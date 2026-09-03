// K7 ESIMD GDN mixer: conv1d K=4 then delta, one in-order go().
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Packed q+k+v C=10240 conv, then 48-head delta with q/k repeat 16->48.
// CONFIG prior: conv ~4.4 + delta 7.1 ~11.5. Rank fused pipe_host vs 11.5.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

namespace {

constexpr int kVl = 16;
constexpr int kWg = 16;
constexpr int kConv = 4;
constexpr int kState = 3;
constexpr int kCq = 2048;
constexpr int kCk = 2048;
constexpr int kCv = 6144;
constexpr int kCtot = kCq + kCk + kCv;
constexpr int kNk = 16;
constexpr int kNv = 48;
constexpr int kDv = 128;
constexpr int kDk = 128;
constexpr int kTiles = kDk / kVl;

struct GdnMixerConvName {};
struct GdnMixerDeltaName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_mixer: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_f16(sycl::half *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        const float v = float(int((i * 17u + seed) % 201u) - 100) / 250.f;
        p[i] = sycl::half(v);
    }
}

int read_int_file(const char *path) {
    FILE *f = std::fopen(path, "r");
    if (!f)
        return -1;
    int v = -1;
    if (std::fscanf(f, "%d", &v) != 1)
        v = -1;
    std::fclose(f);
    return v;
}

void read_str_file(const char *path, char *dst, size_t n) {
    dst[0] = '?';
    dst[1] = 0;
    FILE *f = std::fopen(path, "r");
    if (!f)
        return;
    if (!std::fgets(dst, int(n), f)) {
        dst[0] = '?';
        dst[1] = 0;
        std::fclose(f);
        return;
    }
    std::fclose(f);
    size_t len = std::strlen(dst);
    while (len > 0 && (dst[len - 1] == '\n' || dst[len - 1] == '\r'))
        dst[--len] = 0;
}

void sample_gt(int *act, int *cur, char *power, size_t pn, int *throttle) {
    char path[256];
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/act_freq",
                  g_card);
    *act = read_int_file(path);
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/cur_freq",
                  g_card);
    *cur = read_int_file(path);
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/power_state", g_card);
    read_str_file(path, power, pn);
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/throttle/status",
                  g_card);
    *throttle = read_int_file(path);
}

double median_of(std::vector<double> v) {
    if (v.empty())
        return -1.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2)
        return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

inline float hsum16(esimd::simd<float, kVl> v) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < kVl; ++e)
        s += float(v[e]);
    return s;
}

sycl::event launch_conv(sycl::queue &q, const sycl::half *xd,
                        const sycl::half *wd, const sycl::half *std,
                        sycl::half *yd, sycl::half *stout) {
    const int nwi = kCtot / kVl;
    return q.parallel_for<GdnMixerConvName>(
        sycl::nd_range<1>({size_t(nwi)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int c0 = int(it.get_global_id(0)) * kVl;
            esimd::simd<sycl::half, kVl> xv;
            xv.copy_from(xd + c0);
            esimd::simd<float, kVl> acc(0.f);
#pragma unroll
            for (int t = 0; t < kState; ++t) {
                esimd::simd<sycl::half, kVl> stv;
                esimd::simd<sycl::half, kVl> wv;
                stv.copy_from(std + t * kCtot + c0);
                wv.copy_from(wd + t * kCtot + c0);
                acc += esimd::convert<float>(stv) * esimd::convert<float>(wv);
                if (t > 0)
                    stv.copy_to(stout + (t - 1) * kCtot + c0);
            }
            esimd::simd<sycl::half, kVl> w3;
            w3.copy_from(wd + kState * kCtot + c0);
            acc += esimd::convert<float>(xv) * esimd::convert<float>(w3);
            esimd::simd<sycl::half, kVl> yh;
#pragma unroll
            for (int e = 0; e < kVl; ++e)
                yh[e] = sycl::half(float(acc[e]));
            yh.copy_to(yd + c0);
            xv.copy_to(stout + 2 * kCtot + c0);
        });
}

sycl::event launch_delta(sycl::queue &q, const sycl::half *sd,
                         const sycl::half *yd, const sycl::half *ad,
                         const sycl::half *bd, sycl::half *s2d,
                         sycl::half *od) {
    return q.parallel_for<GdnMixerDeltaName>(
        sycl::nd_range<1>({size_t(kNv * kDv)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int gid = int(it.get_global_id(0));
            const int h = gid / kDv;
            const int i = gid % kDv;
            const int kh = h % kNk;
            const size_t srow =
                (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
            const size_t qbase = size_t(kh) * size_t(kDk);
            const size_t kbase = size_t(kCq) + size_t(kh) * size_t(kDk);
            esimd::simd<sycl::half, kVl> sv[kTiles];
            esimd::simd<sycl::half, kVl> kv[kTiles];
            esimd::simd<sycl::half, kVl> qv[kTiles];
            float vold = 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                sv[t].copy_from(sd + srow + size_t(t * kVl));
                kv[t].copy_from(yd + kbase + size_t(t * kVl));
                qv[t].copy_from(yd + qbase + size_t(t * kVl));
                vold += hsum16(esimd::convert<float>(sv[t]) *
                               esimd::convert<float>(kv[t]));
            }
            esimd::simd<sycl::half, 1> ahv, bhv, viv;
            ahv.copy_from(ad + h);
            bhv.copy_from(bd + h);
            viv.copy_from(yd + size_t(kCq + kCk) + size_t(h) * size_t(kDv) +
                          size_t(i));
            const float ah = float(sycl::half(ahv[0]));
            const float bh = float(sycl::half(bhv[0]));
            const float vi = float(sycl::half(viv[0]));
            float oacc = 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const esimd::simd<float, kVl> sf = esimd::convert<float>(sv[t]);
                const esimd::simd<float, kVl> kf = esimd::convert<float>(kv[t]);
                const esimd::simd<float, kVl> snew =
                    ah * (sf - bh * vold * kf) + bh * vi * kf;
                esimd::simd<sycl::half, kVl> sh;
#pragma unroll
                for (int e = 0; e < kVl; ++e)
                    sh[e] = sycl::half(float(snew[e]));
                sh.copy_to(s2d + srow + size_t(t * kVl));
                oacc += hsum16(snew * esimd::convert<float>(qv[t]));
            }
            esimd::simd<sycl::half, 1> ov;
            ov[0] = sycl::half(oacc);
            ov.copy_to(od + size_t(h) * size_t(kDv) + size_t(i));
        });
}

void host_conv(const sycl::half *x, const sycl::half *w, const sycl::half *st,
               sycl::half *y, int c) {
    for (int i = 0; i < c; ++i) {
        float acc = 0.f;
        for (int t = 0; t < kState; ++t)
            acc += float(st[t * c + i]) * float(w[t * c + i]);
        acc += float(x[i]) * float(w[kState * c + i]);
        y[i] = sycl::half(acc);
    }
}

void host_delta(const sycl::half *s, const sycl::half *y, const sycl::half *a,
                const sycl::half *b, sycl::half *s2, sycl::half *o) {
    for (int h = 0; h < kNv; ++h) {
        const float ah = float(a[h]);
        const float bh = float(b[h]);
        const int kh = h % kNk;
        for (int i = 0; i < kDv; ++i) {
            const size_t srow =
                (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
            float vold = 0.f;
            for (int j = 0; j < kDk; ++j)
                vold += float(s[srow + size_t(j)]) *
                        float(y[size_t(kCq) + size_t(kh) * size_t(kDk) +
                                size_t(j)]);
            const float vi = float(y[size_t(kCq + kCk) + size_t(h) * size_t(kDv) +
                                     size_t(i)]);
            float oacc = 0.f;
            for (int j = 0; j < kDk; ++j) {
                const float kj = float(y[size_t(kCq) + size_t(kh) * size_t(kDk) +
                                         size_t(j)]);
                const float snew =
                    ah * (float(s[srow + size_t(j)]) - bh * vold * kj) +
                    bh * vi * kj;
                s2[srow + size_t(j)] = sycl::half(snew);
                oacc += snew * float(y[size_t(kh) * size_t(kDk) + size_t(j)]);
            }
            o[size_t(h) * size_t(kDv) + size_t(i)] = sycl::half(oacc);
        }
    }
}

void score_f16(const sycl::half *got, const sycl::half *ref, size_t n,
               double *cosine, double *mx) {
    double dot = 0, na2 = 0, nb2 = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        const float x = float(got[i]);
        const float y = float(ref[i]);
        const float d = x > y ? x - y : y - x;
        if (d > m)
            m = d;
        dot += double(x) * double(y);
        na2 += double(x) * double(x);
        nb2 += double(y) * double(y);
    }
    *cosine = (na2 > 0 && nb2 > 0) ? dot / std::sqrt(na2 * nb2) : 0.0;
    *mx = m;
}

void run_shape(sycl::queue &q, int warmup, int iters, int *rc) {
    const size_t nx = size_t(kCtot);
    const size_t nw = size_t(kConv) * nx;
    const size_t nst = size_t(kState) * nx;
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    const size_t no = size_t(kNv) * size_t(kDv);
    const size_t nsc = size_t(kNv);
    std::vector<sycl::half> hx(nx), hw(nw), hst(nst), hy(nx), hyref(nx);
    std::vector<sycl::half> hs(ns), hrefs(ns), hgots(ns), hrefo(no), hgoto(no);
    std::vector<sycl::half> ha(nsc), hb(nsc);
    fill_f16(hx.data(), nx, 1);
    fill_f16(hw.data(), nw, 9);
    fill_f16(hst.data(), nst, 13);
    fill_f16(hs.data(), ns, 17);
    for (int h = 0; h < kNv; ++h) {
        ha[size_t(h)] = sycl::half(0.25f + float(h % 8) * 0.05f);
        hb[size_t(h)] = sycl::half(0.30f + float(h % 5) * 0.08f);
    }
    host_conv(hx.data(), hw.data(), hst.data(), hyref.data(), kCtot);
    host_delta(hs.data(), hyref.data(), ha.data(), hb.data(), hrefs.data(),
               hrefo.data());

    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(nw, q);
    sycl::half *std = sycl::malloc_device<sycl::half>(nst, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *stout = sycl::malloc_device<sycl::half>(nst, q);
    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *ad = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *s2d = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *od = sycl::malloc_device<sycl::half>(no, q);
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), nw * sizeof(sycl::half)).wait();
    q.memcpy(std, hst.data(), nst * sizeof(sycl::half)).wait();
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();
    q.memcpy(ad, ha.data(), nsc * sizeof(sycl::half)).wait();
    q.memcpy(bd, hb.data(), nsc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        (void)launch_conv(q, xd, wd, std, yd, stout);
        return launch_delta(q, sd, yd, ad, bd, s2d, od);
    };

    go().wait_and_throw();
    q.memcpy(hgots.data(), s2d, ns * sizeof(sycl::half)).wait();
    q.memcpy(hgoto.data(), od, no * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_o = 0, mx_o = 0;
    score_f16(hgots.data(), hrefs.data(), ns, &cosine, &mx);
    score_f16(hgoto.data(), hrefo.data(), no, &cosine_o, &mx_o);
    const int ok = (cosine > 0.99 && cosine_o > 0.99) ? 1 : 0;
    if (!ok)
        *rc = 1;

    auto batch_wait = [&](int np) {
        constexpr int kBatch = 256;
        for (int i = 0; i < np; ++i) {
            (void)go();
            if ((i % kBatch) == (kBatch - 1))
                q.wait_and_throw();
        }
        q.wait_and_throw();
    };
    batch_wait(warmup);
    if (g_spin > 0) {
        batch_wait(g_spin);
        int act = -1, cur = -1, throttle = -1;
        char power[32];
        sample_gt(&act, &cur, power, sizeof(power), &throttle);
        std::printf("spin_done n=%d act=%d cur=%d power=%s throttle=%d\n",
                    g_spin, act, cur, power, throttle);
    }

    std::vector<double> all_us;
    uint64_t ns_sum = 0;
    int act0 = -1, cur0 = -1, thr0 = -1;
    char power0[32];
    sample_gt(&act0, &cur0, power0, sizeof(power0), &thr0);
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0, cur0,
                power0, thr0);
    const auto host0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sycl::event ec = launch_conv(q, xd, wd, std, yd, stout);
        ec.wait_and_throw();
        sycl::event ed = launch_delta(q, sd, yd, ad, bd, s2d, od);
        ed.wait_and_throw();
        auto span = [](sycl::event e) {
            const uint64_t t0 =
                e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 =
                e.get_profiling_info<sycl::info::event_profiling::command_end>();
            return t1 - t0;
        };
        const uint64_t ns = span(ec) + span(ed);
        ns_sum += ns;
        all_us.push_back(double(ns) / 1000.0);
    }
    const auto host1 = std::chrono::steady_clock::now();
    const double wait_host =
        std::chrono::duration<double, std::micro>(host1 - host0).count() /
        double(iters);
    const auto pipe0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        (void)go();
    q.wait_and_throw();
    const auto pipe1 = std::chrono::steady_clock::now();
    const double pipe_us =
        std::chrono::duration<double, std::micro>(pipe1 - pipe0).count() /
        double(iters);
    int act1 = -1, cur1 = -1, thr1 = -1;
    char power1[32];
    sample_gt(&act1, &cur1, power1, sizeof(power1), &thr1);
    std::printf("timed_end act=%d cur=%d power=%s throttle=%d\n", act1, cur1,
                power1, thr1);
    const double us = (double(ns_sum) / 1000.0) / double(iters);
    std::printf("phase,c,event_us,wait_host_us,pipe_host_us,cosine,max_abs,"
                "cosine_o,max_abs_o,ok,median_us,min_us,max_us\n");
    std::printf("mixer,%d,%.3f,%.3f,%.3f,%.6f,%.5g,%.6f,%.5g,%d,%.3f,%.3f,"
                "%.3f\n",
                kCtot, us, wait_host, pipe_us, cosine, mx, cosine_o, mx_o, ok,
                median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(xd, q);
    sycl::free(wd, q);
    sycl::free(std, q);
    sycl::free(yd, q);
    sycl::free(stout, q);
    sycl::free(sd, q);
    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(s2d, q);
    sycl::free(od, q);
}

} // namespace

int main(int argc, char **argv) {
    int warmup = 50, iters = 40;
    const char *aff = std::getenv("ZE_AFFINITY_MASK");
    if (aff && aff[0])
        g_card = std::atoi(aff);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--iters")
            take(iters);
        else if (a == "--warmup")
            take(warmup);
        else if (a == "--card")
            take(g_card);
        else if (a == "--spin")
            take(g_spin);
        else if (a == "--mhz")
            take(g_mhz);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "gdn_mixer [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_mixer: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "op=gdn_mixer conv+delta C=%d nv=%d dtype=f16 "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kCtot, kNv, warmup,
                iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

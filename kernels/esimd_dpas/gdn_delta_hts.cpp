// K7 ESIMD GDN fused-recurrent delta T=1 tile-fused scalar hsum. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Qwen3.8-27B: 48 v-heads, S 128x128 f16. T=1 decode.
// Tile-fused acc, scalar hsum not esimd::reduce. Prior: ht 5.54
// max_abs_o=2. Rank pipe_host vs fused 7.1.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

namespace {

constexpr int kVl = 16;
constexpr int kNv = 48;
constexpr int kDv = 128;
constexpr int kDk = 128;
constexpr int kTiles = kDk / kVl;

struct GdnDeltaHtsName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_delta_hts: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_f16(sycl::half *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        const float v = float(int((i * 17u + seed) % 201u) - 100) / 50.f;
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

sycl::event launch(sycl::queue &q, const sycl::half *sd, const sycl::half *qd,
                   const sycl::half *kd, const sycl::half *vd,
                   const sycl::half *ad, const sycl::half *bd, sycl::half *s2d,
                   sycl::half *od) {
    constexpr int kWg = 16;
    return q.parallel_for<GdnDeltaHtsName>(
        sycl::nd_range<1>({size_t(kNv * kDv)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int gid = int(it.get_global_id(0));
            const int h = gid / kDv;
            const int i = gid % kDv;
            const size_t srow = (size_t(h) * size_t(kDv) + size_t(i)) *
                                size_t(kDk);
            const size_t kbase = size_t(h) * size_t(kDk);
            esimd::simd<float, kVl> sf[kTiles];
            esimd::simd<float, kVl> kf[kTiles];
            esimd::simd<float, kVl> qf[kTiles];
            esimd::simd<float, kVl> vacc = 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                esimd::simd<sycl::half, kVl> sv, kv, qv;
                sv.copy_from(sd + srow + size_t(t * kVl));
                kv.copy_from(kd + kbase + size_t(t * kVl));
                qv.copy_from(qd + kbase + size_t(t * kVl));
                sf[t] = esimd::convert<float>(sv);
                kf[t] = esimd::convert<float>(kv);
                qf[t] = esimd::convert<float>(qv);
                vacc += sf[t] * kf[t];
            }
            const float vold = hsum16(vacc);
            esimd::simd<sycl::half, 1> ahv, bhv, viv;
            ahv.copy_from(ad + h);
            bhv.copy_from(bd + h);
            viv.copy_from(vd + size_t(h) * size_t(kDv) + size_t(i));
            const float ah = float(sycl::half(ahv[0]));
            const float bh = float(sycl::half(bhv[0]));
            const float vi = float(sycl::half(viv[0]));
            esimd::simd<float, kVl> oaccv = 0.f;
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const esimd::simd<float, kVl> snew =
                    ah * (sf[t] - bh * vold * kf[t]) + bh * vi * kf[t];
                esimd::simd<sycl::half, kVl> sh;
#pragma unroll
                for (int e = 0; e < kVl; ++e)
                    sh[e] = sycl::half(float(snew[e]));
                sh.copy_to(s2d + srow + size_t(t * kVl));
                oaccv += snew * qf[t];
            }
            const float oacc = hsum16(oaccv);
            esimd::simd<sycl::half, 1> ov;
            ov[0] = sycl::half(oacc);
            ov.copy_to(od + size_t(h) * size_t(kDv) + size_t(i));
        });
}

void host_delta(const sycl::half *s, const sycl::half *q, const sycl::half *k,
                const sycl::half *v, const sycl::half *a, const sycl::half *b,
                sycl::half *s2, sycl::half *o) {
    for (int h = 0; h < kNv; ++h) {
        const float ah = float(a[h]);
        const float bh = float(b[h]);
        for (int i = 0; i < kDv; ++i) {
            const size_t srow =
                (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
            float vold = 0.f;
            for (int j = 0; j < kDk; ++j)
                vold += float(s[srow + size_t(j)]) *
                        float(k[size_t(h) * size_t(kDk) + size_t(j)]);
            const float vi = float(v[size_t(h) * size_t(kDv) + size_t(i)]);
            float oacc = 0.f;
            for (int j = 0; j < kDk; ++j) {
                const float kj = float(k[size_t(h) * size_t(kDk) + size_t(j)]);
                const float snew =
                    ah * (float(s[srow + size_t(j)]) - bh * vold * kj) +
                    bh * vi * kj;
                s2[srow + size_t(j)] = sycl::half(snew);
                oacc += snew * float(q[size_t(h) * size_t(kDk) + size_t(j)]);
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

void run_shape(sycl::queue &q, const char *phase, int warmup, int iters,
               int *rc, int do_spin) {
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    const size_t nq = size_t(kNv) * size_t(kDk);
    const size_t nv = size_t(kNv) * size_t(kDv);
    const size_t nsc = size_t(kNv);
    std::vector<sycl::half> hs(ns), hq(nq), hk(nq), hv(nv), ha(nsc), hb(nsc);
    std::vector<sycl::half> hrefs(ns), hgots(ns), hrefo(nv), hgoto(nv);
    fill_f16(hs.data(), ns, 1);
    fill_f16(hq.data(), nq, 3);
    fill_f16(hk.data(), nq, 5);
    fill_f16(hv.data(), nv, 7);
    for (int h = 0; h < kNv; ++h) {
        ha[size_t(h)] = sycl::half(0.25f + float(h % 8) * 0.05f);
        hb[size_t(h)] = sycl::half(0.30f + float(h % 5) * 0.08f);
    }
    host_delta(hs.data(), hq.data(), hk.data(), hv.data(), ha.data(),
               hb.data(), hrefs.data(), hrefo.data());

    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *qd = sycl::malloc_device<sycl::half>(nq, q);
    sycl::half *kd = sycl::malloc_device<sycl::half>(nq, q);
    sycl::half *vd = sycl::malloc_device<sycl::half>(nv, q);
    sycl::half *ad = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *s2d = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *od = sycl::malloc_device<sycl::half>(nv, q);
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();
    q.memcpy(qd, hq.data(), nq * sizeof(sycl::half)).wait();
    q.memcpy(kd, hk.data(), nq * sizeof(sycl::half)).wait();
    q.memcpy(vd, hv.data(), nv * sizeof(sycl::half)).wait();
    q.memcpy(ad, ha.data(), nsc * sizeof(sycl::half)).wait();
    q.memcpy(bd, hb.data(), nsc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, sd, qd, kd, vd, ad, bd, s2d, od);
    };

    go().wait_and_throw();
    q.memcpy(hgots.data(), s2d, ns * sizeof(sycl::half)).wait();
    q.memcpy(hgoto.data(), od, nv * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_o = 0, mx_o = 0;
    score_f16(hgots.data(), hrefs.data(), ns, &cosine, &mx);
    score_f16(hgoto.data(), hrefo.data(), nv, &cosine_o, &mx_o);
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
    if (do_spin && g_spin > 0) {
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
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ns_sum += (t1 - t0);
        all_us.push_back(double(t1 - t0) / 1000.0);
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
    const double nbytes =
        double((ns + ns + nq + nq + nv + nsc + nsc + nv) * sizeof(sycl::half));
    const double gbs = (nbytes / 1.0e9) / (pipe_us * 1.0e-6);
    std::printf("phase,nv,dv,dk,event_us,wait_host_us,pipe_host_us,GBs,cosine,"
                "max_abs,cosine_o,max_abs_o,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%.6f,%.5g,%d,%.3f,"
                "%.3f,%.3f\n",
                phase, kNv, kDv, kDk, us, wait_host, pipe_us, gbs, cosine, mx,
                cosine_o, mx_o, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(sd, q);
    sycl::free(qd, q);
    sycl::free(kd, q);
    sycl::free(vd, q);
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
            std::fprintf(stderr, "gdn_delta_hts [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_delta_hts: unknown arg %s\n", a.c_str());
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
                "op=gdn_delta_hts_recurrent dtype=f16 nv=%d dv=%d dk=%d VL=%d "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kNv, kDv, kDk, kVl,
                warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, "timed", warmup, iters, &rc, 1);
    return rc;
}

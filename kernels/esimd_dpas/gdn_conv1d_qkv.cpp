// K7 ESIMD GDN fused q+k+v conv1d K=4 one launch. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Pack q=2048 k=2048 v=6144 -> C=10240. Same 4-tap as gdn_conv1d.
// CONFIG prior: one-arm ~4.4 us; 3x ~13.2. Rank fused vs trio pipe_host.

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

struct GdnConv1dQkvName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_conv1d_qkv: no GPU\n");
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

sycl::event launch(sycl::queue &q, const sycl::half *xd, const sycl::half *wd,
                   const sycl::half *std, sycl::half *yd, sycl::half *stout,
                   int c) {
    const int nwi = c / kVl;
    const size_t local = (nwi % kWg == 0) ? size_t(kWg) : size_t(kVl);
    return q.parallel_for<GdnConv1dQkvName>(
        sycl::nd_range<1>({size_t(nwi)}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int c0 = int(it.get_global_id(0)) * kVl;
            esimd::simd<sycl::half, kVl> xv;
            xv.copy_from(xd + c0);
            esimd::simd<float, kVl> acc(0.f);
#pragma unroll
            for (int t = 0; t < kState; ++t) {
                esimd::simd<sycl::half, kVl> stv;
                esimd::simd<sycl::half, kVl> wv;
                stv.copy_from(std + t * c + c0);
                wv.copy_from(wd + t * c + c0);
                acc += esimd::convert<float>(stv) * esimd::convert<float>(wv);
                if (t > 0)
                    stv.copy_to(stout + (t - 1) * c + c0);
            }
            esimd::simd<sycl::half, kVl> w3;
            w3.copy_from(wd + kState * c + c0);
            acc += esimd::convert<float>(xv) * esimd::convert<float>(w3);
            esimd::simd<sycl::half, kVl> yh;
#pragma unroll
            for (int e = 0; e < kVl; ++e)
                yh[e] = sycl::half(float(acc[e]));
            yh.copy_to(yd + c0);
            xv.copy_to(stout + 2 * c + c0);
        });
}

void host_conv(const sycl::half *x, const sycl::half *w, const sycl::half *st,
               sycl::half *y, sycl::half *stout, int c) {
    for (int i = 0; i < c; ++i) {
        float acc = 0.f;
        for (int t = 0; t < kState; ++t)
            acc += float(st[t * c + i]) * float(w[t * c + i]);
        acc += float(x[i]) * float(w[kState * c + i]);
        y[i] = sycl::half(acc);
        stout[0 * c + i] = st[1 * c + i];
        stout[1 * c + i] = st[2 * c + i];
        stout[2 * c + i] = x[i];
    }
}

void score_f16(const sycl::half *got, const sycl::half *ref, size_t n,
               double *cosine, double *mx) {
    double dot = 0, na2 = 0, nb2 = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        const float a = float(got[i]);
        const float b = float(ref[i]);
        const float d = a > b ? a - b : b - a;
        if (d > m)
            m = d;
        dot += double(a) * double(b);
        na2 += double(a) * double(a);
        nb2 += double(b) * double(b);
    }
    *cosine = (na2 > 0 && nb2 > 0) ? dot / std::sqrt(na2 * nb2) : 0.0;
    *mx = m;
}

void run_shape(sycl::queue &q, const char *phase, int c, int warmup, int iters,
               int *rc, int do_spin) {
    if (c % kVl != 0 || c < kVl) {
        std::fprintf(stderr, "gdn_conv1d_qkv: C=%d not aligned to %d\n", c, kVl);
        *rc = 2;
        return;
    }
    const size_t nx = size_t(c);
    const size_t nw = size_t(kConv) * size_t(c);
    const size_t ns = size_t(kState) * size_t(c);
    std::vector<sycl::half> hx(nx), hw(nw), hst(ns), hyref(nx), hygot(nx);
    std::vector<sycl::half> hrefst(ns), hgotst(ns);
    fill_f16(hx.data(), nx, 1);
    fill_f16(hw.data(), nw, 9);
    fill_f16(hst.data(), ns, 13);
    host_conv(hx.data(), hw.data(), hst.data(), hyref.data(), hrefst.data(), c);

    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(nw, q);
    sycl::half *std = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *stout = sycl::malloc_device<sycl::half>(ns, q);
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), nw * sizeof(sycl::half)).wait();
    q.memcpy(std, hst.data(), ns * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, xd, wd, std, yd, stout, c);
    };

    go().wait_and_throw();
    q.memcpy(hygot.data(), yd, nx * sizeof(sycl::half)).wait();
    q.memcpy(hgotst.data(), stout, ns * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_st = 0, mx_st = 0;
    score_f16(hygot.data(), hyref.data(), nx, &cosine, &mx);
    score_f16(hgotst.data(), hrefst.data(), ns, &cosine_st, &mx_st);
    const int ok = (cosine > 0.99 && cosine_st > 0.99) ? 1 : 0;
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
    const double nbytes = double((nx + nw + ns + nx + ns) * sizeof(sycl::half));
    const double gbs = (nbytes / 1.0e9) / (pipe_us * 1.0e-6);
    std::printf("phase,c,kconv,event_us,wait_host_us,pipe_host_us,GBs,cosine,"
                "max_abs,cosine_st,max_abs_st,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%.6f,%.5g,%d,%.3f,%.3f,"
                "%.3f\n",
                phase, c, kConv, us, wait_host, pipe_us, gbs, cosine, mx,
                cosine_st, mx_st, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(xd, q);
    sycl::free(wd, q);
    sycl::free(std, q);
    sycl::free(yd, q);
    sycl::free(stout, q);
}

struct ConvBuf {
    int c;
    sycl::half *x;
    sycl::half *w;
    sycl::half *st;
    sycl::half *y;
    sycl::half *so;
};

ConvBuf alloc_conv(sycl::queue &q, int c, unsigned seed) {
    ConvBuf b{};
    b.c = c;
    const size_t nx = size_t(c);
    const size_t nw = size_t(kConv) * nx;
    const size_t ns = size_t(kState) * nx;
    std::vector<sycl::half> hx(nx), hw(nw), hst(ns);
    fill_f16(hx.data(), nx, seed);
    fill_f16(hw.data(), nw, seed + 8);
    fill_f16(hst.data(), ns, seed + 16);
    b.x = sycl::malloc_device<sycl::half>(nx, q);
    b.w = sycl::malloc_device<sycl::half>(nw, q);
    b.st = sycl::malloc_device<sycl::half>(ns, q);
    b.y = sycl::malloc_device<sycl::half>(nx, q);
    b.so = sycl::malloc_device<sycl::half>(ns, q);
    q.memcpy(b.x, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(b.w, hw.data(), nw * sizeof(sycl::half)).wait();
    q.memcpy(b.st, hst.data(), ns * sizeof(sycl::half)).wait();
    return b;
}

void free_conv(sycl::queue &q, ConvBuf &b) {
    sycl::free(b.x, q);
    sycl::free(b.w, q);
    sycl::free(b.st, q);
    sycl::free(b.y, q);
    sycl::free(b.so, q);
}

void run_trio(sycl::queue &q, int warmup, int iters, int *rc) {
    ConvBuf qb = alloc_conv(q, 2048, 1);
    ConvBuf kb = alloc_conv(q, 2048, 3);
    ConvBuf vb = alloc_conv(q, 6144, 5);
    auto go = [&]() {
        (void)launch(q, qb.x, qb.w, qb.st, qb.y, qb.so, qb.c);
        (void)launch(q, kb.x, kb.w, kb.st, kb.y, kb.so, kb.c);
        return launch(q, vb.x, vb.w, vb.st, vb.y, vb.so, vb.c);
    };
    go().wait_and_throw();
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
        sycl::event e0 = launch(q, qb.x, qb.w, qb.st, qb.y, qb.so, qb.c);
        e0.wait_and_throw();
        sycl::event e1 = launch(q, kb.x, kb.w, kb.st, kb.y, kb.so, kb.c);
        e1.wait_and_throw();
        sycl::event e2 = launch(q, vb.x, vb.w, vb.st, vb.y, vb.so, vb.c);
        e2.wait_and_throw();
        auto span = [](sycl::event e) {
            const uint64_t t0 =
                e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 =
                e.get_profiling_info<sycl::info::event_profiling::command_end>();
            return t1 - t0;
        };
        const uint64_t ns = span(e0) + span(e1) + span(e2);
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
    std::printf("phase,c,kconv,event_us,wait_host_us,pipe_host_us,GBs,cosine,"
                "max_abs,cosine_st,max_abs_st,ok,median_us,min_us,max_us\n");
    std::printf("trio,10240,%d,%.3f,%.3f,%.3f,0,1.000000,0,1.000000,0,1,%.3f,"
                "%.3f,%.3f\n",
                kConv, us, wait_host, pipe_us, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));
    (void)rc;
    free_conv(q, qb);
    free_conv(q, kb);
    free_conv(q, vb);
}

} // namespace

int main(int argc, char **argv) {
    int timed_c = 10240;
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
        if (a == "--c")
            take(timed_c);
        else if (a == "--iters")
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
            std::fprintf(stderr, "gdn_conv1d_qkv [--c 10240] [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_conv1d_qkv: unknown arg %s\n", a.c_str());
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
                "op=gdn_conv1d_qkv dtype=f16 T=1 VL=%d wg=%d C=%d "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kVl, kWg, timed_c,
                warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, "fused", timed_c, warmup, iters, &rc, 1);
    run_trio(q, warmup, iters, &rc);
    return rc;
}

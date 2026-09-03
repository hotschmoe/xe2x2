// K7 ESIMD GDN depthwise conv1d K=4 prefill T=64. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Causal 4-tap over T, zero pad. C=2048. CONFIG prior: eager ~115, decode 4.4.
// f16 in/out, float acc. Rank pipe_host vs 115.

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

struct GdnConv1dTName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_conv1d_t: no GPU\n");
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
                   sycl::half *yd, int c, int tlen) {
    const int nwi = c / kVl;
    const size_t local = (nwi % kWg == 0) ? size_t(kWg) : size_t(kVl);
    return q.parallel_for<GdnConv1dTName>(
        sycl::nd_range<1>({size_t(nwi)}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int c0 = int(it.get_global_id(0)) * kVl;
            esimd::simd<sycl::half, kVl> wv[kConv];
#pragma unroll
            for (int k = 0; k < kConv; ++k)
                wv[k].copy_from(wd + k * c + c0);
            esimd::simd<float, kVl> h0(0.f), h1(0.f), h2(0.f);
            for (int t = 0; t < tlen; ++t) {
                esimd::simd<sycl::half, kVl> xv;
                xv.copy_from(xd + t * c + c0);
                const esimd::simd<float, kVl> xf = esimd::convert<float>(xv);
                const esimd::simd<float, kVl> acc =
                    h0 * esimd::convert<float>(wv[0]) +
                    h1 * esimd::convert<float>(wv[1]) +
                    h2 * esimd::convert<float>(wv[2]) +
                    xf * esimd::convert<float>(wv[3]);
                esimd::simd<sycl::half, kVl> yh;
#pragma unroll
                for (int e = 0; e < kVl; ++e)
                    yh[e] = sycl::half(float(acc[e]));
                yh.copy_to(yd + t * c + c0);
                h0 = h1;
                h1 = h2;
                h2 = xf;
            }
        });
}

void host_conv(const sycl::half *x, const sycl::half *w, sycl::half *y, int c,
               int tlen) {
    for (int i = 0; i < c; ++i) {
        float h0 = 0.f, h1 = 0.f, h2 = 0.f;
        for (int t = 0; t < tlen; ++t) {
            const float xf = float(x[t * c + i]);
            const float acc = h0 * float(w[0 * c + i]) +
                              h1 * float(w[1 * c + i]) +
                              h2 * float(w[2 * c + i]) + xf * float(w[3 * c + i]);
            y[t * c + i] = sycl::half(acc);
            h0 = h1;
            h1 = h2;
            h2 = xf;
        }
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

void run_shape(sycl::queue &q, const char *phase, int c, int tlen, int warmup,
               int iters, int *rc, int do_spin) {
    if (c % kVl != 0 || c < kVl || tlen < 1) {
        std::fprintf(stderr, "gdn_conv1d_t: C=%d T=%d\n", c, tlen);
        *rc = 2;
        return;
    }
    const size_t nx = size_t(tlen) * size_t(c);
    const size_t nw = size_t(kConv) * size_t(c);
    std::vector<sycl::half> hx(nx), hw(nw), hyref(nx), hygot(nx);
    fill_f16(hx.data(), nx, 1);
    fill_f16(hw.data(), nw, 9);
    host_conv(hx.data(), hw.data(), hyref.data(), c, tlen);

    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(nw, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q);
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), nw * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, xd, wd, yd, c, tlen);
    };

    go().wait_and_throw();
    q.memcpy(hygot.data(), yd, nx * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0;
    score_f16(hygot.data(), hyref.data(), nx, &cosine, &mx);
    const int ok = (cosine > 0.99) ? 1 : 0;
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
    const double nbytes = double((nx + nw + nx) * sizeof(sycl::half));
    const double gbs = (nbytes / 1.0e9) / (pipe_us * 1.0e-6);
    std::printf("phase,c,t,kconv,event_us,wait_host_us,pipe_host_us,GBs,cosine,"
                "max_abs,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%d,%.3f,%.3f,%.3f\n",
                phase, c, tlen, kConv, us, wait_host, pipe_us, gbs, cosine, mx,
                ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(xd, q);
    sycl::free(wd, q);
    sycl::free(yd, q);
}

} // namespace

int main(int argc, char **argv) {
    int timed_c = 2048;
    int timed_t = 64;
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
        else if (a == "--t")
            take(timed_t);
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
            std::fprintf(stderr,
                         "gdn_conv1d_t [--c 2048] [--t 64] [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_conv1d_t: unknown arg %s\n", a.c_str());
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
                "op=gdn_conv1d_t dtype=f16 T=%d VL=%d wg=%d C=%d "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), timed_t, kVl, kWg,
                timed_c, warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, "timed", timed_c, timed_t, warmup, iters, &rc, 1);
    return rc;
}

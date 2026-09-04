// K8 ESIMD bf16 GEMV hail-mary: M=1, N=1856, K=2688.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. bf16 A/B, f32 acc, f16 out.
// Rank pipe_host against NVFP4 LUT 84 us and s8 16 us serving floors.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
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
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr int kM = 1;
constexpr int kN = 1856;
constexpr int kK = 2688;
constexpr int kVl = 16;
constexpr int kOutputsPerWi = 4;
constexpr int kWg = 16;

struct GemvBf16M1Name {};

int g_card = 0;
int g_spin = 4000;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gemv_bf16_m1: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_bf16(bf16 *p, size_t n, unsigned seed, float scale) {
    for (size_t i = 0; i < n; ++i)
        p[i] = bf16(float(int((i * 17u + seed) % 201u) - 100) * scale);
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
    if (std::fgets(dst, int(n), f)) {
        size_t len = std::strlen(dst);
        while (len && (dst[len - 1] == '\n' || dst[len - 1] == '\r'))
            dst[--len] = 0;
    }
    std::fclose(f);
}

void sample_gt(int *act, int *cur, char *power, size_t pn, int *throttle) {
    char path[256];
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/act_freq", g_card);
    *act = read_int_file(path);
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/cur_freq", g_card);
    *cur = read_int_file(path);
    std::snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/power_state",
                  g_card);
    read_str_file(path, power, pn);
    std::snprintf(path, sizeof(path),
                  "/sys/class/drm/card%d/device/tile0/gt0/freq0/throttle/status",
                  g_card);
    *throttle = read_int_file(path);
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() & 1 ? v[v.size() / 2]
                        : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

sycl::event launch(sycl::queue &q, const bf16 *a, const bf16 *b,
                   sycl::half *out) {
    constexpr int nwi = kN / kOutputsPerWi;
    return q.parallel_for<GemvBf16M1Name>(
        sycl::nd_range<1>({size_t(nwi)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int n0 = int(it.get_global_id(0)) * kOutputsPerWi;
            esimd::simd<float, kOutputsPerWi> acc(0.f);
            for (int k0 = 0; k0 < kK; k0 += kVl) {
                esimd::simd<bf16, kVl> av;
                av.copy_from(a + k0);
                const esimd::simd<float, kVl> af = esimd::convert<float>(av);
#pragma unroll
                for (int n = 0; n < kOutputsPerWi; ++n) {
                    esimd::simd<bf16, kVl> bv;
                    bv.copy_from(b + size_t(n0 + n) * kK + k0);
                    const esimd::simd<float, kVl> prod =
                        af * esimd::convert<float>(bv);
#pragma unroll
                    for (int lane = 0; lane < kVl; ++lane)
                        acc[n] += prod[lane];
                }
            }
#pragma unroll
            for (int n = 0; n < kOutputsPerWi; ++n)
                out[n0 + n] = sycl::half(acc[n]);
        });
}

void host_oracle(const bf16 *a, const bf16 *b, sycl::half *out) {
    for (int n = 0; n < kN; ++n) {
        float acc = 0.f;
        for (int k = 0; k < kK; ++k)
            acc += float(a[k]) * float(b[size_t(n) * kK + k]);
        out[n] = sycl::half(acc);
    }
}

void run_shape(sycl::queue &q, int warmup, int iters, int *rc) {
    std::vector<bf16> ha(kK), hb(size_t(kN) * kK);
    std::vector<sycl::half> href(kN), hgot(kN);
    fill_bf16(ha.data(), ha.size(), 1, 0.01f);
    fill_bf16(hb.data(), hb.size(), 9, 0.01f);
    host_oracle(ha.data(), hb.data(), href.data());
    bf16 *ad = sycl::malloc_device<bf16>(ha.size(), q);
    bf16 *bd = sycl::malloc_device<bf16>(hb.size(), q);
    sycl::half *od = sycl::malloc_device<sycl::half>(kN, q);
    q.memcpy(ad, ha.data(), ha.size() * sizeof(bf16)).wait();
    q.memcpy(bd, hb.data(), hb.size() * sizeof(bf16)).wait();
    auto go = [&]() { return launch(q, ad, bd, od); };
    go().wait_and_throw();
    q.memcpy(hgot.data(), od, size_t(kN) * sizeof(sycl::half)).wait();
    double dot = 0, got2 = 0, ref2 = 0, mx = 0;
    for (int i = 0; i < kN; ++i) {
        const double got = float(hgot[i]);
        const double ref = float(href[i]);
        dot += got * ref;
        got2 += got * got;
        ref2 += ref * ref;
        mx = std::max(mx, std::abs(got - ref));
    }
    const double cosine = dot / std::sqrt(got2 * ref2);
    const int ok = cosine > 0.99;
    if (!ok)
        *rc = 1;
    auto batch_wait = [&](int count) {
        for (int i = 0; i < count; ++i) {
            (void)go();
            if ((i & 255) == 255)
                q.wait_and_throw();
        }
        q.wait_and_throw();
    };
    batch_wait(warmup);
    if (g_spin > 0) {
        batch_wait(g_spin);
        int act, cur, throttle;
        char power[32];
        sample_gt(&act, &cur, power, sizeof(power), &throttle);
        std::printf("spin_done n=%d act=%d cur=%d power=%s throttle=%d\n",
                    g_spin, act, cur, power, throttle);
    }
    int act0, cur0, thr0;
    char power0[32];
    sample_gt(&act0, &cur0, power0, sizeof(power0), &thr0);
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0, cur0,
                power0, thr0);
    uint64_t event_ns = 0;
    std::vector<double> samples;
    const auto wait0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        event_ns += t1 - t0;
        samples.push_back(double(t1 - t0) / 1000.0);
    }
    const auto wait1 = std::chrono::steady_clock::now();
    const double wait_host_us =
        std::chrono::duration<double, std::micro>(wait1 - wait0).count() / iters;
    const auto pipe0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        (void)go();
    q.wait_and_throw();
    const auto pipe1 = std::chrono::steady_clock::now();
    const double pipe_host_us =
        std::chrono::duration<double, std::micro>(pipe1 - pipe0).count() / iters;
    int act1, cur1, thr1;
    char power1[32];
    sample_gt(&act1, &cur1, power1, sizeof(power1), &thr1);
    std::printf("timed_end act=%d cur=%d power=%s throttle=%d\n", act1, cur1,
                power1, thr1);
    const double ops = 2.0 * kM * kN * kK;
    const double tops = (ops / 1.0e12) / (pipe_host_us * 1.0e-6);
    std::printf("phase,M,N,K,event_us,wait_host_us,pipe_host_us,TOPS,"
                "lut_floor_us,s8_floor_us,cosine,max_abs,ok,median_us,min_us,"
                "max_us\n");
    std::printf("timed,%d,%d,%d,%.3f,%.3f,%.3f,%.4f,84.000,16.000,%.6f,"
                "%.5g,%d,%.3f,%.3f,%.3f\n",
                kM, kN, kK, double(event_ns) / 1000.0 / iters, wait_host_us,
                pipe_host_us, tops, cosine, mx, ok, median_of(samples),
                *std::min_element(samples.begin(), samples.end()),
                *std::max_element(samples.begin(), samples.end()));
    sycl::free(ad, q); sycl::free(bd, q); sycl::free(od, q);
}

} // namespace

int main(int argc, char **argv) {
    int warmup = 50, iters = 40;
    const char *aff = std::getenv("ZE_AFFINITY_MASK");
    if (aff && aff[0])
        g_card = std::atoi(aff);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](int &dst) { if (i + 1 < argc) dst = std::atoi(argv[++i]); };
        if (a == "--warmup") take(warmup);
        else if (a == "--iters") take(iters);
        else if (a == "--card") take(g_card);
        else if (a == "--spin") take(g_spin);
        else if (a == "--mhz") take(g_mhz);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "gemv_bf16_m1 [--warmup 50] [--iters 40] "
                                 "[--card 0] [--spin 4000] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gemv_bf16_m1: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (warmup < 0 || iters < 1 || g_spin < 0)
        return 2;
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0" : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "op=gemv_bf16_m1 M=%d N=%d K=%d dtype=bf16xbf16_accf32_outf16 "
                "arm=dequant_to_bf16_hail_mary rank=pipe_host floors_us=lut84_s8_16 "
                "warmup=%d iters=%d card=%d spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kM, kN, kK, warmup,
                iters, g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

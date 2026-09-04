// K8 ESIMD Lightning MoE router: M=1, hidden=2688, experts=128.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. f16 GEMV, f32 acc, sigmoid.
// Lightning uses n_groups=8, topk_group=1, topk=6; this requested simple arm
// deliberately takes the global top-6 and reports the exact host/GPU sets.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
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

constexpr int kHidden = 2688;
constexpr int kExperts = 128;
constexpr int kGroups = 8;
constexpr int kTopKGroup = 1;
constexpr int kTopK = 6;
constexpr int kVl = 16;
constexpr int kExpertsPerWi = 4;
constexpr int kWg = 8;

struct MoeRouterName {};

int g_card = 0;
int g_spin = 4000;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "moe_router: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_x(sycl::half *p) {
    for (int i = 0; i < kHidden; ++i)
        p[i] = sycl::half(float(int((i * 17u + 3u) % 101u) - 50) / 100.f);
}

void fill_w(sycl::half *p) {
    for (int e = 0; e < kExperts; ++e)
        for (int k = 0; k < kHidden; ++k) {
            const float base = float(int((size_t(e) * 29u + size_t(k) * 13u + 7u) %
                                         97u) - 48) / 500.f;
            const float marker = (k == e) ? 0.25f + 0.002f * e : 0.f;
            p[size_t(e) * kHidden + k] = sycl::half(base + marker);
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

sycl::event launch(sycl::queue &q, const sycl::half *x,
                   const sycl::half *w, float *scores) {
    constexpr int nwi = kExperts / kExpertsPerWi;
    return q.parallel_for<MoeRouterName>(
        sycl::nd_range<1>({size_t(nwi)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int e0 = int(it.get_global_id(0)) * kExpertsPerWi;
            esimd::simd<float, kExpertsPerWi> acc(0.f);
            for (int k0 = 0; k0 < kHidden; k0 += kVl) {
                esimd::simd<sycl::half, kVl> xv;
                xv.copy_from(x + k0);
                const esimd::simd<float, kVl> xf = esimd::convert<float>(xv);
#pragma unroll
                for (int e = 0; e < kExpertsPerWi; ++e) {
                    esimd::simd<sycl::half, kVl> wv;
                    wv.copy_from(w + size_t(e0 + e) * kHidden + k0);
                    const esimd::simd<float, kVl> prod =
                        xf * esimd::convert<float>(wv);
#pragma unroll
                    for (int lane = 0; lane < kVl; ++lane)
                        acc[e] += prod[lane];
                }
            }
#pragma unroll
            for (int e = 0; e < kExpertsPerWi; ++e)
                acc[e] = 1.f / (1.f + esimd::exp(-acc[e]));
            acc.copy_to(scores + e0);
        });
}

void host_oracle(const sycl::half *x, const sycl::half *w, float *scores) {
    for (int e = 0; e < kExperts; ++e) {
        float acc = 0.f;
        for (int k = 0; k < kHidden; ++k)
            acc += float(x[k]) * float(w[size_t(e) * kHidden + k]);
        scores[e] = 1.f / (1.f + std::exp(-acc));
    }
}

std::array<int, kTopK> top6(const float *scores) {
    std::array<int, kExperts> order{};
    for (int i = 0; i < kExperts; ++i)
        order[i] = i;
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [&](int a, int b) {
                          return scores[a] == scores[b] ? a < b
                                                        : scores[a] > scores[b];
                      });
    std::array<int, kTopK> out{};
    std::copy_n(order.begin(), kTopK, out.begin());
    return out;
}

void print_top6(const char *label, const std::array<int, kTopK> &idx,
                const float *scores) {
    std::printf("%s indices=", label);
    for (int i = 0; i < kTopK; ++i)
        std::printf("%s%d", i ? "," : "", idx[i]);
    std::printf(" weights=");
    for (int i = 0; i < kTopK; ++i)
        std::printf("%s%.7f", i ? "," : "", scores[idx[i]]);
    std::printf("\n");
}

void run_shape(sycl::queue &q, int warmup, int iters, int *rc) {
    std::vector<sycl::half> hx(kHidden), hw(size_t(kExperts) * kHidden);
    std::vector<float> href(kExperts), hgot(kExperts);
    fill_x(hx.data());
    fill_w(hw.data());
    host_oracle(hx.data(), hw.data(), href.data());
    sycl::half *xd = sycl::malloc_device<sycl::half>(kHidden, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(hw.size(), q);
    float *sd = sycl::malloc_device<float>(kExperts, q);
    q.memcpy(xd, hx.data(), size_t(kHidden) * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), hw.size() * sizeof(sycl::half)).wait();
    auto go = [&]() { return launch(q, xd, wd, sd); };
    go().wait_and_throw();
    q.memcpy(hgot.data(), sd, size_t(kExperts) * sizeof(float)).wait();
    const auto ref_top = top6(href.data());
    const auto got_top = top6(hgot.data());
    print_top6("host_top6", ref_top, href.data());
    print_top6("gpu_top6", got_top, hgot.data());
    double dot = 0, got2 = 0, ref2 = 0, mx = 0;
    for (int i = 0; i < kExperts; ++i) {
        dot += double(hgot[i]) * href[i];
        got2 += double(hgot[i]) * hgot[i];
        ref2 += double(href[i]) * href[i];
        mx = std::max(mx, std::abs(double(hgot[i]) - href[i]));
    }
    const double cosine = dot / std::sqrt(got2 * ref2);
    const int set_ok = ref_top == got_top;
    const int ok = set_ok && cosine > 0.99;
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
    std::printf("phase,M,hidden,n_experts,n_groups,topk_group,topk,event_us,"
                "wait_host_us,pipe_host_us,cosine,max_abs,set_ok,ok,median_us,"
                "min_us,max_us\n");
    std::printf("timed,1,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.6f,%.5g,%d,%d,"
                "%.3f,%.3f,%.3f\n",
                kHidden, kExperts, kGroups, kTopKGroup, kTopK,
                double(event_ns) / 1000.0 / iters, wait_host_us, pipe_host_us,
                cosine, mx, set_ok, ok, median_of(samples),
                *std::min_element(samples.begin(), samples.end()),
                *std::max_element(samples.begin(), samples.end()));
    sycl::free(xd, q); sycl::free(wd, q); sycl::free(sd, q);
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
            std::fprintf(stderr, "moe_router [--warmup 50] [--iters 40] "
                                 "[--card 0] [--spin 4000] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "moe_router: unknown arg %s\n", a.c_str());
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
                "op=lightning_moe_router selection=global_top6 M=1 hidden=%d "
                "n_experts=%d n_groups=%d topk_group=%d topk=%d "
                "dtype=f16xf16_accf32_sigmoid rank=pipe_host_vs_eager "
                "warmup=%d iters=%d card=%d spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kHidden, kExperts,
                kGroups, kTopKGroup, kTopK, warmup, iters, g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

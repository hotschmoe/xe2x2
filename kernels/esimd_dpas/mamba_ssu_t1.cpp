// K8 ESIMD Mamba-2 SSD state-space update, decode T=1. Not GDN.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx.
//
// CONFIG prior: sibling SSU B8/W4 exists; GDN delta 7.1 is the wrong math.
// One work-item per head, d_state streamed in VL=16 tiles, f32 state/acc.
// Rank pipe_host vs eager napkin. CSV includes event and pipelined host time.

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

constexpr int kHeads = 64;
constexpr int kDHead = 64;
constexpr int kDState = 128;
constexpr int kGroups = 8;
constexpr int kHeadsPerGroup = kHeads / kGroups;
constexpr int kVl = 16;
constexpr int kWg = 16;

struct MambaSsuT1Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "mamba_ssu_t1: no GPU\n");
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

void fill_state(float *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = float(int((i * 29u + seed) % 201u) - 100) / 1000.f;
}

void fill_alog(float *p) {
    for (int h = 0; h < kHeads; ++h)
        p[h] = -4.f + 3.f * float((h * 13) % 64) / 63.f;
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

sycl::event launch(sycl::queue &q, const float *hin, const sycl::half *x,
                   const sycl::half *dt, const float *alog,
                   const sycl::half *b, const sycl::half *c, float *hout,
                   sycl::half *y) {
    return q.parallel_for<MambaSsuT1Name>(
        sycl::nd_range<1>({size_t(kHeads)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int head = int(it.get_global_id(0));
            const int group = head / kHeadsPerGroup;
            const float dtv = float(dt[head]);
            const float aval = -esimd::exp(alog[head]);
            const float decay = esimd::exp(dtv * aval);
            for (int i = 0; i < kDHead; ++i) {
                const float xv = float(x[head * kDHead + i]);
                float acc = 0.f;
#pragma unroll
                for (int j0 = 0; j0 < kDState; j0 += kVl) {
                    esimd::simd<float, kVl> hv;
                    esimd::simd<sycl::half, kVl> bv;
                    esimd::simd<sycl::half, kVl> cv;
                    const size_t state0 =
                        (size_t(head) * kDHead + size_t(i)) * kDState + j0;
                    hv.copy_from(hin + state0);
                    bv.copy_from(b + group * kDState + j0);
                    cv.copy_from(c + group * kDState + j0);
                    esimd::simd<float, kVl> hnew =
                        decay * hv + xv * dtv * esimd::convert<float>(bv);
                    hnew.copy_to(hout + state0);
                    const esimd::simd<float, kVl> prod =
                        hnew * esimd::convert<float>(cv);
#pragma unroll
                    for (int lane = 0; lane < kVl; ++lane)
                        acc += prod[lane];
                }
                y[head * kDHead + i] = sycl::half(acc);
            }
        });
}

void host_oracle(const float *hin, const sycl::half *x, const sycl::half *dt,
                 const float *alog, const sycl::half *b, const sycl::half *c,
                 float *hout, sycl::half *y) {
    for (int head = 0; head < kHeads; ++head) {
        const int group = head / kHeadsPerGroup;
        const float dtv = float(dt[head]);
        const float aval = -std::exp(alog[head]);
        const float decay = std::exp(dtv * aval);
        for (int i = 0; i < kDHead; ++i) {
            const float xv = float(x[head * kDHead + i]);
            float acc = 0.f;
            for (int j = 0; j < kDState; ++j) {
                const size_t idx =
                    (size_t(head) * kDHead + size_t(i)) * kDState + j;
                const float hnew = decay * hin[idx] +
                                   xv * (dtv * float(b[group * kDState + j]));
                hout[idx] = hnew;
                acc += hnew * float(c[group * kDState + j]);
            }
            y[head * kDHead + i] = sycl::half(acc);
        }
    }
}

void add_score(double got, double ref, double *dot, double *got2, double *ref2,
               double *mx) {
    const double d = std::abs(got - ref);
    *mx = std::max(*mx, d);
    *dot += got * ref;
    *got2 += got * got;
    *ref2 += ref * ref;
}

void run_shape(sycl::queue &q, int warmup, int iters, int *rc) {
    const size_t nx = size_t(kHeads) * kDHead;
    const size_t ns = nx * kDState;
    const size_t nbc = size_t(kGroups) * kDState;
    std::vector<float> hhin(ns), hhref(ns), hhgot(ns), halog(kHeads);
    std::vector<sycl::half> hx(nx), hdt(kHeads), hb(nbc), hc(nbc);
    std::vector<sycl::half> hyref(nx), hygot(nx);
    fill_state(hhin.data(), ns, 3);
    fill_f16(hx.data(), nx, 1);
    fill_f16(hdt.data(), kHeads, 5);
    fill_f16(hb.data(), nbc, 9);
    fill_f16(hc.data(), nbc, 13);
    fill_alog(halog.data());
    host_oracle(hhin.data(), hx.data(), hdt.data(), halog.data(), hb.data(),
                hc.data(), hhref.data(), hyref.data());

    float *hd_in = sycl::malloc_device<float>(ns, q);
    float *hd_out = sycl::malloc_device<float>(ns, q);
    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *dtd = sycl::malloc_device<sycl::half>(kHeads, q);
    float *ad = sycl::malloc_device<float>(kHeads, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q);
    q.memcpy(hd_in, hhin.data(), ns * sizeof(float)).wait();
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(dtd, hdt.data(), size_t(kHeads) * sizeof(sycl::half)).wait();
    q.memcpy(ad, halog.data(), size_t(kHeads) * sizeof(float)).wait();
    q.memcpy(bd, hb.data(), nbc * sizeof(sycl::half)).wait();
    q.memcpy(cd, hc.data(), nbc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, hd_in, xd, dtd, ad, bd, cd, hd_out, yd);
    };

    go().wait_and_throw();
    q.memcpy(hhgot.data(), hd_out, ns * sizeof(float)).wait();
    q.memcpy(hygot.data(), yd, nx * sizeof(sycl::half)).wait();
    double dot = 0, got2 = 0, ref2 = 0, mx = 0;
    for (size_t i = 0; i < ns; ++i)
        add_score(hhgot[i], hhref[i], &dot, &got2, &ref2, &mx);
    for (size_t i = 0; i < nx; ++i)
        add_score(float(hygot[i]), float(hyref[i]), &dot, &got2, &ref2, &mx);
    const double cosine =
        (got2 > 0 && ref2 > 0) ? dot / std::sqrt(got2 * ref2) : 0.0;
    const int ok = cosine > 0.99 ? 1 : 0;
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
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ns_sum += t1 - t0;
        all_us.push_back(double(t1 - t0) / 1000.0);
    }
    const auto host1 = std::chrono::steady_clock::now();
    const double wait_host_us =
        std::chrono::duration<double, std::micro>(host1 - host0).count() /
        double(iters);
    const auto pipe0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        (void)go();
    q.wait_and_throw();
    const auto pipe1 = std::chrono::steady_clock::now();
    const double pipe_host_us =
        std::chrono::duration<double, std::micro>(pipe1 - pipe0).count() /
        double(iters);
    int act1 = -1, cur1 = -1, thr1 = -1;
    char power1[32];
    sample_gt(&act1, &cur1, power1, sizeof(power1), &thr1);
    std::printf("timed_end act=%d cur=%d power=%s throttle=%d\n", act1, cur1,
                power1, thr1);
    const double event_us = double(ns_sum) / 1000.0 / double(iters);
    const double nbytes = double((2 * ns) * sizeof(float) +
                                 (2 * nx + size_t(kHeads) + 2 * nbc) *
                                     sizeof(sycl::half) +
                                 size_t(kHeads) * sizeof(float));
    const double gbs = (nbytes / 1.0e9) / (pipe_host_us * 1.0e-6);
    std::printf("phase,heads,d_head,d_state,groups,event_us,wait_host_us,"
                "pipe_host_us,GBs,cosine,max_abs,ok,median_us,min_us,max_us\n");
    std::printf("timed,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%d,%.3f,"
                "%.3f,%.3f\n",
                kHeads, kDHead, kDState, kGroups, event_us, wait_host_us,
                pipe_host_us, gbs, cosine, mx, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(hd_in, q);
    sycl::free(hd_out, q);
    sycl::free(xd, q);
    sycl::free(dtd, q);
    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
    sycl::free(yd, q);
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
        if (a == "--warmup")
            take(warmup);
        else if (a == "--iters")
            take(iters);
        else if (a == "--card")
            take(g_card);
        else if (a == "--spin")
            take(g_spin);
        else if (a == "--mhz")
            take(g_mhz);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "mamba_ssu_t1 [--warmup 50] [--iters 40] "
                         "[--card 0] [--spin 0] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "mamba_ssu_t1: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (warmup < 0 || iters < 1 || g_spin < 0) {
        std::fprintf(stderr, "mamba_ssu_t1: invalid iteration count\n");
        return 2;
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
                "op=mamba2_ssd_t1 dtype=h=f32_xdtbc_y=f16 heads=%d "
                "d_head=%d d_state=%d groups=%d VL=%d wi=one_per_head "
                "warmup=%d iters=%d card=%d spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kHeads, kDHead, kDState,
                kGroups, kVl, warmup, iters, g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

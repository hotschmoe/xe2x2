// K8 ESIMD Mamba decode: depthwise conv K=4 C=4096, then Mamba-2 SSU T=1.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Two in-order launches per go.
// This is the Mamba-2 SSU recurrence, not the GDN delta recurrence.
// Rank pipe_host against conv 4.355 us + SSU 80.064 us = 84.419 us.

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

constexpr int kChannels = 4096;
constexpr int kConv = 4;
constexpr int kConvState = 3;
constexpr int kHeads = 64;
constexpr int kDHead = 64;
constexpr int kDState = 128;
constexpr int kGroups = 8;
constexpr int kHeadsPerGroup = kHeads / kGroups;
constexpr int kVl = 16;
constexpr int kWg = 16;

static_assert(kChannels == kHeads * kDHead, "conv output must feed SSU x");

struct MambaFuseConvName {};
struct MambaFuseSsuName {};

int g_card = 0;
int g_spin = 4000;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "mamba_fuse_t1: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_f16(sycl::half *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = sycl::half(float(int((i * 17u + seed) % 201u) - 100) / 50.f);
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
        std::fclose(f);
        return;
    }
    std::fclose(f);
    size_t len = std::strlen(dst);
    while (len && (dst[len - 1] == '\n' || dst[len - 1] == '\r'))
        dst[--len] = 0;
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
    if (v.empty())
        return -1.0;
    std::sort(v.begin(), v.end());
    return v.size() & 1 ? v[v.size() / 2]
                        : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

sycl::event launch_conv(sycl::queue &q, const sycl::half *x,
                        const sycl::half *w, const sycl::half *state,
                        sycl::half *y, sycl::half *state_out) {
    return q.parallel_for<MambaFuseConvName>(
        sycl::nd_range<1>({size_t(kChannels / kVl)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int c0 = int(it.get_global_id(0)) * kVl;
            esimd::simd<sycl::half, kVl> xv;
            xv.copy_from(x + c0);
            esimd::simd<float, kVl> acc(0.f);
#pragma unroll
            for (int t = 0; t < kConvState; ++t) {
                esimd::simd<sycl::half, kVl> sv, wv;
                sv.copy_from(state + t * kChannels + c0);
                wv.copy_from(w + t * kChannels + c0);
                acc += esimd::convert<float>(sv) * esimd::convert<float>(wv);
                if (t > 0)
                    sv.copy_to(state_out + (t - 1) * kChannels + c0);
            }
            esimd::simd<sycl::half, kVl> wv;
            wv.copy_from(w + kConvState * kChannels + c0);
            acc += esimd::convert<float>(xv) * esimd::convert<float>(wv);
            esimd::simd<sycl::half, kVl> yh;
#pragma unroll
            for (int i = 0; i < kVl; ++i)
                yh[i] = sycl::half(float(acc[i]));
            yh.copy_to(y + c0);
            xv.copy_to(state_out + 2 * kChannels + c0);
        });
}

sycl::event launch_ssu(sycl::queue &q, const float *hin,
                       const sycl::half *x, const sycl::half *dt,
                       const float *alog, const sycl::half *b,
                       const sycl::half *c, float *hout, sycl::half *y) {
    return q.parallel_for<MambaFuseSsuName>(
        sycl::nd_range<1>({size_t(kHeads)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int head = int(it.get_global_id(0));
            const int group = head / kHeadsPerGroup;
            const float dtv = float(dt[head]);
            const float decay = esimd::exp(dtv * -esimd::exp(alog[head]));
            for (int i = 0; i < kDHead; ++i) {
                const float xv = float(x[head * kDHead + i]);
                float acc = 0.f;
#pragma unroll
                for (int j0 = 0; j0 < kDState; j0 += kVl) {
                    esimd::simd<float, kVl> hv;
                    esimd::simd<sycl::half, kVl> bv, cv;
                    const size_t s0 =
                        (size_t(head) * kDHead + size_t(i)) * kDState + j0;
                    hv.copy_from(hin + s0);
                    bv.copy_from(b + group * kDState + j0);
                    cv.copy_from(c + group * kDState + j0);
                    esimd::simd<float, kVl> hn =
                        decay * hv + xv * dtv * esimd::convert<float>(bv);
                    hn.copy_to(hout + s0);
                    const esimd::simd<float, kVl> prod =
                        hn * esimd::convert<float>(cv);
#pragma unroll
                    for (int lane = 0; lane < kVl; ++lane)
                        acc += prod[lane];
                }
                y[head * kDHead + i] = sycl::half(acc);
            }
        });
}

void host_conv(const sycl::half *x, const sycl::half *w,
               const sycl::half *state, sycl::half *y,
               sycl::half *state_out) {
    for (int i = 0; i < kChannels; ++i) {
        float acc = 0.f;
        for (int t = 0; t < kConvState; ++t)
            acc += float(state[t * kChannels + i]) *
                   float(w[t * kChannels + i]);
        acc += float(x[i]) * float(w[kConvState * kChannels + i]);
        y[i] = sycl::half(acc);
        state_out[i] = state[kChannels + i];
        state_out[kChannels + i] = state[2 * kChannels + i];
        state_out[2 * kChannels + i] = x[i];
    }
}

void host_ssu(const float *hin, const sycl::half *x, const sycl::half *dt,
              const float *alog, const sycl::half *b, const sycl::half *c,
              float *hout, sycl::half *y) {
    for (int head = 0; head < kHeads; ++head) {
        const int group = head / kHeadsPerGroup;
        const float dtv = float(dt[head]);
        const float decay = std::exp(dtv * -std::exp(alog[head]));
        for (int i = 0; i < kDHead; ++i) {
            const float xv = float(x[head * kDHead + i]);
            float acc = 0.f;
            for (int j = 0; j < kDState; ++j) {
                const size_t idx =
                    (size_t(head) * kDHead + size_t(i)) * kDState + j;
                const float hn = decay * hin[idx] +
                                 xv * dtv * float(b[group * kDState + j]);
                hout[idx] = hn;
                acc += hn * float(c[group * kDState + j]);
            }
            y[head * kDHead + i] = sycl::half(acc);
        }
    }
}

void add_score(double got, double ref, double *dot, double *got2,
               double *ref2, double *mx) {
    *dot += got * ref;
    *got2 += got * got;
    *ref2 += ref * ref;
    *mx = std::max(*mx, std::abs(got - ref));
}

void run_shape(sycl::queue &q, int warmup, int iters, int *rc) {
    const size_t nc = kChannels;
    const size_t ncs = size_t(kConvState) * nc;
    const size_t ns = nc * kDState;
    const size_t nbc = size_t(kGroups) * kDState;
    std::vector<sycl::half> hx(nc), hw(size_t(kConv) * nc), hcs(ncs);
    std::vector<sycl::half> hconv_ref(nc), hconv_got(nc), hcs_ref(ncs), hcs_got(ncs);
    std::vector<float> hhin(ns), hh_ref(ns), hh_got(ns), halog(kHeads);
    std::vector<sycl::half> hdt(kHeads), hb(nbc), hc(nbc), hy_ref(nc), hy_got(nc);
    fill_f16(hx.data(), nc, 1);
    fill_f16(hw.data(), hw.size(), 9);
    fill_f16(hcs.data(), ncs, 13);
    fill_state(hhin.data(), ns, 3);
    fill_f16(hdt.data(), kHeads, 5);
    fill_f16(hb.data(), nbc, 17);
    fill_f16(hc.data(), nbc, 23);
    fill_alog(halog.data());
    host_conv(hx.data(), hw.data(), hcs.data(), hconv_ref.data(), hcs_ref.data());
    host_ssu(hhin.data(), hconv_ref.data(), hdt.data(), halog.data(), hb.data(),
             hc.data(), hh_ref.data(), hy_ref.data());

    sycl::half *xd = sycl::malloc_device<sycl::half>(nc, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(hw.size(), q);
    sycl::half *csd = sycl::malloc_device<sycl::half>(ncs, q);
    sycl::half *convd = sycl::malloc_device<sycl::half>(nc, q);
    sycl::half *csod = sycl::malloc_device<sycl::half>(ncs, q);
    float *hid = sycl::malloc_device<float>(ns, q);
    float *hod = sycl::malloc_device<float>(ns, q);
    sycl::half *dtd = sycl::malloc_device<sycl::half>(kHeads, q);
    float *ad = sycl::malloc_device<float>(kHeads, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nc, q);
    q.memcpy(xd, hx.data(), nc * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), hw.size() * sizeof(sycl::half)).wait();
    q.memcpy(csd, hcs.data(), ncs * sizeof(sycl::half)).wait();
    q.memcpy(hid, hhin.data(), ns * sizeof(float)).wait();
    q.memcpy(dtd, hdt.data(), size_t(kHeads) * sizeof(sycl::half)).wait();
    q.memcpy(ad, halog.data(), size_t(kHeads) * sizeof(float)).wait();
    q.memcpy(bd, hb.data(), nbc * sizeof(sycl::half)).wait();
    q.memcpy(cd, hc.data(), nbc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        (void)launch_conv(q, xd, wd, csd, convd, csod);
        return launch_ssu(q, hid, convd, dtd, ad, bd, cd, hod, yd);
    };

    go().wait_and_throw();
    q.memcpy(hconv_got.data(), convd, nc * sizeof(sycl::half)).wait();
    q.memcpy(hcs_got.data(), csod, ncs * sizeof(sycl::half)).wait();
    q.memcpy(hh_got.data(), hod, ns * sizeof(float)).wait();
    q.memcpy(hy_got.data(), yd, nc * sizeof(sycl::half)).wait();
    double cdot = 0, cgot2 = 0, cref2 = 0, cmx = 0;
    for (size_t i = 0; i < nc; ++i) {
        add_score(float(hconv_got[i]), float(hconv_ref[i]), &cdot, &cgot2,
                  &cref2, &cmx);
    }
    for (size_t i = 0; i < ncs; ++i)
        add_score(float(hcs_got[i]), float(hcs_ref[i]), &cdot, &cgot2,
                  &cref2, &cmx);
    double sdot = 0, sgot2 = 0, sref2 = 0, smx = 0;
    for (size_t i = 0; i < nc; ++i)
        add_score(float(hy_got[i]), float(hy_ref[i]), &sdot, &sgot2, &sref2,
                  &smx);
    for (size_t i = 0; i < ns; ++i)
        add_score(hh_got[i], hh_ref[i], &sdot, &sgot2, &sref2, &smx);
    const double conv_cosine = cdot / std::sqrt(cgot2 * cref2);
    const double ssu_cosine = sdot / std::sqrt(sgot2 * sref2);
    const int ok = conv_cosine > 0.99 && ssu_cosine > 0.99;
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
    uint64_t last_ns = 0;
    std::vector<double> last_us;
    const auto wait0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        last_ns += t1 - t0;
        last_us.push_back(double(t1 - t0) / 1000.0);
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
    std::printf("phase,T,C,kconv,heads,d_head,d_state,groups,last_event_us,"
                "wait_host_us,pipe_host_us,baseline_us,conv_cosine,conv_max_abs,"
                "ssu_cosine,ssu_max_abs,ok,"
                "median_last_us,min_last_us,max_last_us\n");
    std::printf("timed,1,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,84.419,%.6f,"
                "%.5g,%.6f,%.5g,%d,%.3f,%.3f,%.3f\n",
                kChannels, kConv, kHeads, kDHead, kDState, kGroups,
                double(last_ns) / 1000.0 / iters, wait_host_us, pipe_host_us,
                conv_cosine, cmx, ssu_cosine, smx, ok, median_of(last_us),
                *std::min_element(last_us.begin(), last_us.end()),
                *std::max_element(last_us.begin(), last_us.end()));

    sycl::free(xd, q); sycl::free(wd, q); sycl::free(csd, q);
    sycl::free(convd, q); sycl::free(csod, q); sycl::free(hid, q);
    sycl::free(hod, q); sycl::free(dtd, q); sycl::free(ad, q);
    sycl::free(bd, q); sycl::free(cd, q); sycl::free(yd, q);
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
            std::fprintf(stderr, "mamba_fuse_t1 [--warmup 50] [--iters 40] "
                                 "[--card 0] [--spin 4000] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "mamba_fuse_t1: unknown arg %s\n", a.c_str());
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
                "op=mamba_conv_k4_then_mamba2_ssu_t1 launches=2 in_order=1 "
                "C=%d heads=%d d_head=%d d_state=%d groups=%d dtype=f16_f32 "
                "rank=pipe_host baseline_us=84.419 warmup=%d iters=%d card=%d "
                "spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kChannels, kHeads,
                kDHead, kDState, kGroups, warmup, iters, g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

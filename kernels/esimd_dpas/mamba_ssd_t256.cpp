// K8 ESIMD Mamba-2 SSD prefill, T=256, chunk_size=128. Not GDN delta.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. State carries serially over T.
// Rank pipe_host against the 256 * 80.064 us = 20496.384 us decode napkin.

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

constexpr int kT = 256;
constexpr int kChunk = 128;
constexpr int kHeads = 64;
constexpr int kDHead = 64;
constexpr int kDState = 128;
constexpr int kGroups = 8;
constexpr int kHeadsPerGroup = kHeads / kGroups;
constexpr int kVl = 16;
constexpr int kWg = 16;

struct MambaSsdT256Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "mamba_ssd_t256: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_small_f16(sycl::half *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = sycl::half(float(int((i * 17u + seed) % 201u) - 100) / 500.f);
}

void fill_dt(sycl::half *p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        p[i] = sycl::half(0.005f + 0.015f * float((i * 13u) % 64) / 63.f);
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
    if (v.empty())
        return -1.0;
    std::sort(v.begin(), v.end());
    return v.size() & 1 ? v[v.size() / 2]
                        : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

sycl::event launch(sycl::queue &q, const float *hin, const sycl::half *x,
                   const sycl::half *dt, const float *alog,
                   const sycl::half *b, const sycl::half *c, float *state,
                   sycl::half *y) {
    return q.parallel_for<MambaSsdT256Name>(
        sycl::nd_range<1>({size_t(kHeads)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int head = int(it.get_global_id(0));
            const int group = head / kHeadsPerGroup;
            const float aval = -esimd::exp(alog[head]);
            for (int i = 0; i < kDHead; ++i) {
                for (int chunk0 = 0; chunk0 < kT; chunk0 += kChunk) {
                    for (int t = chunk0; t < chunk0 + kChunk; ++t) {
                        const float dtv = float(dt[t * kHeads + head]);
                        const float decay = esimd::exp(dtv * aval);
                        const float xv =
                            float(x[(t * kHeads + head) * kDHead + i]);
                        float acc = 0.f;
#pragma unroll
                        for (int j0 = 0; j0 < kDState; j0 += kVl) {
                            const size_t s0 =
                                (size_t(head) * kDHead + size_t(i)) * kDState +
                                j0;
                            const size_t bc0 =
                                (size_t(t) * kGroups + size_t(group)) * kDState +
                                j0;
                            esimd::simd<float, kVl> hv;
                            esimd::simd<sycl::half, kVl> bv, cv;
                            if (t == 0)
                                hv.copy_from(hin + s0);
                            else
                                hv.copy_from(state + s0);
                            bv.copy_from(b + bc0);
                            cv.copy_from(c + bc0);
                            esimd::simd<float, kVl> hn =
                                decay * hv +
                                xv * dtv * esimd::convert<float>(bv);
                            hn.copy_to(state + s0);
                            const esimd::simd<float, kVl> prod =
                                hn * esimd::convert<float>(cv);
#pragma unroll
                            for (int lane = 0; lane < kVl; ++lane)
                                acc += prod[lane];
                        }
                        y[(t * kHeads + head) * kDHead + i] = sycl::half(acc);
                    }
                }
            }
        });
}

void host_oracle(const float *hin, const sycl::half *x, const sycl::half *dt,
                 const float *alog, const sycl::half *b, const sycl::half *c,
                 float *state, sycl::half *y) {
    const size_t ns = size_t(kHeads) * kDHead * kDState;
    std::copy_n(hin, ns, state);
    for (int head = 0; head < kHeads; ++head) {
        const int group = head / kHeadsPerGroup;
        const float aval = -std::exp(alog[head]);
        for (int i = 0; i < kDHead; ++i) {
            for (int chunk0 = 0; chunk0 < kT; chunk0 += kChunk) {
                for (int t = chunk0; t < chunk0 + kChunk; ++t) {
                    const float dtv = float(dt[t * kHeads + head]);
                    const float decay = std::exp(dtv * aval);
                    const float xv =
                        float(x[(t * kHeads + head) * kDHead + i]);
                    float acc = 0.f;
                    for (int j = 0; j < kDState; ++j) {
                        const size_t sidx =
                            (size_t(head) * kDHead + size_t(i)) * kDState + j;
                        const size_t bcidx =
                            (size_t(t) * kGroups + size_t(group)) * kDState + j;
                        const float hn = decay * state[sidx] +
                                         xv * dtv * float(b[bcidx]);
                        state[sidx] = hn;
                        acc += hn * float(c[bcidx]);
                    }
                    y[(t * kHeads + head) * kDHead + i] = sycl::half(acc);
                }
            }
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
    const size_t nx1 = size_t(kHeads) * kDHead;
    const size_t nx = size_t(kT) * nx1;
    const size_t ns = nx1 * kDState;
    const size_t nbc1 = size_t(kGroups) * kDState;
    const size_t nbc = size_t(kT) * nbc1;
    std::vector<float> hhin(ns), href_state(ns), hgot_state(ns), halog(kHeads);
    std::vector<sycl::half> hx(nx), hdt(size_t(kT) * kHeads), hb(nbc), hc(nbc);
    std::vector<sycl::half> href_y(nx), hgot_y(nx);
    fill_state(hhin.data(), ns, 3);
    fill_small_f16(hx.data(), nx, 1);
    fill_dt(hdt.data(), hdt.size());
    fill_small_f16(hb.data(), nbc, 9);
    fill_small_f16(hc.data(), nbc, 13);
    fill_alog(halog.data());
    const auto oracle0 = std::chrono::steady_clock::now();
    host_oracle(hhin.data(), hx.data(), hdt.data(), halog.data(), hb.data(),
                hc.data(), href_state.data(), href_y.data());
    const auto oracle1 = std::chrono::steady_clock::now();
    const double oracle_s = std::chrono::duration<double>(oracle1 - oracle0).count();

    float *hid = sycl::malloc_device<float>(ns, q);
    float *sd = sycl::malloc_device<float>(ns, q);
    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *dtd = sycl::malloc_device<sycl::half>(hdt.size(), q);
    float *ad = sycl::malloc_device<float>(kHeads, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nbc, q);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q);
    q.memcpy(hid, hhin.data(), ns * sizeof(float)).wait();
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(dtd, hdt.data(), hdt.size() * sizeof(sycl::half)).wait();
    q.memcpy(ad, halog.data(), size_t(kHeads) * sizeof(float)).wait();
    q.memcpy(bd, hb.data(), nbc * sizeof(sycl::half)).wait();
    q.memcpy(cd, hc.data(), nbc * sizeof(sycl::half)).wait();

    auto go = [&]() { return launch(q, hid, xd, dtd, ad, bd, cd, sd, yd); };
    go().wait_and_throw();
    q.memcpy(hgot_state.data(), sd, ns * sizeof(float)).wait();
    q.memcpy(hgot_y.data(), yd, nx * sizeof(sycl::half)).wait();
    double dot = 0, got2 = 0, ref2 = 0, mx = 0;
    for (size_t i = 0; i < ns; ++i)
        add_score(hgot_state[i], href_state[i], &dot, &got2, &ref2, &mx);
    for (size_t i = 0; i < nx; ++i)
        add_score(float(hgot_y[i]), float(href_y[i]), &dot, &got2, &ref2, &mx);
    const double cosine = dot / std::sqrt(got2 * ref2);
    const int ok = cosine > 0.99;
    if (!ok)
        *rc = 1;
    std::printf("oracle_s=%.3f score_y=full score_state=final samples=%zu\n",
                oracle_s, nx + ns);

    auto batch_wait = [&](int count) {
        for (int i = 0; i < count; ++i) {
            (void)go();
            if ((i & 31) == 31)
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
    std::vector<double> event_us_samples;
    const auto wait0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        event_ns += t1 - t0;
        event_us_samples.push_back(double(t1 - t0) / 1000.0);
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
    std::printf("phase,T,chunk_size,heads,d_head,d_state,groups,event_us,"
                "wait_host_us,pipe_host_us,napkin_us,cosine,max_abs,ok,"
                "median_us,min_us,max_us\n");
    std::printf("timed,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,20496.384,%.6f,"
                "%.5g,%d,%.3f,%.3f,%.3f\n",
                kT, kChunk, kHeads, kDHead, kDState, kGroups,
                double(event_ns) / 1000.0 / iters, wait_host_us, pipe_host_us,
                cosine, mx, ok, median_of(event_us_samples),
                *std::min_element(event_us_samples.begin(), event_us_samples.end()),
                *std::max_element(event_us_samples.begin(), event_us_samples.end()));

    sycl::free(hid, q); sycl::free(sd, q); sycl::free(xd, q);
    sycl::free(dtd, q); sycl::free(ad, q); sycl::free(bd, q);
    sycl::free(cd, q); sycl::free(yd, q);
}

} // namespace

int main(int argc, char **argv) {
    int warmup = 5, iters = 10;
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
            std::fprintf(stderr, "mamba_ssd_t256 [--warmup 5] [--iters 10] "
                                 "[--card 0] [--spin 0] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "mamba_ssd_t256: unknown arg %s\n", a.c_str());
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
                "op=mamba2_ssd_prefill serial_T=1 T=%d chunk_size=%d heads=%d "
                "d_head=%d d_state=%d groups=%d dtype=h=f32_xdtbc_y=f16 "
                "rank=pipe_host napkin_us=20496.384 warmup=%d iters=%d card=%d "
                "spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kT, kChunk, kHeads,
                kDHead, kDState, kGroups, warmup, iters, g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, warmup, iters, &rc);
    return rc;
}

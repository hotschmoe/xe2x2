// K8 ESIMD grouped routed-expert decode: six s8 GEMMs, M=1.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx.
//
// RC=4, K step=64 as two k32 DPAS loads, execN=16, wg=8x2 along N,
// NT=2, transformed B load, and constant W8A8 scale-to-f16 epilogue.
// One in-order queue chains every expert without an intervening host wait.
// Rank pipe_host. CSV reports both summed expert events and the last event.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
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
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

constexpr int kRc = 4;
constexpr int kKc = 32;
constexpr int kK64 = 64;
constexpr int kExecN = 16;
constexpr int kNt = 2;
constexpr int kTn = kNt * kExecN;
constexpr int kWgX = 8;
constexpr int kWgY = 2;
constexpr int kWgN = kWgX * kWgY;
constexpr float kScale = 0.02f;

struct MoeGroupS8M1Name {};

int g_card = 0;
int g_spin = 4000;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "moe_group_s8_m1: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_s8(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 129u) - 64);
}

size_t round_up(int64_t n, int w) {
    return size_t(((n + int64_t(w) - 1) / int64_t(w)) * int64_t(w));
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

sycl::event launch(sycl::queue &q, const int8_t *a, const int8_t *b,
                   sycl::half *out, int rows, int n, int k) {
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = n / kTn;
    const size_t n_wgs = round_up(n_groups, kWgN) / size_t(kWgN);
    const size_t g0 = n_wgs * size_t(kWgX);
    const size_t g1 = size_t(m_blocks) * size_t(kWgY);
    return q.parallel_for<MoeGroupS8M1Name>(
        sycl::nd_range<2>({g0, g1}, {size_t(kWgX), size_t(kWgY)}),
        [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int64_t ng = int64_t(it.get_group(0)) * kWgN +
                               int64_t(it.get_local_id(1)) * kWgX +
                               int64_t(it.get_local_id(0));
            const int64_t mb = int64_t(it.get_group(1));
            if (ng >= n_groups || mb >= m_blocks)
                return;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kTn);
            esimd::simd<int32_t, kRc * kExecN> acc[kNt];
#pragma unroll
            for (int t = 0; t < kNt; ++t)
                acc[t] = 0;
            for (int k0 = 0; k0 < k; k0 += kK64) {
                const esimd::simd<int8_t, kRc * kKc> a0 =
                    xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                        a, unsigned(k - 1), unsigned(rows - 1), unsigned(k - 1),
                        k0, row0);
                const esimd::simd<int8_t, kRc * kKc> a1 =
                    xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                        a, unsigned(k - 1), unsigned(rows - 1), unsigned(k - 1),
                        k0 + kKc, row0);
#pragma unroll
                for (int t = 0; t < kNt; ++t) {
                    const esimd::simd<int8_t, kKc * kExecN> b0 =
                        xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false, true>(
                            b, unsigned(n - 1), unsigned(k - 1), unsigned(n - 1),
                            col0 + t * kExecN, k0);
                    acc[t] = esimd::xmx::dpas<8, kRc, int32_t, int32_t, int8_t,
                                              int8_t>(acc[t], b0, a0);
                    const esimd::simd<int8_t, kKc * kExecN> b1 =
                        xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false, true>(
                            b, unsigned(n - 1), unsigned(k - 1), unsigned(n - 1),
                            col0 + t * kExecN, k0 + kKc);
                    acc[t] = esimd::xmx::dpas<8, kRc, int32_t, int32_t, int8_t,
                                              int8_t>(acc[t], b1, a1);
                }
            }
#pragma unroll
            for (int t = 0; t < kNt; ++t) {
                esimd::simd<sycl::half, kRc * kExecN> h;
#pragma unroll
                for (int i = 0; i < kRc * kExecN; ++i)
                    h[i] = sycl::half(float(acc[t][i]) * kScale * kScale);
                xesimd::lsc_store_2d<sycl::half, kExecN, kRc>(
                    out, unsigned(n * int(sizeof(sycl::half)) - 1),
                    unsigned(rows - 1),
                    unsigned(n * int(sizeof(sycl::half)) - 1),
                    col0 + t * kExecN, row0, h);
            }
        });
}

void run_shape(sycl::queue &q, int experts, int m, int n, int k, int warmup,
               int iters, int *rc) {
    if (experts < 1 || m < 1 || n < kTn || n % kTn != 0 || k < kK64 ||
        k % kK64 != 0) {
        std::fprintf(stderr,
                     "moe_group_s8_m1: shape experts=%d m=%d n=%d k=%d\n",
                     experts, m, n, k);
        *rc = 2;
        return;
    }
    const int rows = ((m + kRc - 1) / kRc) * kRc;
    const size_t na = size_t(rows) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc_pad = size_t(rows) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na, 0), hb(size_t(experts) * nb);
    std::vector<sycl::half> href(size_t(experts) * nc);
    std::vector<sycl::half> hgot(size_t(experts) * nc);
    std::vector<sycl::half> hpad(size_t(experts) * nc_pad);
    fill_s8(ha.data(), size_t(m) * size_t(k), 1);
    for (int e = 0; e < experts; ++e) {
        int8_t *be = hb.data() + size_t(e) * nb;
        fill_s8(be, nb, unsigned(9 + 23 * e));
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                int32_t acc = 0;
                for (int kk = 0; kk < k; ++kk)
                    acc += int32_t(ha[size_t(row) * k + kk]) *
                           int32_t(be[size_t(kk) * n + col]);
                href[size_t(e) * nc + size_t(row) * n + col] =
                    sycl::half(float(acc) * kScale * kScale);
            }
        }
    }

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd = sycl::malloc_device<int8_t>(size_t(experts) * nb, q);
    sycl::half *cd =
        sycl::malloc_device<sycl::half>(size_t(experts) * nc_pad, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(bd, hb.data(), size_t(experts) * nb).wait();

    auto go = [&](std::vector<sycl::event> *events) -> sycl::event {
        sycl::event last;
        if (events)
            events->clear();
        for (int e = 0; e < experts; ++e) {
            last = launch(q, ad, bd + size_t(e) * nb,
                          cd + size_t(e) * nc_pad, rows, n, k);
            if (events)
                events->push_back(last);
        }
        return last;
    };

    go(nullptr).wait_and_throw();
    q.memcpy(hpad.data(), cd,
             size_t(experts) * nc_pad * sizeof(sycl::half)).wait();
    for (int e = 0; e < experts; ++e)
        for (int row = 0; row < m; ++row)
            std::copy_n(hpad.data() + size_t(e) * nc_pad + size_t(row) * n, n,
                        hgot.data() + size_t(e) * nc + size_t(row) * n);
    double dot = 0, got2 = 0, ref2 = 0, mx = 0;
    for (size_t i = 0; i < hgot.size(); ++i) {
        const double a = float(hgot[i]);
        const double b = float(href[i]);
        dot += a * b;
        got2 += a * a;
        ref2 += b * b;
        mx = std::max(mx, std::abs(a - b));
    }
    const double cosine =
        (got2 > 0 && ref2 > 0) ? dot / std::sqrt(got2 * ref2) : 0.0;
    const int ok = cosine > 0.99 ? 1 : 0;
    if (!ok)
        *rc = 1;

    auto batch_wait = [&](int np) {
        constexpr int kBatch = 64;
        for (int i = 0; i < np; ++i) {
            (void)go(nullptr);
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

    uint64_t sum_event_ns = 0;
    uint64_t last_event_ns = 0;
    std::vector<double> all_sum_us;
    std::vector<sycl::event> events;
    events.reserve(size_t(experts));
    int act0 = -1, cur0 = -1, thr0 = -1;
    char power0[32];
    sample_gt(&act0, &cur0, power0, sizeof(power0), &thr0);
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0, cur0,
                power0, thr0);
    const auto host0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        sycl::event last = go(&events);
        last.wait_and_throw();
        uint64_t iter_ns = 0;
        for (const sycl::event &e : events) {
            const uint64_t t0 = e.get_profiling_info<
                sycl::info::event_profiling::command_start>();
            const uint64_t t1 = e.get_profiling_info<
                sycl::info::event_profiling::command_end>();
            iter_ns += t1 - t0;
        }
        const uint64_t lt0 = last.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        const uint64_t lt1 = last.get_profiling_info<
            sycl::info::event_profiling::command_end>();
        sum_event_ns += iter_ns;
        last_event_ns += lt1 - lt0;
        all_sum_us.push_back(double(iter_ns) / 1000.0);
    }
    const auto host1 = std::chrono::steady_clock::now();
    const double wait_host_us =
        std::chrono::duration<double, std::micro>(host1 - host0).count() /
        double(iters);
    const auto pipe0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        (void)go(nullptr);
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
    const double event_us = double(sum_event_ns) / 1000.0 / double(iters);
    const double last_us = double(last_event_ns) / 1000.0 / double(iters);
    const double ops = 2.0 * double(experts) * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (pipe_host_us * 1.0e-6);
    std::printf("phase,experts,m,n,k,event_us,last_event_us,wait_host_us,"
                "pipe_host_us,TOPS,cosine,max_abs,ok,median_sum_us,min_sum_us,"
                "max_sum_us\n");
    std::printf("timed,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.6f,%.5g,%d,"
                "%.3f,%.3f,%.3f\n",
                experts, m, n, k, event_us, last_us, wait_host_us, pipe_host_us,
                tops, cosine, mx, ok, median_of(all_sum_us),
                all_sum_us.empty()
                    ? -1.0
                    : *std::min_element(all_sum_us.begin(), all_sum_us.end()),
                all_sum_us.empty()
                    ? -1.0
                    : *std::max_element(all_sum_us.begin(), all_sum_us.end()));

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int experts = 6, m = 1, n = 1856, k = 2688;
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
        if (a == "--experts")
            take(experts);
        else if (a == "--m")
            take(m);
        else if (a == "--n")
            take(n);
        else if (a == "--k")
            take(k);
        else if (a == "--warmup")
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
                         "moe_group_s8_m1 [--experts 6] [--m 1] [--n 1856] "
                         "[--k 2688] [--warmup 50] [--iters 40] [--card 0] "
                         "[--spin 4000] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "moe_group_s8_m1: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (warmup < 0 || iters < 1 || g_spin < 0) {
        std::fprintf(stderr, "moe_group_s8_m1: invalid iteration count\n");
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
                "op=moe_group_s8_m1 dtype=s8xs8_to_f16 experts=%d m=%d n=%d "
                "k=%d scale=%.4f RC=%d NT=%d kstep=%d wg=%dx%d_alongN "
                "warmup=%d iters=%d card=%d spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), experts, m, n, k,
                double(kScale), kRc, kNt, kK64, kWgX, kWgY, warmup, iters,
                g_card, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, experts, m, n, k, warmup, iters, &rc);
    return rc;
}

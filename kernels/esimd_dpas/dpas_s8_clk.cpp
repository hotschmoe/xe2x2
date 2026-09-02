// K2 ESIMD s8 RC=4, 64 dpas.8x4, 8x2 along N, per-iter GT clocks.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// CONFIG prior: heat-then-decode 1024^3 dropped cur to ~750 MHz and
// the hold was 61-65 us. Warm wgn M=4 is 47-50 vs W8A8 M=1 42-46.
// Steal: no long heat. Tight decode loop. Optional in-order prime
// launches before each timed event so RPS stays up DURING the event.
// Sample act/cur/power after each wait. Rank median us at cur>=mhz.
// Pad M=1 to RC=4. Oracle is the real M rows.
// CSV: iter,us,act,cur,power,throttle

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
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
constexpr int kWgX = 8;
constexpr int kWgY = 2;
constexpr int kWgN = kWgX * kWgY;

struct DpasS8ClkNt2Name {};
struct DpasS8ClkNt4Name {};

int g_card = 0;
int g_prime = 0;
int g_mhz = 2400;
int g_spin = 0;
int g_sample_every = 0;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s8_clk: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_s8(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 255u) - 128);
}

void host_s32(const int8_t *a, const int8_t *b, int32_t *c, int m, int n,
              int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk)
                acc += int32_t(a[i * k + kk]) * int32_t(b[kk * n + j]);
            c[i * n + j] = acc;
        }
}

static size_t round_up(int64_t n, int w) {
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

template <typename Name, int NT, int kUnroll>
sycl::event launch(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                   int32_t *cd, int rows, int cols, int dk) {
    constexpr int kTN = NT * kExecN;
    constexpr int kInnerK = kUnroll * kK64;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kTN;
    const size_t n_wgs = round_up(n_groups, kWgN) / size_t(kWgN);
    const size_t g0 = n_wgs * size_t(kWgX);
    const size_t g1 = size_t(m_blocks) * size_t(kWgY);
    return q.parallel_for<Name>(
        sycl::nd_range<2>({g0, g1}, {size_t(kWgX), size_t(kWgY)}),
        [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int64_t ng = int64_t(it.get_group(0)) * kWgN +
                               int64_t(it.get_local_id(1)) * kWgX +
                               int64_t(it.get_local_id(0));
            const int64_t mb = int64_t(it.get_group(1));
            if (ng >= n_groups || mb >= m_blocks)
                return;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kTN);
            esimd::simd<int32_t, kRc * kExecN> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t)
                acc[t] = 0;
            const int outer = dk / kInnerK;
            for (int o = 0; o < outer; ++o) {
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    const int k0 = o * kInnerK + u * kK64;
                    const esimd::simd<int8_t, kRc * kKc> a0 =
                        xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                            ad, unsigned(dk - 1), unsigned(rows - 1),
                            unsigned(dk - 1), k0, row0);
                    const esimd::simd<int8_t, kRc * kKc> a1 =
                        xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                            ad, unsigned(dk - 1), unsigned(rows - 1),
                            unsigned(dk - 1), k0 + kKc, row0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const esimd::simd<int8_t, kKc * kExecN> b0 =
                            xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false,
                                                true>(
                                bd, unsigned(cols - 1), unsigned(dk - 1),
                                unsigned(cols - 1), col0 + t * kExecN, k0);
                        acc[t] = esimd::xmx::dpas<8, kRc, int32_t, int32_t,
                                                  int8_t, int8_t>(acc[t], b0,
                                                                  a0);
                        const esimd::simd<int8_t, kKc * kExecN> b1 =
                            xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false,
                                                true>(
                                bd, unsigned(cols - 1), unsigned(dk - 1),
                                unsigned(cols - 1), col0 + t * kExecN,
                                k0 + kKc);
                        acc[t] = esimd::xmx::dpas<8, kRc, int32_t, int32_t,
                                                  int8_t, int8_t>(acc[t], b1,
                                                                  a1);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t)
                xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                    cd, unsigned(cols * int(sizeof(int32_t)) - 1),
                    unsigned(rows - 1),
                    unsigned(cols * int(sizeof(int32_t)) - 1),
                    col0 + t * kExecN, row0, acc[t]);
        });
}

int max_abs_diff(const int32_t *got, const int32_t *ref, size_t n) {
    int mx = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = got[i] > ref[i] ? got[i] - ref[i] : ref[i] - got[i];
        if (d > mx)
            mx = d;
    }
    return mx;
}

void run_shape(sycl::queue &q, int nt, int unroll, const char *phase, int m,
               int n, int k, int warmup, int iters, int *rc, int per_iter) {
    const int tn = nt * kExecN;
    const int inner_k = unroll * kK64;
    if (m < 1 || n % tn != 0 || k % inner_k != 0) {
        std::fprintf(stderr,
                     "dpas_s8_clk: shape m=%d n=%d k=%d nt=%d unroll=%d\n", m, n,
                     k, nt, unroll);
        *rc = 2;
        return;
    }
    const int rows = ((m + kRc - 1) / kRc) * kRc;
    const size_t na = size_t(rows) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc_pad = size_t(rows) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na, 0), hb(nb);
    std::vector<int32_t> href(nc), hgot(nc), hpad(nc_pad);
    fill_s8(ha.data(), size_t(m) * size_t(k), 1);
    fill_s8(hb.data(), nb, 9);
    host_s32(ha.data(), hb.data(), href.data(), m, n, k);

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd = sycl::malloc_device<int8_t>(nb, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc_pad, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(bd, hb.data(), nb).wait();

    auto go = [&]() -> sycl::event {
        if (nt == 4)
            return launch<DpasS8ClkNt4Name, 4, 8>(q, ad, bd, cd, rows, n, k);
        return launch<DpasS8ClkNt2Name, 2, 16>(q, ad, bd, cd, rows, n, k);
    };

    go().wait_and_throw();
    q.memcpy(hpad.data(), cd, nc_pad * sizeof(int32_t)).wait();
    for (size_t i = 0; i < nc; ++i)
        hgot[i] = hpad[i];
    const int max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    const int ok = (max_abs == 0) ? 1 : 0;
    if (!ok)
        *rc = 1;

    auto batch_wait = [&](int n) {
        constexpr int kBatch = 256;
        for (int i = 0; i < n; ++i) {
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
    std::vector<double> high_us;
    std::vector<double> all_cur;
    uint64_t ns_sum = 0;
    int act0 = -1, cur0 = -1, thr0 = -1;
    char power0[32];
    sample_gt(&act0, &cur0, power0, sizeof(power0), &thr0);
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0, cur0,
                power0, thr0);
    if (per_iter && g_sample_every > 0)
        std::printf("iter,us,act,cur,power,throttle\n");
    const auto host0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        for (int p = 0; p < g_prime; ++p)
            (void)go();
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        const double us = double(t1 - t0) / 1000.0;
        ns_sum += (t1 - t0);
        all_us.push_back(us);
        if (g_sample_every > 0 && (i % g_sample_every) == 0) {
            int act = -1, cur = -1, throttle = -1;
            char power[32];
            sample_gt(&act, &cur, power, sizeof(power), &throttle);
            all_cur.push_back(double(cur));
            if (cur >= g_mhz)
                high_us.push_back(us);
            if (per_iter)
                std::printf("%d,%.3f,%d,%d,%s,%d\n", i, us, act, cur, power,
                            throttle);
        }
    }
    const auto host1 = std::chrono::steady_clock::now();
    const double host_us =
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
    const double ops = 2.0 * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    std::printf("phase,nt,unroll,m,n,k,event_us,wait_host_us,pipe_host_us,TOPS,"
                "max_abs,ok,median_us,min_us,max_us,median_cur,n_cur_ge_mhz,"
                "median_us_high,mhz,prime\n");
    std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.4f,%d,%d,%.3f,%.3f,%.3f,%.1f,"
                "%zu,%.3f,%d,%d\n",
                phase, nt, unroll, m, n, k, us, host_us, pipe_us, tops, max_abs,
                ok, median_of(all_us),
                all_us.empty() ? -1.0 : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0 : *std::max_element(all_us.begin(), all_us.end()),
                median_of(all_cur), high_us.size(), median_of(high_us), g_mhz,
                g_prime);
    std::printf("# note in-loop sysfs only if sample_every>0; duty-cycle "
                "clocks live in the .freq log\n");

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int nt = 2;
    int timed_m = 1, timed_n = 5120, timed_k = 5120;
    int warmup = 30, iters = 40;
    const char *aff = std::getenv("ZE_AFFINITY_MASK");
    if (aff && aff[0])
        g_card = std::atoi(aff);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--nt")
            take(nt);
        else if (a == "--m")
            take(timed_m);
        else if (a == "--n")
            take(timed_n);
        else if (a == "--k")
            take(timed_k);
        else if (a == "--iters")
            take(iters);
        else if (a == "--warmup")
            take(warmup);
        else if (a == "--card")
            take(g_card);
        else if (a == "--prime")
            take(g_prime);
        else if (a == "--mhz")
            take(g_mhz);
        else if (a == "--spin")
            take(g_spin);
        else if (a == "--sample-every")
            take(g_sample_every);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "dpas_s8_clk --nt 2|4 [--m 1] [--n 5120] [--k 5120] "
                         "[--prime 0] [--spin 0] [--sample-every 0] "
                         "[--card 0] [--mhz 2400]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s8_clk: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (nt != 2 && nt != 4) {
        std::fprintf(stderr, "dpas_s8_clk: --nt must be 2 or 4\n");
        return 2;
    }
    const int unroll = (nt == 4) ? 8 : 16;
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s dtype=s8xs8s32 "
                "RC=%d K64=%d execN=%d NT=%d tileN=%d unroll=%d dpas=%d "
                "innerK=%d wg=%dx%d_alongN padM=RC Bload=Transformed warmup=%d "
                "iters=%d card=%d prime=%d spin=%d mhz=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kRc, kK64, kExecN, nt,
                nt * kExecN, unroll, 2 * nt * unroll, unroll * kK64, kWgX, kWgY,
                warmup, iters, g_card, g_prime, g_spin, g_mhz);
    int rc = 0;
    run_shape(q, nt, unroll, "check", kRc, nt * kExecN, unroll * kK64, 1, 1, &rc,
              0);
    run_shape(q, nt, unroll, "timed", timed_m, timed_n, timed_k, warmup, iters,
              &rc, 1);
    return rc;
}

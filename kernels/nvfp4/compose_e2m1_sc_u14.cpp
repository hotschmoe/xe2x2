// K8 E2M1 two-term s4 compose on the s4 decode tile.
// NT=2 unroll=14 makes inner_k=896 divide Lightning hidden 2688.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// HAIL-MARY. Never bitcast E2M1 onto s4 DPAS. Split q=2*E2M1 to
// w_lo+8*w_hi, both in s4 [-8,7]. A is s4. Two s4xs4 DPAS per k64
// on the K2 RC=4 8x2-N scale-to-f16 tile. Rank pipe_host vs s4 16.5,
// s8 34, nibble_lut_sc 158.

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
namespace xmx = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int kRc = 4;
constexpr int kKc = 64;
constexpr int kK64 = 64;
constexpr int kExecN = 16;
constexpr int kWgX = 8;
constexpr int kWgY = 2;
constexpr int kWgN = kWgX * kWgY;
constexpr int kPack = 2;
constexpr int kAPacked = kKc / kPack;
constexpr int kBPackedH = kKc / kPack;
constexpr float kScale = 0.02f;
constexpr int8_t kMag2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

struct DpasE2M1ScU14Nt2Name {};
struct DpasE2M1ScU14Nt4Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "compose_e2m1_sc_u14: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

uint8_t pack_s4(int8_t lo, int8_t hi) {
    return uint8_t((uint8_t(lo) & 0xf) | ((uint8_t(hi) & 0xf) << 4));
}

void fill_s4(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 16u) - 8);
}

inline int8_t e2m1_nibble_to_q(uint8_t nib) {
    int8_t q = kMag2[nib & 7];
    if (nib & 8)
        q = int8_t(-q);
    return q;
}

void split_q(int8_t q, int8_t *lo, int8_t *hi) {
    if (q >= 8) {
        *hi = 1;
        *lo = int8_t(q - 8);
    } else if (q <= -8) {
        *hi = -1;
        *lo = int8_t(q + 8);
    } else {
        *hi = 0;
        *lo = q;
    }
}

void fill_e2m1_split(int8_t *q, int8_t *lo, int8_t *hi, size_t n,
                     unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        q[i] = e2m1_nibble_to_q(uint8_t((i * 13u + seed) & 15u));
        split_q(q[i], &lo[i], &hi[i]);
    }
}

void pack_a(const int8_t *a, uint8_t *out, int m, int k) {
    const int kp = k / kPack;
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; kk += kPack)
            out[i * kp + kk / kPack] = pack_s4(a[i * k + kk], a[i * k + kk + 1]);
}

void pack_b(const int8_t *b, uint8_t *out, int k, int n) {
    for (int kk = 0; kk < k; kk += kPack)
        for (int j = 0; j < n; ++j)
            out[(kk / kPack) * n + j] = pack_s4(b[kk * n + j], b[(kk + 1) * n + j]);
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
sycl::event launch(sycl::queue &q, const uint8_t *ad, const uint8_t *bd_lo,
                   const uint8_t *bd_hi, const float *asd, const float *bsd,
                   sycl::half *cd, int rows, int cols, int dk) {
    constexpr int kTN = NT * kExecN;
    constexpr int kInnerK = kUnroll * kK64;
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
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
            esimd::simd<int32_t, kRc * kExecN> acc_lo[NT];
            esimd::simd<int32_t, kRc * kExecN> acc_hi[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc_lo[t] = 0;
                acc_hi[t] = 0;
            }
            const int outer = dk / kInnerK;
            for (int o = 0; o < outer; ++o) {
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    const int k0 = o * kInnerK + u * kK64;
                    const int pk0 = k0 / kPack;
                    const esimd::simd<uint8_t, kRc * kAPacked> a0 =
                        xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                            ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                            unsigned(a_pitch - 1), pk0, row0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const esimd::simd<uint8_t, kBPackedH * kExecN> blo =
                            xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1,
                                                false, true>(
                                bd_lo, unsigned(cols - 1), unsigned(b_rows - 1),
                                unsigned(cols - 1), col0 + t * kExecN, pk0);
                        acc_lo[t] = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t,
                                              uint8_t,
                                              xmx::dpas_argument_type::s4,
                                              xmx::dpas_argument_type::s4>(
                            acc_lo[t], blo, a0);
                        const esimd::simd<uint8_t, kBPackedH * kExecN> bhi =
                            xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1,
                                                false, true>(
                                bd_hi, unsigned(cols - 1), unsigned(b_rows - 1),
                                unsigned(cols - 1), col0 + t * kExecN, pk0);
                        acc_hi[t] = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t,
                                              uint8_t,
                                              xmx::dpas_argument_type::s4,
                                              xmx::dpas_argument_type::s4>(
                            acc_hi[t], bhi, a0);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t)
                acc_lo[t] = acc_lo[t] + acc_hi[t] * 8;
            esimd::simd<float, kRc> asv;
            asv.copy_from(asd + row0);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                esimd::simd<float, kExecN> bsv;
                bsv.copy_from(bsd + col0 + t * kExecN);
                esimd::simd<float, kRc * kExecN> f(acc_lo[t]);
#pragma unroll
                for (int r = 0; r < kRc; ++r) {
                    const float ar = asv[r];
#pragma unroll
                    for (int c = 0; c < kExecN; ++c)
                        f[r * kExecN + c] *= ar * bsv[c];
                }
                esimd::simd<sycl::half, kRc * kExecN> h;
#pragma unroll
                for (int i = 0; i < kRc * kExecN; ++i)
                    h[i] = sycl::half(float(f[i]));
                xesimd::lsc_store_2d<sycl::half, kExecN, kRc>(
                    cd, unsigned(cols * int(sizeof(sycl::half)) - 1),
                    unsigned(rows - 1),
                    unsigned(cols * int(sizeof(sycl::half)) - 1),
                    col0 + t * kExecN, row0, h);
            }
        });
}

void run_shape(sycl::queue &q, int nt, int unroll, const char *phase, int m,
               int n, int k, int warmup, int iters, int *rc, int do_spin) {
    const int tn = nt * kExecN;
    const int inner_k = unroll * kK64;
    if (m < 1 || n % tn != 0 || k % inner_k != 0 || k % kPack != 0) {
        std::fprintf(stderr,
                     "compose_e2m1_sc_u14: shape m=%d n=%d k=%d nt=%d unroll=%d\n", m, n,
                     k, nt, unroll);
        *rc = 2;
        return;
    }
    const int rows = ((m + kRc - 1) / kRc) * kRc;
    const size_t na = size_t(rows) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t na_p = size_t(rows) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    const size_t nc_pad = size_t(rows) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na, 0), hq(nb), hlo(nb), hhi(nb);
    std::vector<uint8_t> pa(na_p), pb_lo(nb_p), pb_hi(nb_p);
    std::vector<float> has(size_t(rows), 0.f), hbs(size_t(n), kScale);
    std::vector<sycl::half> href(nc), hgot(nc), hpad(nc_pad);
    fill_s4(ha.data(), size_t(m) * size_t(k), 1);
    fill_e2m1_split(hq.data(), hlo.data(), hhi.data(), nb, 9);
    pack_a(ha.data(), pa.data(), rows, k);
    pack_b(hlo.data(), pb_lo.data(), k, n);
    pack_b(hhi.data(), pb_hi.data(), k, n);
    for (int i = 0; i < m; ++i)
        has[size_t(i)] = kScale;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int kk = 0; kk < k; ++kk)
                acc += float(ha[size_t(i) * size_t(k) + size_t(kk)]) *
                       float(hq[size_t(kk) * size_t(n) + size_t(j)]);
            href[size_t(i) * size_t(n) + size_t(j)] =
                sycl::half(acc * has[size_t(i)] * hbs[size_t(j)]);
        }
    }

    uint8_t *ad = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *bd_lo = sycl::malloc_device<uint8_t>(nb_p, q);
    uint8_t *bd_hi = sycl::malloc_device<uint8_t>(nb_p, q);
    float *asd = sycl::malloc_device<float>(size_t(rows), q);
    float *bsd = sycl::malloc_device<float>(size_t(n), q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nc_pad, q);
    q.memcpy(ad, pa.data(), na_p).wait();
    q.memcpy(bd_lo, pb_lo.data(), nb_p).wait();
    q.memcpy(bd_hi, pb_hi.data(), nb_p).wait();
    q.memcpy(asd, has.data(), size_t(rows) * sizeof(float)).wait();
    q.memcpy(bsd, hbs.data(), size_t(n) * sizeof(float)).wait();

    auto go = [&]() -> sycl::event {
        if (nt == 4)
            return launch<DpasE2M1ScU14Nt4Name, 4, 8>(q, ad, bd_lo, bd_hi, asd,
                                                   bsd, cd, rows, n, k);
        return launch<DpasE2M1ScU14Nt2Name, 2, 14>(q, ad, bd_lo, bd_hi, asd,
                                                   bsd, cd, rows, n, k);
    };

    go().wait_and_throw();
    q.memcpy(hpad.data(), cd, nc_pad * sizeof(sycl::half)).wait();
    for (size_t i = 0; i < nc; ++i)
        hgot[i] = hpad[i];
    double dot = 0, na2 = 0, nb2 = 0, mx = 0;
    for (size_t i = 0; i < nc; ++i) {
        const float x = float(hgot[i]);
        const float y = float(href[i]);
        const float d = x > y ? x - y : y - x;
        if (d > mx)
            mx = d;
        dot += double(x) * double(y);
        na2 += double(x) * double(x);
        nb2 += double(y) * double(y);
    }
    const double cosine =
        (na2 > 0 && nb2 > 0) ? dot / std::sqrt(na2 * nb2) : 0.0;
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
    const double ops = 2.0 * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    std::printf("phase,nt,unroll,m,n,k,event_us,wait_host_us,pipe_host_us,TOPS,"
                "cosine,max_abs,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.4f,%.6f,%.5g,%d,%.3f,%.3f,"
                "%.3f\n",
                phase, nt, unroll, m, n, k, us, wait_host, pipe_us, tops,
                cosine, mx, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(ad, q);
    sycl::free(bd_lo, q);
    sycl::free(bd_hi, q);
    sycl::free(asd, q);
    sycl::free(bsd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int nt = 2;
    int timed_m = 1, timed_n = 5120, timed_k = 5120;
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
        else if (a == "--spin")
            take(g_spin);
        else if (a == "--mhz")
            take(g_mhz);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "compose_e2m1_sc_u14 --nt 2|4 [--m 1] [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "compose_e2m1_sc_u14: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (nt != 2 && nt != 4) {
        std::fprintf(stderr, "compose_e2m1_sc_u14: --nt must be 2 or 4\n");
        return 2;
    }
    const int unroll = (nt == 4) ? 8 : 14;
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "dtype=s4xE2M1_two_term->f16_scaled never_bitcast_s4 "
                "RC=%d NT=%d unroll=%d dpas_lo_hi=%d "
                "wg=%dx%d_alongN padM=RC pack=2 a_scale=b_scale=%.4f out=f16 "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kRc, nt, unroll,
                2 * nt * unroll, kWgX, kWgY, double(kScale), warmup, iters, g_card,
                g_spin);
    int rc = 0;
    run_shape(q, nt, unroll, "check", kRc, nt * kExecN, unroll * kK64, 1, 1, &rc,
              0);
    run_shape(q, nt, unroll, "timed", timed_m, timed_n, timed_k, warmup, iters,
              &rc, 1);
    return rc;
}

// K2 ESIMD s8 4-acc + wg 4x2x4 (32 thr, M on Y), k64 A-db, f16,
// M=64, no SLM.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// CONFIG prior: 4-acc wg 4x2 (8 thr) is 115 us vs 4x8 A-db 75 vs
// W8A8 46. sc84 4x2x4+SLM was 136. ngen M=64 is wg 4x2x4 k64
// grf256. Steal 32-thread 4x2x4 without SLM: X/Z along N, Y
// along M. Same 64 dpas, k64 A-db. Rank pipe_host vs 115 and 75.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
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
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

namespace {

constexpr int kRc = 8;
constexpr int kMTiles = 4;
constexpr int kRowsTh = kRc * kMTiles;
constexpr int kKc = 32;
constexpr int kK64 = 64;
constexpr int kExecN = 16;
constexpr int kWgX = 4;
constexpr int kWgY = 2;
constexpr int kWgZ = 4;
constexpr int kWgN = kWgX * kWgZ;
constexpr float kScale = 0.02f;

struct DpasS8Sc8m424Nt2Name {};
struct DpasS8Sc8m424Nt4Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s8_sc8m424: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_s8(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 129u) - 64);
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
                   const float *asd, const float *bsd, sycl::half *cd, int rows,
                   int cols, int dk) {
    constexpr int kTN = NT * kExecN;
    constexpr int kInnerK = kUnroll * kK64;
    const int64_t m_blocks = rows / kRowsTh;
    const int64_t n_groups = cols / kTN;
    const size_t n_wgs = round_up(n_groups, kWgN) / size_t(kWgN);
    const size_t m_wgs = round_up(m_blocks, kWgY) / size_t(kWgY);
    const size_t g0 = n_wgs * size_t(kWgX);
    const size_t g1 = m_wgs * size_t(kWgY);
    const size_t g2 = size_t(kWgZ);
    syclex::properties props{intelex::grf_size<256>};
    return q.parallel_for<Name>(
        sycl::nd_range<3>({g0, g1, g2},
                          {size_t(kWgX), size_t(kWgY), size_t(kWgZ)}),
        props, [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
            const int lid_n = int(it.get_local_id(2)) * kWgX +
                              int(it.get_local_id(0));
            const int64_t ng = int64_t(it.get_group(0)) * kWgN + int64_t(lid_n);
            const int64_t mb = int64_t(it.get_group(1)) * kWgY +
                               int64_t(it.get_local_id(1));
            if (ng >= n_groups || mb >= m_blocks)
                return;
            const int row0 = int(mb * kRowsTh);
            const int col0 = int(ng * kTN);
            esimd::simd<int32_t, kRc * kExecN> acc[kMTiles][NT];
#pragma unroll
            for (int r = 0; r < kMTiles; ++r)
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    acc[r][t] = 0;
            esimd::simd<int8_t, kRc * kKc> a0[kMTiles];
            esimd::simd<int8_t, kRc * kKc> a1[kMTiles];
#pragma unroll
            for (int r = 0; r < kMTiles; ++r) {
                a0[r] = xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                    ad, unsigned(dk - 1), unsigned(rows - 1), unsigned(dk - 1),
                    0, row0 + r * kRc);
                a1[r] = xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                    ad, unsigned(dk - 1), unsigned(rows - 1), unsigned(dk - 1),
                    kKc, row0 + r * kRc);
            }
            const int outer = dk / kInnerK;
            for (int o = 0; o < outer; ++o) {
#pragma unroll
                for (int u = 0; u < kUnroll; ++u) {
                    const int k0 = o * kInnerK + u * kK64;
                    const bool more_u = (u + 1 < kUnroll);
                    esimd::simd<int8_t, kRc * kKc> a0n[kMTiles];
                    esimd::simd<int8_t, kRc * kKc> a1n[kMTiles];
                    const int knext = more_u ? (k0 + kK64)
                                             : ((o + 1) * kInnerK);
                    const bool load_next = more_u || (o + 1 < outer);
                    if (load_next) {
#pragma unroll
                        for (int r = 0; r < kMTiles; ++r) {
                            a0n[r] = xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                                ad, unsigned(dk - 1), unsigned(rows - 1),
                                unsigned(dk - 1), knext, row0 + r * kRc);
                            a1n[r] = xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                                ad, unsigned(dk - 1), unsigned(rows - 1),
                                unsigned(dk - 1), knext + kKc,
                                row0 + r * kRc);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const esimd::simd<int8_t, kKc * kExecN> b0 =
                            xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false,
                                                true>(
                                bd, unsigned(cols - 1), unsigned(dk - 1),
                                unsigned(cols - 1), col0 + t * kExecN, k0);
#pragma unroll
                        for (int r = 0; r < kMTiles; ++r)
                            acc[r][t] = esimd::xmx::dpas<8, kRc, int32_t,
                                                         int32_t, int8_t,
                                                         int8_t>(acc[r][t], b0,
                                                                 a0[r]);
                        const esimd::simd<int8_t, kKc * kExecN> b1 =
                            xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false,
                                                true>(
                                bd, unsigned(cols - 1), unsigned(dk - 1),
                                unsigned(cols - 1), col0 + t * kExecN,
                                k0 + kKc);
#pragma unroll
                        for (int r = 0; r < kMTiles; ++r)
                            acc[r][t] = esimd::xmx::dpas<8, kRc, int32_t,
                                                         int32_t, int8_t,
                                                         int8_t>(acc[r][t], b1,
                                                                 a1[r]);
                    }
                    if (load_next) {
#pragma unroll
                        for (int r = 0; r < kMTiles; ++r) {
                            a0[r] = a0n[r];
                            a1[r] = a1n[r];
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kMTiles; ++r) {
                esimd::simd<float, kRc> asv;
                asv.copy_from(asd + row0 + r * kRc);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    esimd::simd<float, kExecN> bsv;
                    bsv.copy_from(bsd + col0 + t * kExecN);
                    esimd::simd<float, kRc * kExecN> f(acc[r][t]);
#pragma unroll
                    for (int rr = 0; rr < kRc; ++rr) {
                        const float ar = asv[rr];
#pragma unroll
                        for (int c = 0; c < kExecN; ++c)
                            f[rr * kExecN + c] *= ar * bsv[c];
                    }
                    esimd::simd<sycl::half, kRc * kExecN> h;
#pragma unroll
                    for (int i = 0; i < kRc * kExecN; ++i)
                        h[i] = sycl::half(float(f[i]));
                    xesimd::lsc_store_2d<sycl::half, kExecN, kRc>(
                        cd, unsigned(cols * int(sizeof(sycl::half)) - 1),
                        unsigned(rows - 1),
                        unsigned(cols * int(sizeof(sycl::half)) - 1),
                        col0 + t * kExecN, row0 + r * kRc, h);
                }
            }
        });
}

void run_shape(sycl::queue &q, int nt, int unroll, const char *phase, int m,
               int n, int k, int warmup, int iters, int *rc, int do_spin) {
    const int tn = nt * kExecN;
    const int inner_k = unroll * kK64;
    if (m < 1 || n % tn != 0 || k % inner_k != 0) {
        std::fprintf(stderr,
                     "dpas_s8_sc8m424: shape m=%d n=%d k=%d nt=%d unroll=%d\n", m, n,
                     k, nt, unroll);
        *rc = 2;
        return;
    }
    const int rows = ((m + kRowsTh - 1) / kRowsTh) * kRowsTh;
    const size_t na = size_t(rows) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc_pad = size_t(rows) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na, 0), hb(nb);
    std::vector<float> has(size_t(rows), 0.f), hbs(size_t(n), kScale);
    std::vector<sycl::half> href(nc), hgot(nc), hpad(nc_pad);
    fill_s8(ha.data(), size_t(m) * size_t(k), 1);
    fill_s8(hb.data(), nb, 9);
    for (int i = 0; i < m; ++i)
        has[size_t(i)] = kScale;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int kk = 0; kk < k; ++kk)
                acc += float(ha[size_t(i) * size_t(k) + size_t(kk)]) *
                       float(hb[size_t(kk) * size_t(n) + size_t(j)]);
            href[size_t(i) * size_t(n) + size_t(j)] =
                sycl::half(acc * has[size_t(i)] * hbs[size_t(j)]);
        }
    }

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd = sycl::malloc_device<int8_t>(nb, q);
    float *asd = sycl::malloc_device<float>(size_t(rows), q);
    float *bsd = sycl::malloc_device<float>(size_t(n), q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nc_pad, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(bd, hb.data(), nb).wait();
    q.memcpy(asd, has.data(), size_t(rows) * sizeof(float)).wait();
    q.memcpy(bsd, hbs.data(), size_t(n) * sizeof(float)).wait();

    auto go = [&]() -> sycl::event {
        if (nt == 4)
            return launch<DpasS8Sc8m424Nt4Name, 4, 2>(q, ad, bd, asd, bsd, cd,
                                                      rows, n, k);
        return launch<DpasS8Sc8m424Nt2Name, 2, 4>(q, ad, bd, asd, bsd, cd,
                                                  rows, n, k);
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
    sycl::free(bd, q);
    sycl::free(asd, q);
    sycl::free(bsd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int nt = 2;
    int timed_m = 64, timed_n = 5120, timed_k = 5120;
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
            std::fprintf(stderr,
                         "dpas_s8_sc8m424 --nt 2|4 [--m 64] [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s8_sc8m424: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (nt != 2 && nt != 4) {
        std::fprintf(stderr, "dpas_s8_sc8m424: --nt must be 2 or 4\n");
        return 2;
    }
    const int unroll = (nt == 4) ? 2 : 4;
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "dtype=s8xs8->f16_scaled RC=%d Mtiles=%d rows_th=%d NT=%d "
                "unroll=%d dpas=%d wg=%dx%dx%d_NxMxN A_double_buffer=1 no_slm=1 "
                "grf256_request=1 a_scale=b_scale=%.4f out=f16 "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kRc, kMTiles, kRowsTh, nt,
                unroll, 2 * nt * unroll * kMTiles, kWgX, kWgY, kWgZ,
                double(kScale),
                warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, nt, unroll, "check", kRowsTh * kWgY, nt * kExecN * kWgN,
              unroll * kK64, 1, 1, &rc, 0);
    run_shape(q, nt, unroll, "timed", timed_m, timed_n, timed_k, warmup, iters,
              &rc, 1);
    return rc;
}

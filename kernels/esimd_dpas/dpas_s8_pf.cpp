// K2 ESIMD s8 RC=4, 64 dpas.8x4 + lsc_prefetch_2d. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx -fsycl (intel/llvm#21741).
//
// CONFIG prior: ngen M=1 catalog has ff (null-dest UGM prefetch) around
// the 64x dpas.8x4. u64 has the count, no prefetch. Prefetch next k64 A
// and B (cached/cached) before the current step's loads+dpas.
// CSV: phase,nt,unroll,m,n,k,us,TOPS,max_abs,ok

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

namespace {

constexpr int kRc = 4;
constexpr int kKc = 32;
constexpr int kK64 = 64;
constexpr int kExecN = 16;
constexpr int kSg = 16;

struct DpasS8PfNt2Name {};
struct DpasS8PfNt4Name {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s8_pf: no GPU\n");
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

template <typename Name, int NT, int kUnroll>
sycl::event launch(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                   int32_t *cd, int rows, int cols, int dk) {
    constexpr int kTN = NT * kExecN;
    constexpr int kInnerK = kUnroll * kK64;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kTN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    return q.parallel_for<Name>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
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
                    const int kn = k0 + kK64;
                    if (kn < dk) {
                        xesimd::lsc_prefetch_2d<
                            int8_t, kKc, kRc, 1, xesimd::cache_hint::cached,
                            xesimd::cache_hint::cached>(
                            ad, unsigned(dk - 1), unsigned(rows - 1),
                            unsigned(dk - 1), kn, row0);
                        xesimd::lsc_prefetch_2d<
                            int8_t, kTN, kKc, 1, xesimd::cache_hint::cached,
                            xesimd::cache_hint::cached>(
                            bd, unsigned(cols - 1), unsigned(dk - 1),
                            unsigned(cols - 1), col0, kn);
                    }
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
               int n, int k, int warmup, int iters, int *rc) {
    const int tn = nt * kExecN;
    const int inner_k = unroll * kK64;
    if (m % kRc != 0 || n % tn != 0 || k % inner_k != 0) {
        std::fprintf(stderr,
                     "dpas_s8_pf: shape m=%d n=%d k=%d nt=%d unroll=%d\n", m, n,
                     k, nt, unroll);
        *rc = 2;
        return;
    }
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na), hb(nb);
    std::vector<int32_t> href(nc), hgot(nc);
    fill_s8(ha.data(), na, 1);
    fill_s8(hb.data(), nb, 9);
    host_s32(ha.data(), hb.data(), href.data(), m, n, k);

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd = sycl::malloc_device<int8_t>(nb, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(bd, hb.data(), nb).wait();

    auto go = [&]() -> sycl::event {
        if (nt == 4)
            return launch<DpasS8PfNt4Name, 4, 8>(q, ad, bd, cd, m, n, k);
        return launch<DpasS8PfNt2Name, 2, 16>(q, ad, bd, cd, m, n, k);
    };

    go().wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    const int max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    const int ok = (max_abs == 0) ? 1 : 0;
    if (!ok)
        *rc = 1;

    for (int i = 0; i < warmup; ++i)
        go().wait_and_throw();
    uint64_t ns_sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = go();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ns_sum += (t1 - t0);
    }
    const double us = (double(ns_sum) / 1000.0) / double(iters);
    const double ops = 2.0 * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.4f,%d,%d\n", phase, nt, unroll, m, n,
                k, us, tops, max_abs, ok);

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int nt = 2;
    int timed_m = 4, timed_n = 5120, timed_k = 5120;
    int warmup = 5, iters = 20;
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
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "dpas_s8_pf --nt 2|4 [--m 4] [--n 5120] [--k 5120]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s8_pf: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (nt != 2 && nt != 4) {
        std::fprintf(stderr, "dpas_s8_pf: --nt must be 2 or 4\n");
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
                "innerK=%d prefetch=lsc_prefetch_2d_cached Bload=Transformed "
                "warmup=%d iters=%d\n",
                backend, name.c_str(), driver.c_str(), kRc, kK64, kExecN, nt,
                nt * kExecN, unroll, 2 * nt * unroll, unroll * kK64, warmup,
                iters);
    std::printf("phase,nt,unroll,m,n,k,us,TOPS,max_abs,ok\n");
    int rc = 0;
    run_shape(q, nt, unroll, "check", kRc, nt * kExecN, unroll * kK64, warmup,
              iters, &rc);
    run_shape(q, nt, unroll, "timed", timed_m, timed_n, timed_k, warmup, iters,
              &rc);
    return rc;
}

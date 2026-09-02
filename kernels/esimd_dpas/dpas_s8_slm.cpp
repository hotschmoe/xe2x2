// K2 ESIMD s8 RC=4, WG=16 (8x2), A in SLM. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx -fsycl (intel/llvm#21741).
//
// CONFIG prior: W8A8 M=1 is dpas.8x4 wg 8x2 + SLM. Share one A 4x32
// tile across 16 threads (N=256 per WG) instead of reloading A per N.
// B still lsc_load_2d Transformed. CSV: phase,m,n,k,us,TOPS,max_abs,ok

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
constexpr int kExecN = 16;
constexpr int kSg = 16;
constexpr int kWgN = kSg * kExecN; // 256
constexpr int kSlmA = kRc * kKc;   // 128 bytes

struct DpasS8SlmName {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s8_slm: no GPU\n");
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

sycl::event launch(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                   int32_t *cd, int rows, int cols, int dk) {
    const int64_t m_blocks = rows / kRc;
    const int64_t n_wgs = cols / kWgN;
    const size_t groups = size_t(m_blocks * n_wgs);
    return q.parallel_for<DpasS8SlmName>(
        sycl::nd_range<1>({groups * size_t(kSg)}, {size_t(kSg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            esimd::slm_init<256>();
            const int64_t gid = int64_t(it.get_group(0));
            const int lid = int(it.get_local_id(0));
            const int64_t mb = gid % m_blocks;
            const int64_t nw = gid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(nw * kWgN) + lid * kExecN;
            esimd::simd<int32_t, kRc * kExecN> acc(0);
            const int chunks = dk / kKc;
            for (int kc = 0; kc < chunks; ++kc) {
                if (lid == 0) {
                    const esimd::simd<int8_t, kSlmA> afrag =
                        xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                            ad, unsigned(dk - 1), unsigned(rows - 1),
                            unsigned(dk - 1), kc * kKc, row0);
                    esimd::slm_block_store<int8_t, kSlmA>(0, afrag);
                }
                esimd::barrier();
                const esimd::simd<int8_t, kSlmA> afrag =
                    esimd::slm_block_load<int8_t, kSlmA>(0);
                const esimd::simd<int8_t, kKc * kExecN> bt =
                    xesimd::lsc_load_2d<int8_t, kExecN, kKc, 1, false, true>(
                        bd, unsigned(cols - 1), unsigned(dk - 1),
                        unsigned(cols - 1), col0, kc * kKc);
                acc = esimd::xmx::dpas<8, kRc, int32_t, int32_t, int8_t,
                                       int8_t>(acc, bt, afrag);
                esimd::barrier();
            }
            xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                cd, unsigned(cols * int(sizeof(int32_t)) - 1),
                unsigned(rows - 1),
                unsigned(cols * int(sizeof(int32_t)) - 1), col0, row0, acc);
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

void run_shape(sycl::queue &q, const char *phase, int m, int n, int k,
               int warmup, int iters, int *rc) {
    if (m % kRc != 0 || n % kWgN != 0 || k % kKc != 0) {
        std::fprintf(stderr, "dpas_s8_slm: shape m=%d n=%d k=%d\n", m, n, k);
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

    launch(q, ad, bd, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    const int max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    const int ok = (max_abs == 0) ? 1 : 0;
    if (!ok)
        *rc = 1;

    for (int i = 0; i < warmup; ++i)
        launch(q, ad, bd, cd, m, n, k).wait_and_throw();
    uint64_t ns_sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = launch(q, ad, bd, cd, m, n, k);
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
    std::printf("%s,%d,%d,%d,%.3f,%.4f,%d,%d\n", phase, m, n, k, us, tops,
                max_abs, ok);

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int timed_m = 4, timed_n = 5120, timed_k = 5120;
    int warmup = 5, iters = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--m")
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
            std::fprintf(stderr, "dpas_s8_slm [--m 4] [--n 5120] [--k 5120]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s8_slm: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s dtype=s8xs8s32 "
                "RC=%d K=%d execN=%d wgN=%d slmA=%d Bload=lsc_load_2d_Transformed "
                "warmup=%d iters=%d\n",
                backend, name.c_str(), driver.c_str(), kRc, kKc, kExecN, kWgN,
                kSlmA, warmup, iters);
    std::printf("phase,m,n,k,us,TOPS,max_abs,ok\n");
    int rc = 0;
    run_shape(q, "check", kRc, kWgN, kKc, warmup, iters, &rc);
    run_shape(q, "timed", timed_m, timed_n, timed_k, warmup, iters, &rc);
    return rc;
}

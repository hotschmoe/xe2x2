// K2 ESIMD DPAS s2 x s2 -> s32. Backend: sycl+l0. AOT intel_gpu_bmg_g31.
// Standalone icpx -fsycl binary (intel/llvm#21741: fat SYCL trees lie on B70).
//
// Optional arm. Tile: RC=8, K=64 per dpas (OPC=8, both <8-bit), exec N=16.
// This is NOT the 2508.06753 mix (that is s2xs8, K=32). s2xs2 would be
// the 2x-K-per-instruction arm if IGC accepts it.
// IGC s2 range is [-2, 1], not E2M1. Never bitcast NVFP4 onto s2.
//
// B feed: packed 4 s2/byte along K, lsc_load_2d Transformed=true.
// If this TU does not compile, keep the source and the refusal log.
// CSV: phase,m,n,k,us,TOPS,max_abs,ok

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
namespace xmx = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int kRc = 8;
constexpr int kKc = 64;
constexpr int kExecN = 16;
constexpr int kSg = 16;
constexpr int kPack = 4; // s2 per uint8 along K

struct DpasS2Name {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s2: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

uint8_t pack_s2(int8_t v0, int8_t v1, int8_t v2, int8_t v3) {
    return uint8_t((uint8_t(v0) & 3) | ((uint8_t(v1) & 3) << 2) |
                   ((uint8_t(v2) & 3) << 4) | ((uint8_t(v3) & 3) << 6));
}

void fill_s2(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 4u) - 2);
}

void pack_a(const int8_t *a, uint8_t *out, int m, int k) {
    const int kp = k / kPack;
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; kk += kPack)
            out[i * kp + kk / kPack] =
                pack_s2(a[i * k + kk], a[i * k + kk + 1], a[i * k + kk + 2],
                        a[i * k + kk + 3]);
}

void pack_b(const int8_t *b, uint8_t *out, int k, int n) {
    for (int kk = 0; kk < k; kk += kPack)
        for (int j = 0; j < n; ++j)
            out[(kk / kPack) * n + j] =
                pack_s2(b[kk * n + j], b[(kk + 1) * n + j],
                        b[(kk + 2) * n + j], b[(kk + 3) * n + j]);
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

sycl::event launch(sycl::queue &q, const uint8_t *ad, const uint8_t *bd,
                   int32_t *cd, int rows, int cols, int dk) {
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    constexpr int kAPacked = kKc / kPack;  // 16 uint8
    constexpr int kBPackedH = kKc / kPack; // 16 rows
    return q.parallel_for<DpasS2Name>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kExecN);
            esimd::simd<int32_t, kRc * kExecN> acc(0);
            const int chunks = dk / kKc;
            for (int kc = 0; kc < chunks; ++kc) {
                const esimd::simd<uint8_t, kRc * kAPacked> afrag =
                    xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                        ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                        unsigned(a_pitch - 1), kc * kAPacked, row0);
                const esimd::simd<uint8_t, kBPackedH * kExecN> bt =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1, false,
                                        true>(
                        bd, unsigned(cols - 1), unsigned(b_rows - 1),
                        unsigned(cols - 1), col0, kc * kBPackedH);
                acc = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                                xmx::dpas_argument_type::s2,
                                xmx::dpas_argument_type::s2>(acc, bt, afrag);
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
    if (m % kRc != 0 || n % kExecN != 0 || k % kKc != 0) {
        std::fprintf(stderr, "dpas_s2: shape m=%d n=%d k=%d not aligned\n", m, n,
                     k);
        *rc = 2;
        return;
    }
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    const size_t na_p = size_t(m) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    std::vector<int8_t> ha(na), hb(nb);
    std::vector<uint8_t> pa(na_p), pb(nb_p);
    std::vector<int32_t> href(nc), hgot(nc);
    fill_s2(ha.data(), na, 1);
    fill_s2(hb.data(), nb, 9);
    pack_a(ha.data(), pa.data(), m, k);
    pack_b(hb.data(), pb.data(), k, n);
    host_s32(ha.data(), hb.data(), href.data(), m, n, k);

    uint8_t *ad = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *bd = sycl::malloc_device<uint8_t>(nb_p, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc, q);
    q.memcpy(ad, pa.data(), na_p).wait();
    q.memcpy(bd, pb.data(), nb_p).wait();

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
    int check_m = kRc, check_n = kExecN, check_k = kKc;
    int timed_m = 1024, timed_n = 512, timed_k = 1024;
    int warmup = 3, iters = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--check-m")
            take(check_m);
        else if (a == "--check-n")
            take(check_n);
        else if (a == "--check-k")
            take(check_k);
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
                         "dpas_s2 [--check-m 8] [--check-n 16] [--check-k 64] "
                         "[--m 1024] [--n 512] [--k 1024] [--iters N] [--warmup W]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s2: unknown arg %s\n", a.c_str());
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
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s dtype=s2xs2s32 "
                "RC=%d K=%d N=%d OPC=8 s2_range=[-2,1] "
                "Bload=lsc_load_2d_Transformed warmup=%d iters=%d\n",
                backend, name.c_str(), driver.c_str(), kRc, kKc, kExecN, warmup,
                iters);
    std::printf("phase,m,n,k,us,TOPS,max_abs,ok\n");

    int rc = 0;
    run_shape(q, "check", check_m, check_n, check_k, warmup, iters, &rc);
    run_shape(q, "timed", timed_m, timed_n, timed_k, warmup, iters, &rc);
    return rc;
}

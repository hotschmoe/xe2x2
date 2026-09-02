// K3 E2M1 two-term hail-mary: w = w_lo + 8*w_hi vs native s8 LUT.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31.
// Standalone icpx -fsycl (intel/llvm#21741: fat SYCL trees lie on B70).
//
// HAIL-MARY (label in CONFIG). Never bitcast E2M1 nibbles onto s4 DPAS.
// E2M1 mag {0, 0.5, 1, 1.5, 2, 3, 4, 6}; q = 2*w is
// {0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12}. s4 is [-8, 7];
// 8 and 12 overflow. Decode nibble -> q, then split integer planes.
//
// Split (both planes fit s4):
//   |q|<=6:  w_lo=q,   w_hi=0
//   q=8:     w_lo=0,   w_hi=1
//   q=12:    w_lo=4,   w_hi=1
//   q=-8:    w_lo=0,   w_hi=-1
//   q=-12:   w_lo=-4,  w_hi=-1
// A is s4 activations. Two s4xs4 DPAS, acc = acc_lo + 8*acc_hi.
// Native arm: A widened to s8, q stored as s8, one s8xs8 DPAS.
//
// CONFIG prior (NOT a RESULT): two s4 terms ~2/1.49 ~= 1.34x native s8
// DPAS-only, plus extra B plane and epilogue.
//
// Tile: RC=8, exec N=16. s8 K/dpas=32, s4 K/dpas=64.
// B feed: lsc_load_2d Transformed=true. Host s32 oracle on integer q.
// CSV: arm,m,n,k,terms,us,TOPS,max_abs,ok

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <climits>
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
constexpr int kKcS8 = 32;
constexpr int kKcS4 = 64;
constexpr int kExecN = 16;
constexpr int kSg = 16;
constexpr int kPack = 2;

// E2M1 magnitude codes after *2. Index is bits [2:0] of the nibble.
constexpr int8_t kMag2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

struct NativeS8E2M1Name {};
struct E2M1TwoTermName {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "compose_e2m1_two_term: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

int8_t e2m1_nibble_to_q(uint8_t nib) {
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

uint8_t pack_s4(int8_t lo, int8_t hi) {
    return uint8_t((uint8_t(lo) & 0xf) | ((uint8_t(hi) & 0xf) << 4));
}

void fill_s4(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 16u) - 8);
    if (n >= 2) {
        p[0] = int8_t(-8);
        p[1] = int8_t(7);
    }
}

void fill_e2m1_nibbles(uint8_t *nib, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        nib[i] = uint8_t((i * 13u + seed) & 15u);
}

void pack_a_s4(const int8_t *a, uint8_t *out, int m, int k) {
    const int kp = k / kPack;
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; kk += kPack)
            out[i * kp + kk / kPack] =
                pack_s4(a[i * k + kk], a[i * k + kk + 1]);
}

void pack_b_s4(const int8_t *b, uint8_t *out, int k, int n) {
    for (int kk = 0; kk < k; kk += kPack)
        for (int j = 0; j < n; ++j)
            out[(kk / kPack) * n + j] =
                pack_s4(b[kk * n + j], b[(kk + 1) * n + j]);
}

int host_s32(const int8_t *a, const int8_t *b, int32_t *c, int m, int n,
             int k) {
    int overflow = 0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            int64_t acc = 0;
            for (int kk = 0; kk < k; ++kk)
                acc += int64_t(a[i * k + kk]) * int64_t(b[kk * n + j]);
            if (acc < int64_t(INT32_MIN) || acc > int64_t(INT32_MAX))
                overflow = 1;
            c[i * n + j] = int32_t(acc);
        }
    return overflow;
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

sycl::event launch_s8(sycl::queue &q, const int8_t *ad, const int8_t *bd,
                      int32_t *cd, int rows, int cols, int dk) {
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    return q.parallel_for<NativeS8E2M1Name>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kExecN);
            esimd::simd<int32_t, kRc * kExecN> acc(0);
            const int chunks = dk / kKcS8;
            for (int kc = 0; kc < chunks; ++kc) {
                const esimd::simd<int8_t, kRc * kKcS8> afrag =
                    xesimd::lsc_load_2d<int8_t, kKcS8, kRc>(
                        ad, unsigned(dk - 1), unsigned(rows - 1),
                        unsigned(dk - 1), kc * kKcS8, row0);
                const esimd::simd<int8_t, kKcS8 * kExecN> bt =
                    xesimd::lsc_load_2d<int8_t, kExecN, kKcS8, 1, false, true>(
                        bd, unsigned(cols - 1), unsigned(dk - 1),
                        unsigned(cols - 1), col0, kc * kKcS8);
                acc = esimd::xmx::dpas<8, kRc, int32_t, int32_t, int8_t,
                                       int8_t>(acc, bt, afrag);
            }
            xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                cd, unsigned(cols * int(sizeof(int32_t)) - 1),
                unsigned(rows - 1),
                unsigned(cols * int(sizeof(int32_t)) - 1), col0, row0, acc);
        });
}

sycl::event launch_two_term(sycl::queue &q, const uint8_t *ad,
                            const uint8_t *blo, const uint8_t *bhi,
                            int32_t *cd, int rows, int cols, int dk) {
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    constexpr int kAPacked = kKcS4 / kPack;
    constexpr int kBPackedH = kKcS4 / kPack;
    return q.parallel_for<E2M1TwoTermName>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kExecN);
            esimd::simd<int32_t, kRc * kExecN> acc_lo(0);
            esimd::simd<int32_t, kRc * kExecN> acc_hi(0);
            const int chunks = dk / kKcS4;
            for (int kc = 0; kc < chunks; ++kc) {
                const esimd::simd<uint8_t, kRc * kAPacked> afrag =
                    xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                        ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                        unsigned(a_pitch - 1), kc * kAPacked, row0);
                const esimd::simd<uint8_t, kBPackedH * kExecN> bt_lo =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1, false,
                                        true>(
                        blo, unsigned(cols - 1), unsigned(b_rows - 1),
                        unsigned(cols - 1), col0, kc * kBPackedH);
                const esimd::simd<uint8_t, kBPackedH * kExecN> bt_hi =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1, false,
                                        true>(
                        bhi, unsigned(cols - 1), unsigned(b_rows - 1),
                        unsigned(cols - 1), col0, kc * kBPackedH);
                acc_lo = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                                   xmx::dpas_argument_type::s4,
                                   xmx::dpas_argument_type::s4>(acc_lo, bt_lo,
                                                                afrag);
                acc_hi = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                                   xmx::dpas_argument_type::s4,
                                   xmx::dpas_argument_type::s4>(acc_hi, bt_hi,
                                                                afrag);
            }
            const esimd::simd<int32_t, kRc * kExecN> acc = acc_lo + acc_hi * 8;
            xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                cd, unsigned(cols * int(sizeof(int32_t)) - 1),
                unsigned(rows - 1),
                unsigned(cols * int(sizeof(int32_t)) - 1), col0, row0, acc);
        });
}

void emit_row(const char *arm, int m, int n, int k, int terms, double us,
              int max_abs, int ok) {
    const double ops = 2.0 * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    std::printf("%s,%d,%d,%d,%d,%.3f,%.4f,%d,%d\n", arm, m, n, k, terms, us,
                tops, max_abs, ok);
}

template <typename Launch>
double time_launch(int warmup, int iters, Launch launch) {
    for (int i = 0; i < warmup; ++i)
        launch().wait_and_throw();
    uint64_t ns_sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = launch();
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ns_sum += (t1 - t0);
    }
    return (double(ns_sum) / 1000.0) / double(iters);
}

void run_shape(sycl::queue &q, int m, int n, int k, int warmup, int iters,
               int *rc) {
    if (m % kRc != 0 || n % kExecN != 0 || k % kKcS4 != 0) {
        std::fprintf(stderr,
                     "compose_e2m1_two_term: shape m=%d n=%d k=%d not aligned\n",
                     m, n, k);
        *rc = 2;
        return;
    }
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    const size_t na_p = size_t(m) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    std::vector<int8_t> ha(na), hq(nb), wlo(nb), whi(nb);
    std::vector<uint8_t> nib(nb), pa(na_p), pb_lo(nb_p), pb_hi(nb_p);
    std::vector<int32_t> href(nc), hgot(nc);
    fill_s4(ha.data(), na, 1);
    fill_e2m1_nibbles(nib.data(), nb, 9);
    for (size_t i = 0; i < nb; ++i) {
        hq[i] = e2m1_nibble_to_q(nib[i]);
        split_q(hq[i], &wlo[i], &whi[i]);
    }
    pack_a_s4(ha.data(), pa.data(), m, k);
    pack_b_s4(wlo.data(), pb_lo.data(), k, n);
    pack_b_s4(whi.data(), pb_hi.data(), k, n);
    const int ovf = host_s32(ha.data(), hq.data(), href.data(), m, n, k);
    if (ovf) {
        std::fprintf(stderr, "compose_e2m1_two_term: oracle overflow s32\n");
        *rc = 1;
    }

    int8_t *ad8 = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd8 = sycl::malloc_device<int8_t>(nb, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc, q);
    q.memcpy(ad8, ha.data(), na).wait();
    q.memcpy(bd8, hq.data(), nb).wait();
    launch_s8(q, ad8, bd8, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    int max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    int ok = (max_abs == 0 && !ovf) ? 1 : 0;
    if (!ok)
        *rc = 1;
    const double us8 = time_launch(warmup, iters, [&]() {
        return launch_s8(q, ad8, bd8, cd, m, n, k);
    });
    emit_row("native_s8_e2m1_lut", m, n, k, 1, us8, max_abs, ok);
    sycl::free(ad8, q);
    sycl::free(bd8, q);

    uint8_t *ad4 = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *blo = sycl::malloc_device<uint8_t>(nb_p, q);
    uint8_t *bhi = sycl::malloc_device<uint8_t>(nb_p, q);
    q.memcpy(ad4, pa.data(), na_p).wait();
    q.memcpy(blo, pb_lo.data(), nb_p).wait();
    q.memcpy(bhi, pb_hi.data(), nb_p).wait();
    launch_two_term(q, ad4, blo, bhi, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    ok = (max_abs == 0 && !ovf) ? 1 : 0;
    if (!ok)
        *rc = 1;
    const double us2 = time_launch(warmup, iters, [&]() {
        return launch_two_term(q, ad4, blo, bhi, cd, m, n, k);
    });
    emit_row("e2m1_two_term", m, n, k, 2, us2, max_abs, ok);

    sycl::free(ad4, q);
    sycl::free(blo, q);
    sycl::free(bhi, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int check_m = kRc, check_n = kExecN, check_k = kKcS4;
    int timed_m = 1024, timed_n = 1024, timed_k = 1024;
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
                         "compose_e2m1_two_term [--check-m 8] [--check-n 16] "
                         "[--check-k 64] [--m 1024] [--n 1024] [--k 1024] "
                         "[--iters N] [--warmup W]\n");
            return 0;
        } else {
            std::fprintf(stderr, "compose_e2m1_two_term: unknown arg %s\n",
                         a.c_str());
            return 2;
        }
    }

    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver =
        dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf(
        "# CONFIG backend=%s device=\"%s\" driver=%s "
        "target=emulate_E2M1 hail_mary=1 "
        "decode=nibble_to_q_then_split never_bitcast_e2m1_onto_s4 "
        "RC=%d N=%d Ks8=%d Ks4=%d Bload=lsc_load_2d_Transformed "
        "prior_two_term_vs_s8=1.34x_from_k2_1.49x NOT_A_RESULT "
        "warmup=%d iters=%d\n",
        backend, name.c_str(), driver.c_str(), kRc, kExecN, kKcS8, kKcS4,
        warmup, iters);
    std::printf("arm,m,n,k,terms,us,TOPS,max_abs,ok\n");

    int rc = 0;
    run_shape(q, check_m, check_n, check_k, warmup, iters, &rc);
    run_shape(q, timed_m, timed_n, timed_k, warmup, iters, &rc);
    return rc;
}

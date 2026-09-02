// K3 schoolbook s8-from-s4 vs native s8 DPAS in one binary.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31.
// Standalone icpx -fsycl (intel/llvm#21741: fat SYCL trees lie on B70).
//
// Tile: RC=8, exec N=16. Native s8 K/dpas=32. 4-bit K/dpas=64.
// B feed: lsc_load_2d Transformed=true. No flat host-prepacked VNNI path.
//
// Split (covers full s8 [-128,127]):
//   a = 16 * a1 + a0,  a0 = u4 [0,15],  a1 = s4 [-8,7]
//   a*b = a0*b0 + 16*(a0*b1 + a1*b0) + 256*a1*b1
// Four 4-bit DPAS (u4xu4, s4xu4, u4xs4, s4xs4) plus *16/*256 adds.
// A pure s4xs4 nibble split does not cover full s8 (low digit is unsigned).
//
// CONFIG prior (NOT a RESULT): K2 at 1024^3 matched ~583 MHz, native s8
// 374 us, native s4 250 us (1.49x s8, not 2x). Four schoolbook 4-bit terms
// cost ~4/1.49 ~= 2.7x native s8, ignoring extra loads and epilogue.
//
// Host s32 oracle (int64 accum, must fit s32). Logical ops = 2*m*n*k s8 MACs.
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

struct NativeS8Name {};
struct SchoolbookS4Name {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "compose_s8_from_s4: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

int s8_lo_u4(int8_t a) { return int(uint8_t(a) & 0x0f); }

int s8_hi_s4(int8_t a) {
    const int h = int(uint8_t(a) >> 4);
    return h >= 8 ? h - 16 : h;
}

uint8_t pack_nib(int lo, int hi) {
    return uint8_t((lo & 0xf) | ((hi & 0xf) << 4));
}

void fill_s8(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 256u) - 128);
    if (n >= 4) {
        p[0] = int8_t(-128);
        p[1] = int8_t(127);
        p[2] = int8_t(-1);
        p[3] = int8_t(8);
    }
}

void pack_a_lo(const int8_t *a, uint8_t *out, int m, int k) {
    const int kp = k / kPack;
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; kk += kPack)
            out[i * kp + kk / kPack] =
                pack_nib(s8_lo_u4(a[i * k + kk]), s8_lo_u4(a[i * k + kk + 1]));
}

void pack_a_hi(const int8_t *a, uint8_t *out, int m, int k) {
    const int kp = k / kPack;
    for (int i = 0; i < m; ++i)
        for (int kk = 0; kk < k; kk += kPack)
            out[i * kp + kk / kPack] =
                pack_nib(s8_hi_s4(a[i * k + kk]), s8_hi_s4(a[i * k + kk + 1]));
}

void pack_b_lo(const int8_t *b, uint8_t *out, int k, int n) {
    for (int kk = 0; kk < k; kk += kPack)
        for (int j = 0; j < n; ++j)
            out[(kk / kPack) * n + j] =
                pack_nib(s8_lo_u4(b[kk * n + j]), s8_lo_u4(b[(kk + 1) * n + j]));
}

void pack_b_hi(const int8_t *b, uint8_t *out, int k, int n) {
    for (int kk = 0; kk < k; kk += kPack)
        for (int j = 0; j < n; ++j)
            out[(kk / kPack) * n + j] =
                pack_nib(s8_hi_s4(b[kk * n + j]), s8_hi_s4(b[(kk + 1) * n + j]));
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
    return q.parallel_for<NativeS8Name>(
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

sycl::event launch_schoolbook(sycl::queue &q, const uint8_t *a0d,
                              const uint8_t *a1d, const uint8_t *b0d,
                              const uint8_t *b1d, int32_t *cd, int rows,
                              int cols, int dk) {
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    constexpr int kAPacked = kKcS4 / kPack;
    constexpr int kBPackedH = kKcS4 / kPack;
    return q.parallel_for<SchoolbookS4Name>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kExecN);
            esimd::simd<int32_t, kRc * kExecN> acc(0);
            esimd::simd<int32_t, kRc * kExecN> z(0);
            const int chunks = dk / kKcS4;
            for (int kc = 0; kc < chunks; ++kc) {
                const esimd::simd<uint8_t, kRc * kAPacked> a0 =
                    xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                        a0d, unsigned(a_pitch - 1), unsigned(rows - 1),
                        unsigned(a_pitch - 1), kc * kAPacked, row0);
                const esimd::simd<uint8_t, kRc * kAPacked> a1 =
                    xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                        a1d, unsigned(a_pitch - 1), unsigned(rows - 1),
                        unsigned(a_pitch - 1), kc * kAPacked, row0);
                const esimd::simd<uint8_t, kBPackedH * kExecN> b0 =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1, false,
                                        true>(
                        b0d, unsigned(cols - 1), unsigned(b_rows - 1),
                        unsigned(cols - 1), col0, kc * kBPackedH);
                const esimd::simd<uint8_t, kBPackedH * kExecN> b1 =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1, false,
                                        true>(
                        b1d, unsigned(cols - 1), unsigned(b_rows - 1),
                        unsigned(cols - 1), col0, kc * kBPackedH);
                auto t00 =
                    xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                              xmx::dpas_argument_type::u4,
                              xmx::dpas_argument_type::u4>(z, b0, a0);
                auto t01 =
                    xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                              xmx::dpas_argument_type::s4,
                              xmx::dpas_argument_type::u4>(z, b1, a0);
                auto t10 =
                    xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                              xmx::dpas_argument_type::u4,
                              xmx::dpas_argument_type::s4>(z, b0, a1);
                auto t11 =
                    xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                              xmx::dpas_argument_type::s4,
                              xmx::dpas_argument_type::s4>(z, b1, a1);
                acc = acc + t00 + (t01 + t10) * 16 + t11 * 256;
            }
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
                     "compose_s8_from_s4: shape m=%d n=%d k=%d not aligned\n",
                     m, n, k);
        *rc = 2;
        return;
    }
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    const size_t na_p = size_t(m) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    std::vector<int8_t> ha(na), hb(nb);
    std::vector<uint8_t> a0(na_p), a1(na_p), b0(nb_p), b1(nb_p);
    std::vector<int32_t> href(nc), hgot(nc);
    fill_s8(ha.data(), na, 1);
    fill_s8(hb.data(), nb, 9);
    pack_a_lo(ha.data(), a0.data(), m, k);
    pack_a_hi(ha.data(), a1.data(), m, k);
    pack_b_lo(hb.data(), b0.data(), k, n);
    pack_b_hi(hb.data(), b1.data(), k, n);
    const int ovf = host_s32(ha.data(), hb.data(), href.data(), m, n, k);
    if (ovf) {
        std::fprintf(stderr, "compose_s8_from_s4: oracle overflow s32\n");
        *rc = 1;
    }

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    int8_t *bd = sycl::malloc_device<int8_t>(nb, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(bd, hb.data(), nb).wait();
    launch_s8(q, ad, bd, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    int max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    int ok = (max_abs == 0 && !ovf) ? 1 : 0;
    if (!ok)
        *rc = 1;
    const double us8 = time_launch(warmup, iters, [&]() {
        return launch_s8(q, ad, bd, cd, m, n, k);
    });
    emit_row("native_s8", m, n, k, 1, us8, max_abs, ok);
    sycl::free(ad, q);
    sycl::free(bd, q);

    uint8_t *a0d = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *a1d = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *b0d = sycl::malloc_device<uint8_t>(nb_p, q);
    uint8_t *b1d = sycl::malloc_device<uint8_t>(nb_p, q);
    q.memcpy(a0d, a0.data(), na_p).wait();
    q.memcpy(a1d, a1.data(), na_p).wait();
    q.memcpy(b0d, b0.data(), nb_p).wait();
    q.memcpy(b1d, b1.data(), nb_p).wait();
    launch_schoolbook(q, a0d, a1d, b0d, b1d, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    max_abs = max_abs_diff(hgot.data(), href.data(), nc);
    ok = (max_abs == 0 && !ovf) ? 1 : 0;
    if (!ok)
        *rc = 1;
    const double us4 = time_launch(warmup, iters, [&]() {
        return launch_schoolbook(q, a0d, a1d, b0d, b1d, cd, m, n, k);
    });
    emit_row("schoolbook_s4", m, n, k, 4, us4, max_abs, ok);

    sycl::free(a0d, q);
    sycl::free(a1d, q);
    sycl::free(b0d, q);
    sycl::free(b1d, q);
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
            std::fprintf(
                stderr,
                "compose_s8_from_s4 [--check-m 8] [--check-n 16] [--check-k 64] "
                "[--m 1024] [--n 1024] [--k 1024] [--iters N] [--warmup W]\n");
            return 0;
        } else {
            std::fprintf(stderr, "compose_s8_from_s4: unknown arg %s\n",
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
        "target=emulate_s8 digits=u4_lo+s4_hi "
        "RC=%d N=%d Ks8=%d Ks4=%d Bload=lsc_load_2d_Transformed "
        "prior_schoolbook_vs_s8=2.7x_from_k2_1.49x NOT_A_RESULT "
        "warmup=%d iters=%d\n",
        backend, name.c_str(), driver.c_str(), kRc, kExecN, kKcS8, kKcS4,
        warmup, iters);
    std::printf("arm,m,n,k,terms,us,TOPS,max_abs,ok\n");

    int rc = 0;
    run_shape(q, check_m, check_n, check_k, warmup, iters, &rc);
    run_shape(q, timed_m, timed_n, timed_k, warmup, iters, &rc);
    return rc;
}

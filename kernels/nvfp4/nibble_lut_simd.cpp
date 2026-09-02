// K6 vectorized in-register E2M1 nibble LUT -> VNNI4 -> s8 DPAS.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx -fsycl.
// Never bitcast E2M1 onto s4.
//
// CONFIG prior: scalar-unroll in-register LUT was max_abs=0 at 2316 us;
// two-launch unpack+DPAS was 84-305 us. This arm uses simd LUT + simd
// VNNI4 select, same numeric contract.

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

constexpr int kRc = 8;
constexpr int kKc = 32;
constexpr int kExecN = 16;
constexpr int kSg = 16;
constexpr int kPackedN = (kKc / 2) * kExecN;
constexpr int8_t kMag2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

struct FusedSimdName {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "nibble_lut_simd: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

inline int8_t nibble_to_q(uint8_t nib) {
    int8_t q = kMag2[nib & 7];
    if (nib & 8)
        q = int8_t(-q);
    return q;
}

void fill_s8(int8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = int8_t(int((i * 17u + seed) % 255u) - 128);
}

void fill_nibbles(uint8_t *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = uint8_t((i * 13u + seed) & 15u);
}

void pack_b(const uint8_t *nib, uint8_t *packed, int k, int n) {
    for (int kk = 0; kk < k; kk += 2)
        for (int j = 0; j < n; ++j)
            packed[(kk / 2) * n + j] =
                uint8_t((nib[kk * n + j] & 15) |
                        ((nib[(kk + 1) * n + j] & 15) << 4));
}

void unpack_host(const uint8_t *packed, int8_t *s8, int k, int n) {
    for (int kk = 0; kk < k; kk += 2)
        for (int j = 0; j < n; ++j) {
            const uint8_t p = packed[(kk / 2) * n + j];
            s8[kk * n + j] = nibble_to_q(uint8_t(p & 15));
            s8[(kk + 1) * n + j] = nibble_to_q(uint8_t(p >> 4));
        }
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

int max_abs_diff(const int32_t *got, const int32_t *ref, size_t n) {
    int mx = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = got[i] > ref[i] ? got[i] - ref[i] : ref[i] - got[i];
        if (d > mx)
            mx = d;
    }
    return mx;
}

inline esimd::simd<int8_t, kPackedN>
decode_nibbles(const esimd::simd<uint8_t, kPackedN> &nib) {
    const esimd::simd<int16_t, kPackedN> idx =
        esimd::convert<int16_t>(nib & uint8_t(7));
    esimd::simd<int16_t, kPackedN> mag = idx;
    mag.merge(mag + int16_t(1), idx > int16_t(4));
    mag.merge(mag + int16_t(1), idx > int16_t(5));
    mag.merge(mag + int16_t(3), idx > int16_t(6));
    esimd::simd<int16_t, kPackedN> q = mag;
    q.merge(-mag, (nib & uint8_t(8)) != uint8_t(0));
    return esimd::convert<int8_t>(q);
}

inline esimd::simd<int8_t, kKc * kExecN>
lut_packed_to_s8(const esimd::simd<uint8_t, kPackedN> &p) {
    esimd::simd<int8_t, kPackedN> qlo = decode_nibbles(p & uint8_t(15));
    esimd::simd<int8_t, kPackedN> qhi = decode_nibbles(p >> uint8_t(4));
    esimd::simd<int8_t, kKc * kExecN> brm(0);
#pragma unroll
    for (int r = 0; r < kKc / 2; ++r) {
        brm.template select<kExecN, 1>((2 * r) * kExecN) =
            qlo.template select<kExecN, 1>(r * kExecN);
        brm.template select<kExecN, 1>((2 * r + 1) * kExecN) =
            qhi.template select<kExecN, 1>(r * kExecN);
    }
    return brm;
}

inline esimd::simd<int8_t, kKc * kExecN>
pack_vnni4(esimd::simd<int8_t, kKc * kExecN> brm) {
    esimd::simd<int8_t, kKc * kExecN> bv(0);
#pragma unroll
    for (int g = 0; g < kKc / 4; ++g) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            bv.template select<kExecN, 4>((g * kExecN) * 4 + r) =
                brm.template select<kExecN, 1>((g * 4 + r) * kExecN);
        }
    }
    return bv;
}

sycl::event launch_fused(sycl::queue &q, const int8_t *ad, const uint8_t *pd,
                         int32_t *cd, int rows, int cols, int dk) {
    const int packed_rows = dk / 2;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    return q.parallel_for<FusedSimdName>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ng = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kExecN);
            esimd::simd<int32_t, kRc * kExecN> acc(0);
            const int chunks = dk / kKc;
            constexpr int kPackedH = kKc / 2;
            for (int kc = 0; kc < chunks; ++kc) {
                const esimd::simd<int8_t, kRc * kKc> afrag =
                    xesimd::lsc_load_2d<int8_t, kKc, kRc>(
                        ad, unsigned(dk - 1), unsigned(rows - 1),
                        unsigned(dk - 1), kc * kKc, row0);
                const esimd::simd<uint8_t, kPackedH * kExecN> packed =
                    xesimd::lsc_load_2d<uint8_t, kExecN, kPackedH, 1, false,
                                        false>(
                        pd, unsigned(cols - 1), unsigned(packed_rows - 1),
                        unsigned(cols - 1), col0, kc * kPackedH);
                const esimd::simd<int8_t, kKc * kExecN> bt =
                    pack_vnni4(lut_packed_to_s8(packed));
                acc = esimd::xmx::dpas<8, kRc, int32_t, int32_t, int8_t,
                                       int8_t>(acc, bt, afrag);
            }
            xesimd::lsc_store_2d<int32_t, kExecN, kRc>(
                cd, unsigned(cols * int(sizeof(int32_t)) - 1),
                unsigned(rows - 1),
                unsigned(cols * int(sizeof(int32_t)) - 1), col0, row0, acc);
        });
}

double event_us(sycl::event e) {
    e.wait_and_throw();
    const uint64_t t0 =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t t1 =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t1 - t0) / 1000.0;
}

void run_shape(sycl::queue &q, const char *phase, int m, int n, int k,
               int warmup, int iters, int *rc) {
    if (m % kRc != 0 || n % kExecN != 0 || k % kKc != 0) {
        std::fprintf(stderr, "nibble_lut_simd: shape not aligned\n");
        *rc = 2;
        return;
    }
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t np = size_t(k / 2) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na), hb_s8(nb);
    std::vector<uint8_t> nib(nb), packed(np);
    std::vector<int32_t> href(nc), hgot(nc);
    fill_s8(ha.data(), na, 1);
    fill_nibbles(nib.data(), nb, 9);
    pack_b(nib.data(), packed.data(), k, n);
    unpack_host(packed.data(), hb_s8.data(), k, n);
    host_s32(ha.data(), hb_s8.data(), href.data(), m, n, k);

    int8_t *ad = sycl::malloc_device<int8_t>(na, q);
    uint8_t *pd = sycl::malloc_device<uint8_t>(np, q);
    int32_t *cd = sycl::malloc_device<int32_t>(nc, q);
    q.memcpy(ad, ha.data(), na).wait();
    q.memcpy(pd, packed.data(), np).wait();

    launch_fused(q, ad, pd, cd, m, n, k).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(int32_t)).wait();
    const int mx = max_abs_diff(hgot.data(), href.data(), nc);
    const int ok = mx == 0 ? 1 : 0;
    if (!ok)
        *rc = 1;

    for (int i = 0; i < warmup; ++i)
        launch_fused(q, ad, pd, cd, m, n, k).wait_and_throw();
    double us = 0.0;
    for (int i = 0; i < iters; ++i)
        us += event_us(launch_fused(q, ad, pd, cd, m, n, k));
    us /= double(iters);
    const double ops = 2.0 * double(m) * double(n) * double(k);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    const double gbs = (double(k) * double(n) * 0.5 / 1.0e9) / (us * 1.0e-6);
    std::printf("%s,reg_lut_simd_vnni4,%d,%d,%d,1,%.3f,%.4f,%.3f,%d,%d\n", phase,
                m, n, k, us, tops, gbs, mx, ok);

    sycl::free(ad, q);
    sycl::free(pd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    int check_m = kRc, check_n = kExecN, check_k = kKc;
    int timed_m = 1024, timed_n = 1024, timed_k = 1024;
    int warmup = 5, iters = 20;
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
                         "nibble_lut_simd [--m 1024] [--n 1024] [--k 1024]\n");
            return 0;
        } else {
            std::fprintf(stderr, "nibble_lut_simd: unknown arg %s\n", a.c_str());
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
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "arm=e2m1_nibble_lut_simd_vnni4 never_bitcast_s4 warmup=%d "
                "iters=%d\n",
                backend, name.c_str(), driver.c_str(), warmup, iters);
    std::printf("phase,arm,m,n,k,launches,us,TOPS,GBs_packedB,max_abs,ok\n");
    int rc = 0;
    run_shape(q, "check", check_m, check_n, check_k, warmup, iters, &rc);
    run_shape(q, "timed", timed_m, timed_n, timed_k, warmup, iters, &rc);
    return rc;
}

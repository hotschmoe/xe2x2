// K6: ESIMD s4 DPAS + GPTQ group-scale f16 epilogue.
// Backend sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
// A synthetic s4 * 0.02. B real GPTQ s4 with per-group f16 scales.
// Partial s32 over gs=128, then * scale, f32 acc, store f16.
// Never E2M1 bitcast.
//
// Dump: uint32 k,n,gs; int8[k*n]; f16[(k/gs)*n]
// CSV: phase,m,n,k,us,cosine,max_abs,ok

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr int kPack = 2;
constexpr float kAScale = 0.02f;

struct DpasS4GptqName {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s4_gptq: no GPU\n");
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

sycl::event launch(sycl::queue &q, const uint8_t *ad, const uint8_t *bd,
                   const sycl::half *sd, sycl::half *cd, int rows, int cols,
                   int dk, int gs) {
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
    const int ng = dk / gs;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kExecN;
    const size_t threads = size_t(m_blocks * n_groups);
    const size_t local = (threads % size_t(kSg) == 0) ? size_t(kSg) : 1;
    constexpr int kAPacked = kKc / kPack;
    constexpr int kBPackedH = kKc / kPack;
    const int dpas_per_g = gs / kKc;
    return q.parallel_for<DpasS4GptqName>(
        sycl::nd_range<1>({threads}, {local}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int64_t tid = int64_t(it.get_global_id(0));
            const int64_t mb = tid % m_blocks;
            const int64_t ngi = tid / m_blocks;
            const int row0 = int(mb * kRc);
            const int col0 = int(ngi * kExecN);
            esimd::simd<float, kRc * kExecN> facc(0);
            for (int g = 0; g < ng; ++g) {
                esimd::simd<int32_t, kRc * kExecN> acc(0);
                for (int p = 0; p < dpas_per_g; ++p) {
                    const int kc = g * dpas_per_g + p;
                    const esimd::simd<uint8_t, kRc * kAPacked> afrag =
                        xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                            ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                            unsigned(a_pitch - 1), kc * kAPacked, row0);
                    const esimd::simd<uint8_t, kBPackedH * kExecN> bt =
                        xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1,
                                            false, true>(
                            bd, unsigned(cols - 1), unsigned(b_rows - 1),
                            unsigned(cols - 1), col0, kc * kBPackedH);
                    acc = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t, uint8_t,
                                    xmx::dpas_argument_type::s4,
                                    xmx::dpas_argument_type::s4>(acc, bt, afrag);
                }
                esimd::simd<sycl::half, kExecN> bsh;
                bsh.copy_from(sd + size_t(g) * size_t(cols) + size_t(col0));
                esimd::simd<float, kExecN> bsf = esimd::convert<float>(bsh);
#pragma unroll
                for (int r = 0; r < kRc; ++r) {
#pragma unroll
                    for (int c = 0; c < kExecN; ++c) {
                        const float s = kAScale * float(bsf[c]);
                        facc[r * kExecN + c] += float(acc[r * kExecN + c]) * s;
                    }
                }
            }
            esimd::simd<sycl::half, kRc * kExecN> h;
#pragma unroll
            for (int i = 0; i < kRc * kExecN; ++i)
                h[i] = sycl::half(facc[i]);
            xesimd::lsc_store_2d<sycl::half, kExecN, kRc>(
                cd, unsigned(cols * int(sizeof(sycl::half)) - 1),
                unsigned(rows - 1),
                unsigned(cols * int(sizeof(sycl::half)) - 1), col0, row0, h);
        });
}

int load_dump(const char *path, std::vector<int8_t> *hb,
              std::vector<sycl::half> *hs, int *k, int *n, int *gs) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "dpas_s4_gptq: open %s\n", path);
        return 2;
    }
    uint32_t hk = 0, hn = 0, hgs = 0;
    if (std::fread(&hk, 4, 1, f) != 1 || std::fread(&hn, 4, 1, f) != 1 ||
        std::fread(&hgs, 4, 1, f) != 1) {
        std::fclose(f);
        return 2;
    }
    *k = int(hk);
    *n = int(hn);
    *gs = int(hgs);
    hb->assign(size_t(*k) * size_t(*n), 0);
    if (std::fread(hb->data(), 1, hb->size(), f) != hb->size()) {
        std::fclose(f);
        return 2;
    }
    const size_t ns = size_t(*k / *gs) * size_t(*n);
    std::vector<uint16_t> raw(ns);
    if (std::fread(raw.data(), 2, ns, f) != ns) {
        std::fclose(f);
        return 2;
    }
    std::fclose(f);
    hs->resize(ns);
    for (size_t i = 0; i < ns; ++i) {
        uint16_t bits = raw[i];
        sycl::half h;
        std::memcpy(&h, &bits, 2);
        (*hs)[i] = h;
    }
    return 0;
}

void run_shape(sycl::queue &q, const char *phase, int m, int n, int k, int gs,
               const int8_t *hb_full, const sycl::half *hs_full, int dump_n,
               int warmup, int iters, int *rc) {
    if (m % kRc != 0 || n % kExecN != 0 || k % gs != 0 || gs % kKc != 0) {
        std::fprintf(stderr, "dpas_s4_gptq: shape m=%d n=%d k=%d gs=%d\n", m, n,
                     k, gs);
        *rc = 2;
        return;
    }
    const int ng = k / gs;
    const size_t na = size_t(m) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    const size_t na_p = size_t(m) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    const size_t ns = size_t(ng) * size_t(n);
    std::vector<int8_t> ha(na), hb(nb);
    std::vector<uint8_t> pa(na_p), pb(nb_p);
    std::vector<sycl::half> hs(ns), href(nc), hgot(nc);
    for (int kk = 0; kk < k; ++kk)
        for (int j = 0; j < n; ++j)
            hb[size_t(kk) * size_t(n) + size_t(j)] =
                hb_full[size_t(kk) * size_t(dump_n) + size_t(j)];
    for (int g = 0; g < ng; ++g)
        for (int j = 0; j < n; ++j)
            hs[size_t(g) * size_t(n) + size_t(j)] =
                hs_full[size_t(g) * size_t(dump_n) + size_t(j)];
    fill_s4(ha.data(), na, 1);
    pack_a(ha.data(), pa.data(), m, k);
    pack_b(hb.data(), pb.data(), k, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int g = 0; g < ng; ++g) {
                int32_t s32 = 0;
                for (int kk = 0; kk < gs; ++kk) {
                    const int ki = g * gs + kk;
                    s32 += int32_t(ha[size_t(i) * size_t(k) + size_t(ki)]) *
                           int32_t(hb[size_t(ki) * size_t(n) + size_t(j)]);
                }
                acc += float(s32) * kAScale * float(hs[size_t(g) * size_t(n) + size_t(j)]);
            }
            href[size_t(i) * size_t(n) + size_t(j)] = sycl::half(acc);
        }
    }

    uint8_t *ad = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *bd = sycl::malloc_device<uint8_t>(nb_p, q);
    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nc, q);
    q.memcpy(ad, pa.data(), na_p).wait();
    q.memcpy(bd, pb.data(), nb_p).wait();
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();

    launch(q, ad, bd, sd, cd, m, n, k, gs).wait_and_throw();
    q.memcpy(hgot.data(), cd, nc * sizeof(sycl::half)).wait();
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

    for (int i = 0; i < warmup; ++i)
        launch(q, ad, bd, sd, cd, m, n, k, gs).wait_and_throw();
    uint64_t ns_sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = launch(q, ad, bd, sd, cd, m, n, k, gs);
        e.wait_and_throw();
        const uint64_t t0 =
            e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 =
            e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ns_sum += (t1 - t0);
    }
    const double us = (double(ns_sum) / 1000.0) / double(iters);
    std::printf("%s,%d,%d,%d,%.3f,%.6f,%.5g,%d\n", phase, m, n, k, us, cosine,
                mx, ok);

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(sd, q);
    sycl::free(cd, q);
}

} // namespace

int main(int argc, char **argv) {
    const char *bin = nullptr;
    int warmup = 5, iters = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--b-bin") {
            if (i + 1 < argc)
                bin = argv[++i];
        } else if (a == "--iters")
            take(iters);
        else if (a == "--warmup")
            take(warmup);
        else {
            std::fprintf(stderr, "dpas_s4_gptq: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (!bin) {
        std::fprintf(stderr, "dpas_s4_gptq: need --b-bin\n");
        return 2;
    }

    std::vector<int8_t> hb;
    std::vector<sycl::half> hs;
    int bk = 0, bn = 0, gs = 0;
    if (load_dump(bin, &hb, &hs, &bk, &bn, &gs) != 0)
        return 2;

    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s dtype=s4xs4_gptq_f16 "
                "arm=dpas_s4_gptq gs=%d dump_k=%d dump_n=%d a_scale=%.4f RC=%d "
                "Kc=%d\n",
                backend, name.c_str(), driver.c_str(), gs, bk, bn,
                double(kAScale), kRc, kKc);
    std::printf("phase,m,n,k,us,cosine,max_abs,ok\n");

    int rc = 0;
    run_shape(q, "check", kRc, kExecN, gs, gs, hb.data(), hs.data(), bn, warmup,
              iters, &rc);
    run_shape(q, "tile", kRc, (bn / kExecN) * kExecN, (bk / gs) * gs, gs,
              hb.data(), hs.data(), bn, warmup, iters, &rc);
    return rc;
}

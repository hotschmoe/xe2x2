// K6 ESIMD s4 RC=8 GPTQ group-scale f16, A-db, wg 4x8, M=64.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// 8x2-N GPTQ lost at M=64 (123.5 us) vs W8A8 46 and s4 4x8 33.6.
// Steal 4x8 A-db. Partial s32 per g128, then * a_scale * b_scale[g,n].
// Never E2M1. Rank pipe_host vs s4 33.6, mix 43.3, W8A8 46.

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
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

namespace {

constexpr int kRc = 8;
constexpr int kKc = 64;
constexpr int kK64 = 64;
constexpr int kGs = 128;
constexpr int kExecN = 16;
constexpr int kWgX = 4;
constexpr int kWgY = 8;
constexpr int kPack = 2;
constexpr int kAPacked = kKc / kPack;
constexpr int kBPackedH = kKc / kPack;
constexpr float kScale = 0.02f;

struct DpasS4GptqDb48Nt2Name {};
struct DpasS4GptqDb48Nt4Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;
const int8_t *g_hb = nullptr;
const sycl::half *g_hs = nullptr;
int g_dump_k = 0;
int g_dump_n = 0;
int g_dump_gs = 0;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "dpas_s4_gptq_db48: no GPU\n");
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
sycl::event launch(sycl::queue &q, const uint8_t *ad, const uint8_t *bd,
                   const float *asd, const sycl::half *sd, sycl::half *cd,
                   int rows, int cols, int dk) {
    constexpr int kTN = NT * kExecN;
    const int a_pitch = dk / kPack;
    const int b_rows = dk / kPack;
    const int ngrp = dk / kGs;
    const int dpas_per_g = kGs / kK64;
    const int64_t m_blocks = rows / kRc;
    const int64_t n_groups = cols / kTN;
    const size_t n_wgs = round_up(n_groups, kWgX) / size_t(kWgX);
    const size_t m_wgs = round_up(m_blocks, kWgY) / size_t(kWgY);
    const size_t g0 = n_wgs * size_t(kWgX);
    const size_t g1 = m_wgs * size_t(kWgY);
    (void)kUnroll;
    syclex::properties props{intelex::grf_size<256>};
    return q.parallel_for<Name>(
        sycl::nd_range<2>({g0, g1}, {size_t(kWgX), size_t(kWgY)}), props,
        [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
            const int64_t ng = int64_t(it.get_group(0)) * kWgX +
                               int64_t(it.get_local_id(0));
            const int64_t mb = int64_t(it.get_group(1)) * kWgY +
                               int64_t(it.get_local_id(1));
            if (ng >= n_groups || mb >= m_blocks)
                return;
            const int row0 = int(mb * kRc);
            const int col0 = int(ng * kTN);
            esimd::simd<float, kRc * kExecN> facc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t)
                facc[t] = 0;
            esimd::simd<float, kRc> asv;
            asv.copy_from(asd + row0);
            esimd::simd<uint8_t, kRc * kAPacked> a0 =
                xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                    ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                    unsigned(a_pitch - 1), 0, row0);
            for (int g = 0; g < ngrp; ++g) {
                esimd::simd<int32_t, kRc * kExecN> acc[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    acc[t] = 0;
                for (int p = 0; p < dpas_per_g; ++p) {
                    const int k0 = g * kGs + p * kK64;
                    const int pk0 = k0 / kPack;
                    const int knext = k0 + kK64;
                    const bool more = knext < dk;
                    esimd::simd<uint8_t, kRc * kAPacked> a0n = a0;
                    if (more)
                        a0n = xesimd::lsc_load_2d<uint8_t, kAPacked, kRc>(
                            ad, unsigned(a_pitch - 1), unsigned(rows - 1),
                            unsigned(a_pitch - 1), knext / kPack, row0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const esimd::simd<uint8_t, kBPackedH * kExecN> bt =
                            xesimd::lsc_load_2d<uint8_t, kExecN, kBPackedH, 1,
                                                false, true>(
                                bd, unsigned(cols - 1), unsigned(b_rows - 1),
                                unsigned(cols - 1), col0 + t * kExecN, pk0);
                        acc[t] = xmx::dpas<8, kRc, int32_t, int32_t, uint8_t,
                                           uint8_t, xmx::dpas_argument_type::s4,
                                           xmx::dpas_argument_type::s4>(
                            acc[t], bt, a0);
                    }
                    if (more)
                        a0 = a0n;
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    esimd::simd<sycl::half, kExecN> bsh;
                    bsh.copy_from(sd + size_t(g) * size_t(cols) +
                                  size_t(col0 + t * kExecN));
                    esimd::simd<float, kExecN> bsf = esimd::convert<float>(bsh);
#pragma unroll
                    for (int r = 0; r < kRc; ++r) {
                        const float ar = asv[r];
#pragma unroll
                        for (int c = 0; c < kExecN; ++c)
                            facc[t][r * kExecN + c] +=
                                float(acc[t][r * kExecN + c]) * ar *
                                float(bsf[c]);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                esimd::simd<sycl::half, kRc * kExecN> h;
#pragma unroll
                for (int i = 0; i < kRc * kExecN; ++i)
                    h[i] = sycl::half(float(facc[t][i]));
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
    if (m < 1 || n % tn != 0 || k % inner_k != 0 || k % kGs != 0 ||
        k > g_dump_k || n > g_dump_n) {
        std::fprintf(stderr,
                     "dpas_s4_gptq_sc: shape m=%d n=%d k=%d nt=%d unroll=%d\n",
                     m, n, k, nt, unroll);
        *rc = 2;
        return;
    }
    const int rows = ((m + kRc - 1) / kRc) * kRc;
    const int ngrp = k / kGs;
    const size_t na = size_t(rows) * size_t(k);
    const size_t nb = size_t(k) * size_t(n);
    const size_t na_p = size_t(rows) * size_t(k / kPack);
    const size_t nb_p = size_t(k / kPack) * size_t(n);
    const size_t ns = size_t(ngrp) * size_t(n);
    const size_t nc_pad = size_t(rows) * size_t(n);
    const size_t nc = size_t(m) * size_t(n);
    std::vector<int8_t> ha(na, 0), hb(nb);
    std::vector<uint8_t> pa(na_p), pb(nb_p);
    std::vector<float> has(size_t(rows), 0.f);
    std::vector<sycl::half> hs(ns), href(nc), hgot(nc), hpad(nc_pad);
    fill_s4(ha.data(), size_t(m) * size_t(k), 1);
    for (int kk = 0; kk < k; ++kk)
        for (int j = 0; j < n; ++j)
            hb[size_t(kk) * size_t(n) + size_t(j)] =
                g_hb[size_t(kk) * size_t(g_dump_n) + size_t(j)];
    for (int g = 0; g < ngrp; ++g)
        for (int j = 0; j < n; ++j)
            hs[size_t(g) * size_t(n) + size_t(j)] =
                g_hs[size_t(g) * size_t(g_dump_n) + size_t(j)];
    pack_a(ha.data(), pa.data(), rows, k);
    pack_b(hb.data(), pb.data(), k, n);
    for (int i = 0; i < m; ++i)
        has[size_t(i)] = kScale;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int g = 0; g < ngrp; ++g) {
                int32_t s32 = 0;
                for (int kk = 0; kk < kGs; ++kk) {
                    const int ki = g * kGs + kk;
                    s32 += int32_t(ha[size_t(i) * size_t(k) + size_t(ki)]) *
                           int32_t(hb[size_t(ki) * size_t(n) + size_t(j)]);
                }
                acc += float(s32) * has[size_t(i)] *
                       float(hs[size_t(g) * size_t(n) + size_t(j)]);
            }
            href[size_t(i) * size_t(n) + size_t(j)] = sycl::half(acc);
        }
    }

    uint8_t *ad = sycl::malloc_device<uint8_t>(na_p, q);
    uint8_t *bd = sycl::malloc_device<uint8_t>(nb_p, q);
    float *asd = sycl::malloc_device<float>(size_t(rows), q);
    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *cd = sycl::malloc_device<sycl::half>(nc_pad, q);
    q.memcpy(ad, pa.data(), na_p).wait();
    q.memcpy(bd, pb.data(), nb_p).wait();
    q.memcpy(asd, has.data(), size_t(rows) * sizeof(float)).wait();
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        if (nt == 4)
            return launch<DpasS4GptqScNt4Name, 4, 8>(q, ad, bd, asd, sd, cd,
                                                     rows, n, k);
        return launch<DpasS4GptqScNt2Name, 2, 16>(q, ad, bd, asd, sd, cd, rows,
                                                  n, k);
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
    sycl::free(sd, q);
    sycl::free(cd, q);
}

int load_dump(const char *path, std::vector<int8_t> *hb,
              std::vector<sycl::half> *hs, int *k, int *n, int *gs) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "dpas_s4_gptq_sc: open %s\n", path);
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

} // namespace

int main(int argc, char **argv) {
    int nt = 2;
    int timed_m = 1, timed_n = 5120, timed_k = 5120;
    int warmup = 50, iters = 40;
    const char *bin = nullptr;
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
        else if (a == "--b-bin") {
            if (i + 1 < argc)
                bin = argv[++i];
        } else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "dpas_s4_gptq_sc --b-bin file --nt 2|4 [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "dpas_s4_gptq_sc: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (!bin) {
        std::fprintf(stderr, "dpas_s4_gptq_sc: need --b-bin\n");
        return 2;
    }
    std::vector<int8_t> hb_store;
    std::vector<sycl::half> hs_store;
    if (load_dump(bin, &hb_store, &hs_store, &g_dump_k, &g_dump_n, &g_dump_gs) !=
        0)
        return 2;
    if (g_dump_gs != kGs) {
        std::fprintf(stderr, "dpas_s4_gptq_sc: gs %d != %d\n", g_dump_gs, kGs);
        return 2;
    }
    g_hb = hb_store.data();
    g_hs = hs_store.data();
    if (nt != 2 && nt != 4) {
        std::fprintf(stderr, "dpas_s4_gptq_sc: --nt must be 2 or 4\n");
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
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "dtype=s4xs4_gptq->f16 RC=%d NT=%d unroll=%d gs=%d "
                "wg=%dx%d_alongN padM=RC pack=2 a_scale=%.4f dump_k=%d dump_n=%d "
                "out=f16 warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), kRc, nt, unroll, kGs,
                kWgX, kWgY, double(kScale), g_dump_k, g_dump_n, warmup, iters,
                g_card, g_spin);
    int rc = 0;
    run_shape(q, nt, unroll, "check", kRc, nt * kExecN, unroll * kK64, 1, 1, &rc,
              0);
    run_shape(q, nt, unroll, "timed", timed_m, timed_n, timed_k, warmup, iters,
              &rc, 1);
    return rc;
}

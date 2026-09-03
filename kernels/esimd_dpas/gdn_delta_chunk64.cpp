// K7 ESIMD GDN chunk/WY delta prefill T=256 C=64. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// FLA default chunk 64. C=16 lost 2.92x vs fused 1100.
// Same oracle as fused gdn_delta_t. Rank pipe_host vs 1100.

#include <sycl/ext/intel/esimd.hpp>
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

namespace {

constexpr int kVl = 16;
constexpr int kWg = 16;
constexpr int kC = 64;
constexpr int kPer = kC / kWg;
constexpr int kNv = 48;
constexpr int kDv = 128;
constexpr int kDk = 128;
constexpr int kTiles = kDk / kVl;

constexpr int kOffK = 0;
constexpr int kOffQ = kC * kDk * int(sizeof(sycl::half));
constexpr int kOffA = kOffQ + kC * kDk * int(sizeof(sycl::half));
constexpr int kOffW = kOffA + kC * kC * int(sizeof(float));
constexpr int kOffG = kOffW + kC * kDk * int(sizeof(float));
constexpr int kOffB = kOffG + kC * int(sizeof(float));
constexpr int kOffAl = kOffB + kC * int(sizeof(float));
constexpr int kOffQK = kOffAl + kC * int(sizeof(float));
constexpr int kSlmBytes = kOffQK + kC * kC * int(sizeof(float));

struct GdnDeltaChunk64Name {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_delta_chunk64: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void fill_f16(sycl::half *p, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        const float v = float(int((i * 17u + seed) % 201u) - 100) / 250.f;
        p[i] = sycl::half(v);
    }
}

void l2_norm_rows(sycl::half *p, int rows, int dim) {
    for (int r = 0; r < rows; ++r) {
        float ss = 0.f;
        for (int j = 0; j < dim; ++j) {
            const float x = float(p[size_t(r) * size_t(dim) + size_t(j)]);
            ss += x * x;
        }
        const float inv = (ss > 0.f) ? 1.f / std::sqrt(ss) : 0.f;
        for (int j = 0; j < dim; ++j) {
            const size_t idx = size_t(r) * size_t(dim) + size_t(j);
            p[idx] = sycl::half(float(p[idx]) * inv);
        }
    }
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

inline float hsum16(esimd::simd<float, kVl> v) {
    float s = 0.f;
#pragma unroll
    for (int e = 0; e < kVl; ++e)
        s += float(v[e]);
    return s;
}

sycl::event launch(sycl::queue &q, const sycl::half *sd, const sycl::half *qd,
                   const sycl::half *kd, const sycl::half *vd,
                   const sycl::half *ad, const sycl::half *bd, sycl::half *s2d,
                   sycl::half *od, int tlen) {
    const int nchunks = tlen / kC;
    return q.parallel_for<GdnDeltaChunk64Name>(
        sycl::nd_range<1>({size_t(kNv * kDv)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            esimd::slm_init<kSlmBytes>();
            const int gid = int(it.get_global_id(0));
            const int lid = int(it.get_local_id(0));
            const int h = gid / kDv;
            const int i = gid % kDv;
            const size_t srow = (size_t(h) * size_t(kDv) + size_t(i)) *
                                size_t(kDk);
            esimd::simd<float, kVl> sf[kTiles];
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                esimd::simd<sycl::half, kVl> sv;
                sv.copy_from(sd + srow + size_t(tile * kVl));
                sf[tile] = esimd::convert<float>(sv);
            }
            for (int ch = 0; ch < nchunks; ++ch) {
                const int t0 = ch * kC;
                for (int r = 0; r < kPer; ++r) {
                    const int tloc = lid + r * kWg;
                    const size_t kbase =
                        (size_t(t0 + tloc) * size_t(kNv) + size_t(h)) *
                        size_t(kDk);
#pragma unroll
                    for (int tile = 0; tile < kTiles; ++tile) {
                        esimd::simd<sycl::half, kVl> kv, qv;
                        kv.copy_from(kd + kbase + size_t(tile * kVl));
                        qv.copy_from(qd + kbase + size_t(tile * kVl));
                        const int koff = kOffK + (tloc * kDk + tile * kVl) *
                                                     int(sizeof(sycl::half));
                        const int qoff = kOffQ + (tloc * kDk + tile * kVl) *
                                                     int(sizeof(sycl::half));
                        esimd::slm_block_store<sycl::half, kVl>(koff, kv);
                        esimd::slm_block_store<sycl::half, kVl>(qoff, qv);
                    }
                    esimd::simd<sycl::half, 1> ahv, bhv;
                    const size_t abase =
                        size_t(t0 + tloc) * size_t(kNv) + size_t(h);
                    ahv.copy_from(ad + abase);
                    bhv.copy_from(bd + abase);
                    esimd::simd<float, 1> af, bf;
                    af[0] = float(sycl::half(ahv[0]));
                    bf[0] = float(sycl::half(bhv[0]));
                    esimd::slm_block_store<float, 1>(
                        kOffAl + tloc * int(sizeof(float)), af);
                    esimd::slm_block_store<float, 1>(
                        kOffB + tloc * int(sizeof(float)), bf);
                }
                it.barrier(sycl::access::fence_space::local_space);
                float alpha[kC], beta[kC], g[kC];
                for (int t = 0; t < kC; ++t) {
                    esimd::simd<float, 1> af, bf;
                    af = esimd::slm_block_load<float, 1>(kOffAl +
                                                         t * int(sizeof(float)));
                    bf = esimd::slm_block_load<float, 1>(kOffB +
                                                         t * int(sizeof(float)));
                    alpha[t] = float(af[0]);
                    beta[t] = float(bf[0]);
                }
                float pref = 0.f;
                for (int t = 0; t < kC; ++t) {
                    pref += sycl::log2(alpha[t]);
                    g[t] = pref;
                    if (lid == 0) {
                        esimd::simd<float, 1> gf;
                        gf[0] = g[t];
                        esimd::slm_block_store<float, 1>(
                            kOffG + t * int(sizeof(float)), gf);
                    }
                }
                for (int r = 0; r < kPer; ++r) {
                    const int tloc = lid + r * kWg;
                    for (int s = 0; s < kC; ++s) {
                        float kacc = 0.f;
                        float qacc = 0.f;
#pragma unroll
                        for (int tile = 0; tile < kTiles; ++tile) {
                            const int kt = kOffK + (tloc * kDk + tile * kVl) *
                                                       int(sizeof(sycl::half));
                            const int qt = kOffQ + (tloc * kDk + tile * kVl) *
                                                       int(sizeof(sycl::half));
                            const int ks = kOffK + (s * kDk + tile * kVl) *
                                                       int(sizeof(sycl::half));
                            const esimd::simd<sycl::half, kVl> a =
                                esimd::slm_block_load<sycl::half, kVl>(kt);
                            const esimd::simd<sycl::half, kVl> b =
                                esimd::slm_block_load<sycl::half, kVl>(ks);
                            const esimd::simd<sycl::half, kVl> qq =
                                esimd::slm_block_load<sycl::half, kVl>(qt);
                            kacc += hsum16(esimd::convert<float>(a) *
                                           esimd::convert<float>(b));
                            qacc += hsum16(esimd::convert<float>(qq) *
                                           esimd::convert<float>(b));
                        }
                        float lts = 0.f;
                        if (tloc > s)
                            lts = beta[tloc] * sycl::exp2(g[tloc] - g[s]) * kacc;
                        esimd::simd<float, 1> lv, qv;
                        lv[0] = lts;
                        qv[0] = qacc;
                        esimd::slm_block_store<float, 1>(
                            kOffA + (tloc * kC + s) * int(sizeof(float)), lv);
                        esimd::slm_block_store<float, 1>(
                            kOffQK + (tloc * kC + s) * int(sizeof(float)), qv);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid == 0) {
                    for (int t = 0; t < kC; ++t) {
                        for (int s = 0; s < t; ++s) {
                            float acc = 0.f;
                            for (int p = s; p < t; ++p) {
                                esimd::simd<float, 1> atp, aps;
                                atp = esimd::slm_block_load<float, 1>(
                                    kOffA + (t * kC + p) * int(sizeof(float)));
                                aps = esimd::slm_block_load<float, 1>(
                                    kOffA + (p * kC + s) * int(sizeof(float)));
                                acc += float(atp[0]) * float(aps[0]);
                            }
                            esimd::simd<float, 1> av;
                            av[0] = -acc;
                            esimd::slm_block_store<float, 1>(
                                kOffA + (t * kC + s) * int(sizeof(float)), av);
                        }
                        esimd::simd<float, 1> one;
                        one[0] = 1.f;
                        esimd::slm_block_store<float, 1>(
                            kOffA + (t * kC + t) * int(sizeof(float)), one);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                if (lid < kTiles) {
                    const int tile = lid;
                    for (int t = 0; t < kC; ++t) {
                        esimd::simd<float, kVl> wacc(0.f);
                        for (int s = 0; s < kC; ++s) {
                            esimd::simd<float, 1> av;
                            av = esimd::slm_block_load<float, 1>(
                                kOffA + (t * kC + s) * int(sizeof(float)));
                            const float scale =
                                float(av[0]) * beta[s] * sycl::exp2(g[s]);
                            const int ks = kOffK + (s * kDk + tile * kVl) *
                                                       int(sizeof(sycl::half));
                            const esimd::simd<sycl::half, kVl> kh =
                                esimd::slm_block_load<sycl::half, kVl>(ks);
                            wacc += esimd::convert<float>(kh) * scale;
                        }
                        esimd::slm_block_store<float, kVl>(
                            kOffW + (t * kDk + tile * kVl) * int(sizeof(float)),
                            wacc);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                float vv[kC];
                for (int t = 0; t < kC; ++t) {
                    const size_t vbase =
                        (size_t(t0 + t) * size_t(kNv) + size_t(h)) * size_t(kDv);
                    esimd::simd<sycl::half, 1> viv;
                    viv.copy_from(vd + vbase + size_t(i));
                    vv[t] = float(sycl::half(viv[0]));
                }
                float u[kC], vnew[kC], s0q[kC];
                for (int t = 0; t < kC; ++t) {
                    float accu = 0.f;
                    for (int s = 0; s < kC; ++s) {
                        esimd::simd<float, 1> av;
                        av = esimd::slm_block_load<float, 1>(
                            kOffA + (t * kC + s) * int(sizeof(float)));
                        accu += float(av[0]) * beta[s] * vv[s];
                    }
                    u[t] = accu;
                    float wdot = 0.f;
                    float qdot = 0.f;
#pragma unroll
                    for (int tile = 0; tile < kTiles; ++tile) {
                        const esimd::simd<float, kVl> wv =
                            esimd::slm_block_load<float, kVl>(
                                kOffW + (t * kDk + tile * kVl) *
                                            int(sizeof(float)));
                        wdot += hsum16(wv * sf[tile]);
                        const int qo = kOffQ + (t * kDk + tile * kVl) *
                                                   int(sizeof(sycl::half));
                        const esimd::simd<sycl::half, kVl> qh =
                            esimd::slm_block_load<sycl::half, kVl>(qo);
                        qdot += hsum16(sf[tile] * esimd::convert<float>(qh));
                    }
                    vnew[t] = u[t] - wdot;
                    s0q[t] = qdot;
                }
                const float glast = g[kC - 1];
                const float eg = sycl::exp2(glast);
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    esimd::simd<float, kVl> acc = sf[tile] * eg;
                    for (int t = 0; t < kC; ++t) {
                        const float vn = vnew[t] * sycl::exp2(glast - g[t]);
                        const int ko = kOffK + (t * kDk + tile * kVl) *
                                                   int(sizeof(sycl::half));
                        const esimd::simd<sycl::half, kVl> kh =
                            esimd::slm_block_load<sycl::half, kVl>(ko);
                        acc += esimd::convert<float>(kh) * vn;
                    }
                    sf[tile] = acc;
                }
                for (int t = 0; t < kC; ++t) {
                    float oacc = sycl::exp2(g[t]) * s0q[t];
                    for (int s = 0; s <= t; ++s) {
                        esimd::simd<float, 1> qk;
                        qk = esimd::slm_block_load<float, 1>(
                            kOffQK + (t * kC + s) * int(sizeof(float)));
                        oacc += sycl::exp2(g[t] - g[s]) * float(qk[0]) * vnew[s];
                    }
                    esimd::simd<sycl::half, 1> ov;
                    ov[0] = sycl::half(oacc);
                    const size_t vbase =
                        (size_t(t0 + t) * size_t(kNv) + size_t(h)) * size_t(kDv);
                    ov.copy_to(od + vbase + size_t(i));
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                esimd::simd<sycl::half, kVl> sh;
#pragma unroll
                for (int e = 0; e < kVl; ++e)
                    sh[e] = sycl::half(float(sf[tile][e]));
                sh.copy_to(s2d + srow + size_t(tile * kVl));
            }
        });
}

void host_delta(const sycl::half *s, const sycl::half *q, const sycl::half *k,
                const sycl::half *v, const sycl::half *a, const sycl::half *b,
                sycl::half *s2, sycl::half *o, int tlen) {
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    std::vector<float> sf(ns);
    for (size_t i = 0; i < ns; ++i)
        sf[i] = float(s[i]);
    for (int t = 0; t < tlen; ++t) {
        for (int h = 0; h < kNv; ++h) {
            const float ah = float(a[size_t(t) * size_t(kNv) + size_t(h)]);
            const float bh = float(b[size_t(t) * size_t(kNv) + size_t(h)]);
            for (int i = 0; i < kDv; ++i) {
                const size_t srow =
                    (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
                const size_t kbase =
                    (size_t(t) * size_t(kNv) + size_t(h)) * size_t(kDk);
                const size_t vbase =
                    (size_t(t) * size_t(kNv) + size_t(h)) * size_t(kDv);
                float vold = 0.f;
                for (int j = 0; j < kDk; ++j)
                    vold += sf[srow + size_t(j)] * float(k[kbase + size_t(j)]);
                const float vi = float(v[vbase + size_t(i)]);
                float oacc = 0.f;
                for (int j = 0; j < kDk; ++j) {
                    const float kj = float(k[kbase + size_t(j)]);
                    const float snew =
                        ah * (sf[srow + size_t(j)] - bh * vold * kj) +
                        bh * vi * kj;
                    sf[srow + size_t(j)] = snew;
                    oacc += snew * float(q[kbase + size_t(j)]);
                }
                o[vbase + size_t(i)] = sycl::half(oacc);
            }
        }
    }
    for (size_t i = 0; i < ns; ++i)
        s2[i] = sycl::half(sf[i]);
}

void score_f16(const sycl::half *got, const sycl::half *ref, size_t n,
               double *cosine, double *mx) {
    double dot = 0, na2 = 0, nb2 = 0, m = 0;
    for (size_t i = 0; i < n; ++i) {
        const float x = float(got[i]);
        const float y = float(ref[i]);
        const float d = x > y ? x - y : y - x;
        if (d > m)
            m = d;
        dot += double(x) * double(y);
        na2 += double(x) * double(x);
        nb2 += double(y) * double(y);
    }
    *cosine = (na2 > 0 && nb2 > 0) ? dot / std::sqrt(na2 * nb2) : 0.0;
    *mx = m;
}

void run_shape(sycl::queue &q, const char *phase, int tlen, int warmup,
               int iters, int *rc, int do_spin) {
    if (tlen < kC || (tlen % kC) != 0) {
        std::fprintf(stderr, "gdn_delta_chunk64: T=%d need multiple of %d\n", tlen,
                     kC);
        *rc = 2;
        return;
    }
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    const size_t nq = size_t(tlen) * size_t(kNv) * size_t(kDk);
    const size_t nv = size_t(tlen) * size_t(kNv) * size_t(kDv);
    const size_t nsc = size_t(tlen) * size_t(kNv);
    std::vector<sycl::half> hs(ns), hq(nq), hk(nq), hv(nv), ha(nsc), hb(nsc);
    std::vector<sycl::half> hrefs(ns), hgots(ns), hrefo(nv), hgoto(nv);
    fill_f16(hs.data(), ns, 1);
    fill_f16(hq.data(), nq, 3);
    fill_f16(hk.data(), nq, 5);
    fill_f16(hv.data(), nv, 7);
    l2_norm_rows(hq.data(), tlen * kNv, kDk);
    l2_norm_rows(hk.data(), tlen * kNv, kDk);
    for (int t = 0; t < tlen; ++t) {
        for (int h = 0; h < kNv; ++h) {
            const size_t idx = size_t(t) * size_t(kNv) + size_t(h);
            ha[idx] = sycl::half(0.25f + float(h % 8) * 0.05f);
            hb[idx] = sycl::half(0.30f + float(h % 5) * 0.08f);
        }
    }
    host_delta(hs.data(), hq.data(), hk.data(), hv.data(), ha.data(),
               hb.data(), hrefs.data(), hrefo.data(), tlen);

    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *qd = sycl::malloc_device<sycl::half>(nq, q);
    sycl::half *kd = sycl::malloc_device<sycl::half>(nq, q);
    sycl::half *vd = sycl::malloc_device<sycl::half>(nv, q);
    sycl::half *ad = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *s2d = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *od = sycl::malloc_device<sycl::half>(nv, q);
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();
    q.memcpy(qd, hq.data(), nq * sizeof(sycl::half)).wait();
    q.memcpy(kd, hk.data(), nq * sizeof(sycl::half)).wait();
    q.memcpy(vd, hv.data(), nv * sizeof(sycl::half)).wait();
    q.memcpy(ad, ha.data(), nsc * sizeof(sycl::half)).wait();
    q.memcpy(bd, hb.data(), nsc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, sd, qd, kd, vd, ad, bd, s2d, od, tlen);
    };

    go().wait_and_throw();
    q.memcpy(hgots.data(), s2d, ns * sizeof(sycl::half)).wait();
    q.memcpy(hgoto.data(), od, nv * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_o = 0, mx_o = 0;
    score_f16(hgots.data(), hrefs.data(), ns, &cosine, &mx);
    score_f16(hgoto.data(), hrefo.data(), nv, &cosine_o, &mx_o);
    const int ok = (cosine > 0.99 && cosine_o > 0.99) ? 1 : 0;
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
    const double nbytes =
        double((ns + ns + nq + nq + nv + nsc + nsc + nv) * sizeof(sycl::half));
    const double gbs = (nbytes / 1.0e9) / (pipe_us * 1.0e-6);
    std::printf("phase,t,nv,dv,dk,C,event_us,wait_host_us,pipe_host_us,GBs,"
                "cosine,max_abs,cosine_o,max_abs_o,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%.6f,%.5g,%d,"
                "%.3f,%.3f,%.3f\n",
                phase, tlen, kNv, kDv, kDk, kC, us, wait_host, pipe_us, gbs,
                cosine, mx, cosine_o, mx_o, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(sd, q);
    sycl::free(qd, q);
    sycl::free(kd, q);
    sycl::free(vd, q);
    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(s2d, q);
    sycl::free(od, q);
}

} // namespace

int main(int argc, char **argv) {
    int timed_t = 256;
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
        if (a == "--t")
            take(timed_t);
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
            std::fprintf(stderr, "gdn_delta_chunk64 [--t 256] [--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_delta_chunk64: unknown arg %s\n", a.c_str());
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
                "op=gdn_delta_chunk64 dtype=f16 T=%d C=%d nv=%d dv=%d dk=%d VL=%d "
                "warmup=%d iters=%d card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), timed_t, kC, kNv, kDv,
                kDk, kVl, warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, "timed", timed_t, warmup, iters, &rc, 1);
    return rc;
}

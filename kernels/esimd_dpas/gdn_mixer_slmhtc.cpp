// K7 ESIMD GDN mixer T=256: in-kernel conv1d plus slmht delta. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Raw packed q+k+v C=10240. Each slmht load applies causal K=4 FIR;
// q/k are L2-normalized in registers before their existing half SLM stores.
// CONFIG prior: sequential conv 38 + slmht 260 ~298. Rank pipe_host vs 298.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

namespace {

constexpr int kVl = 16;
constexpr int kWg = 16;
constexpr int kConv = 4;
constexpr int kCq = 2048;
constexpr int kCk = 2048;
constexpr int kCv = 6144;
constexpr int kCtot = kCq + kCk + kCv;
constexpr int kNk = 16;
constexpr int kNv = 48;
constexpr int kDv = 128;
constexpr int kDk = 128;
constexpr int kTiles = kDk / kVl;
constexpr int kBlk = 16;
constexpr int kOffK = 0;
constexpr int kOffQ = kBlk * kDk * int(sizeof(sycl::half));
constexpr int kSlmBytes = kOffQ + kBlk * kDk * int(sizeof(sycl::half));

struct GdnMixerSlmhtcName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_mixer_slmhtc: no GPU\n");
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
    return esimd::reduce<float>(v, std::plus<>{});
}

sycl::event launch(sycl::queue &q, const sycl::half *sd,
                   const sycl::half *xd, const sycl::half *wd,
                   const sycl::half *ad, const sycl::half *bd,
                   sycl::half *s2d, sycl::half *od, int tlen) {
    return q.parallel_for<GdnMixerSlmhtcName>(
        sycl::nd_range<1>({size_t(kNv * kDv)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            esimd::slm_init<kSlmBytes>();
            const int gid = int(it.get_global_id(0));
            const int lid = int(it.get_local_id(0));
            const int h = gid / kDv;
            const int i = gid % kDv;
            const int kh = h % kNk;
            const int qchan = kh * kDk;
            const int kchan = kCq + kh * kDk;
            const int vchan = kCq + kCk + h * kDv + i;
            const size_t srow =
                (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
            esimd::simd<float, kVl> sf[kTiles];
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                esimd::simd<sycl::half, kVl> sv;
                sv.copy_from(sd + srow + size_t(tile * kVl));
                sf[tile] = esimd::convert<float>(sv);
            }

            float vw[kConv];
#pragma unroll
            for (int tap = 0; tap < kConv; ++tap) {
                esimd::simd<sycl::half, 1> wv;
                wv.copy_from(wd + size_t(tap) * size_t(kCtot) +
                             size_t(vchan));
                vw[tap] = float(sycl::half(wv[0]));
            }

            const int nblk = tlen / kBlk;
            for (int b = 0; b < nblk; ++b) {
                const int t0 = b * kBlk;
                const int tl = t0 + lid;
                esimd::simd<sycl::half, kVl> qraw[kTiles];
                esimd::simd<sycl::half, kVl> kraw[kTiles];
                float qss = 0.f, kss = 0.f;
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    esimd::simd<float, kVl> qacc = 0.f;
                    esimd::simd<float, kVl> kacc = 0.f;
#pragma unroll
                    for (int tap = 0; tap < kConv; ++tap) {
                        const int xt = tl - (kConv - 1 - tap);
                        if (xt >= 0) {
                            esimd::simd<sycl::half, kVl> qx, kx, qw, kw;
                            qx.copy_from(xd + size_t(xt) * size_t(kCtot) +
                                         size_t(qchan + tile * kVl));
                            kx.copy_from(xd + size_t(xt) * size_t(kCtot) +
                                         size_t(kchan + tile * kVl));
                            qw.copy_from(wd + size_t(tap) * size_t(kCtot) +
                                         size_t(qchan + tile * kVl));
                            kw.copy_from(wd + size_t(tap) * size_t(kCtot) +
                                         size_t(kchan + tile * kVl));
                            qacc += esimd::convert<float>(qx) *
                                    esimd::convert<float>(qw);
                            kacc += esimd::convert<float>(kx) *
                                    esimd::convert<float>(kw);
                        }
                    }
#pragma unroll
                    for (int e = 0; e < kVl; ++e) {
                        qraw[tile][e] = sycl::half(float(qacc[e]));
                        kraw[tile][e] = sycl::half(float(kacc[e]));
                    }
                    const esimd::simd<float, kVl> qf =
                        esimd::convert<float>(qraw[tile]);
                    const esimd::simd<float, kVl> kf =
                        esimd::convert<float>(kraw[tile]);
                    qss += hsum16(qf * qf);
                    kss += hsum16(kf * kf);
                }
                const float qinv = qss > 0.f ? 1.f / std::sqrt(qss) : 0.f;
                const float kinv = kss > 0.f ? 1.f / std::sqrt(kss) : 0.f;
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const esimd::simd<float, kVl> qf =
                        esimd::convert<float>(qraw[tile]) * qinv;
                    const esimd::simd<float, kVl> kf =
                        esimd::convert<float>(kraw[tile]) * kinv;
                    esimd::simd<sycl::half, kVl> qn, kn;
#pragma unroll
                    for (int e = 0; e < kVl; ++e) {
                        qn[e] = sycl::half(float(qf[e]));
                        kn[e] = sycl::half(float(kf[e]));
                    }
                    esimd::slm_block_store<sycl::half, kVl>(
                        kOffK + (lid * kDk + tile * kVl) * int(sizeof(sycl::half)),
                        kn);
                    esimd::slm_block_store<sycl::half, kVl>(
                        kOffQ + (lid * kDk + tile * kVl) * int(sizeof(sycl::half)),
                        qn);
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int tt = 0; tt < kBlk; ++tt) {
                    const int t = t0 + tt;
                    const size_t abase = size_t(t) * size_t(kNv) + size_t(h);
                    esimd::simd<sycl::half, kVl> kv[kTiles];
                    esimd::simd<sycl::half, kVl> qv[kTiles];
                    esimd::simd<float, kVl> vacc = 0.f;
#pragma unroll
                    for (int tile = 0; tile < kTiles; ++tile) {
                        kv[tile] = esimd::slm_block_load<sycl::half, kVl>(
                            kOffK + (tt * kDk + tile * kVl) *
                                        int(sizeof(sycl::half)));
                        qv[tile] = esimd::slm_block_load<sycl::half, kVl>(
                            kOffQ + (tt * kDk + tile * kVl) *
                                        int(sizeof(sycl::half)));
                        vacc += sf[tile] * esimd::convert<float>(kv[tile]);
                    }
                    const float vold = hsum16(vacc);
                    esimd::simd<sycl::half, 1> ahv, bhv;
                    ahv.copy_from(ad + abase);
                    bhv.copy_from(bd + abase);
                    const float ah = float(sycl::half(ahv[0]));
                    const float bh = float(sycl::half(bhv[0]));
                    float vi = 0.f;
#pragma unroll
                    for (int tap = 0; tap < kConv; ++tap) {
                        const int xt = t - (kConv - 1 - tap);
                        if (xt >= 0) {
                            esimd::simd<sycl::half, 1> xv;
                            xv.copy_from(xd + size_t(xt) * size_t(kCtot) +
                                         size_t(vchan));
                            vi += float(sycl::half(xv[0])) * vw[tap];
                        }
                    }
                    esimd::simd<float, kVl> oaccv = 0.f;
#pragma unroll
                    for (int tile = 0; tile < kTiles; ++tile) {
                        const esimd::simd<float, kVl> kf =
                            esimd::convert<float>(kv[tile]);
                        const esimd::simd<float, kVl> snew =
                            ah * (sf[tile] - bh * vold * kf) + bh * vi * kf;
                        sf[tile] = snew;
                        oaccv += snew * esimd::convert<float>(qv[tile]);
                    }
                    const float oacc = hsum16(oaccv);
                    esimd::simd<sycl::half, 1> ov;
                    ov[0] = sycl::half(oacc);
                    ov.copy_to(od + (size_t(t) * size_t(kNv) + size_t(h)) *
                                        size_t(kDv) +
                               size_t(i));
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

float host_fir(const sycl::half *x, const sycl::half *w, int t, int c) {
    float acc = 0.f;
    for (int tap = 0; tap < kConv; ++tap) {
        const int xt = t - (kConv - 1 - tap);
        if (xt >= 0) {
            acc += float(x[size_t(xt) * size_t(kCtot) + size_t(c)]) *
                   float(w[size_t(tap) * size_t(kCtot) + size_t(c)]);
        }
    }
    return acc;
}

void l2_row(float *p, int dim) {
    float ss = 0.f;
    for (int j = 0; j < dim; ++j)
        ss += p[j] * p[j];
    const float inv = ss > 0.f ? 1.f / std::sqrt(ss) : 0.f;
    for (int j = 0; j < dim; ++j)
        p[j] = float(sycl::half(p[j] * inv));
}

void host_mixer_delta(const sycl::half *s, const sycl::half *x,
                      const sycl::half *w, const sycl::half *a,
                      const sycl::half *b, sycl::half *s2, sycl::half *o,
                      int tlen) {
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    std::vector<float> sf(ns);
    for (size_t n = 0; n < ns; ++n)
        sf[n] = float(s[n]);
    std::vector<float> qn(size_t(kDk), 0.f);
    std::vector<float> kn(size_t(kDk), 0.f);
    for (int t = 0; t < tlen; ++t) {
        for (int h = 0; h < kNv; ++h) {
            const int kh = h % kNk;
            const int qchan = kh * kDk;
            const int kchan = kCq + kh * kDk;
            for (int j = 0; j < kDk; ++j) {
                qn[size_t(j)] =
                    float(sycl::half(host_fir(x, w, t, qchan + j)));
                kn[size_t(j)] =
                    float(sycl::half(host_fir(x, w, t, kchan + j)));
            }
            l2_row(qn.data(), kDk);
            l2_row(kn.data(), kDk);
            const float ah = float(a[size_t(t) * size_t(kNv) + size_t(h)]);
            const float bh = float(b[size_t(t) * size_t(kNv) + size_t(h)]);
            for (int i = 0; i < kDv; ++i) {
                const size_t srow =
                    (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
                const int vchan = kCq + kCk + h * kDv + i;
                const float vi = host_fir(x, w, t, vchan);
                float vold = 0.f;
                for (int j = 0; j < kDk; ++j)
                    vold += sf[srow + size_t(j)] * kn[size_t(j)];
                float oacc = 0.f;
                for (int j = 0; j < kDk; ++j) {
                    const float kj = kn[size_t(j)];
                    const float snew =
                        ah * (sf[srow + size_t(j)] - bh * vold * kj) +
                        bh * vi * kj;
                    sf[srow + size_t(j)] = snew;
                    oacc += snew * qn[size_t(j)];
                }
                o[(size_t(t) * size_t(kNv) + size_t(h)) * size_t(kDv) +
                  size_t(i)] = sycl::half(oacc);
            }
        }
    }
    for (size_t n = 0; n < ns; ++n)
        s2[n] = sycl::half(sf[n]);
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
    if (tlen < kBlk || (tlen % kBlk) != 0) {
        std::fprintf(stderr, "gdn_mixer_slmhtc: T=%d need multiple of %d\n",
                     tlen, kBlk);
        *rc = 2;
        return;
    }
    const size_t nx = size_t(tlen) * size_t(kCtot);
    const size_t nw = size_t(kConv) * size_t(kCtot);
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    const size_t no = size_t(tlen) * size_t(kNv) * size_t(kDv);
    const size_t nsc = size_t(tlen) * size_t(kNv);
    std::vector<sycl::half> hx(nx), hw(nw), hs(ns), ha(nsc), hb(nsc);
    std::vector<sycl::half> hrefs(ns), hgots(ns), hrefo(no), hgoto(no);
    fill_f16(hx.data(), nx, 1);
    fill_f16(hw.data(), nw, 9);
    fill_f16(hs.data(), ns, 17);
    for (int t = 0; t < tlen; ++t) {
        for (int h = 0; h < kNv; ++h) {
            const size_t idx = size_t(t) * size_t(kNv) + size_t(h);
            ha[idx] = sycl::half(0.25f + float(h % 8) * 0.05f);
            hb[idx] = sycl::half(0.30f + float(h % 5) * 0.08f);
        }
    }
    host_mixer_delta(hs.data(), hx.data(), hw.data(), ha.data(), hb.data(),
                     hrefs.data(), hrefo.data(), tlen);

    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q);
    sycl::half *wd = sycl::malloc_device<sycl::half>(nw, q);
    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *ad = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nsc, q);
    sycl::half *s2d = sycl::malloc_device<sycl::half>(ns, q);
    sycl::half *od = sycl::malloc_device<sycl::half>(no, q);
    q.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q.memcpy(wd, hw.data(), nw * sizeof(sycl::half)).wait();
    q.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();
    q.memcpy(ad, ha.data(), nsc * sizeof(sycl::half)).wait();
    q.memcpy(bd, hb.data(), nsc * sizeof(sycl::half)).wait();

    auto go = [&]() -> sycl::event {
        return launch(q, sd, xd, wd, ad, bd, s2d, od, tlen);
    };

    go().wait_and_throw();
    q.memcpy(hgots.data(), s2d, ns * sizeof(sycl::half)).wait();
    q.memcpy(hgoto.data(), od, no * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_o = 0, mx_o = 0;
    score_f16(hgots.data(), hrefs.data(), ns, &cosine, &mx);
    score_f16(hgoto.data(), hrefo.data(), no, &cosine_o, &mx_o);
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
        ns_sum += t1 - t0;
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
    const double nbytes = double((nx + nw + ns + nsc + nsc + ns + no) *
                                 sizeof(sycl::half));
    const double gbs = (nbytes / 1.0e9) / (pipe_us * 1.0e-6);
    std::printf("phase,t,c,nv,dv,dk,event_us,wait_host_us,pipe_host_us,GBs,"
                "cosine,max_abs,cosine_o,max_abs_o,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.2f,%.6f,%.5g,%.6f,"
                "%.5g,%d,%.3f,%.3f,%.3f\n",
                phase, tlen, kCtot, kNv, kDv, kDk, us, wait_host, pipe_us, gbs,
                cosine, mx, cosine_o, mx_o, ok, median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(), all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(), all_us.end()));

    sycl::free(xd, q);
    sycl::free(wd, q);
    sycl::free(sd, q);
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
            std::fprintf(stderr, "gdn_mixer_slmhtc [--t 256] [--spin 0]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_mixer_slmhtc: unknown arg %s\n",
                         a.c_str());
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
                "op=gdn_mixer_slmhtc conv_reg+l2_reg+slmht dtype=f16 T=%d "
                "C=%d nv=%d dv=%d dk=%d VL=%d blk=%d warmup=%d iters=%d "
                "card=%d spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), timed_t, kCtot, kNv,
                kDv, kDk, kVl, kBlk, warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q, "timed", timed_t, warmup, iters, &rc, 1);
    return rc;
}

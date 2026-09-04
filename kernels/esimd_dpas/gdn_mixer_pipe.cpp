// K7 ESIMD GDN mixer T=256: T-chunk two-queue pipeline. Backend: sycl+l0.
// AOT intel_gpu_bmg_g31. Standalone icpx (intel/llvm#21741).
//
// Conv and slmht delta remain separate; a, b, and v remain unpacked.
// CONFIG prior: seq conv+slmht ~298, mixer-slmht 471, slmhtc 570 STOP.
// Steal T-chunk two-queue pipeline.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
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
constexpr int kT = 256;
constexpr int kNchunk = 4;
constexpr int kTchunk = 64;
constexpr int kOffK = 0;
constexpr int kOffQ = kBlk * kDk * int(sizeof(sycl::half));
constexpr int kSlmBytes = kOffQ + kBlk * kDk * int(sizeof(sycl::half));

struct GdnMixerPipeConvName {};
struct GdnMixerPipeDeltaName {};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "gdn_mixer_pipe: no GPU\n");
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

sycl::event launch_conv(sycl::queue &q, const sycl::half *xd,
                        const sycl::half *wd, sycl::half *yd, int t0,
                        int tlen, float *hdelay) {
    const int nwi = kCtot / kVl;
    return q.parallel_for<GdnMixerPipeConvName>(
        sycl::nd_range<1>({size_t(nwi)}, {size_t(kWg)}),
        [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            const int c0 = int(it.get_global_id(0)) * kVl;
            esimd::simd<sycl::half, kVl> wv[kConv];
#pragma unroll
            for (int k = 0; k < kConv; ++k)
                wv[k].copy_from(wd + k * kCtot + c0);
            esimd::simd<float, kVl> h0(0.f), h1(0.f), h2(0.f);
            if (t0 != 0) {
                h0.copy_from(hdelay + 0 * kCtot + c0);
                h1.copy_from(hdelay + 1 * kCtot + c0);
                h2.copy_from(hdelay + 2 * kCtot + c0);
            }
            for (int tt = 0; tt < tlen; ++tt) {
                const int t = t0 + tt;
                esimd::simd<sycl::half, kVl> xv;
                xv.copy_from(xd + size_t(t) * size_t(kCtot) + c0);
                const esimd::simd<float, kVl> xf = esimd::convert<float>(xv);
                const esimd::simd<float, kVl> acc =
                    h0 * esimd::convert<float>(wv[0]) +
                    h1 * esimd::convert<float>(wv[1]) +
                    h2 * esimd::convert<float>(wv[2]) +
                    xf * esimd::convert<float>(wv[3]);
                esimd::simd<sycl::half, kVl> yh;
#pragma unroll
                for (int e = 0; e < kVl; ++e)
                    yh[e] = sycl::half(float(acc[e]));
                yh.copy_to(yd + size_t(t) * size_t(kCtot) + c0);
                h0 = h1;
                h1 = h2;
                h2 = xf;
            }
            h0.copy_to(hdelay + 0 * kCtot + c0);
            h1.copy_to(hdelay + 1 * kCtot + c0);
            h2.copy_to(hdelay + 2 * kCtot + c0);
        });
}

sycl::event launch_delta(sycl::queue &q, const sycl::half *sd,
                         const sycl::half *yd, const sycl::half *ad,
                         const sycl::half *bd, sycl::half *s2d, sycl::half *od,
                         int t0, int tlen, const sycl::event &conv_ready) {
    return q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(conv_ready);
        cgh.parallel_for<GdnMixerPipeDeltaName>(
            sycl::nd_range<1>({size_t(kNv * kDv)}, {size_t(kWg)}),
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
            esimd::slm_init<kSlmBytes>();
            const int gid = int(it.get_global_id(0));
            const int lid = int(it.get_local_id(0));
            const int h = gid / kDv;
            const int i = gid % kDv;
            const int kh = h % kNk;
            const size_t srow =
                (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
            esimd::simd<float, kVl> sf[kTiles];
#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                esimd::simd<sycl::half, kVl> sv;
                sv.copy_from(sd + srow + size_t(tile * kVl));
                sf[tile] = esimd::convert<float>(sv);
            }
            const int nblk = tlen / kBlk;
            for (int b = 0; b < nblk; ++b) {
                const int bt0 = t0 + b * kBlk;
                const size_t ybase =
                    size_t(bt0 + lid) * size_t(kCtot);
                const size_t qbase = ybase + size_t(kh) * size_t(kDk);
                const size_t kbase =
                    ybase + size_t(kCq) + size_t(kh) * size_t(kDk);
                esimd::simd<sycl::half, kVl> qraw[kTiles];
                esimd::simd<sycl::half, kVl> kraw[kTiles];
                float qss = 0.f, kss = 0.f;
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    qraw[tile].copy_from(yd + qbase + size_t(tile * kVl));
                    kraw[tile].copy_from(yd + kbase + size_t(tile * kVl));
                    const esimd::simd<float, kVl> qf =
                        esimd::convert<float>(qraw[tile]);
                    const esimd::simd<float, kVl> kf =
                        esimd::convert<float>(kraw[tile]);
                    qss += hsum16(qf * qf);
                    kss += hsum16(kf * kf);
                }
                const float qinv = (qss > 0.f) ? 1.f / sycl::sqrt(qss) : 0.f;
                const float kinv = (kss > 0.f) ? 1.f / sycl::sqrt(kss) : 0.f;
#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    esimd::simd<sycl::half, kVl> qn, kn;
                    const esimd::simd<float, kVl> qf =
                        esimd::convert<float>(qraw[tile]) * qinv;
                    const esimd::simd<float, kVl> kf =
                        esimd::convert<float>(kraw[tile]) * kinv;
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
                    const int t = bt0 + tt;
                    const size_t vbase = size_t(t) * size_t(kCtot) +
                                         size_t(kCq + kCk) +
                                         size_t(h) * size_t(kDv);
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
                    esimd::simd<sycl::half, 1> ahv, bhv, viv;
                    ahv.copy_from(ad + abase);
                    bhv.copy_from(bd + abase);
                    viv.copy_from(yd + vbase + size_t(i));
                    const float ah = float(sycl::half(ahv[0]));
                    const float bh = float(sycl::half(bhv[0]));
                    const float vi = float(sycl::half(viv[0]));
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
    });
}

void host_conv(const sycl::half *x, const sycl::half *w, sycl::half *y,
               int tlen) {
    for (int i = 0; i < kCtot; ++i) {
        float h0 = 0.f, h1 = 0.f, h2 = 0.f;
        for (int t = 0; t < tlen; ++t) {
            const float xf = float(x[size_t(t) * size_t(kCtot) + size_t(i)]);
            const float acc = h0 * float(w[0 * kCtot + i]) +
                              h1 * float(w[1 * kCtot + i]) +
                              h2 * float(w[2 * kCtot + i]) +
                              xf * float(w[3 * kCtot + i]);
            y[size_t(t) * size_t(kCtot) + size_t(i)] = sycl::half(acc);
            h0 = h1;
            h1 = h2;
            h2 = xf;
        }
    }
}

void l2_row(float *p, int dim) {
    float ss = 0.f;
    for (int j = 0; j < dim; ++j)
        ss += p[j] * p[j];
    const float inv = (ss > 0.f) ? 1.f / std::sqrt(ss) : 0.f;
    for (int j = 0; j < dim; ++j)
        p[j] *= inv;
}

void host_delta(const sycl::half *s, const sycl::half *y, const sycl::half *a,
                const sycl::half *b, sycl::half *s2, sycl::half *o, int tlen) {
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    std::vector<float> sf(ns);
    for (size_t i = 0; i < ns; ++i)
        sf[i] = float(s[i]);
    std::vector<float> qn(size_t(kDk), 0.f);
    std::vector<float> kn(size_t(kDk), 0.f);
    for (int t = 0; t < tlen; ++t) {
        const size_t ybase = size_t(t) * size_t(kCtot);
        for (int h = 0; h < kNv; ++h) {
            const int kh = h % kNk;
            const float ah = float(a[size_t(t) * size_t(kNv) + size_t(h)]);
            const float bh = float(b[size_t(t) * size_t(kNv) + size_t(h)]);
            for (int j = 0; j < kDk; ++j) {
                qn[size_t(j)] = float(
                    y[ybase + size_t(kh) * size_t(kDk) + size_t(j)]);
                kn[size_t(j)] = float(y[ybase + size_t(kCq) +
                                        size_t(kh) * size_t(kDk) + size_t(j)]);
            }
            l2_row(qn.data(), kDk);
            l2_row(kn.data(), kDk);
            for (int i = 0; i < kDv; ++i) {
                const size_t srow =
                    (size_t(h) * size_t(kDv) + size_t(i)) * size_t(kDk);
                const float vi = float(y[ybase + size_t(kCq + kCk) +
                                         size_t(h) * size_t(kDv) + size_t(i)]);
                float vold = 0.f;
                for (int j = 0; j < kDk; ++j)
                    vold += sf[srow + size_t(j)] * kn[size_t(j)];
                float oacc = 0.f;
                for (int j = 0; j < kDk; ++j) {
                    const float snew =
                        ah * (sf[srow + size_t(j)] - bh * vold * kn[size_t(j)]) +
                        bh * vi * kn[size_t(j)];
                    sf[srow + size_t(j)] = snew;
                    oacc += snew * qn[size_t(j)];
                }
                o[(size_t(t) * size_t(kNv) + size_t(h)) * size_t(kDv) +
                  size_t(i)] = sycl::half(oacc);
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
        const float yv = float(ref[i]);
        const float d = x > yv ? x - yv : yv - x;
        if (d > m)
            m = d;
        dot += double(x) * double(yv);
        na2 += double(x) * double(x);
        nb2 += double(yv) * double(yv);
    }
    *cosine = (na2 > 0 && nb2 > 0) ? dot / std::sqrt(na2 * nb2) : 0.0;
    *mx = m;
}

void run_shape(sycl::queue &q_conv, sycl::queue &q_delta, const char *phase,
               int tlen, int warmup, int iters, int *rc) {
    if (tlen != kT || (kTchunk % kBlk) != 0 || kNchunk * kTchunk != kT) {
        std::fprintf(stderr, "gdn_mixer_pipe: T=%d need fixed T=%d\n", tlen,
                     kT);
        *rc = 2;
        return;
    }
    const size_t nx = size_t(tlen) * size_t(kCtot);
    const size_t nw = size_t(kConv) * size_t(kCtot);
    const size_t ns = size_t(kNv) * size_t(kDv) * size_t(kDk);
    const size_t no = size_t(tlen) * size_t(kNv) * size_t(kDv);
    const size_t nsc = size_t(tlen) * size_t(kNv);
    std::vector<sycl::half> hx(nx), hw(nw), hyref(nx);
    std::vector<sycl::half> hs(ns), hrefs(ns), hgots(ns), hrefo(no), hgoto(no);
    std::vector<sycl::half> ha(nsc), hb(nsc);
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
    host_conv(hx.data(), hw.data(), hyref.data(), tlen);
    host_delta(hs.data(), hyref.data(), ha.data(), hb.data(), hrefs.data(),
               hrefo.data(), tlen);

    sycl::half *xd = sycl::malloc_device<sycl::half>(nx, q_conv);
    sycl::half *wd = sycl::malloc_device<sycl::half>(nw, q_conv);
    sycl::half *yd = sycl::malloc_device<sycl::half>(nx, q_conv);
    sycl::half *sd = sycl::malloc_device<sycl::half>(ns, q_conv);
    sycl::half *ad = sycl::malloc_device<sycl::half>(nsc, q_conv);
    sycl::half *bd = sycl::malloc_device<sycl::half>(nsc, q_conv);
    sycl::half *s2d = sycl::malloc_device<sycl::half>(ns, q_conv);
    sycl::half *od = sycl::malloc_device<sycl::half>(no, q_conv);
    float *hdelay = sycl::malloc_device<float>(size_t(3 * kCtot), q_conv);
    q_conv.memcpy(xd, hx.data(), nx * sizeof(sycl::half)).wait();
    q_conv.memcpy(wd, hw.data(), nw * sizeof(sycl::half)).wait();
    q_conv.memcpy(sd, hs.data(), ns * sizeof(sycl::half)).wait();
    q_conv.memcpy(ad, ha.data(), nsc * sizeof(sycl::half)).wait();
    q_conv.memcpy(bd, hb.data(), nsc * sizeof(sycl::half)).wait();

    std::array<sycl::event, kNchunk> delta_events;
    auto go = [&]() -> sycl::event {
        std::array<sycl::event, kNchunk> conv_events;
        for (int chunk = 0; chunk < kNchunk; ++chunk) {
            const int t0 = chunk * kTchunk;
            conv_events[size_t(chunk)] =
                launch_conv(q_conv, xd, wd, yd, t0, kTchunk, hdelay);
            const sycl::half *chunk_sd = chunk == 0 ? sd : s2d;
            delta_events[size_t(chunk)] =
                launch_delta(q_delta, chunk_sd, yd, ad, bd, s2d, od, t0,
                             kTchunk, conv_events[size_t(chunk)]);
        }
        return delta_events.back();
    };

    go().wait_and_throw();
    q_conv.memcpy(hgots.data(), s2d, ns * sizeof(sycl::half)).wait();
    q_conv.memcpy(hgoto.data(), od, no * sizeof(sycl::half)).wait();
    double cosine = 0, mx = 0, cosine_o = 0, mx_o = 0;
    score_f16(hgots.data(), hrefs.data(), ns, &cosine, &mx);
    score_f16(hgoto.data(), hrefo.data(), no, &cosine_o, &mx_o);
    const int ok = (cosine > 0.99 && cosine_o > 0.99) ? 1 : 0;
    if (!ok)
        *rc = 1;

    auto run_and_wait = [&](int np) {
        for (int i = 0; i < np; ++i)
            go().wait_and_throw();
    };
    run_and_wait(warmup);
    if (g_spin > 0) {
        run_and_wait(g_spin);
        int act = -1, cur = -1, throttle = -1;
        char power[32];
        sample_gt(&act, &cur, power, sizeof(power), &throttle);
        std::printf("spin_done n=%d act=%d cur=%d power=%s throttle=%d\n",
                    g_spin, act, cur, power, throttle);
    }

    std::vector<double> all_pipe_us;
    uint64_t event_ns_sum = 0;
    double pipe_us_sum = 0.0;
    int act0 = -1, cur0 = -1, thr0 = -1;
    char power0[32];
    sample_gt(&act0, &cur0, power0, sizeof(power0), &thr0);
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0, cur0,
                power0, thr0);
    for (int i = 0; i < iters; ++i) {
        const auto pipe0 = std::chrono::steady_clock::now();
        sycl::event last = go();
        last.wait_and_throw();
        const auto pipe1 = std::chrono::steady_clock::now();
        const double pipe_us =
            std::chrono::duration<double, std::micro>(pipe1 - pipe0).count();
        pipe_us_sum += pipe_us;
        all_pipe_us.push_back(pipe_us);
        auto span = [](const sycl::event &e) {
            const uint64_t t0 =
                e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 =
                e.get_profiling_info<sycl::info::event_profiling::command_end>();
            return t1 - t0;
        };
        for (const sycl::event &e : delta_events)
            event_ns_sum += span(e);
    }
    int act1 = -1, cur1 = -1, thr1 = -1;
    char power1[32];
    sample_gt(&act1, &cur1, power1, sizeof(power1), &thr1);
    std::printf("timed_end act=%d cur=%d power=%s throttle=%d\n", act1, cur1,
                power1, thr1);
    const double event_us =
        (double(event_ns_sum) / 1000.0) / double(iters);
    const double pipe_host_us = pipe_us_sum / double(iters);
    std::printf("phase,t,c,nv,event_us,pipe_host_us,cosine,max_abs,cosine_o,"
                "max_abs_o,ok,median_us,min_us,max_us\n");
    std::printf("%s,%d,%d,%d,%.3f,%.3f,%.6f,%.5g,%.6f,%.5g,%d,%.3f,%.3f,"
                "%.3f\n",
                phase, tlen, kCtot, kNv, event_us, pipe_host_us, cosine, mx,
                cosine_o, mx_o, ok, median_of(all_pipe_us),
                all_pipe_us.empty()
                    ? -1.0
                    : *std::min_element(all_pipe_us.begin(), all_pipe_us.end()),
                all_pipe_us.empty()
                    ? -1.0
                    : *std::max_element(all_pipe_us.begin(), all_pipe_us.end()));

    sycl::free(xd, q_conv);
    sycl::free(wd, q_conv);
    sycl::free(yd, q_conv);
    sycl::free(sd, q_conv);
    sycl::free(ad, q_conv);
    sycl::free(bd, q_conv);
    sycl::free(s2d, q_conv);
    sycl::free(od, q_conv);
    sycl::free(hdelay, q_conv);
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
            std::fprintf(stderr, "gdn_mixer_pipe [--t 256] [--spin 0]\n");
            return 0;
        } else {
            std::fprintf(stderr, "gdn_mixer_pipe: unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    sycl::device dev = pick_device();
    sycl::context ctx(dev);
    sycl::queue q_conv(ctx, dev, {sycl::property::queue::in_order{},
                                  sycl::property::queue::enable_profiling{}});
    sycl::queue q_delta(ctx, dev, {sycl::property::queue::in_order{},
                                   sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char *backend =
        q_conv.get_backend() == sycl::backend::ext_oneapi_level_zero
            ? "sycl+l0"
            : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "op=gdn_mixer_pipe conv+slmht dtype=f16 T=%d C=%d nv=%d "
                "blk=%d nchunk=%d tchunk=%d warmup=%d iters=%d card=%d "
                "spin=%d heat=none\n",
                backend, name.c_str(), driver.c_str(), timed_t, kCtot, kNv,
                kBlk, kNchunk, kTchunk, warmup, iters, g_card, g_spin);
    int rc = 0;
    run_shape(q_conv, q_delta, "timed", timed_t, warmup, iters, &rc);
    return rc;
}

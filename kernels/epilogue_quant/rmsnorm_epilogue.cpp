// K5 RMSNorm then quant vs fused RMSNorm-epilogue s8.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31.
// Standalone icpx -fsycl (not a serving wrap).
//
// Contract: per-row RMSNorm, gamma=1, eps=1e-6, then symmetric s8
// with qmax=127 and per-row absmax scale. Residual stays f16 on
// the two-launch path; fused writes s8+scale only.
// Path A: 2 launches (rmsnorm f16, then quant).
// Path B: 1 launch (rmsnorm writes s8+scale).
// Rank us and launch count. TOPS is not the ranking key.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr float kEps = 1e-6f;
constexpr float kQmax = 127.f;

struct RmsName {};
struct QuantName {};
struct FusedName {};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "rmsnorm_epilogue: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

int8_t quant_one(float y, float scale) {
    float q = scale > 0.f ? y / scale : 0.f;
    q = std::nearbyintf(q);
    if (q > kQmax)
        q = kQmax;
    if (q < -kQmax)
        q = -kQmax;
    return int8_t(q);
}

void host_rmsnorm_quant(const float *x, int8_t *q, float *scale, int m, int k) {
    for (int i = 0; i < m; ++i) {
        const float *row = x + i * k;
        float ss = 0.f;
        for (int j = 0; j < k; ++j)
            ss += row[j] * row[j];
        const float inv = 1.f / std::sqrt(ss / float(k) + kEps);
        float amax = 0.f;
        for (int j = 0; j < k; ++j) {
            const float y = row[j] * inv;
            const float ay = y < 0.f ? -y : y;
            if (ay > amax)
                amax = ay;
        }
        float s = amax / kQmax;
        if (s <= 0.f)
            s = 1.f;
        scale[i] = s;
        for (int j = 0; j < k; ++j)
            q[i * k + j] = quant_one(row[j] * inv, s);
    }
}

sycl::event launch_rmsnorm(sycl::queue &q, const sycl::half *xin,
                           sycl::half *yout, int m, int k) {
    return q.parallel_for<RmsName>(sycl::range<1>(size_t(m)), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        const sycl::half *row = xin + i * k;
        float ss = 0.f;
        for (int j = 0; j < k; ++j) {
            const float v = float(row[j]);
            ss += v * v;
        }
        const float inv = sycl::rsqrt(ss / float(k) + kEps);
        sycl::half *out = yout + i * k;
        for (int j = 0; j < k; ++j)
            out[j] = sycl::half(float(row[j]) * inv);
    });
}

sycl::event launch_quant(sycl::queue &q, const sycl::half *yin, int8_t *qout,
                         float *sout, int m, int k) {
    return q.parallel_for<QuantName>(sycl::range<1>(size_t(m)), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        const sycl::half *row = yin + i * k;
        float amax = 0.f;
        for (int j = 0; j < k; ++j) {
            const float y = float(row[j]);
            const float ay = y < 0.f ? -y : y;
            if (ay > amax)
                amax = ay;
        }
        float s = amax / kQmax;
        if (s <= 0.f)
            s = 1.f;
        sout[i] = s;
        int8_t *qo = qout + i * k;
        for (int j = 0; j < k; ++j) {
            float v = float(row[j]) / s;
            v = sycl::rint(v);
            if (v > kQmax)
                v = kQmax;
            if (v < -kQmax)
                v = -kQmax;
            qo[j] = int8_t(v);
        }
    });
}

sycl::event launch_fused(sycl::queue &q, const sycl::half *xin, int8_t *qout,
                         float *sout, int m, int k) {
    return q.parallel_for<FusedName>(sycl::range<1>(size_t(m)), [=](sycl::id<1> id) {
        const int i = int(id[0]);
        const sycl::half *row = xin + i * k;
        float ss = 0.f;
        for (int j = 0; j < k; ++j) {
            const float v = float(row[j]);
            ss += v * v;
        }
        const float inv = sycl::rsqrt(ss / float(k) + kEps);
        float amax = 0.f;
        for (int j = 0; j < k; ++j) {
            const float y = float(row[j]) * inv;
            const float ay = y < 0.f ? -y : y;
            if (ay > amax)
                amax = ay;
        }
        float s = amax / kQmax;
        if (s <= 0.f)
            s = 1.f;
        sout[i] = s;
        int8_t *qo = qout + i * k;
        for (int j = 0; j < k; ++j) {
            float v = (float(row[j]) * inv) / s;
            v = sycl::rint(v);
            if (v > kQmax)
                v = kQmax;
            if (v < -kQmax)
                v = -kQmax;
            qo[j] = int8_t(v);
        }
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

int max_abs_s8(const int8_t *a, const int8_t *b, size_t n) {
    int mx = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = int(a[i]) - int(b[i]);
        const int ad = d < 0 ? -d : d;
        if (ad > mx)
            mx = ad;
    }
    return mx;
}

void run_shape(sycl::queue &q, int m, int k, int warmup, int iters, int *rc) {
    const size_t n = size_t(m) * size_t(k);
    std::vector<float> hx(n);
    for (size_t i = 0; i < n; ++i)
        hx[i] = float(int((i * 17u) % 200u) - 100) * 0.02f;
    std::vector<sycl::half> h16(n);
    for (size_t i = 0; i < n; ++i)
        h16[i] = sycl::half(hx[i]);
    std::vector<int8_t> href(n), hgot(n);
    std::vector<float> hs(static_cast<size_t>(m), 0.f);
    host_rmsnorm_quant(hx.data(), href.data(), hs.data(), m, k);

    sycl::half *xin = sycl::malloc_device<sycl::half>(n, q);
    sycl::half *y = sycl::malloc_device<sycl::half>(n, q);
    int8_t *qd = sycl::malloc_device<int8_t>(n, q);
    float *sd = sycl::malloc_device<float>(size_t(m), q);
    q.memcpy(xin, h16.data(), n * sizeof(sycl::half)).wait();

    launch_fused(q, xin, qd, sd, m, k).wait_and_throw();
    q.memcpy(hgot.data(), qd, n).wait();
    const int max_abs_f = max_abs_s8(hgot.data(), href.data(), n);
    const int ok_f = (max_abs_f <= 1) ? 1 : 0;
    if (!ok_f)
        *rc = 1;

    launch_rmsnorm(q, xin, y, m, k).wait_and_throw();
    launch_quant(q, y, qd, sd, m, k).wait_and_throw();
    q.memcpy(hgot.data(), qd, n).wait();
    const int max_abs_t = max_abs_s8(hgot.data(), href.data(), n);
    const int ok_t = (max_abs_t <= 1) ? 1 : 0;
    if (!ok_t)
        *rc = 1;

    for (int i = 0; i < warmup; ++i) {
        launch_rmsnorm(q, xin, y, m, k).wait_and_throw();
        launch_quant(q, y, qd, sd, m, k).wait_and_throw();
        launch_fused(q, xin, qd, sd, m, k).wait_and_throw();
    }
    double two_us = 0.0;
    for (int i = 0; i < iters; ++i) {
        two_us += event_us(launch_rmsnorm(q, xin, y, m, k));
        two_us += event_us(launch_quant(q, y, qd, sd, m, k));
    }
    two_us /= double(iters);
    double one_us = 0.0;
    for (int i = 0; i < iters; ++i)
        one_us += event_us(launch_fused(q, xin, qd, sd, m, k));
    one_us /= double(iters);

    std::printf("two_launch,%d,%d,2,%.3f,%d,%d\n", m, k, two_us, max_abs_t,
                ok_t);
    std::printf("fused,%d,%d,1,%.3f,%d,%d\n", m, k, one_us, max_abs_f, ok_f);

    sycl::free(xin, q);
    sycl::free(y, q);
    sycl::free(qd, q);
    sycl::free(sd, q);
}

} // namespace

int main(int argc, char **argv) {
    int warmup = 5, iters = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto take = [&](int &dst) {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        if (a == "--iters")
            take(iters);
        else if (a == "--warmup")
            take(warmup);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "rmsnorm_epilogue [--warmup W] [--iters N]\n");
            return 0;
        } else {
            std::fprintf(stderr, "rmsnorm_epilogue: unknown arg %s\n", a.c_str());
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
                "contract=rmsnorm_gamma1_eps1e-6_sym_s8_qmax127 warmup=%d "
                "iters=%d\n",
                backend, name.c_str(), driver.c_str(), warmup, iters);
    std::printf("arm,m,k,launches,us,max_abs,ok\n");
    int rc = 0;
    const int shapes[][2] = {{1, 5120}, {1, 17408}, {64, 5120}, {64, 17408}};
    for (const auto &s : shapes)
        run_shape(q, s[0], s[1], warmup, iters, &rc);
    return rc;
}

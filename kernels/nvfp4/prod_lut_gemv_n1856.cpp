// K8 Lightning expert-down GEMV hail-mary with a 16x16 E2M1 product LUT.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31.
// A and B remain E2M1 nibble codes. Never bitcast E2M1 to s4.

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

namespace {

constexpr int8_t kMag2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

int g_card = 0;
int g_spin = 0;
int g_mhz = 2400;

int8_t nibble_to_q(uint8_t nib) {
    int8_t q = kMag2[nib & 7];
    if (nib & 8)
        q = int8_t(-q);
    return q;
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

} // namespace

int main(int argc, char **argv) {
    int m = 1, n = 1856, k = 2688;
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
        if (a == "--m")
            take(m);
        else if (a == "--n")
            take(n);
        else if (a == "--k")
            take(k);
        else if (a == "--warmup")
            take(warmup);
        else if (a == "--iters")
            take(iters);
        else if (a == "--card")
            take(g_card);
        else if (a == "--spin")
            take(g_spin);
        else if (a == "--mhz")
            take(g_mhz);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "prod_lut_gemv_n1856 [--m 1] [--n 1856] [--k 2688] "
                         "[--spin 4000]\n");
            return 0;
        } else {
            std::fprintf(stderr, "prod_lut_gemv_n1856: unknown arg %s\n",
                         a.c_str());
            return 2;
        }
    }
    if (m != 1 || n < 1 || k < 1 || warmup < 0 || iters < 1) {
        std::fprintf(stderr,
                     "prod_lut_gemv_n1856: shape m=%d n=%d k=%d warmup=%d "
                     "iters=%d\n",
                     m, n, k, warmup, iters);
        return 2;
    }

    std::vector<uint8_t> ha(size_t(k), uint8_t(0));
    std::vector<uint8_t> hb(size_t(k) * size_t(n));
    int16_t table[16][16];
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            table[i][j] =
                int16_t(nibble_to_q(uint8_t(i)) * nibble_to_q(uint8_t(j)));
    for (int i = 0; i < k; ++i)
        ha[size_t(i)] = uint8_t((i * 17u) & 15u);
    for (size_t i = 0; i < hb.size(); ++i)
        hb[i] = uint8_t((i * 13u) & 15u);

    std::vector<int32_t> href(size_t(n), 0);
    for (int j = 0; j < n; ++j) {
        int32_t acc = 0;
        for (int kk = 0; kk < k; ++kk)
            acc += int32_t(
                table[ha[size_t(kk)]]
                     [hb[size_t(kk) * size_t(n) + size_t(j)]]);
        href[size_t(j)] = acc;
    }

    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "prod_lut_gemv_n1856: no GPU\n");
        return 2;
    }
    sycl::device dev = devs[0];
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver =
        dev.get_info<sycl::info::device::driver_version>();
    const char *backend =
        q.get_backend() == sycl::backend::ext_oneapi_level_zero
            ? "sycl+l0"
            : "sycl+other";
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s "
                "arm=prod_lut_gemv_n1856 W4A4 table16x16 never_bitcast_s4 "
                "m=1 n=%d k=%d warmup=%d iters=%d card=%d spin=%d mhz=%d "
                "heat=none\n",
                backend, name.c_str(), driver.c_str(), n, k, warmup, iters,
                g_card, g_spin, g_mhz);

    uint8_t *ad = sycl::malloc_device<uint8_t>(size_t(k), q);
    uint8_t *bd =
        sycl::malloc_device<uint8_t>(size_t(k) * size_t(n), q);
    int32_t *cd = sycl::malloc_device<int32_t>(size_t(n), q);
    int16_t *td = sycl::malloc_device<int16_t>(256, q);
    q.memcpy(ad, ha.data(), size_t(k)).wait();
    q.memcpy(bd, hb.data(), hb.size()).wait();
    q.memcpy(td, table, sizeof(table)).wait();

    auto go = [&]() -> sycl::event {
        return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int j = int(id[0]);
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                const uint8_t ai = ad[kk] & 15;
                const uint8_t bj =
                    bd[size_t(kk) * size_t(n) + size_t(j)] & 15;
                acc += int32_t(td[int(ai) * 16 + int(bj)]);
            }
            cd[j] = acc;
        });
    };

    go().wait_and_throw();
    std::vector<int32_t> hgot(size_t(n), int32_t(0));
    q.memcpy(hgot.data(), cd, size_t(n) * sizeof(int32_t)).wait();
    double dot = 0.0, na2 = 0.0, nb2 = 0.0;
    int mx = 0;
    for (int j = 0; j < n; ++j) {
        const int32_t x = hgot[size_t(j)];
        const int32_t y = href[size_t(j)];
        int d = x - y;
        if (d < 0)
            d = -d;
        if (d > mx)
            mx = d;
        dot += double(x) * double(y);
        na2 += double(x) * double(x);
        nb2 += double(y) * double(y);
    }
    const double cosine =
        (na2 > 0.0 && nb2 > 0.0) ? dot / std::sqrt(na2 * nb2) : 0.0;
    const int ok = cosine > 0.99 ? 1 : 0;

    auto batch_wait = [&](int count) {
        constexpr int kBatch = 256;
        for (int i = 0; i < count; ++i) {
            (void)go();
            if ((i % kBatch) == kBatch - 1)
                q.wait_and_throw();
        }
        q.wait_and_throw();
    };
    batch_wait(warmup);
    if (g_spin > 0) {
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
    std::printf("timed_begin act=%d cur=%d power=%s throttle=%d\n", act0,
                cur0, power0, thr0);
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
    const double pipe_host =
        std::chrono::duration<double, std::micro>(pipe1 - pipe0).count() /
        double(iters);
    int act1 = -1, cur1 = -1, thr1 = -1;
    char power1[32];
    sample_gt(&act1, &cur1, power1, sizeof(power1), &thr1);
    std::printf("timed_end act=%d cur=%d power=%s throttle=%d\n", act1, cur1,
                power1, thr1);

    const double event_us = double(ns_sum) / 1000.0 / double(iters);
    std::printf("phase,m,n,k,event_us,wait_host_us,pipe_host_us,cosine,max_abs,"
                "ok,median_us,min_us,max_us\n");
    std::printf("timed,1,%d,%d,%.3f,%.3f,%.3f,%.6f,%d,%d,%.3f,%.3f,%.3f\n",
                n, k, event_us, wait_host, pipe_host, cosine, mx, ok,
                median_of(all_us),
                all_us.empty() ? -1.0
                               : *std::min_element(all_us.begin(),
                                                   all_us.end()),
                all_us.empty() ? -1.0
                               : *std::max_element(all_us.begin(),
                                                   all_us.end()));

    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
    sycl::free(td, q);
    return ok ? 0 : 1;
}

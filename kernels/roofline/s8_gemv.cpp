// K0 M=1 s8 GEMV -- decode-shaped, sycl+l0, not DPAS.
// C[n] = sum_k A[k] * B[k, n], A is 1xK, B is KxN row-major.
// Prior: bandwidth / launch bound. TOPS will be a lie if XMX is idle.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "s8_gemv: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void host_gemv(const int8_t *a, const int8_t *b, int32_t *c, int k, int n) {
    for (int j = 0; j < n; ++j) {
        int32_t acc = 0;
        for (int i = 0; i < k; ++i)
            acc += int32_t(a[i]) * int32_t(b[i * n + j]);
        c[j] = acc;
    }
}

} // namespace

int main(int argc, char **argv) {
    int k = 5120;
    int n = 17408;
    int warmup = 10;
    int iters = 50;
    int do_check = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--k" && i + 1 < argc)
            k = std::atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc)
            n = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (a == "--no-check")
            do_check = 0;
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr, "s8_gemv [--k 5120] [--n 17408] [--iters N]\n");
            return 0;
        } else {
            std::fprintf(stderr, "s8_gemv: unknown arg %s\n", a.c_str());
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
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s m=1 k=%d n=%d "
                "note=GEMV_not_DPAS warmup=%d iters=%d\n",
                backend, name.c_str(), driver.c_str(), k, n, warmup, iters);
    std::printf("k,n,iters,us,GBs,TOPS,max_abs,ok\n");

    std::vector<int8_t> ha(static_cast<size_t>(k));
    std::vector<int8_t> hb(static_cast<size_t>(k) * static_cast<size_t>(n));
    for (int i = 0; i < k; ++i)
        ha[i] = int8_t((i * 17) % 255 - 128);
    for (size_t i = 0; i < hb.size(); ++i)
        hb[i] = int8_t((int(i * 29) % 255) - 128);

    int8_t *a = sycl::malloc_device<int8_t>(size_t(k), q);
    int8_t *b = sycl::malloc_device<int8_t>(size_t(k) * size_t(n), q);
    int32_t *c = sycl::malloc_device<int32_t>(size_t(n), q);
    q.memcpy(a, ha.data(), size_t(k)).wait();
    q.memcpy(b, hb.data(), hb.size()).wait();

    auto launch = [&]() -> sycl::event {
        return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int j = int(id[0]);
            int32_t acc = 0;
            for (int i = 0; i < k; ++i)
                acc += int32_t(a[i]) * int32_t(b[i * n + j]);
            c[j] = acc;
        });
    };

    int max_abs = 0;
    int ok = 1;
    if (do_check) {
        launch().wait_and_throw();
        std::vector<int32_t> hc(static_cast<size_t>(n));
        std::vector<int32_t> href(static_cast<size_t>(n));
        q.memcpy(hc.data(), c, size_t(n) * sizeof(int32_t)).wait();
        host_gemv(ha.data(), hb.data(), href.data(), k, n);
        for (int j = 0; j < n; ++j) {
            const int d = hc[j] > href[j] ? hc[j] - href[j] : href[j] - hc[j];
            if (d > max_abs)
                max_abs = d;
        }
        ok = (max_abs == 0) ? 1 : 0;
    }

    for (int i = 0; i < warmup; ++i)
        launch().wait_and_throw();
    uint64_t ns_sum = 0;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = launch();
        e.wait_and_throw();
        ns_sum += e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                  e.get_profiling_info<sycl::info::event_profiling::command_start>();
    }
    const double us = (double(ns_sum) / 1000.0) / double(iters);
    const double bytes = double(k) + double(k) * double(n) + double(n) * 4.0;
    const double gbs = (bytes / 1.0e9) / (us * 1.0e-6);
    const double ops = 2.0 * double(k) * double(n);
    const double tops = (ops / 1.0e12) / (us * 1.0e-6);
    std::printf("%d,%d,%d,%.3f,%.3f,%.6f,%d,%d\n", k, n, iters, us, gbs, tops,
                max_abs, ok);

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return ok ? 0 : 1;
}

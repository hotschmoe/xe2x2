// K0 s8 square GEMM -- boring SYCL work-group tile, sycl+l0, BMG-G31.
// Standalone: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31
//
// Prior (CONFIG, not RESULT): this kernel does not call DPAS / XMX.
// Datasheet 367 INT8 TOPS is an XMX number. Expect XVE-class TOPS.
// oneDNN / ESIMD DPAS are separate labeled arms (K1 / K2).

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "s8_square_gemm: no GPU\n");
        std::exit(2);
    }
    return devs[0];
}

void host_s8s8s32(const int8_t *a, const int8_t *b, int32_t *c, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < n; ++k)
                acc += int32_t(a[i * n + k]) * int32_t(b[k * n + j]);
            c[i * n + j] = acc;
        }
}

void launch_gemm(sycl::queue &q, const int8_t *a, const int8_t *b, int32_t *c,
                 int n) {
    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<int8_t, 2> as({kTile, kTile}, h);
        sycl::local_accessor<int8_t, 2> bs({kTile, kTile}, h);
        h.parallel_for(sycl::nd_range<2>({size_t(n), size_t(n)},
                                         {size_t(kTile), size_t(kTile)}),
                       [=](sycl::nd_item<2> it) {
                           const int gi = int(it.get_global_id(0));
                           const int gj = int(it.get_global_id(1));
                           const int li = int(it.get_local_id(0));
                           const int lj = int(it.get_local_id(1));
                           int32_t acc = 0;
                           const int tiles = n / kTile;
                           for (int t = 0; t < tiles; ++t) {
                               as[li][lj] = a[gi * n + (t * kTile + lj)];
                               bs[li][lj] = b[(t * kTile + li) * n + gj];
                               it.barrier(sycl::access::fence_space::local_space);
                               for (int k = 0; k < kTile; ++k)
                                   acc += int32_t(as[li][k]) * int32_t(bs[k][lj]);
                               it.barrier(sycl::access::fence_space::local_space);
                           }
                           c[gi * n + gj] = acc;
                       });
    });
}

} // namespace

int main(int argc, char **argv) {
    std::vector<int> ns = {256, 1024, 2048};
    int warmup = 3;
    int iters = 10;
    int check_n = 64;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n" && i + 1 < argc) {
            ns.clear();
            char *p = argv[++i];
            while (p && *p) {
                ns.push_back(int(std::strtol(p, &p, 10)));
                if (*p == ',')
                    ++p;
            }
        } else if (a == "--iters" && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (a == "--check-n" && i + 1 < argc)
            check_n = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "s8_square_gemm [--n 256,1024,2048] [--iters N] [--warmup W]\n");
            return 0;
        } else {
            std::fprintf(stderr, "s8_square_gemm: unknown arg %s\n", a.c_str());
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
    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s tile=%d "
                "note=XVE_not_DPAS warmup=%d iters=%d\n",
                backend, name.c_str(), driver.c_str(), kTile, warmup, iters);
    std::printf("n,iters,us,TOPS,pct_367,max_abs,ok\n");

    int rc = 0;
    auto run_n = [&](int n, bool do_check) {
        if (n % kTile != 0) {
            std::fprintf(stderr, "s8_square_gemm: n=%d not multiple of tile %d\n", n,
                         kTile);
            rc = 2;
            return;
        }
        const size_t nn = size_t(n) * size_t(n);
        std::vector<int8_t> ha(nn), hb(nn);
        std::vector<int32_t> href(do_check ? nn : 0);
        for (size_t i = 0; i < nn; ++i) {
            ha[i] = int8_t((int(i * 17) % 255) - 128);
            hb[i] = int8_t((int(i * 29) % 255) - 128);
        }
        int8_t *a = sycl::malloc_device<int8_t>(nn, q);
        int8_t *b = sycl::malloc_device<int8_t>(nn, q);
        int32_t *c = sycl::malloc_device<int32_t>(nn, q);
        q.memcpy(a, ha.data(), nn).wait();
        q.memcpy(b, hb.data(), nn).wait();

        int max_abs = 0;
        int ok = 1;
        if (do_check) {
            host_s8s8s32(ha.data(), hb.data(), href.data(), n);
            launch_gemm(q, a, b, c, n);
            q.wait_and_throw();
            std::vector<int32_t> hc(nn);
            q.memcpy(hc.data(), c, nn * sizeof(int32_t)).wait();
            for (size_t i = 0; i < nn; ++i) {
                const int d = hc[i] > href[i] ? hc[i] - href[i] : href[i] - hc[i];
                if (d > max_abs)
                    max_abs = d;
            }
            ok = (max_abs == 0) ? 1 : 0;
            if (!ok)
                rc = 1;
        }

        for (int i = 0; i < warmup; ++i) {
            launch_gemm(q, a, b, c, n);
            q.wait_and_throw();
        }
        uint64_t ns_sum = 0;
        for (int i = 0; i < iters; ++i) {
            sycl::event e = q.submit([&](sycl::handler &h) {
                // Re-submit via helper: profiling on the gemm event.
                sycl::local_accessor<int8_t, 2> as({kTile, kTile}, h);
                sycl::local_accessor<int8_t, 2> bs({kTile, kTile}, h);
                h.parallel_for(sycl::nd_range<2>({size_t(n), size_t(n)},
                                                 {size_t(kTile), size_t(kTile)}),
                               [=](sycl::nd_item<2> it) {
                                   const int gi = int(it.get_global_id(0));
                                   const int gj = int(it.get_global_id(1));
                                   const int li = int(it.get_local_id(0));
                                   const int lj = int(it.get_local_id(1));
                                   int32_t acc = 0;
                                   const int tiles = n / kTile;
                                   for (int t = 0; t < tiles; ++t) {
                                       as[li][lj] = a[gi * n + (t * kTile + lj)];
                                       bs[li][lj] = b[(t * kTile + li) * n + gj];
                                       it.barrier(sycl::access::fence_space::local_space);
                                       for (int k = 0; k < kTile; ++k)
                                           acc += int32_t(as[li][k]) *
                                                  int32_t(bs[k][lj]);
                                       it.barrier(sycl::access::fence_space::local_space);
                                   }
                                   c[gi * n + gj] = acc;
                               });
            });
            e.wait_and_throw();
            const uint64_t t0 =
                e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 =
                e.get_profiling_info<sycl::info::event_profiling::command_end>();
            ns_sum += (t1 - t0);
        }
        const double us = (double(ns_sum) / 1000.0) / double(iters);
        const double ops = 2.0 * double(n) * double(n) * double(n);
        const double tops = (ops / 1.0e12) / (us * 1.0e-6);
        const double pct = 100.0 * tops / 367.0;
        std::printf("%d,%d,%.3f,%.4f,%.4f,%d,%d\n", n, iters, us, tops, pct, max_abs,
                    ok);

        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
    };

    if (check_n > 0)
        run_n(check_n, true);
    for (int n : ns)
        run_n(n, false);
    return rc;
}

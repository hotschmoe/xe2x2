// K0 copy roof -- sycl+l0 USM H2D / D2H / D2D on BMG-G31.
// Standalone: icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31
// Pin with ZE_AFFINITY_MASK and gpu-run --card N.
//
// Prior (CONFIG, not RESULT): datasheet 608 GB/s HBM. This binary
// measures host-visible USM copies, not a tuned STREAM.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Run {
    const char *kind;
    size_t bytes;
    int iters;
    double us;
    double gbs;
};

sycl::device pick_device() {
    auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devs.empty()) {
        std::fprintf(stderr, "copy_roof: no GPU (ONEAPI_DEVICE_SELECTOR / ZE_AFFINITY_MASK)\n");
        std::exit(2);
    }
    return devs[0];
}

double ns_to_us(uint64_t ns) { return double(ns) / 1000.0; }

void fill_host(uint8_t *p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; ++i)
        p[i] = uint8_t(seed + i * 131u);
}

bool bytes_equal(const uint8_t *a, const uint8_t *b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}

} // namespace

int main(int argc, char **argv) {
    std::vector<size_t> sizes = {
        4096ull, 65536ull, 1048576ull, 16777216ull, 268435456ull};
    int warmup = 5;
    int iters = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc)
            iters = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (a == "--bytes" && i + 1 < argc) {
            sizes.clear();
            char *p = argv[++i];
            while (p && *p) {
                sizes.push_back(std::strtoull(p, &p, 10));
                if (*p == ',')
                    ++p;
            }
        } else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "copy_roof [--bytes 4096,65536,...] [--iters N] [--warmup W]\n");
            return 0;
        } else {
            std::fprintf(stderr, "copy_roof: unknown arg %s\n", a.c_str());
            return 2;
        }
    }

    sycl::device dev = pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const std::string name = dev.get_info<sycl::info::device::name>();
    const std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const std::string backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                                    ? "sycl+l0"
                                    : "sycl+other";

    std::printf("# CONFIG backend=%s device=\"%s\" driver=%s warmup=%d iters=%d\n",
                backend.c_str(), name.c_str(), driver.c_str(), warmup, iters);
    std::printf("kind,bytes,iters,us,GBs,ok\n");

    int rc = 0;
    for (size_t bytes : sizes) {
        uint8_t *h_src = sycl::malloc_host<uint8_t>(bytes, q);
        uint8_t *h_dst = sycl::malloc_host<uint8_t>(bytes, q);
        uint8_t *d_a = sycl::malloc_device<uint8_t>(bytes, q);
        uint8_t *d_b = sycl::malloc_device<uint8_t>(bytes, q);
        if (!h_src || !h_dst || !d_a || !d_b) {
            std::fprintf(stderr, "copy_roof: alloc failed bytes=%zu\n", bytes);
            return 1;
        }
        fill_host(h_src, bytes, 0xA5);
        std::memset(h_dst, 0, bytes);

        auto time_copy = [&](const char *kind, auto launch) -> Run {
            for (int i = 0; i < warmup; ++i) {
                launch();
                q.wait_and_throw();
            }
            uint64_t ns_sum = 0;
            for (int i = 0; i < iters; ++i) {
                sycl::event e = launch();
                e.wait_and_throw();
                const uint64_t t0 =
                    e.get_profiling_info<sycl::info::event_profiling::command_start>();
                const uint64_t t1 =
                    e.get_profiling_info<sycl::info::event_profiling::command_end>();
                ns_sum += (t1 - t0);
            }
            Run r;
            r.kind = kind;
            r.bytes = bytes;
            r.iters = iters;
            r.us = ns_to_us(ns_sum) / double(iters);
            r.gbs = (double(bytes) / 1.0e9) / (r.us * 1.0e-6);
            return r;
        };

        Run h2d = time_copy("h2d", [&] { return q.memcpy(d_a, h_src, bytes); });
        Run d2h = time_copy("d2h", [&] { return q.memcpy(h_dst, d_a, bytes); });
        // seed d_b from host once so D2D has known bytes
        q.memcpy(d_b, h_src, bytes).wait_and_throw();
        Run d2d = time_copy("d2d", [&] { return q.memcpy(d_a, d_b, bytes); });

        q.memcpy(h_dst, d_a, bytes).wait_and_throw();
        const int ok = bytes_equal(h_src, h_dst, bytes) ? 1 : 0;
        if (!ok)
            rc = 1;

        auto emit = [&](const Run &r) {
            std::printf("%s,%zu,%d,%.3f,%.3f,%d\n", r.kind, r.bytes, r.iters, r.us,
                        r.gbs, ok);
        };
        emit(h2d);
        emit(d2h);
        emit(d2d);

        sycl::free(h_src, q);
        sycl::free(h_dst, q);
        sycl::free(d_a, q);
        sycl::free(d_b, q);
    }
    return rc;
}

// K6 hail-mary: 16x16 E2M1 product LUT decode GEMV. W4A4-style. Never s4 bitcast.
// A and B are E2M1 nibbles. table[ai][bj] = q(ai)*q(bj). M=1.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;

constexpr int8_t kMag2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

int8_t nibble_to_q(uint8_t nib) {
    int8_t q = kMag2[nib & 7];
    if (nib & 8)
        q = int8_t(-q);
    return q;
}

int main() {
    const int n = 5120, k = 5120;
    std::vector<uint8_t> ha(k);
    std::vector<uint8_t> hb(k * n);
    int16_t table[16][16];
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            table[i][j] = int16_t(nibble_to_q(uint8_t(i)) * nibble_to_q(uint8_t(j)));
    for (int i = 0; i < k; ++i)
        ha[size_t(i)] = uint8_t((i * 17u) & 15u);
    for (size_t i = 0; i < hb.size(); ++i)
        hb[i] = uint8_t((i * 13u) & 15u);

    std::vector<int32_t> href(size_t(n), 0);
    for (int j = 0; j < n; ++j) {
        int32_t acc = 0;
        for (int kk = 0; kk < k; ++kk)
            acc += int32_t(table[ha[size_t(kk)]][hb[size_t(kk) * size_t(n) + size_t(j)]]);
        href[size_t(j)] = acc;
    }

    sycl::device dev = sycl::device::get_devices(sycl::info::device_type::gpu)[0];
    sycl::queue q(dev, {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s arm=prod_lut_gemv W4A4 table16x16 m=1 n=%d k=%d\n",
                backend, n, k);
    uint8_t *ad = sycl::malloc_device<uint8_t>(size_t(k), q);
    uint8_t *bd = sycl::malloc_device<uint8_t>(size_t(k) * size_t(n), q);
    int32_t *cd = sycl::malloc_device<int32_t>(size_t(n), q);
    int16_t *td = sycl::malloc_device<int16_t>(256, q);
    q.memcpy(ad, ha.data(), size_t(k)).wait();
    q.memcpy(bd, hb.data(), hb.size()).wait();
    q.memcpy(td, table, sizeof(table)).wait();

    auto go = [&]() {
        return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
            const int j = int(id[0]);
            int32_t acc = 0;
            for (int kk = 0; kk < k; ++kk) {
                const uint8_t ai = ad[kk] & 15;
                const uint8_t bj = bd[size_t(kk) * size_t(n) + size_t(j)] & 15;
                acc += int32_t(td[int(ai) * 16 + int(bj)]);
            }
            cd[j] = acc;
        });
    };
    go().wait_and_throw();
    std::vector<int32_t> hgot(n);
    q.memcpy(hgot.data(), cd, size_t(n) * 4).wait();
    int mx = 0;
    for (int j = 0; j < n; ++j) {
        int d = hgot[size_t(j)] - href[size_t(j)];
        if (d < 0)
            d = -d;
        if (d > mx)
            mx = d;
    }
    const int ok = mx == 0 ? 1 : 0;
    for (int i = 0; i < 5; ++i)
        go().wait_and_throw();
    uint64_t ns = 0;
    const int iters = 10;
    for (int i = 0; i < iters; ++i) {
        sycl::event e = go();
        e.wait_and_throw();
        ns += e.get_profiling_info<sycl::info::event_profiling::command_end>() -
              e.get_profiling_info<sycl::info::event_profiling::command_start>();
    }
    const double us = (double(ns) / 1000.0) / double(iters);
    std::printf("timed,1,%d,%d,%.3f,%d,%d\n", n, k, us, mx, ok);
    std::printf("DONE\n");
    sycl::free(ad, q);
    sycl::free(bd, q);
    sycl::free(cd, q);
    sycl::free(td, q);
    return ok ? 0 : 1;
}

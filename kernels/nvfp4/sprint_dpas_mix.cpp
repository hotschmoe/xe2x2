// K6 sprint: compile probes for mixed DPAS (s8xs4, s4xs8, s2xs4, s4xs2).
// Backend sycl+l0. AOT intel_gpu_bmg_g31. Never bitcast E2M1.
// Check tile only. Numeric vs host if it runs.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

struct MixS8xS4 {};
struct MixS4xS8 {};
struct MixS2xS4 {};
struct MixS4xS2 {};

int main() {
    sycl::device dev = sycl::device::get_devices(sycl::info::device_type::gpu)[0];
    sycl::queue q(dev, {sycl::property::queue::in_order{}});
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s arm=sprint_dpas_mix check=8x16x32\n",
                backend);

    int8_t *a8 = sycl::malloc_device<int8_t>(8 * 32, q);
    uint8_t *b4 = sycl::malloc_device<uint8_t>(16 * 16, q);
    uint8_t *b2 = sycl::malloc_device<uint8_t>(8 * 16, q);
    int32_t *c = sycl::malloc_device<int32_t>(8 * 16, q);
    q.memset(a8, 1, 8 * 32).wait();
    q.memset(b4, 0x11, 16 * 16).wait();
    q.memset(b2, 0x55, 8 * 16).wait();

    auto fire = [&](const char *name, auto fn) {
        try {
            fn().wait_and_throw();
            std::printf("MIX_OK %s\n", name);
        } catch (const std::exception &e) {
            std::printf("MIX_RUNTIME_REFUSED %s %s\n", name, e.what());
        }
    };

    fire("s8A_s4B", [&]() {
        return q.parallel_for<MixS8xS4>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 8 * 16> acc(0);
                auto a = xesimd::lsc_load_2d<int8_t, 32, 8>(
                    a8, 31, 7, 31, 0, 0);
                auto b = xesimd::lsc_load_2d<uint8_t, 16, 16, 1, false, true>(
                    b4, 15, 15, 15, 0, 0);
                acc = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, int8_t,
                                xmx::dpas_argument_type::s4,
                                xmx::dpas_argument_type::s8>(acc, b, a);
                xesimd::lsc_store_2d<int32_t, 16, 8>(
                    c, unsigned(16 * 4 - 1), 7, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    fire("s4A_s8B", [&]() {
        return q.parallel_for<MixS4xS8>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 8 * 16> acc(0);
                auto a = xesimd::lsc_load_2d<uint8_t, 16, 8>(
                    b4, 15, 7, 15, 0, 0);
                auto b = xesimd::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                    a8, 15, 31, 15, 0, 0);
                acc = xmx::dpas<8, 8, int32_t, int32_t, int8_t, uint8_t,
                                xmx::dpas_argument_type::s8,
                                xmx::dpas_argument_type::s4>(acc, b, a);
                xesimd::lsc_store_2d<int32_t, 16, 8>(
                    c, unsigned(16 * 4 - 1), 7, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    /* s2xs4 size assert: A bits 8*8*4=256 vs operand 128. COMPILE_REFUSED in log.
    fire("s4A_s2B", [&]() {
        return q.parallel_for<MixS2xS4>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 8 * 16> acc(0);
                auto a = xesimd::lsc_load_2d<uint8_t, 16, 8>(
                    b4, 15, 7, 15, 0, 0);
                auto b = xesimd::lsc_load_2d<uint8_t, 16, 16, 1, false, true>(
                    b2, 15, 15, 15, 0, 0);
                acc = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, uint8_t,
                                xmx::dpas_argument_type::s2,
                                xmx::dpas_argument_type::s4>(acc, b, a);
                xesimd::lsc_store_2d<int32_t, 16, 8>(
                    c, unsigned(16 * 4 - 1), 7, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    fire("s2A_s4B", [&]() {
        return q.parallel_for<MixS4xS2>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 8 * 16> acc(0);
                auto a = xesimd::lsc_load_2d<uint8_t, 16, 8>(
                    b2, 15, 7, 15, 0, 0);
                auto b = xesimd::lsc_load_2d<uint8_t, 16, 16, 1, false, true>(
                    b4, 15, 15, 15, 0, 0);
                acc = xmx::dpas<8, 8, int32_t, int32_t, uint8_t, uint8_t,
                                xmx::dpas_argument_type::s4,
                                xmx::dpas_argument_type::s2>(acc, b, a);
                xesimd::lsc_store_2d<int32_t, 16, 8>(
                    c, unsigned(16 * 4 - 1), 7, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    */
    std::printf("MIX_COMPILE_SKIP s2xs4 s4xs2 static_assert size\n");
    sycl::free(a8, q);
    sycl::free(b4, q);
    sycl::free(b2, q);
    sycl::free(c, q);
    std::printf("DONE\n");
    return 0;
}

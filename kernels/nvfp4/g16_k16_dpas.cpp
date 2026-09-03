// K6 sprint idea 8: can s8 DPAS isolate NVFP4 group-16 (K=16)?
// Backend sycl+l0. AOT intel_gpu_bmg_g31. Never bitcast E2M1.
// Landmine: serving tiles use dpas<8,4> s8 which is K=32 = two NVFP4 groups.
// Probe: dpas<4,4> s8 (SystolicDepth 4 -> K=16) vs dpas<8,4> s8 (K=32).

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace esimd = sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

struct DpasK16 {};
struct DpasK32 {};

int main() {
    sycl::device dev = sycl::device::get_devices(sycl::info::device_type::gpu)[0];
    sycl::queue q(dev, {sycl::property::queue::in_order{}});
    const char *backend = q.get_backend() == sycl::backend::ext_oneapi_level_zero
                              ? "sycl+l0"
                              : "sycl+other";
    std::printf("# CONFIG backend=%s arm=g16_k16_dpas s8_K16_vs_K32\n", backend);

    int8_t *a = sycl::malloc_device<int8_t>(8 * 32, q);
    int8_t *b = sycl::malloc_device<int8_t>(16 * 32, q);
    int32_t *c = sycl::malloc_device<int32_t>(8 * 16, q);
    q.memset(a, 1, 8 * 32).wait();
    q.memset(b, 1, 16 * 32).wait();

    auto fire = [&](const char *name, auto fn) {
        try {
            fn().wait_and_throw();
            std::printf("G16_OK %s\n", name);
        } catch (const std::exception &e) {
            std::printf("G16_RUNTIME_REFUSED %s %s\n", name, e.what());
        }
    };

    fire("s8_dpas8x4_K32", [&]() {
        return q.parallel_for<DpasK32>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 4 * 16> acc(0);
                auto av = xesimd::lsc_load_2d<int8_t, 32, 4>(a, 31, 3, 31, 0, 0);
                auto bv = xesimd::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                    b, 15, 31, 15, 0, 0);
                acc = xmx::dpas<8, 4, int32_t, int32_t, int8_t, int8_t,
                                xmx::dpas_argument_type::s8,
                                xmx::dpas_argument_type::s8>(acc, bv, av);
                xesimd::lsc_store_2d<int32_t, 16, 4>(
                    c, unsigned(16 * 4 - 1), 3, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    fire("s8_dpas4x4_K16", [&]() {
        return q.parallel_for<DpasK16>(sycl::nd_range<1>({16}, {16}),
            [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
                esimd::simd<int32_t, 4 * 16> acc(0);
                auto av = xesimd::lsc_load_2d<int8_t, 16, 4>(a, 15, 3, 15, 0, 0);
                auto bv = xesimd::lsc_load_2d<int8_t, 16, 16, 1, false, true>(
                    b, 15, 15, 15, 0, 0);
                acc = xmx::dpas<4, 4, int32_t, int32_t, int8_t, int8_t,
                                xmx::dpas_argument_type::s8,
                                xmx::dpas_argument_type::s8>(acc, bv, av);
                xesimd::lsc_store_2d<int32_t, 16, 4>(
                    c, unsigned(16 * 4 - 1), 3, unsigned(16 * 4 - 1), 0, 0,
                    acc);
            });
    });
    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    std::printf("DONE\n");
    return 0;
}

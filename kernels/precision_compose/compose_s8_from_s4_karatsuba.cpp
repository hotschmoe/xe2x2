// K3 Karatsuba s8-from-s4: three s4 DPAS skipped.
// Backend named for the campaign: sycl+l0. AOT intel_gpu_bmg_g31.
// Standalone icpx -fsycl (intel/llvm#21741).
//
// Algebra (three products):
//   p0 = a0*b0
//   p2 = a1*b1
//   ps = (a0+a1)*(b0+b1)
//   a*b = p0 + 16*(ps - p0 - p2) + 256*p2
//
// Digit split that covers full s8: a0 in u4 [0,15], a1 in s4 [-8,7].
// Then a0+a1 lives in [-8, 22]. That is not a subset of s4 [-8,7] or
// u4 [0,15]. Both-s4 digits ([-8,7]+[-8,7]) land in [-16,14], also not
// s4. Recoding the sum as a 4-bit residue plus a carry/overflow plane
// adds DPAS terms, so the kernel is no longer three s4 products.
// An s8 DPAS middle term is 2 s4 + 1 s8, which cannot beat native s8.
//
// CONFIG prior (NOT a RESULT): K2 s4 is 1.49x s8, so three s4 terms
// would cost ~2.0x native s8 IF the sums fit s4. They do not.
//
// This binary is a host algebra witness (no ESIMD kernel). CSV us=0
// means not timed on device. CSV: arm,m,n,k,terms,us,TOPS,max_abs,ok

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Host algebra only. Linked as a SYCL AOT binary so the compile line
// matches the other K3 TUs. No device kernel is emitted.

namespace {

int s8_lo_u4(int8_t a) { return int(uint8_t(a) & 0x0f); }

int s8_hi_s4(int8_t a) {
    const int h = int(uint8_t(a) >> 4);
    return h >= 8 ? h - 16 : h;
}

int in_s4(int v) { return v >= -8 && v <= 7; }
int in_u4(int v) { return v >= 0 && v <= 15; }

} // namespace

int main(int argc, char **argv) {
    int ncase = 65536;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n" && i + 1 < argc)
            ncase = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                         "compose_s8_from_s4_karatsuba [--n 65536]\n"
                         "Host algebra only. Three s4 DPAS refused.\n");
            return 0;
        } else {
            std::fprintf(stderr,
                         "compose_s8_from_s4_karatsuba: unknown arg %s\n",
                         a.c_str());
            return 2;
        }
    }

    std::printf(
        "# CONFIG backend=host_cpu target=emulate_s8 "
        "arm=karatsuba_s4 SKIP "
        "reason=\"(a0+a1) in [-8,22] not subset of s4[-8,7] or u4[0,15]\" "
        "prior_karatsuba_vs_s8=2.0x_from_k2_1.49x NOT_A_RESULT "
        "digits=u4_lo+s4_hi\n");
    std::printf("arm,m,n,k,terms,us,TOPS,max_abs,ok\n");

    int max_abs = 0;
    int sum_not_s4 = 0;
    int sum_not_u4 = 0;
    int sum_not_4b = 0;
    for (int i = 0; i < ncase; ++i) {
        const int8_t a = int8_t(int((unsigned(i) * 17u) % 256u) - 128);
        const int8_t b = int8_t(int((unsigned(i) * 29u + 9u) % 256u) - 128);
        const int a0 = s8_lo_u4(a);
        const int a1 = s8_hi_s4(a);
        const int b0 = s8_lo_u4(b);
        const int b1 = s8_hi_s4(b);
        const int sa = a0 + a1;
        const int sb = b0 + b1;
        if (!in_s4(sa) || !in_s4(sb))
            sum_not_s4 += 1;
        if (!in_u4(sa) || !in_u4(sb))
            sum_not_u4 += 1;
        if (!(in_s4(sa) || in_u4(sa)) || !(in_s4(sb) || in_u4(sb)))
            sum_not_4b += 1;
        const int p0 = a0 * b0;
        const int p2 = a1 * b1;
        const int ps = sa * sb;
        const int recon = p0 + 16 * (ps - p0 - p2) + 256 * p2;
        const int direct = int(a) * int(b);
        int d = recon > direct ? recon - direct : direct - recon;
        if (d > max_abs)
            max_abs = d;
    }
    const int ok = (max_abs == 0) ? 1 : 0;
    std::printf(
        "# karatsuba_host_identity n=%d max_abs=%d sum_not_s4=%d "
        "sum_not_u4=%d sum_not_s4_or_u4=%d\n",
        ncase, max_abs, sum_not_s4, sum_not_u4, sum_not_4b);
    // terms=3 is the algebraic count; us=0: no device kernel.
    std::printf("karatsuba_s4_skip,1,1,1,3,0.000,0.0000,%d,%d\n", max_abs, ok);
    return ok ? 0 : 1;
}

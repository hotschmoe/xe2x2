// K8 closed-form E2M1 nibble LUT, NT=2 unroll=14, inner_k=896.
// Backend: sycl+l0. AOT intel_gpu_bmg_g31. Never bitcast E2M1 to s4.
#define NIBBLE_LUT_SCF_PROGRAM "nibble_lut_scf_u14"
#define NIBBLE_LUT_SCF_ARM "e2m1_nibble_lut_scf_u14"
#define NIBBLE_LUT_SCF_NT2_UNROLL 14
#include "nibble_lut_scf.cpp"

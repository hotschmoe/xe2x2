# kernels

Device-side work for Xe2 / Arc Pro B70.

Put kernel sources, microbenchmarks, IGC dumps, and SYCL / Level Zero
harnesses here. One question per experiment directory. Name the
backend in the experiment README (`sycl+l0` default; OpenCL and
Vulkan are controls). AOT for these cards is `intel_gpu_bmg_g31`.
See ../docs/BACKENDS.md.

Questions this tree exists to answer:

- What actually runs well on Battlemage EU / XVE / XMX for this host.
- How IGC compiles the kernels we care about (GEMM, attention, epilogue,
  collect, copy).
- Single-card vs two-card occupancy, bandwidth, and cache behavior.
- Which kernel families are the bottleneck once TP=2 or PP=2 is attached.

Do not dump serving wrappers here. If a kernel only matters as part of a
parallel map, keep the kernel here and the parallel protocol under
`../parallel/`.

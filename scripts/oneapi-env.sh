#!/usr/bin/env bash
# Source from bash:  source /mnt/vm_8tb/github/xe2x2/scripts/oneapi-env.sh
# Host oneAPI pin for standalone icpx -fsycl (intel_gpu_bmg_g31).
# Relocated tree; do not assume /opt/intel/oneapi.

ONEAPI_ROOT="${ONEAPI_ROOT:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260901/oneapi-root/opt/intel/oneapi}"
# shellcheck disable=SC1091
source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZES_ENABLE_SYSMAN="${ZES_ENABLE_SYSMAN:-1}"
export CCL_TOPO_P2P_ACCESS="${CCL_TOPO_P2P_ACCESS:-0}"

# Live adapter on this host is Unified Runtime over Level-Zero V2
# (immediate lists only). Do not copy the flashnext
# SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMAND_LISTS=0 override unless that
# is the labeled control in CONFIG.

export CXX="${CXX:-icpx}"
export CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-icpx}"

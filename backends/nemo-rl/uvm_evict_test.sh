#!/bin/bash
# Builds and runs uvm_evict_test.cu on GPU 0, once per mode given as arguments
# (default: managed managed-forced managed-accessed-by device), each in its own
# process (a fault leaves the CUDA context unusable). Run on a compute node in
# the NeMo-RL container. Output: logs/uvm-evict-test/result-<time>.txt.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR=/iopsstor/scratch/cscs/dmelikidze/nemo-rl/logs/uvm-evict-test
mkdir -p "${OUT_DIR}"
BIN="${OUT_DIR}/uvm_evict_test"
LOG="${OUT_DIR}/result-$(date +%H%M%S).txt"

RUNS=(managed managed-forced managed-accessed-by device)
if [[ $# -gt 0 ]]; then
  RUNS=("$@")
fi

unset CUDA_ENABLE_COREDUMP_ON_EXCEPTION CUDA_MANAGED_FORCE_DEVICE_ALLOC

NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"

{
  date
  hostname
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
  "${NVCC}" --version | tail -2
  "${NVCC}" -O2 -arch=sm_90 -o "${BIN}" "${HERE}/uvm_evict_test.cu" || { echo "nvcc failed"; exit 1; }

  for run in "${RUNS[@]}"; do
    echo
    echo "===== ${run}"
    if [[ "${run}" == managed-forced ]]; then
      CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 timeout 300 "${BIN}" managed
    else
      timeout 300 "${BIN}" "${run}"
    fi
    echo "exit code $?"
  done
} 2>&1 | tee "${LOG}"

#!/usr/bin/env bash
# Run only through rl-infra's check_env_job.sh, inside its allocated container.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'A Slurm allocation is required.' >&2; exit 2; }
export NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_CUMEM_ENABLE=1 CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
endpoint="http://$(hostname):8080/v1"
echo "Judge ${SAFETY_JUDGE_MODEL}: ${endpoint} (job ${SLURM_JOB_ID})"
# Foreground service; Slurm owns its lifetime. No retry or background service.
# Match the currently working SML image startup; torchcodec is unused by text judges.
pip uninstall -y -q torchcodec
exec vllm serve \
  --model "${SAFETY_JUDGE_DIR}" --served-model-name "${SAFETY_JUDGE_MODEL}" \
  --host 0.0.0.0 --port 8080 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --tensor-parallel-size 1

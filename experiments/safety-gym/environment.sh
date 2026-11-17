#!/usr/bin/env bash
# Reuse the already staged, owned runtime; never run setup/fetch/patch on existing checkouts.
export SAFETY_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export APERTUS2_DIR="${SAFETY_EXPERIMENT_DIR}/../../backends/nemo-rl/apertus2"
export SITE=hangang USER="${USER:-hanint14}" SCRATCH="${SCRATCH:-/scratch/${USER:-hanint14}}"
export PROJECTS_DIR="${PROJECTS_DIR:-${SCRATCH}/safetygym/src}"
export SHARED="${SCRATCH}"
export OUT="${OUT:-${SCRATCH}/safetygym/runs}"
export SAFETY_GYM_COMMIT=6cdaeea52b25e2a48a2477ba2f8fffcf7113b2ee
export SAFETY_GYM_DIR="${SAFETY_GYM_DIR:-${SCRATCH}/safetygym/Gym-${SAFETY_GYM_COMMIT}-py312}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${HOME}/.cache/safetygym/gym-venvs/${SAFETY_GYM_COMMIT}}"
export JOB_UV_CACHE_DIR="${JOB_UV_CACHE_DIR:-${HOME}/.cache/safetygym/uv/${SAFETY_GYM_COMMIT}}"
export PYDEPS_GYM=1 PYDEPS="${PYDEPS:-${SCRATCH}/safetygym/pydeps-gym-${SAFETY_GYM_COMMIT}}"
export SAFETY_UV_BIN="${SAFETY_UV_BIN:-${SCRATCH}/safetygym/tools/uv-0.13.8/bin}"
export SAFETY_CHONK_MODEL="${SAFETY_CHONK_MODEL:-${SCRATCH}/models/chonk-lc-16k-v0.2-iter_0018315-ifgym}"
export SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SCRATCH}/datasets/msafetemp-safetygym}"
export SAFETY_VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-500}"
export SAFETY_DATA_DIR="${OUT}/data/validation-${SAFETY_VALIDATION_SAMPLES}"
export WANDB_PROJECT="${WANDB_PROJECT:-safety-gym}" WANDB_ENTITY="${WANDB_ENTITY:-nutemmax-epfl}"
export WANDB_KEY_FILE="${WANDB_KEY_FILE:-${OUT}/private/wandb.key}"
export SAFETY_ROLLOUT_SAVE_PERIOD=50 DRIVER_RETRIES=0
export JUDGE_KIND="${JUDGE_KIND:-oss}"
case "$JUDGE_KIND" in
  oss) judge_repo=openai/gpt-oss-safeguard-120b; judge_folder=gpt-oss-safeguard-120b ;;
  nemotron) judge_repo=nvidia/Nemotron-Content-Safety-Reasoning-4B; judge_folder=Nemotron-Content-Safety-Reasoning-4B ;;
  *) echo 'JUDGE_KIND must be oss or nemotron.' >&2; return 2 ;;
esac
export SAFETY_JUDGE_MODEL="${JUDGE_MODEL:-${USER}/${judge_repo}}"
export SAFETY_JUDGE_DIR="${JUDGE_MODEL_DIR:-${SCRATCH}/models/${judge_folder}}"
export SAFETY_JUDGE_BASE_URL="${JUDGE_URL:-}"
export SAFETY_CHECKPOINT_DEADLINE="${CHECKPOINT_DEADLINE:-00:01:40:00}"
export JOB_TIME="${JOB_TIME:-02:00:00}"
source "${SAFETY_EXPERIMENT_DIR}/../../sites/load.sh"
rl_infra_load "${APERTUS2_DIR}/env"
export NEMO_RL_DIR MEGATRON_DIR BRIDGE_DIR VLLM_RUNNER VLLM_APERTUS2 PIN_FILE
export PYDEPS_APERTUS2="${PYDEPS_APERTUS2:-}"
# HANGANG agent policy prohibits inspecting another user's files. Supply an owned
# image; the historical shared-user image is deliberately not tested or copied.
export IMAGE_WORK="${TRAIN_IMAGE:-${SCRATCH}/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh}"
export IMAGE_MASTER="${IMAGE_WORK}"
export APERTUS2_IMAGE_MASTER="${IMAGE_WORK}" APERTUS2_IMAGE_WORK="${IMAGE_WORK}"
export DRIVER_PYTHON="bash ${SAFETY_EXPERIMENT_DIR}/driver.sh"
export SCRIPT="${SAFETY_EXPERIMENT_DIR}/train.py"

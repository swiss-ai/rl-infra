#!/usr/bin/env bash
# Reuse the already staged, owned runtime; never run setup/fetch/patch on existing checkouts.
export SAFETY_EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAFETY_PROFILE="${SAFETY_PROFILE:-legacy}"
case "${SAFETY_PROFILE}" in
  legacy) ;;
  adaptive|single_turn_bridge|honeypot_single_turn|adaptive_honeypot)
    export SAFETY_ADAPTIVE_ROOT="${SAFETY_ADAPTIVE_ROOT:-/scratch/${USER}/safetygym-adaptive}"
    ADAPTIVE_GYM_COMMIT="${ADAPTIVE_GYM_COMMIT:-7017aab3e68300d4b177bc900ecd27bebc3ae2a8}"
    export SAFETY_GYM_COMMIT="${ADAPTIVE_GYM_COMMIT}"
    export SAFETY_GYM_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Gym-${SAFETY_GYM_COMMIT}-py312"
    export GYM_VENV_DIR="${SAFETY_ADAPTIVE_ROOT}/cache/gym-server-venvs-${SAFETY_GYM_COMMIT}"
    export SAFETY_GYM_SITE_PACKAGES="${SAFETY_ADAPTIVE_ROOT}/cache/gym-test-venv/lib/python3.12/site-packages"
    export JOB_UV_CACHE_DIR="${SAFETY_ADAPTIVE_ROOT}/cache/uv-${SAFETY_GYM_COMMIT}"
    export PYDEPS="${SAFETY_ADAPTIVE_ROOT}/pydeps/gym-${SAFETY_GYM_COMMIT}"
    export OUT="${OUT:-${SAFETY_ADAPTIVE_ROOT}/runs}"
    export SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SAFETY_ADAPTIVE_ROOT}/data/route-e1e8a681}"
    export SAFETY_ADAPTIVE_VALIDATION_FILE="${SAFETY_ADAPTIVE_VALIDATION_FILE:-${SAFETY_DATA_SOURCE}/adaptive_validation_500.jsonl}"
    export SAFETY_ADAPTIVE_GYM_CONFIG="${SAFETY_ADAPTIVE_GYM_CONFIG:-${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/configs/adaptive_safety.yaml}"
    export SAFETY_ADAPTIVE_LIMITS_CONFIG="${SAFETY_ADAPTIVE_LIMITS_CONFIG:-${SAFETY_EXPERIMENT_DIR}/adaptive_limits.yaml}"
  export SAFETY_SWISSAI_ENV_FILE="${SAFETY_SWISSAI_ENV_FILE:-/home01/${USER}/safety-gym/.env}"
  export SAFETY_SWISSAI_KEY_NAME="SWISSAI_RESEARCH_API_KEY"
  export SAFETY_SWISSAI_API_BASE="https://api.swissai.svc.cscs.ch/v1"
  export SAFETY_SWISSAI_ATTACKER_MODEL="SwissAI-Research/Qwen/Qwen3.8-27B"
  export SAFETY_SWISSAI_JUDGE_MODEL="RCP-AIaaS/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  export SAFETY_SWISSAI_VALIDATOR_MODEL="RCP-AIaaS/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  export SAFETY_SWISSAI_MODEL_REVISION="unknown"
  export SAFETY_ADAPTIVE_JUDGE_MODEL="${SAFETY_ADAPTIVE_JUDGE_MODEL:-${USER}/openai/gpt-oss-safeguard-120b}"
  export SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE="${SAFETY_ADAPTIVE_JUDGE_ENDPOINT_FILE:-${SAFETY_ADAPTIVE_ROOT}/runtime/judge-endpoint-${RUN:-unbound}.json}"
  export SAFETY_ADAPTIVE_TRAINING_JOB_FILE="${SAFETY_ADAPTIVE_TRAINING_JOB_FILE:-${SAFETY_ADAPTIVE_ROOT}/runtime/training-job-${RUN:-unbound}.json}"
  export SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS="${SAFETY_ADAPTIVE_JUDGE_WAIT_SECONDS:-1200}"
  export SAFETY_ADAPTIVE_JUDGE_BASE_URL="${JUDGE_URL:-http://127.0.0.1:8080/v1}"
  export SAFETY_SWISSAI_QUOTA_KEY="swissai:research-account"
    export SAFETY_ADAPTIVE_RUNTIME="${SAFETY_ADAPTIVE_RUNTIME:-${SAFETY_ADAPTIVE_ROOT}/src/nemo-rl-apertus2}"
    export NEMO_RL_DIR="${SAFETY_ADAPTIVE_RUNTIME}"
    export MEGATRON_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Megatron-LM-MoE"
    export BRIDGE_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Megatron-Bridge"
    export PYDEPS_APERTUS2="${SAFETY_ADAPTIVE_ROOT}/src/apertus-common"
    export SAFETY_UV_BIN="${SAFETY_ADAPTIVE_ROOT}/cache/gym-test-venv/bin"
    ;;
  *) echo "Unknown SAFETY_PROFILE: ${SAFETY_PROFILE}" >&2; return 2 ;;
esac
export APERTUS2_DIR="${SAFETY_EXPERIMENT_DIR}/../../backends/nemo-rl/apertus2"
export SITE=hangang USER="${USER:-$(id -un)}"
export SCRATCH="${SCRATCH:-/scratch/${USER}}"
export PROJECTS_DIR="${PROJECTS_DIR:-${SCRATCH}/safetygym/src}"
export SHARED="${SCRATCH}"
export OUT="${OUT:-${SCRATCH}/safetygym/runs}"
export SAFETY_GYM_COMMIT=6cdaeea52b25e2a48a2477ba2f8fffcf7113b2ee
export SAFETY_GYM_DIR="${SAFETY_GYM_DIR:-${SCRATCH}/safetygym/Gym-${SAFETY_GYM_COMMIT}-py312}"
export GYM_VENV_DIR="${GYM_VENV_DIR:-${HOME}/.cache/safetygym/gym-venvs/${SAFETY_GYM_COMMIT}}"
export JOB_UV_CACHE_DIR="${JOB_UV_CACHE_DIR:-${HOME}/.cache/safetygym/uv/${SAFETY_GYM_COMMIT}}"
export PYDEPS_GYM=1 PYDEPS="${PYDEPS:-${SCRATCH}/safetygym/pydeps-gym-${SAFETY_GYM_COMMIT}}"
export SAFETY_UV_BIN="${SAFETY_UV_BIN:-${SCRATCH}/safetygym/tools/uv-0.13.8/bin}"
export SAFETY_CHONK_SOURCE="${SAFETY_CHONK_SOURCE:-/appsdata/hanint05/hf-exports/bridge/chonk-lc-16k-v0.2-iter_0018315}"
export SAFETY_CHONK_MODEL="${SAFETY_CHONK_MODEL:-${SCRATCH}/safetygym/models/chonk-lc-16k-v0.2-iter_0018315-safetygym}"
export SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SCRATCH}/datasets/msafetemp-safetygym}"
export SAFETY_VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-500}"
export SAFETY_DATA_DIR="${OUT}/data/validation-${SAFETY_VALIDATION_SAMPLES}"
export WANDB_PROJECT="${WANDB_PROJECT:-safety-gym}" WANDB_ENTITY="${WANDB_ENTITY:-nutemmax-epfl}"
export WANDB_KEY_FILE="${WANDB_KEY_FILE:-${OUT}/private/wandb.key}"
export SAFETY_ROLLOUT_SAVE_PERIOD=50 DRIVER_RETRIES=0
if [[ "${SAFETY_PROFILE:-legacy}" != legacy ]]; then export SAFETY_ROLLOUT_SAVE_PERIOD=25; fi
export JUDGE_KIND="${JUDGE_KIND:-oss}"
if [[ "${SAFETY_PROFILE}" != legacy ]]; then
  export SAFETY_ADAPTIVE_ROOT="${SAFETY_ADAPTIVE_ROOT:-/scratch/${USER}/safetygym-adaptive}"
  export OUT="${SAFETY_ADAPTIVE_USER_OUT:-${SAFETY_ADAPTIVE_ROOT}/runs}"
  export SAFETY_GYM_COMMIT="${ADAPTIVE_GYM_COMMIT:?Set ADAPTIVE_GYM_COMMIT to the reviewed adaptive Gym commit}"
  export SAFETY_GYM_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Gym-${SAFETY_GYM_COMMIT}-py312"
  export GYM_VENV_DIR="${SAFETY_ADAPTIVE_ROOT}/cache/gym-server-venvs-${SAFETY_GYM_COMMIT}"
  export JOB_UV_CACHE_DIR="${SAFETY_ADAPTIVE_ROOT}/cache/uv-${SAFETY_GYM_COMMIT}"
  export PYDEPS="${SAFETY_ADAPTIVE_ROOT}/pydeps/gym-${SAFETY_GYM_COMMIT}"
  export SAFETY_DATA_SOURCE="${SAFETY_DATA_SOURCE:-${SAFETY_ADAPTIVE_ROOT}/data/route-e1e8a681}"
  export SAFETY_DATA_DIR="${SAFETY_DATA_SOURCE}"
  export SAFETY_ADAPTIVE_GYM_CONFIG="${SAFETY_ADAPTIVE_GYM_CONFIG:-${SAFETY_GYM_DIR}/responses_api_agents/adaptive_safety/configs/adaptive_safety.yaml}"
  export SAFETY_ADAPTIVE_RUNTIME="${SAFETY_ADAPTIVE_RUNTIME:-${SAFETY_ADAPTIVE_ROOT}/src/nemo-rl-apertus2}"
  export NEMO_RL_DIR="${SAFETY_ADAPTIVE_RUNTIME}"
  export MEGATRON_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Megatron-LM-MoE"
  export BRIDGE_DIR="${SAFETY_ADAPTIVE_ROOT}/src/Megatron-Bridge"
  export PYDEPS_APERTUS2="${SAFETY_ADAPTIVE_ROOT}/src/apertus-common"
fi
if [[ "${SAFETY_PROFILE}" != legacy ]]; then
  export PYDEPS_GYM=""
  gym_shared_venv="${SAFETY_ADAPTIVE_ROOT}/cache/gym-test-venv"
  [[ -x "${gym_shared_venv}/bin/python" && -r "${gym_shared_venv}/bin/activate" ]] || {
    echo "Feature Gym venv is incomplete: ${gym_shared_venv}" >&2; return 2; }
  for server_path in     responses_api_models/vllm_model     responses_api_models/openai_model     resources_servers/jailbreak_detection     resources_servers/over_refusal_detection     responses_api_agents/adaptive_safety; do
    target="${GYM_VENV_DIR}/${server_path}/.venv"
    mkdir -p "$(dirname "${target}")"
    if [[ -e "${target}" || -L "${target}" ]]; then
      [[ -L "${target}" && "$(readlink -f "${target}")" == "$(readlink -f "${gym_shared_venv}")" ]] || {
        echo "Unexpected Gym server venv; refusing to replace: ${target}" >&2; return 2; }
    else
      ln -s "${gym_shared_venv}" "${target}"
    fi
  done
fi

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

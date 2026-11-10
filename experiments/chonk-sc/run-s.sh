#!/bin/bash
# Run S: run B on the SFT checkpoint. Apertus 2 Chonk SFT (Rob's HF export of iter_0019374), 12 nodes
# (8 trainer Megatron PP2 EP16 + 4 vLLM TP1 EP4 FP8), Harbor micro-tasks through Gym's harbor_agent with
# Terminus-2 in the Apertus 2 message format, every task container a pod on OpenSandbox, token capture
# at the model endpoint, 8 prompts x 8 generations, 16384 tokens (6144 new per turn), 3 steps.
# See README.md. Run check.sh first (RUNS=s bash experiments/chonk-sc/check.sh).
#   bash experiments/chonk-sc/run-s.sh [overrides...]
# Before submitting, on the login node:
#   1. setup.sh with the apertus-common overlay (PYDEPS_APERTUS2, as ../../run.sh does for an SFT
#      recipe) into an EDF of its own (SFT_EDF_NAME, default nemo-rl-apertus2-sft). It clones
#      swiss-ai/apertus-common, a private repository, over SSH (APERTUS_COMMON_URL): your SSH key on
#      Clariden needs access to it.
#   2. stage_model.sh: SFT_MODEL_SRC -> $SCRATCH/models/chonk-lc-16k-v0.1-ep1-iter_0019374 (shards
#      linked, small files copied, modeling code from hfconverter added).
#   3. the Gym checkout and NeMo-RL clone checks of run B.
# Then ../../run-harbor-osb.sh with RECIPE ../../recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml, TASK
# a copy of ../../envs/harbor-osb-sft.yaml without its defaults/script lines, OVERLAY run-b-harbor-sc.yaml
# run-s-sft.yaml, SCRIPT=examples/run_grpo_single_controller.py.
# AGENT=opencode: OpenCode in each pod instead of Terminus-2, calling the model through the per-pod
# tunnel (../harbor-capture-sc/opencode-sc.yaml + tunnel-sc.yaml, as R2 job 5314745 on Qwen) plus
# run-s-opencode.yaml (Gym model-call capture of every request). Smoke (16 samples: the trainer's
# data-parallel size is 16):
#   AGENT=opencode bash experiments/chonk-sc/run-s.sh grpo.num_prompts_per_step=2 \
#       grpo.num_generations_per_prompt=8 grpo.max_num_steps=1 policy.train_global_batch_size=16
#   OPENCODE_VERSION, OPENCODE_TIMEOUT_SEC, OPENCODE_DEBUG, CHISEL_DIR as ../harbor-capture-sc/run.sh.
# Env: SFT_MODEL_SRC, SFT_EDF_NAME, ACCOUNT, PARTITION (default normal), JOB_TIME (default 03:00:00), RUN,
# HARBOR_ROOT (default $SCRATCH/osb/harbor_micro), OPENSANDBOX_API_KEY_FILE (default ~/.opensandbox_token),
# OVERLAY (extra overlays, merged after ours), DRIVER_RETRIES (default 0), everything else
# run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${HERE}/common.sh"
AGENT="${AGENT:-terminus}"
case "${AGENT}" in
    terminus) export RUN="${RUN:-chonk-sc-s-harbor-sft-$(date +%Y%m%d-%H%M%S)}" ;;
    opencode) export RUN="${RUN:-chonk-sc-s-opencode-$(date +%Y%m%d-%H%M%S)}" ;;
    *) echo "ERROR: AGENT must be terminus or opencode, not ${AGENT}" >&2; exit 1 ;;
esac

[[ -r "${SFT_MODEL_SRC}/config.json" && -r "${SFT_MODEL_SRC}/model.safetensors.index.json" ]] \
    || { echo "ERROR: ${SFT_MODEL_SRC} has no readable config.json + model.safetensors.index.json (an HF export)" >&2; exit 1; }
OPENSANDBOX_API_KEY_FILE="$(realpath "${OPENSANDBOX_API_KEY_FILE:-${HOME}/.opensandbox_token}")"
[[ -r "${OPENSANDBOX_API_KEY_FILE}" ]] || { echo "ERROR: no OpenSandbox key file ${OPENSANDBOX_API_KEY_FILE}" >&2; exit 1; }
export OPENSANDBOX_API_KEY_FILE

# 1. setup.sh as ../../run.sh runs it for a recipe with sft_export, but with an EDF of its own; exported,
#    so ../../run-harbor-osb.sh and launch.sh submit with the same EDF and overlay.
# apertus-common is private: clone it over SSH (APERTUS_COMMON_URL overrides; the pin's default is HTTPS)
export APERTUS_COMMON_URL="${APERTUS_COMMON_URL:-git@github.com:swiss-ai/apertus-common.git}"
source "${PIN_FILE}"
export EDF_NAME="${SFT_EDF_NAME}"
export PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
export PYDEPS="${PYDEPS}-apertus2"
mkdir -p "${OUT}/configs"
LOG="${OUT}/setup-${RUN}.log"
echo "== setup for run S (EDF ${EDF_NAME}, overlay ${PYDEPS}; log ${LOG})"
# 2. the model, as ../../run.sh stages an sft_export
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" \
    && SRC="${SFT_MODEL_SRC}" DST="${SFT_MODEL_DIR}" CHONK_MODEL_STAGE=copy STAGE_SHARDS=link \
       bash "${ROOT}/experiments/grpo-chonk/stage_model.sh"; } > "${LOG}" 2>&1 \
    || { tail -n 20 "${LOG}"; echo "ERROR: setup or model staging failed, see ${LOG}" >&2; exit 1; }
grep -E '^== (EDF|done|image)|^WARNING' "${LOG}" || true

# 3. as run-b.sh: fail before the queue wait
if ! out="$(check_gym_capture)"; then
    grep -v '^  ok' <<< "${out}" >&2
    echo "ERROR: the Gym checkout lacks the capture patches (bash ${APERTUS2_DIR}/setup-gym.sh)" >&2
    exit 1
fi
if ! out="$(check_nemo_rl_clone)"; then
    grep -v '^  ok' <<< "${out}" >&2
    echo "ERROR: the NeMo-RL clone ${NEMO_RL_DIR} is not ready (bash ${HERE}/check.sh)" >&2
    exit 1
fi

AGENT_OVERLAY=""
if [[ "${AGENT}" == opencode ]]; then
    # OpenCode in the pod, as ../harbor-capture-sc/run.sh AGENT=opencode (R2 job 5314745), on Chonk
    GYM_DIR="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym"
    grep -q opencode_workdir "${GYM_DIR}/responses_api_agents/harbor_agent/custom_agents/opencode_nemo_gym.py" 2>/dev/null \
        && grep -q server_start_script "${GYM_DIR}/responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/model_tunnel.py" 2>/dev/null \
        || { echo "ERROR: AGENT=opencode needs Gym patches 0013-0014 (tunnel) and 0023-0026 (OpenCode) in ${GYM_DIR} (bash ${APERTUS2_DIR}/setup-gym.sh)" >&2; exit 1; }
    for v in OPENCODE_VERSION OPENCODE_TIMEOUT_SEC OPENCODE_DEBUG; do
        [[ -z "${!v:-}" ]] || export "${v}"
    done
    export CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}" CHISEL_DIR="${CHISEL_DIR:-${SCRATCH}/chisel}"
    bash "${APERTUS2_DIR}/fetch-chisel.sh"
    CAPTURE_SC="${ROOT}/experiments/harbor-capture-sc"
    AGENT_OVERLAY=" ${CAPTURE_SC}/opencode-sc.yaml ${CAPTURE_SC}/tunnel-sc.yaml ${HERE}/run-s-opencode.yaml"
    export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-opencode-sft-sc}"
fi

TASK_COPY="${OUT}/configs/${RUN}-task.yaml"
sft_task_copy "${TASK_COPY}"
export SCRIPT="${SC_SCRIPT}" RECIPE="${SFT_RECIPE}" TASK="${TASK_COPY}"
export OVERLAY="${HERE}/run-b-harbor-sc.yaml ${HERE}/run-s-sft.yaml${AGENT_OVERLAY}${OVERLAY:+ ${OVERLAY}}"
export PARTITION="${PARTITION:-normal}" DRIVER_RETRIES="${DRIVER_RETRIES:-0}"
# ~10-15 min start-up (16k layout, Megatron import of the HF export) + 3 steps; worst case
# 3 x rollout_timeout_s (50 min)
export JOB_TIME="${JOB_TIME:-03:00:00}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-harbor-sft-sc}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

#!/bin/bash
# Run A: Apertus 2 Chonk on GSM8K through NeMo-RL's Single-Controller path, 12 nodes (8 trainer
# EP32 + 4 vLLM EP4), 32 prompts x 16 generations, 3 steps, no Gym / OpenSandbox / token capture.
# See README.md. Run check.sh first.
#   bash experiments/chonk-sc/run-a.sh [overrides...]
# Goes through the normal chain: ../../run.sh (setup.sh + model staging, then the recipe
# ../../recipes/apertus2-chonk-vllm-adamw.yaml + task run-a-gsm8k.yaml) -> apertus2/launch.sh ->
# backends/nemo-rl/launch.sh (preflight, sbatch ray.sub) with SCRIPT=examples/run_grpo_single_controller.py.
# Env: ACCOUNT (run.sh default infra01; Daint: csstaff), PARTITION (default normal), JOB_TIME
# (default 01:30:00), RUN, SCRATCH (Daint: export SCRATCH=/iopsstor/scratch/cscs/$USER first),
# DRIVER_RETRIES (default 0: SC writes no train_data_step*.jsonl, so driver_retry.sh would not
# retry anyway).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${HERE}/common.sh"

# Fail before the queue wait if the clone lacks #2517 or lost f196's refit code (check.sh (b)).
if ! out="$(check_nemo_rl_clone)"; then
    grep -v '^  ok' <<< "${out}" >&2
    echo "ERROR: the NeMo-RL clone ${NEMO_RL_DIR} is not ready for run A (bash ${HERE}/check.sh)" >&2
    exit 1
fi

export SCRIPT="${SC_SCRIPT}" RECIPE="${CHONK_RECIPE}"
export PARTITION="${PARTITION:-normal}" JOB_TIME="${JOB_TIME:-01:30:00}" DRIVER_RETRIES="${DRIVER_RETRIES:-0}"
export RUN="${RUN:-chonk-sc-a-gsm8k-$(date +%Y%m%d-%H%M%S)}"
exec bash "${ROOT}/run.sh" "${HERE}/run-a-gsm8k.yaml" "$@"

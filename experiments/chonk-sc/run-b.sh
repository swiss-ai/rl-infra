#!/bin/bash
# Run B: the Single-Controller version of Daint job 5184052. Apertus 2 Chonk, 12 nodes (8 trainer
# EP32 + 4 vLLM EP4), Harbor micro-tasks through Gym's harbor_agent with Terminus-2, every task
# container a pod on OpenSandbox, token capture at the model endpoint, 8 prompts x 8 generations,
# 4096 tokens (1024 new per turn), 3 steps. See README.md. Run check.sh first.
#   bash experiments/chonk-sc/run-b.sh [overrides...]
# Goes through ../../run-harbor-osb.sh (OpenSandbox key check, Harbor paths, sandbox reaper) with
# RECIPE ../../recipes/apertus2-chonk-vllm-adamw.yaml, TASK ../../envs/harbor-osb.yaml and
# OVERLAY run-b-harbor-sc.yaml, SCRIPT=examples/run_grpo_single_controller.py.
# Env: ACCOUNT (default infra01; Daint: csstaff), PARTITION (default normal), JOB_TIME (default
# 02:30:00), RUN, HARBOR_ROOT (default $SCRATCH/osb/harbor_micro), SCRATCH (Daint: export
# SCRATCH=/iopsstor/scratch/cscs/$USER first), OVERLAY (extra overlays, merged after ours),
# DRIVER_RETRIES (default 0), everything else run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${HERE}/common.sh"

# Fail before the queue wait: the capture patches in the Gym checkout (as ../harbor-capture-sc/run.sh)
# and #2517 / f196 refit code in the clone (check.sh (b) and (c)).
if ! out="$(check_gym_capture)"; then
    grep -v '^  ok' <<< "${out}" >&2
    echo "ERROR: the Gym checkout lacks the capture patches; without 0010 capture records nothing and every row is rejected" >&2
    exit 1
fi
if ! out="$(check_nemo_rl_clone)"; then
    grep -v '^  ok' <<< "${out}" >&2
    echo "ERROR: the NeMo-RL clone ${NEMO_RL_DIR} is not ready for run B (bash ${HERE}/check.sh)" >&2
    exit 1
fi

export SCRIPT="${SC_SCRIPT}" RECIPE="${CHONK_RECIPE}" TASK="${ROOT}/envs/harbor-osb.yaml"
export OVERLAY="${HERE}/run-b-harbor-sc.yaml${OVERLAY:+ ${OVERLAY}}"
export PARTITION="${PARTITION:-normal}" DRIVER_RETRIES="${DRIVER_RETRIES:-0}"
# ~10 min start-up (5184052: 9.5) + 3 steps; worst case 3 x rollout_timeout_s (30 min)
export JOB_TIME="${JOB_TIME:-02:30:00}"
export RUN="${RUN:-chonk-sc-b-harbor-$(date +%Y%m%d-%H%M%S)}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-harbor-sc}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

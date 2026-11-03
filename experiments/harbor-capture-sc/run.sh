#!/bin/bash
# Harbor micro-tasks through Gym's harbor_agent with NeMo-RL's Single-Controller path and token
# capture at the model endpoint, Qwen3-4B on one node. See README.md.
#   bash experiments/harbor-capture-sc/run.sh [overrides...]
# Env:
#   AGENT                  terminus (default) or opencode
#   HARBOR_MODEL_BASE_URL  opencode only: model-server URL as reachable from the pods (required)
#   OPENCODE_VERSION       opencode only: opencode-ai npm version (default: latest)
#   TUNNEL                 1: per-sandbox reverse tunnel sandbox 127.0.0.1:8000 -> policy_model (Gym patch
#                          0003, tunnel-sc.yaml); every trial checks GET /health through it before starting.
#                          With terminus the agent still calls the model from the node (step A: the tunnel
#                          runs alongside). CHISEL_DIR (default $SCRATCH/chisel) is filled on first use.
#   HARBOR_ROOT            micro-task tree, train.jsonl, validation.jsonl (default $SCRATCH/osb/harbor_micro)
#   RUN, PARTITION, JOB_TIME (default 02:00:00), everything run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"

GYM_DIR="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym"
grep -q harbor_model_base_url "${GYM_DIR}/responses_api_agents/harbor_agent/app.py" 2>/dev/null || {
    echo "ERROR: the Gym checkout in ${GYM_DIR} lacks the capture patches; run" >&2
    echo "  GYM_BRANCH=apertus2-osb-capture GYM_PATCHES=\"$(ls "${APERTUS2_DIR}"/patches/gym/*.patch | tr '\n' ' ')\" \\" >&2
    echo "    bash ${APERTUS2_DIR}/setup-gym.sh" >&2
    exit 1
}

AGENT="${AGENT:-terminus}"
export OVERLAY=""
case "${AGENT}" in
    terminus) ;;
    opencode)
        [[ -n "${HARBOR_MODEL_BASE_URL:-}" ]] || {
            echo "ERROR: AGENT=opencode needs HARBOR_MODEL_BASE_URL (model server as reachable from the pods)" >&2
            exit 1
        }
        export HARBOR_MODEL_BASE_URL
        export OVERLAY="${HERE}/opencode-sc.yaml"
        ;;
    *) echo "ERROR: AGENT must be terminus or opencode, not ${AGENT}" >&2; exit 1 ;;
esac
[[ -z "${OPENCODE_VERSION:-}" ]] || export OPENCODE_VERSION

if [[ "${TUNNEL:-0}" == 1 ]]; then
    [[ -f "${GYM_DIR}/responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/model_tunnel.py" ]] || {
        echo "ERROR: TUNNEL=1 needs Gym patch 0003 in ${GYM_DIR}; re-run setup-gym.sh as above" >&2
        exit 1
    }
    export CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}" CHISEL_DIR="${CHISEL_DIR:-${SCRATCH}/chisel}"
    bash "${APERTUS2_DIR}/fetch-chisel.sh"
    export OVERLAY="${OVERLAY:+${OVERLAY} }${HERE}/tunnel-sc.yaml"
fi

export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}"
export TASK="${HERE}/terminus-sc.yaml" RECIPE="${ROOT}/experiments/token-capture-sc/qwen3-4b-1n-sc.yaml"
export SCRIPT=examples/run_grpo_single_controller.py
export RUN="${RUN:-harbor-sc-${AGENT}$([[ "${TUNNEL:-0}" == 1 ]] && echo -tunnel)-$(date +%Y%m%d-%H%M%S)}"
# 8 x 4 x 3 steps: ~5 min start-up + ~5-10 min per step expected; 3 x rollout_timeout_s (30 min)
# plus start-up in the worst case
export JOB_TIME="${JOB_TIME:-02:00:00}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-harbor-capture-sc}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

#!/bin/bash
# Harbor micro-tasks through Gym's harbor_agent with NeMo-RL's Single-Controller path and token
# capture at the model endpoint, Qwen3-4B on one node. See README.md.
#   bash experiments/harbor-capture-sc/run.sh [overrides...]
# Env:
#   AGENT                  terminus (default) or opencode
#   HARBOR_MODEL_BASE_URL  opencode only: model-server URL as reachable from the pods (required)
#   OPENCODE_VERSION       opencode only: opencode-ai npm version (default: latest)
#   HARBOR_ROOT            micro-task tree, train.jsonl, validation.jsonl (default $SCRATCH/osb/harbor_micro)
#   RUN, PARTITION, JOB_TIME (default 01:00:00), everything run-harbor-osb.sh reads.
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

export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}"
export TASK="${HERE}/terminus-sc.yaml" RECIPE="${ROOT}/experiments/token-capture-sc/qwen3-4b-1n-sc.yaml"
export SCRIPT=examples/run_grpo_single_controller.py
export RUN="${RUN:-harbor-sc-${AGENT}-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-01:00:00}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-harbor-capture-sc}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

#!/bin/bash
# Harbor micro-tasks through Gym's harbor_agent with NeMo-RL's Single-Controller path and token
# capture at the model endpoint, Qwen3-4B on one node. See README.md.
#   bash experiments/harbor-capture-sc/run.sh [overrides...]
# Env:
#   AGENT                  terminus (default), mini (mini-swe-agent in the sandbox, implies TUNNEL=1; Gym
#                          patch 0018, mini-sc.yaml) or opencode (OpenCode in the sandbox, implies TUNNEL=1;
#                          Gym patches 0024+0029, opencode-sc.yaml)
#   MSWEA_VERSION          mini only: li-boxuan/mini-swe-agent git ref (default: the commit pinned in mini-sc.yaml)
#   OPENCODE_VERSION       opencode only: default 1.17.11 (the version whose keys 0026 was checked against)
#   OPENCODE_TIMEOUT_SEC   opencode only: OpenCode's own time limit, below the agent timeout (default 870)
#   OPENCODE_DEBUG         opencode only: true = OpenCode debug logs in agent/opencode.txt (default false)
#   TUNNEL                 1: per-sandbox reverse tunnel sandbox 127.0.0.1:8000 -> policy_model (Gym patch
#                          0016, tunnel-sc.yaml); every trial checks GET /health through it before starting.
#                          With terminus the agent still calls the model from the node (step A: the tunnel
#                          runs alongside). CHISEL_DIR (default $SCRATCH/chisel) is filled on first use.
#   HARBOR_ROOT            micro-task tree, train.jsonl, validation.jsonl (default $SCRATCH/osb/harbor_micro)
#   RUN, PARTITION, JOB_TIME (default 02:00:00), everything run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"

GYM_DIR="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym"
grep -q harbor_model_base_url "${GYM_DIR}/responses_api_agents/harbor_agent/app.py" 2>/dev/null || {
    echo "ERROR: the Gym checkout in ${GYM_DIR} lacks the capture patches; run" >&2
    echo "  bash ${APERTUS2_DIR}/setup-gym.sh   (applies every patch in ${APERTUS2_DIR}/patches/gym)" >&2
    exit 1
}

AGENT="${AGENT:-terminus}"
export OVERLAY=""
case "${AGENT}" in
    terminus) ;;
    mini)
        grep -q MiniSweAgentNemoGym \
            "${GYM_DIR}/responses_api_agents/harbor_agent/custom_agents/mini_swe_agent_nemo_gym.py" 2>/dev/null || {
            echo "ERROR: AGENT=mini needs Gym patches 0018-0020 (mini-swe-agent) in ${GYM_DIR}; re-run setup-gym.sh as above" >&2
            exit 1
        }
        # the agent calls the model from inside the sandbox: only through the tunnel
        TUNNEL=1
        export OVERLAY="${HERE}/mini-sc.yaml"
        ;;
    opencode)
        grep -q opencode_workdir \
            "${GYM_DIR}/responses_api_agents/harbor_agent/custom_agents/opencode_nemo_gym.py" 2>/dev/null || {
            echo "ERROR: AGENT=opencode needs Gym patches 0026-0029 (OpenCode adapter) in ${GYM_DIR}; re-run setup-gym.sh as above" >&2
            exit 1
        }
        # OpenCode always streams: without 0024 every call gets 422 under token capture
        grep -q "def _stream_served_response" "${GYM_DIR}/nemo_gym/base_responses_api_model.py" 2>/dev/null || {
            echo "ERROR: AGENT=opencode needs Gym patches 0023-0024 (streaming under capture) in ${GYM_DIR}; re-run setup-gym.sh as above" >&2
            exit 1
        }
        # the agent calls the model from inside the sandbox: only through the tunnel
        TUNNEL=1
        export OVERLAY="${HERE}/opencode-sc.yaml"
        ;;
    *) echo "ERROR: AGENT must be terminus, mini or opencode, not ${AGENT}" >&2; exit 1 ;;
esac
for v in OPENCODE_VERSION OPENCODE_TIMEOUT_SEC OPENCODE_DEBUG; do
    [[ -z "${!v:-}" ]] || export "${v}"
done
[[ -z "${MSWEA_VERSION:-}" ]] || export MSWEA_VERSION

if [[ "${TUNNEL:-0}" == 1 ]]; then
    grep -q server_start_script \
        "${GYM_DIR}/responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/model_tunnel.py" 2>/dev/null || {
        echo "ERROR: TUNNEL=1 needs Gym patches 0016+0017 (tunnel) in ${GYM_DIR}; re-run setup-gym.sh as above" >&2
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

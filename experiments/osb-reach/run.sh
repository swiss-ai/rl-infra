#!/bin/bash
# Network between OpenSandbox pods and a CSCS compute node, on one debug node. See README.md.
#   bash experiments/osb-reach/run.sh                         # REACH_TESTS=gymtunnel (default)
#   OLD_START=1 bash experiments/osb-reach/run.sh             # job 5245999's server start, for comparison
#   REACH_TESTS="tcp ws tunnel" bash experiments/osb-reach/run.sh
# tests: tcp = pod -> node TCP; ws = node -> pod HTTP/WebSocket via the OpenSandbox proxy;
#        tunnel = chisel reverse tunnel per sandbox, pod 127.0.0.1:8000 -> node (standalone scripts);
#        gymtunnel = the same through Gym's model_tunnel.py (patch 0016 in the NeMo-RL clone's Gym).
# Env: ACCOUNT (infra01), PARTITION (debug), JOB_TIME (00:20:00), REACH_PORT, EXTRA_TARGETS ("host:port ..."),
#      TUNNELS (tunnel 4, gymtunnel 32), STREAM_S, IDLE_S, GYM_TUNNEL_TIMEOUT_S (900), OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY (else ~/.opensandbox_token).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"

if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    [[ -r "${HOME}/.opensandbox_token" ]] || { echo "ERROR: no OPENSANDBOX_API_KEY and no ~/.opensandbox_token" >&2; exit 1; }
    OPENSANDBOX_API_KEY="$(tr -d '\n' < "${HOME}/.opensandbox_token")"
fi
export OPENSANDBOX_API_KEY
export OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-https://opensandbox-gvisor.breithorn.svc.cscs.ch}"
export NEMO_GYM_RUN_ID="osb-reach-$(date +%Y%m%d-%H%M%S)" NEMO_GYM_WORKLOAD=osb-reach
export GYM_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
export REACH_PORT="${REACH_PORT:-}" EXTRA_TARGETS="${EXTRA_TARGETS:-}" REACH_PYTHON="${REACH_PYTHON:-}" \
    REACH_TESTS="${REACH_TESTS:-gymtunnel}" OLD_START="${OLD_START:-0}" WS_PORT="${WS_PORT:-8765}" HOLD_S="${HOLD_S:-60}" \
    TUNNELS="${TUNNELS:-}" GYM_TUNNEL_TIMEOUT_S="${GYM_TUNNEL_TIMEOUT_S:-}" STREAM_S="${STREAM_S:-}" IDLE_S="${IDLE_S:-}"
# Gym server venvs of the Harbor runs (run-harbor-osb.sh default); reach_node.sh takes the
# harbor_agent one, which has the OpenSandbox SDK.
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}/venvs}"

OUT="${SCRATCH}/osb-reach/${NEMO_GYM_RUN_ID}"

# chisel for the tunnel test (login node, pinned sha256): arm64 for the node, amd64/arm64 for the pod.
export CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}" CHISEL_DIR="${CHISEL_DIR:-${SCRATCH}/chisel}"
[[ " ${REACH_TESTS} " != *tunnel* ]] || bash "${APERTUS2_DIR}/fetch-chisel.sh"
if [[ " ${REACH_TESTS} " == *" gymtunnel "* ]]; then
    grep -q server_start_script \
        "${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/model_tunnel.py" \
        2>/dev/null || { echo "ERROR: gymtunnel needs Gym patches 0016+0017 in the NeMo-RL clone; run setup-gym.sh" \
                              "(see experiments/harbor-capture-sc/README.md)" >&2; exit 1; }
fi
mkdir -p "${OUT}"
echo "== ${NEMO_GYM_RUN_ID}: output in ${OUT}"
cd "${OUT}"
srun -A "${ACCOUNT:-infra01}" -p "${PARTITION:-debug}" -N1 -n1 -t "${JOB_TIME:-00:20:00}" -J osb-reach \
    --environment="${EDF_NAME}" bash "${HERE}/reach_node.sh" 2>&1 | tee reach.log

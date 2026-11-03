#!/bin/bash
# Can an OpenSandbox pod open a TCP connection to a CSCS compute node? Starts a listener on one compute
# node and one sandbox (created as in the Harbor runs) that connects to it. See README.md.
#   bash experiments/osb-reach/run.sh
# Env: ACCOUNT (infra01), PARTITION (debug), REACH_PORT (random 5000-5999), EXTRA_TARGETS ("host:port ..."),
#      OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY (else ~/.opensandbox_token).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${APERTUS2_DIR}/env"
[[ ! -r "${ROOT}/site.env" ]] || source "${ROOT}/site.env"

if [[ -z "${OPENSANDBOX_API_KEY:-}" ]]; then
    [[ -r "${HOME}/.opensandbox_token" ]] || { echo "ERROR: no OPENSANDBOX_API_KEY and no ~/.opensandbox_token" >&2; exit 1; }
    OPENSANDBOX_API_KEY="$(tr -d '\n' < "${HOME}/.opensandbox_token")"
fi
export OPENSANDBOX_API_KEY
export OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-https://opensandbox-gvisor.breithorn.svc.cscs.ch}"
export NEMO_GYM_RUN_ID="osb-reach-$(date +%Y%m%d-%H%M%S)" NEMO_GYM_WORKLOAD=osb-reach
export GYM_DIR=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
export REACH_PORT="${REACH_PORT:-}" EXTRA_TARGETS="${EXTRA_TARGETS:-}" REACH_PYTHON="${REACH_PYTHON:-}" \
    REACH_WS="${REACH_WS:-1}" WS_PORT="${WS_PORT:-8765}" HOLD_S="${HOLD_S:-60}"
# Gym server venvs of the Harbor runs (run-harbor-osb.sh default); reach_node.sh takes the
# harbor_agent one, which has the OpenSandbox SDK.
export NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${HARBOR_ROOT:-${SCRATCH}/osb/harbor_micro}/venvs}"

OUT="${SCRATCH}/osb-reach/${NEMO_GYM_RUN_ID}"
mkdir -p "${OUT}"
echo "== ${NEMO_GYM_RUN_ID}: output in ${OUT}"
cd "${OUT}"
srun -A "${ACCOUNT:-infra01}" -p "${PARTITION:-debug}" -N1 -n1 -t 00:15:00 -J osb-reach \
    --environment="${EDF_NAME}" bash "${HERE}/reach_node.sh" 2>&1 | tee reach.log

#!/bin/bash
# Token-capture test: mini_swe_agent_2 on OpenSandbox with Qwen2.5-1.5B on one node, through
# NeMo-RL's legacy GRPO entrypoint (run-harbor-osb.sh). See README.md.
#   bash experiments/token-capture-test/run.sh [overrides...]
# Env: PARTITION (default normal), JOB_TIME (default 01:00:00), RUN, HARBOR_ROOT
# (default $SCRATCH/osb/mini_swe), everything run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
source "${ROOT}/backends/nemo-rl/apertus2/env"
source "${ROOT}/site.env"

export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/mini_swe}"
# run-harbor-osb.sh insists on a Harbor task tree; this agent reads its tasks from the rows.
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${HARBOR_ROOT}/tasks}"
mkdir -p "${HARBOR_TASKS_DIR}"
python3 "${HERE}/make_rows.py" --gym "${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym" --out "${HARBOR_ROOT}"

export TASK="${HERE}/mini-swe-osb.yaml" RECIPE="${HERE}/qwen2.5-1.5b-1n.yaml"
export RUN="${RUN:-token-test-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-01:00:00}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-token-capture-test}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

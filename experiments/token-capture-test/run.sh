#!/bin/bash
# Token-capture test: mini_swe_agent_2 on OpenSandbox with Qwen2.5-1.5B on one node, through
# NeMo-RL's legacy GRPO entrypoint (run-harbor-osb.sh). See README.md.
#   bash experiments/token-capture-test/run.sh [overrides...]
# Env: PARTITION (default normal), JOB_TIME (default 01:00:00), RUN, HARBOR_ROOT
# (default $SCRATCH/osb/mini_swe), everything run-harbor-osb.sh reads.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${APERTUS2_DIR}/env"
source "${ROOT}/site.env"

export HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/mini_swe}"
# run-harbor-osb.sh insists on a Harbor task tree; this agent reads its tasks from the rows.
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:-${HARBOR_ROOT}/tasks}"
mkdir -p "${HARBOR_TASKS_DIR}"
python3 "${HERE}/make_rows.py" --gym "${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym" --out "${HARBOR_ROOT}"

# Nothing pins litellm, so uv takes the newest; from 1.103 it needs mcp>=2, while Gym caps mcp<2,
# and 1.106 imports mcp.types.CacheableResult on every completion call (job 5211452). Gym builds
# its server venvs with plain `uv pip install`, which honours UV_CONSTRAINT from the job's env.
printf 'litellm<1.103\n' > "${HARBOR_ROOT}/uv-constraints.txt"
export UV_CONSTRAINT="${HARBOR_ROOT}/uv-constraints.txt"
# A venv left over from a run without the constraint keeps the bad litellm; rebuild it.
MSA_VENV="${NEMO_GYM_VENV_DIR:-${HARBOR_ROOT}/venvs}/responses_api_agents/mini_swe_agent_2/.venv"
litellm_v="$(ls -d "${MSA_VENV}"/lib/python3.*/site-packages/litellm-*.dist-info 2>/dev/null \
    | sed -E 's/.*litellm-([0-9.]+)\.dist-info$/\1/' | head -1)"
if [[ -n "${litellm_v}" && "$(printf '%s\n' 1.103 "${litellm_v}" | sort -V | head -1)" == 1.103 ]]; then
    echo "removing ${MSA_VENV} (litellm ${litellm_v})"
    rm -rf "${MSA_VENV}"
fi

export TASK="${HERE}/mini-swe-osb.yaml" RECIPE="${HERE}/qwen2.5-1.5b-1n.yaml"
export RUN="${RUN:-token-test-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-01:00:00}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-token-capture-test}"
exec bash "${ROOT}/run-harbor-osb.sh" "$@"

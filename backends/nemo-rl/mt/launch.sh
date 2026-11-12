#!/bin/bash
# Same arguments as ../launch.sh; use an MT data processor and env_name: mt.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load
PATCH="${HERE}/patches/0001-grpo-reserve-mt-reward-resources.patch"
if ! git -C "${NEMO_RL_DIR}" apply --reverse --check "${PATCH}" 2>/dev/null; then
    echo "ERROR: run bash backends/nemo-rl/mt/setup.sh before launching MT" >&2
    exit 1
fi
export PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}"
export SCRIPT="${HERE}/run_grpo.py"
exec bash "${HERE}/../launch.sh" "$@"

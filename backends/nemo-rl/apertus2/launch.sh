#!/bin/bash
# Submit a NeMo-RL run on the Apertus 2 stack: backends/nemo-rl/launch.sh with this directory's
# overrides (fork clone, image, EDF, driver). Same arguments:
#   bash backends/nemo-rl/apertus2/launch.sh <config> <num_nodes> <run_name> [overrides...]
set -euo pipefail

export APERTUS2_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${APERTUS2_DIR}/env"
export SCRATCH NEMO_RL_DIR PIN_FILE EDF_NAME DRIVER_PYTHON IMAGE_MASTER IMAGE_WORK
exec bash "${APERTUS2_DIR}/../launch.sh" "$@"

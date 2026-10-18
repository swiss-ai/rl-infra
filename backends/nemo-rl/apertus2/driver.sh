#!/bin/bash
# Driver entry on the Ray head inside the Apertus 2 NeMo-RL container (cwd /opt/nemo-rl).
# Everything runs on the image's system Python (the image sets NEMO_RL_PY_EXECUTABLES_SYSTEM=1,
# so Ray actors use it too). First (re)installs the runtime overlay (pydeps.sh) if it is missing
# or out of date. Exports RL_INFRA_DIR (this repository) for configs that reference files in it.
set -euo pipefail

export RL_INFRA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
bash "$(dirname "$0")/pydeps.sh"

exec /usr/bin/python3.12 "$@"

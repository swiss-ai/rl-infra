#!/bin/bash
# Run inside the nemo-rl container. Keep scoring packages in a scratch venv,
# and reuse the container's GPU/Ray packages through an explicit .pth file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER=/opt/nemo_rl_venv/bin/python
SCORER_VENV="${SCORER_VENV:-/iopsstor/scratch/cscs/${USER}/mt-reward-venv}"
[[ -x "${DRIVER}" ]] || { echo "ERROR: run this script inside the nemo-rl container" >&2; exit 1; }
uv venv --allow-existing --python "${DRIVER}" "${SCORER_VENV}"
DRIVER_SITE="$(${DRIVER} -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
SCORER_SITE="$("${SCORER_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "${DRIVER_SITE}" > "${SCORER_SITE}/nemo_container.pth"
uv pip install --python "${SCORER_VENV}/bin/python" -r "${HERE}/requirements-scorer.txt"
"${SCORER_VENV}/bin/python" - <<'PY'
import sys
import ray
import torch
import transformers
import fasttext
import sacrebleu
print("Python:", sys.version)
print("Ray:", ray.__version__)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Reward environment imports passed")
PY
echo "scorer_python: ${SCORER_VENV}/bin/python"

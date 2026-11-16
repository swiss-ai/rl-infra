#!/bin/bash
# Install the isolated COMET runtime on shared scratch (no Ray dependency).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KIWI_VENV="${KIWI_VENV:-/iopsstor/scratch/cscs/${USER}/mt-kiwi-venv}"
# The container exports training constraints (including protobuf>=6), and its
# workdir is a Python 3.13 project. Neither applies to this isolated COMET env.
unset UV_CONSTRAINT PIP_CONSTRAINT UV_BUILD_CONSTRAINT PIP_BUILD_CONSTRAINT
unset UV_OVERRIDE UV_CONFIG_FILE UV_PROJECT_ENVIRONMENT VIRTUAL_ENV PYTHONHOME PYTHONPATH
KIWI_SETUP_DIR="$(mktemp -d)"
trap 'rmdir "${KIWI_SETUP_DIR}"' EXIT
cd "${KIWI_SETUP_DIR}"
uv --no-config venv --allow-existing --python 3.12 "${KIWI_VENV}"
uv --no-config pip install --python "${KIWI_VENV}/bin/python" -r "${HERE}/requirements-kiwi.txt"
"${KIWI_VENV}/bin/python" - <<'PY'
import torch
from comet import download_model, load_from_checkpoint
print("COMET imports passed; PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
if torch.version.cuda is None:
    raise RuntimeError("COMET environment needs a CUDA-enabled PyTorch build for GPU scoring")
PY
printf 'cometkiwi_python: %s/bin/python\n' "${KIWI_VENV}"

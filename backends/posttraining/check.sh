#!/bin/bash
# Check of the SFT environment inside the container (bash backends/posttraining/submit.sh v11-check check.sh): the
# Python overlay and the converted mixture (made on first use, as in train.sh), then check_env.py. Safe to re-run.
set -euo pipefail
bash "${PT_DIR}/build_env.sh"
python3 "${PT_DIR}/prepare_data.py" "${HF_MIXTURE_DIR}" "${SFT_ROOT}/data/apertus-sft-mixture"
python3 "${PT_DIR}/check_env.py" "${SFT_ROOT}/data/apertus-sft-mixture" \
    "${MODELS_DIR}/Apertus-v1.1-0.5B" "${MODELS_DIR}/Apertus-v1.1-0.5B-Instruct"

#!/bin/bash
# Download the coding dataset, then write the two files envs/code-gen-sft.yaml reads. A row holds its
# unit tests and can reach hundreds of MB, so you choose how large a row may be, file by file. Keep the
# validation cap below the sandbox's SANDBOX_MAX_REQUEST_BYTES: a larger row stops the run. Filtering
# needs some CPU time, so run this on a compute node.
#   MAX_TRAIN_ROW_BYTES=4194304 MAX_VALIDATION_ROW_BYTES=200000000 bash setup-data.sh
set -euo pipefail

: "${MAX_TRAIN_ROW_BYTES:?Set the largest training row to keep, in bytes.}"
: "${MAX_VALIDATION_ROW_BYTES:?Set the largest validation row to keep, in bytes.}"

SANDBOX_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SANDBOX_DIR}/../../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
OUT="${SCRATCH}/code-gen"
SOURCE="${OUT}/source"
URL="https://huggingface.co/datasets/nvidia/nemotron-RL-coding-competitive_coding/resolve/main"
mkdir -p "${SOURCE}"

if [[ -z "${SKIP_DOWNLOAD:-}" ]]; then
    curl --fail --location --retry 5 --continue-at - --output "${SOURCE}/train.jsonl" \
        "${URL}/opencodereasoning_filtered_25k_train.jsonl"
    curl --fail --location --retry 5 --continue-at - --output "${SOURCE}/validation.jsonl" \
        "${URL}/validation.jsonl"
fi

python3.12 -I "${SANDBOX_DIR}/prepare_data.py" --input "${SOURCE}/train.jsonl" \
    --output "${OUT}/train.jsonl" --max-row-bytes "${MAX_TRAIN_ROW_BYTES}" \
    --agent-name code_gen_simple_agent
python3.12 -I "${SANDBOX_DIR}/prepare_data.py" --input "${SOURCE}/validation.jsonl" \
    --output "${OUT}/validation.jsonl" --max-row-bytes "${MAX_VALIDATION_ROW_BYTES}" \
    --agent-name code_gen_simple_agent

#!/bin/bash
# Start the code sandbox on one node of its own, as a job of its own, and print its job ID. The
# server writes SANDBOX_ENDPOINT_FILE (default $SCRATCH/code-sandbox/endpoint.json) once it has checked that sandboxes are isolated,
# and Gym's code_gen server reads the address and token there. Stop it with scancel.
#   JOB_TIME=06:00:00 SANDBOX_WORKERS=128 SANDBOX_MAX_REQUEST_BYTES=268435456 bash start.sh
set -euo pipefail

SANDBOX_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SANDBOX_DIR}/../../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
OUT="${SCRATCH}/code-sandbox"
mkdir -p "${OUT}/logs"
ENDPOINT_FILE="${SANDBOX_ENDPOINT_FILE:-${OUT}/endpoint.json}"
rm -f "${ENDPOINT_FILE}"
cd "${OUT}"

# shellcheck disable=SC2086  # SBATCH_EXTRA holds the site's several flags.
sbatch --parsable --nodes=1 ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time="${JOB_TIME:-05:00:00}" \
    --job-name=code-sandbox --output="${OUT}/logs/%j.out" \
    --wrap "exec python3.12 ${SANDBOX_DIR}/sandbox_server.py --env-dir ${OUT}/env \
        --endpoint-file ${ENDPOINT_FILE} --workers ${SANDBOX_WORKERS:-128} \
        --max-request-bytes ${SANDBOX_MAX_REQUEST_BYTES:-268435456}"

#!/bin/bash
# Start the Lean 4 sandbox on one intTest node of its own, as a job of its own, and print its job ID. It
# runs the NeMo-Skills sandbox image (Lean 4 + Mathlib, arm64) as Gym math_formal_lean expects it:
# POST /execute on port 6000. It runs in bubblewrap on the unpacked image, so it sees no scratch or home,
# though compiled code keeps the network (see sandbox_job.sh). Once it answers, the job writes the node
# name to LEAN_SANDBOX_ENDPOINT_FILE (default $SCRATCH/lean/sandbox-endpoint); pass that node as
#   env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.sandbox_host=<node>
# Stop it with scancel.
#   JOB_TIME=06:00:00 LEAN_SANDBOX_WORKERS=128 bash start.sh
set -euo pipefail

LEAN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_DIR}/../../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
OUT="${SCRATCH}/lean"
IMAGE="${LEAN_SANDBOX_IMAGE:-${OUT}/images/nemo-skills-sandbox-0.7.1-arm64.sqsh}"
ENDPOINT_FILE="${LEAN_SANDBOX_ENDPOINT_FILE:-${OUT}/sandbox-endpoint}"
[[ -s "${IMAGE}" ]] || { echo "ERROR: no sandbox image at ${IMAGE}; run import_image.sh first" >&2; exit 1; }
mkdir -p "${OUT}/logs"
rm -f "${ENDPOINT_FILE}"
cd "${OUT}"

# shellcheck disable=SC2086  # SBATCH_EXTRA holds the site several flags.
sbatch --parsable --nodes=1 ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time="${JOB_TIME:-05:00:00}" \
    --job-name=lean-sandbox --output="${OUT}/logs/%j-sandbox.out" \
    --export=ALL,LEAN_IMAGE="${IMAGE}",ENDPOINT_FILE="${ENDPOINT_FILE}",NUM_WORKERS="${LEAN_SANDBOX_WORKERS:-128}" \
    "${LEAN_DIR}/sandbox_job.sh"

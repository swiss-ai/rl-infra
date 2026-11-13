#!/bin/bash
# Start a Kimina Lean Server on one node, as a job of its own, and print its job ID; envs/lean-kimina-sft.yaml
# names its endpoint at launch. Stop it with scancel after the training run. Run setup.sh and setup-data.sh first.
# NOT YET RUN WITH A TRAINING JOB (see README.md).
#   JOB_TIME=06:00:00 MAX_REPLS=128 bash backends/nemo-rl/lean-kimina/start.sh
# 128 REPLs: about 2 GB each, within a job's 450 GB next to the Lean copy; 64 already check an RL step of
# 2,048 proofs as fast (the fork's docs/MANY_CORE_NODES.md).
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
[[ -x "$(kimina_repl_bin)" ]] || { echo "ERROR: no Kimina install in ${KIMINA_INSTALL}; run setup.sh first" >&2; exit 1; }
rows="${HEADER_ROWS:-${LEAN_DATA}/data/train.v434.jsonl}"
[[ -s "${rows}" ]] || { echo "ERROR: no training rows ${rows}; run setup-data.sh first" >&2; exit 1; }
OUT_DIR="${SCRATCH}/lean-kimina"
mkdir -p "${OUT_DIR}/logs"
# shellcheck disable=SC2086
sbatch --parsable --nodes=1 --exclusive ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time="${JOB_TIME:-05:00:00}" \
    --job-name=lean-kimina --output="${OUT_DIR}/logs/%j-kimina.out" \
    --export=ALL,LEAN_KIMINA_DIR="${LEAN_KIMINA_DIR}",ROOT="${ROOT}",ENDPOINT_FILE="${OUT_DIR}/endpoint",MAX_REPLS="${MAX_REPLS:-128}",PORT="${PORT:-8000}",HEADER_ROWS="${rows}" \
    "${LEAN_KIMINA_DIR}/server_job.sh"

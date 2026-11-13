#!/bin/bash
# The Lean task's rows for a Kimina verifier, made in one job on Alps (prints the job ID). The rows are those of
# backends/nemo-rl/lean/setup-data.sh, made by the same scripts (download.py, reduce_nemotron.py, Gym's
# prepare_nemotron_math_proofs.py at the Gym pin, fix_rows.py), but on the host with the venv of setup.sh
# instead of a pyxis container, and without a NeMo-RL clone (only Gym's two files are fetched, at its pin).
# Then a Kimina server in the same job keeps the rows whose statement elaborates on its Lean/Mathlib:
#   $LEAN_DATA/data/train.jsonl, validation.jsonl          as backends/nemo-rl/lean writes them
#   $LEAN_DATA/data/train.v434.jsonl, validation.v434.jsonl the rows that elaborate (envs/lean-kimina-sft.yaml)
#   $LEAN_DATA/data/*.v434.jsonl.dropped.jsonl              the others, with Lean's first error
# Nemotron-Math-Proofs-v1 (29.5 GB) stays in $LEAN_DATA/source, so a rerun skips the download.
# Run setup.sh first. A rerun skips every step whose output exists.
#   PARTITION=debug bash backends/nemo-rl/lean-kimina/setup-data.sh
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
[[ -x "$(kimina_repl_bin)" && -x "${KIMINA_INSTALL}/venv/bin/python" ]] \
    || { echo "ERROR: no Kimina install in ${KIMINA_INSTALL}; run setup.sh first" >&2; exit 1; }
mkdir -p "${LEAN_DATA}/logs" "${LEAN_DATA}/source" "${LEAN_DATA}/data"
cd "${LEAN_DATA}"
# shellcheck disable=SC2086
sbatch --parsable --nodes=1 --exclusive ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time="${JOB_TIME:-01:25:00}" \
    --job-name=lean-kimina-data --output="${LEAN_DATA}/logs/%j-kimina-data.out" \
    --export=ALL,LEAN_KIMINA_DIR="${LEAN_KIMINA_DIR}",ROOT="${ROOT}" \
    "${LEAN_KIMINA_DIR}/setup-data-job.sh"

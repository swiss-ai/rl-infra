#!/bin/bash
# Lean task rows, made on one intTest node (prints the job ID): Nemotron-Math-Proofs-v1 (29.5 GB, pinned
# revision) selected by Gym prepare_nemotron_math_proofs.py with its defaults (hard, medium and easy theorems
# not used in Nano v3 SFT: 9,605), then fix_rows.py; validation is Gym miniF2F test (244). Each row gets the
# agent_ref of math_formal_lean_simple_agent. Writes $SCRATCH/lean/data/{train,validation}.jsonl and a sample
# of 400 verified proofs from the data for test_sandbox.py. Needs the NeMo-RL clone with Gym (run.sh or
# setup-gym.sh).
set -euo pipefail
LEAN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${ROOT}/backends/nemo-rl/apertus2/env"
OUT="${SCRATCH}/lean"
GYM_LEAN="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym/resources_servers/math_formal_lean"
[[ -f "${GYM_LEAN}/prepare_nemotron_math_proofs.py" ]] || { echo "ERROR: no Gym in ${NEMO_RL_DIR}" >&2; exit 1; }
mkdir -p "${OUT}/logs" "${OUT}/source" "${OUT}/data"
cd "${OUT}"
# shellcheck disable=SC2086
sbatch --parsable --nodes=1 ${SBATCH_EXTRA} --account="${ACCOUNT}" \
    --partition="${PARTITION:-${RUN_PARTITION:-${DEFAULT_PARTITION}}}" --time=02:00:00 \
    --job-name=lean-data --output="${OUT}/logs/%j-data.out" \
    --export=ALL,OUT="${OUT}",GYM_LEAN="${GYM_LEAN}",LEAN_DIR="${LEAN_DIR}",IMAGE="${IMAGE_WORK}" \
    --wrap "bash ${LEAN_DIR}/setup-data-job.sh"

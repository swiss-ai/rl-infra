#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --job-name=qwen3-8b-mt-smoke
#SBATCH --output=mt-smoke-%j.out

# Submit from the rl-infra repository root:
#   sbatch experiments/grpo-mt/run_smoke.sh
# Slurm copies this script into its spool, so use SLURM_SUBMIT_DIR to find the repo.
set -euo pipefail
ROOT="${RL_INFRA_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}}"
[[ -f "${ROOT}/backends/nemo-rl/mt/launch.sh" ]] || {
    echo "ERROR: submit from the rl-infra root, or set RL_INFRA_DIR to its absolute path" >&2
    exit 1
}
cd "${ROOT}"
export MT_DATA_PATH="${MT_DATA_PATH:-/capstor/scratch/cscs/${USER}/data_rl/es-de.jsonl}"
export MT_SCORER_CONFIG="${MT_SCORER_CONFIG:-${ROOT}/experiments/grpo-mt/scorer.json}"
[[ -r "${MT_DATA_PATH}" && -r "${MT_SCORER_CONFIG}" ]] || {
    echo "ERROR: check MT_DATA_PATH=${MT_DATA_PATH} and MT_SCORER_CONFIG=${MT_SCORER_CONFIG}" >&2
    exit 1
}
export RUN_IN_ALLOCATION=1
export PARTITION="${SLURM_JOB_PARTITION}"
export JOB_TIME=00:30:00
exec bash backends/nemo-rl/mt/launch.sh \
    experiments/grpo-mt/qwen3-8b-smoke.yaml 1 qwen3-8b-mt-smoke "$@"

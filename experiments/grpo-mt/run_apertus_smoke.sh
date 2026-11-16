#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --job-name=apertus-8b-instruct-2509-mt-smoke
#SBATCH --output=mt-smoke-%j.out

# Submit from the rl-infra repository root:
#   sbatch experiments/grpo-mt/run_apertus_smoke.sh
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
# The check loads config/tokenizer only; no policy weights are downloaded.
srun --nodes=1 --ntasks=1 --environment="${EDF_NAME:-nemo-rl}" \
    /opt/nemo_rl_venv/bin/python "${ROOT}/backends/nemo-rl/mt/check_policy.py" \
    --model swiss-ai/Apertus-8B-Instruct-2509
export RUN_IN_ALLOCATION=1
export PARTITION="${SLURM_JOB_PARTITION}"
export JOB_TIME=00:10:00
exec bash backends/nemo-rl/mt/launch.sh \
    experiments/grpo-mt/apertus-8b-instruct-2509-smoke.yaml 1 apertus-8b-instruct-2509-mt-smoke "$@"

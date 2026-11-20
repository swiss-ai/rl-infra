#!/bin/bash
# Checks of the native Enigmata environment (test_env.py) on one intTest node, no model.
# Usage: sbatch test_env.sbatch   (NEMO_RL_DIR overrides the NeMo-RL copy)
#SBATCH --job-name=enigmata-envtest
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/hanint06/data/stem-rl/enigmata/logs/%x-%j.out
set -euo pipefail
E=/scratch/hanint06/data/stem-rl/enigmata
N=${NEMO_RL_DIR:-${HOME}/projects/nemo-rl-apertus2}
srun --ntasks=1 --cpu-bind=none --cpus-per-task=64 \
    --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
    --container-writable --no-container-mount-home --container-workdir="${E}" \
    python3 -u ${RL_INFRA_DIR:-${HOME}/projects/rl-infra}/experts/stem/scripts/enigmata/test_env.py "${N}" "${E}/repo" "${E}/pool"

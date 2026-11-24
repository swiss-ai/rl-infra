#!/bin/bash
#SBATCH --job-name=build-run3-1
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/hanint06/data/stem-rl/stats/logs/%x-%j.out
set -euo pipefail
srun --ntasks=1 --cpus-per-task=64 \
    --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
    --container-writable --no-container-mount-home --container-workdir=/scratch/hanint06/data/stem-rl/stats \
    bash -c "export HF_HUB_OFFLINE=1; python3 -I /scratch/hanint06/data/stem-rl/scripts/build_run3.py"

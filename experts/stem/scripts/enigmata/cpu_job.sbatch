#!/bin/bash
# Run CMD (a shell command) in the NeMo-RL container on one intTest node, for small data steps and checks.
# Usage: sbatch --job-name=<name> --export=ALL,CMD="<command>" cpu_job.sbatch
#SBATCH --job-name=enigmata-cpu
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/hanint06/data/stem-rl/enigmata/logs/%x-%j.out
set -euo pipefail
srun --ntasks=1 --cpu-bind=none --cpus-per-task=64 \
    --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
    --container-writable --no-container-mount-home --container-workdir=/scratch/hanint06/data/stem-rl/enigmata \
    bash -c "${CMD}"

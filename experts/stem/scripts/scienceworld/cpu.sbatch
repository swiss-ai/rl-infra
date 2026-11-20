#!/bin/bash
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home01/hanint06/sciworld/logs/%x-%j.out
# CPU work for ScienceWorld in the NeMo-RL container, with the scienceworld package on PYTHONPATH:
#   sbatch --job-name=<name> --export=ALL,CMD="python3 -u <script> <args>" cpu.sbatch
set -euo pipefail
SW=/home01/hanint06/sciworld
IMAGE=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh
srun --ntasks=1 --cpu-bind=none --cpus-per-task=$SLURM_CPUS_ON_NODE --network=disable_rdzv_get \
    --container-image=$IMAGE --container-mounts=/scratch:/scratch,/home01:/home01 --container-workdir=$SW/tools \
    env PYTHONPATH=$SW/pydeps PYTHONDONTWRITEBYTECODE=1 bash -c "$CMD"

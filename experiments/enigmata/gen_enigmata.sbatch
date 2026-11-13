#!/bin/bash
# Enigmata puzzle pool: gen_enigmata.py on one intTest node (CPU work only; writes to OUT).
# Usage: sbatch --export=ALL,COUNT=<n>,OUT=<dir>,TIMEOUT=<s> gen_enigmata.sbatch
#SBATCH --job-name=enigmata-gen
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/hanint06/data/stem-rl/enigmata/logs/%x-%j.out
set -euo pipefail
E=/scratch/hanint06/data/stem-rl/enigmata
srun --ntasks=1 --cpu-bind=none --cpus-per-task="${SLURM_CPUS_ON_NODE}" \
    --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
    --container-writable --no-container-mount-home --container-workdir="${E}" \
    python3 -I /home01/hanint06/projects/rl-infra-enigmata/experiments/enigmata/gen_enigmata.py --repo "${E}/repo" --out "${OUT}" --count "${COUNT}" \
        --cell-timeout "${TIMEOUT}" --workers 96 --pydeps "${E}/pydeps" \
        --tokenizer /scratch/hanint06/models/chonk-lc-16k-v0.2-iter_0018315/tokenizer.json

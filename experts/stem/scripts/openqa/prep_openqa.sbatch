#!/bin/bash
#SBATCH --job-name=openqa-prep-1
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/hanint06/rl-analysis/openqa-judge/logs/%x-%j.out
set -euo pipefail
A=/scratch/hanint06/rl-analysis/openqa-judge
O=/scratch/hanint06/data/stem-rl/openqa
mkdir -p $O
srun --ntasks=1 --cpus-per-task=64 \
    --container-image=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh \
    --container-writable --no-container-mount-home --container-workdir=$A \
    python3 -I $A/prep_openqa.py /scratch/hanint06/data/stem-rl/raw/openqa/train.parquet $O
srun --ntasks=1 --cpus-per-task=128 \
    --container-image=/scratch/hanint06/container-images/vllm-openai-v0.33.0-aarch64.sqsh \
    --container-writable --no-container-mount-home --container-workdir=$A \
    bash -c "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 HF_HOME=$A/cache/hf VLLM_CACHE_ROOT=$A/cache/vllm XDG_CACHE_HOME=$A/cache/xdg TRITON_CACHE_DIR=$A/cache/triton TORCHINDUCTOR_CACHE_DIR=$A/cache/inductor NCCL_NET_PLUGIN=none; for w in 0 1 2 3; do CUDA_VISIBLE_DEVICES=\$w VLLM_PORT=\$((29500+100*w)) python3 $A/label_openqa.py \$w 4 $O /scratch/hanint06/models/gemma-4-26B-A4B-it > $A/logs/label-\$w.log 2>&1 & sleep 10; done; wait; echo labels-done"

#!/bin/bash
#SBATCH --job-name=chonk-serve
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#
# Serves one HF export of Chonk with vLLM on a whole node (TP1 DP4 EP) until the time limit or scancel.
# Once the model is up, in about 10 minutes, the URL is in ${ROOT}/endpoint, where chat.py finds it.
# Without --enforce-eager one step takes about 22 ms instead of 650 ms. CUDA graph capture needs a
# Mamba cache block per sequence, and vLLM's default of 1024 sequences is more than fit at memory 0.88.
# Submit from ${ROOT}:
#   cd /scratch/$USER/chonk-sft
#   sbatch --export=ALL,MODEL=<export>,IMAGE=<image> ~/projects/rl-infra/experiments/chonk-sft/serve.sbatch

set -euo pipefail
ROOT=${ROOT:-/scratch/$USER/chonk-sft}
MODEL=${MODEL:?Set MODEL to an HF export of Chonk.}
IMAGE=${IMAGE:?Set IMAGE to the NeMo-RL image.}
PORT=${PORT:-8011}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
LOG=$ROOT/logs/vllm-$SLURM_JOB_ID.log
export UV_DEFAULT_INDEX=https://pypi.org/simple

srun --ntasks=1 --network=disable_rdzv_get --container-image="$IMAGE" --container-writable \
    --container-mounts=/scratch:/scratch,/appsdata:/appsdata --container-env=UV_DEFAULT_INDEX \
    --container-workdir="$ROOT" bash -s <<EOF
set -uo pipefail
export PYTHONPATH=$ROOT/pydeps-vllm:\${PYTHONPATH:-}
export NCCL_NET=Socket NCCL_PROTO='^LL128' VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
export TOKENIZERS_PARALLELISM=false HF_HOME=$ROOT/cache/hf VLLM_CACHE_ROOT=$ROOT/cache/vllm XDG_CACHE_HOME=$ROOT/cache
export FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer TRITON_CACHE_DIR=/tmp/triton
mkdir -p $ROOT/cache/vllm
start=\$(date +%s)
vllm serve $MODEL --served-model-name chonk --port $PORT \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization 0.88 --max-num-seqs 256 --max-num-batched-tokens 16384 \
    --trust-remote-code > $LOG 2>&1 &
pid=\$!
until curl -sf localhost:$PORT/health > /dev/null; do
    kill -0 \$pid 2>/dev/null || { echo "SERVER_DIED, see $LOG"; tail -40 $LOG; exit 1; }
    sleep 10
done
echo "http://$SLURMD_NODENAME:$PORT" > $ROOT/endpoint
echo "READY after \$(( \$(date +%s) - start ))s: http://$SLURMD_NODENAME:$PORT serves $MODEL"
wait \$pid
EOF

#!/bin/bash
# Chonk on one KISTI node with `vllm serve`, then one completion. Checks the image, the checkpoint copy and the
# single-node settings. Takes about 15 minutes (load about 8 min, first request about 4 min of Triton compilation).
#   source kisti/env.sh && cd $KISTI_SCRATCH
#   sbatch "${KISTI_SLURM[@]}" -N1 -t 00:45:00 -o chonk_serve_%j.log \
#     --wrap="srun --container-image=$KISTI_IMAGE --container-mounts=$KISTI_SCRATCH:$KISTI_SCRATCH,$HOME/rl-infra:$HOME/rl-infra \
#             --container-writable $HOME/rl-infra/kisti/chonk_serve.sh"
# Expected: LOAD_SECONDS around 480, then a completion such as " Bern.\nThe official language of Switzerland is ...".
set -uo pipefail
source "${RL_INFRA:-${HOME}/rl-infra}/kisti/env.sh"

# The image's uv points at CSCS's JFrog mirror, which KISTI cannot reach.
export UV_DEFAULT_INDEX=https://pypi.org/simple
# The image's openai (2.7.2) is too old for its vLLM (ImportError: NamespaceTool); install a current one beside it.
PYDEPS="${KISTI_SCRATCH}/pydeps-kisti"
[[ -d "${PYDEPS}/openai" ]] || uv pip install --python /usr/bin/python3.12 --no-deps --target "${PYDEPS}" openai
export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"

# Single node: KISTI assigns no Slingshot VNI, so the OFI plugin cannot open a domain ("NCCL error: invalid usage").
# Sockets are only the fallback transport; GPUs in one node talk over NVLink.
export NCCL_NET=Socket NCCL_PROTO='^LL128'
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${KISTI_SCRATCH}/hf" VLLM_CACHE_ROOT="${KISTI_SCRATCH}/vllm_cache" XDG_CACHE_HOME="${KISTI_SCRATCH}/.cache"
export FLASHINFER_WORKSPACE_BASE="${KISTI_SCRATCH}/.cache/flashinfer" TRITON_CACHE_DIR=/tmp/triton
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}"

LOG="${KISTI_SCRATCH}/chonk_serve_full_${SLURM_JOB_ID:-local}.log"
start=$(date +%s)
# Chonk has 7 KV heads, so tensor parallelism must be 1: 4 data-parallel replicas with expert parallelism
# (the blueprint's vLLM layout, TP1 EP4 per node).
vllm serve "${KISTI_MODEL}" --served-model-name chonk --port 8011 \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager --trust-remote-code \
    --enable-chunked-prefill > "${LOG}" 2>&1 &
pid=$!
until curl -sf localhost:8011/health > /dev/null; do
    kill -0 "${pid}" 2> /dev/null || { echo "SERVER_DIED, see ${LOG}"; tail -30 "${LOG}"; exit 1; }
    sleep 10
done
echo "LOAD_SECONDS $(( $(date +%s) - start ))"
curl -s localhost:8011/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "chonk", "prompt": "The capital of Switzerland is", "max_tokens": 24, "temperature": 0}'
echo
grep -E 'Loading weights took|Model loading took|Available KV cache' "${LOG}" | head -3
kill "${pid}"
wait

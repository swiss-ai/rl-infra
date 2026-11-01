#!/bin/bash
# Smallest test of the judge, before queueing a whole training job: one vLLM on one GPU in the
# Apertus 2 container, without NeMo-RL or a Gym server. It loads the judge model, asks it to score a
# good and a bad answer with the real rubrics, and fails unless the good one scores higher. Runs
# as a one-node job, after setup-gym.sh on the login node (it reads the rubrics from the Gym checkout):
#   PYDEPS_GYM=1 PYDEPS=$SCRATCH/nemo-rl-apertus2/pydeps-gym CHECK_SCRIPT=check_judge.sh \
#       bash backends/nemo-rl/apertus2/check_env_job.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${JUDGE_MODEL:-${SCRATCH}/models/Qwen3.8-27B-FP8}"
PORT="${JUDGE_PORT:-8011}"

bash "${HERE}/pydeps.sh"

CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.12 -c 'from vllm.entrypoints.cli.main import main; main()' serve "${MODEL}" \
    --port "${PORT}" --max-model-len 8192 --gpu-memory-utilization 0.88 --max-num-seqs 16 &
server=$!
trap 'kill "${server}" 2>/dev/null || true' EXIT

echo "== waiting for ${MODEL}"
for _ in $(seq 1 180); do
    kill -0 "${server}" 2>/dev/null || { echo "ERROR: vLLM exited during start-up" >&2; exit 1; }
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "ERROR: vLLM not ready after 15 minutes" >&2; exit 1; }

/usr/bin/python3.12 "${HERE}/check_judge.py" "${MODEL}" "http://127.0.0.1:${PORT}/v1/chat/completions"

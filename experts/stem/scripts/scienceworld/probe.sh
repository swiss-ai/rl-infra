#!/bin/bash
# bash probe.sh <task file> [extra hydra overrides...]
# ScienceWorld episodes with SFT v0.2 on one node, vLLM only (sciworld_score.py through run.sh, NeMo-RL copy with
# the scienceworld environment). Results: /scratch/hanint06/nemo-rl/logs/<RUN>/{episodes.jsonl,summary.json}.
# RUN (default: the task file's name), JOB_TIME (default 2 h) and DRY_RUN pass through.
set -euo pipefail
TASK_FILE=$(realpath "$1"); shift
cd "$(dirname "$0")/../../../.."
export SCRIPT=${PWD}/experts/stem/scripts/scienceworld/sciworld_score.py PREFLIGHT=0
export RECIPE=${PWD}/experts/stem/scripts/scienceworld/recipe-sftv02-1node.yaml
export RUN=${RUN:-sciworld-$(basename "${TASK_FILE}" .yaml)}
export JOB_TIME=${JOB_TIME:-02:00:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN}"
DEBUG=0 bash run.sh "${TASK_FILE}" logger.wandb_enabled=false checkpointing.enabled=false \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 \
    ++policy.generation.vllm_cfg.env_vars.NCCL_NET_PLUGIN=none ++policy.generation.vllm_cfg.env_vars.NCCL_NET=Socket "$@"

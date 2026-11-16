#!/bin/bash
# bash launch.sh [extra hydra overrides...]
# Run 5: run 4 (experiments/stem-mix-run4/launch.sh) with RDKit chemistry added and a new split (envs/stem-mix-run5.yaml):
# 32 prompts each from Reasoning Gym, Enigmata, ScienceWorld and chemistry, 96 each from DAPO, NuminaMath, MCQ and
# openqa. 16 trainer + 16 vLLM + 1 judge node, variant B of handover/rl-throughput-and-512p-config.md (512 prompts x 16
# answers, one optimizer step, no log-prob pass).
# Checkpoints every 20 steps, weights only (user, 2026-11-16; SAVE_OPTIMIZER=true adds the optimizer state); only the
# latest is kept (keep_top_k 1 by recency). A failed driver is restarted in the allocation from the latest checkpoint
# (driver_retry.sh), with a fresh Adam state; a resume job launched with the same RUN and
# SBATCH_MORE=--dependency=afternotok:<job> continues from it on new nodes.
# RUN, JOB_TIME, STALENESS (default 3), SAVE_OPTIMIZER (default false), SBATCH_MORE and DRY_RUN pass through.
set -euo pipefail
cd /home01/hanint06/projects/rl-infra-stem5
export RECIPE=/home01/hanint06/projects/rl-infra-stem5/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n-judge1.yaml
export RUN=${RUN:-sftv02-stem-mix-run5-33n}
# Run 4's clone, unchanged (no NeMo-RL change is needed for chemistry). Other jobs import from it too: refuse to
# launch if its commit has moved.
export NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-stemall
head=$(git -C "$NEMO_RL_DIR" rev-parse --short=9 HEAD)
[[ "$head" == 862d358d6 ]] || { echo "NeMo-RL clone at $head, expected 862d358d6"; exit 1; }
for f in /scratch/hanint06/data/stem-rl/chem_faithful/train.jsonl /scratch/hanint06/data/stem-rl/chem_faithful/validation.jsonl; do
    [[ -s "$f" ]] || { echo "missing $f (envs/prepare_chem_rdkit.py <rows dir> <out> --val-rows 246)"; exit 1; }
done
export JOB_TIME=${JOB_TIME:-24:00:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN} ${SBATCH_MORE:-}"
bash run.sh envs/stem-mix-run5.yaml \
    grpo.num_prompts_per_step=512 policy.train_global_batch_size=8192 \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=${STALENESS:-3} \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=true \
    checkpointing.enabled=true checkpointing.save_period=20 checkpointing.keep_top_k=1 \
    checkpointing.metric_name=null checkpointing.save_optimizer=${SAVE_OPTIMIZER:-false} \
    "$@"

#!/bin/bash
# bash debug16.sh [extra hydra overrides...]
# Pipeline debug of the scienceworld environment on 16 nodes (task-debug-16n.yaml): 128 prompts per step, one
# optimizer step, no log-prob pass, staleness 2, lr 2e-6, PyTorch cap 0.84, compile threads 1, max_num_seqs 256,
# no rollout dump, no checkpoints, no W&B. RUN, JOB_TIME (default 1.5 h) and DRY_RUN pass through.
set -euo pipefail
cd /home01/hanint06/projects/rl-infra-sciworld
export RECIPE=/home01/hanint06/projects/rl-infra-sciworld/experiments/scienceworld/recipe-sftv02-16n.yaml
export RUN=${RUN:-sciworld-debug-16n}
export NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-sciworld
export JOB_TIME=${JOB_TIME:-01:30:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN}"
bash run.sh experiments/scienceworld/task-debug-16n.yaml \
    grpo.num_prompts_per_step=128 policy.train_global_batch_size=2048 \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=2 \
    checkpointing.enabled=false logger.wandb_enabled=false policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=true "$@"

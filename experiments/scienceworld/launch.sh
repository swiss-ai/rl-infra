#!/bin/bash
# bash launch.sh [extra hydra overrides...]
# GRPO of SFT v0.2 on the run-3 STEM mix plus ScienceWorld (TASK, default experts/stem/archive/stem-mix-run3-sciworld.yaml) on
# 16 trainer + 16 vLLM nodes with the 512-prompt variant B of handover/rl-throughput-and-512p-config.md (one optimizer
# step, no log-prob pass, staleness 2, lr 2e-6, PyTorch cap 0.84, compile threads 1, max_num_seqs 256, no rollout
# dump, no checkpoints), from the NeMo-RL copy with the scienceworld environment (branch apertus2-sciworld).
# RUN, JOB_TIME (default 12 h), SBATCH_MORE and DRY_RUN pass through; extra overrides go last, e.g. for a debug run
#   grpo.max_num_steps=4 logger.wandb_enabled=false
set -euo pipefail
cd /home01/hanint06/projects/rl-infra-sciworld
export RECIPE=/home01/hanint06/projects/rl-infra-sciworld/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n.yaml
export RUN=${RUN:-sftv02-stem-mix-sciworld-32n}
export NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-sciworld
export JOB_TIME=${JOB_TIME:-12:00:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN} ${SBATCH_MORE:-}"
DEBUG=0 bash run.sh "${TASK:-experts/stem/archive/stem-mix-run3-sciworld.yaml}" \
    grpo.num_prompts_per_step=512 policy.train_global_batch_size=8192 \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=2 \
    checkpointing.enabled=false policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=true "$@"

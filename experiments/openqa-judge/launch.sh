#!/bin/bash
# bash launch.sh <judge nodes> [extra hydra overrides...]
# A short GRPO run of SFT v0.2 on the run-3 mix plus Nemotron-RL-knowledge-openqa graded by the llm_judge
# environment (envs/stem-mix-openqa-judge.yaml), on 16 trainer + 16 vLLM + <judge nodes> nodes, with the
# 512-prompt setup of handover/rl-throughput-and-512p-config.md (variant B: one optimizer step, no log-prob
# pass, staleness 2) and the NeMo-RL copy that has the llm_judge environment.
# RUN, JOB_TIME, STEPS (default 8) and DRY_RUN pass through. Validation and checkpoints are off.
set -euo pipefail
JUDGE_NODES=$1; shift
cd /home01/hanint06/projects/rl-infra-openqa
export RECIPE=/home01/hanint06/projects/rl-infra-openqa/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n-judge${JUDGE_NODES}.yaml
export RUN=${RUN:-openqa-judge-${JUDGE_NODES}n-debug}
export NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-openqa
export JOB_TIME=${JOB_TIME:-02:00:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN}"
bash run.sh envs/stem-mix-openqa-judge.yaml \
    grpo.num_prompts_per_step=512 policy.train_global_batch_size=8192 \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=2 \
    checkpointing.enabled=false logger.wandb_enabled=false \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=true \
    grpo.max_num_steps=${STEPS:-8} grpo.val_at_start=false grpo.val_period=1000 grpo.val_at_end=false \
    env.judge_pool.num_nodes=${JUDGE_NODES} "$@"

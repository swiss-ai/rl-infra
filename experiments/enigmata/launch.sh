#!/bin/bash
# bash launch.sh [extra hydra overrides...]
# GRPO of SFT v0.2 on the run-3 STEM mix plus Enigmata puzzles (experts/stem/archive/stem-mix-enigmata.yaml) on 16 trainer
# + 16 vLLM nodes, with variant B of handover/rl-throughput-and-512p-config.md (512 prompts x 16 answers,
# one optimizer step, no log-prob pass, staleness 2) and the NeMo-RL copy that has the enigmata environment.
# The overrides are run 3's (job 1272134) apart from the task file and the NeMo-RL copy.
# RUN, JOB_TIME, DRY_RUN and SBATCH_MORE pass through. STEPS caps the steps (default: the task file's 100);
# VAL=0 turns validation off (debug runs). TASK replaces the task file. W&B gets metrics only, as in run 3.
set -euo pipefail
cd /home01/hanint06/projects/rl-infra-enigmata
export RECIPE=/home01/hanint06/projects/rl-infra-enigmata/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n.yaml
export RUN=${RUN:-sftv02-stem-mix-enigmata-32n}
export NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-enigmata
export JOB_TIME=${JOB_TIME:-12:00:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN} ${SBATCH_MORE:-}"
EXTRA=()
if [[ -n "${STEPS:-}" ]]; then
    EXTRA+=("grpo.max_num_steps=${STEPS}")
fi
if [[ "${VAL:-1}" == 0 ]]; then
    EXTRA+=(grpo.val_at_start=false grpo.val_period=1000 grpo.val_at_end=false)
fi
bash run.sh "${TASK:-envs/stem/archive/stem-mix-enigmata.yaml}" \
    grpo.num_prompts_per_step=512 policy.train_global_batch_size=8192 \
    checkpointing.enabled=false \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=2 \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=true \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"

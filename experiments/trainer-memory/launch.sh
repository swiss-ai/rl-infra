#!/bin/bash
# Trainer GPU memory tests on the 16k coding task (envs/code-gen-osb-sft.yaml): 3 steps each, 16 nodes,
# 16 sandbox pods, no validation or checkpoints. Each variant changes one thing; all but "base" train
# with one optimizer step per GRPO step. The runs keep the base config's memory forensics (per-phase
# [gpumem] lines, forensics/mem-rank*.tsv, NCCL INFO/ALLOC logs).
#   bash experiments/trainer-memory/launch.sh <variant>...     (variants: see V below)
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NEMO_RL_DIR:=${HOME}/projects/nemo-rl-apertus2-mem}"
export NEMO_RL_DIR
COMMON=(
    env.nemo_gym.code_gen.resources_servers.code_gen.opensandbox.api_key_file="${HOME}/.opensandbox_token"
    env.nemo_gym.code_gen.resources_servers.code_gen.opensandbox.num_pods=16
    grpo.max_num_steps=3 grpo.val_at_start=false grpo.val_at_end=false
    grpo.async_grpo.max_trajectory_age_steps=2
    checkpointing.enabled=false
)
# Variants named *-ckpt save a checkpoint (with optimizer state, as the real run does) and validate
# after step 2: in job 1213376 memory outside PyTorch grew by up to 4 GB at checkpoint saves.
COMMON_CKPT=("${COMMON[@]/checkpointing.enabled=false/checkpointing.enabled=true}")
COMMON_CKPT+=(grpo.val_period=2 checkpointing.save_period=2)
E=++policy.megatron_cfg.env_vars
G=++policy.generation.vllm_cfg.env_vars
# Env values must be strings: the job's command goes through one shell, which turns "'0'" into the
# Hydra string '0' (a bare 0 is an int and fails validation).
ONE_OPT=policy.train_global_batch_size=2048
declare -A V=(
    [base]=""
    [1opt]="${ONE_OPT}"
    [1opt-split512]="${ONE_OPT} ${E}.PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
    [1opt-pp22-20]="${ONE_OPT} ++policy.megatron_cfg.num_layers_in_first_pipeline_stage=22 ++policy.megatron_cfg.num_layers_in_last_pipeline_stage=20"
    [1opt-bf16grad]="${ONE_OPT} policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=false"
    [1opt-nocumem]="${ONE_OPT} ${E}.NCCL_CUMEM_ENABLE=\"'0'\""
    # Channel limits must match on every rank of a communicator, and the weight-sync one spans trainer
    # and vLLM (job 1221366: "Message truncated" in init_collective with the trainer alone limited).
    [1opt-nccl16]="${ONE_OPT} ${E}.NCCL_MAX_NCHANNELS=\"'16'\" ${E}.NCCL_MAX_P2P_NCHANNELS=\"'16'\" ${G}.NCCL_MAX_NCHANNELS=\"'16'\" ${G}.NCCL_MAX_P2P_NCHANNELS=\"'16'\""
    # NCCL sets up every connection when a communicator is created, not at its first use mid-step.
    # PyTorch's pool capped at 80% (76 GB) by NeMo-RL patch 0033, leaving 19 GB to NCCL, Triton and the
    # CUDA context (17.3 GB outside PyTorch at most in job 1213376); -gc also reclaims cached blocks
    # from 90% of the cap on, instead of all at once when the cap is hit.
    [1opt-cap80]="${ONE_OPT} ${E}.NRL_CUDA_MEMORY_FRACTION=\"'0.80'\""
    [1opt-cap80-gc]="${ONE_OPT} ${E}.NRL_CUDA_MEMORY_FRACTION=\"'0.80'\" ${E}.PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9"
    # 16 NCCL channels keep NCCL at about 10 GB (job 1221412), so the cap can rise to 86% (82 GB),
    # with the 22/20 split for the last stage.
    [1opt-combo]="${ONE_OPT} ${E}.NRL_CUDA_MEMORY_FRACTION=\"'0.86'\" ${E}.NCCL_MAX_NCHANNELS=\"'16'\" ${E}.NCCL_MAX_P2P_NCHANNELS=\"'16'\" ${G}.NCCL_MAX_NCHANNELS=\"'16'\" ${G}.NCCL_MAX_P2P_NCHANNELS=\"'16'\" ++policy.megatron_cfg.num_layers_in_first_pipeline_stage=22 ++policy.megatron_cfg.num_layers_in_last_pipeline_stage=20"
    # PyTorch capped alone, NCCL left at its defaults: 80% (76 GB) and 78% (74 GB, 21 GB left for NCCL,
    # whose largest use was 19.4 GB at a checkpoint save in job 1213376).
    [1opt-cap80-ckpt]="${ONE_OPT} ${E}.NRL_CUDA_MEMORY_FRACTION=\"'0.80'\""
    [1opt-cap78-ckpt]="${ONE_OPT} ${E}.NRL_CUDA_MEMORY_FRACTION=\"'0.78'\""
    # Set on both sides as well: with the trainer alone eager, job 1221367 stalled in init_collective.
    [1opt-eagerconnect]="${ONE_OPT} ${E}.NCCL_RUNTIME_CONNECT=\"'0'\" ${G}.NCCL_RUNTIME_CONNECT=\"'0'\""
)
for v in "$@"; do
    [[ -v "V[${v}]" ]] || { echo "unknown variant ${v}" >&2; exit 1; }
    if [[ "${v}" == *-ckpt ]]; then args=("${COMMON_CKPT[@]}"); else args=("${COMMON[@]}"); fi
    # shellcheck disable=SC2086
    RUN="mem-${v}" JOB_TIME="${JOB_TIME:-02:30:00}" bash run.sh envs/code-gen-osb-sft.yaml "${args[@]}" ${V[${v}]}
done

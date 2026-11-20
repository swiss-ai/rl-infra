#!/bin/bash
# One SFT run of an Apertus-v1.1 base model on one node of four GPUs, inside the container (sft.sh submits it):
#   train.sh <size: 0.5B | 1.5B | 4B> <conversations> [Hydra overrides...]
# First the Python overlay and the converted mixture, made on first use (build_env.sh, prepare_data.py), then training.
#
# The recipe is the final Apertus SFT recipe (posttraining: reproducibility-scripts/sft/final/ademamix.py): AdEMAMix,
# one epoch, global batch 512, loss on assistant tokens only (plw 0), the apertus chat template, EOS 68, left padding,
# DeepSpeed ZeRO-2, 4,096 tokens. As in the Apertus-v1.1 report, only the learning rate changes: it is the model's
# learning rate at the end of pre-training (min-lr of the distillation runs, swiss-ai/Megatron-LM-Distill, branch
# 8b_distill). The conversations are a random subset of the mixture, drawn with seed 42.
set -euo pipefail
SIZE=$1; N=$2; shift 2
# Per size: the learning rate, and gradient checkpointing (GC). GC and the sequences per GPU (DEVICE_BS) change only
# speed and memory, as the accumulation steps follow from the global batch; the defaults are the fastest settings that
# fit on the 95 GB GPUs of Alps (backends/posttraining/README.md).
case "${SIZE}" in
    0.5B) DEFAULT_LR=6e-5; DEFAULT_GC=false ;;
    1.5B) DEFAULT_LR=3e-5; DEFAULT_GC=true ;;
    4B)   DEFAULT_LR=2e-5; DEFAULT_GC=true ;;
    *) echo "ERROR: size must be 0.5B, 1.5B or 4B" >&2; exit 1 ;;
esac
LR="${LR:-${DEFAULT_LR}}" GC="${GC:-${DEFAULT_GC}}"
GBS=512 GPUS=4 DEVICE_BS="${DEVICE_BS:-4}"
ACC=$(( GBS / (GPUS * DEVICE_BS) ))
NAME="${RUN_NAME:-apertus-v11-${SIZE}-sft-n${N}}"

bash "${PT_DIR}/build_env.sh"
python3 "${PT_DIR}/prepare_data.py" "${HF_MIXTURE_DIR}" "${SFT_ROOT}/data/apertus-sft-mixture"

# RESUME=1 (long runs, preemptible partitions): full checkpoints with optimizer state every SAVE_STEPS optimizer steps
# (the last two kept), in a folder named after the configuration, and a job that starts again (for example requeued
# after preemption) continues from the last one. Otherwise only the final model is saved.
if [[ "${RESUME:-0}" == 1 ]]; then
    SAVE_ARGS=(resuming.resume=true training_args.save_steps="${SAVE_STEPS:-200}" training_args.save_total_limit=2)
    CKPT_GLOB=checkpoints/*/checkpoint-*
    # A resumed Trainer takes the interval from the checkpoint's trainer_state.json, not from its arguments:
    # a SAVE_STEPS given here is written there too.
    if [[ -n "${SAVE_STEPS:-}" ]]; then
        for state in "${SFT_ROOT}"/artifacts/private/outputs/train_sft/"${NAME}"/${CKPT_GLOB}/trainer_state.json; do
            if [[ -f "${state}" ]]; then sed -i "s/\"save_steps\": [0-9]*/\"save_steps\": ${SAVE_STEPS}/" "${state}"; fi
        done
    fi
else
    SAVE_ARGS=(+training_args.save_only_model=true)
    CKPT_GLOB=runs/*/checkpoint-*
fi

# W&B: the key file is read here, never put on a command line.
if [[ -r "${WANDB_KEY_FILE:-}" ]]; then
    WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"; export WANDB_API_KEY
fi
export WANDB_ENTITY="${WANDB_ENTITY:-apertus}"

cd "${POSTTRAINING_DIR}"
accelerate launch --config-file src/post_training/configs/accelerate/ds-zero2.yaml \
    --num_machines 1 --num_processes "${GPUS}" --main_process_ip 127.0.0.1 --machine_rank 0 \
    -m post_training.train_sft \
    model=apertus-v11 dataset=apertus-sft-mixture-hf \
    model_args.model_name_or_path="${MODELS_DIR}/Apertus-v1.1-${SIZE}" \
    dataset_args.dataset_name="${SFT_ROOT}/data/apertus-sft-mixture" \
    dataset_args.debug_subsample.train="${N}" \
    trainer=plw plw_args.prompt_loss_weight=0.0 \
    training_args.optim=ademamix training_args.learning_rate="${LR}" training_args.max_grad_norm=1.0 \
    training_args.num_train_epochs=1 training_args.max_seq_length=4096 \
    training_args.per_device_train_batch_size="${DEVICE_BS}" training_args.gradient_accumulation_steps="${ACC}" \
    training_args.gradient_checkpointing="${GC}" \
    "${SAVE_ARGS[@]}" \
    artifacts_dir="${SFT_ROOT}/artifacts" job_subdir="${NAME}" \
    wandb.project="${WANDB_PROJECT:-apertus-v11}" wandb.run_name="${NAME}" \
    global_batch_size="${GBS}" num_nodes=1 seed=42 "$@"

# The final checkpoint (an HF model folder with the tokenizer and chat template) under a fixed name:
# $SFT_ROOT/models/<run name>, which GRPO takes as policy.model_name.
last="$(ls -d "${SFT_ROOT}"/artifacts/private/outputs/train_sft/"${NAME}"/${CKPT_GLOB} 2>/dev/null \
    | sort -V | tail -n 1)"
[[ -n "${last}" ]] || { echo "ERROR: no checkpoint found for ${NAME}" >&2; exit 1; }
mkdir -p "${SFT_ROOT}/models"
ln -sfn "${last}" "${SFT_ROOT}/models/${NAME}"
echo "== final checkpoint: ${SFT_ROOT}/models/${NAME} -> ${last}"

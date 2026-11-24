#!/bin/bash
# The 200-step GRPO run of Chonk SFT v0.2 on the Lean task (RUNBOOK.md, "Next runs" and "The 200-step run"):
# 12 nodes (8 Megatron trainers + 4 vLLM), Kimina inside the job, 128 problems x 8 rollouts per step.
#   DRY_RUN=1 bash backends/nemo-rl/lean-kimina/run-200.sh   setup, preflight and the sbatch command; submits nothing
#   bash backends/nemo-rl/lean-kimina/run-200.sh             submits. Run it again with the same RUN after the job
#                                                            ends: the new job resumes from the run's latest checkpoint
# Settings (environment):
#   RUN              default lean-kimina-v02-grpo200-128x8 (no date in it, so that a later job resumes the same run);
#                    a job whose RUN has checkpoints resumes from the latest one
#   PARTITION        default preemptable (requeued after a preemption; at most 1 day per job)
#   JOB_TIME         default 24:00:00; train.sh saves a checkpoint and stops 30 min before it
#   ROWS             default $LEAN_DATA/goedel-full-5c99dce/goedel-clean.v434.jsonl (37,953 rows, checked below)
#   FT_SAVE_PERIOD   default 1: a recovery checkpoint after every step, of which only the newest is kept, so a
#                    preemption or crash loses at most the step in progress. A save takes 39-55 s (1.4 TB) and runs
#                    while vLLM generates the next batch, which the trainers wait for anyway (job 4008441: about 520 s
#                    per step). 10 was the setting until 2026-11-24.
#   TRITON_CACHE     default $SCRATCH/lean/jit-cache/triton: Triton's kernel cache, with autotuning results
#                    (TRITON_CACHE_AUTOTUNING=1), for the trainers and vLLM, kept across jobs. In node-local /tmp (the
#                    EDF's default) every job compiles and tunes again: step 1 took 282 s of training and 180 s of
#                    log-probs, later steps about 140 and 40 s (job 4008441). Empty: the EDF's /tmp.
#   EXTRA_OVERRIDES  more overrides, space-separated, appended last (they win)
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
D="${LEAN_DATA}/data"
mkdir -p "${D}"

check() {  # check <file> <sha256>
    [[ -s "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }
    echo "$2  $1" | sha256sum -c --quiet || { echo "ERROR: $1 is not the expected file" >&2; exit 1; }
}
# Training rows: all Goedel RL V2 completion statements that elaborate on Lean v4.34.0, without chonk-tests'
# evaluation sample and without near-duplicates of corrected miniF2F (RUNBOOK.md, "The larger row set").
ROWS="${ROWS:-${LEAN_DATA}/goedel-full-5c99dce/goedel-clean.v434.jsonl}"
[[ -n "${ROWS_SHA256:-}" ]] || ROWS_SHA256=95c620a7b3ab21858c7d2ab124d0e47e5b65079091dcfbc8fd9c2f2fb2ecde3e
check "${ROWS}" "${ROWS_SHA256}"
# Validation: corrected miniF2F, valid 244 + test 242, each twice (972 rows), as in the ablations.
VAL="${D}/minif2f-corrected-x2.v434.jsonl"
if [[ ! -s "${VAL}" ]]; then
    gunzip -c "${LEAN_KIMINA_DIR}/data/minif2f-corrected-x2.v434.jsonl.gz" > "${VAL}.part" && mv "${VAL}.part" "${VAL}"
fi
check "${VAL}" f8e04393118ba0236cb7cd33dc5b44d4a65df1e5f6f15e94c674bf3017964465

overrides=(
    # Data
    data.train.data_path="${ROWS}"
    data.validation.data_path="${VAL}"
    # Rollouts: 128 problems x 8 samples = 1,024 per step, one optimizer update per step
    grpo.num_prompts_per_step=128
    grpo.num_generations_per_prompt=8
    policy.train_global_batch_size=1024
    grpo.max_num_steps=200
    # Sampling and proof check, as in every run since 2026-11-17
    policy.generation.temperature=1.0
    policy.generation.top_p=1.0
    policy.generation.max_new_tokens=16000
    env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.compilation_timeout=120
    # Optimizer: AdamW, learning rate 1e-5 cosine-decayed to 1e-6 over the 200 updates, epsilon 1e-14
    policy.megatron_cfg.optimizer.lr=1.0e-5
    policy.megatron_cfg.optimizer.min_lr=1.0e-6
    policy.megatron_cfg.scheduler.lr_decay_style=cosine
    policy.megatron_cfg.scheduler.lr_decay_iters=200
    policy.megatron_cfg.scheduler.lr_warmup_iters=0
    policy.megatron_cfg.optimizer.adam_eps=1.0e-14
    # Mask a sequence whose train/inference probability error exceeds 1.5 (the step-40 outlier of the eps ablation)
    grpo.seq_logprob_error_threshold=1.5
    # Trainer GPU memory: without it 128 x 8 ran out of GPU memory from fragmentation (job 3984550)
    ++policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # Validation every 20 steps (each about 15-26 min at this load); none at the start (the baseline is known)
    grpo.val_period=20
    grpo.val_at_start=false
    # Checkpoints (1.4 TB each): every 20 steps, all kept (one per validation, for picking a checkpoint on valid);
    # plus a recovery checkpoint every FT_SAVE_PERIOD steps of which only the newest is kept
    checkpointing.save_period=20
    checkpointing.keep_top_k=null
    ++checkpointing.ft_save_period="${FT_SAVE_PERIOD:-1}"
    ++checkpointing.ft_keep_latest_k=1
)
TRITON_CACHE="${TRITON_CACHE-${LEAN_DATA}/jit-cache/triton}"
if [[ -n "${TRITON_CACHE}" ]]; then
    mkdir -p "${TRITON_CACHE}"
    # '"1"': the driver command is parsed by a shell once more at job start, and Hydra must get "1" quoted, a
    # string: Ray takes only strings as environment values, and the Megatron workers' env_vars reach it unconverted.
    for w in policy.megatron_cfg.env_vars policy.generation.vllm_cfg.env_vars; do
        overrides+=("++${w}.TRITON_CACHE_DIR=${TRITON_CACHE}" "++${w}.TRITON_CACHE_AUTOTUNING='\"1\"'")
    done
fi
read -r -a more <<< "${EXTRA_OVERRIDES:-}"

cd "${ROOT}"
export RUN="${RUN:-lean-kimina-v02-grpo200-128x8}" PARTITION="${PARTITION:-preemptable}" \
    JOB_TIME="${JOB_TIME:-24:00:00}" RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-12n.yaml HEADER_ROWS="${ROWS}"
exec bash "${LEAN_KIMINA_DIR}/train.sh" "${overrides[@]}" ${more[@]+"${more[@]}"}

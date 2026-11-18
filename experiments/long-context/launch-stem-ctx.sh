#!/bin/bash
# bash launch-stem-ctx.sh <context tokens> [extra hydra overrides...]
# The run-4 STEM mix (envs/stem-mix-run4.yaml: 512 prompts x 16, one optimizer step, no log-prob pass,
# lr 2e-6, staleness 3, same overrides as experiments/stem-mix-run4/launch.sh) with a longer context and a
# context-parallel trainer. Prompt plus answer, vLLM's limit and the trainer's packed micro-batch are <context tokens>.
#
#   CP           context-parallel size of the trainer (default 1): each GPU holds <context tokens>/CP tokens of a
#                micro-batch. The trainer stays PP2 x EP16; the ranks of a pipeline stage are CP x data-parallel.
#   KCP          1: the KCP scheme for the KDA layers (tokens stay split over the CP ranks; Megatron copy
#                ~/projects/megatron-lm-moe-apertus2-kcp, branch rl-kcp, with flash-linear-attention 0.5.2 from
#                fla-0.5.2/ ahead of the image's). Any CP that divides the ranks of a stage.
#                0 (default): the pinned Megatron's scheme, heads split over the CP ranks; CP 1, 2 or 4 only.
#   MIX          run4 (default): envs/stem-mix-run4.yaml, all seven environments, 1 judge node, staleness 3.
#                run3: envs/stem-mix-run3.yaml (per step DAPO-Math 128, NuminaMath 256, multiple-choice science 128;
#                validation AIME 2024 x32 and GPQA-Diamond x8), no judge node, staleness 2, as run 3.
#                run5: envs/stem-mix-run5.yaml, run 4 plus chemistry (per step DAPO, NuminaMath,
#                multiple-choice science and openqa 96 each; Reasoning Gym, Enigmata, ScienceWorld and chemistry 32
#                each), 1 judge node, staleness 3.
#   STALENESS    grpo.async_grpo.max_trajectory_age_steps; default as the mix's run.
#   TRAIN_NODES  8 or 16 trainer nodes (default 16); 16 vLLM nodes in both cases (plus the judge node for run4).
#   GEN_NODES    vLLM nodes: 16 (default) or, for run5 with 16 trainer nodes, 32 (49 nodes in all).
#   NEMO_RL_DIR  the NeMo-RL clone (default ${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2, set up by setup.sh).
#   KEEP         checkpoints kept (checkpointing.keep_top_k, default 1 = the latest only).
#   NCCL_MIN     minutes a collective of the trainer waits for its peers (default 10).
#   GEN_CTX      limit of prompt plus answer for generation when it should be shorter than the trainer's micro-batch
#                (debugging the trainer without waiting for long answers). Default: <context tokens>.
#   CAP          PyTorch's share of each trainer GPU (NRL_CUDA_MEMORY_FRACTION); default 0.84 as run 4.
#   STEPS        default 3.   DUMP=1 writes each step's rollouts (train_data_step<N>.jsonl: text, rewards and the
#                per-token arrays, about 6 GB per step at 16K); DUMP=text the same without the per-token arrays.
#   VAL=1        validation as runs 3 and 4 (start, every 20 steps, end); default none.
#   CKPT         1: checkpoints as run 4 (every 20 steps, latest only, with optimizer state, about 1.4 TB);
#                weights: the same without optimizer state; default none (as run 3).
#   SAVE_BY      checkpointing.checkpoint_must_save_by (D:HH:MM:SS): the run saves and stops then; the recipes say
#                00:23:30:00, meant for 24 h jobs. Set it with JOB_TIME for a longer job.
#   RUN, JOB_TIME (default 02:30:00), SBATCH_MORE and DRY_RUN pass through.
#
# vLLM refuses a limit above the export's max_position_embeddings (16,384, the SFT sequence length) unless
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set; the model has no position table, so the value is only that check.
set -euo pipefail
SCRATCH=${SCRATCH:-/scratch/${USER}}   # generated pin, container spec and task file go to ${SCRATCH}/long-context
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$(cd "${HERE}/../.." && pwd)"   # the rl-infra worktree
CTX=${1:?context length in tokens, e.g. 65536}
shift
CP=${CP:-1}
KCP=${KCP:-0}
TRAIN_NODES=${TRAIN_NODES:-16}
GEN_CTX=${GEN_CTX:-${CTX}}
# Every packed sequence is padded to a multiple of 2 x CP (NeMo-RL checks it only at the first training step).
PAD=$(( CP > 1 ? 2 * CP : 2 ))
(( CTX % PAD == 0 )) || { echo "ERROR: ${CTX} is not a multiple of 2 x CP (${PAD})" >&2; exit 1; }

MIX=${MIX:-run4}
GEN_NODES=${GEN_NODES:-16}
case "${MIX}:${TRAIN_NODES}:${GEN_NODES}" in
    run4:16:16|run5:16:16) export RECIPE=${INFRA}/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n-judge1.yaml ;;
    run5:16:32) export RECIPE=${HERE}/recipes/sftv02-stem-16t-32g-judge1.yaml ;;
    run4:8:16)  export RECIPE=${HERE}/recipes/sftv02-stem-8t-16g-judge1.yaml ;;
    run3:16:16) export RECIPE=${INFRA}/recipes/apertus2-chonk-sftv02-18315-vllm-adamw-dapo-32n.yaml ;;
    run3:8:16)  export RECIPE=${HERE}/recipes/sftv02-stem-8t-16g.yaml ;;
    *)  echo "ERROR: no recipe for MIX=${MIX} TRAIN_NODES=${TRAIN_NODES} GEN_NODES=${GEN_NODES}" >&2; exit 1 ;;
esac
case "${MIX}" in
    run4) TASK=envs/stem-mix-run4.yaml; STALENESS=${STALENESS:-3} ;;
    run5) TASK=envs/stem-mix-run5.yaml; STALENESS=${STALENESS:-3} ;;
    run3) TASK=envs/stem-mix-run3.yaml; STALENESS=${STALENESS:-2} ;;
esac
RANKS=$(( TRAIN_NODES * 4 / 2 ))   # ranks of one pipeline stage (PP2)
(( RANKS % CP == 0 )) || { echo "ERROR: CP ${CP} does not divide the ${RANKS} ranks of a pipeline stage" >&2; exit 1; }

EXTRA=()
if [[ "${KCP}" == 1 ]]; then
    # The Megatron copy with the KCP path, at its own commit: setup.sh checks out the pin's commit otherwise.
    export MEGATRON_DIR=${KCP_MEGATRON_DIR:-${PROJECTS_DIR:-${HOME}/projects}/megatron-lm-moe-apertus2-kcp}
    KCP_COMMIT=$(git -C "${MEGATRON_DIR}" rev-parse rl-kcp)
    mkdir -p ${SCRATCH}/long-context/launch
    export PIN_FILE=${SCRATCH}/long-context/launch/pin-kcp
    sed "s/^MEGATRON_COMMIT=.*/MEGATRON_COMMIT=${KCP_COMMIT}/" "${INFRA}/backends/nemo-rl/apertus2/pin" > "${PIN_FILE}"
    # The stack's container spec with flash-linear-attention 0.5.2 (fla.ops.cp) ahead of the image's copy.
    export CONTAINER_SPEC=${SCRATCH}/long-context/launch/edf-kcp.toml.template
    sed "s|:@@MEGATRON_DIR@@:|:@@MEGATRON_DIR@@:${KCP_FLA_DIR:-${PROJECTS_DIR:-${HOME}/projects}/fla-0.5.2}:|" \
        "${INFRA}/backends/nemo-rl/apertus2/edf.toml.template" > "${CONTAINER_SPEC}"
    grep -q "fla-0.5.2" "${CONTAINER_SPEC}" || { echo "ERROR: could not add fla-0.5.2 to PYTHONPATH" >&2; exit 1; }
    EXTRA+=(++policy.megatron_cfg.model_overrides.linear_attention_cp_impl=kcp)
elif (( CP != 1 && CP != 2 && CP != 4 )); then
    echo "ERROR: without KCP=1 the KDA layers split their 28 heads over the CP ranks: CP must be 1, 2 or 4" >&2; exit 1
fi
if [[ -n "${CAP:-}" ]]; then
    EXTRA+=("++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'${CAP}'\"")
else
    EXTRA+=("++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'0.84'\"")
fi
case "${DUMP:-0}" in
    1)    NO_DUMP=false ;;
    text) NO_DUMP=false; EXTRA+=(++env.train_dump_text_only=true) ;;
    *)    NO_DUMP=true ;;
esac
if [[ "${VAL:-0}" != 1 ]]; then EXTRA+=(grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=1000); fi
case "${CKPT:-0}" in
    1)       EXTRA+=(checkpointing.enabled=true checkpointing.save_period=20 checkpointing.keep_top_k=${KEEP:-1}
                     checkpointing.metric_name=null checkpointing.save_optimizer=true) ;;
    weights) EXTRA+=(checkpointing.enabled=true checkpointing.save_period=20 checkpointing.keep_top_k=${KEEP:-1}
                     checkpointing.metric_name=null checkpointing.save_optimizer=false) ;;
    *)       EXTRA+=(checkpointing.enabled=false) ;;
esac
# NCCL_MIN: minutes a collective of the trainer waits for its peers (default 10); needs NeMo-RL patch 0040.
if [[ -n "${NCCL_MIN:-}" ]]; then
    EXTRA+=("++policy.megatron_cfg.env_vars.NRL_NCCL_TIMEOUT_MINUTES=\"'${NCCL_MIN}'\"")
fi
if [[ -n "${SAVE_BY:-}" ]]; then EXTRA+=("checkpointing.checkpoint_must_save_by=\"'${SAVE_BY}'\""); fi

# Every run logs to W&B (entity apertus, project chonk-rl). The key is in ~/.wandb-api-key (hyphens: the user's
# file; rl-infra's default name is ~/.wandb_api_key, so WANDB_KEY_FILE names it) and is read at job start.
# DEBUG=1 (default) marks a debug run: its name ends with -DEBUG. DEBUG=0 is a full run and needs the key.
# launch.sh only checks that the key file is readable; an empty one would fail inside the job.
DEBUG=${DEBUG:-1}
WANDB_KEY=${WANDB_KEY_FILE:-${HOME}/.wandb_api_key}
if [[ -s "${WANDB_KEY}" ]]; then
    export WANDB_PROJECT=chonk-rl WANDB_ENTITY=apertus WANDB_KEY_FILE="${WANDB_KEY}"
elif [[ "${DEBUG}" == 0 ]]; then
    echo "ERROR: a full run must log to W&B, but ~/.wandb-api-key is missing or empty" >&2; exit 1
else
    echo "WARNING: ~/.wandb-api-key is missing or empty: this debug run logs to TensorBoard only" >&2
fi
export RUN=${RUN:-sftv02-stem-${MIX}-$(( CTX / 1024 ))k-cp${CP}-${TRAIN_NODES}t}
if [[ "${DEBUG}" == 1 && "${RUN}" != *-DEBUG ]]; then export RUN="${RUN}-DEBUG"; fi
# The NeMo-RL clone that setup.sh made. Slingshot (ray.sub), NRL_NCCL_TIMEOUT_MINUTES and env.train_dump_text_only
# (DUMP=text) are NeMo-RL patches 0041, 0040 and 0043; the checks below stop a clone that lacks them.
export NEMO_RL_DIR=${NEMO_RL_DIR:-${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2}
if [[ "${DUMP:-0}" == text ]] && ! grep -q train_dump_text_only "${NEMO_RL_DIR}/nemo_rl/algorithms/grpo.py"; then
    echo "ERROR: DUMP=text needs env.train_dump_text_only, which ${NEMO_RL_DIR} does not have" >&2; exit 1
fi
if [[ -n "${NCCL_MIN:-}" ]] && ! grep -q NRL_NCCL_TIMEOUT_MINUTES "${NEMO_RL_DIR}/nemo_rl/models/megatron/setup.py"; then
    echo "ERROR: NCCL_MIN needs NRL_NCCL_TIMEOUT_MINUTES, which ${NEMO_RL_DIR} does not have" >&2; exit 1
fi
# HSN=1 (default, the rule for every run): Ray on the first Slingshot interface (400 Gbit) of each node instead of
# the address the node name resolves to (1 Gbit management Ethernet, 36 times slower for the batch that reaches the
# trainer nodes through Ray). ray.sub reads RAY_NODE_NAME_SUFFIX from NeMo-RL patch 0041 on; an older ray.sub ignores it without a word, hence the check.
if [[ "${HSN:-1}" == 1 ]]; then
    grep -q RAY_NODE_NAME_SUFFIX "${NEMO_RL_DIR}/ray.sub" || {
        echo "ERROR: ${NEMO_RL_DIR}/ray.sub does not read RAY_NODE_NAME_SUFFIX: rerun setup.sh (NeMo-RL patch 0041) (HSN=0 runs on the slow network)" >&2; exit 1; }
    export RAY_NODE_NAME_SUFFIX=-hsn0
    echo "Ray on Slingshot: RAY_NODE_NAME_SUFFIX=${RAY_NODE_NAME_SUFFIX}"
else
    unset RAY_NODE_NAME_SUFFIX
    echo "WARNING: HSN=0, Ray on the 1 Gbit management Ethernet" >&2
fi
export JOB_TIME=${JOB_TIME:-02:30:00}
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN} ${SBATCH_MORE:-}"
cd "${INFRA}"
bash run.sh "${TASK}" \
    grpo.num_prompts_per_step=512 policy.train_global_batch_size=8192 \
    loss_fn.force_on_policy_ratio=true grpo.async_grpo.max_trajectory_age_steps=${STALENESS} \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 ++env.should_log_nemo_gym_responses=${NO_DUMP} \
    policy.max_total_sequence_length=${GEN_CTX} policy.generation.max_new_tokens=${GEN_CTX} \
    policy.sequence_packing.train_mb_tokens=${CTX} policy.sequence_packing.logprob_mb_tokens=${CTX} \
    policy.megatron_cfg.context_parallel_size=${CP} policy.make_sequence_length_divisible_by=${PAD} \
    "++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"" \
    grpo.max_num_steps=${STEPS:-3} \
    "${EXTRA[@]}" "$@"

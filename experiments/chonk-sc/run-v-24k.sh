#!/bin/bash
# Run V at 24k context: run-v.sh with policy.max_total_sequence_length=CTX (default 24576) instead of 16384.
# Why: in job 5356101 (16k) 53-94% of rollouts per step overflowed the context, and Terminus-2 (summarization
# off) ends the episode on overflow, so most SWE trials never finished.
# The Chonk SFT checkpoint has no RoPE (config.json no_rope_layers all 0; swiss-ai/vllm apertus2.py use_rope only
# where no_rope_layers[i] == 1) and was trained at 16384 (max_position_embeddings 16384, "lc-16k"): longer
# context is out of distribution, not impossible. Compare with 5356101: overflow rate
# (train/harbor_agent/context_length_exceeded_error/mean), reward per task family, token_mult_prob_error and
# gen_kl_error (first sign of the model misbehaving past 16k), trainer memory.
# What changes:
# - policy.max_total_sequence_length: vLLM max_model_len and Gym's max_input_tokens follow it
#   (envs/harbor-verified-sft.yaml)
# - sequence_packing train_mb_tokens / logprob_mb_tokens = CTX: a packed microbatch must hold the longest
#   sequence (16384 in the DAPO recipe); 1.5x the activations of a 16k microbatch (16k peak: 74.8 GB reserved,
#   ~7.5 GB free per GPU, job 5302914), so the first training step is the memory test
# - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for the vLLM workers (vllm_cfg.env_vars reach every vLLM worker,
#   nemo_rl vllm_generation.py): vLLM refuses max_model_len > max_position_embeddings otherwise. Not
#   policy.hf_config_overrides.max_position_embeddings: that also changes the Megatron conversion dir
#   (__hfovr_<hash>) and would convert the 202 GB checkpoint again.
# max_new_tokens stays 6144 per turn. vllm_kwargs.max_num_batched_tokens stays 16384 (chunked prefill).
# Job 5358015 (24576 on 12 nodes = 8 trainer nodes, EP16) ran out of GPU memory in the backward pass of step 5:
# 9.11 GiB asked, 8.17 GiB free, 13.45 GiB held by PyTorch's cache but in pieces too small (fragmentation); peak
# allocated 69.7 GB per GPU (16k: 59.9), lowest free 0.1 GB. Hence by default:
# - RECIPE recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools-20n.yaml: 20 nodes, 16 trainer nodes with EP32,
#   half the experts per GPU. RECIPE=recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml for 12 nodes.
# - ALLOC=1 (default 0): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the trainer workers only
#   (policy.megatron_cfg.env_vars; vLLM keeps its own allocator), so freed blocks can be merged.
#   bash experiments/chonk-sc/run-v-24k.sh [overrides...]     same env as run-v.sh; CTX, STEPS (20), JOB_TIME
#   (08:00:00), RECIPE, ALLOC
#
# Long context (CTX > 24576; plan research/reports/2026-11-20-long-context-agent-runs-plan.md, user decisions
# 2026-11-20): CTX=65536 selects common.sh's long-context profile (long_ctx_defaults, every knob overridable):
# SFT v0.2 iter_0018315 (recipes/apertus2-chonk-sftv02-alps-vllm-adamw-tools-20n.yaml, 20 nodes), thinking high
# (envs/apertus2-system-high.json, no budget), trainer CP8 with KCP (main's 64K layout; data-parallel size 4) and
# make_sequence_length_divisible_by 16, NRL_CUDA_MEMORY_FRACTION 0.74, NRL_NCCL_TIMEOUT_MINUTES 60, 16384 tokens per
# turn, 100 turns, agent cap 3000 s, TTL 4800 s, harbor_mask_agent_timeout, hang watchdog 3600/14400 s; STEPS 10,
# JOB_TIME 12:00:00; SMOKE=1: 1 step, 4 x 8, JOB_TIME 03:00:00 (Daint's limit for normal: UNVERIFIED). KCP needs
# experiments/long-context/setup-kcp.sh once (README.md, "Long context"). The job reads a copy of the recipe made now.
#   SMOKE=1 CTX=65536 bash experiments/chonk-sc/run-v-24k.sh      then      CTX=65536 bash experiments/chonk-sc/run-v-24k.sh
# Up to 24576 nothing of this applies unless a knob is set (e.g. THINKING=high, SFT_CKPT=v0.2).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${HERE}/common.sh"
long_ctx_defaults || exit 1
if [[ "${LONG_CTX}" == 1 && "${SMOKE:-0}" == 1 ]]; then
    : "${STEPS:=1}" "${PROMPTS:=4}" "${JOB_TIME:=03:00:00}"
elif [[ "${LONG_CTX}" == 1 ]]; then
    : "${STEPS:=10}" "${JOB_TIME:=12:00:00}"
fi
STEPS="${STEPS:-20}"
# run-v.sh reads PROMPTS (default 8)
[[ -z "${PROMPTS:-}" ]] || export PROMPTS
if [[ "${SFT_CKPT}" == v0.1 ]]; then
    export RUN="${RUN:-sft19374-harbor-verified-ctx${CTX}-$(date +%Y%m%d-%H%M%S)}"
else
    export RUN="${RUN:-sftv02-harbor-verified-ctx${CTX}-$(date +%Y%m%d-%H%M%S)}"
fi
export JOB_TIME="${JOB_TIME:-08:00:00}"
if [[ -z "${RECIPE:-}" ]]; then
    # the 20-node recipes: 16 trainer nodes, PP2, so 32 ranks per pipeline stage, data-parallel size 32 / CP
    (( 32 % CP == 0 )) || { echo "ERROR: CP ${CP} does not divide the 32 ranks of a pipeline stage" >&2; exit 1; }
    (( (${PROMPTS:-8} * 8) % (32 / CP) == 0 )) || {
        echo "ERROR: PROMPTS=${PROMPTS:-8} gives a batch of $(( ${PROMPTS:-8} * 8 )), not a multiple of the trainer's data-parallel size $(( 32 / CP ))" >&2
        exit 1; }
    if [[ "${SFT_CKPT}" == v0.1 ]]; then
        RECIPE="${HERE}/../../recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools-20n.yaml"
    else
        RECIPE="${SFT_RECIPE_20N}"
    fi
fi
EXTRA=()
[[ "${ALLOC:-0}" == 1 ]] && EXTRA+=(++policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
long_ctx_args
EXTRA+=(${LONG_CTX_ARGS[@]+"${LONG_CTX_ARGS[@]}"})
if [[ "${LONG_CTX}" == 1 ]]; then
    RECIPE="$(recipe_copy "${RECIPE}" "${RUN}")" || exit 1
fi
export RECIPE
# KCP: the Megatron checkout with the KCP path, fla 0.5.2 and an EDF of their own (run.sh -> setup.sh, launch.sh)
if [[ "${KCP}" == 1 ]]; then kcp_setup "${EDF_NAME}" || exit 1; fi
long_ctx_job_env
echo "== run V at ${CTX} tokens, ${STEPS} steps, ALLOC=${ALLOC:-0}"
[[ "${LONG_CTX}" != 1 ]] || echo "== long context: SFT ${SFT_CKPT}, thinking ${THINKING}, CP ${CP} KCP ${KCP}, cap ${CAP:-none}," \
     "${MAX_NEW:-6144} tokens per turn, ${MAX_TURNS:-30} turns, agent cap ${AGENT_CAP:-1800} s, TTL ${TTL:-3600} s," \
     "mask agent timeout ${MASK_AGENT_TIMEOUT}, mask overflow ${MASK_OVERFLOW}, JOB_TIME ${JOB_TIME}, recipe copy ${RECIPE}"
exec bash "${HERE}/run-v.sh" \
    grpo.max_num_steps="${STEPS}" \
    policy.max_total_sequence_length="${CTX}" \
    policy.sequence_packing.train_mb_tokens="${CTX}" \
    policy.sequence_packing.logprob_mb_tokens="${CTX}" \
    ++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"\'1\'\" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"

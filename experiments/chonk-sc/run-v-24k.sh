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
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CTX="${CTX:-24576}"
STEPS="${STEPS:-20}"
export RUN="${RUN:-sft19374-harbor-verified-ctx${CTX}-$(date +%Y%m%d-%H%M%S)}"
export JOB_TIME="${JOB_TIME:-08:00:00}"
export RECIPE="${RECIPE:-${HERE}/../../recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools-20n.yaml}"
EXTRA=()
[[ "${ALLOC:-0}" == 1 ]] && EXTRA+=(++policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
echo "== run V at ${CTX} tokens, ${STEPS} steps, ALLOC=${ALLOC:-0}"
exec bash "${HERE}/run-v.sh" \
    grpo.max_num_steps="${STEPS}" \
    policy.max_total_sequence_length="${CTX}" \
    policy.sequence_packing.train_mb_tokens="${CTX}" \
    policy.sequence_packing.logprob_mb_tokens="${CTX}" \
    ++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"\'1\'\" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"

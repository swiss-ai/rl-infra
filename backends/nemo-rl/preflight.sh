#!/bin/bash
# Checks a NeMo-RL config before submission, on the login node (yq v3, no Python):
# parses every file of its `defaults` chain, merges them with the command-line
# overrides, prints the settings that matter for a run and fails on known mistakes.
# Interpolations (${...}) stay unresolved. launch.sh runs it before sbatch.
#
# Usage: preflight.sh <config> <num_nodes> [overrides...]   (env: NEMO_RL_DIR, JOB_TIME)
set -uo pipefail
CONFIG=$1; NODES=$2; shift 2
TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT
errors=0; warnings=0
err()  { echo "  ERROR: $*"; errors=$((errors + 1)); }
warn() { echo "  WARNING: $*"; warnings=$((warnings + 1)); }

# Container paths of upstream recipes map to the NeMo-RL clone.
host_path() { [[ "$1" == /opt/nemo-rl/* ]] && echo "${NEMO_RL_DIR}/${1#/opt/nemo-rl/}" || echo "$1"; }

# Resolve `defaults` as NeMo-RL does: each parent is resolved on its own, the parents are
# merged in list order (later wins), then the file itself goes on top. Entries are relative
# to their file's directory.
chain=()
resolve() {
  local f=$1 out=$2 d p base part
  [[ -f "$f" ]] || { echo "ERROR: config file not found: $f"; exit 1; }
  if ! yq r "$f" >/dev/null 2>"${TMP}/err"; then
    echo "ERROR: $f does not parse: $(cat "${TMP}/err")"; exit 1
  fi
  chain+=("$f")
  base="$(mktemp -p "${TMP}")"; echo "{}" > "${base}"
  for d in $(yq r "$f" defaults 2>/dev/null | sed 's/^- //'); do
    [[ "$d" == /* ]] && p="$(host_path "$d")" || p="$(dirname "$f")/$d"
    part="$(mktemp -p "${TMP}")"
    resolve "$p" "${part}"
    yq m -x -a overwrite "${base}" "${part}" > "${base}.m" && mv "${base}.m" "${base}"
  done
  yq m -x -a overwrite "${base}" "$f" > "${out}"
}
merged="${TMP}/merged.yaml"
resolve "$(host_path "${CONFIG}")" "${merged}"
new_keys=()
for o in "$@"; do
  [[ "$o" == *=* ]] || continue
  key="${o%%=*}"
  if [[ "${key}" == +* ]]; then
    key="${key#+}"
  elif [[ -z "$(yq r "${merged}" "${key}")" ]]; then
    new_keys+=("${key}")
  fi
  yq w -i "${merged}" "${key}" "${o#*=}"
done
r() { yq r "${merged}" "$1"; }

echo "== preflight: $(basename "${CONFIG}") ($(( ${#chain[@]} )) files: $(for f in "${chain[@]}"; do basename "$f"; done | tr '\n' ' '))"
gen_nodes=$(r policy.generation.colocated.resources.num_nodes)
colocated=$(r policy.generation.colocated.enabled)
train_nodes=${NODES}
[[ "${colocated}" == false ]] && train_nodes=$(( NODES - ${gen_nodes:-0} ))
megatron=$(r policy.megatron_cfg.enabled)
prompts=$(r grpo.num_prompts_per_step); gens=$(r grpo.num_generations_per_prompt)
gbs=$(r policy.train_global_batch_size); maxlen=$(r policy.max_total_sequence_length)
packing=$(r policy.sequence_packing.enabled)
mb_tokens=$(r policy.sequence_packing.train_mb_tokens); lp_tokens=$(r policy.sequence_packing.logprob_mb_tokens)
async=$(r grpo.async_grpo.enabled)
backend=$(r policy.generation.backend)
g() { r "policy.generation.mcore_generation_config.$1"; }
gen_tp=$(g tensor_model_parallel_size); gen_tp=${gen_tp:-$(r policy.megatron_cfg.tensor_model_parallel_size)}
gen_pp=$(g pipeline_model_parallel_size); gen_pp=${gen_pp:-$(r policy.megatron_cfg.pipeline_model_parallel_size)}
gen_ep=$(g expert_model_parallel_size); gen_ep=${gen_ep:-$(r policy.megatron_cfg.expert_model_parallel_size)}
gen_dispatcher=$(g moe_token_dispatcher_type); gen_dispatcher=${gen_dispatcher:-$(r policy.megatron_cfg.moe_token_dispatcher_type)}
gen_deepep=$(g moe_enable_deepep); gen_deepep=${gen_deepep:-$(r policy.megatron_cfg.moe_enable_deepep)}
echo "  nodes: ${NODES} = ${train_nodes} train + ${gen_nodes:-0} generation (colocated: ${colocated})"
if [[ "${backend}" == megatron ]]; then
  echo "  generation: megatron TP${gen_tp} PP${gen_pp} EP${gen_ep}, dispatcher ${gen_dispatcher} (deepep ${gen_deepep}), buffer $(g buffer_size_gb) GB, max requests $(g max_requests), cuda graphs $(g cuda_graph_impl)/$(g inference_cuda_graph_scope), kv cache $(g kv_cache_management_mode), logprobs $(g logprobs_mode), refit $(r policy.generation.refit_transport)"
else
  echo "  generation: ${backend} TP$(r policy.generation.vllm_cfg.tensor_parallel_size) EP$(r policy.generation.vllm_cfg.expert_parallel_size), async engine $(r policy.generation.vllm_cfg.async_engine), enforce eager $(r policy.generation.vllm_cfg.enforce_eager), prefix caching $(v=$(r policy.generation.vllm_cfg.enable_prefix_caching); echo "${v:-on (NeMo-RL default)}"), gpu mem $(r policy.generation.vllm_cfg.gpu_memory_utilization)"
fi
echo "  trainer: $([[ "${megatron}" == true ]] && echo "megatron TP$(r policy.megatron_cfg.tensor_model_parallel_size) PP$(r policy.megatron_cfg.pipeline_model_parallel_size) EP$(r policy.megatron_cfg.expert_model_parallel_size) CP$(r policy.megatron_cfg.context_parallel_size), dispatcher $(r policy.megatron_cfg.moe_token_dispatcher_type) (deepep $(r policy.megatron_cfg.moe_enable_deepep)), activation ckpt $(r policy.megatron_cfg.activation_checkpointing), lr $(r policy.megatron_cfg.optimizer.lr), warmup $(r policy.megatron_cfg.scheduler.lr_warmup_iters) from $(r policy.megatron_cfg.scheduler.lr_warmup_init)" || echo "dtensor, lr $(r policy.optimizer.kwargs.lr)")"
echo "  batch: ${prompts} prompts x ${gens} = $(( prompts * gens )) samples, global batch ${gbs} ($(( prompts * gens / gbs )) optimizer steps), micro-batch $(r policy.train_micro_batch_size)"
echo "  length: total ${maxlen}, new tokens $(r policy.generation.max_new_tokens), packing ${packing} (train ${mb_tokens}, logprob ${lp_tokens} tokens)"
echo "  grpo: ${max_steps:=$(r grpo.max_num_steps)} steps, val every $(r grpo.val_period) (start $(r grpo.val_at_start), end $(r grpo.val_at_end)), async ${async} (age $(r grpo.async_grpo.max_trajectory_age_steps), in-flight $(r grpo.async_grpo.in_flight_weight_updates))"
echo "  loss: IS correction $(r loss_fn.use_importance_sampling_correction) ($(r loss_fn.truncated_importance_sampling_type) $(r loss_fn.truncated_importance_sampling_ratio)), KL $(r loss_fn.reference_policy_kl_penalty), clip $(r loss_fn.ratio_clip_min)/$(r loss_fn.ratio_clip_max), dual-clip $(r loss_fn.ratio_clip_c); advantage: normalized $(r grpo.normalize_rewards), leave-one-out $(r grpo.use_leave_one_out_baseline)"
echo "  checkpoints: enabled $(r checkpointing.enabled), every $(r checkpointing.save_period), keep $(r checkpointing.keep_top_k), save-and-stop by $(r checkpointing.checkpoint_must_save_by); job time ${JOB_TIME:-?}"
o() { r "policy.megatron_cfg.optimizer.$1"; }
opt=$(o optimizer)
if [[ "${megatron}" == true ]]; then
  echo "  optimizer: ${opt}, lr $(o lr)$([[ "${opt}" == md_decoupling ]] && echo ", matrix lr $(o matrix_lr), gains lr $(o gains_lr)"), min lr $(o min_lr), weight decay $(o weight_decay), betas $(o adam_beta1)/$(o adam_beta2), distributed $(o use_distributed_optimizer), layer-wise $(o use_layer_wise_distributed_optimizer), cpu offload $(o optimizer_cpu_offload) ($(o optimizer_offload_fraction)), offload for logprobs $(r policy.offload_optimizer_for_logprob)"
fi

isnum() { [[ "$1" =~ ^[0-9]+$ ]]; }
(( train_nodes > 0 )) || err "no trainer nodes left (${NODES} nodes, ${gen_nodes} for generation)"
if [[ "${megatron}" == true ]]; then
  tp=$(r policy.megatron_cfg.tensor_model_parallel_size); pp=$(r policy.megatron_cfg.pipeline_model_parallel_size)
  ep=$(r policy.megatron_cfg.expert_model_parallel_size)
  if isnum "${tp}" && isnum "${pp}" && isnum "${ep}"; then
    train_gpus=$(( train_nodes * 4 ))
    (( train_gpus % (tp * pp) == 0 && (train_gpus / (tp * pp)) % ep == 0 )) \
      || err "trainer EP${ep} does not divide the data-parallel size of ${train_gpus} GPUs at TP${tp} PP${pp}"
  fi
  if [[ "${opt}" == md_decoupling ]]; then
    [[ "$(o use_distributed_optimizer)" == false ]] || err "md_decoupling needs use_distributed_optimizer: false (it shards state with use_layer_wise_distributed_optimizer)"
    [[ "$(o optimizer_cpu_offload)" == false ]] || err "md_decoupling has no CPU-offload path; set optimizer_cpu_offload: false"
    [[ "$(o use_precision_aware_optimizer)" == false ]] || err "use_precision_aware_optimizer is Adam-only; set it to false for md_decoupling"
    [[ "$(o use_layer_wise_distributed_optimizer)" == true ]] || warn "md_decoupling without use_layer_wise_distributed_optimizer keeps the full optimizer state on every rank"
  fi
  # OptimizerParamScheduler asserts min_lr <= lr and lr_warmup_init <= lr, even without warmup.
  lr=$(o lr); minlr=$(o min_lr); initlr=$(r policy.megatron_cfg.scheduler.lr_warmup_init)
  gt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 > b + 0) }'; }
  gt "${minlr}" "${lr}" && err "min_lr ${minlr} is above lr ${lr}; Megatron's scheduler rejects it"
  gt "${initlr}" "${lr}" && err "lr_warmup_init ${initlr} is above lr ${lr}; Megatron's scheduler rejects it (even with lr_warmup_iters 0)"
  # UCCL's internode kernels (flex dispatcher, EP across nodes) take their chunk sizes from
  # UCCL_EP_{DISPATCH,COMBINE}_CONFIG = sms,nvl_send,nvl_recv,rdma_send,rdma_recv and assert
  # these rules for the number of nodes an EP group spans (ep/src/internode.cu, ep_config.hpp).
  if [[ "$(r policy.megatron_cfg.moe_token_dispatcher_type)" == flex ]] && isnum "${ep}" && (( ep > 4 )); then
    rr=$(( ep / 4 ))
    [[ " 2 3 4 8 16 " == *" ${rr} "* ]] || err "UCCL internode kernels exist for EP groups of 2, 3, 4, 8 or 16 nodes; EP${ep} spans ${rr}"
    declare -A sms=()
    for kind in DISPATCH COMBINE; do
      cfg=$(r "policy.megatron_cfg.env_vars.UCCL_EP_${kind}_CONFIG")
      [[ -n "${cfg}" ]] || { warn "UCCL_EP_${kind}_CONFIG unset: UCCL's built-in table assumes 8 GPUs per node, check it for EP${ep}"; continue; }
      IFS=, read -r s ns nr rs rv <<<"${cfg}"
      sms[${kind}]=$s
      rva=$(( (rv + rs - 1) / rs * rs ))
      (( s % 2 == 0 )) || err "UCCL ${kind} '${cfg}': SM count must be even"
      (( ns < nr )) || err "UCCL ${kind} '${cfg}': nvl_send must be below nvl_recv"
      (( rs < rva && rs <= rva / 2 )) || err "UCCL ${kind} '${cfg}': rdma_send must be at most half of rdma_recv"
      if [[ "${kind}" == COMBINE ]]; then
        (( nr % rr == 0 )) || err "UCCL COMBINE '${cfg}': nvl_recv must be a multiple of the ${rr} nodes"
        mx=$(( rs > ns ? rs : ns ))
        (( nr / rr > mx )) || err "UCCL COMBINE '${cfg}': nvl_recv/${rr} nodes = $(( nr / rr )) must exceed max(rdma_send, nvl_send) = ${mx}"
        fw=$(( 16 / rr > 1 ? 16 / rr : 1 ))
        (( rs >= fw )) || err "UCCL COMBINE '${cfg}': rdma_send must be at least ${fw} at ${rr} nodes"
      fi
    done
    [[ -n "${sms[DISPATCH]:-}" && -n "${sms[COMBINE]:-}" && "${sms[DISPATCH]}" != "${sms[COMBINE]}" ]] \
      && err "UCCL dispatch and combine configs must use the same SM count"
  fi
fi
if isnum "${prompts}" && isnum "${gens}" && isnum "${gbs}"; then
  (( prompts * gens % gbs == 0 )) || err "samples per step ($(( prompts * gens ))) not a multiple of the global batch (${gbs})"
fi
if [[ "${packing}" == true ]]; then
  for t in "${mb_tokens}" "${lp_tokens}"; do
    if isnum "$t" && isnum "${maxlen}"; then
      (( t >= maxlen )) || err "packing budget ${t} tokens < max_total_sequence_length ${maxlen}"
    else
      warn "packing budget '${t}' is not a number; check it resolves to >= ${maxlen}"
    fi
  done
fi
if [[ "${async}" == true ]]; then
  [[ "$(r loss_fn.use_importance_sampling_correction)" == true ]] || err "async GRPO needs loss_fn.use_importance_sampling_correction"
  if [[ "${backend}" == vllm ]]; then
    [[ "$(r policy.generation.vllm_cfg.async_engine)" == true ]] || err "async GRPO needs policy.generation.vllm_cfg.async_engine"
  fi
fi
if [[ "${backend}" == megatron ]]; then
  [[ "${megatron}" == true ]] || err "megatron generation needs the Megatron trainer (policy.megatron_cfg.enabled)"
  if [[ "${colocated}" == false ]] && isnum "${gen_nodes}" && isnum "${gen_tp}" && isnum "${gen_pp}" && isnum "${gen_ep}"; then
    gen_gpus=$(( gen_nodes * 4 ))
    (( gen_gpus % (gen_tp * gen_pp) == 0 && (gen_gpus / (gen_tp * gen_pp)) % gen_ep == 0 )) \
      || err "generation EP${gen_ep} does not divide the data-parallel size of ${gen_gpus} GPUs at TP${gen_tp} PP${gen_pp}"
  fi
  [[ "${gen_deepep}" == true && "${gen_dispatcher}" != flex ]] && err "generation sets moe_enable_deepep with dispatcher ${gen_dispatcher}; Megatron requires flex for DeepEP"
  [[ "$(r loss_fn.use_importance_sampling_correction)" == true && "$(g logprobs_mode)" != raw_logprobs ]] \
    && warn "generation logprobs_mode $(g logprobs_mode): importance weights compare against the trainer's raw log-probs; raw_logprobs matches them"
  [[ "$(g kv_cache_management_mode)" == recompute ]] && [[ "$(r grpo.async_grpo.recompute_kv_cache_after_weight_updates)" != true ]] \
    && err "kv_cache_management_mode recompute needs grpo.async_grpo.recompute_kv_cache_after_weight_updates"
fi
# Refit chunking must agree on both sides of the weight broadcast. Megatron generation
# inherits the trainer's env_vars unless mcore_generation_config.env_vars replaces them.
for v in NRL_REFIT_BUFFER_MEMORY_RATIO NRL_REFIT_NUM_BUFFERS; do
  t=$(r "policy.megatron_cfg.env_vars.$v"); [[ "${megatron}" == true ]] || t=$(r "policy.dtensor_cfg.env_vars.$v")
  if [[ "${backend}" == megatron ]]; then
    gv=$t; [[ -n "$(g env_vars)" ]] && gv=$(g "env_vars.$v")
  else
    gv=$(r "policy.generation.vllm_cfg.env_vars.$v")
  fi
  [[ "$t" == "$gv" ]] || err "$v differs between trainer ('$t') and generation ('$gv'); the refit broadcast hangs"
done
must=$(r checkpointing.checkpoint_must_save_by)
if [[ -n "${must}" && "${must}" != null && -n "${JOB_TIME:-}" ]]; then
  secs() { local s=$1 d=0; [[ "$s" == *-* ]] && { d=${s%%-*}; s=${s#*-}; }; IFS=: read -r a b c <<<"$s"; echo $(( d*86400 + 10#$a*3600 + 10#$b*60 + 10#${c:-0} )); }
  IFS=: read -r md mh mm ms <<<"${must}"
  must_s=$(( 10#$md*86400 + 10#$mh*3600 + 10#$mm*60 + 10#$ms ))
  (( must_s < $(secs "${JOB_TIME}") )) || warn "checkpoint_must_save_by ${must} is not before the job time ${JOB_TIME}: the job is killed before it saves"
fi
for k in "${new_keys[@]}"; do
  err "override ${k} sets a key the config does not have; NeMo-RL rejects that, add it as +${k}=..."
done
echo "== preflight: ${errors} error(s), ${warnings} warning(s)"
(( errors == 0 ))

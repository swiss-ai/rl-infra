#!/bin/bash
# The MegaChonk vLLM matrix of 2026-11-23: replicas of 8 nodes (TP4 x EP32) and 4 nodes (TP4 x EP16), each with the
# all-gather dispatch, FP8 KV cache, DeepEP low-latency (512 batched tokens: its buffer grows with them; 4,096 ran out
# of memory, jobs 1346545/1346548) and DeepEP high-throughput. One job per variant (48 nodes in all).
#   bash shared/debug/genbench-megachonk-matrix.sh <run config.yaml>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
C=$1
G=policy.generation
EP16=$G.vllm_cfg.expert_parallel_size=16
LL="$G.vllm_kwargs.all2all_backend=deepep_low_latency $G.vllm_kwargs.max_num_batched_tokens=512"
HT=$G.vllm_kwargs.all2all_backend=deepep_high_throughput
FP8KV=$G.vllm_cfg.kv_cache_dtype=fp8
sub() { bash "${HERE}/genbench.sh" "$@" 2>&1 | grep -o 'Submitted batch job [0-9]*' | sed "s/^/$1: /"; }
sub mc8-base 8 "$C"
sub mc8-fp8kv 8 "$C" $FP8KV
sub mc8-deepll512 8 "$C" $LL
sub mc8-deepht 8 "$C" $HT
sub mc4-base 4 "$C" $EP16
sub mc4-fp8kv 4 "$C" $EP16 $FP8KV
sub mc4-deepll512 4 "$C" $EP16 $LL
sub mc4-deepht 4 "$C" $EP16 $HT

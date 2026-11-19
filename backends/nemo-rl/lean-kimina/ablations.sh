#!/bin/bash
# The GRPO ablations of 2026-11-18 (RUNBOOK.md, "Ablations"): Chonk SFT v0.2 on Goedel-Prover rows, 12 nodes
# (8 Megatron trainers + 4 vLLM), Kimina inside the job.
#   DRY_RUN=1 bash backends/nemo-rl/lean-kimina/ablations.sh <variant> [partition]   setup, preflight, sbatch command
#   bash backends/nemo-rl/lean-kimina/ablations.sh <variant> [partition]             submits (partition: preemptable)
# Variants; everything else as run 3965303 (Goedel rows, 32 prompts x 16, temperature 1.0, top_p 1.0, 16,000 new
# tokens, 120 s proof check, learning rate 2e-6, global batch 512 = 1 optimizer update per step):
#   h1     learning rate 1e-5                                          (the expert of experts round 1, step 40)
#   h2     global batch 128 (4 optimizer updates per step)              (ran out of host memory on 8 trainer nodes)
#   h3     only the 736 prompts with mixed rewards in run 3965303 (23 steps per epoch)
#   h8     learning rate 5e-6 + global batch 128 + the mixed prompts; validates at start on preemptable (the baseline)
#   h8s    learning rate 5e-6 + the mixed prompts (global batch 512); validates at start
#   nostd  advantages not divided by the group's standard deviation (grpo.normalize_rewards=false)
# All: validation = corrected miniF2F (data/minif2f-corrected-x2.v434.jsonl.gz: valid 244 + test 242, each twice)
# every 10 steps; a checkpoint every 5 steps, all kept; JOB_TIME (default 06:20:00; checkpoint-and-stop 30 min
# before). Settings:
#   RUN              default lean-kimina-v02-abl-<variant>-<partition>-<date><RUN_SUFFIX>; a job with the same RUN
#                    resumes from that run's latest checkpoint
#   VAL_AT_START     true/false, overrides the variant's choice (false when resuming)
#   EXTRA_OVERRIDES  more overrides, space-separated. Resuming with more epochs than the run started with needs
#                    "grpo.max_num_epochs=<n> ++policy.megatron_cfg.scheduler.override_opt_param_scheduler=true"
#                    (Megatron's scheduler otherwise refuses the checkpoint; safe with a constant learning rate)
set -euo pipefail
variant=${1:?variant: h1 h2 h3 h8 h8s nostd} partition=${2:-preemptable}
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
D="${LEAN_DATA}/data"
mkdir -p "${D}"

unpack() {  # unpack <name> <sha256>: data/<name>.gz -> $LEAN_DATA/data/<name>, once, checked
    local dst="${D}/$1"
    if [[ ! -s "${dst}" ]]; then
        [[ -s "${LEAN_KIMINA_DIR}/data/$1.gz" ]] || { echo "ERROR: no ${dst} and no data/$1.gz" >&2; exit 1; }
        gunzip -c "${LEAN_KIMINA_DIR}/data/$1.gz" > "${dst}.part" && mv "${dst}.part" "${dst}"
    fi
    echo "$2  ${dst}" | sha256sum -c --quiet || { echo "ERROR: ${dst} is not the expected file" >&2; exit 1; }
    echo "${dst}"
}
ALL=$(unpack goedel-train-s1.v434.jsonl 47e7b60208716969386980210fa0530a30b17588d7f44690bb8e75f69e7e4f40)
VAL=$(unpack minif2f-corrected-x2.v434.jsonl f8e04393118ba0236cb7cd33dc5b44d4a65df1e5f6f15e94c674bf3017964465)
MIXED="${D}/goedel-train-s1-mixed.v434.jsonl"   # see RUNBOOK.md, "Not in this repository yet"

rows=${ALL} start=false extra=()
case ${variant} in
    h1) extra=(policy.megatron_cfg.optimizer.lr=1.0e-5) ;;
    h2) extra=(policy.train_global_batch_size=128) ;;
    h3) rows=${MIXED} ;;
    h8) rows=${MIXED} extra=(policy.megatron_cfg.optimizer.lr=5.0e-6 policy.train_global_batch_size=128)
        [[ ${partition} == preemptable ]] && start=true ;;
    h8s) rows=${MIXED} start=true extra=(policy.megatron_cfg.optimizer.lr=5.0e-6) ;;
    nostd) extra=(grpo.normalize_rewards=false) ;;
    *) echo "unknown variant ${variant}" >&2; exit 2 ;;
esac
start=${VAL_AT_START:-${start}}
read -r -a more <<< "${EXTRA_OVERRIDES:-}"
[[ -s ${rows} ]] || { echo "ERROR: missing ${rows}" >&2; exit 1; }

cd "${ROOT}"
export RUN="${RUN:-lean-kimina-v02-abl-${variant}-${partition}-$(date +%Y%m%d)${RUN_SUFFIX:-}}" PARTITION=${partition} \
    JOB_TIME=${JOB_TIME:-06:20:00} RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-12n.yaml HEADER_ROWS=${rows}
exec bash "${LEAN_KIMINA_DIR}/train.sh" \
    data.train.data_path="${rows}" data.validation.data_path="${VAL}" \
    grpo.num_prompts_per_step=32 grpo.val_period=10 grpo.val_at_start=${start} \
    checkpointing.save_period=5 checkpointing.keep_top_k=null \
    policy.generation.temperature=1.0 policy.generation.top_p=1.0 policy.generation.max_new_tokens=16000 \
    env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.compilation_timeout=120 \
    "${extra[@]}" ${more[@]+"${more[@]}"}

#!/bin/bash
# Stage-1 jobs of the four-node design: prints the sbatch commands; submits them only with --yes.
#   bash submit.sh [--yes] [host] [vllm-auto] [vllm-fp8]      (no job names: all three)
# Each job is 1 node on PARTITION (default preemptable; debug's 2-job limit is taken by other work), requeued when
# preempted. Code in $STAGE1/code, logs in $STAGE1/logs, results in $STAGE1/out. Report: python3 report.py $STAGE1/out
set -euo pipefail
STAGE1=${STAGE1:-${SCRATCH}/lean/fournode-stage1}
PARTITION=${PARTITION:-preemptable}
yes=0
jobs=()
for a in "$@"; do
    case "${a}" in
        --yes) yes=1 ;;
        host|vllm-auto|vllm-fp8) jobs+=("${a}") ;;
        *) echo "unknown argument ${a}" >&2; exit 2 ;;
    esac
done
(( ${#jobs[@]} )) || jobs=(host vllm-auto vllm-fp8)
for f in host_job.sbatch vllm_job.sbatch host_bench.py conc_client.py; do
    [[ -f "${STAGE1}/code/${f}" ]] || { echo "ERROR: no ${STAGE1}/code/${f}" >&2; exit 1; }
done
mkdir -p "${STAGE1}/logs" "${STAGE1}/out"
for j in "${jobs[@]}"; do
    case "${j}" in
        host) cmd=(sbatch --partition="${PARTITION}" --output="${STAGE1}/logs/%j-host.out" --export=ALL,STAGE1="${STAGE1}"
                   "${STAGE1}/code/host_job.sbatch") ;;
        vllm-*) kv=${j#vllm-}
                cmd=(sbatch --partition="${PARTITION}" --output="${STAGE1}/logs/%j-vllm-kv${kv}.out"
                     --export=ALL,STAGE1="${STAGE1}",KV="${kv}" "${STAGE1}/code/vllm_job.sbatch") ;;
    esac
    echo "${cmd[*]}"
    if (( yes )); then "${cmd[@]}"; fi
done
(( yes )) || echo "(not submitted: add --yes)"

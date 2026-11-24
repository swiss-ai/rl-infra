#!/bin/bash
# Status of the 200-step run: the newest chain's jobs, the guards' decisions, the newest checkpoint, and the last
# step lines of the training job that runs now. Only reads.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/settings.sh"
echo "RUN ${RUN}, OUT ${OUT}; newest checkpoint: step $(latest_step) of ${MAX_STEPS}"
[[ -e "${CHAIN}/DONE" ]] && echo "DONE $(cat "${CHAIN}/DONE")"
[[ -e "${CHAIN}/STOP" ]] && echo "STOP $(cat "${CHAIN}/STOP")"
echo "== chain (k, training job, guard, partition, submitted; Slurm state)"
running=""
while IFS=$'\t' read -r k j g p t; do
    sj=$(sacct -j "$j" -X -n -P -o State,Restarts,Elapsed,Start 2>/dev/null | head -n 1)
    sg=$(sacct -j "$g" -X -n -P -o State 2>/dev/null | head -n 1)
    printf '%s  J %s [%s]  G %s [%s]  %s  %s\n' "$k" "$j" "${sj:-?}" "$g" "${sg:-?}" "$p" "$t"
    [[ "${sj%%|*}" == RUNNING ]] && running=$j
done < <(chain_rows)
echo "== guard decisions"
[[ -f "${CHAIN}/guard.tsv" ]] && tail -n 8 "${CHAIN}/guard.tsv"
if [[ -n "${running}" ]]; then
    log=$(ls -t "${OUT}/slurm/${running}"-*logs/ray-driver.log "${OUT}/slurm/${running}-logs/ray-driver.log" 2>/dev/null | head -n 1)
    echo "== ${running}: ${log:-no driver log yet}"
    [[ -n "${log}" ]] && grep -E "Step [0-9]+/|train/lr|Avg Reward|Validation" "${log}" | tail -n 6
fi

#!/bin/bash
# Stop the chain: writes STOP (so no guard lets another job start) and cancels the newest chain's jobs that are still
# in the queue. A running training job is cancelled too: it loses the steps since its newest checkpoint (at most 10).
#   bash stop.sh          lists what it would cancel
#   bash stop.sh --yes    stops
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/settings.sh"
ids=()
while IFS=$'\t' read -r k j g _; do ids+=("$j" "$g"); done < <(chain_rows)
live=()
for id in "${ids[@]}"; do
    s=$(squeue -h -j "${id}" -o %T 2>/dev/null || true)
    if [[ -n "$s" ]]; then live+=("${id}"); echo "${id}: ${s}"; fi
done
if [[ "${1:-}" != --yes ]]; then echo "dry run: ${#live[@]} job(s) would be cancelled; bash stop.sh --yes to stop"; exit 0; fi
date '+%F %T' > "${CHAIN}/STOP"
if (( ${#live[@]} )); then scancel "${live[@]}"; fi
echo "STOP written; cancelled: ${live[*]:-none}. Remove ${CHAIN}/STOP before start.sh starts a new chain."

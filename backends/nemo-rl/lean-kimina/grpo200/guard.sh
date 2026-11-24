#!/bin/bash
# Guard G<k> of the chain (start.sh submits it with --dependency=afterany:J<k>): runs once training job J<k> has
# ended for good and decides whether J<k+1> may start.
#   exit 0  continue: the run is not done, there is no STOP file, and it made progress since the previous guard
#           (or this is the chain's first start without progress, which is forgiven once: a first step can fail
#           once by chance, e.g. a GPU running out of memory)
#   exit 1  stop: cancels the chain's later jobs (and J<k+1> could not start anyway: afterok)
# Every decision is a line in guard.tsv: k, training job, its state, restarts, elapsed, newest step, decision, time.
set -uo pipefail
: "${CHAIN_DIR:?}" "${CHAIN_K:?}" "${TRAIN_JOB:?}"
source "${CHAIN_DIR}/settings.sh" || exit 1
say() { echo "[guard ${CHAIN_K} $(date '+%F %T')] $*"; }
IFS='|' read -r state elapsed restarts exitcode <<< "$(sacct -j "${TRAIN_JOB}" -X -n -P -o State,Elapsed,Restarts,ExitCode 2>/dev/null | head -n 1)"
state=${state%% *}
step=$(latest_step)
# The step at the previous decision of this chain (or at its start).
prev=$(awk -F'\t' '/^# chain started/ {p = $0; sub(/.*from step /, "", p); s = p + 0; nop = 0; next}
                    {s = $6; if ($7 == "continue-no-progress") nop++} END {print s + 0, nop + 0}' "${CHAIN}/guard.tsv" 2>/dev/null)
read -r prev_step strikes <<< "${prev:-0 0}"
rows=()
while IFS= read -r r; do rows+=("$r"); done < <(chain_rows)
last_k=${#rows[@]}

# Still in the queue: requeued after a preemption, if Slurm ever releases afterany then. J<k+1> may go ahead:
# singleton keeps the two from running at the same time, and both resume from the newest checkpoint.
alive=$(squeue -h -j "${TRAIN_JOB}" -o %T 2>/dev/null | head -n 1)
if [[ -e "${CHAIN}/STOP" ]]; then decision=stop-file
elif (( step >= MAX_STEPS )); then decision=done; date '+%F %T' > "${CHAIN}/DONE"
elif [[ -n "${alive}" ]]; then decision=continue-alive; state="${alive}"
elif (( step > prev_step )); then decision=continue
elif (( strikes < 1 )); then decision=continue-no-progress
else decision=stop-no-progress
fi
if [[ "${decision}" == continue* && "${CHAIN_K}" -ge "${last_k}" ]]; then decision=chain-exhausted; fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${CHAIN_K}" "${TRAIN_JOB}" "${state:-unknown}" "${restarts:-?}" \
    "${elapsed:-?}" "${step}" "${decision}" "$(date '+%F %T')" >> "${CHAIN}/guard.tsv"
say "J${CHAIN_K} = ${TRAIN_JOB}: ${state:-unknown} (exit ${exitcode:-?}, ${restarts:-?} requeues, ${elapsed:-?}); newest checkpoint step ${step} (before: ${prev_step}); ${decision}"

case "${decision}" in
    continue|continue-no-progress|continue-alive) exit 0 ;;
esac
# Stop: cancel the chain's later jobs (training and guards), and earlier training jobs still in the queue.
later=()
for r in "${rows[@]}"; do
    IFS=$'\t' read -r k j g _ <<< "$r"
    if (( k > CHAIN_K )); then later+=("$j" "$g")
    elif [[ -n "$(squeue -h -j "$j" -o %T 2>/dev/null)" ]]; then later+=("$j")
    fi
done
if (( ${#later[@]} )); then say "cancelling the chain's later jobs: ${later[*]}"; scancel "${later[@]}"; fi
case "${decision}" in
    done) say "the run is complete (step ${step})" ;;
    chain-exhausted) say "the chain's last job ended at step ${step} < ${MAX_STEPS}: run start.sh again to continue" ;;
    *) say "stopped (${decision}); see ${CHAIN}/guard.tsv; start.sh starts a new chain" ;;
esac
exit 1

#!/bin/bash
# Status of training jobs of this task, read-only, one "<tag> <event>" line per event: Slurm state, Ray head and
# Kimina server start, every step's reward and time, every validation accuracy, checkpoints, and the errors that end
# a run (out of memory, NCCL, dead Ray actors, preemption).
#   bash backends/nemo-rl/lean-kimina/status.sh <jobid>:<tag> [<jobid>:<tag> ...]
# Reads $SCRATCH/nemo-rl/slurm/<job>-logs/ray-driver.log and the job's .out, and the Kimina logs of its run.
# STEP numbers count the steps in that job's log: a job that resumed a run starts again at 1.
L=${SCRATCH}/nemo-rl/slurm
for jt in "$@"; do
    j=${jt%%:*} tag=${jt#*:}
    S=$(sacct -X -n -P -j "${j}" -o State | head -1)
    R=$(squeue -h -j "${j}" -o %r 2>/dev/null)
    echo "${tag} STATE ${S:-?}${R:+ (${R})}"
    [[ "${S}" == PENDING || -z "${S}" ]] && continue
    run=$(sacct -X -n -P -j "${j}" -o JobName | head -1); run=${run#nrl-}
    D=${L}/${j}-logs/ray-driver.log
    [[ -f ${L}/${j}-logs/STARTED_RAY_HEAD ]] && echo "${tag} ray head up"
    grep -hE "^\[kimina [^]]*\] (the Ray head|the server ended|no Ray head|ray.sub ended|signal|restart)" \
        "${L}/${j}-${run}.out" 2>/dev/null | sed "s/^/${tag} KIMINA /"
    grep -h "Kimina server up" "${SCRATCH}/lean-kimina/runs/${run}/logs/"*-kimina-*.out 2>/dev/null | sort -u \
        | sed "s/^/${tag} KIMINA /"
    awk -v t="${tag}" '/Validation Results/{v=1} v&&/Accuracy:/{print t " VAL accuracy " $NF; v=0}
        /Training Results/{tr=1; n++} tr&&/Avg Reward:/{r=$NF}
        tr&&/Total step time:/{print t " STEP " n ": reward " r ", step time " $NF; tr=0}' "${D}" 2>/dev/null
    grep -hoE "Starting async GRPO training|Saved checkpoint[^[]{0,60}|CUDA out of memory|OutOfMemoryError|oom-kill|NCCL error|DistBackendError|RayActorError|ActorDiedError|preempted|DUE TO PREEMPTION|CANCELLED AT" \
        "${D}" "${L}/${j}-${run}.out" 2>/dev/null | sort -u | sed "s/^/${tag} DRIVER /"
done

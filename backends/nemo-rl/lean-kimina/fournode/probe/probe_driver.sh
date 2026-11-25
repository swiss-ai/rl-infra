#!/bin/bash
# The probe's driver (probe.sh sets DRIVER_PYTHON to it; launch.sh runs it through driver_retry.sh on the Ray head):
# the NeMo-RL driver twice in one allocation.
#   phase A  steps 1..PROBE_FIRST_STEPS from the SFT export; the run saves its checkpoint at the last one
#            (probe.sh sets checkpointing.save_period to PROBE_FIRST_STEPS). Skipped when that checkpoint exists
#            (a job started again on the same RUN).
#   phase B  a new driver, so new Ray workers, a new Megatron optimizer and ScratchAdamW: it resumes from the newest
#            checkpoint as a requeued job does, and runs to PROBE_STEPS.
# Every grpo.max_num_steps in the arguments is replaced by the phase's own. Phase B gets
# checkpointing.checkpoint_must_save_by from NRL_JOB_END_EPOCH as driver_retry.sh computes it (NeMo-RL counts it from
# the driver's start). Phase marks go to <logger.log_dir>/probe-phases.tsv (epoch, phase, event) and to stdout
# (ray-driver.log), where report.py splits the log by them.
#   env: PROBE_STEPS (default 5), PROBE_FIRST_STEPS (default 3), PROBE_DRIVER (default apertus2/driver.sh),
#        NRL_JOB_END_EPOCH, NRL_SAVE_MARGIN_S (default 1800), PROBE_GPU_WAIT_S (default 600)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r -a DRIVER <<< "${PROBE_DRIVER:-bash ${HERE}/../../../apertus2/driver.sh}"
STEPS="${PROBE_STEPS:-5}"
FIRST="${PROBE_FIRST_STEPS:-3}"
KEY=checkpointing.checkpoint_must_save_by
log_dir="" ckpt_dir=""
for a in "$@"; do
    [[ "${a}" == logger.log_dir=* ]] && log_dir="${a#*=}"
    [[ "${a}" == checkpointing.checkpoint_dir=* ]] && ckpt_dir="${a#*=}"
done
[[ -n "${log_dir}" && -n "${ckpt_dir}" ]] \
    || { echo "[probe_driver] ERROR: no logger.log_dir or checkpointing.checkpoint_dir in the arguments" >&2; exit 2; }
mkdir -p "${log_dir}"
mark() {
    printf '%s\t%s\t%s\n' "$(date +%s)" "$1" "$2" >> "${log_dir}/probe-phases.tsv"
    echo "[probe_driver] $(date +%FT%T) phase $1 $2"
}
# args = "$@" with every grpo.max_num_steps set to $1 and, given $2, KEY set to it (added when missing).
phase_args() {
    local steps="$1" save_by="${2:-}" a found=0
    args=()
    shift 2
    for a in "$@"; do
        [[ "${a}" == grpo.max_num_steps=* ]] && a="grpo.max_num_steps=${steps}"
        if [[ -n "${save_by}" && "${a}" == "${KEY}="* ]]; then a="${KEY}=${save_by}"; found=1; fi
        args+=("${a}")
    done
    args+=("grpo.max_num_steps=${steps}")
    [[ -z "${save_by}" ]] || (( found )) || args+=("${KEY}=${save_by}")
}
save_by() {
    [[ "${NRL_JOB_END_EPOCH:-}" =~ ^[0-9]+$ ]] || return 0
    local left=$(( NRL_JOB_END_EPOCH - $(date +%s) - ${NRL_SAVE_MARGIN_S:-1800} ))
    (( left < 300 )) && left=300
    printf '%02d:%02d:%02d:%02d' $(( left / 86400 )) $(( left % 86400 / 3600 )) $(( left % 3600 / 60 )) $(( left % 60 ))
}

if [[ -d "${ckpt_dir}/step_${FIRST}" ]]; then
    mark A "skipped (step_${FIRST} exists)"
else
    phase_args "${FIRST}" "" "$@"
    mark A "start steps=${FIRST}"
    "${DRIVER[@]}" "${args[@]}"
    rc=$?
    mark A "end rc=${rc}"
    (( rc == 0 )) || exit "${rc}"
    [[ -d "${ckpt_dir}/step_${FIRST}" ]] \
        || { echo "[probe_driver] ERROR: phase A ended without ${ckpt_dir}/step_${FIRST}" >&2; exit 3; }
    # The old driver's workers must be gone before the new one asks Ray for all GPUs.
    used=""
    for _ in $(seq $(( ${PROBE_GPU_WAIT_S:-600} / 10 ))); do
        used="$(ray status 2>/dev/null | awk '$2 == "GPU" {split($1, a, "/"); print a[1]; exit}')"
        [[ -z "${used}" || "${used}" == "0.0" ]] && break
        sleep 10
    done
    echo "[probe_driver] GPUs in use before phase B: ${used:-unknown}"
fi
phase_args "${STEPS}" "$(save_by)" "$@"
mark B "start steps=${STEPS}"
"${DRIVER[@]}" "${args[@]}"
rc=$?
mark B "end rc=${rc}"
exit "${rc}"

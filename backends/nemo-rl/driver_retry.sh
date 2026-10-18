#!/bin/bash
# Runs the NeMo-RL driver and, when it fails after making progress, starts it again in the
# same allocation. The new driver resumes from the latest checkpoint in the run's checkpoint
# directory, so a crash (or a hang that hang_watch.sh ends) costs the steps since that
# checkpoint instead of a new wait in the queue. A failure is retried when the run has shown
# that it works: this attempt or an earlier one in the allocation wrote a training-step record
# (train_data_step*.jsonl under logger.log_dir), or the run already has a checkpoint
# (step_* under checkpointing.checkpoint_dir), so a resumed attempt that fails in its first
# step is retried too. A failure before any of that is taken as a setup or configuration
# error and not retried. Without a logger.log_dir argument, running for DRIVER_RETRY_MIN_S
# (default 1800 s) counts instead.
# Before a retry it waits until Ray reports all GPUs free again. NeMo-RL measures
# checkpoint_must_save_by from the driver's own start, so a retry gets that override
# shortened to the time that is left (and no retry when less than DRIVER_RETRY_MIN_S is
# left). launch.sh puts this in front of the driver.
#
# Usage: driver_retry.sh <driver command...>   (env: DRIVER_RETRIES, default 2; DRIVER_RETRY_MIN_S)
set -uo pipefail
RETRIES="${DRIVER_RETRIES:-2}"
MIN_S="${DRIVER_RETRY_MIN_S:-1800}"
KEY=checkpointing.checkpoint_must_save_by
deadline_s=""
log_dir=""
ckpt_dir=""
for a in "$@"; do
  if [[ "${a}" == "${KEY}="* ]]; then
    IFS=: read -r d h m s <<<"${a#*=}"
    deadline_s=$(( 10#${d}*86400 + 10#${h}*3600 + 10#${m}*60 + 10#${s} ))
  fi
  [[ "${a}" == logger.log_dir=* ]] && log_dir="${a#*=}"
  [[ "${a}" == checkpointing.checkpoint_dir=* ]] && ckpt_dir="${a#*=}"
done
total_steps=0
marker="$(mktemp)"
trap 'rm -f "${marker}"' EXIT
attempt=0
while true; do
  start=${SECONDS}
  touch "${marker}"
  echo "[driver_retry] attempt ${attempt} started $(date +%FT%T)"
  "$@"
  rc=$?
  ran=$(( SECONDS - start ))
  echo "[driver_retry] attempt ${attempt} exited with ${rc} after ${ran}s"
  (( rc == 0 )) && exit 0
  if (( attempt >= RETRIES )); then echo "[driver_retry] no retries left"; exit "${rc}"; fi
  if [[ -n "${log_dir}" ]]; then
    steps=$(find "${log_dir}" -name 'train_data_step*.jsonl' -newer "${marker}" 2>/dev/null | wc -l)
    total_steps=$(( total_steps + steps ))
    ckpts=$(ls -d "${ckpt_dir:-/nonexistent}"/step_* 2>/dev/null | wc -l)
    echo "[driver_retry] attempt ${attempt} completed ${steps} training step(s) before failing (${total_steps} in this allocation, ${ckpts} checkpoint(s) in the run)"
    if (( total_steps == 0 && ckpts == 0 )); then echo "[driver_retry] no step completed and no checkpoint yet, not retrying"; exit "${rc}"; fi
  elif (( ran < MIN_S )); then
    echo "[driver_retry] failed within ${MIN_S}s, not retrying"; exit "${rc}"
  fi
  attempt=$(( attempt + 1 ))
  for _ in $(seq 60); do
    used="$(ray status 2>/dev/null | awk '$2 == "GPU" {split($1, a, "/"); print a[1]; exit}')"
    [[ -z "${used}" || "${used}" == "0.0" ]] && break
    sleep 10
  done
  echo "[driver_retry] GPUs in use before retry: ${used:-unknown}"
  if [[ -n "${deadline_s}" ]]; then
    left=$(( deadline_s - SECONDS ))
    if (( left < MIN_S )); then echo "[driver_retry] ${left}s left before ${KEY}, not retrying"; exit "${rc}"; fi
    new="$(printf '%02d:%02d:%02d:%02d' $(( left / 86400 )) $(( left % 86400 / 3600 )) $(( left % 3600 / 60 )) $(( left % 60 )))"
    args=()
    for a in "$@"; do
      [[ "${a}" == "${KEY}="* ]] && a="${KEY}=${new}"
      args+=("${a}")
    done
    set -- "${args[@]}"
    echo "[driver_retry] ${KEY}=${new} for this attempt"
  fi
done

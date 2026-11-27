#!/bin/bash
# Why do a run's task pods stay Pending: the registry, or Breithorn's capacity? (login node, ~15 min)
#   bash experiments/osb-reach/registry-diag.sh [task]     # task: a regex on the task dir name (default sdmetrics,
#                                                          # one of the four tasks of job 5374114)
# 1. registry_check.sh on the task's agent and verifier image: may the robot account pull them?
# 2. pending.py starts three pods at once and prints their Kubernetes events while they wait:
#      small      python:3.12-slim (Docker Hub), 1 CPU / 1 GiB    -> does OpenSandbox start anything?
#      swe-small  the task's image, 1 CPU / 1 GiB                 -> can the image be pulled?
#      swe-full   the task's image with the task's resources       -> is there room for a real task pod?
# Reading the summary at the end:
#   small Running, swe-small + swe-full Pending with ErrImagePull/ImagePullBackOff -> registry or credentials
#   small + swe-small Running, swe-full Pending with FailedScheduling/Insufficient -> capacity (Breithorn admins)
#   all three Pending                                                               -> namespace/cluster (admins)
#   all three Running                                                               -> fine now: rerun the job
# Env: HARBOR_ROOT ($SCRATCH/osb/harbor_verified), RL_REGISTRY_CRED_FILE (~/.rl_registry_cred),
# OPENSANDBOX_API_KEY_FILE (~/.opensandbox_token), WAIT_S (900). Secrets are read from those files, never printed.
# Output also goes to $SCRATCH/osb-reach/registry-diag-<date>.log (send that file).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PATTERN="${1:-sdmetrics}"
HARBOR_ROOT="${HARBOR_ROOT:-${SCRATCH}/osb/harbor_verified}"
mkdir -p "${SCRATCH}/osb-reach"
LOG="${SCRATCH}/osb-reach/registry-diag-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "${LOG}") 2>&1

dir="$(find "${HARBOR_ROOT}/tasks" -mindepth 1 -maxdepth 1 -type d | grep -E -- "${PATTERN}" | sort | head -n1)"
[[ -n "${dir}" && -r "${dir}/task.toml" ]] || { echo "ERROR: no task dir matching /${PATTERN}/ in ${HARBOR_ROOT}/tasks"; exit 1; }
toml="${dir}/task.toml"
image_of() {
    awk -v want="$1" '/^[[:space:]]*\[/ { s = $0; gsub(/[][[:space:]]/, "", s); next }
        s == want && /^[[:space:]]*docker_image[[:space:]]*=/ { sub(/^[^"]*"/, ""); sub(/".*/, ""); print; exit }' "${toml}"
}
echo "== $(date '+%F %T %Z') task $(basename "${dir}"); log ${LOG}"

rc=0
for sec in environment verifier.environment; do
    img="$(image_of "${sec}")"
    [[ -n "${img}" ]] || continue
    echo; echo "#### 1. registry check, [${sec}] image"
    bash "${HERE}/registry_check.sh" "${img}" || rc=1
done

echo; echo "#### 2. three test pods (up to ${WAIT_S:-900} s)"
PY="$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3)"
WAIT_S="${WAIT_S:-900}" "${PY}" "${HERE}/pending.py" --cases small,swe-small,swe-full --task-toml "${toml}" || rc=1

echo; echo "== done (exit ${rc}); how to read it: the header of ${HERE}/registry-diag.sh; log ${LOG}"
exit "${rc}"

#!/bin/bash
# Before a submission: can Breithorn pull the run's task images? Job 5374114 (run OV smoke) spent 10.7
# node-hours on 64 creates that all stayed Pending (POD_READY_TIMEOUT), while rl-registry pulls failed.
#   bash experiments/osb-reach/registry_precheck.sh <train.jsonl> <harbor_data_dir>
# Takes the first row's task (instance_id `<alias>::<task>` -> <harbor_data_dir>/tasks/<task>/task.toml) and
# checks its agent image ([environment]) and verifier image ([verifier.environment], if any) with
# registry_check.sh (the robot account in RL_REGISTRY_CRED_FILE must get the manifest: HTTP 200).
# REGISTRY_CHECK: 1 (default) = that; pod = also start one pod of the agent image with the task's resources
# (pending.py --cases swe-full, WAIT_S default 600; catches capacity as well, takes up to WAIT_S); 0 = skip.
# Exit 0 = go, 1 = do not submit (the output says why; registry-diag.sh tells registry from capacity).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TRAIN="${1:?usage: registry_precheck.sh <train.jsonl> <harbor_data_dir>}"
DATA="${2:?usage: registry_precheck.sh <train.jsonl> <harbor_data_dir>}"
MODE="${REGISTRY_CHECK:-1}"
case "${MODE}" in
    0) echo "== registry check skipped (REGISTRY_CHECK=0)"; exit 0 ;;
    1|pod) ;;
    *) echo "ERROR: REGISTRY_CHECK must be 0, 1 or pod, not ${MODE}" >&2; exit 1 ;;
esac

task="$(head -n1 "${TRAIN}" | sed -nE 's/.*"instance_id": *"[^":]*::([^"]+)".*/\1/p')"
toml="${DATA}/tasks/${task}/task.toml"
[[ -n "${task}" && -r "${toml}" ]] || { echo "ERROR: registry check: no task.toml for the first row of ${TRAIN} (${toml})" >&2; exit 1; }

image_of() {  # image_of <section>: docker_image of that table of ${toml}
    awk -v want="$1" '/^[[:space:]]*\[/ { s = $0; gsub(/[][[:space:]]/, "", s); next }
        s == want && /^[[:space:]]*docker_image[[:space:]]*=/ { sub(/^[^"]*"/, ""); sub(/".*/, ""); print; exit }' "${toml}"
}
agent_image="$(image_of environment)"
verifier_image="$(image_of verifier.environment)"
[[ -n "${agent_image}" ]] || { echo "ERROR: registry check: no [environment] docker_image in ${toml}" >&2; exit 1; }

LOG="$(mktemp "${SCRATCH:-${HOME}}/registry-precheck.XXXXXX")"
for img in "${agent_image}" ${verifier_image:+"${verifier_image}"}; do
    if [[ "${img}" != */*/* && "${img}" != *.*/* ]]; then
        echo "== registry check: ${img} is not in a private registry, nothing to check"; continue
    fi
    if bash "${HERE}/registry_check.sh" "${img}" > "${LOG}" 2>&1; then
        echo "== registry check: $(sed -n 's/^== //p' "${LOG}" | grep '^PULL')"
    else
        grep -E '^(1\.|4\. C|== PULL)' "${LOG}" >&2
        echo "ERROR: registry check failed for task ${task}: the pods would stay Pending. Not submitting." >&2
        echo "       Diagnose: bash ${HERE}/registry-diag.sh ${task}   (skip the check: REGISTRY_CHECK=0)" >&2
        rm -f "${LOG}"; exit 1
    fi
done
rm -f "${LOG}"

if [[ "${MODE}" == pod ]]; then
    echo "== registry check: starting one pod of ${agent_image##*/} with the task's resources (up to ${WAIT_S:-600} s)"
    PY="$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3)"
    WAIT_S="${WAIT_S:-600}" "${PY}" "${HERE}/pending.py" --cases swe-full --task-toml "${toml}" | tail -n 3 \
        || { echo "ERROR: the test pod did not reach Running: not submitting (bash ${HERE}/registry-diag.sh ${task})" >&2; exit 1; }
fi
exit 0

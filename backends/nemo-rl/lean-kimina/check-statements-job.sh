#!/bin/bash
# Keeps the Gym Lean rows of SRC whose statement elaborates on the Kimina server's Lean and Mathlib (the training's
# verifier): DST, and DST.dropped.jsonl with Lean's first error. One node; a Kimina server with KIMINA_REPLS REPLs
# (default 128) in this job, its REPLs warmed with the rows' most common import header, as setup-data-job.sh does.
#   sbatch --nodes=1 --exclusive --account=... --partition=... --time=01:30:00 \
#       --output=<dir>/logs/%j-check.out \
#       --export=ALL,ROOT=<rl-infra checkout>,SRC=<rows.jsonl>,DST=<rows.v434.jsonl>,LOGS=<dir>/logs \
#       backends/nemo-rl/lean-kimina/check-statements-job.sh
# Writes only DST, DST.dropped.jsonl, the server log in LOGS, and /dev/shm (Lean staged into node memory, deleted at
# exit). The checkout and the Kimina install are only read (Python runs with -B: no __pycache__). A rerun with DST
# present does nothing.
set -euo pipefail
: "${ROOT:?}" "${SRC:?}" "${DST:?}" "${LOGS:?}"
LEAN_KIMINA_DIR="${ROOT}/backends/nemo-rl/lean-kimina"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
PY=("${KIMINA_INSTALL}/venv/bin/python" -B)
SHM="/dev/shm/${USER}-check-statements-${SLURM_JOB_ID}"
REPLS="${KIMINA_REPLS:-128}"
step() { echo "$(date +%T) == $*"; }
[[ -s "${DST}" ]] && { step "${DST} exists: nothing to do"; exit 0; }
cleanup() {
    local rc=$?
    set +e
    [[ -n "${KIMINA_PID:-}" ]] && kill "${KIMINA_PID}" 2>/dev/null && sleep 2
    pkill -u "${USER}" -f "$(kimina_repl_bin)" 2>/dev/null
    rm -rf "${SHM}" "${DST}.part" "${DST}.part.dropped.jsonl"
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 143' TERM
step "$(wc -l < "${SRC}") rows in ${SRC}"
header=$("${PY[@]}" "${LEAN_KIMINA_DIR}/dominant_header.py" "${KIMINA_INSTALL}/kimina-lean-server" "${SRC}")
step "staging Lean into ${SHM}"
kimina_stage "${SHM}"
key=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')
url="http://127.0.0.1:8100"
confine=no
if kimina_confinable; then confine=yes; fi
export PYTHONPATH="${KIMINA_INSTALL}/kimina-lean-server/client"
for attempt in 1 2; do
    step "Kimina server, ${REPLS} REPLs, bubblewrap: ${confine}"
    kimina_serve 127.0.0.1 8100 "${REPLS}" "${header}" "${key}" "${LOGS}/${SLURM_JOB_ID}-kimina-server-${confine}.log" "${confine}"
    if kimina_wait "${url}" 900 && "${PY[@]}" "${LEAN_KIMINA_DIR}/check_server.py" controls "${url}" "${key}" "${header}"; then
        break
    fi
    kill "${KIMINA_PID}" 2>/dev/null || true
    sleep 3
    pkill -u "${USER}" -f "$(kimina_repl_bin)" 2>/dev/null || true
    [[ "${confine}" == yes ]] || { echo "ERROR: the Kimina server failed its controls" >&2; exit 1; }
    echo "WARNING: the controls failed with REPLs in bubblewrap; trying unconfined (see the server log)" >&2
    confine=no
done
step "checking the statements on Lean ${REPL_TAG}"
"${PY[@]}" "${LEAN_KIMINA_DIR}/validate_statements.py" --url "${url}" --api-key "${key}" --max-inflight "${REPLS}" \
    "${SRC}" "${DST}.part"
mv "${DST}.part.dropped.jsonl" "${DST}.dropped.jsonl"
mv "${DST}.part" "${DST}"
curl -s "${url}/stats"; echo
wc -l "${DST}" "${DST}.dropped.jsonl"
step "done"

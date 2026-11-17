#!/bin/bash
# Body of the Kimina server job (start.sh submits it): Lean into node memory, then a Kimina server whose REPLs
# have imported the training rows' header, reachable from the other nodes. Once it answers, the job writes
# http://<node>:<port> to $ENDPOINT_FILE and the API key to $ENDPOINT_FILE.key (mode 600). Stop it with scancel.
# It served the training runs of 2026-11-15 (see README.md). train.sh runs it inside the training job
# (in_job.sbatch) or as a job of its own (server_watch_job.sh, SERVER=debug).
# KEEP_KEY=1 reuses the key in $ENDPOINT_FILE.key when there is one (in_job.sbatch restarts a server under the same
# address and key, which the training read at its start). SERVER_LOG names the server's own log.
set -euo pipefail
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"
SHM="${KIMINA_SHM:-/dev/shm/${USER}-lean-kimina-${SLURM_JOB_ID}}"
# Keeps the job's own exit status: under `set -e` a failing command in an EXIT trap would replace it.
cleanup() {
    local rc=$?
    set +e
    [[ -n "${KIMINA_PID:-}" ]] && kill "${KIMINA_PID}" 2>/dev/null && sleep 2
    pkill -u "${USER}" -f "$(kimina_repl_bin)" 2>/dev/null
    rm -rf "${SHM}"
    exit "${rc}"
}
trap cleanup EXIT
key=""
if [[ "${KEEP_KEY:-0}" == 1 && -s "${ENDPOINT_FILE}.key" ]]; then key="$(< "${ENDPOINT_FILE}.key")"; fi
rm -f "${ENDPOINT_FILE}" "${ENDPOINT_FILE}.key" "${ENDPOINT_FILE}.key.part"

header=$("${KIMINA_INSTALL}/venv/bin/python" "${LEAN_KIMINA_DIR}/dominant_header.py" "${KIMINA_INSTALL}/kimina-lean-server" "${HEADER_ROWS}")
kimina_stage "${SHM}"
confine=no
if kimina_confinable; then confine=yes; else echo "WARNING: bubblewrap unusable: REPLs run unconfined" >&2; fi
[[ -n "${key}" ]] || key=$(head -c 24 /dev/urandom | base64 | tr -d '/+=')
kimina_serve 0.0.0.0 "${PORT}" "${MAX_REPLS}" "${header}" "${key}" \
    "${SERVER_LOG:-$(dirname "${ENDPOINT_FILE}")/logs/${SLURM_JOB_ID}-server.log}" "${confine}"
url="http://$(hostname):${PORT}"
kimina_wait "${url}" 1800
PYTHONPATH="${KIMINA_INSTALL}/kimina-lean-server/client" "${KIMINA_INSTALL}/venv/bin/python" \
    "${LEAN_KIMINA_DIR}/check_server.py" controls "${url}" "${key}" "${header}"
# The key file is made private before the key goes in. The umask alone is not enough: a default ACL on the
# folder overrides it (Alps scratch has one: job 3931932's key file came out readable by two groups). chmod 600
# also sets the ACL mask to none.
(umask 077 && : > "${ENDPOINT_FILE}.key.part" && chmod 600 "${ENDPOINT_FILE}.key.part")
echo "${key}" > "${ENDPOINT_FILE}.key.part"
mv -f "${ENDPOINT_FILE}.key.part" "${ENDPOINT_FILE}.key"
echo "${url}" > "${ENDPOINT_FILE}"
echo "== Kimina server up on ${url} with ${MAX_REPLS} warm REPLs (bubblewrap: ${confine})"
wait "${KIMINA_PID}"

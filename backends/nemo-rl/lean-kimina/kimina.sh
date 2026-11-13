# Functions for jobs that run a Kimina Lean Server (source it). Needs env loaded and setup.sh done.
#   kimina_stage <dir>                           Lean into node memory; sets KIMINA_LEAN_PATH, KIMINA_LEAN_SYSROOT
#                                                (KIMINA_STAGE=0: read it in place, for tests)
#   kimina_serve <host> <port> <repls> <header> <key> <log> <confine>
#                                                start the server in the background; sets KIMINA_PID, KIMINA_LOG
#   kimina_wait <url> <seconds>                  wait until /health answers (1 if the server died, its start-up
#                                                failed, or time ran out)
#   kimina_confinable                            0 if bubblewrap can confine REPLs on this node
# The server settings are the ones measured on Clariden (the fork's docs/MANY_CORE_NODES.md).

LEAN_KIMINA_DIR="${LEAN_KIMINA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

kimina_stage() {
    local dst=$1 staged
    if [[ "${KIMINA_STAGE:-1}" == 0 ]]; then
        # Read Lean where it is (tests on a machine without room for a copy): no copy, slower imports.
        mkdir -p "${dst}"
        KIMINA_LEAN_PATH=$(ls -d "${LEAN_PROJECT}"/.lake/packages/*/.lake/build/lib/lean | paste -sd: -)
        KIMINA_LEAN_SYSROOT="${LEAN_TOOLCHAIN}"
        KIMINA_STAGE_DIR="${dst}"
        echo "Lean read in place (KIMINA_STAGE=0)"
        return 0
    fi
    staged=$("${KIMINA_INSTALL}/venv/bin/python" "${LEAN_KIMINA_DIR}/stage_lean.py" \
        --toolchain "${LEAN_TOOLCHAIN}" --project "${LEAN_PROJECT}" --dst "${dst}" --jobs 64) || return 1
    echo "${staged}"
    KIMINA_LEAN_PATH=$("${KIMINA_INSTALL}/venv/bin/python" -c 'import json,sys; print(json.loads(sys.argv[1])["LEAN_PATH"])' "${staged}")
    KIMINA_LEAN_SYSROOT=$("${KIMINA_INSTALL}/venv/bin/python" -c 'import json,sys; print(json.loads(sys.argv[1])["LEAN_SYSROOT"])' "${staged}")
    KIMINA_STAGE_DIR="${dst}"
}

kimina_confinable() {
    command -v bwrap >/dev/null && bwrap --ro-bind / / --unshare-all true 2>/dev/null
}

kimina_serve() {
    local host=$1 port=$2 repls=$3 header=$4 key=$5 log=$6 confine=$7 repl_path init
    repl_path="$(kimina_repl_bin)"
    if [[ "${confine}" == yes ]]; then
        repl_path="${LEAN_KIMINA_DIR}/repl-bwrap.sh"
    fi
    # INIT_REPLS is JSON, and the header has newlines in it.
    init=$("${KIMINA_INSTALL}/venv/bin/python" -c 'import json,sys; print(json.dumps({sys.argv[1]: int(sys.argv[2])}))' "${header}" "${repls}")
    (cd "${KIMINA_INSTALL}/kimina-lean-server" && exec env \
        REPL_BIN="$(kimina_repl_bin)" LEAN_STAGE="${KIMINA_STAGE_DIR}" \
        LEAN_SERVER_HOST="${host}" LEAN_SERVER_PORT="${port}" LEAN_SERVER_API_KEY="${key}" \
        LEAN_SERVER_ENVIRONMENT=prod LEAN_SERVER_LOG_LEVEL=WARNING \
        LEAN_SERVER_REPL_PATH="${repl_path}" LEAN_SERVER_PROJECT_DIR="${KIMINA_STAGE_DIR}" \
        LEAN_SERVER_MAX_REPLS="${repls}" LEAN_SERVER_INIT_REPLS="${init}" \
        LEAN_SERVER_MAX_REPL_MEM=32G LEAN_SERVER_MAX_WAIT=3600 LEAN_SERVER_HEADER_TIMEOUT=900 \
        LEAN_SERVER_LOG_SNIPPETS=false LEAN_SERVER_MONITOR_REPLS=false LEAN_SERVER_PREWARM=true \
        LEAN_SERVER_CHECK_HEADER=true LEAN_SERVER_DIRECT_EXEC=true \
        LEAN_SERVER_LEAN_PATH="${KIMINA_LEAN_PATH}" LEAN_SERVER_LEAN_SYSROOT="${KIMINA_LEAN_SYSROOT}" \
        PYTHONPATH="${KIMINA_INSTALL}/kimina-lean-server/client:${KIMINA_INSTALL}/kimina-lean-server" \
        "${KIMINA_INSTALL}/venv/bin/python" -m server) > "${log}" 2>&1 &
    KIMINA_PID=$!
    KIMINA_LOG="${log}"
}

kimina_wait() {
    local url=$1 limit=$2 t0=${SECONDS}
    until curl -sf "${url}/health" > /dev/null; do
        kill -0 "${KIMINA_PID}" 2>/dev/null || { echo "ERROR: Kimina server exited" >&2; return 1; }
        # A server whose start-up failed can stay alive: Python then waits forever at exit for the REPLs that were
        # still starting (asyncio cannot cancel a process start it is in the middle of; job 3925556 waited 900 s).
        # uvicorn has logged the failure by then, so the caller can stop it.
        if grep -q "Application startup failed" "${KIMINA_LOG:-/dev/null}" 2>/dev/null; then
            echo "ERROR: Kimina server start-up failed after $((SECONDS - t0)) s (see ${KIMINA_LOG})" >&2
            return 1
        fi
        (( SECONDS - t0 > limit )) && { echo "ERROR: Kimina server not up after ${limit} s" >&2; return 1; }
        sleep 1
    done
    echo "Kimina server up after $((SECONDS - t0)) s"
}

kimina_repl_bin() {
    echo "${KIMINA_INSTALL}/repl-${REPL_TAG}/.lake/build/bin/repl"
}

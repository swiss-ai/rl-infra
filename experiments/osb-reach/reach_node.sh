#!/bin/bash
# Runs on the compute node inside the training image (started by run.sh through srun): starts the TCP
# listener, lets a sandbox connect to every address of this node (plus EXTRA_TARGETS), then shows what
# the listener saw.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/usr/bin/python3.12
PORT="${REACH_PORT:-$((5000 + RANDOM % 1000))}"

echo "== node $(hostname), listener port ${PORT}"
ip -o -4 addr show 2>/dev/null | awk '{print "   " $2, $4}' || hostname -I
"${PY}" "${HERE}/listen.py" "${PORT}" > listen.log 2>&1 &
LPID=$!
sleep 2

# Every global IPv4 address of this node, plus its hostname (DNS from inside the pod).
mapfile -t ADDRS < <(ip -o -4 addr show scope global 2>/dev/null | awk '{split($4, a, "/"); print a[1]}')
[[ ${#ADDRS[@]} -gt 0 ]] || read -r -a ADDRS <<< "$(hostname -I 2>/dev/null)"
[[ ${#ADDRS[@]} -gt 0 ]] || ADDRS=("$("${PY}" -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')")
TARGETS=()
for a in "${ADDRS[@]}"; do TARGETS+=("${a}:${PORT}"); done
TARGETS+=("$(hostname -f):${PORT}")
# Self-test from the node itself, so a dead listener is not mistaken for a blocked network.
"${PY}" "${HERE}/probe.py" "${ADDRS[0]}:${PORT}" | tail -1 | sed 's/^/   node self-test: /'
echo "   (the CONNECTION line it causes in the listener log is this self-test, source = the node itself)"
# Control: internet (expected to work).
TARGETS+=(pypi.org:443 ${EXTRA_TARGETS:-})

echo "== sandbox probes: ${TARGETS[*]}"
# The image's system Python lacks the OpenSandbox SDK and tenacity; the runs create sandboxes from
# Gym's harbor_agent server venv (built by the first Harbor run under NEMO_GYM_VENV_DIR), so use that.
if [[ -z "${REACH_PYTHON:-}" ]]; then
    for p in "${NEMO_GYM_VENV_DIR}/responses_api_agents/harbor_agent/.venv/bin/python" \
             "${NEMO_GYM_VENV_DIR}"/*/*/.venv/bin/python; do
        [[ -x "${p}" ]] && "${p}" -c "import opensandbox, tenacity, omegaconf, nemo_gym" 2>/dev/null \
            && { REACH_PYTHON="${p}"; break; }
    done
fi
[[ -n "${REACH_PYTHON:-}" ]] || { echo "ERROR: no Gym venv with opensandbox+tenacity under ${NEMO_GYM_VENV_DIR}; set REACH_PYTHON" >&2; kill "${LPID}"; exit 1; }
echo "== sandbox client python: ${REACH_PYTHON}"
"${REACH_PYTHON}" "${HERE}/reach.py" "${TARGETS[@]}"
RC=$?
sleep 2
kill "${LPID}" 2>/dev/null
echo "== listener log (a CONNECTION line from the sandbox = pods can reach this node)"
cat listen.log

# Reverse direction, the transport of a per-sandbox tunnel: node -> pod through the OpenSandbox server
# proxy, plain HTTP and WebSocket (ws.py / ws_server.py). REACH_WS=0 skips it.
if [[ "${REACH_WS:-1}" == 1 ]]; then
    echo "== node -> pod through the OpenSandbox proxy (HTTP + WebSocket)"
    "${REACH_PYTHON}" "${HERE}/ws.py" || RC=$?
fi
exit "${RC}"

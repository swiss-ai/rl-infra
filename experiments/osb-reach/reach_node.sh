#!/bin/bash
# Runs on the compute node inside the training image (started by run.sh through srun). REACH_TESTS
# selects: tcp (sandbox -> every address of this node, TCP listener), ws (node -> pod HTTP/WebSocket
# through the OpenSandbox proxy), tunnel (chisel reverse tunnel per sandbox, pod 127.0.0.1:8000 -> node),
# gymtunnel (the same through Gym's model_tunnel.py, patch 0013).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/usr/bin/python3.12
PORT="${REACH_PORT:-$((5000 + RANDOM % 1000))}"
TESTS=" ${REACH_TESTS:-tcp ws tunnel} "
RC=0

# The image's system Python lacks the OpenSandbox SDK and tenacity; the runs create sandboxes from
# Gym's harbor_agent server venv (built by the first Harbor run under NEMO_GYM_VENV_DIR), so use that.
echo "== node $(hostname) ($(uname -m)), $(date +%T): looking for the sandbox client python under ${NEMO_GYM_VENV_DIR}"
if [[ -z "${REACH_PYTHON:-}" ]]; then
    for p in "${NEMO_GYM_VENV_DIR}/responses_api_agents/harbor_agent/.venv/bin/python" \
             "${NEMO_GYM_VENV_DIR}"/*/*/.venv/bin/python; do
        [[ -x "${p}" ]] || continue
        echo "   $(date +%T) trying ${p}"
        # job 5253171 hung 20 min without output somewhere before the first test; bound every step
        timeout 180 "${p}" -X importtime -c "import opensandbox, tenacity, omegaconf, nemo_gym" 2>import_check.log \
            && { REACH_PYTHON="${p}"; break; }
        echo "   $(date +%T) failed (rc $?), last imports:"; tail -n 5 import_check.log | sed 's/^/     /'
    done
fi
[[ -n "${REACH_PYTHON:-}" ]] || { echo "ERROR: no Gym venv with opensandbox+tenacity under ${NEMO_GYM_VENV_DIR}; set REACH_PYTHON" >&2; exit 1; }
echo "== $(date +%T) tests:${TESTS}, sandbox client python: ${REACH_PYTHON}"

if [[ "${TESTS}" == *" tcp "* ]]; then
echo "== pod -> node TCP, listener port ${PORT}"

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
"${REACH_PYTHON}" "${HERE}/reach.py" "${TARGETS[@]}" || RC=$?
sleep 2
kill "${LPID}" 2>/dev/null
echo "== listener log (a CONNECTION line from the sandbox = pods can reach this node)"
cat listen.log
fi

# Reverse direction, the transport of a per-sandbox tunnel: node -> pod through the OpenSandbox server
# proxy, plain HTTP and WebSocket (ws.py / ws_server.py).
if [[ "${TESTS}" == *" ws "* ]]; then
    echo "== node -> pod through the OpenSandbox proxy (HTTP + WebSocket)"
    "${REACH_PYTHON}" "${HERE}/ws.py" || RC=$?
fi

# The tunnel itself: node opens it, pod uses it (tunnel.py / tunnel_probe.py / model_stub.py, chisel).
if [[ "${TESTS}" == *" tunnel "* ]]; then
    echo "== per-sandbox reverse tunnel (chisel): pod 127.0.0.1:8000 -> node"
    "${REACH_PYTHON}" "${HERE}/tunnel.py" || RC=$?
fi

# Gym's own tunnel code (patch 0013, ModelTunnel) through AsyncSandbox, as NemoGymSandboxEnvironment runs it.
if [[ "${TESTS}" == *" gymtunnel "* ]]; then
    echo "== Gym model_tunnel (patch 0013): ${TUNNELS:-32} sandboxes, OLD_START=${OLD_START:-0}"
    # with -X faulthandler, `timeout -s ABRT` makes a hang dump every thread's stack (GYM_TUNNEL_TIMEOUT_S)
    PYTHONUNBUFFERED=1 timeout -s ABRT "${GYM_TUNNEL_TIMEOUT_S:-900}" \
        "${REACH_PYTHON}" -X faulthandler "${HERE}/gym_tunnel.py"
    rc=$?
    echo "== $(date +%T) gym_tunnel.py exit ${rc} (124 or 134 = timeout)"
    [[ ${rc} -eq 0 ]] || RC=${rc}
fi
exit "${RC}"

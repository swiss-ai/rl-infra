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
# Controls: internet (expected to work), and the address of job 5236016 (nothing listening).
TARGETS+=(pypi.org:443 phillipmueller.ch:443 ${EXTRA_TARGETS:-})

echo "== sandbox probes: ${TARGETS[*]}"
"${PY}" "${HERE}/reach.py" "${TARGETS[@]}"
RC=$?
sleep 2
kill "${LPID}" 2>/dev/null
echo "== listener log (a CONNECTION line from the sandbox = pods can reach this node)"
cat listen.log
exit "${RC}"

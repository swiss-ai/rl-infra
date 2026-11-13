#!/bin/bash
# Run the Lean REPL inside bubblewrap: no network, no writable file system, and only the staged Lean copy plus
# the system's libraries visible. Lean runs code while it elaborates (`#eval`, `IO`, `run_cmd`), so a model's
# proof could otherwise read or delete files on scratch/home or open connections. The Kimina server itself
# (which needs the network to answer) stays outside; only its REPLs are confined.
#
#   LEAN_SERVER_REPL_PATH=<this script>  REPL_BIN=<the real repl binary>  LEAN_STAGE=<Lean copy in /dev/shm>
#
# The server starts REPL_PATH with LEAN_PATH/LEAN_SYSROOT set (direct_exec), which pass through.
# bubblewrap applies its mounts in order: /proc, /dev and /tmp come first, because the Lean copy lives under
# /dev/shm, and a later `--dev /dev` would mount a fresh /dev over it (the import then fails silently).
set -euo pipefail
: "${REPL_BIN:?}" "${LEAN_STAGE:?}"
exec bwrap \
    --proc /proc --dev /dev --tmpfs /tmp \
    --ro-bind /usr /usr --ro-bind-try /lib /lib --ro-bind-try /lib64 /lib64 --ro-bind-try /bin /bin \
    --ro-bind-try /etc/ld.so.cache /etc/ld.so.cache \
    --ro-bind "${LEAN_STAGE}" "${LEAN_STAGE}" --ro-bind "${REPL_BIN}" "${REPL_BIN}" \
    --chdir /tmp --unshare-all --die-with-parent --new-session \
    "${REPL_BIN}"

#!/bin/bash
# Body of the Lean sandbox job (start.sh submits it). The NeMo-Skills image is unpacked on node-local disk
# and its own server (/app/main.py under uWSGI) runs inside bubblewrap with that tree as its root: Lean runs
# code at compile time (#eval, IO), and in the enroot container this site mounts /scratch and /home01
# whatever pyxis is told. Only the network is shared, so Gym reaches port 6000; compiled code can still open
# sockets (the 0.7.1 image predates NeMo-Skills network blocking).
set -euo pipefail
ROOTFS="/tmp/lean-sandbox-${SLURM_JOB_ID}"
trap "rm -rf ${ROOTFS}" EXIT
echo "== unpacking ${LEAN_IMAGE} into ${ROOTFS}"
unsquashfs -q -no-xattrs -d "${ROOTFS}" "${LEAN_IMAGE}" > /dev/null
mkdir -p "${ROOTFS}/tmp/uwsgi"
# PATH and HOME of the image (Lean toolchain under /root/.elan).
bwrap --bind "${ROOTFS}" / --proc /proc --dev /dev --tmpfs /dev/shm \
    --unshare-user --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup-try --share-net \
    --uid 0 --gid 0 --die-with-parent --chdir /app --clearenv \
    --setenv HOME /root --setenv LEAN_PATH /lean4/my_project \
    --setenv PATH /lean4/my_project:/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --setenv PYTHONUNBUFFERED 1 \
    uwsgi --http-socket "0.0.0.0:${LEAN_SANDBOX_PORT:-6000}" --module main --callable app --master \
        --processes "${NUM_WORKERS}" --listen 4096 --http-timeout 300 --socket-timeout 300 \
        --die-on-term --disable-logging &
server=$!
for _ in $(seq 300); do
    if curl -sf "http://$(hostname):${LEAN_SANDBOX_PORT:-6000}/health" > /dev/null; then
        echo "$(hostname)" > "${ENDPOINT_FILE}"
        echo "== Lean sandbox up on $(hostname):${LEAN_SANDBOX_PORT:-6000}, ${NUM_WORKERS} workers"
        break
    fi
    kill -0 "${server}" 2> /dev/null || { echo "ERROR: sandbox exited" >&2; exit 1; }
    sleep 2
done
wait "${server}"

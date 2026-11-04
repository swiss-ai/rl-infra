#!/bin/bash
# Fetch chisel (github.com/jpillora/chisel, MIT) for the per-sandbox model tunnel (Gym patch 0012) into
# CHISEL_DIR, checked against the release's published sha256 (pinned here). Run on a login node (needs
# internet); idempotent. Both architectures: arm64 for the GH200 nodes (client), amd64 for the
# OpenSandbox pods on Breithorn (server, uploaded into every sandbox).
#   CHISEL_DIR=$SCRATCH/chisel bash backends/nemo-rl/apertus2/fetch-chisel.sh
# Leaves chisel_<version>_linux_{amd64,arm64} in CHISEL_DIR, the layout model_tunnel.chisel_dir expects.
set -euo pipefail
CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}"
CHISEL_DIR="${CHISEL_DIR:?set CHISEL_DIR}"
declare -A SHA256=(  # chisel_1.12.0_checksums.txt (release 2026-08-29)
    [1.12.0_amd64]=f3f180f1d93aa72cce4e6386f98cc06569a0146fbd65eb4423cf83e6434bcfe6
    [1.12.0_arm64]=2ec6152cd2c74fe0146d4d79e4e7aa174521368c56e433d55e023a92ea404ec3
)
mkdir -p "${CHISEL_DIR}"
for arch in amd64 arm64; do
    sum="${SHA256[${CHISEL_VERSION}_${arch}]:-}"
    [[ -n "${sum}" ]] || { echo "ERROR: no pinned sha256 for chisel ${CHISEL_VERSION} ${arch}; add it here" >&2; exit 1; }
    f="${CHISEL_DIR}/chisel_${CHISEL_VERSION}_linux_${arch}"
    [[ -x "${f}" ]] && continue
    curl -fsSL -o "${f}.gz" \
        "https://github.com/jpillora/chisel/releases/download/v${CHISEL_VERSION}/chisel_${CHISEL_VERSION}_linux_${arch}.gz"
    echo "${sum}  ${f}.gz" | sha256sum -c --quiet - || { echo "ERROR: chisel ${arch} checksum mismatch" >&2; rm -f "${f}.gz"; exit 1; }
    gunzip -f "${f}.gz"
    chmod +x "${f}"
done
echo "== chisel ${CHISEL_VERSION} in ${CHISEL_DIR}"

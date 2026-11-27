#!/bin/bash
# Fetch the pi release archive (github.com/earendil-works/pi, MIT) into PI_DIR once per run, checked against the
# release's published sha256 (pinned here). Run on a login node (needs internet); idempotent. Gym patch 0048
# (PiNemoGym) uploads it from there into every sandbox, so the sandboxes need no internet and no Node.
#   PI_DIR=$SCRATCH/pi bash backends/nemo-rl/apertus2/fetch-pi.sh
# Leaves PI_DIR/<version>/<asset>, the layout harbor_agent_kwargs.pi_archive_dir expects.
set -euo pipefail
PI_VERSION="${PI_VERSION:-1.5.2}"
PI_DIR="${PI_DIR:?set PI_DIR}"
# glibc x86_64, built for bun-linux-x64-baseline (no AVX2 needed); the Gym adapter's default asset
ASSET=pi-linux-x64.tar.gz
declare -A SHA256=(  # "digest" of the asset in the GitHub release API; the same pin as the Gym adapter
    [1.5.2]=811c6f2c9e7aaee252b5404252e556fb713aa209c1f7f924226aa77d4ef70cb5
)
sum="${SHA256[${PI_VERSION}]:-}"
[[ -n "${sum}" ]] || { echo "ERROR: no pinned sha256 for pi ${PI_VERSION}; add it here" >&2; exit 1; }
f="${PI_DIR}/${PI_VERSION}/${ASSET}"
if [[ -s "${f}" ]] && echo "${sum}  ${f}" | sha256sum -c --quiet - 2>/dev/null; then
    echo "== pi ${PI_VERSION} in ${f}"
    exit 0
fi
mkdir -p "$(dirname "${f}")"
tmp="${f}.part.$$"
trap 'rm -f "${tmp}"' EXIT
curl -fsSL --retry 3 -o "${tmp}" "https://github.com/earendil-works/pi/releases/download/v${PI_VERSION}/${ASSET}"
echo "${sum}  ${tmp}" | sha256sum -c --quiet - || { echo "ERROR: pi ${PI_VERSION} checksum mismatch" >&2; exit 1; }
# a rename, never an overwrite in place: running jobs that read the old file keep reading it
mv -f "${tmp}" "${f}"
echo "== pi ${PI_VERSION} fetched to ${f}"

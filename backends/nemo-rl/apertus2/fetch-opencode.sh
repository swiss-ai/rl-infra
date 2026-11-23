#!/bin/bash
# Fetch the OpenCode release archive (github.com/anomalyco/opencode, MIT) into OPENCODE_DIR once per run, checked
# against the release's published sha256 (pinned here). Run on a login node (needs internet); idempotent. Gym
# patch 0045 (opencode_install: upload) uploads it from there into every sandbox, so the sandboxes need no
# internet and no python3/curl/wget (job 5427428: 8 of 640 trials failed setup on an image without them).
#   OPENCODE_DIR=$SCRATCH/opencode bash backends/nemo-rl/apertus2/fetch-opencode.sh
# Leaves OPENCODE_DIR/<version>/<asset>, the layout harbor_agent_kwargs.opencode_archive_dir expects.
set -euo pipefail
OPENCODE_VERSION="${OPENCODE_VERSION:-1.17.11}"
OPENCODE_DIR="${OPENCODE_DIR:?set OPENCODE_DIR}"
# glibc x86_64 without AVX2 (the Breithorn sandboxes are amd64); the Gym adapter's default asset
ASSET=opencode-linux-x64-baseline.tar.gz
declare -A SHA256=(  # "digest" of the asset in the GitHub release API; the same pin as the Gym adapter
    [1.17.11]=36064327d5a3fba662264f5726f517eafa26d1aefb0d59bc95e66bddccf3497b
)
sum="${SHA256[${OPENCODE_VERSION}]:-}"
[[ -n "${sum}" ]] || { echo "ERROR: no pinned sha256 for OpenCode ${OPENCODE_VERSION}; add it here" >&2; exit 1; }
f="${OPENCODE_DIR}/${OPENCODE_VERSION}/${ASSET}"
if [[ -s "${f}" ]] && echo "${sum}  ${f}" | sha256sum -c --quiet - 2>/dev/null; then
    echo "== OpenCode ${OPENCODE_VERSION} in ${f}"
    exit 0
fi
mkdir -p "$(dirname "${f}")"
tmp="${f}.part.$$"
trap 'rm -f "${tmp}"' EXIT
curl -fsSL --retry 3 -o "${tmp}" \
    "https://github.com/anomalyco/opencode/releases/download/v${OPENCODE_VERSION}/${ASSET}"
echo "${sum}  ${tmp}" | sha256sum -c --quiet - || { echo "ERROR: OpenCode ${OPENCODE_VERSION} checksum mismatch" >&2; exit 1; }
# a rename, never an overwrite in place: running jobs that read the old file keep reading it
mv -f "${tmp}" "${f}"
echo "== OpenCode ${OPENCODE_VERSION} fetched to ${f}"

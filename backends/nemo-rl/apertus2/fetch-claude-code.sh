#!/bin/bash
# Fetch the Claude Code CLI (Anthropic's native release binary, downloads.claude.ai) into CC_DIR once per run,
# checked against the sha256 pinned here (the checksum in the release's manifest.json), and keep it gzipped. Run on a
# login node (needs internet); idempotent. Gym patch 0050 (ClaudeCodeNemoGym) uploads the .gz into every sandbox and
# checks the binary's sha256 there again, so the sandboxes need no internet, curl, Node.js or npm, only gzip.
#   CC_DIR=$SCRATCH/claude-code bash backends/nemo-rl/apertus2/fetch-claude-code.sh
#   CC_VERSION (2.1.326), CC_PLATFORM (linux-x64, the pods; linux-arm64 for the probe on Alps login nodes)
# Leaves CC_DIR/<version>/claude-<platform>.gz (gzip -n: no name or time stamp) and .gz.sha256 (the digest of the
# .gz, which the adapter checks on the node before each upload), the layout harbor_agent_kwargs.claude_archive_dir
# expects.
# Why gzip: the linux-x64 binary of 2.1.326 is 371 MB and goes through the OpenSandbox gateway into each pod; the
# release's own claude.zst is smaller (134 MB) but task images rarely have zstd.
set -euo pipefail
CC_VERSION="${CC_VERSION:-2.1.326}"
CC_PLATFORM="${CC_PLATFORM:-linux-x64}"
CC_DIR="${CC_DIR:?set CC_DIR}"
declare -A SHA256=(  # platforms.<platform>.checksum in <version>/manifest.json; the same pins as the Gym adapter
    [2.1.326/linux-x64]=b6b3647d981130dc44fc670846ae97b6cde531ffb667cf9c4d17cfcafb38cb37
    [2.1.326/linux-arm64]=98d56f6ffe53469767a74864a38eaf359d62ef557a8de1b1ad524d61a66ec119
)
sum="${SHA256[${CC_VERSION}/${CC_PLATFORM}]:-}"
[[ -n "${sum}" ]] || { echo "ERROR: no pinned sha256 for Claude Code ${CC_VERSION} ${CC_PLATFORM}; add it here" >&2; exit 1; }
gz="${CC_DIR}/${CC_VERSION}/claude-${CC_PLATFORM}.gz"
# The .gz is written only after its binary matched the pin, so a .gz that matches its sidecar is that binary.
if [[ -s "${gz}" && -s "${gz}.sha256" && "$(sha256sum "${gz}" | cut -d' ' -f1)" == "$(cat "${gz}.sha256")" ]]; then
    echo "== Claude Code ${CC_VERSION} (${CC_PLATFORM}) in ${gz}"
    exit 0
fi
mkdir -p "$(dirname "${gz}")"
tmp="${gz%.gz}.part.$$"
trap 'rm -f "${tmp}" "${tmp}.gz" "${tmp}.gz.sha256"' EXIT
curl -fsSL --retry 3 -o "${tmp}" "https://downloads.claude.ai/claude-code-releases/${CC_VERSION}/${CC_PLATFORM}/claude"
echo "${sum}  ${tmp}" | sha256sum -c --quiet - \
    || { echo "ERROR: Claude Code ${CC_VERSION} ${CC_PLATFORM}: checksum mismatch" >&2; exit 1; }
gzip -n -6 -c "${tmp}" > "${tmp}.gz"
sha256sum "${tmp}.gz" | cut -d' ' -f1 > "${tmp}.gz.sha256"
# renames, never an overwrite in place: running jobs that read the old file keep reading it
mv -f "${tmp}.gz" "${gz}"
mv -f "${tmp}.gz.sha256" "${gz}.sha256"
# the size as uploaded (du would show Lustre's compressed allocation)
echo "== Claude Code ${CC_VERSION} (${CC_PLATFORM}) fetched to ${gz} ($(( $(stat -c %s "${gz}") / 1000000 )) MB)"

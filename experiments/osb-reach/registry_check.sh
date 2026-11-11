#!/bin/bash
# Can the SWE-Lego image be pulled anonymously, and with the run's placeholder credentials?
# pending.py (2026-11-11): every pull of rl-registry.swissai.svc.cscs.ch/swe_lego_v05_1/... got
# "HEAD .../manifests/sha256:...: 401 Unauthorized" on the Breithorn node.
# Does what a pull does: HEAD the manifest (a registry answers 401 + a token URL even for public
# projects), get a token from that URL (A: anonymous, B: anonymous/unused as run-harbor-osb.sh
# sends), HEAD the manifest again with the token. Tokens are not printed.
#   bash experiments/osb-reach/registry_check.sh [image@sha256:...]
# Result lines: "manifest with token -> HTTP 200" = pull allowed, 401/403/404 = not.
set -uo pipefail
IMAGE="${1:-rl-registry.swissai.svc.cscs.ch/swe_lego_v05_1/adamchainz__time-machine-298@sha256:cc8246f55a927fcb7b7739ab45721d11f99b86611704afcab826d65e5415900c}"
HOST="${IMAGE%%/*}"
REST="${IMAGE#*/}"
REPO="${REST%@*}"
DIGEST="${REST#*@}"
URL="https://${HOST}/v2/${REPO}/manifests/${DIGEST}"
ACCEPT='Accept: application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json'

echo "== ${URL}"
hdr="$(curl -sS -m 30 -I -H "${ACCEPT}" "${URL}" 2>&1)"
echo "1. manifest without token -> $(head -n1 <<< "${hdr}" | tr -d '\r')"
www="$(grep -i '^www-authenticate:' <<< "${hdr}" | tr -d '\r')"
echo "   ${www:-no WWW-Authenticate header}"
realm="$(sed -nE 's/.*realm="([^"]+)".*/\1/p' <<< "${www}")"
service="$(sed -nE 's/.*service="([^"]+)".*/\1/p' <<< "${www}")"
[[ -n "${realm}" ]] || { echo "no token URL; stop"; exit 1; }

try() {  # $1 label, rest: extra curl args for the token request
    local label="$1"; shift
    local body code token
    body="$(curl -sS -m 30 -G -w '\n%{http_code}' "$@" "${realm}" \
        --data-urlencode "service=${service}" --data-urlencode "scope=repository:${REPO}:pull" 2>&1)"
    code="$(tail -n1 <<< "${body}")"
    token="$(head -n -1 <<< "${body}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("token") or d.get("access_token") or "")' 2>/dev/null)"
    echo "${label}: token request -> HTTP ${code}, token $([[ -n "${token}" ]] && echo received || echo none)"
    [[ "${code}" == 200 && -n "${token}" ]] || { head -n -1 <<< "${body}" | head -c 300; echo; return; }
    echo "${label}: manifest with token -> HTTP $(curl -sS -m 30 -o /dev/null -I -w '%{http_code}' \
        -H "${ACCEPT}" -H "Authorization: Bearer ${token}" "${URL}")"
}
try "2. A anonymous"
try "3. B anonymous/unused (the run's image_auth)" -u "${RL_REGISTRY_USERNAME:-anonymous}:${RL_REGISTRY_PASSWORD:-unused}"

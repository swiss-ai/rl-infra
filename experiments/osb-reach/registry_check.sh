#!/bin/bash
# Can the SWE-Lego image be pulled anonymously, and with the run's placeholder credentials?
# pending.py (2026-11-11): every pull of rl-registry.swissai.svc.cscs.ch/swe_lego_v05_1/... got
# "HEAD .../manifests/sha256:...: 401 Unauthorized" on the Breithorn node.
# Does what a pull does: HEAD the manifest (a registry answers 401 + a token URL even for public
# projects), get a token from that URL (A: anonymous, B: anonymous/unused as run-harbor-osb.sh
# sends), C: the robot account in RL_REGISTRY_CRED_FILE (default ~/.rl_registry_cred, `name:password`,
# what run-s.sh passes to Gym patch 0028; handed to curl on stdin, not on the command line), HEAD the
# manifest again with the token. Tokens and passwords are not printed.
#   bash experiments/osb-reach/registry_check.sh [image@sha256:...]
# Result lines: "manifest with token -> HTTP 200" = pull allowed, 401/403/404 = not. The token is issued even
# for wrong credentials (2026-11-11); its `sub` says whether they were accepted ('' = anonymous) and
# `access` what they allow ("actions": [] = no pull permission on the repository).
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

try() {  # $1 label, $2 curl config text for the token request (stdin), rest: extra curl args
    local label="$1" conf="$2"; shift 2
    local body code token
    body="$(curl -sS -m 30 -G -w '\n%{http_code}' -K - "$@" "${realm}" <<< "${conf}" \
        --data-urlencode "service=${service}" --data-urlencode "scope=repository:${REPO}:pull" 2>&1)"
    code="$(tail -n1 <<< "${body}")"
    token="$(head -n -1 <<< "${body}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("token") or d.get("access_token") or "")' 2>/dev/null)"
    echo "${label}: token request -> HTTP ${code}, token $([[ -n "${token}" ]] && echo received || echo none)"
    [[ "${code}" == 200 && -n "${token}" ]] || { head -n -1 <<< "${body}" | head -c 300; echo; return; }
    # the token's readable part: who it was issued to (empty = anonymous) and what it allows
    echo "${label}: token for $(python3 -c '
import base64, json, sys
part = sys.argv[1].split(".")[1]
claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
print("sub=%r access=%s" % (claims.get("sub", ""), json.dumps(claims.get("access"))))' "${token}" 2>/dev/null || echo "(not a JWT)")"
    echo "${label}: manifest with token -> HTTP $(curl -sS -m 30 -o /dev/null -I -w '%{http_code}' -K - \
        -H "${ACCEPT}" "${URL}" <<< "header = \"Authorization: Bearer ${token}\"")"
}
try "2. A anonymous" ""
try "3. B anonymous/unused (the old placeholder image_auth)" \
    "user = \"${RL_REGISTRY_USERNAME:-anonymous}:${RL_REGISTRY_PASSWORD:-unused}\""
cred_file="${RL_REGISTRY_CRED_FILE:-${HOME}/.rl_registry_cred}"
if [[ -r "${cred_file}" ]]; then
    cred="$(head -n1 "${cred_file}" | tr -d '\r\n')"
    [[ "${cred}" == ?*:?* ]] || { echo "4. C ${cred_file}: not one name:password line"; exit 1; }
    cred="${cred//\\/\\\\}"; cred="${cred//\"/\\\"}"   # curl config quoting
    try "4. C robot account (${cred_file}, name ${cred%%:*})" "user = \"${cred}\""
    # the projects the account can see (this Harbor treats wrong credentials as anonymous: only `public`)
    api="$(curl -sS -m 30 -w '\n%{http_code}' -K - "https://${HOST}/api/v2.0/projects?page_size=100" \
        <<< "user = \"${cred}\"" 2>&1)"
    echo "4. C Harbor API /projects with the account -> HTTP $(tail -n1 <<< "${api}"): $(head -n -1 <<< "${api}" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(", ".join("%s(public=%s)" % (p.get("name"), (p.get("metadata") or {}).get("public")) for p in d) if isinstance(d, list) else str(d)[:200])' 2>/dev/null)"
    unset cred
else
    echo "4. C no ${cred_file}"
fi

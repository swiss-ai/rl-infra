#!/bin/bash
# Where is the SWE-Lego image now? registry_check.sh (2026-11-11): the robot account is accepted and sees
# 42 projects, but not swe_lego_v05_1 (manifest 401, no pull permission). Asks Harbor's API with the account:
#   1. does project swe_lego_v05_1 still exist (HEAD /projects?project_name=: 200 yes, 404 no)?
#   2. which visible project holds the image: the same repository name, and the same digest (= the
#      identical image) or another one?
# Credentials from RL_REGISTRY_CRED_FILE (default ~/.rl_registry_cred, `name:password`), handed to curl on
# stdin; nothing secret is printed.
#   bash experiments/osb-reach/registry_find.sh [image@sha256:...]
set -uo pipefail
IMAGE="${1:-rl-registry.swissai.svc.cscs.ch/swe_lego_v05_1/adamchainz__time-machine-298@sha256:cc8246f55a927fcb7b7739ab45721d11f99b86611704afcab826d65e5415900c}"
HOST="${IMAGE%%/*}"
REST="${IMAGE#*/}"
PROJECT="${REST%%/*}"
REPO="${REST#*/}"; REPO="${REPO%@*}"
DIGEST="${REST#*@}"
API="https://${HOST}/api/v2.0"
cred_file="${RL_REGISTRY_CRED_FILE:-${HOME}/.rl_registry_cred}"
cred="$(head -n1 "${cred_file}" 2>/dev/null | tr -d '\r\n')"
[[ "${cred}" == ?*:?* ]] || { echo "no name:password line in ${cred_file}"; exit 1; }
cred="${cred//\\/\\\\}"; cred="${cred//\"/\\\"}"   # curl config quoting
CONF="user = \"${cred}\""
unset cred
get() {  # $1 method, $2 URL -> "<code>\n<body>"
    curl -sS -m 30 -X "$1" -w '\n%{http_code}' -K - "$2" <<< "${CONF}" 2>&1
}
code() { tail -n1 <<< "$1"; }
body() { head -n -1 <<< "$1"; }
# Harbor double-encodes a repository name with slashes in the path
repo_enc="$(python3 -c 'import sys, urllib.parse as u; print(u.quote(u.quote(sys.argv[1], safe=""), safe=""))' "${REPO}")"

echo "== image ${PROJECT}/${REPO}@${DIGEST:0:19}..."
r="$(curl -sS -m 30 -I -o /dev/null -w '%{http_code}' -K - "${API}/projects?project_name=${PROJECT}" <<< "${CONF}")"
echo "1. project ${PROJECT} exists? HEAD /projects?project_name= -> HTTP ${r} (200 yes, 404 no; 401/403: this account may not ask)"

r="$(get GET "${API}/projects?page_size=100")"
projects="$(body "${r}" | python3 -c 'import json,sys; print("\n".join(p["name"] for p in json.load(sys.stdin)))' 2>/dev/null)"
echo "2. $(wc -l <<< "${projects}") visible projects; looking for repository ${REPO} in each:"
found=0
while read -r p; do
    [[ -n "${p}" ]] || continue
    r="$(get GET "${API}/projects/${p}/repositories/${repo_enc}")"
    [[ "$(code "${r}")" == 200 ]] || continue
    found=1
    a="$(get GET "${API}/projects/${p}/repositories/${repo_enc}/artifacts/${DIGEST}")"
    if [[ "$(code "${a}")" == 200 ]]; then
        echo "   ${p}/${REPO}: SAME digest (identical image)"
    else
        latest="$(get GET "${API}/projects/${p}/repositories/${repo_enc}/artifacts?page_size=3&sort=-push_time")"
        echo "   ${p}/${REPO}: other digest(s): $(body "${latest}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("; ".join("%s... pushed %s tags %s" % (a.get("digest", "")[:19], a.get("push_time"), [t.get("name") for t in a.get("tags") or []]) for a in d))' 2>/dev/null)"
    fi
done <<< "${projects}"
[[ "${found}" == 1 ]] || echo "   in none of them"

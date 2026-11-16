#!/bin/bash
# Prints the address or the API key of a running Kimina server job, for a training job that was submitted BEFORE
# the server started (train.sh):
#   ...kimina_url=$(KIMINA_ENDPOINT_FILE=<file> bash endpoint.sh url)
#   ...kimina_api_key=$(KIMINA_ENDPOINT_FILE=<file> bash endpoint.sh key)
# train.sh puts these on the training's command line unexpanded. ray.sub writes the driver command to a file and
# runs it with bash on the head node, in the container, once the Ray cluster is up, so they run at job start.
# It waits until server_job.sh's endpoint files exist AND the server answers /health, so a file left by an earlier
# server job is never used. After KIMINA_ENDPOINT_WAIT seconds (default 1800: another debug job of the same user
# can delay the server) it prints nothing and fails; the Gym server then refuses the empty kimina_url and the run
# ends early.
# Only the value goes to stdout; messages go to stderr (the driver log). The key is not in the submitted command
# or the sbatch script; it is filled in at job start. It is made per server job and is worthless once that job
# ends.
set -u
what="${1:?usage: KIMINA_ENDPOINT_FILE=<file> endpoint.sh url|key}"
F="${KIMINA_ENDPOINT_FILE:?KIMINA_ENDPOINT_FILE is the endpoint file of the server job}"
LIMIT="${KIMINA_ENDPOINT_WAIT:-1800}"

alive() {
    if command -v curl > /dev/null; then
        curl -sf -m 5 "$1/health" > /dev/null 2>&1
    else
        python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/health", timeout=5)' "$1" 2> /dev/null
    fi
}

t0=${SECONDS}
while true; do
    if [[ -s "${F}" && -s "${F}.key" ]]; then
        url="$(< "${F}")"
        alive "${url}" && break
    fi
    if (( SECONDS - t0 >= LIMIT )); then
        echo "kimina-endpoint: no live Kimina server behind ${F} after ${LIMIT} s" >&2
        exit 1
    fi
    sleep 5
done
case "${what}" in
    url) echo "kimina-endpoint: ${url} answered after $(( SECONDS - t0 )) s" >&2; printf '%s' "${url}" ;;
    key) printf '%s' "$(< "${F}.key")" ;;
    *)   echo "kimina-endpoint: unknown argument ${what}" >&2; exit 2 ;;
esac

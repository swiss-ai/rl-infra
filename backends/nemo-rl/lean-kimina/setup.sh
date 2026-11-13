#!/bin/bash
# Install the Kimina Lean Server for the Lean task into KIMINA_INSTALL (default $SCRATCH/lean-kimina/install);
# safe to re-run. Runs on a login node, about a minute:
#   - kimina-lean-server (swiss-ai's fork) cloned at the pin;
#   - a Python venv (KIMINA_PYTHON, default python3.11) with requirements.txt;
#   - the Lean REPL at REPL_TAG, built with LEAN_TOOLCHAIN.
# It adds no Lean toolchain and no Mathlib build: LEAN_TOOLCHAIN and LEAN_PROJECT (see env) are only read, and
# must be at the REPL's Lean version.
#   bash backends/nemo-rl/lean-kimina/setup.sh
set -euo pipefail
LEAN_KIMINA_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${LEAN_KIMINA_DIR}/../../.." && pwd)"
source "${ROOT}/sites/load.sh" && rl_infra_load "${LEAN_KIMINA_DIR}/env"
source "${LEAN_KIMINA_DIR}/pin"
source "${LEAN_KIMINA_DIR}/kimina.sh"

lean_version="${REPL_TAG#v}"
[[ -n "${LEAN_TOOLCHAIN}" && -n "${LEAN_PROJECT}" ]] || {
    echo "ERROR: set LEAN_TOOLCHAIN (a Lean ${REPL_TAG} toolchain directory) and LEAN_PROJECT (a Lake project with" >&2
    echo "       Mathlib compiled for it), e.g. in site.env; see backends/nemo-rl/lean-kimina/README.md" >&2
    exit 1
}
[[ -x "${LEAN_TOOLCHAIN}/bin/lean" ]] || { echo "ERROR: no Lean toolchain at LEAN_TOOLCHAIN=${LEAN_TOOLCHAIN}" >&2; exit 1; }
"${LEAN_TOOLCHAIN}/bin/lean" --version | grep -q "version ${lean_version}[,)]" \
    || { echo "ERROR: ${LEAN_TOOLCHAIN} is not Lean ${lean_version}: $("${LEAN_TOOLCHAIN}/bin/lean" --version)" >&2; exit 1; }
grep -q "v${lean_version}\$" "${LEAN_PROJECT}/lean-toolchain" 2>/dev/null \
    || { echo "ERROR: LEAN_PROJECT=${LEAN_PROJECT} is not a Lake project on Lean ${lean_version}" >&2; exit 1; }
ls -d "${LEAN_PROJECT}"/.lake/packages/mathlib/.lake/build/lib/lean > /dev/null \
    || { echo "ERROR: no compiled Mathlib in ${LEAN_PROJECT}/.lake/packages/mathlib" >&2; exit 1; }
mkdir -p "${KIMINA_INSTALL}"

# The server at the pin (swiss-ai's fork). A clone made from another URL (older versions of this script cloned
# upstream and applied a patch) is moved to the pin's URL and commit, and its working tree reset.
K="${KIMINA_INSTALL}/kimina-lean-server"
if [[ ! -d "${K}/.git" ]]; then
    echo "== cloning kimina-lean-server into ${K}"
    git clone --quiet "${KIMINA_URL}" "${K}"
fi
git -C "${K}" remote set-url origin "${KIMINA_URL}"
git -C "${K}" remote set-url --push origin no_push
# Only changes to tracked files count: the server writes __pycache__ into the clone when it runs.
if [[ "$(git -C "${K}" rev-parse HEAD)" != "${KIMINA_COMMIT}" \
      || -n "$(git -C "${K}" status --porcelain --untracked-files=no)" ]]; then
    echo "== kimina-lean-server at ${KIMINA_COMMIT:0:9}"
    git -C "${K}" fetch --quiet origin "${KIMINA_COMMIT}"
    git -C "${K}" clean -fdxq   # untracked files of an older patch would be in the checkout's way
    git -C "${K}" checkout --quiet --force --detach "${KIMINA_COMMIT}"
    rm -f "${K}/.git/rl-infra-patches"
fi

# The venv, reinstalled when requirements.txt changes.
V="${KIMINA_INSTALL}/venv"
[[ -x "${V}/bin/python" ]] || "${KIMINA_PYTHON}" -m venv "${V}"
req_stamp="$(git hash-object "${LEAN_KIMINA_DIR}/requirements.txt")"
if [[ "$(cat "${V}/.rl-infra-requirements" 2>/dev/null)" != "${req_stamp}" ]]; then
    echo "== installing requirements.txt into ${V}"
    "${V}/bin/python" -m pip install --quiet --no-cache-dir --upgrade pip
    "${V}/bin/python" -m pip install --quiet --no-cache-dir --only-binary :all: -r "${LEAN_KIMINA_DIR}/requirements.txt"
    echo "${req_stamp}" > "${V}/.rl-infra-requirements"
fi

# The REPL, built with the given toolchain's lake (not elan's), so nothing outside KIMINA_INSTALL changes.
R="${KIMINA_INSTALL}/repl-${REPL_TAG}"
if [[ ! -x "$(kimina_repl_bin)" ]]; then
    echo "== building the Lean REPL ${REPL_TAG} in ${R}"
    [[ -d "${R}/.git" ]] || git -c advice.detachedHead=false clone --quiet --depth 1 --branch "${REPL_TAG}" "${REPL_URL}" "${R}"
    # Lake's cache would default to a directory inside the (read-only for us) toolchain.
    (cd "${R}" && LAKE_CACHE_DIR="${KIMINA_INSTALL}/lake-cache" "${LEAN_TOOLCHAIN}/bin/lake" build 2>&1 | tail -n 2)
fi

# Self-checks: the server imports, and the REPL answers on the toolchain.
PYTHONPATH="${K}/client:${K}" "${V}/bin/python" -c "import server.main, kimina_client.stream" > /dev/null
answer=$(printf '{"cmd": "#check Nat"}\n\n' | LEAN_SYSROOT="${LEAN_TOOLCHAIN}" LEAN_PATH= timeout 120 "$(kimina_repl_bin)")
grep -q '"Nat : Type"' <<< "${answer}" || { echo "ERROR: the REPL did not answer #check Nat: ${answer}" >&2; exit 1; }
echo "== Kimina Lean Server ready in ${KIMINA_INSTALL}: $(git -C "${K}" log --oneline -1 | cut -c1-60), $("${V}/bin/python" --version), REPL ${REPL_TAG}"

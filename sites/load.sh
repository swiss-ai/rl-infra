# Loads the rl-infra configuration into the calling shell (source it, do not run it):
#   source "${ROOT}/sites/load.sh" && rl_infra_load [<stack env>...]
# (A function, as a sourced file without arguments would see the caller's own arguments.)
# Every file only sets defaults (`: "${VAR:=...}"`), so the first one to set a variable wins:
#   1. the environment at launch time
#   2. site.env in the repository root (personal overrides, optional, gitignored)
#   3. sites/<SITE>.env, the cluster profile (filesystems, account, container runtime, ...)
#   4. the stack env files given as arguments (e.g. backends/nemo-rl/apertus2/env), in order
#   5. backends/nemo-rl/env, the stock NeMo-RL stack
# SITE is detected from the filesystems when it is not set.
_RL_INFRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rl_infra_load() {
    if [[ -f "${_RL_INFRA_ROOT}/site.env" ]]; then
        # The site.env of the old template set the stack's image and clone paths, which would now
        # win over the stack env files (it used to be the other way round).
        if grep -qE '^: "\$\{(IMAGE_MASTER|EDF_NAME):=' "${_RL_INFRA_ROOT}/site.env"; then
            echo "ERROR: ${_RL_INFRA_ROOT}/site.env is from the old template (it sets IMAGE_MASTER/EDF_NAME)." >&2
            echo "       Move it aside (mv site.env site.env.old) and copy site.env.example again." >&2
            return 1
        fi
        source "${_RL_INFRA_ROOT}/site.env"
    fi

    if [[ -z "${SITE:-}" ]]; then
        if [[ -d /iopsstor && -d /capstor ]]; then
            SITE=alps
        elif [[ -d /home01 && -d /appsdata ]]; then
            SITE=hangang
        else
            echo "ERROR: cannot tell which cluster this is; set SITE (one of: $(cd "${_RL_INFRA_ROOT}/sites" && ls *.env | sed 's/\.env$//' | tr '\n' ' '))" >&2
            return 1
        fi
    fi
    [[ -f "${_RL_INFRA_ROOT}/sites/${SITE}.env" ]] || { echo "ERROR: no profile sites/${SITE}.env" >&2; return 1; }
    source "${_RL_INFRA_ROOT}/sites/${SITE}.env"

    local env_file
    for env_file in "$@"; do
        source "${env_file}"
    done
    source "${_RL_INFRA_ROOT}/backends/nemo-rl/env"
    export SITE
}

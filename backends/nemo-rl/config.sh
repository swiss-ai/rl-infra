# Config helpers for run.sh and preflight.sh (source it). yq v3, no Python on the login node.
#
#   config_resolve <file> <out>      merged config of <file> and its `defaults` chain, as NeMo-RL merges it
#   config_freeze  <file> <dir> [nodefaults]
#                                    copies <file> and every file of its `defaults` chain into <dir>, with the
#                                    `defaults` entries pointing at the copies; sets FROZEN to the copy of <file>
#
# NeMo-RL (nemo_rl/utils/config.py) resolves each parent of a file on its own, merges the parents in list order
# (a later one wins), then the file itself; lists are replaced, not merged, and a top-level section with
# `_override_: true` replaces the section below it. Entries are relative to the file that names them; an
# absolute /opt/nemo-rl/... entry is a file of the NeMo-RL clone and stays as it is (it belongs to the code).
#
# rl-infra keys: top-level keys read by run.sh, not NeMo-RL config keys. config_freeze leaves them out of the
# copies: recipe, size, context, job_time, pydeps, script, template.
RL_INFRA_KEYS='recipe|size|context|job_time|pydeps|script|template'

_config_host_path() {
    if [[ "$1" == /opt/nemo-rl/* ]]; then echo "${NEMO_RL_DIR:?NEMO_RL_DIR is not set}/${1#/opt/nemo-rl/}"; else echo "$1"; fi
}

# The `defaults` entries of a file, one per line, as written (a string or a list).
_config_defaults() {
    yq r "$1" defaults 2>/dev/null | sed -e 's/^- //' -e '/^$/d'
}

# Path of a `defaults` entry of <file>, on the host.
_config_parent() {
    local file=$1 entry=$2
    if [[ "${entry}" == /* ]]; then _config_host_path "${entry}"; else realpath -m "$(dirname "${file}")/${entry}"; fi
}

config_resolve() {
    local file=$1 out=$2 base part entry key
    [[ -f "${file}" ]] || { echo "ERROR: config file not found: ${file}" >&2; return 1; }
    yq r "${file}" > /dev/null 2>&1 || { echo "ERROR: ${file} does not parse" >&2; return 1; }
    base="$(mktemp)"; echo '{}' > "${base}"
    while read -r entry; do
        [[ -n "${entry}" ]] || continue
        # CONFIG_SKIP_MISSING_NEMO_RL=1: a file of a NeMo-RL clone that setup.sh has not made yet is left out
        # (run.sh reads its own keys before setup; NeMo-RL's files hold none of them).
        if [[ "${entry}" == /opt/nemo-rl/* && "${CONFIG_SKIP_MISSING_NEMO_RL:-0}" == 1 \
              && ! -f "$(_config_parent "${file}" "${entry}")" ]]; then
            continue
        fi
        part="$(mktemp)"
        config_resolve "$(_config_parent "${file}" "${entry}")" "${part}" || return 1
        _config_merge "${base}" "${part}"
        rm -f "${part}"
    done < <(_config_defaults "${file}")
    _config_merge "${base}" "${file}"
    yq d -i "${base}" defaults
    mv "${base}" "${out}"
}

# Merges <over> into <base> in place, with NeMo-RL's _override_ rule for top-level sections.
_config_merge() {
    local base=$1 over=$2 key tmp
    for key in $(yq r "${over}" --printMode p '*._override_' 2>/dev/null | sed 's/\._override_$//'); do
        [[ "$(yq r "${over}" "${key}._override_")" == true ]] && yq d -i "${base}" "${key}"
    done
    tmp="$(mktemp)"
    yq m -x -a overwrite "${base}" "${over}" > "${tmp}"
    for key in $(yq r "${tmp}" --printMode p '*._override_' 2>/dev/null); do yq d -i "${tmp}" "${key}"; done
    mv "${tmp}" "${base}"
}

# The copy keeps every other byte of the file: only the `defaults` block and the rl-infra keys are rewritten.
config_freeze() {
    local file=$1 dir=$2 nodefaults=${3:-} entry parents=() copy
    mkdir -p "${dir}"
    if [[ -z "${nodefaults}" ]]; then
        while read -r entry; do
            [[ -n "${entry}" ]] || continue
            if [[ "${entry}" == /opt/nemo-rl/* ]]; then
                parents+=("${entry}")
            else
                config_freeze "$(_config_parent "${file}" "${entry}")" "${dir}" || return 1
                parents+=("${FROZEN}")
            fi
        done < <(_config_defaults "${file}")
    fi
    _CONFIG_FROZEN_N=$(( ${_CONFIG_FROZEN_N:-0} + 1 ))
    copy="${dir}/$(printf '%02d' "${_CONFIG_FROZEN_N}")-$(basename "${file}")"
    {
        echo "# Copy of ${file} made at launch; the job reads this file, not the original."
        if (( ${#parents[@]} )); then
            echo "defaults:"
            printf '  - %s\n' "${parents[@]}"
        fi
        awk -v keys="^(${RL_INFRA_KEYS}):" '
            /^defaults:/ { in_defaults = 1; next }
            in_defaults && /^[[:space:]]+-/ { next }
            { in_defaults = 0 }
            $0 ~ keys { next }
            { print }
        ' "${file}"
    } > "${copy}"
    FROZEN="${copy}"
}

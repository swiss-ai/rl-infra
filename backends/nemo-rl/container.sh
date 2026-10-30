# Container spec helpers, sourced by setup.sh and launch.sh. A stack describes its container once,
# as an EDF template (CONTAINER_SPEC, e.g. apertus2/edf.toml.template), and the cluster profile's
# CONTAINER_RUNTIME decides how it is used:
#   edf    CSCS Container Engine: the rendered template is installed as ~/.edf/<EDF_NAME>.toml
#          and jobs run with --environment=<EDF_NAME>.
#   pyxis  enroot/pyxis (e.g. HANGANG, which has no EDF support): the rendered template becomes
#          CONTAINER, MOUNTS and PYXIS_EXTRA_ARGS for ray.sub, and its [env] (plus the profile's
#          SITE_ENV) is exported for the job, as pyxis passes the host environment in.

# Renders CONTAINER_SPEC to stdout: @@VAR@@ placeholders from the environment, and the
# @@SITE_MOUNTS@@ line as one mount per line (the profile's SITE_MOUNTS).
container_render() {
    awk -v mounts="${SITE_MOUNTS[*]-}" '
        /^[[:space:]]*@@SITE_MOUNTS@@[[:space:]]*$/ {
            n = split(mounts, m, " ")
            for (i = 1; i <= n; i++) printf "  \"%s\",\n", m[i]
            next
        }
        { print }
    ' "${CONTAINER_SPEC}" | sed \
        -e "s|@@IMAGE_WORK@@|${IMAGE_WORK}|g" \
        -e "s|@@NEMO_RL_DIR@@|${NEMO_RL_DIR}|g" \
        -e "s|@@SCRATCH@@|${SCRATCH}|g" \
        -e "s|@@PYDEPS@@|${PYDEPS:-}|g" \
        -e "s|@@MEGATRON_DIR@@|${MEGATRON_DIR:-}|g" \
        -e "s|@@BRIDGE_DIR@@|${BRIDGE_DIR:-}|g" \
        -e "s|@@VLLM_RUNNER@@|${VLLM_RUNNER:-}|g" \
        -e "s|@@VLLM_APERTUS2@@|${VLLM_APERTUS2:-}|g"
}

# edf: installs the rendered spec as ~/.edf/<EDF_NAME>.toml (the old one is backed up if it differs).
container_install_edf() {
    local edf_file="${HOME}/.edf/${EDF_NAME}.toml" rendered
    mkdir -p "${HOME}/.edf"
    rendered="$(mktemp)"
    container_render > "${rendered}"
    if [[ -f "${edf_file}" ]] && cmp -s "${rendered}" "${edf_file}"; then
        echo "== EDF up to date: ${edf_file}"
        rm -f "${rendered}"
    else
        if [[ -f "${edf_file}" ]]; then
            cp "${edf_file}" "${edf_file}.bak.$(date +%Y%m%d-%H%M%S)"
            echo "== EDF changed; previous version backed up next to it"
        fi
        mv "${rendered}" "${edf_file}"
        echo "== EDF written: ${edf_file}"
    fi
}

# SITE_PRELOAD: C sources (relative to the repository) built here into shared libraries in
# ${OUT}/preload, rebuilt when the source is newer, and added to SITE_ENV as LD_PRELOAD.
container_build_preload() {
    local src lib libs=()
    for src in "${SITE_PRELOAD[@]+"${SITE_PRELOAD[@]}"}"; do
        lib="${OUT}/preload/lib$(basename "${src}" .c).so"
        if [[ ! "${lib}" -nt "${_RL_INFRA_ROOT}/${src}" ]]; then
            mkdir -p "${OUT}/preload"
            # Build beside it and rename, so a running job never maps a half-written library.
            gcc -O2 -Wall -shared -fPIC -o "${lib}.$$" "${_RL_INFRA_ROOT}/${src}" \
                && mv -f "${lib}.$$" "${lib}" || { echo "ERROR: cannot build ${src}" >&2; return 1; }
        fi
        libs+=("${lib}")
    done
    if (( ${#libs[@]} )); then
        SITE_ENV+=("LD_PRELOAD=$(IFS=:; echo "${libs[*]}")")
    fi
}

# pyxis: sets CONTAINER, MOUNTS and PYXIS_EXTRA_ARGS (exported, for ray.sub) and exports the
# spec's [env] and SITE_ENV. EDF-only parts are dropped: [annotations] (CSCS hooks) and PATH
# (the CE does not apply the image's own environment, pyxis does). Every exported variable is
# listed in --container-env so that it also replaces a value the image sets itself.
container_pyxis_env() {
    local kind a b mounts=() names=() writable=0 kv
    container_build_preload || return 1
    CONTAINER=""
    while IFS=$'\t' read -r kind a b; do
        case "${kind}" in
            IMAGE) CONTAINER="${a}" ;;
            MOUNT) [[ "${a}" == *:* ]] && mounts+=("${a}") || mounts+=("${a}:${a}") ;;
            WRITABLE) writable=1 ;;
            ENV) [[ "${a}" == PATH ]] && continue
                 export "${a}=${b}"; names+=("${a}") ;;
        esac
    done < <(container_render | awk '
        function unquote(s) { gsub(/^[[:space:]]*"|"[[:space:]]*,?[[:space:]]*$/, "", s); return s }
        /^[[:space:]]*(#|$)/ { next }
        /^\[/ { section = $0; next }
        in_mounts && /^[[:space:]]*\]/ { in_mounts = 0; next }
        in_mounts { print "MOUNT\t" unquote($0); next }
        section == "" && /^image[[:space:]]*=/ { sub(/^[^=]*=/, ""); print "IMAGE\t" unquote($0); next }
        section == "" && /^mounts[[:space:]]*=[[:space:]]*\[/ { in_mounts = 1; next }
        section == "" && /^writable[[:space:]]*=[[:space:]]*true/ { print "WRITABLE"; next }
        section == "[env]" && /=/ {
            key = $0; sub(/[[:space:]]*=.*/, "", key); gsub(/^[[:space:]]+/, "", key)
            val = $0; sub(/^[^=]*=/, "", val)
            print "ENV\t" key "\t" unquote(val)
        }
    ')
    for kv in "${SITE_ENV[@]+"${SITE_ENV[@]}"}"; do
        export "${kv}"; names+=("${kv%%=*}")
    done
    [[ -n "${CONTAINER}" ]] || { echo "ERROR: no image in ${CONTAINER_SPEC}" >&2; return 1; }
    MOUNTS="$(IFS=,; echo "${mounts[*]-}")"
    PYXIS_EXTRA_ARGS=""
    (( writable )) && PYXIS_EXTRA_ARGS+=" --container-writable"
    (( ${#names[@]} )) && PYXIS_EXTRA_ARGS+=" --container-env=$(IFS=,; echo "${names[*]}")"
    PYXIS_EXTRA_ARGS="${PYXIS_EXTRA_ARGS# }"
    export CONTAINER MOUNTS PYXIS_EXTRA_ARGS
}

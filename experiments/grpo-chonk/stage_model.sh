#!/bin/bash
# Stage the Chonk HF checkpoint on scratch and make it loadable by Transformers
# (safe to re-run; also the recovery path after the scratch purge):
#   1. copy the shards from the persistent master (parallel, skips complete files)
#   2. add the HF modeling code, which the export does not ship (config only),
#      from swiss-ai/hfconverter at a pinned commit
#   3. point auto_map at it so AutoModelForCausalLM works with trust_remote_code
# vLLM does not need step 2/3: it has its own Apertus2KDAForCausalLM.
# CHONK_MODEL_STAGE=link (sites/hangang.env) instead links DST to a source that already went
# through all three steps (a shared copy that is only read), after checking its modeling code.
# SRC and the mode come from the cluster profile (sites/<SITE>.env).
set -euo pipefail

source "$(dirname "$0")/../../sites/load.sh" && rl_infra_load
SRC="${SRC:-${CHONK_MODEL_SRC}}"
DST="${DST:-${SCRATCH}/models/apertus2-chonk-lc-iter3576}"
JOBS="${JOBS:-8}"
# hfconverter apertus2/main; its configuration_apertus2.py adds linear_attn_a_log_per_channel
# (default False = per-head A_log, which matches this checkpoint's A_log shape [28]).
HFCONV_COMMIT="${HFCONV_COMMIT:-c2ab4910948b}"
# Fetched through the GitHub API.
HFCONV_API="https://api.github.com/repos/swiss-ai/hfconverter/contents"

if [[ "${CHONK_MODEL_STAGE}" == link ]]; then
    [[ "$(cat "${SRC}/MODELING_SOURCE" 2>/dev/null)" == "hfconverter ${HFCONV_COMMIT}" ]] \
        || { echo "ERROR: ${SRC} lacks the hfconverter ${HFCONV_COMMIT} modeling code; stage a copy instead (CHONK_MODEL_STAGE=copy)" >&2; exit 1; }
    grep -q '"AutoModelForCausalLM": "modeling_apertus2.Apertus2ForCausalLM"' "${SRC}/config.json" \
        || { echo "ERROR: ${SRC}/config.json has no auto_map for the modeling code" >&2; exit 1; }
    if [[ -L "${DST}" && "$(readlink "${DST}")" == "${SRC}" ]]; then
        echo "== link up to date: ${DST} -> ${SRC}"
    else
        [[ ! -e "${DST}" || -L "${DST}" ]] || { echo "ERROR: ${DST} exists and is not a link" >&2; exit 1; }
        mkdir -p "$(dirname "${DST}")"
        ln -sfn "${SRC}" "${DST}"
        echo "== linked ${DST} -> ${SRC}"
    fi
    n=$(find "${SRC}/" -maxdepth 1 -name '*.safetensors' | wc -l)
    n_index=$(jq -r '.weight_map[]' "${SRC}/model.safetensors.index.json" | sort -u | wc -l)
    echo "== ${n}/${n_index} shards in ${SRC}"
    [[ "${n}" == "${n_index}" ]]
    exit
fi

mkdir -p "${DST}"
lfs setstripe -c 8 -S 4M "${DST}" 2>/dev/null || true

# 1. Shards and small files; per-file temp name + rename, so DST only holds complete files.
copy_one() {
    local src="$1" dst="$2/$(basename "$1")"
    if [[ -f "${dst}" && $(stat -c %s "${dst}") == $(stat -c %s "${src}") ]]; then return 0; fi
    cp "${src}" "${dst}.tmp.$$" && mv "${dst}.tmp.$$" "${dst}"
}
export -f copy_one
echo "== copying ${SRC} -> ${DST} (${JOBS} parallel)"
find "${SRC}" -maxdepth 1 -type f ! -name configuration_apertus2.py ! -name config.json -print0 \
    | xargs -0 -P "${JOBS}" -I{} bash -c 'copy_one "$1" "$2"' _ {} "${DST}"

# 2. Modeling + matching configuration code (the checkpoint's own configuration file predates
#    the modeling file's version, so both come from the same commit).
#    The repository is public; a GitHub token in ~/.git-credentials is used when present.
token="$(grep -m1 'github.com' ~/.git-credentials 2>/dev/null | sed -E 's#https://[^:]*:([^@]*)@.*#\1#' || true)"
auth=(); [[ -n "${token}" ]] && auth=(-H "Authorization: Bearer ${token}")
for f in modeling_apertus2.py configuration_apertus2.py; do
    curl -fsSL ${auth[@]+"${auth[@]}"} -H "Accept: application/vnd.github.raw" \
        "${HFCONV_API}/${f}?ref=${HFCONV_COMMIT}" -o "${DST}/${f}.tmp" && mv "${DST}/${f}.tmp" "${DST}/${f}"
done
echo "hfconverter ${HFCONV_COMMIT}" > "${DST}/MODELING_SOURCE"

# 3. config.json with the model classes registered (architectures unchanged: vLLM selects
#    its model by that name). Text insert, not jq: jq 1.6 rewrites -5.0 as -5, and the strict
#    Apertus2Config then rejects the int gate_lower_bound. Every other byte stays the export's.
sed 's#^\(    "AutoConfig": "configuration_apertus2.Apertus2Config"\)$#\1,\n    "AutoModelForCausalLM": "modeling_apertus2.Apertus2ForCausalLM",\n    "AutoModel": "modeling_apertus2.Apertus2Model"#' \
    "${SRC}/config.json" > "${DST}/config.json.tmp"
[[ $(diff "${SRC}/config.json" "${DST}/config.json.tmp" | grep -c '^>') == 3 ]] \
    || { echo "ERROR: auto_map insert did not apply cleanly" >&2; exit 1; }
mv "${DST}/config.json.tmp" "${DST}/config.json"

n_src=$(find "${SRC}" -maxdepth 1 -name '*.safetensors' | wc -l)
n_dst=$(find "${DST}" -maxdepth 1 -name '*.safetensors' | wc -l)
echo "== done: ${n_dst}/${n_src} shards, $(du -sh "${DST}" | cut -f1) in ${DST}"
[[ "${n_dst}" == "${n_src}" ]]

#!/bin/bash
# One-time login-node setup for talking to Chonk SFT checkpoints (safe to re-run; also the recovery
# path after the scratch purge). Needs no job and no GPU:
#   1. a Python 3.13 venv with apertus-common, the format's reference library (the image has 3.12)
#   2. the instruct tokenizer on its own, so that the client never opens a model directory
#   3. the openai overlay that the image's vLLM server needs
#   4. the sources and the config-only HF reference that convert.sh needs
set -euo pipefail

ROOT="${ROOT:-/scratch/${USER}/chonk-sft}"
UV_VERSION=0.13.6
COMMON_COMMIT=494c2be79080aa8fbcd46409b96560f72707cd69     # tokenized the v0.1 SFT data (run.json)
TOKENIZER_COMMIT=b801ff1b00d933d10bba90b0c22307157299faa7  # Apertus_2_instruct, as trained
# The commits of the first conversion that passed parity (job 1185634, all three clean).
HFCONVERTER_COMMIT=acf1dbefb4fe453332bdbc3cd22b8fe43fd0bcfa
BRIDGE_COMMIT=6ccdb3c369491374f1fef4320886e8bb1394bb3e
MEGATRON_COMMIT=42e7a63aaf49e78e33ccfdd0373a423d198b9fb0

mkdir -p "${ROOT}"/{logs,results,src,tokenizer}

export UV_CACHE_DIR="${ROOT}/uv-cache" UV_PYTHON_INSTALL_DIR="${ROOT}/python"
if [[ ! -x "${ROOT}/uv/bin/uv" ]]; then
    /usr/bin/python3.12 -m pip install -q --target "${ROOT}/uv" "uv==${UV_VERSION}"
fi
uv="${ROOT}/uv/bin/uv"

# 1. Reference library.
"${uv}" python install -q 3.13
[[ -x "${ROOT}/venv/bin/python" ]] || "${uv}" venv -q --python 3.13 "${ROOT}/venv"
"${uv}" pip install -q --python "${ROOT}/venv/bin/python" numpy pyarrow \
    "apertus-common[hf] @ git+https://github.com/swiss-ai/apertus-common@${COMMON_COMMIT}"

# Blobless clone at one commit; the checkout fetches what the worktree needs.
clone() {
    local url="$1" dir="$2" commit="$3" ref="${4:-$3}"
    [[ -d "${dir}/.git" ]] || git clone -q --filter=blob:none --no-checkout "${url}" "${dir}"
    git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null || git -C "${dir}" fetch -q origin "${ref}"
    git -C "${dir}" checkout -q --detach "${commit}"
}

# 2. Tokenizer.
clone https://github.com/swiss-ai/apertus-omni-tokenizer "${ROOT}/src/apertus-omni-tokenizer" "${TOKENIZER_COMMIT}"
cp "${ROOT}/src/apertus-omni-tokenizer/tokenizers/Apertus_2_instruct/"* "${ROOT}/tokenizer/"

# 3. The image's openai (2.7.2) is too old for its own vLLM server (ImportError: NamespaceTool).
if [[ ! -f "${ROOT}/pydeps-vllm/.complete" ]]; then
    "${uv}" pip install -q --python /usr/bin/python3.12 --no-deps --target "${ROOT}/pydeps-vllm" openai==2.44.0
    touch "${ROOT}/pydeps-vllm/.complete"
fi

# 4. Conversion sources and the reference: chonk config from hfconverter, instruct tokenizer.
clone https://github.com/swiss-ai/hfconverter "${ROOT}/src/hfconverter" "${HFCONVERTER_COMMIT}"
clone https://github.com/swiss-ai/Megatron-Bridge "${ROOT}/src/Megatron-Bridge" "${BRIDGE_COMMIT}"
clone https://github.com/swiss-ai/Megatron-LM-MoE "${ROOT}/src/Megatron-LM-MoE-bridge" "${MEGATRON_COMMIT}" pull/88/head
mkdir -p "${ROOT}/reference-chonk-instruct"
cp "${ROOT}/src/hfconverter/examples/apertus2-chonk-hf-reference/"{config.json,configuration_apertus2.py} \
    "${ROOT}/reference-chonk-instruct/"
cp "${ROOT}/tokenizer/"{tokenizer.json,tokenizer_config.json,special_tokens_map.json,generation_config.json} \
    "${ROOT}/reference-chonk-instruct/"
echo "tokenizer: apertus-omni-tokenizer ${TOKENIZER_COMMIT}" > "${ROOT}/reference-chonk-instruct/SOURCES.txt"
echo "config: hfconverter ${HFCONVERTER_COMMIT} examples/apertus2-chonk-hf-reference" >> "${ROOT}/reference-chonk-instruct/SOURCES.txt"

echo "== ready: ${ROOT}"
"${ROOT}/venv/bin/python" -I -c "
import apertus_common as c
e = c.load_encoding('${ROOT}/tokenizer')
print('== apertus-common', e.controls, 'wait', e.generation_constraints.eos_token_id)" 2>/dev/null

#!/bin/bash
# Python overlay for posttraining on top of the container's PyTorch (run inside the container; train.sh and check.sh
# call it). posttraining's tokenization needs Transformers 4.x, where apply_chat_template returns a
# tensor; the other versions follow installation/docker-arm64-cuda/requirements-freeze.txt where an
# aarch64 build exists (bitsandbytes for optim=ademamix). Everything goes in with --no-deps so the
# container's torch, flash-attn, datasets and accelerate stay; the list holds the missing dependencies.
set -euo pipefail
: "${SFT_PYDEPS:?}"

PKGS=(
    transformers==4.57.6 tokenizers==0.22.1 huggingface-hub==0.36.0
    trl==0.16.0 peft==0.15.1
    bitsandbytes==0.50.2
    deepspeed==0.16.9 hjson==3.1.0 py-cpuinfo==9.0.0
)
STAMP="${PKGS[*]}"
if [[ "$(cat "${SFT_PYDEPS}/.complete" 2>/dev/null)" == "${STAMP}" ]]; then
    echo "== overlay up to date: ${SFT_PYDEPS}"; exit 0
fi
tmp="${SFT_PYDEPS}.tmp.$$"
mkdir -p "$(dirname "${SFT_PYDEPS}")"
# DeepSpeed is a source package; DS_BUILD_OPS=0 builds no CUDA kernels now (they compile on first use),
# and --no-build-isolation lets its setup see the container's torch.
DS_BUILD_OPS=0 uv pip install --python "$(command -v python3)" --no-deps --no-build-isolation \
    --target "${tmp}" "${PKGS[@]}"
printf '%s' "${STAMP}" > "${tmp}/.complete"
rm -rf "${SFT_PYDEPS}"
mv "${tmp}" "${SFT_PYDEPS}"
echo "== overlay installed: ${SFT_PYDEPS}"

#!/bin/bash
# Packages David's image lacks for the Gym path, installed beside the image (no dependencies pulled in, so no
# image package is shadowed). Runs inside the container; launch-harbor-osb.sh and chonk_serve.sh call it.
#   openai==2.44.0           Gym pins it; the image's 2.7.2 also breaks vLLM's server import (NamespaceTool)
#   devtools, yappi, pydot   Gym dependencies missing from the image ("No module named 'devtools'" at Gym start-up)
#   asttokens, executing     devtools' own dependencies, also missing
# Same idea as backends/nemo-rl/apertus2/pydeps.sh. Not KISTI-specific: Alps runs of this image need it too.
set -euo pipefail
: "${PYDEPS_KISTI:?set by kisti/env.sh}"
PKGS=("openai==2.44.0" "devtools==0.13.0" "yappi==1.7.6" "pydot==4.0.1" "asttokens==3.0.2" "executing==2.3.0")

if [[ "$(cat "${PYDEPS_KISTI}/.complete" 2>/dev/null)" != "${PKGS[*]}" ]]; then
    # The image's uv points at CSCS's JFrog mirror and its cache directory is read-only.
    export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX_KISTI:-https://pypi.org/simple}" UV_CACHE_DIR="${KISTI_SCRATCH}/.cache/uv"
    tmp="${PYDEPS_KISTI}.tmp.$$"
    uv pip install --python /usr/bin/python3.12 --no-deps --target "${tmp}" "${PKGS[@]}"
    printf '%s' "${PKGS[*]}" > "${tmp}/.complete"
    rm -rf "${PYDEPS_KISTI}"
    mv "${tmp}" "${PYDEPS_KISTI}"
fi

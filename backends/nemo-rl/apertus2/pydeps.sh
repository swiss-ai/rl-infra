#!/bin/bash
# Runtime overlay for the Apertus 2 image: a directory of Python packages the image lacks, installed with
# `uv pip install --target` and put on PYTHONPATH by the container spec (PYDEPS). Run inside the container at
# the start of a job (driver.sh, check_env.sh): it installs the overlay when it is missing or was built from
# another package list, which its stamp file (.complete) records, and does nothing otherwise.
#   PYDEPS             the overlay directory (run.sh names it after its package list, so each list has its own)
#   PYDEPS_PKGS        extra packages, separated by "|": the task's `pydeps:` list and, for NeMo Gym tasks, what
#                      Gym needs (PYDEPS_GYM_PKGS in apertus2/env); empty for the plain overlay
#   PYDEPS_PKGS_PY313  packages that declare Python 3.13 but run on the image's 3.12 (apertus-common, set by
#                      run.sh when the recipe uses the Apertus 2 message format)
# The overlay is built in a temporary directory and swapped in when complete, so a job never imports from a
# half-built one.
set -euo pipefail
: "${PYDEPS:?PYDEPS is set in the container spec}"

# What every job needs and this image build lacks (the image's own Chonk example installs the same,
# alps-extended-images#64):
PKGS=(
    # nemo_rl.telemetry imports nemo.lens; rev = the clone's pyproject.toml pin
    "nemo-lens[sdk] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b85578fc2b736a1804705e537001b5f45e9c715d"
    # NeMo-RL's dataset registry imports the audio datasets at module level
    soundfile==0.13.1
)
if [[ -n "${PYDEPS_PKGS:-}" ]]; then IFS='|' read -r -a extra <<< "${PYDEPS_PKGS}"; PKGS+=("${extra[@]}"); fi
PKGS_PY313=()
if [[ -n "${PYDEPS_PKGS_PY313:-}" ]]; then IFS='|' read -r -a PKGS_PY313 <<< "${PYDEPS_PKGS_PY313}"; fi
STAMP="${PKGS[*]}${PKGS_PY313[*]:+ ${PKGS_PY313[*]}}"

if [[ "$(cat "${PYDEPS}/.complete" 2>/dev/null)" != "${STAMP}" ]]; then
    mkdir -p "$(dirname "${PYDEPS}")"
    tmp="${PYDEPS}.tmp.$$"
    # No uv cache: the overlay is built only when its package list changes, and a scratch purge leaves the cache
    # with entries whose files are gone (uv then fails on a missing METADATA; Clariden 2026-11-20).
    export UV_NO_CACHE=1
    uv pip install --python /usr/bin/python3.12 --no-deps --target "${tmp}" "${PKGS[@]}"
    if (( ${#PKGS_PY313[@]} )); then
        # It asks for Python 3.13 and tokenizers 0.23.2, but runs on the image's 3.12 and 0.23.1.
        uv pip install --python /usr/bin/python3.12 --python-version 3.13 --no-deps --target "${tmp}" "${PKGS_PY313[@]}"
    fi
    printf '%s' "${STAMP}" > "${tmp}/.complete"
    rm -rf "${PYDEPS}"
    mv "${tmp}" "${PYDEPS}"
fi

#!/bin/bash
# Runtime overlay for the Apertus 2 image (PYDEPS, on PYTHONPATH via the EDF): packages NeMo-RL
# imports at module level that this image build lacks. Same overlay as the image's own Chonk
# example (alps-extended-images#64). Installed again when the list changes or the scratch purge
# removed it. Run inside the container (driver.sh and check_env.sh call it).
set -euo pipefail
: "${PYDEPS:?PYDEPS is set in the EDF}"

PKGS=(
    # nemo_rl.telemetry imports nemo.lens; rev = the clone's pyproject.toml pin
    "nemo-lens[sdk] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b85578fc2b736a1804705e537001b5f45e9c715d"
    # NeMo-RL's dataset registry imports the audio datasets at module level
    soundfile==0.13.1
)
# NeMo Gym tasks (PYDEPS_GYM, set by run.sh with an overlay of their own, so these never reach the
# overlay other jobs run from): the openai Gym pins (the image's 2.7.2 lacks what Gym and vLLM's
# server import), and devtools, yappi and pydot with devtools' two dependencies, missing from the image.
if [[ -n "${PYDEPS_GYM:-}" ]]; then
    PKGS+=(openai==2.44.0 devtools==0.13.0 yappi==1.7.6 pydot==4.0.1 asttokens==3.0.2 executing==2.3.0)
fi
# Reasoning Gym scored in NeMo-RL's math environment (PYDEPS_RG, set by run.sh for a task with
# `pydeps: reasoning_gym`, overlay of its own): reasoning-gym and the dependencies the image lacks.
# The image has numpy, sympy, matplotlib, pyyaml, pytz, rich and tabulate (0.10.0 for the 0.9.0 pin).
# pycosat has no wheels and compiles here (gcc-13 and the Python 3.12 headers are in the image).
# 0.1.25, as Gym's reasoning_gym server resolves (>=0.1.19): 0.1.19 lacks coin_flip, kakurasu and survo
# (its __version__ still says 0.1.19 in 0.1.25).
if [[ -n "${PYDEPS_RG:-}" ]]; then
    PKGS+=(reasoning-gym==0.1.25 arckit==0.1.0 drawsvg==2.4.2 bfi==1.0.4 cellpylib==2.4.0 magiccube==0.3.0
           pycosat==0.6.6 pyfiglet==1.0.2 zss==1.2.0)
fi
# Tasks in the Apertus 2 message format (PYDEPS_APERTUS2, set by run.sh with an overlay of their own,
# so this never reaches the overlay other jobs run from) install apertus-common from the checkout
# setup.sh made. It asks for Python 3.13 and tokenizers 0.23.2, but runs on the image's 3.12 and
# 0.23.1, and encodes the SFT data exactly there.
PKGS_PY313=()
if [[ -n "${PYDEPS_APERTUS2:-}" ]]; then
    PKGS_PY313=("${PYDEPS_APERTUS2}")
fi
STAMP="${PKGS[*]}${PKGS_PY313[*]:+ ${PKGS_PY313[*]}}"

if [[ "$(cat "${PYDEPS}/.complete" 2>/dev/null)" != "${STAMP}" ]]; then
    mkdir -p "$(dirname "${PYDEPS}")"
    tmp="${PYDEPS}.tmp.$$"
    uv pip install --python /usr/bin/python3.12 --no-deps --target "${tmp}" "${PKGS[@]}"
    if (( ${#PKGS_PY313[@]} )); then
        uv pip install --python /usr/bin/python3.12 --python-version 3.13 --no-deps --target "${tmp}" "${PKGS_PY313[@]}"
    fi
    printf '%s' "${STAMP}" > "${tmp}/.complete"
    rm -rf "${PYDEPS}"
    mv "${tmp}" "${PYDEPS}"
fi

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

if [[ "$(cat "${PYDEPS}/.complete" 2>/dev/null)" != "${PKGS[*]}" ]]; then
    mkdir -p "$(dirname "${PYDEPS}")"
    tmp="${PYDEPS}.tmp.$$"
    uv pip install --python /usr/bin/python3.12 --no-deps --target "${tmp}" "${PKGS[@]}"
    printf '%s' "${PKGS[*]}" > "${tmp}/.complete"
    rm -rf "${PYDEPS}"
    mv "${tmp}" "${PYDEPS}"
fi

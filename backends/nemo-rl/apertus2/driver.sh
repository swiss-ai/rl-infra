#!/bin/bash
# Driver entry on the Ray head inside the Apertus 2 NeMo-RL container (cwd /opt/nemo-rl).
# Everything runs on the image's system Python (the image sets NEMO_RL_PY_EXECUTABLES_SYSTEM=1,
# so Ray actors use it too). First (re)installs the runtime overlay (pydeps.sh) if it is missing
# or out of date. Exports RL_INFRA_DIR (this repository) for configs that reference files in it.
set -euo pipefail

export RL_INFRA_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
bash "$(dirname "$0")/pydeps.sh"
# W&B key for runs with WANDB_PROJECT (launch.sh); the container's HOME is not the user's.
if [[ -n "${WANDB_KEY_FILE:-}" && -r "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY="$(< "${WANDB_KEY_FILE}")"
fi
# W&B does not copy the driver's console to the run: the driver prints every Ray worker's output,
# several hundred thousand lines at the start of a 48-node megachonk run. W&B sends each write
# through one socket, and once wandb-core falls behind, every further write in the client costs
# time proportional to the backlog (Python 3.12 asyncio sums its whole write buffer on each
# write). In job 1299422 the client was 32 minutes behind after half an hour and then stopped,
# and the first wandb.log of the run waited on it forever. The console stays in ray-driver.log.
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
# W&B's console wrapper used to write the driver's prints through at once. Without it, Python
# buffers stdout in blocks of the file's st_blksize (4 MiB on Lustre), so the step lines reach
# ray-driver.log hours late and hang_watch.sh takes a healthy driver for a hung one (job 1300476).
export PYTHONUNBUFFERED=1

exec /usr/bin/python3.12 "$@"

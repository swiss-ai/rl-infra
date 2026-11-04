#!/bin/bash
# Download a Hugging Face repository to scratch from a compute node, in the Apertus 2 container, which
# has huggingface_hub and hf_xet. Much faster than envs/stage_judge_model.sh on a login node for a big
# model. HF_DEST defaults to $SCRATCH/models/<name>, and finished files are skipped on a re-run:
#   HF_REPO=openai/gpt-oss-safeguard-120b CHECK_SCRIPT=download_model.sh JOB_TIME=01:00:00 \
#       bash backends/nemo-rl/apertus2/check_env_job.sh
set -euo pipefail

export HF_REPO DEST="${HF_DEST:-${SCRATCH}/models/$(basename "${HF_REPO:?HF_REPO names the repository}")}"
export HF_WORKERS="${HF_WORKERS:-16}" HF_XET_HIGH_PERFORMANCE=1
# No chunk cache: it would keep a second copy of the model on scratch.
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0

/usr/bin/python3.12 -c '
import os, time
from huggingface_hub import snapshot_download

start = time.time()
path = snapshot_download(os.environ["HF_REPO"], local_dir=os.environ["DEST"], max_workers=int(os.environ["HF_WORKERS"]))
print(f"== {os.environ['"'"'HF_REPO'"'"']} in {path}, {time.time() - start:.0f} s", flush=True)
'
du -sh "${DEST}"

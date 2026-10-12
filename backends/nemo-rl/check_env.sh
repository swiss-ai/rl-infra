#!/bin/bash
# Sanity check of the container: run it INSIDE the nemo-rl EDF on a GPU node (cwd
# /opt/nemo-rl) before the first job, e.g. from an interactive allocation.
set -u

echo "== user / dirs";   id; ls -ld /root /opt/nemo-rl; pwd
echo "== python";        which python uv; python -V
echo "== imports";       python -c "import nemo_rl, torch; print('nemo_rl', nemo_rl.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
echo "== worker venvs";  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo "== fingerprint";   diff <(python tools/generate_fingerprint.py) /opt/nemo_rl_container_fingerprint && echo "fingerprint OK"
echo "== env";           echo "NRL_CONTAINER=${NRL_CONTAINER:-<unset>} NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR:-<unset>}"; echo "HF_HOME=$HF_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME UV_CACHE_DIR=${UV_CACHE_DIR:-<unset>}"

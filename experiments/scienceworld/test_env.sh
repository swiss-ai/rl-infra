#!/bin/bash
# CPU check of the ScienceWorld environment (test_env.py) with the sciworld NeMo-RL copy; submit from /scratch/$USER:
#   sbatch --job-name=sciworld-envtest --export=ALL,CMD="bash <this dir>/test_env.sh" <this dir>/cpu.sbatch
set -euo pipefail
export PYTHONPATH=/home01/hanint06/projects/nemo-rl-apertus2-sciworld:/scratch/hanint06/nemo-rl-apertus2/pydeps-apertus2:/home01/hanint06/sciworld/pydeps
export RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false
cd /scratch/hanint06
python3 -u /home01/hanint06/projects/rl-infra-sciworld/experiments/scienceworld/test_env.py

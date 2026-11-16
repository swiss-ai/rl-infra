#!/bin/bash
# check_env.sh, plus what it leaves out for a run on an SFT checkpoint: apertus-common imported from the
# overlay, and the checkpoint's encoding loaded, as NeMo-RL patches 0022, 0023 and 0025 do. Runs inside the
# container through check_env_job.sh:
#   CHECK_SCRIPT=check_env_sft.sh MODEL=$SCRATCH/models/<staged checkpoint> PARTITION=debug \
#       bash backends/nemo-rl/apertus2/check_env_job.sh
# Writes nothing besides what check_env.sh writes (the overlay and the caches).
set -u
APERTUS2_DIR="${APERTUS2_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL="${MODEL:?MODEL is the staged checkpoint}"

bash "${APERTUS2_DIR}/check_env.sh"
echo "== check_env.sh exit $?"

echo "== apertus-common from the overlay"
MODEL="${MODEL}" /usr/bin/python3.12 -c "
import os, apertus_common
print('apertus_common', apertus_common.__file__)
print('inside the overlay:', apertus_common.__file__.startswith(os.environ['PYDEPS'] + '/'))
# The names the NeMo-RL patches import.
from apertus_common import (Call, Claim, Conversation, MappingPolicy, Reply, Result, StreamParseError,
                            SystemPrompt, Think, User, from_openai, load_encoding)
e = load_encoding(os.environ['MODEL'])
print('encoding', e.controls, '| eos', e.generation_constraints.eos_token_id)
import nemo_rl.data.apertus2_format
print('nemo_rl.data.apertus2_format', nemo_rl.data.apertus2_format.__file__)
"
echo "== apertus-common exit $?"

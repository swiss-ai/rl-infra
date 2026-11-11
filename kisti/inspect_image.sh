#!/bin/bash
# Runs inside the image: GPUs, Python packages and the Slingshot (CXI) provider. Submit from the login node:
#   source kisti/env.sh && cd $KISTI_SCRATCH
#   srun "${KISTI_SLURM[@]}" -N1 -t 00:15:00 --container-image=$KISTI_IMAGE \
#     --container-mounts=$KISTI_SCRATCH:$KISTI_SCRATCH bash ~/rl-infra/kisti/inspect_image.sh
echo "host $(hostname)"
nvidia-smi -L
nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1
python3 - <<'EOF'
import importlib
for m in ["torch", "vllm", "uccl", "deep_ep", "transformer_engine", "transformers", "ray",
          "megatron.core", "nemo_rl", "nemo_gym"]:
    try:
        x = importlib.import_module(m)
        print(f"{m:20s} {getattr(x, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:20s} MISSING ({type(e).__name__})")
EOF
fi_info -p cxi 2>&1 | head -3

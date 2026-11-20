# SFT backend: swiss-ai/posttraining

[swiss-ai/posttraining](https://github.com/swiss-ai/posttraining) (TRL, accelerate, DeepSpeed, Hydra) is the code that
trained the Apertus and Apertus-v1.1 instruct models. This folder drives it the way `backends/nemo-rl` drives NeMo-RL:
a clone at a pinned commit with our patches on top, run inside a container image with a Python overlay, one job per
run. The entry point is `sft.sh` at the top of the repository; the recipe, the run sizes and the results are in
[docs/apertus-v11.md](../../docs/apertus-v11.md).

```
bash sft.sh 1.5B 320000
 └ sft.sh ─ sites/load.sh (sites/<SITE>.env, site.env, backends/posttraining/env, backends/nemo-rl/apertus2/env)
   ├ backends/posttraining/setup.sh     login node: posttraining clone at pin + patches/, container image staged
   └ backends/posttraining/submit.sh    one-node sbatch: pyxis flags (HANGANG) or a per-job EDF (Alps)
       └ train.sh (in the container)  build_env.sh (overlay), prepare_data.py (mixture), accelerate launch, model link
```

| File | Purpose |
|---|---|
| `pin` | posttraining's repository and commit |
| `patches/0001-*` | posttraining configs: the `apertus-v11` model (Apertus-v1.1 base models, `apertus` chat template, EOS 68, flash attention) and the `apertus-sft-mixture-hf` dataset (the public mixture after `prepare_data.py`) |
| `patches/0002-*` | `try_sync_wandb` tolerates a missing `scontrol`, which the container lacks (a finished run crashed in it) |
| `env` | Defaults: clone, scratch folders (`SFT_ROOT=$SCRATCH/apertus-v11-sft`), downloads, CPUs per node |
| `setup.sh` | Clones posttraining and applies the patches it lacks (matched by subject); stages the container image |
| `submit.sh` | Runs one script of this folder as a one-node job inside the container |
| `train.sh` | One SFT run: overlay, mixture, training, final model linked as `$SFT_ROOT/models/<run name>` |
| `check.sh`, `check_env.py` | Check job: package versions, one AdEMAMix step on a GPU, the encoding of 20 conversations against the chat template of the released 0.5B-Instruct model (identical on all 20), token lengths of the mixture (mean 622, 0.5% over 4,096) |
| `build_env.sh` | Python overlay `$SFT_ROOT/pydeps`: Transformers 4.57, TRL 0.16, DeepSpeed 0.16.9, bitsandbytes (AdEMAMix), without dependencies, on top of the image's PyTorch |
| `prepare_data.py` | Saves the downloaded mixture (parquet) in the format `load_from_disk` reads, once |

## Container and sites

The container is the Apertus 2 NeMo-RL image (`IMAGE_WORK` of `backends/nemo-rl/apertus2/env`: PyTorch 2.13, CUDA 13,
flash-attn, datasets, accelerate); posttraining needs Transformers 4.x, where `apply_chat_template` returns a tensor,
hence the overlay. Account, partition, mandatory sbatch flags, mounts, `SITE_ENV` and the container runtime come from
`sites/<SITE>.env`, as for NeMo-RL. A job takes a whole node (`--exclusive`: on Alps, `preemptable` otherwise gives
it one CPU and no GPU) and NCCL's socket transport (one node, NVLink; on HANGANG the network plugin fails to start
without a Slingshot service). The partition defaults to the site's `DEFAULT_PARTITION` (HANGANG `intTest`, Alps
`normal`); `PARTITION=debug` (Alps, at most 1.5 hours) or `preemptable` with `RESUME=1` often start sooner.

## Throughput and memory

`DEVICE_BS` (sequences per GPU, default 4) and `GC` (gradient checkpointing; default off for 0.5B, on for 1.5B and
4B) change only speed and memory, as the accumulation steps follow from the global batch of 512. Times per optimizer
step on one node, from runs on 8,192 conversations (16 steps):

| Model | Sequences per GPU | Gradient checkpointing | Site | Time per step of 512 | Conversations per hour |
|---|---|---|---|---|---|
| 0.5B | 2 | on | HANGANG | 29 s | 63,000 |
| 0.5B | 4 (default) | off (default) | Alps | 14.1 s | 130,000 |
| 0.5B | 8 | off | Alps | out of memory (95 GB) | |
| 1.5B | 4 | off | Alps | out of memory (95 GB) | |
| 1.5B | 4 (default) | on (default) | Alps | 22.6 s | 81,000 |
| 4B | 4 (default) | on (default) | Alps | 42.7 s | 43,000 |

The four-hour runs on HANGANG reached the same rates (0.5B: 500,000 conversations in 3.95 h).

## Long runs and preemption

`RESUME=1 SAVE_STEPS=<n>` keeps full checkpoints (with optimizer state, the last two) every n optimizer steps in a
folder named after the configuration, makes the job requeueable, and a job that starts again continues from the last
checkpoint. A resumed HF Trainer takes the interval from the checkpoint's `trainer_state.json`; `train.sh` writes
`SAVE_STEPS` there too.

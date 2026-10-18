# rl-infra

Launch scripts and experiment configs for running RL post-training on the CSCS Alps
(Clariden) GH200 nodes. The training frameworks themselves are not part of this repo:
each backend is an untouched local clone that this repo pins, mounts into a container
and launches, the same way a verl checkout sits next to the code that drives it. The
only backend so far is [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Quick start: GRPO on Apertus 2

```bash
git clone https://github.com/swiss-ai/rl-infra.git && cd rl-infra
bash run.sh envs/gsm8k.yaml
```

This trains Apertus 2 Chonk on GSM8K with GRPO, on 12 nodes of the `infra01` account, in
about an hour and a half, and saves the final weights. A new RL environment needs only a
task file like `envs/gsm8k.yaml`, with its dataset, prompt and reward; the training recipe
stays as it is. The first run also sets everything up, cloning the pinned sources and
copying the 200 GB model to your scratch, so it takes a while to start; later runs start
right away. `run.sh` prints where the training log, the metrics and the weights go, and
`RECIPE=recipes/<name>.yaml` selects one of the other setups in `recipes/`;
[experiments/grpo-chonk/README.md](experiments/grpo-chonk/README.md) shows how to
reproduce the comparison of vLLM against Megatron inference and of AdamW against Muon.
The sections below describe the general launcher underneath.

## How a NeMo-RL run is put together

NeMo-RL ships as an NGC container image (`nvcr.io/nvidia/nemo-rl`) that already contains
every Python environment it needs: the driver venv and one prebuilt venv per Ray worker
type (policy training, vLLM generation, ...). Those venvs are built for one specific
NeMo-RL commit, and NeMo-RL refuses to start if the source it runs does not match it. So
nothing is pip-installed here. Instead, `backends/nemo-rl/pin` records the commit the
image was built from, `setup.sh` clones NeMo-RL at exactly that commit, and the container
environment mounts the clone over `/opt/nemo-rl`, where the image's venvs expect the
source. Patching NeMo-RL is then just committing on top of the pin in the clone; only a
dependency change (a new `uv.lock`) needs a new image, or `NRL_FORCE_REBUILD_VENVS=true`.

`backends/nemo-rl/launch.sh` turns a config and a node count into a Slurm job. It hands
NeMo-RL's own Slurm launcher, `ray.sub` in the clone, the training command, and `ray.sub`
starts a Ray head on the first node and Ray workers on the others, all inside the
container. Stock `ray.sub` does not run on Alps, so the clone sits on a local branch
(`alps`) with one commit on top of the pin, kept here as `backends/nemo-rl/patches/` and
applied by `setup.sh`. Once all GPUs have joined, the head runs `examples/run_grpo.py` from the clone
with the given config, and the job's exit code is the training script's.

Configs in `experiments/` inherit from NeMo-RL's own example configs through
`defaults: /opt/nemo-rl/examples/configs/...` and only state what differs. Overrides on
the command line (`grpo.max_num_steps=50`) are applied last.

## Setup

```bash
cp site.env.example site.env   # then set account, scratch and clone paths
bash setup.sh                  # clones NeMo-RL, writes ~/.edf/nemo-rl.toml, stages the image
```

## Running

```bash
PARTITION=preemptable bash backends/nemo-rl/launch.sh \
    experiments/grpo-math/qwen3-32b-4n4g-fsdp2tp2.yaml 4 qwen3-32b-probe \
    grpo.max_num_steps=3 checkpointing.enabled=false
```

The arguments are the config, the number of nodes, a run name, and any overrides. Each
job writes under `$OUT` (from `site.env`): `slurm/<jobid>-logs/ray-driver.log` is the
training output, `slurm/<jobid>-<run>.out` is the Slurm-side trace of `ray.sub`,
`logs/<run>/` holds metrics and `results/<run>/` checkpoints. Vision-language recipes use
`SCRIPT=examples/run_vlm_grpo.py`.

## Alps specifics baked in

A few things differ from a stock NeMo-RL cluster and are handled by the launcher and the
container environment, each with the reasoning next to it in the code:

- Head and workers start in **one** `srun` step, because Slingshot authorizes the network
  per step and NCCL across separate steps fails.
- Each GPU's HBM shows up as a memory-only NUMA node next to the four CPU memory nodes.
  NeMo-RL's per-GPU memory binding (one ~120 GB node) is too small for loading 30B+
  models, and no binding at all lets host memory spill onto GPU memory, so every step is
  bound to the four CPU nodes (`--mem-bind=mask_mem:0xf`). `numa_watch.sh` logs any host
  memory that still lands on HBM.
- Triton and Inductor caches are node-local, the other caches live on scratch, and the
  image is re-staged from a persistent copy when the scratch purge removes it.

## Status

Dense models train with the PyTorch (DTensor) backend and colocated vLLM: Qwen3-8B-Base
on one node and Qwen3-32B on four (about 10 minutes per GRPO step with 64 prompts × 32
generations). Qwen3-32B needs the older v1 DTensor worker (`dtensor_cfg._v2: false`),
which loads the checkpoint once and broadcasts it instead of loading it on every GPU.
The Megatron backend's weight refit to vLLM is broken in NeMo-RL v0.7.0 for the MoE/VLM
models tried (`experiments/vlm-geo3k/`).

The patch in `backends/nemo-rl/patches/` modifies NeMo-RL's `ray.sub` (Copyright NVIDIA,
Apache-2.0); it adds the container-engine (EDF) mode, the single-step launch, memory
binding and the NUMA watcher.

# rl-infra

Launch scripts and experiment configs for running RL post-training on GH200 clusters: CSCS
Alps (Clariden) and KISTI HANGANG (see [Clusters](#clusters)). The training frameworks themselves are not part of this repo:
each backend is an untouched local clone that this repo pins, mounts into a container
and launches, the same way a verl checkout sits next to the code that drives it. The
only backend so far is [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

## Quick start: GRPO on Apertus 2

```bash
git clone https://github.com/swiss-ai/rl-infra.git && cd rl-infra
bash run.sh envs/gsm8k.yaml
```

This trains Apertus 2 Chonk on GSM8K with GRPO, on 12 nodes of the cluster profile's account
(`infra01` on Alps, `in0138` on HANGANG), in about an hour and a half, and saves the final weights. A new RL environment needs only a
task file like `envs/gsm8k.yaml`, with its dataset, prompt and reward; the training recipe
stays as it is. The first run also sets everything up, cloning the pinned sources and
copying the 200 GB model to your scratch, so it takes a while to start; later runs start
right away. `run.sh` prints where the training log, the metrics and the weights go, and
`RECIPE=recipes/<name>.yaml` selects one of the other setups in `recipes/`;
[experiments/grpo-chonk/README.md](experiments/grpo-chonk/README.md) shows how to
reproduce the comparison of vLLM against Megatron inference and of AdamW against Muon.
The sections below describe the general launcher underneath.

DAPO-Math with AIME 2024 validation, on 8 trainer and 8 vLLM nodes:

```bash
PARTITION=normal JOB_TIME=12:00:00 bash run.sh envs/dapo-math.yaml
```

The task names its recipe in `defaults:`, and `RECIPE` still replaces it. This is a 12-hour job,
about 120 GRPO steps: the recipe saves the weights and stops after 11:30, so another `JOB_TIME`
needs `checkpointing.checkpoint_must_save_by=DD:HH:MM:SS`, some time before the job ends, on the
command. Re-running on an existing checkout also brings its NeMo-RL clone up to date.

General chat on the UltraFeedback prompts, rewarded by an LLM judge on rubrics and validated with
AlpacaEval, on 8 trainer, 4 vLLM and 1 judge node of one job:

```bash
python3.12 envs/prepare_ultrafeedback.py $SCRATCH/rubric-judge/ultrafeedback
python3.12 envs/prepare_alpaca_eval.py $SCRATCH/rubric-judge/alpaca_eval
bash envs/stage_judge_model.sh
bash run.sh envs/ultrafeedback-rubric.yaml grpo.max_num_steps=3 grpo.val_at_start=false
```

The task names its entry `script`, the NeMo Gym one, so `run.sh` also sets up Gym, without touching
the image: a checkout on `PYTHONPATH`, its own Python overlay, and server venvs built on scratch at
job start. `env.judge_pool` serves the judge model through Ray next to the trainer and inference
partitions, and the `rubric_judge` server in `apertus2/patches/gym` fills in the rubrics and calls it.
The rubrics and their weights are in the task. Validation is AlpacaEval 2.0's prompts, GPT-4 Turbo
references and judge prompt with the same judge model, reported as
`validation/alpaca_eval_simple_agent/reward/mean`. Every rollout is saved to `<checkpoint_dir>/rollouts/`.
The few-shot prompt is written into the data rows by the two `prepare_*` scripts, so changing
`envs/ultrafeedback-few-shot.txt` means running them again.

To see how loaded the judge is, each replica writes its state to `<checkpoint_dir>/judge_stats.jsonl`
every few seconds. `python3.12 backends/nemo-rl/apertus2/judge_stats.py <file> [start] [end]` sums it
up per replica, for a window in seconds from the start of the run. Requests waiting and `queue_s`
well above 0 mean the judge's GPUs are full: check `max_num_batched_tokens` in its `vllm_kwargs`, then
add judge nodes with `env.judge_pool.num_nodes` and a recipe of the matching size, as in
`recipes/apertus2-chonk-vllm-adamw-rubric-2judge.yaml`. Replicas with room while batches still take much longer than their generation mean the
limit is in front of them, in Gym's servers, so try `num_workers`. Replicas with very different load
mean Gym is not spreading the calls evenly.

### Several judge models in one run

A run can use several judge models, each on nodes of its own through `env.extra_judge_pools`.
`data.train.gym_agent_choices` draws one judge's agent for each prompt when it is read, so all rollouts
of a prompt go to one judge and no score is averaged across judges. Advantages are normalized per
prompt, so a lenient judge only moves its own prompts. Validation stays on `env.judge_pool`, so
AlpacaEval stays comparable with one-judge runs. Each pool writes its own stats file and each agent logs
its own metrics, so read the reward per agent.

`envs/ultrafeedback-rubric-sft-3judges.yaml` runs three judges on the SFT checkpoint, 16 nodes in all:
Qwen and Gemma 4 on one node each, and Nemotron 3 Super as one replica across two.
`envs/ultrafeedback-rubric-sft-2judges.yaml` leaves Nemotron out, on 14 nodes. After the data steps at
the top of `envs/ultrafeedback-rubric-sft.yaml`:

```bash
bash envs/stage_judge_model.sh
HF_REPO=google/gemma-4-26B-A4B-it CHECK_SCRIPT=download_model.sh bash backends/nemo-rl/apertus2/check_env_job.sh
HF_REPO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 CHECK_SCRIPT=download_model.sh bash backends/nemo-rl/apertus2/check_env_job.sh
bash run.sh envs/ultrafeedback-rubric-sft-3judges.yaml grpo.max_num_steps=3
```

The commands of the full runs are in `experiments/runs.md`. Nemotron's first start builds its kernels
into scratch, which `check_judge.sh` does in about 13 minutes on 64 CPUs, after `setup-gym.sh`:

```bash
JOB_TIME=01:00:00 JUDGE_MODEL=$SCRATCH/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 JUDGE_GPUS=0,1 \
    JUDGE_VLLM_ARGS="--tensor-parallel-size 2 --dtype auto --kv-cache-dtype fp8 --enable-chunked-prefill --mamba-ssm-cache-dtype float32 --enable-prefix-caching --trust-remote-code" \
    PYDEPS_GYM=1 PYDEPS=$SCRATCH/nemo-rl-apertus2/pydeps-gym CHECK_SCRIPT=check_judge.sh \
    bash backends/nemo-rl/apertus2/check_env_job.sh
```

The 4 rubrics and their weights are in `rubric_weights`, once per judge (`rubric_judge`,
`rubric_judge_gemma`, `rubric_judge_nemotron`), so change every block. Each reply takes one judge call
per rubric.

For another judge, add a pool to `env.extra_judge_pools`, copy the three Gym entries at the end of the
task file, and add its agent to `data.train.gym_agent_choices`. Set `cluster.num_nodes` and
`sft_export` in a recipe of its own: `run.sh` reads both from that file alone, not through `defaults`.
Scores come from the judge's first token, so a judge that thinks first gets a noise score with no
error. Turn thinking off with `chat_template_kwargs: {enable_thinking: false}`, and run
`check_judge.sh`, which fails when under 95% of that token's probability is on a verdict. A replica
across nodes needs the three settings in the Nemotron block, or vLLM hangs at start-up.

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
bash setup.sh                  # clones NeMo-RL, writes ~/.edf/nemo-rl.toml (Alps), stages the image
```

The cluster is detected and its profile `sites/<SITE>.env` supplies the account, partition,
filesystems and container runtime; `site.env` (copied from `site.env.example`, optional)
overrides any of it for you, and `sites/load.sh` documents the order.

## Running

```bash
PARTITION=preemptable bash backends/nemo-rl/launch.sh \
    experiments/grpo-math/qwen3-32b-4n4g-fsdp2tp2.yaml 4 qwen3-32b-probe \
    grpo.max_num_steps=3 checkpointing.enabled=false
```

The arguments are the config, the number of nodes, a run name, and any overrides;
`DRY_RUN=1` prints the sbatch command and the container setup instead of submitting. Each
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

## Clusters

| | Alps (`sites/alps.env`) | HANGANG (`sites/hangang.env`) |
|---|---|---|
| Container runtime | CSCS Container Engine: `--environment=<EDF>` | pyxis/enroot: `--container-image` etc. |
| Slurm | account `infra01`, `run.sh` on `preemptable` | account `in0138`, partition `intTest`, plus the mandatory `--gres=gpu:4 --exclusive --comment=field=...;appl=...` |
| Scratch, clones | `/iopsstor/scratch/cscs/$USER`, `/users/$USER/projects` | `/scratch/$USER` (purged after 15 days unaccessed), `$HOME/projects` |
| Images, Chonk model | copied to scratch from capstor | hanint02's copies in `/scratch/hanint02`, read in place (the model is linked into scratch) |
| Network | EDF hooks and `[env]` | the image's libfabric plugin, plus KISTI's `NCCL_PROTO=^LL128 FI_PROVIDER=cxi NCCL_IB_DISABLE=1` (LL128 gives wrong inter-node all-reduce results there) |

Each stack describes its container once, as an EDF template (`edf.toml.template`). On Alps
`setup.sh` installs it in `~/.edf`; on pyxis sites `backends/nemo-rl/container.sh` turns the same
file into `ray.sub`'s `CONTAINER`, `MOUNTS` and `PYXIS_EXTRA_ARGS` and exports its `[env]` for
the job. A new cluster needs a `sites/<name>.env` and, if it cannot be told from its
filesystems, `SITE=<name>`.

## Status

Dense models train with the PyTorch (DTensor) backend and colocated vLLM: Qwen3-8B-Base
on one node and Qwen3-32B on four (about 10 minutes per GRPO step with 64 prompts × 32
generations). Qwen3-32B needs the older v1 DTensor worker (`dtensor_cfg._v2: false`),
which loads the checkpoint once and broadcasts it instead of loading it on every GPU.
The Megatron backend's weight refit to vLLM is broken in NeMo-RL v0.7.0 for the MoE/VLM
models tried (`experiments/vlm-geo3k/`).

The patches in `backends/nemo-rl/patches/` modify NeMo-RL's `ray.sub` (Copyright NVIDIA,
Apache-2.0); it adds the container-engine (EDF) mode, the single-step launch, memory
binding, the NUMA watcher and extra pyxis flags (`PYXIS_EXTRA_ARGS`).

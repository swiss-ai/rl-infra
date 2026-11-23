# rl-infra

Launch scripts and experiment configs for running RL post-training on GH200 clusters: CSCS
Alps (Clariden) and KISTI HANGANG (see [Clusters](#clusters)). The training frameworks themselves are not part of this repo:
each backend is an untouched local clone that this repo pins, mounts into a container
and launches, the same way a verl checkout sits next to the code that drives it. The
only backend so far is [NeMo-RL](https://github.com/NVIDIA-NeMo/RL).

> **Branch based on main before the restructure (November 2026)?** Task files moved from `envs/` to
> `experts/<expert>/` and `shared/`, launch settings became task keys and node tiers, and every NeMo-RL change is a
> patch here. [docs/MIGRATION.md](docs/MIGRATION.md) lists what changed and the steps to move a branch over;
> [docs/repository-map.md](docs/repository-map.md) describes every file.

## Quick start: GRPO on Apertus 2

```bash
git clone https://github.com/swiss-ai/rl-infra.git && cd rl-infra
bash run.sh experts/stem/dev/gsm8k.yaml
```

A full run logs to Weights & Biases and needs `WANDB_PROJECT` and the key file in `site.env` (see
`site.env.example`); without them, start with `DEBUG=1 bash run.sh ...`, a 3-step debug run.

This trains Apertus 2 Chonk on GSM8K with GRPO, on 12 nodes of the cluster profile's account
(`infra01` on Alps, `in0138` on HANGANG), in about an hour and a half, and saves the final weights. A new RL environment needs only a
task file like `experts/stem/dev/gsm8k.yaml`, with its dataset, prompt and reward; the training recipe
stays as it is. The first run also sets everything up, cloning the pinned sources and
copying the 200 GB model to your scratch, so it takes a while to start; later runs start
right away. `run.sh` prints where the training log, the metrics and the weights go, and
`RECIPE=recipes/<name>.yaml` selects one of the other setups in `recipes/`;
[recipes/archive/chain/README.md](recipes/archive/chain/README.md) shows how to
reproduce the comparison of vLLM against Megatron inference and of AdamW against Muon.
The sections below describe the general launcher underneath.

DAPO-Math with AIME 2024 validation, on 8 trainer and 8 vLLM nodes:

```bash
PARTITION=normal JOB_TIME=12:00:00 bash run.sh experts/stem/dev/dapo-math.yaml
```

The task names its recipe in `defaults:`, and `RECIPE` still replaces it. This is a 12-hour job,
about 120 GRPO steps: the recipe saves the weights and stops after 11:30, so another `JOB_TIME`
needs `checkpointing.checkpoint_must_save_by=DD:HH:MM:SS`, some time before the job ends, on the
command. Re-running on an existing checkout also brings its NeMo-RL clone up to date.

General chat on the UltraFeedback prompts, rewarded by an LLM judge on rubrics and validated with
AlpacaEval, on 8 trainer, 4 vLLM and 1 judge node of one job:

```bash
python3.12 experts/chat/scripts/prepare_ultrafeedback.py $SCRATCH/rubric-judge/ultrafeedback
python3.12 experts/chat/scripts/prepare_alpaca_eval.py $SCRATCH/rubric-judge/alpaca_eval
bash shared/scripts/stage_judge_model.sh
bash run.sh experts/chat/archive/ultrafeedback-rubric.yaml grpo.max_num_steps=3 grpo.val_at_start=false
```

The task names its entry `script`, the NeMo Gym one, so `run.sh` also sets up Gym, without touching
the image: a checkout on `PYTHONPATH`, its own Python overlay, and server venvs built on scratch at
job start. `env.judge_pool` serves the judge model through Ray next to the trainer and inference
partitions, and the `rubric_judge` server in `apertus2/patches/gym` fills in the rubrics and calls it.
The rubrics and their weights are in the task. Validation is AlpacaEval 2.0's prompts, GPT-4 Turbo
references and judge prompt with the same judge model, reported as
`validation/alpaca_eval_simple_agent/reward/mean`. Every rollout is saved to `<checkpoint_dir>/rollouts/`.
The few-shot prompt is written into the data rows by the two `prepare_*` scripts, so changing
`experts/chat/scripts/ultrafeedback-few-shot.txt` means running them again.

To see how loaded the judge is, each replica writes its state to the server's `stats_path` (e.g. `<log_dir>/judge_stats-<model>.jsonl`)
every few seconds. `python3.12 backends/nemo-rl/apertus2/judge_stats.py <file> [start] [end]` sums it
up per replica, for a window in seconds from the start of the run. Requests waiting and `queue_s`
well above 0 mean the judge's GPUs are full: check `max_num_batched_tokens` in its `vllm_kwargs`, then
add judge nodes (`num_nodes` of the server in the expert's base.yaml; older task files: `env.judge_pool.num_nodes`), as in
`recipes/archive/apertus2-chonk-vllm-adamw-rubric-2judge.yaml`. Replicas with room while batches still take much longer than their generation mean the
limit is in front of them, in Gym's servers, so try `num_workers`. Replicas with very different load
mean Gym is not spreading the calls evenly.

### Several judge models in one run

A run can use several judge models, each on nodes of its own through `env.extra_judge_pools`.
`data.train.gym_agent_choices` draws one judge's agent for each prompt when it is read, so all rollouts
of a prompt go to one judge and no score is averaged across judges. Advantages are normalized per
prompt, so a lenient judge only moves its own prompts. Validation stays on `env.judge_pool`, so
AlpacaEval stays comparable with one-judge runs. Each pool writes its own stats file and each agent logs
its own metrics, so read the reward per agent.

`experts/chat/archive/ultrafeedback-rubric-sft-3judges.yaml` runs three judges on the SFT checkpoint, 16 nodes in all:
Qwen and Gemma 4 on one node each, and Nemotron 3 Super as one replica across two.
`experts/chat/archive/ultrafeedback-rubric-sft-2judges.yaml` leaves Nemotron out, on 14 nodes. After the data steps at
the top of `experts/chat/archive/ultrafeedback-rubric-sft.yaml`:

```bash
bash shared/scripts/stage_judge_model.sh
HF_REPO=google/gemma-4-26B-A4B-it CHECK_SCRIPT=download_model.sh bash backends/nemo-rl/apertus2/check_env_job.sh
HF_REPO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 CHECK_SCRIPT=download_model.sh bash backends/nemo-rl/apertus2/check_env_job.sh
bash run.sh experts/chat/archive/ultrafeedback-rubric-sft-3judges.yaml grpo.max_num_steps=3
```

The commands of the full runs are in `docs/runs.md`. Nemotron's first start builds its kernels
into scratch, which `check_judge.sh` does in about 13 minutes on 64 CPUs, after `setup-gym.sh`:

```bash
JOB_TIME=01:00:00 JUDGE_MODEL=$SCRATCH/models/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 JUDGE_GPUS=0,1 \
    JUDGE_VLLM_ARGS="--tensor-parallel-size 2 --dtype auto --kv-cache-dtype fp8 --enable-chunked-prefill --mamba-ssm-cache-dtype float32 --enable-prefix-caching --trust-remote-code" \
    PYDEPS_PKGS="openai==2.44.0|devtools==0.13.0|yappi==1.7.6|pydot==4.0.1|asttokens==3.0.2|executing==2.3.0" \
    PYDEPS=$SCRATCH/nemo-rl-apertus2/pydeps-check-gym CHECK_SCRIPT=check_judge.sh \
    bash backends/nemo-rl/apertus2/check_env_job.sh
```

The 4 rubrics and their weights are in `rubric_weights`, once per judge (`rubric_judge`,
`rubric_judge_gemma`, `rubric_judge_nemotron`), so change every block. Each reply takes one judge call
per rubric.

For another judge, add a pool to `env.extra_judge_pools`, copy the three Gym entries at the end of the
task file, and add its agent to `data.train.gym_agent_choices`. Set `cluster.num_nodes` and
`sft_export` in a recipe of its own (or take a node tier, see below).
Scores come from the judge's first token, so a judge that thinks first gets a noise score with no
error. Turn thinking off with `chat_template_kwargs: {enable_thinking: false}`, and run
`check_judge.sh`, which fails when under 95% of that token's probability is on a verdict. A replica
across nodes needs the three settings in the Nemotron block, or vLLM hangs at start-up.

## Task files, recipes, node tiers and context

```
experts/                what to train on (experts/README.md)
  main.yaml             the run every expert shares: tier, time limit, context, checkpoints, rollout dump, validation
  <expert>/             medical, safety, chat, stem, code, knowledge, tool-use, translation, lean
    base.yaml           the expert's official run: datasets, environments, validation, recipe, steps
    dev/<name>.yaml     variants in progress (defaults: ../base.yaml), e.g. another mix or thinking level
    archive/            task files of runs that happened, as they ran (to reproduce them)
    scripts/            data builders and checks of the expert
shared/                 pieces task files take in: frames/ (system frames: thinking level, tools),
                        judges/ (llm-judge.yaml: the grader; servers/: one file per judge model), prompts/ (few-shot), scripts/ (judge download, few-shot removal), debug/
recipes/                how to train: one flat file per model (chonk-sftv02.yaml: checkpoint, trainer, optimizer,
                        vLLM, GRPO); earlier recipes in recipes/archive/ (recipes/README.md)
```

A run is `bash run.sh <expert> [overrides]` (the expert's `base.yaml`) or `bash run.sh <task file>
[overrides]` (a dev or archive file); `run.sh` puts `experts/main.yaml` between the recipe and the
expert's files. Besides NeMo-RL settings, a task file holds rl-infra
keys, inherited through `defaults` like any other key: `recipe` (a file under `recipes/`), `context`
(longest prompt + answer), `size` (a node tier), `job_time`, `pydeps` and `script`. From them
`run.sh` derives the node counts, CP (context / the recipe's `context_parallel.tokens_per_gpu`, rounded up to a
power of 2: 32K gets CP 4 on Chonk; KCP above CP 4, set up on first use), packing (= context), the memory
fraction (`context_parallel.memory_fraction`), and the checkpoint deadline 30 minutes before `job_time`. `RECIPE`, `SIZE` and `JOB_TIME` in the
environment replace the task's values, and overrides on the command line win over everything.

Node tiers come from the recipe's base layout (`nodes:`, Chonk 8 trainer x 8 vLLM, MegaChonk 32 x 32); judge
nodes come from the task, on top:

| `size` | Trainer | vLLM | Chonk |
|---|---|---|---|
| `lite` | base | half the base | 8 + 4 |
| `cheap` | base | base | 8 + 8 (the default in `experts/main.yaml`) |
| `expensive` | 2 x base | 2 x base | 16 + 16 |
| `ultra` | 2 x base | 4 x base | 16 + 32 (run 5) |
| `<t>+<g>` | t | g | explicit counts, e.g. `16+64` |

```bash
bash run.sh stem                                  # full run; refuses to start without W&B (site.env)
DEBUG=1 bash run.sh stem                          # debug run: name ends in -DEBUG, 3 steps, no validation,
                                                  # no checkpoints, 2 hours (the command line can change each)
DRY_RUN=1 bash run.sh stem                        # shows nodes, context, overrides, config copies, preflight
                                                  # and the sbatch command; no setup, no job (also with DEBUG=1)
SIZE=cheap bash run.sh experts/stem/dev/<name>.yaml
```

The job reads copies of every config file of the run, made at launch in
`$OUT/configs/<run>-<time>/` with the setup log next to them: editing a task or recipe while a job
waits, runs or retries does not change it. The code (the NeMo-RL clone, Megatron) is read when the job
starts, so leave the clone of a queued job alone. A thinking level, a mix or a context is a key in a
run file; a list such as `data.train` is replaced whole when files merge, which is why `base.yaml`
refers to `frames.<dataset>` and run files override those keys.

## Developing NeMo-RL changes

The NeMo-RL a run uses is the swiss-ai fork at the commit in `backends/nemo-rl/apertus2/pin`, the PRs
listed there and the patches in `backends/nemo-rl/patches/` and `backends/nemo-rl/apertus2/patches/nemo-rl/`,
applied in order by `setup.sh` into `~/projects/nemo-rl-apertus2`. A fresh clone of this repository
needs nothing else. Work on NeMo-RL in a clone of your own and keep that shared one as setup made it:

```bash
git -C ~/projects/nemo-rl-apertus2 worktree add ~/projects/nemo-rl-<topic>        # or a separate clone
# commit your change in ~/projects/nemo-rl-<topic>, then test it:
NEMO_RL_DIR=~/projects/nemo-rl-<topic> bash run.sh experts/<expert>/dev/<name>.yaml grpo.max_num_steps=3
# when it works, export each commit as the next patch and commit it here, on a branch for main:
git -C ~/projects/nemo-rl-<topic> format-patch --start-number <next> -o backends/nemo-rl/apertus2/patches/nemo-rl/ <base>..HEAD
```

`setup.sh` applies to an existing clone only the patches whose subject it lacks, so the shared clone
catches up on the next launch. Work that stays in a personal clone or worktree is lost to everyone
else: every change a run depends on goes into this repository's patches, and files on main name only
the pinned clone, never a personal one. To check that a patch series is complete, apply it to a fresh
worktree at the pin (the PR merges, then every patch with `git am -3`) and compare the tree with the
clone you tested on.

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

Configs in `backends/nemo-rl/examples/` inherit from NeMo-RL's own example configs through
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
    backends/nemo-rl/examples/qwen3-32b-4n4g-fsdp2tp2.yaml 4 qwen3-32b-probe \
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
models tried (`backends/nemo-rl/examples/`).

The patches in `backends/nemo-rl/patches/` modify NeMo-RL's `ray.sub` (Copyright NVIDIA,
Apache-2.0); it adds the container-engine (EDF) mode, the single-step launch, memory
binding, the NUMA watcher and extra pyxis flags (`PYXIS_EXTRA_ARGS`).

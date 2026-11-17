# Runbook: GRPO for Chonk on the Lean task, checked by a Kimina server (Clariden)

How to train Apertus 2 Chonk with GRPO on `envs/lean-kimina-sft.yaml`, with proofs checked by a Kimina Lean
Server, starting from an account that has never run this stack. `README.md` explains the server, its settings and
its tests; this file is the procedure. Everything here runs on a Clariden login node, from the repository root.

## What runs

One job of 16 nodes (8 Megatron trainers, 8 vLLM), on `normal` by default: NeMo-RL GRPO through `run.sh`, with
the Kimina server (16 Lean REPLs) inside it. Gym's `math_formal_lean` sends every proof to the server.

`train.sh` submits the job with `in_job.sbatch` as its batch script. On the first node (the Ray head) it starts
the server outside the container, then runs NeMo-RL's `ray.sub`. At job start the training reads the server's
address and key through `endpoint.sh`, which waits for a live server. The server:
- ends with the job, and starts again with it when a preempted job is requeued;
- restarts up to 3 times under the same address and key if it dies; after that the job is cancelled;
- shares its node with the training: port 1120, memory on the CPU NUMA nodes, a lower CPU priority.

The job is also cancelled if its Ray head is not up 10 minutes after it started.

The run can be as long as the partition allows: 12 h on `normal`; 1 day on `preemptable`, where a preempted run
is requeued and resumes from its last checkpoint. It saves by `JOB_TIME` - 30 min.

`SERVER=debug` runs the server as a 1-node debug job of its own instead (how the runs of 2026-11-15 worked). It
limits the training to about 1:20 h and takes your one running debug job; see the comments in `train.sh`.

## Status (2026-11-17)

Seen working on Clariden (2026-11-15, see "Training runs" in `README.md`):
- start-up;
- generation in the Apertus 2 format;
- Gym → Kimina with 0 unchecked proofs;
- correct rewards: 6 verified proofs from Chonk SFT v0.2, all genuine;
- non-zero losses, weight refits and a run finishing its 3 steps with exit 0.

Not yet shown in one run:
- a run with rewards that finishes;
- the task's 16,384 tokens through a whole run;
- validation;
- checkpoint saving;
- `train.sh` itself, which puts together the hand-made scripts of those runs, and the server inside the job
  (`in_job.sbatch`). Both are tested only locally, with stand-ins for Slurm, ray.sub and the server.
- checkpoint saving on this task. Other Chonk GRPO runs in rl-infra save (about 1.5 TB per checkpoint with
  AdamW) and resume.

## Once per account

### 1. rl-infra on scratch

    git clone -b lean-kimina-verifier https://github.com/swiss-ai/rl-infra.git $SCRATCH/rl-infra
    cd $SCRATCH/rl-infra
    cp site.env.example site.env

`site.env` is gitignored and holds your settings. Never put keys in it.

### 2. Lean v4.34.0 and Mathlib

The server reads a Lean toolchain and a Lake project with Mathlib built, both at v4.34.0. Set both in
`site.env`:

    : "${LEAN_TOOLCHAIN:=/path/to/toolchains/leanprover--lean4---v4.34.0}"
    : "${LEAN_PROJECT:=/path/to/a/lake/project/with/mathlib}"

- **Reuse first.** Disk is tight, and a working install exists on Clariden. Ask whoever ran this before for the
  two lines of their `site.env`, and check that you can read both paths.
- An install on scratch is deleted after 14 days without access. Each server job reads it, which keeps it.
- To build your own, see "Before the first run" in `README.md` (about 6 GB).

### 3. apertus-common: no GitHub credentials on Clariden

`swiss-ai/apertus-common` is private, and Clariden has no GitHub credentials. Any run on an SFT checkpoint needs
it. Bundle the pinned commit on a machine that can read the repository:

    git clone https://github.com/swiss-ai/apertus-common.git && cd apertus-common
    git branch pinned 494c2be79080aa8fbcd46409b96560f72707cd69     # APERTUS_COMMON_COMMIT in ../apertus2/pin
    git bundle create apertus-common-494c2be.bundle pinned
    scp apertus-common-494c2be.bundle clariden:/iopsstor/scratch/cscs/<user>/

Then, on Clariden, clone it to the path `run.sh` uses. `setup.sh` then finds the commit and skips the network:

    mkdir -p $SCRATCH/nemo-rl-apertus2
    git clone $SCRATCH/apertus-common-494c2be.bundle $SCRATCH/nemo-rl-apertus2/apertus-common-494c2be79
    git -C $SCRATCH/nemo-rl-apertus2/apertus-common-494c2be79 checkout --detach 494c2be79080aa8fbcd46409b96560f72707cd69

The folder name ends in the first 9 characters of the pin. A new pin needs a new bundle.

### 4. The Kimina server and the task's data

    bash backends/nemo-rl/lean-kimina/setup.sh                       # ~2 min: the server at the pin, its venv, the REPL
    PARTITION=debug bash backends/nemo-rl/lean-kimina/setup-data.sh  # one node, < 30 min; prints the job ID

`setup-data.sh` writes `$SCRATCH/lean/data/{train,validation}.v434.jsonl`: 9,128 training statements and 212
validation statements (miniF2F test) that elaborate on Lean v4.34.0.

### 5. The Apertus 2 stack and the model: a dry run

    RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml DRY_RUN=1 \
        bash backends/nemo-rl/lean-kimina/train.sh

The first time, this does the following (about 10 min; the log is `$SCRATCH/nemo-rl/setup.log`):
- clones NeMo-RL, Automodel, Megatron-LM-MoE and Megatron-Bridge into `/users/<user>/projects`;
- writes `~/.edf/nemo-rl-apertus2.toml`;
- copies the 28.4 GB image to `$SCRATCH/container-images/`;
- links the checkpoint's shards into `$SCRATCH/models/`;
- sets up Gym with rl-infra's patches.

It then prints preflight's summary and both sbatch commands. It submits nothing. Preflight must report 0 errors.

### 6. Environment check (one debug node, about 3 min)

    CHECK_SCRIPT=check_env_sft.sh MODEL=$SCRATCH/models/chonk-lc-16k-v0.2-iter_0018315 \
        PARTITION=debug bash backends/nemo-rl/apertus2/check_env_job.sh

It builds the Python overlay on the compute node, imports the stack, and builds Chonk on the meta device against
the checkpoint. It then imports apertus-common and loads the checkpoint's encoding. Read
`$SCRATCH/nemo-rl/slurm/<job>-check_env.out`; a good one ends with `apertus-common exit 0`. Expected and
harmless:
- 39 missing `e_score_correction_bias` tensors (a zero buffer, unused with quantile balancing);
- Triton-kernels and nixl_ep import tracebacks;
- an `FI_CXI_RDZV_PROTO` warning.

## A run

A short run: 3 steps of 32 prompts × 16 rollouts, validation at start, no checkpoints.

    RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml RUN=lean-kimina-v02-smoke-$(date +%Y%m%d) \
        JOB_TIME=01:30:00 bash backends/nemo-rl/lean-kimina/train.sh \
        grpo.max_num_steps=3 grpo.num_prompts_per_step=32 \
        grpo.val_at_start=true grpo.val_at_end=false grpo.val_period=1000 \
        checkpointing.enabled=false

A long run: 12 h, a checkpoint every 10 steps, validation every 20.

    RECIPE=recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code.yaml RUN=lean-kimina-v02-$(date +%Y%m%d) \
        bash backends/nemo-rl/lean-kimina/train.sh \
        grpo.num_prompts_per_step=32 checkpointing.save_period=10 grpo.val_period=20

Use `DRY_RUN=1` first. These settings change the defaults (the comments in `train.sh` list all of them):

| Setting | Default | Note |
|---|---|---|
| `PARTITION` | `normal` | Never preempted, at most 12 h, but 16 nodes waited 9.5 h once (median 7 h for 12-h jobs in 2026-11). `preemptable` starts sooner, up to 1 day, but preempts: about half of 5–16-node runs lasted under 1 h |
| `JOB_TIME` | `12:00:00` | The run saves by `JOB_TIME` - 30 min |
| `MAX_REPLS` | `16` | The server's REPLs, on the first node, which is also a trainer and the Ray head (about 80 GB to spare at its lowest in run 3948553; a REPL takes about 2.2 GB at peak) |
| `RUN` | `lean-kimina-<date>-<time>` | The server's files go to `$SCRATCH/lean-kimina/runs/<RUN>/` |

`train.sh` takes your secrets out of the jobs' environment (sbatch passes it on, and ray.sub prints it).

Debug stays free for your own jobs: nothing here uses it. With `SERVER=debug`, keep debug free before you
submit: while another debug job of yours runs, the server waits, and the training waits for it in `endpoint.sh`
for at most 30 min, then fails.

### After the training starts

| When | Look at | Good sign |
|---|---|---|
| ~1 min | `$SCRATCH/nemo-rl/slurm/<training>-logs/STARTED_RAY_HEAD` | The file exists. If not, the job is cancelled after 10 min |
| ~2 min | `$SCRATCH/nemo-rl/slurm/<training>-<RUN>.out` | `[kimina ...] the Ray head is up`, and no `[kimina ...] the server ended` |
| ~2 min | `$SCRATCH/lean-kimina/runs/<RUN>/logs/<training>-r0-kimina-0.out` | Server up (about 40–70 s, 16 REPLs, controls passed) |
| ~3 min | `$SCRATCH/nemo-rl/slurm/<training>-logs/ray-driver.log` | `kimina-endpoint: http://<server node>:8000 answered` |
| ~11 min | the same | `Starting async GRPO training`, then "Training Results" blocks per step (Loss, Avg Reward) |

Start-up takes about 11 min on warm caches: setup 326–398 s, then the first buffer. Expect it to be slower the
first time.

The outputs:
- metrics (TensorBoard): `$SCRATCH/nemo-rl/logs/<RUN>`;
- rollouts: `$SCRATCH/nemo-rl/results/<RUN>/rollouts/*.jsonl` (`reward`, `proof_status`, `predicted_proof`,
  `response.usage.output_tokens`);
- the server's counters:
  `curl -H "Authorization: Bearer $(cat $SCRATCH/lean-kimina/runs/<RUN>/endpoint.key)" $(cat $SCRATCH/lean-kimina/runs/<RUN>/endpoint)/stats`,
  while it runs.

### Stopping

- `scancel <training>`. The server ends with it.
- With `SERVER=debug`: the server job sees it within about a minute and ends by itself. A training cancelled
  **before it started** leaves that server pending forever (`DependencyNeverSatisfied`): `scancel` it too.

### Weights & Biases (optional)

- Put `WANDB_PROJECT` (and `WANDB_ENTITY`, the team) in `site.env`, and the API key in `~/.wandb_api_key` with
  `chmod 600`. The key is read from that file and never enters the job's command or environment.
- Without `WANDB_PROJECT`, W&B stays off.
- CSCS asks for `WANDB_DIR=/capstor/store/cscs/swissai/infra01/wandb_logs/<github-user>/`, which rl-infra does
  not set yet.

## Known problems and lessons

- **Head node container never started** (job 3949635): 16 nodes idled for 1:30 h, because ray.sub waits for the
  Ray head without a limit. Such a training is now cancelled after 10 min (`in_job.sbatch`, or the debug
  server's watch).
- **Stale file handle in the flashinfer cache** on a cold `$SCRATCH/.cache/flashinfer`: 32 vLLM workers compile
  the same kernel at once (job 3947435). A rerun on the now warm cache did not show it.
- **Preemption:** `--time-min` does not protect a preemptable job. Clariden has `PreemptExemptTime=NONE` and
  `GraceTime=240`, so only `normal` is safe from it. A preempted in-job run is requeued and starts its server
  again; it resumes from its last checkpoint and loses the steps since. With `SERVER=debug`, `train.sh` adds
  `--no-requeue`, so a preempted run ends instead of restarting against a server that is gone.
- **Slurm's start estimates** (`squeue --start`, `sbatch --test-only`) were off by hours, both ways.
- **sbatch refuses a partition list** (`normal,preemptable`).
- **Concurrent runs share** `$SCRATCH/vllm_cache_<n>` and `$SCRATCH/.cache`. It worked, but a cold cache broke a
  start once. Every run gets its own endpoint file, so runs never read each other's server.
- **Token cap:** with `policy.generation.max_new_tokens=4096`, 84 % of v0.2's rollouts stopped before a reply.
  Keep the task's 16,384.
- **The v0.1 checkpoint** (`recipes/apertus2-chonk-sftv01-19374-vllm-adamw-code.yaml`) had no Lean in its SFT
  data and scored 0 on every rollout. Use v0.2 or later.
- **Ports inside the job:** `ray.sub` keeps 1150–8999 (Ray, Gym 5000–5999, vLLM 7000–8999) and reads `PORT`
  as Ray's own port. The in-job server uses 1120 (`KIMINA_PORT`), and `train.sh` passes its settings under
  `KIMINA_` names only.

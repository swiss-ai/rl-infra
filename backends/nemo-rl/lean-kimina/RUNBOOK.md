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

## Status (2026-11-19)

Working on Clariden, shown by the runs below:
- `train.sh` with the server inside the job, on 9 and 12 nodes; no server restart was seen in these runs
  (16 REPLs, 32 GB address space each);
- 16,000-token rollouts, validation (in-run, every 10–20 steps), checkpoints (1.4 TB each, about 1 min to save),
  resuming from a checkpoint (requeue or a new job with the same `RUN`), keeping all checkpoints;
- export of a checkpoint to HF, and a validate-only evaluation of an HF checkpoint with this pipeline (see
  "Expert handover");
- correct rewards and rollout logs: every step's logged `Avg Reward` equals the mean reward of its rollouts.

Results so far (details in the sections below):
- 9 nodes, learning rate 2e-6 (job 3965303): 44 steps, no measurable change on miniF2F; the policy barely moved.
- 12 nodes, six ablations (2026-11-18; one failed): every arm that ran ended above the baseline on
  miniF2F-test, most within noise; the best,
  learning rate 1e-5 at step 40, gained +4.4 ± 1.7 points on test and +3.7 ± 1.7 on valid (in-run validation),
  and +5.5 ± 2.0 on test when evaluated again after HF export. It is the Lean expert of experts round 1.
- Learning is slow: 40–50 steps (about 6 h on 12 nodes) give a few points. Why, and how to get more per step, is
  the open question before a run of 200+ steps.

Open:
- the 736 mixed-reward rows are not in this repository yet (see "Tools of 2026-11-18");
- `eval/run.sh` of the expert entry has run on the expert, not yet on `start` (its `start` row comes from the
  same pipeline's step-0 validations).

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

## The Goedel run (2026-11-17)

Clariden job 3965303. `run-goedel-9n.sh` repeats it as one command (`DRY_RUN=1` first). It needs everything above,
including `validation.v434.jsonl`.

**Outcome.** It ran 2026-11-18 00:58–07:28 with its time limit shortened to 6:30 (Clariden maintenance at 08:00):
44 training steps, checkpoints at steps 10, 20, 30, 40 (the time limit stopped it before step 50).
- miniF2F (212 statements, 1 sample each): 44.34 % at the start, 42.92 % at step 20, 40.09 % at step 40. With one
  sample per statement a difference has a standard error of about 3.8 points: no measurable change.
- Training reward flat at about 0.20 (mean 0.2005, step to step 0.086–0.283).
- The policy barely moved: entropy (about 0.27), mean reply length (about 8,990 tokens) and termination rate show
  no trend. The PPO ratio was exactly 1 (one optimizer update per batch), the gradient norm 0.009–0.017 (clip 1.0),
  the importance-weight cap (2.0) hit on about 3e-6 of tokens: no clipping acted.
- Not starved of signal: 51 % of the 1,408 trained prompt groups had mixed rewards (48 % all 0, 0.6 % all 1),
  about 16 informative groups per step. By source: OMR_problem 72 % all 0, lean_workbook_plus 36 %.
- 12.3 % of rollouts were cut at 16,000 tokens (Gym `incomplete`, reason `max_output_tokens`; these are the
  `empty_generation` rows, reward 0); 2.0 % timed out in Kimina (120 s).
- Kimina never restarted; the head node kept at least about 82 GB free.
- The rollouts are in `results/<RUN>/rollouts/train_weights<v>.jsonl`, one file per generating weight version:
  steps 1 and 2 trained on version 0, step s >= 3 on version s-2 (asynchronous GRPO, trajectory age 1).

| Setting | Value | Why |
|---|---|---|
| Model | Chonk SFT v0.2, iteration 18315 | the first Chonk with Lean in its SFT data |
| Nodes | 9: 8 Megatron trainers, 1 vLLM; `normal`, 12 h | the trainers hold AdamW (state offloaded to CPU memory); in the 14 days to 2026-11-17, 12-h `normal` jobs of 5–8 nodes waited a median of 3.3 h, of 13–16 nodes 22.7 h |
| Kimina | inside the job, 16 REPLs | the first node is also a trainer and the Ray head: about 80 GB to spare at its lowest (run 3948553) |
| Training rows | 2,357 Goedel-Prover RL V2 statements (`data/goedel-train-s1.v434.jsonl.gz`) | v0.2 proves 14–16 % of Goedel statements on the first try, 3.1 % of the Nemotron-Math-Proofs v1 rows, which were chosen as hard (`initial_success_count` 1–3) |
| Sampling | temperature 1.0, top_p 1.0 | at 0.6 / 0.95, 38 % of rollouts reached the length limit, 76 % of those in literal repetition loops; at 1.0, 16 % and 5 %; first-try rate about the same (14.0 against 14.6 %) |
| Length | 16,000 new tokens; 16,384 with the prompt, the checkpoint's context | of rollouts still running at 14,000 tokens, 0 of 195 (temperature 1.0) and 3 of 471 (0.6) ended in a verified proof |
| Proof check | 120 s (`compilation_timeout`; Gym's default is 30 s) | 1,983 replies of chonk-tests' Goedel evaluation, checked again with 1,200 s: correct proofs took a median 1.8 s, p95 28.7 s, p99 108 s, at most 587 s. 30 / 60 / 120 / 300 s keep 95.1 / 97.0 / 99.0 / 99.8 % of them; 120 s keeps about a third of the 16 REPLs busy |
| Step | 32 statements × 16 rollouts, one optimizer step; AdamW, constant learning rate 2e-6 | |
| Validation | miniF2F test (212 statements), at the start, every 20 steps and at the end | |
| Checkpoints | every 10 steps, all kept (about 1.5 TB each), the last by 11:30 h | insurance against a crash, and the first save is an early test of saving on Alps |

How the training rows were made, from Goedel-LM/RL_dataset_V2 at revision `5c99dce`:
1. `goedel_rows.py --cache <dir> --n 3000 --seed 1 --exclude-sample 500:0 --exclude-statements
   $SCRATCH/lean/data/validation.v434.jsonl <rows>` (needs pyarrow; downloads 336 MB once): 3,000 of the 49,913
   distinct statements, then 2,570 rows. Dropped: 392 statements with a `let ... :=` (Gym's proof assembly cuts
   them, so they score 0 whatever the model writes), 33 of the 500 that chonk-tests evaluates on, 5 duplicates;
   none is in miniF2F.
2. The rows whose statement elaborates on Lean v4.34.0 + Mathlib. The run's file was checked with the `lean`
   command line (chonk-tests' `proofnet.py check-statements`), which kept 2,357. Here the check is
   `validate_statements.py` against a Kimina server (`start.sh`) on the same Lean and Mathlib; it has not been
   run on these rows, so it may keep a slightly different set. Given the run's kept rows, `goedel_rows.py`
   reproduces the committed file byte for byte (sha256 `47e7b602...`, checked by `run-goedel-9n.sh`).

Versions: this branch of rl-infra; NeMo-RL, Megatron, Bridge and Gym at `../apertus2/pin`; Lean v4.34.0 and
Mathlib from `site.env`. The run's Kimina install is at 97d7683; `pin` names b3c6c3e, which only adds exiting when
the server's start-up fails.

## Ablations (2026-11-18)

Why: the Goedel run's policy barely moved (one update of about 2e-6 per step), so every arm changes how far or how
usefully the policy moves per step. All start from Chonk SFT v0.2 iteration 18315 and keep the Goedel run's
settings (Goedel rows, 32 prompts × 16 rollouts, temperature 1.0, 16,000 tokens, 120 s, constant learning rate,
no KL), except:
- **12 nodes: 8 Megatron trainers, 4 vLLM** (recipe `apertus2-chonk-sftv02-18315-vllm-adamw-code-12n.yaml`). One
  vLLM node made generation the bottleneck; with four, a step takes about 7 min (alternating about 10–13 min and
  2 min) and a 972-row validation about 7 min.
- **Validation: corrected miniF2F** (the user's `corrected_miniF2F/minif2f.jsonl`, valid 244 + test 244; 486 of
  488 elaborate on v4.34: test `amc12a_2020_p7` uses `∑ … in`, `amc12a_2020_p15` uses `Complex.abs`), every
  statement twice (972 rows), every 10 steps. Checkpoints are picked on **valid** and reported on **test**.
- Checkpoints every 5 steps, all kept; `JOB_TIME` 6:20; `preemptable` with requeue.

**Baseline** (the same starting model validated twice, jobs 3972333 and 3972629, step 0): valid 45.70 / 48.16 %,
test 39.26 / 39.46 %; pooled (4 samples per statement) **valid 46.93 %, test 39.36 %**. The two identical
measurements differed by +2.5 ± 2.0 (valid) and +0.2 ± 2.0 (test) points: a checkpoint needs roughly 3.5 points
over the pooled baseline to stand out (2 standard errors).

| Arm | Job | Change | Steps | Best validation (change vs baseline, ± paired SE) |
|---|---|---|---|---|
| **H1** | 3972337 | learning rate 1e-5 | 47 (time limit) | step 40: valid +3.69 ± 1.66, test +4.44 ± 1.74; step 30: test +5.06 ± 1.89; step 20: valid +4.51 ± 2.00, test −0.31 |
| H2 | 3972342 | global batch 128 (4 optimizer updates per step) | 1 | failed in step 2: host memory (Slurm `OUT_OF_MEMORY`), see Known problems |
| H3 | 3972343 | only the 736 prompts with mixed rewards in run 3965303 | 23 (one epoch) | step 20: valid +2.66 ± 1.51, test +2.58 ± 1.66; step 23: +1.43 / +0.93 |
| H8 | 3972333, resumed 3973965 | learning rate 5e-6 + global batch 128 + the 736 mixed prompts | 23, then 24–41 | step 23: valid +2.87 ± 1.61, test +3.00 ± 1.88; second epoch: step 30 +2.46 / +0.52, step 40 +3.07 / +1.55 |
| H8 twin | 3972629, resumed 3973966 | learning rate 5e-6 + the 736 mixed prompts (global batch 512) | 23, then 24– | step 23: +2.05 / +2.79; step 30: valid +2.46 ± 1.56, test +4.24 ± 1.61 |
| No-std | 3972454, rerun 3972630 | advantages not divided by the group's standard deviation (`grpo.normalize_rewards=false`) | 51 | step 40: +1.23 / +2.79; step 50: +1.02 / +2.38 |
| (H4) | not run | `loss_fn.positive_example_nll_weight` (NLL on correct samples) | | no effect at this NeMo-RL pin: GRPO's training batch carries no `rewards`, so the term is never computed |

What they say, with the caveat that each arm ran once:
- More movement per step helps: learning rate 1e-5 (H1) is the only arm up on both splits beyond 2 standard
  errors; 5e-6 (H8, H8 twin) is up on both by about 2–4 points.
- Training only on prompts with mixed rewards (H3, H8, H8 twin) gives earlier gains, but the 736 prompts run out
  after 23 steps, and a second epoch on the same prompts (H8) did not add to miniF2F.
- Removing the standard-deviation scaling (no-std) changed little.
- The training reward rose clearly only on the mixed prompts (H8: about 0.33 at the start, 0.42 at step 10, 0.55
  at step 20, still in the first epoch).

The launcher of these runs is `ablations.sh <h1|h2|h3|h8|h8s|nostd> [partition]`; see "Tools of 2026-11-18".

## Expert handover (Apertus 2 experts, round 1)

H1 step 40 is staged as `lean__lean-kimina-v02-abl-h1-20261118` in
`/capstor/store/cscs/swissai/infra01/apertus-2-expert-staging/experts-r1-2026-11-18/`, following
`apertus-program/docs/engineering/models/apertus-2/apertus2-experts.md` (posted to issue #1348):
- `ckpt`: HF export of `results/<RUN>/step_40/policy/weights/iter_0000000` with NeMo-RL
  `examples/converters/convert_megatron_to_hf.py` (8 min on one debug node). The single-process export fails
  with DeepEP, so it reads a shadow copy of the checkpoint folder: symlinks to the shards and a `run_config.yaml`
  with `moe_enable_deepep: false` and `moe_flex_dispatcher_backend: null` (weights unchanged). The converter
  writes its own config, tokenizer and modeling files; they are replaced by those of `start` (originals kept in
  `ckpt_export_originals/`). Checked: 30,978 tensors with the names, shapes and dtypes of `start`, every shard
  complete against its header; the sampled tensors differ from `start`.
- `prompts/`: the 2,357 training rows and the 446-statement Goedel held-out sample in the handover schema
  (Apertus 2 native conversation, frame `{"thinking":"medium","tools":[]}`).
- `rollouts/`: every trained rollout, `step001`–`step047.jsonl`, with `prompt_id`, `step`, `sample`, `messages`
  (prompt, raw reply with the think block, Kimina's verdict) and `reward`. Version 0 holds the batches of steps
  1 and 2 interleaved; they are split by the collector's log ("Buffered prompt group (_ng_task_index=..,
  target_weight=..)"). Every step's mean reward equals the run's logged `Avg Reward`.
- `eval/run.sh`: `MODEL=<HF dir> bash eval/run.sh` validates MODEL once with this pipeline (rl-infra at 302695a,
  12 nodes) and stops: `grpo.val_at_start=true grpo.stop_at_validation_metric=accuracy
  grpo.stop_at_validation_threshold=0.0 checkpointing.enabled=false`. (`grpo.max_num_steps=0` returns before
  the validation at start.) Lean comes from the shared install
  `/capstor/store/cscs/swissai/infra01/lean_apertus_v2/lean-v4.34.0` (Lean v4.34.0, Mathlib `5ed29652`, the same
  as the training's project). On the expert (job 3974712, 18 min) it gave test 44.83 %, valid 49.80 %, within
  noise of the run's own step-40 validation (43.80 / 50.61 %).

## Tools of 2026-11-18 (this directory)

| File | What |
|---|---|
| `../../../recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-12n.yaml` | the 12-node layout: 8 Megatron trainers + 4 vLLM |
| `ablations.sh <variant> [partition]` | the ablation launcher (h1, h2, h3, h8, h8s, nostd); `RUN`, `VAL_AT_START`, `EXTRA_OVERRIDES` for resumes |
| `data/minif2f-corrected-x2.v434.jsonl.gz` | the validation file: corrected miniF2F, valid 244 + test 242, each twice (972 rows, sha256 `f8e04393...`); `ablations.sh` unpacks it |
| `status.sh <job>:<tag> ...` | status of running jobs: state, server start, steps, validations, checkpoints, fatal errors |
| `val_report.py --val ... --baseline <RUN> <TAG>=<RUN> ...` | pass@1 per split for every validation of each run, paired against a pooled step-0 baseline |
| `export_hf.sbatch` | HF export of a checkpoint (DeepEP off in a shadow `run_config.yaml`); then `compare_safetensors.py` |
| `compare_safetensors.py <ckpt> <start>` | tensor names, shapes and dtypes of two HF checkpoints, from their safetensors headers |
| `handover_lean.py` | prompts and rollouts of a run in the experts-handover schema |
| `expert-eval/run.sh`, `expert-eval/score.py` | copies of the expert entry's `eval/`, for reference: validate-only evaluation of an HF checkpoint, pinned to rl-infra 302695a. Run it from the entry, which also holds the validation file and the apertus-common bundle it needs |

These were run on Clariden from copies with fixed paths; the versions here take the checkout, run and files as
arguments and are only syntax-checked so far.

### Not in this repository yet

- `goedel-train-s1-mixed.v434.jsonl`, the 736 rows with mixed rewards in run 3965303 (variants h3, h8, h8s): on
  Clariden at `$SCRATCH/lean/data/` of the account that ran them. It was made from the run's rollouts: the training
  rows whose 16 rollouts had 1-15 rewards, from all 1,444 generated groups.

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
- **Global batch 128 (4 optimizer updates per step) runs out of host memory on 8 trainer nodes** (H2, 3972342:
  Slurm `OUT_OF_MEMORY` on the head node in step 2's training). The optimizer state is offloaded to CPU memory;
  with 4 updates per step the trainer nodes had about 75–180 GB free against about 310 GB at global batch 512, and
  single NUMA nodes went down to 6–8 GB. The job gets Slurm's default 450 GB per node (no `--mem`). H8 survived the
  same setting with a thin margin. Use more trainer nodes before trying more updates per step.
- **GPU out of memory in the first training step, once** (no-std 3972454, one rank at 0.02 GB free). An identical
  rerun passed. `driver_retry.sh` does not retry a run that has not completed a step.
- **`grpo.max_num_epochs: 1` (the recipe) ends a run after one pass over its rows**: the 736 mixed prompts gave 23
  steps. Set the epochs a run needs before its first step: changing them on resume changes the total iterations
  that Megatron's learning-rate scheduler stored in the checkpoint, and every trainer fails with
  `AssertionError: OptimizerParamScheduler: class input value ... and checkpoint value ... do not match`
  (3973489, 3973492, 3973508). With a constant learning rate it is safe to resume anyway with
  `++policy.megatron_cfg.scheduler.override_opt_param_scheduler=true` (3973965, 3973966).
- **`normal` has a 350-node QoS cap** (`normalp`) shared by all its users; a job waits with `QOSGrpNodeLimit`
  while it is full, and `preemptable` (no cap, same 1,308 nodes) filled up in the evening of 2026-11-18 (estimated
  starts after midnight). A 12-node job started at once in the afternoon.
- **`loss_fn.positive_example_nll_weight` does nothing in GRPO** at this pin: the term needs `rewards` in the
  training batch, and `grpo.py` never puts them there.
- **Validation logs** (`logs/<RUN>/exp_*/val_data_step<N>.jsonl`) have empty `content` (prompts go to vLLM as
  token IDs) but an `idx` per row, 0 to N-1 once each, taken as the row of the validation file (consistent with
  every check so far; not confirmed in NeMo-RL's code). Split scores and paired comparisons rely on it.
- **Validation with one sample per statement is too noisy** to compare checkpoints: two identical baselines
  differed by 2.5 points with two samples. Use at least two samples per statement and a baseline measured twice.

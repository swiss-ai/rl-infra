# MegaChonk in rl-infra (KISTI Hangang)

Record of bringing the Apertus 2 MegaChonk pretraining checkpoint into rl-infra and NeMo-RL, from
2026-11-15 to 2026-11-18: layout, minimum node counts, the runs so far and the problems fixed on
the way. All runs use the pretrained checkpoint at iteration 42,000.

## Model and checkpoint

| | MegaChonk | Chonk (for comparison) |
|---|---|---|
| Parameters | 593.7 B (560.5 B in routed experts, 33.2 B outside them) | 5.2 B outside the experts |
| Layers | 61 (45 linear-attention KDA, 16 full attention), hidden 8,192 | |
| Experts | 256 routed, top 8, latent MoE 4,096, 2 shared | |
| Pretraining context | 8,192 tokens | |

- Checkpoint: iteration 42,000, the newest official HF export (23 Oct). It is 14% of the planned
  pretraining (lr at 86% of its peak). Private HF repo `swiss-ai/megachonk-pretraining-iter_0042000`;
  on KISTI in `/scratch/hanint06/models/megachonk-pretraining-iter_0042000`.
- Staged for runs as `/scratch/hanint06/models/apertus2-megachonk-iter42000`: links to the 70 shards,
  plus the Apertus 2 HF modeling code (vLLM builds the model from it), plus `AutoModel` entries in
  `config.json`. `~/transfer/stage_megachonk.sh <export> <staged folder>` reproduces this.
- **Later checkpoints do not work with this stack yet.** Between iterations 88,000 and 90,000 the
  pretraining run switched on `--embedding-norm`: an RMSNorm with a learnable gain on the embedding
  output (8,192 parameters). The converter, the HF modeling code and our Megatron do not have it. An
  export of 90,000 silently dropped the gain, and the model then emits one repeated token (greedy
  GSM8K 0 of 1,319). Iteration 88,000 is the newest checkpoint the current tools can run.

## Layout and minimum node counts

Nodes have 4 GH200 GPUs (95 GB each) and about 411 GB of host memory.

**Trainer: 32 nodes minimum** (128 GPUs), with full-precision AdamW and trainable experts.

| Setting | Value |
|---|---|
| Parallelism | TP4 × PP8 × EP16; CP4 at 16K context (4,096 tokens per GPU) |
| Pipeline | 8 stages over 61 layers: 8 in the first, 5 in the last |
| Optimizer | AdamW (12 bytes per parameter), 70% of its state in host memory (`optimizer_offload_fraction: 0.7`) |
| GPU memory at 2K | lowest free GPU memory 6.0 GB, lowest free host memory 41 GB per node (GSM8K full run) |
| GPU memory at 16K, CP4 | 29 to 44 GB allocated per GPU, at least 31.7 GB free (debug 1297757) |

- 16 trainer nodes do not fit (job 1291990, TP4 × PP8 × EP8): 61.7 GB of weights and gradient
  buffers per GPU, 79.7 GB peak while loading, before about 20 GB of communication buffers.
- Without tensor parallelism (TP1) the AdamW state on the host exceeds host memory at the first
  optimizer step. TP4 alone was not enough either; TP4 plus 70% offload works.

**Inference: one vLLM replica needs at least 4 nodes.** Each node runs one engine with TP4; the
experts are spread over all GPUs of the replica. FP8 expert weights, BF16 attention.

| Replica | Experts | Weights per GPU | KV cache per GPU | Tokens per engine | Full 16K answers per engine |
|---|---|---|---|---|---|
| 8 nodes (used) | EP32 | 37.8 GiB | 35.2 GiB | 434,000 | about 26 |
| 4 nodes | EP16 | 54.1 GiB | 21.7 GiB | 267,000 | about 16 |

- 2 nodes per replica leave no room for the KV cache.
- On GSM8K, one 4-node replica instead of one 8-node replica made the slower steps about 13% longer
  (job 1292023).

Layouts used:

| Run type | Nodes | Trainer | Inference |
|---|---|---|---|
| GSM8K, 2K | 40 | 32 | 8 (one 8-node replica, 8 engines) |
| STEM mix, 16K | 48 | 32 | 16 (two 8-node replicas, 16 engines) |
| 16K debug runs | 36 | 32 | 4 (one 4-node replica) |

## GSM8K full run

Job 1285740, run `megachonk-gsm8k-58steps-1`, 2026-11-16, 09:21 to 13:15 KST.

| Setting | Value |
|---|---|
| Task | GSM8K train, five-shot prompt (`envs/gsm8k.yaml`) |
| Context | 2,048 tokens in total, up to 2,048 generated |
| Batch | 128 prompts × 16 answers per step, 4 optimizer steps (global batch 512) |
| Optimizer | AdamW, lr 1e-6 |
| Asynchrony | staleness 1 |
| Steps | 58 (one epoch) |
| Validation | GSM8K test, 1,319 questions, at steps 0, 20, 40 and 58 |
| Layout | 40 nodes: 32 trainer + 8 inference |
| Recipe | `recipes/apertus2-megachonk-vllm-adamw.yaml` with `envs/gsm8k.yaml` |

**Result: completed 58 of 58 steps in 3 h 53 min 37 s.**

| Step | MegaChonk validation accuracy | Chonk, same recipe (job 1173380) |
|---|---|---|
| 0 | 11.1% | 47.8% |
| 20 | 54.9% | 85.3% |
| 40 | 64.6% | 90.5% |
| 58 | 68.5% | 91.7% |

- Training reward: 11.8% at step 1, 52% at step 20, 61% at step 40, 68% at step 58.
- Mean answer length: 120 tokens at the start, 160 at the end; no answer hit the limit.
- Greedy validation at step 0 (separate job): MegaChonk 39.1%, Chonk 75.3%.

**Step time** (steps 2 to 58; step 1 took 1,442 s, almost all one-time kernel compilation):

| | Mean | Median | 10th to 90th percentile |
|---|---|---|---|
| Total step | 200 s | 218 s | 138 to 271 s |
| Training | 74.3 s | 74.1 s | 72 to 76 s |
| Log-prob pass | 23.4 s | 23.4 s | 23 to 24 s |
| Weight sync to vLLM | 37.3 s | 37.2 s | 37 to 38 s |
| Waiting for generation | 51.2 s | 2.7 s | 1.5 to 120 s |

- Steps alternate between about 163 s and 239 s: with staleness 1, every other step waits about
  two minutes for generation.
- Trainer throughput: 14,400 tokens per second (all 128 GPUs).
- vLLM against the trainer: generation KL error 0.001 to 0.002 throughout.
- **W&B: none.** The run predates W&B logging for these runs. Metrics are in TensorBoard,
  `/scratch/hanint06/nemo-rl/logs/megachonk-gsm8k-58steps-1/exp_001/tensorboard`; validation answers
  in `val_data_step{0,20,40,58}.jsonl` next to it. It can be uploaded afterwards with
  `~/projects/rl-context-length/submit_wandb_sync.sh megachonk-gsm8k-58steps-1` (scalars only).

**Why the start is low.** The checkpoint, not the stack:
- Our trainer's loss on 600 held-out pretraining documents matches the pretraining log
  (1.75 to 1.78 against 1.788 nats per token).
- vLLM and the trainer agree (mean log-prob difference 0.023).
- The gap to Chonk is constant along the sequence, so there is no precision error that builds up
  with position.
- Iteration 42,000 is a weaker language model than Chonk: five-shot prompt loss 1.20 against
  0.83 nats per token.

## 16K STEM mix

Data per step: 128 DAPO-Math, 256 NuminaMath and 128 science multiple-choice prompts (the run-3
mix), raw row text as the prompt. Validation: AIME 2024 × 32 and GPQA-Diamond × 8.

- **Debug 1299026** (36 nodes, 3 steps of 64 × 16, answers capped at 2,048 tokens): completed.

  | Step | Total | Training | Weight sync | Rows trained | Correct of 1,024 |
  |---|---|---|---|---|---|
  | 1 | 1,416 s | 1,370 s (compilation) | 44 s | 64 | 4 |
  | 2 | 103 s | 65 s | 37 s | 112 | 10 |
  | 3 | 52 s | 13 s | 38 s | 160 | 13 |

  Generation KL error 0.0008 to 0.0010. A 4-node engine produced about 1,200 to 2,000 tokens per
  second at full load.
- **Full run 1300476** `megachonk-stem-mix-run3-16k-512p-48n-c`, started 2026-11-18 17:34 KST:
  512 prompts × 16 answers per step, one optimizer step, lr 2e-6, staleness 2, 16K context,
  zero-advantage groups skipped, truncated answers rewarded 0, 100 steps planned, 2-day limit,
  48 nodes. W&B: https://wandb.ai/apertus/chonk-rl/runs/0cwmi5vg
  - Start validation (1,202 s): AIME 2024 0.1% (1 of 960), GPQA-Diamond 4.5%, mean answer
    538 tokens.
  - The first batch took about 52 minutes to generate.

## Problems found and fixed

| Symptom | Cause | Fix |
|---|---|---|
| vLLM fails at start-up (`AutoModel.from_config`) | export lacks the HF modeling code | stage the model with the modeling code |
| vLLM out of memory (80 GiB MoE workspace) | all-gather workspace grows with ranks × batched tokens | `max_num_batched_tokens: 4096` |
| "Disk quota exceeded" | group-wide file limit on scratch | vLLM and FlashInfer caches in home (`sites/hangang.env`) |
| 10-minute timeout in the first training step | kernels compile stage by stage, about 95 s each | `distributed_timeout_minutes` (new key) |
| host memory exhausted at the first optimizer step | AdamW state on the host | trainer TP4 and `optimizer_offload_fraction: 0.7` |
| preflight rejects TP4 | wrong divisibility rule | fixed in `backends/nemo-rl/preflight.sh` |
| `make_sequence_length_divisible_by (8) is not a multiple of 32` | CP4 × TP4 needs 32 | set to 32 for 16K |
| 10-minute wait for a communicator id at the first optimizer step | gradient-norm communicator first created there, stages arrive minutes apart | warm it up at start (`nemo_rl/utils/nccl_warmup.py`); `NRL_NCCL_TIMEOUT_MINUTES` and the Megatron timeout at 60 |
| driver hangs after setup on 48 nodes | W&B copies the console; 377,000 lines of worker output; the client's backlog grows without bound | `WANDB_CONSOLE=off` in the driver |
| driver log silent for an hour, hang watcher would stop a healthy run | without the W&B console wrapper, Python buffers stdout in 4 MiB blocks on Lustre | `PYTHONUNBUFFERED=1` in the driver (from the next run on) |
| one node never starts | cluster fault (node2009, node2068) | `--exclude` |
| converted iteration 90,000 emits one token | embedding norm missing in the tools | stay on 42,000 |

Every MegaChonk launch also sets Ray on Slingshot (`RAY_NODE_NAME_SUFFIX=-hsn0`, check `ip=10.169.`
in the driver log) and W&B (`WANDB_PROJECT=chonk-rl WANDB_ENTITY=apertus
WANDB_KEY_FILE=/home01/hanint06/.wandb-api-key`).

## Files

Recipes and experiment files:

| File | What |
|---|---|
| `experiments/grpo-chonk/megachonk-gsm8k-40n-megatron-128x16-2k-async-fp8.yaml` | base settings: model, trainer and vLLM layout, memory, timeouts |
| `recipes/apertus2-megachonk-vllm-adamw.yaml` | GSM8K, 40 nodes |
| `recipes/apertus2-megachonk-vllm-adamw-36n.yaml` | GSM8K, 36 nodes (one 4-node vLLM replica) |
| `experiments/grpo-chonk/megachonk-stem-16k-megatron-512x16-async-fp8.yaml` | 16K, 512 × 16, one optimizer step, CP4 |
| `recipes/apertus2-megachonk-vllm-adamw-stem16k.yaml` | 16K, 48 nodes (two 8-node vLLM replicas) |
| `recipes/apertus2-megachonk-vllm-adamw-stem16k-36n.yaml`, `-64n.yaml` | 16K on 36 nodes (debug runs), on 64 nodes (four replicas, not run) |
| `envs/stem-mix-run3-base.yaml` | STEM mix task: DAPO-Math, NuminaMath, science multiple choice; AIME and GPQA validation |
| `experiments/grpo-chonk/megachonk-dapo-16k-megatron-128x16-async-fp8-skip-trunc0.yaml`, `recipes/apertus2-megachonk-vllm-adamw-dapo.yaml` | DAPO-Math only with AIME validation, the setup of the pretrained-Chonk AIME runs (not run) |
| `envs/score-text.yaml` | debug tool: the trainer's log-probabilities of plain text |
| `envs/stem-mix-four-shot.txt` | four worked examples as a prompt prefix for the STEM mix (not run) |

NeMo-RL changes, as patches applied by `setup.sh` (`backends/nemo-rl/apertus2/patches/nemo-rl/`):
- `0050`: the `policy.megatron_cfg.distributed_timeout_minutes` key.
- `0051`: the gradient-norm communicator in the NCCL warm-up.

The runs also use two NeMo-RL changes added with the long-context work: `0040`,
`NRL_NCCL_TIMEOUT_MINUTES` (the megachonk YAML sets it to 60 in `megatron_cfg.env_vars`), and
`0041`, Ray on Slingshot in `ray.sub` (`RAY_NODE_NAME_SUFFIX`, `-hsn0` by default on Hangang
through `sites/hangang.env`). The clone `~/projects/nemo-rl-apertus2-megachonk` has all four.

Other changes: `backends/nemo-rl/apertus2/driver.sh` (W&B console off, unbuffered stdout),
`backends/nemo-rl/preflight.sh` (expert-parallel rule), `sites/hangang.env` (vLLM caches in the
home).

## Launching

The model must be staged first (once). The Chonk staging script does the same steps:

```bash
SRC=/scratch/hanint06/models/megachonk-pretraining-iter_0042000 \
DST=$SCRATCH/models/apertus2-megachonk-iter42000 CHONK_MODEL_STAGE=copy STAGE_SHARDS=link \
bash experiments/grpo-chonk/stage_model.sh
```

(The folder used so far was staged by hand with the same content; the command above has not been
run in this form.)

Every launch sets these in front of `run.sh`:

```bash
COMMON="RAY_NODE_NAME_SUFFIX=-hsn0 WANDB_PROJECT=chonk-rl WANDB_ENTITY=apertus \
  WANDB_KEY_FILE=/home01/hanint06/.wandb-api-key JOB_TIME=2-00:00:00 \
  NEMO_RL_DIR=/home01/hanint06/projects/nemo-rl-apertus2-megachonk"
SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-<run name>"
```

GSM8K, 2K context, 40 nodes (the run above; it used `distributed_timeout_minutes: 30` and none
of the timeout and warm-up patches, which later 16K runs needed):

```bash
env $COMMON RUN=<run name> SBATCH_EXTRA="$SBATCH_EXTRA" \
RECIPE=recipes/apertus2-megachonk-vllm-adamw.yaml bash run.sh envs/gsm8k.yaml
```

STEM mix, 16K context, 48 nodes (the current full run):

```bash
env $COMMON RUN=<run name> SBATCH_EXTRA="$SBATCH_EXTRA" \
bash run.sh envs/stem-mix-run3-base.yaml ++env.should_log_nemo_gym_responses=true
```

DAPO-Math with AIME validation, 16K context, 48 nodes (not run yet):

```bash
env $COMMON RUN=<run name> SBATCH_EXTRA="$SBATCH_EXTRA" \
RECIPE=recipes/apertus2-megachonk-vllm-adamw-dapo.yaml bash run.sh envs/dapo-math.yaml \
  grpo.max_num_steps=140 ++env.should_log_nemo_gym_responses=true
```

A debug run: append `-DEBUG` to the run name and use the 36-node recipes with fewer prompts, for
example `RECIPE=recipes/apertus2-megachonk-vllm-adamw-stem16k-36n.yaml` with
`grpo.max_num_steps=3 grpo.val_at_start=false grpo.num_prompts_per_step=64
policy.train_global_batch_size=1024` and the three `data.num_prompts_per_dataloader.*` set to
16, 32 and 16.

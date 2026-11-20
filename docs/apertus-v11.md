# Apertus-v1.1 on one node: SFT and GRPO

The small dense Apertus-v1.1 models (0.5B, 1.5B and 4B, [swiss-ai/Apertus-v1.1-4B](https://huggingface.co/swiss-ai/Apertus-v1.1-4B);
Apertus 1 tokenizer and chat template, 4,096-token context) can be post-trained on a single node of four GH200 GPUs in
two stages of one to four hours each:

1. **SFT** of a base model on a subset of the public Apertus SFT mixture, with the final Apertus SFT recipe:
   `sft.sh`, which drives [swiss-ai/posttraining](https://github.com/swiss-ai/posttraining) (`backends/posttraining`).
2. **GRPO** on GSM8K from the SFT model (or from a released Instruct model): `run.sh` with
   `experts/stem/dev/gsm8k-cot.yaml` and the recipe `recipes/apertus-v11.yaml` (NeMo-RL, DTensor trainer, vLLM).

Both run on KISTI HANGANG (120 GB GPUs) and CSCS Alps (95 GB GPUs) with the cluster profiles in `sites/`. The models
are not reasoning models: the reasoning is written in the answer itself.

## Setup (once)

Download the models and the mixture, on a machine with internet access (the paths are the defaults of
`backends/posttraining/env` and `recipes/apertus-v11.yaml`):

```bash
for m in 0.5B 1.5B 4B; do
    hf download swiss-ai/Apertus-v1.1-$m --local-dir $SCRATCH/models/apertus-v1.1/Apertus-v1.1-$m
done
hf download swiss-ai/Apertus-v1.1-0.5B-Instruct --local-dir $SCRATCH/models/apertus-v1.1/Apertus-v1.1-0.5B-Instruct
hf download swiss-ai/apertus-sft-mixture --repo-type dataset --local-dir $SCRATCH/data/apertus-sft-mixture
```

Runs of these models log to the W&B project `apertus-v11`: set `WANDB_PROJECT=apertus-v11` (and `WANDB_ENTITY`,
`WANDB_KEY_FILE`) in `site.env` or in the environment of the commands below. The first launch of each stage sets up
the rest: the posttraining and NeMo-RL clones at their pins with the patches of this repository, the Python
overlays, and the mixture in the format `load_from_disk` reads. An optional check job compares the SFT encoding with
the chat template of the released Instruct model and runs one optimizer step:

```bash
bash backends/posttraining/submit.sh v11-check check.sh
```

## SFT

```bash
bash sft.sh <0.5B | 1.5B | 4B> <conversations> [Hydra overrides]
DEBUG=1 bash sft.sh 0.5B 4096          # 3 optimizer steps, run name ending in -DEBUG
```

The recipe is the final Apertus SFT recipe (posttraining: `reproducibility-scripts/sft/final/ademamix.py`): AdEMAMix
(beta3 0.999, alpha 8), one epoch over the subset, global batch 512, loss on assistant tokens only, the `apertus`
chat template with end-of-turn token 68 as EOS, left padding, gradient clipping at 1.0, linear decay after 3%
warm-up, DeepSpeed ZeRO-2 and sequences of up to 4,096 tokens. As in the Apertus-v1.1 report, only the learning
rate changes with the model: it is the model's learning rate at the end of pre-training, the minimum learning rate
of its distillation run (WSD with a 1-sqrt cooldown to a tenth of the peak; `swiss-ai/Megatron-LM-Distill`, branch
`8b_distill`):

| | 0.5B | 1.5B | 4B |
|---|---|---|---|
| Learning rate | 6e-5 | 3e-5 | 2e-5 |
| Conversations for about one hour | 120,000 | 75,000 | 40,000 |
| Conversations for about four hours (recommended) | 500,000 | 320,000 | 170,000 |

The conversations are a random subset of the 3.94M of the mixture (shuffle with seed 42). The final model is linked
as `$SCRATCH/apertus-v11-sft/models/<run name>` (default name `apertus-v11-<size>-sft-n<conversations>`; `sft.sh`
prints the folder), the folder GRPO takes as `policy.model_name`. `LR` replaces the learning rate, `RUN_NAME` the
name and `JOB_TIME` the default limit of 5 hours. On a preemptible partition, `RESUME=1 SAVE_STEPS=100` keeps
checkpoints with optimizer state and lets a requeued job continue from the last one. Throughput, memory settings
and the backend's mechanics: `backends/posttraining/README.md`.

## GRPO

```bash
bash run.sh experts/stem/dev/gsm8k-cot.yaml policy.model_name=<SFT model folder>        # synchronous
RECIPE=recipes/apertus-v11-async.yaml bash run.sh experts/stem/dev/gsm8k-cot.yaml policy.model_name=<folder>
DEBUG=1 bash run.sh experts/stem/dev/gsm8k-cot.yaml                                     # 3 steps, 0.5B Instruct
```

Without `policy.model_name` the run starts from Apertus-v1.1-0.5B-Instruct. One GSM8K epoch is 116 steps of 64
questions with 8 answers each (512 answers, four optimizer steps of 128 at learning rate 2e-6, no KL penalty);
validation uses the 1,319 test questions every 10 steps. `run.sh` prints where the training log, the metrics and
the checkpoints go.

| Recipe | Layout |
|---|---|
| `recipes/apertus-v11.yaml` | Trainer and vLLM share the four GPUs and take turns (colocated, synchronous): every answer comes from the current policy. |
| `recipes/apertus-v11-async.yaml` | vLLM keeps two GPUs and generates the next batch while the trainer updates on the other two (answers at most one step old, importance-sampling correction). |

The task puts each question in the user turn of the chat template, after the instruction in
`shared/prompts/gsm8k-cot.txt` (reason step by step, final answer in `\boxed{}`); the reward is 1 when math-verify
matches the boxed answer to the reference. The Apertus 1 tokenizer marks LaTeX pieces such as `\boxed{` as special
tokens, which NeMo-RL removed when it decoded an answer: a correct answer then earned 0. The recipe keeps them
(`policy.generation.skip_special_tokens: false`, NeMo-RL patch 0062).

The recipe puts one sequence of 4,096 tokens in a micro-batch and recomputes activations, which fits all three
sizes on the 95 GB GPUs of Alps in both layouts (four sequences ran out of memory with 1.5B, at the 8 GB of fp32
logits of a micro-batch). On the 120 GB GPUs of HANGANG, 0.5B also fits four sequences without recomputation, which
is faster and gives the same update:

```bash
bash run.sh experts/stem/dev/gsm8k-cot.yaml policy.sequence_packing.train_mb_tokens=16384 \
    policy.sequence_packing.logprob_mb_tokens=16384 policy.dtensor_cfg.activation_checkpointing=false
```

A one-node job on HANGANG gets no Slingshot network service; `launch.sh` sets `NCCL_NET=Socket
NCCL_NET_PLUGIN=none` for it (`SINGLE_NODE_ENV` in `sites/hangang.env`), and the GPUs communicate over NVLink.

## Results (November 2026)

### SFT

Measured on the answers to the first 64 GSM8K training questions (8 each, at temperature 1.0, in the GRPO setup) and
on the 1,319 GSM8K test questions:

| Model, SFT | Steps | Training time | Mean loss | Answers with foreign script | Mean answer length | GSM8K test |
|---|---|---|---|---|---|---|
| 0.5B, 1 h | 234 | 54.6 min | 1.33 | 89 of 512 | 754 tokens | 1.5% |
| 0.5B, 4 h | 975 | 3.95 h | 1.21 | 146 of 512 | 1,067 tokens | 1.6% |
| 1.5B, 1 h | 147 | 54.8 min | 1.13 | 164 of 512 | 1,209 tokens | 3.9% |
| 1.5B, 4 h | 624 | 4.04 h | 1.03 | 123 of 512 | 946 tokens | 4.5% |
| 4B, 1 h | 78 | 55.6 min | 0.99 | 510 of 512 | 3,998 tokens | 10.2% |
| 4B, 4 h | 332 | 3.25 h | 0.71 | 45 of 512 | 411 tokens | 16.5% |
| 0.5B-Instruct (released) | | | | 4 of 512 | | 6.8% |

One hour of SFT (1 to 3% of the mixture) is too short for 4B: after 78 optimizer steps it rarely ends its answers,
and it answers English questions with stretches of Arabic, Korean, Chinese or Russian. Four hours fix both. On 0.5B
and 1.5B, four hours change little: the drift stays, and GSM8K accuracy rises by under one point. The released
Instruct models start higher (0.5B 6.8%, 1.5B 16.2%, 4B 46.1%); which SFT data and how much of it they used is not
stated in the Apertus-v1.1 report. A lower learning rate does not help: 0.5B at 5e-6 (the Apertus 8B rate, 1 h) has
foreign script in 506 of 512 answers and 1.3% on GSM8K.

### GRPO from the SFT models

Synchronous layout, one sequence per micro-batch with activation checkpointing:

| SFT model | GRPO steps | Accuracy before | Accuracy after | Mean step time | Site |
|---|---|---|---|---|---|
| 0.5B, 1 h SFT | 116 | 1.5% | 2.0% | 25.8 s | Alps |
| 0.5B, 1 h SFT, GRPO learning rate 1e-5 | 116 | 1.5% | 2.1% | 32.4 s | Alps |
| 0.5B, 4 h SFT | 116 | 1.6% | 3.0% | 27.1 s | HANGANG |
| 1.5B, 1 h SFT | 116 | 3.9% | 12.9% | 35.1 s | Alps |
| 1.5B, 4 h SFT | 116 | 4.5% | 16.9% | 25.1 s | HANGANG |
| 4B, 1 h SFT | 32 (job time limit) | 10.2% | 10.5% | 184 s | Alps |
| 4B, 4 h SFT | 60 | 16.5% | 37.5% | 37.7 s | Alps |

Accuracy at every validation (steps 0, 10, ..., 110 and the last step), W&B project `apertus/apertus-v11`:

- 4B, 4 h SFT: 16.5, 16.2, 16.8, 22.5, 28.8, 31.8, 37.5% (44 minutes; run `uq61nqbp`)
- 1.5B, 4 h SFT: 4.5, 3.3, 3.1, 4.9, 4.0, 5.5, 9.6, 12.2, 15.2, 14.7, 16.5, 17.4, 16.9% (55 minutes; `crjqpiro`)
- 0.5B, 4 h SFT: 1.6, 1.4, 1.6, 1.4, 2.2, 1.8, 2.4, 2.4, 2.4, 2.9, 2.7, 3.6, 3.0% (59 minutes; `ue3ff2oj`)

GRPO needs a few percent of correct answers to start from: at 1 to 2% almost every group of 8 answers earns no
reward, and the policy moves slowly (0.5B) or not at all; a five times higher GRPO learning rate does not change that.
The 1.5B model ends one epoch at the level of the released 1.5B Instruct model (16.2% before GRPO). After one hour of
SFT the 4B model rarely ends its answers (3,998 tokens on average), which makes every step slow and earns no reward.

### GRPO from the released Instruct models

Apertus-v1.1-0.5B-Instruct, one epoch, HANGANG, four sequences per micro-batch without recomputation:

| Layout | Accuracy before | Accuracy after | Mean step time | Job time |
|---|---|---|---|---|
| Synchronous (job 1308872) | 6.8% | 14.7% | 18.3 s | 40 min |
| Asynchronous (job 1308873) | 5.8% | 13.5% | 15.7 s | 55 min (two attempts) |

Accuracy was still rising at the end of the epoch. The asynchronous job lost its first attempt at the step-50
checkpoint: the checkpoint's background process picked a port another process held (`EADDRINUSE`), and the job
restarted from step 0.

1.5B and 4B Instruct on Alps (95 GB GPUs), 20 steps, one sequence per micro-batch with activation checkpointing:

| Model | Layout | Accuracy at step 0 | Accuracy at step 20 | Mean step time |
|---|---|---|---|---|
| 1.5B Instruct | synchronous | 16.2% | 17.3% | 26.7 s |
| 1.5B Instruct | asynchronous | 15.8% | 17.2% | 22.7 s |
| 4B Instruct | synchronous | 46.1% | 47.4% | 50.1 s |
| 4B Instruct | asynchronous | 43.5% | 44.4% | 49.1 s |

One epoch therefore takes about 45 minutes on 1.5B and about 95 minutes on 4B. On 4B the asynchronous layout hides
generation completely (0.2 s of a 51 s step at step 16), but training on two GPUs takes 39 s, against 18 s on four
GPUs in the synchronous layout, where generation adds 16 s: both layouts end up near 50 s.

# Recipes: how to train

A recipe is the model and the training algorithm, apart from what is trained on: the checkpoint and its message
format, the Megatron trainer (parallelism, optimizer, learning rate, memory), vLLM inference (precision,
parallelism) and the GRPO algorithm (batch, async, staleness, loss, advantages). An expert names its recipe in its
`base.yaml` (`recipe: recipes/<file>`), and `run.sh` puts the recipe first, under `experts/main.yaml` and the
expert's files. Each active recipe is one file that builds only on NeMo-RL's own defaults
(`examples/configs/grpo_math_1B.yaml` in the NeMo-RL clone), so reading it shows everything it sets.

| Recipe | Model | Used by |
|---|---|---|
| `chonk-sftv02.yaml` | Chonk SFT v0.2 (iteration 18,315), async GRPO: 512 x 16, one optimizer step, lr 2e-6 constant, staleness 4, vLLM FP8 | STEM, and the templates of medical, safety, knowledge, translation |
| `megachonk.yaml` | MegaChonk pretraining checkpoint (iteration 42,000), pretraining format; TP4 x PP8 x EP16 (32 trainer nodes or more), vLLM replicas of 8 nodes, staleness 2 | template: no expert yet; needs a tier with 32+ trainer nodes |

`sft_export` names an SFT checkpoint (relative to $SCRATCH or absolute; a symlink works); run.sh stages it and sets
`policy.model_name`, so the recipe names it once. `nodes: {trainer, inference}` is the recipe's base node layout,
the `cheap` tier; run.sh derives `lite`, `expensive` and `ultra` from it (`experts/main.yaml`). `context_parallel: {tokens_per_gpu, memory_fraction}` sets context
parallelism from the run's `context:`: CP = context / tokens_per_gpu, rounded up to a power of 2 (Chonk 8,192, MegaChonk
4,096), packing = context, KCP above CP 4, and PyTorch's share of a trainer GPU.

A variant (another optimizer, another checkpoint) is a small file with `defaults: <recipe>.yaml` and what it
changes; a new model gets a flat file of its own when it is trained. Keep a recipe free of what other files set:
datasets and environments (`experts/<expert>/base.yaml`), sequence lengths, packing and context parallelism
(`run.sh`, from `context`), the node tier, checkpoints, validation schedule and logging
(`experts/main.yaml`).

`archive/` holds the recipes of earlier runs as they were, and `archive/chain/` the experiment chain they build on
(GSM8K, DAPO, FP8, Apertus 2 format, ...): the archived task files under `experts/*/archive/` and `dev/`, the
coding and experiment launchers and the commands in `docs/runs.md` use them. `chonk-sftv02.yaml` was written
out from `archive/apertus2-chonk-sftv02-grpo.yaml` (14 files deep); the STEM run gives the same config with either,
apart from five settings of the DTensor trainer, which this recipe switches off.

## To consider

- **Weight decay with Adam on a Muon-trained model** (2026-11-22): `chonk-sftv02.yaml` inherits Adam's
  `weight_decay: 0.01` from NeMo-RL's defaults, while Chonk was pretrained and SFT-trained with Muon. Advice from a
  colleague: either train RL with Muon, or with Adam and no weight decay. At lr 2e-6 the decay shrinks the weights by
  2e-8 per step (about 3e-6 over 135 steps), so its effect is small; setting
  `policy.megatron_cfg.optimizer.weight_decay: 0.0` costs nothing. The larger open question is Adam's update geometry
  on a Muon-trained model (an RL run with Muon: `recipes/archive/*muon*` have the settings of earlier trials).

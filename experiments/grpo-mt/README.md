# GRPO for machine translation

Experiment workspace for training a translation model with GRPO using the
repository's NeMo-RL backend.

Before adding a runnable configuration, choose:

- Source and target languages.
- Base model.
- Parallel training data and a held-out validation set.
- Translation prompt and reward function.

The first training run should be a small smoke test to verify data loading,
translation generation, reward calculation, and a GRPO update.

## Data and reward design

Use a single many-to-many training file, with one directed translation example
per row: `source_lang`, `target_lang`, `source`, and `reference` internally.
The first test uses `data_rl/es-de.jsonl` directly, mapping its `src`, `tgt`,
`source_language`, and `target_language` fields to those internal names.
Keep validation examples separate and keep references out of the model's input
prompt.

## Translation prompts

The data adapter must support multiple translation instruction templates, using
an explicit language-code-to-name mapping (for example, `es: Spanish` and
`de: German`). Render full language names in prompts; retain codes in dataset
metadata for reward scoring and per-direction metrics. Reject unmapped codes
rather than falling back to displaying a code in the prompt.

Example templates:

```text
Translate the following {source_language} text into {target_language}.
Return only the translation.

{source}
```

```text
Provide a {target_language} translation of this {source_language} passage.
Output only the translated text.

{source}
```

Select templates reproducibly using a configured seed. All candidates in a GRPO
group must use the same rendered prompt. Record which template was selected
so runs can be inspected and reproduced.

## Reward

Adapt [WALAR](https://arxiv.org/html/2603.13045v1#S3.SS2) by adding a
reference-based, sentence-level chrF++ term:

```text
reward = -25                                      if language alignment fails
reward = metricx(source, hypothesis, reference)
         + 20 * word_alignment(reference, hypothesis)
         + chrf_weight * chrf_pp(hypothesis, reference) / 100
         + cometkiwi_weight * cometkiwi(source, hypothesis)
                                                  otherwise
```

Adapt WALAR to use reference-based MetricX alongside CTFAlign reference–hypothesis word alignment
and a GlotLID language gate. Alignment uses the full generated translation without language filtering. The MetricX term is the
negative error score (higher is better, in [-25, 0]); do not add MetricX's raw
error score with a positive sign.
The language failure branch overrides all other terms, including chrF++.

Compute chrF++ on the original generated translation and its reference, using
[SacreBLEU](https://github.com/mjpost/sacrebleu) with `char_order=6`,
`word_order=2`, `beta=2`, `lowercase=False`, `whitespace=False`, and
`eps_smoothing=False`. Divide its 0–100 score by 100 before weighting it.

Proposed initial `chrf_weight`: 20 (to be tuned; this is our addition, not a
WALAR setting). Require a nonempty reference for every training example.
Log MetricX, word alignment, language alignment, chrF++, and total reward separately,
including summaries per translation direction.

The reward implementation and NeMo-RL environment live in
[`backends/nemo-rl/mt`](../../backends/nemo-rl/mt/README.md), with setup and testing
instructions there. `scorer.example.json` is the reward-actor configuration
template. The JSONL adapter and one-step `qwen3-8b-smoke.yaml` recipe are implemented;
the full training recipe remains to be configured.

## Qwen3-8B smoke test

`qwen3-8b-smoke.yaml` uses `Qwen/Qwen3-8B` in non-thinking mode, with three
source prompts and four candidates per prompt for one GRPO step (12 candidates).
It allocates one node: three GPUs for policy training/generation and one GPU
for the reward actor. The global training batch is 12, divisible by the three
policy GPUs. Checkpoint saving and validation runs are disabled; 32 dataset
rows are held out of training for later validation.

First validate the file on the login node:

```bash
python3 backends/nemo-rl/mt/check_data.py --input /absolute/path/to/data_rl/es-de.jsonl
```

Then, after the model-backed and Ray reward checks pass:

```bash
sbatch experiments/grpo-mt/run_smoke.sh
```

Submit from the repository root. `run_smoke.sh` defaults to
`/capstor/scratch/cscs/$USER/data_rl/es-de.jsonl` and
`experiments/grpo-mt/scorer.json` in this checkout. Override `MT_DATA_PATH` or
`MT_SCORER_CONFIG` at submission if needed; these paths must be absolute and
visible inside the container. The script runs Ray directly in its allocation,
using the shared launcher with `RUN_IN_ALLOCATION=1`. Slurm output goes to
`mt-smoke-<jobid>.out` in the repository root; Ray logs remain under
`$OUT/slurm/<jobid>-logs/` on scratch.

The instruction
model `Qwen/Qwen3-8B` is a different checkpoint from the earlier math test's
`Qwen/Qwen3-8B-Base`; its first run may download weights. Prompt templates and
language mappings are defined in `backends/nemo-rl/mt/mt_reward/prompts.py`.

## Apertus 8B Instruct 2509 one-step configuration

`apertus-8b-instruct-2509-smoke.yaml` selects `swiss-ai/Apertus-8B-Instruct-2509`,
with 3 prompts × 4 candidates, one GRPO step, three policy GPUs and one reward GPU.
It uses the same verified scorer config. Checkpoints and validation are disabled.
The processor renders this checkpoint's own tokenizer chat template, without
inserting Qwen control tokens or exposing references. `enable_thinking=False`
disables reasoning in templates that support the option; templates without it
ignore the extra argument.

This text-only Apertus model supports `AutoModelForCausalLM` in Transformers
4.56.0 and later; see its [model card](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509).
The wrapper checks the actual config and tokenizer in the training container
before launching:

```bash
sbatch experiments/grpo-mt/run_apertus_smoke.sh
```

It requests one exclusive four-GPU node with a ten-minute limit and uses the same
`MT_DATA_PATH` and `MT_SCORER_CONFIG` overrides as `run_smoke.sh`. It also works via
`bash experiments/grpo-mt/run_apertus_smoke.sh` inside an existing Slurm allocation.
The preflight prints the rendered prompt and checks the causal-model mapping
without downloading weights. Model loading, backward-pass and the separate
vLLM generation runtime still need verification on Clariden. Logs use
`mt-smoke-<jobid>.out`; training metrics use the run name
`apertus-8b-instruct-2509-mt-smoke`.

## Additional reference-free QE term

The scorer JSONs enable `Unbabel/wmt22-cometkiwi-da` with `cometkiwi_weight=20`.
Run `bash backends/nemo-rl/mt/setup-kiwi.sh` before using these configs; COMET
requires an isolated Python 3.12 environment. The supplied `scorer.json` points
to the default scratch environment; customize `cometkiwi_python` for another
location. Setting the weight to zero restores the previous reward and skips
Kiwi loading. See the backend README for setup, checkpoint access and the full
reward check, which now reports raw Kiwi scores and weighted bonuses.

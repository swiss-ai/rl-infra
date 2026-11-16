# MT reward environment

Implements an adapted WALAR reward with CTFAlign reference alignment and sentence-level chrF++ for the NeMo-RL commit in `../pin`.
The training-side `mt` Ray environment calls a GPU reward actor in the same
Ray cluster and Slurm job. The actor owns the scoring models and runs in a
separate Python environment. A small source patch reserves its GPU resources;
it does not change NeMo-RL's dependency fingerprint or affect math runs.

## Reward

```text
if output is empty or GlotLID predicts a different target language:
    reward = -25
else:
    reward = -MetricX_error(source, original_output, reference)
             + alignment_weight * alignment(reference, MaskLID_target_text)
             + chrf_weight * sentence_chrF++(original_output, reference) / 100
             + cometkiwi_weight * COMET_Kiwi(source, original_output)
```

Alignment and chrF++ weights default to 20. COMET-Kiwi defaults to disabled
in code (`cometkiwi_weight=0`); the supplied scorer JSONs enable it with weight 20. The gate is based on the original output, matching
[WALAR's released code](https://github.com/LeiLiLab/WALAR/blob/963746136c5fbff4d3a36cf09f3e4ae1f33ebcb1/openrlhf/openrlhf/cli/serve_rm.py).
MetricX is adapted to **reference-based** scoring: its input contains the source,
generated translation, and reference in the order specified by
[Google's MetricX-24 inference code](https://github.com/google-research/metricx/blob/main/metricx24/predict.py).
The same hybrid checkpoint supports both modes. We use the reference-based mode,
remove the final EOS token before inference, and negate the predicted error.
MetricX inference explicitly disables generation caching: it needs only one
decoder step, and WALAR's standalone decoder configuration otherwise shares
self/cross-attention cache entries under Transformers 4.57.6.
The default input limit is 1536 tokens, matching the documented training limit;
longer inputs are truncated. The adapter rejects a missing reference or an
incompletely loaded checkpoint. MetricX component rewards are logged as
`metricx`, rather than the former `qe` field.
MaskLID extracts target-language segments for alignment against the reference,
with both sides tokenized as the target language. [CTFAlign](https://github.com/ZurichNLP/CTFAlign)
uses its coarse-to-fine method, argmax extraction, and word units. The bonus is
F1 of unique covered reference/hypothesis words, so repeated edges do not inflate
coverage. The default encoder is `BidirLM/BidirLM-1B-Embedding`; override
`alignment_model` and optionally `alignment_layer` in the scorer JSON.
BidirLM uses the `transformers-v4` revision with Transformers 4.57.6 and
`alignment_trust_remote_code=true`, as required by its custom architecture.
The revision is resolved to a local Hugging Face snapshot before CTFAlign loads
it, so tokenizer and model code come from the same revision. For offline runs,
set `alignment_model` to that snapshot directory, remove `alignment_revision`,
and keep the explicit layer and trust settings.
BidirLM uses hidden layer 15 for alignment. Other encoders use CTFAlign's
document-level defaults unless `alignment_layer` is set. The encoder
handles inputs beyond its context window in chunks; the MetricX `max_length`
setting does not truncate alignment inputs.
Chinese uses WALAR's HanLP tokenization and punctuation filtering; other
languages use whitespace word splitting. Segmented words are joined with spaces
before CTFAlign so its returned indices match the coverage denominators.

The adaptation follows the formula above even when the MetricX reward is exactly
-25 and language identification passes. Upstream's code also suppresses the
alignment bonus at that MetricX reward floor; our explicitly language-based gate does not.
Setting `chrf_weight=0` removes the chrF++ term.

SacreBLEU settings are `char_order=6`, `word_order=2`, `beta=2`,
`lowercase=False`, `whitespace=False`, `eps_smoothing=False`.
Reference strings must be nonempty. MetricX, CTFAlign, and chrF++ use the reference;
the policy never receives it.
Invalid scores and model errors propagate through Ray and stop training instead
of silently substituting a reward.

## GPU actor setup

Run the normal repository `setup.sh` first, then install the MT resource patch:

```bash
bash backends/nemo-rl/mt/setup.sh
```

The MT setup script applies the code-only patch as a commit to the pinned
NeMo-RL clone and is safe to re-run. The MT launcher checks that the patch is
installed before submitting. Existing math launches do not use the MT branch
of the resource accounting.

Use a separate environment with **the same Python version and Ray version as
the training container**, visible at the same absolute path on every compute
node, with a CUDA-enabled PyTorch build appropriate for GH200. Your sanity check
reported Python 3.13.13. This version matching is required by
[Ray's runtime environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html).
Install `requirements-scorer.txt` into that environment. The pinned Transformers 4.57.6 supports BidirLM's `transformers-v4` branch.
Run the model-backed MetricX check below after updating the scorer environment. Leave `/opt/nemo_rl_venv` and `/opt/ray_venvs` alone.
The scorer environment needs no NeMo-RL or OpenRLHF installation.

Provide a checkout of `https://github.com/LeiLiLab/WALAR` at commit
`963746136c5fbff4d3a36cf09f3e4ae1f33ebcb1`. Only `code/models.py` and
`code/masklid.py` are imported. Obtain the model assets listed in WALAR's
README, including GlotLID's `model.bin` and MaskLID's `model_v3.bin`.

Copy `experiments/grpo-mt/scorer.example.json` to a personal config and set the
local paths. `languages` maps dataset language codes to GlotLID labels. Include
every source and target language. This list also defines the candidate
languages considered by MaskLID; keep it fixed across comparisons. For Chinese,
add `hanlp==2.1.1` and set `chinese_tokenizer` to the downloaded
`coarse_electra_small_20220616_012050` tokenizer directory. Model IDs can be
replaced with downloaded local directories for offline execution.

## NeMo-RL integration

Launch an MT training config with `backends/nemo-rl/mt/launch.sh`, using the
same arguments as `backends/nemo-rl/launch.sh`. The wrapper adds this package to
`PYTHONPATH` for the driver and Ray actors, registers `mt` and its Python
environment, then invokes the upstream GRPO entry point.

The training config needs the following section (the scorer JSON can instead
be expressed inline under `scorer`):

```yaml
env:
  mt:
    scorer_python: /absolute/path/to/mt-venv/bin/python
    resources:
      num_nodes: 1
      gpus_per_node: 1
      num_cpus: 4
    timeout: 1800
    scorer_config: /absolute/path/to/scorer.json
data:
  default:
    env_name: mt
    # An MT data processor must supply the metadata described below.
```

Alternatively, replace `scorer_config` with an inline `scorer` dictionary.
The actor loads the models once, then handles batched Ray
calls sequentially. Ray assigns one GPU through `CUDA_VISIBLE_DEVICES`; the
actor uses logical `cuda:0` and preserves Ray's device isolation. The CPU
NeMo-RL environment waits for model readiness and propagates actor failures
and timeouts. Its `shutdown()` method explicitly kills its owned GPU actor.

This initial implementation supports **one reward actor on one GPU**. That
GPU must fit MetricX XXL, the CTFAlign encoder, and their activations. On a two-node job,
NeMo-RL reserves one node for reward scoring and uses the other node for
policy training/inference. On a one-node job with colocated inference, the
reward uses one GPU and the policy gets the remaining three; model parallelism
and batch settings must be compatible with those three GPUs. The included
`qwen3-8b-smoke.yaml` uses this one-node layout, with 3 prompts × 4 candidates
and a global batch of 12. Extra GPUs on a reward-only node in a multi-node job
are left idle in this initial implementation.

Each processed example must provide these `extra_env_info` keys:

```json
{"source_lang":"en","target_lang":"de","source":"Hello world.","reference":"Hallo Welt."}
```

The environment reads the hypothesis from the final assistant message and ends
the episode after scoring. It returns a CPU float32 reward tensor and boolean
termination tensor through NeMo-RL's `EnvironmentReturn` interface. Component
scores are stored in the terminal environment observation for rollout metrics.
Metrics include overall and per-direction means. MetricX/alignment means include
only examples passing the language gate; language alignment reports pass rate.

The MT integration includes `mt_reward.data.TranslationDataset` and the
`translation_data_processor`, registered by the MT entry point. The dataset
loads the parallel JSONL directly, maps language codes to full names, and
chooses one of four prompt templates reproducibly before rollouts are grouped.
The processor uses the policy tokenizer's own chat template with thinking
disabled and places references only in reward metadata. See `experiments/grpo-mt/README.md` for the one-step
Qwen3-8B recipe and launch command.

## CPU verification

Only SacreBLEU is needed for the default tests; they do not download models:

```bash
python -m unittest discover -s backends/nemo-rl/mt -v
```

The tests cover scoring, rollout metrics, actor batch processing and error
propagation, and Ray runtime/resource options without downloading models or
starting a Ray cluster. Full GPU inference and Ray scheduling must be checked
on Clariden with the configured model assets.

## Model-backed MetricX check

The unit tests verify adapter behavior with mocked MetricX outputs; they do not
prove the real checkpoint loads or that the GPU environment is compatible.
Before training, run this in the scorer's Python environment inside a GPU
allocation, from the repository root:

```bash
/absolute/path/to/mt-venv/bin/python backends/nemo-rl/mt/check_metricx.py \
    --config /absolute/path/to/scorer.json
```

This loads the same MetricX adapter and actual checkpoint used by the Ray actor,
checks loading diagnostics, and runs reference-based inference on two examples.
It reports package versions, raw errors, and negative rewards. Inspect that the
correct translation has lower error than the unrelated translation; the script
does not impose a universal quality threshold. It needs no Ray cluster, CTFAlign encoder,
GlotLID, or MaskLID models. `--input examples.jsonl` accepts your own examples
with `source`, `hypothesis`, and `reference` fields. This check validates MetricX
inference; full Ray execution remains a separate cluster check.

## Document alignment check

Run `check_alignment.py` in the scorer environment on a GPU to load only the
production CTFAlign/BidirLM component (layer 15), without MetricX, language
models, or Ray. From the `rl-infra` root on Clariden:

```bash
srun --account=infra01 --partition=debug --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:10:00 --environment=nemo-rl \
  /iopsstor/scratch/cscs/$USER/mt-reward-venv/bin/python \
  "$PWD/backends/nemo-rl/mt/check_alignment.py" \
  --config "$PWD/experiments/grpo-mt/scorer.json" \
  --output "$PWD/alignment-check.jsonl"
```

The built-in German documents compare an identical-reference control, an
idiomatic paraphrase, unrelated content, and plausible text with incorrect facts.
The table reports reference coverage, hypothesis coverage, coverage F1, and
`alignment_weight * F1`: the alignment contribution used by GRPO, not the total
reward. The JSONL output contains all text, scores, index pairs, and aligned word
strings for inspection. No universal quality threshold is imposed.

The check treats hypotheses as already filtered target-language text. Training
also runs the language gate and MaskLID before alignment; those steps can change
the actual bonus. High word coverage alone does not establish semantic or factual
correctness; inspect the negative controls and word pairs.

Use `--input documents.jsonl` for your own data, one object per line with
`name`, `target_lang`, `reference`, and `hypothesis`. Set the desired target
language mapping in the scorer config. `--device cpu` is available for machines
without CUDA. The first run downloads the model if it is not cached, which may
require a longer allocation than the example's ten minutes.

The BidirLM adapter captures the selected packed layer activation directly,
disables upstream intermediate-state output, removes BidirLM's internally
prepended BOS, and restores the batch dimension. This works around the
`transformers-v4` implementation's broken intermediate-state unpacking and
end-token trimming. It requires CTFAlign's single, unpadded encoding calls and
checks the token count before returning embeddings. The tokenizer is reloaded
with `fix_mistral_regex=False` before encoding, preserving its Gemma-style
whitespace split. Transformers 4.57.6 falsely identifies this local custom model
as Mistral; applying the Mistral patch to its standalone `Split` pre-tokenizer
raises a TypeError and would change the checkpoint's tokenization. The initial
CTFAlign construction may still print that false-positive warning before the
explicit reload.

## Full document reward check

After the alignment-only check, run all production components together, using
`RewardWorker` exactly as the GPU Ray actor does. From the `rl-infra` root:

```bash
srun --account=infra01 --partition=debug --nodes=1 --ntasks=1 \
  --gpus-per-node=1 --cpus-per-task=4 --time=00:10:00 --environment=nemo-rl \
  /iopsstor/scratch/cscs/$USER/mt-reward-venv/bin/python \
  "$PWD/backends/nemo-rl/mt/check_reward.py" \
  --config "$PWD/experiments/grpo-mt/scorer.json" \
  --output "$PWD/reward-check.jsonl"
```

The 10-minute limit is a maximum runtime; the job exits as soon as the check
finishes. All scorer models must fit on one GPU.
The six examples include the four alignment documents with a Spanish source,
plus a Spanish source copy and empty output. GlotLID and MaskLID run normally.
The table reports the language gate, negative MetricX error, alignment F1,
weighted alignment bonus, raw chrF++, weighted chrF++ bonus, and total reward.
Gate failures get -25, with both effective bonuses reported as zero.

The built-in check asserts that the good paraphrase beats unrelated content and
incorrect facts in both MetricX and total reward, and verifies the language/empty
penalties. A failed ranking exits nonzero after saving the scores; it means this
example did not pass the reward-quality check, even if inference worked. These
are example-specific sanity checks, not a general metric evaluation. Ray actor
startup/scheduling still requires the separate `check_ray_reward.py` check.
Use `--input` for custom JSONL containing `source_lang`, `target_lang`, `source`,
`reference`, `hypothesis`, and optionally `name`; custom inputs print scores
without applying the built-in ranking assertions. Configured language mappings
must cover all input languages (the built-ins use `es` and `de`).

## COMET-Kiwi reference-free reward

The supplied scorer configs add `20 * cometkiwi(source, original_output)` using
[Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da).
This input contains only `src` and `mt`, never the reference or masked output.
The raw higher-is-better score (documented range 0–1) is logged as `cometkiwi`;
`check_reward.py` also prints the weighted `cometkiwi_bonus` and includes it in
the total. The language gate overrides this term too. Nonfinite/out-of-range
scores and subprocess failures stop scoring. Set `cometkiwi_weight=0` to disable
both inference and model loading, reproducing the previous reward.

COMET 2.2.7 requires NumPy <2, incompatible with the Python 3.13/NumPy 2 Ray
scorer. Install it in a separate Python 3.12 environment on shared scratch:

```bash
bash backends/nemo-rl/mt/setup-kiwi.sh
```

The script uses `uv` to provision Python 3.12 and dependencies; it requires
network access and checks that PyTorch is a CUDA build. The Kiwi requirements
pin setuptools 80.9.0 because COMET's torchmetrics 0.10.x imports `pkg_resources`,
which newer setuptools releases removed. Setup runs outside the
NeMo-RL project with uv config discovery disabled and inherited pip/uv constraints
removed, so container training pins do not leak into the Kiwi environment. `KIWI_VENV` overrides
its default `/iopsstor/scratch/cscs/$USER/mt-kiwi-venv`. Update `cometkiwi_python`
in the scorer JSON if using a different path. No changes to the training or
existing scorer venv are needed. Accept the checkpoint's Hugging Face access
conditions and provide `HF_TOKEN` if necessary.

The scorer starts one persistent Kiwi subprocess on the same logical `cuda:0`,
inheriting Ray's `CUDA_VISIBLE_DEVICES` and Hugging Face credentials/cache.
It loads once and scores sequentially without starting Lightning trainers or
additional Ray actors. It strips parent Python environment paths so Python
versions remain isolated. On Linux the subprocess exits when its owner dies.
All reward models must now fit on the same GPU, including Kiwi.

Rerun the existing ten-minute `check_reward.py` command after setup to verify
real Kiwi loading and inference alongside MetricX and alignment. CPU tests do
not load COMET weights. Kiwi is a sentence-level QE model: its encoder's own
context limit/truncation applies to document inputs; this integration does not
add document segmentation or aggregate sentence scores.

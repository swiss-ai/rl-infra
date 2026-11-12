# MT reward environment

Implements WALAR + sentence-level chrF++ for the NeMo-RL commit in `../pin`.
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
             + alignment_weight * alignment(source, MaskLID_target_text)
             + chrf_weight * sentence_chrF++(original_output, reference) / 100
```

Both weights default to 20. The gate is based on the original output, matching
[WALAR's released code](https://github.com/LeiLiLab/WALAR/blob/963746136c5fbff4d3a36cf09f3e4ae1f33ebcb1/openrlhf/openrlhf/cli/serve_rm.py).
MetricX is adapted to **reference-based** scoring: its input contains the source,
generated translation, and reference in the order specified by
[Google's MetricX-24 inference code](https://github.com/google-research/metricx/blob/main/metricx24/predict.py).
The same hybrid checkpoint supports both modes. We use the reference-based mode,
remove the final EOS token before inference, and negate the predicted error.
The default input limit is 1536 tokens, matching the documented training limit;
longer inputs are truncated. The adapter rejects a missing reference or an
incompletely loaded checkpoint. MetricX component rewards are logged as
`metricx`, rather than the former `qe` field.
MaskLID extracts target-language segments for alignment. BGE-M3 alignment uses
layer 24, intersected directional softmax probabilities above 1e-3, and F1 of
unique covered source/target words. Truncated words remain in coverage denominators.
Chinese uses WALAR's HanLP tokenization and punctuation filtering; other
languages use whitespace word splitting, as in WALAR.

The adaptation follows the formula above even when the MetricX reward is exactly
-25 and language identification passes. Upstream's code also suppresses the
alignment bonus at that MetricX reward floor; our explicitly language-based gate does not.
Setting `chrf_weight=0` removes the chrF++ term.

SacreBLEU settings are `char_order=6`, `word_order=2`, `beta=2`,
`lowercase=False`, `whitespace=False`, `eps_smoothing=False`.
Reference strings must be nonempty. Both MetricX and chrF++ use the reference;
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
Install `requirements-scorer.txt` into that environment. The pinned Transformers version matches WALAR's
regression implementation. Leave `/opt/nemo_rl_venv` and `/opt/ray_venvs` alone.
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
GPU must fit MetricX XXL, BGE-M3, and their activations. On a two-node job,
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
The processor uses Qwen3's non-thinking chat template and places references
only in reward metadata. See `experiments/grpo-mt/README.md` for the one-step
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
does not impose a universal quality threshold. It needs no Ray cluster, BGE-M3,
GlotLID, or MaskLID models. `--input examples.jsonl` accepts your own examples
with `source`, `hypothesis`, and `reference` fields. This check validates MetricX
inference; full Ray execution remains a separate cluster check.

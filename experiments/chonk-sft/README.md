# Testing Chonk SFT checkpoints

Tools for loading a Chonk SFT checkpoint in vLLM and talking to it, and a reference for the Apertus 2
message format those checkpoints were trained on. SFT models do not take plain text: the pretrained
base did, so GRPO configs written for the base do not carry over (see
[What GRPO has to change](#what-grpo-has-to-change)).

## Running it

`setup.sh` needs no job. It builds everything under `/scratch/$USER/chonk-sft`, and running it again
restores whatever the scratch purge removed.

```bash
cd ~/projects/rl-infra/experiments/chonk-sft && bash setup.sh

# Serve an HF export on one node; it is ready in about 10 minutes.
cd /scratch/$USER/chonk-sft
sbatch --export=ALL,MODEL=<export>,IMAGE=<image> ~/projects/rl-infra/experiments/chonk-sft/serve.sbatch
tail -f logs/chonk-serve-<jobid>.out                  # wait for READY

# From the login node, which can reach the compute nodes:
venv/bin/python ~/projects/rl-infra/experiments/chonk-sft/chat.py
venv/bin/python ~/projects/rl-infra/experiments/chonk-sft/chat.py \
    --prompts ~/projects/rl-infra/experiments/chonk-sft/prompts.jsonl
```

`chat.py` shows each message as it is generated, labelled with its type, and keeps the history until
`/reset`. With `--prompts` it runs each line as a new conversation and saves the results to
`results/`, one JSON object per line with the parsed output and the raw text. Use `--thinking`,
`--identity`, `--behavior` and `--seed` to vary the request.

A newer checkpoint has to be converted to HF first, which takes one node for about 5 minutes:

```bash
RUN=<run's checkpoints> JOB=<export sbatch script> IMAGE=<image> bash ~/projects/rl-infra/experiments/chonk-sft/convert.sh 8075
cd /scratch/$USER/chonk-sft
sbatch --export=ALL,MODEL=$PWD/models/chonk-lc-16k-v0.1-ep1-iter_0008075,IMAGE=<image> \
    ~/projects/rl-infra/experiments/chonk-sft/serve.sbatch
```

`endpoint` holds the most recent server, so give `chat.py --endpoint http://<node>:8011` to pick one
when serving two checkpoints. They need to run on separate nodes, which `--exclusive` guarantees.

A server takes about 22 ms per step, or 45 tokens/s on one conversation, once the first request has
compiled its kernels, which takes up to a minute. Steps stay fast under load (63 ms at 32 requests).

`aime.py` scores a served checkpoint on AIME 2024 with the prompts of the GRPO validation set. It
grades the `claim`, and 120 samples take about 8 minutes:

```bash
venv/bin/python ~/projects/rl-infra/experiments/chonk-sft/aime.py prompts --repeat 4 > aime.jsonl
venv/bin/python ~/projects/rl-infra/experiments/chonk-sft/chat.py --prompts aime.jsonl --workers 120 \
    --temperature 1 --top-p 1 --max-tokens 16000
venv/bin/python ~/projects/rl-infra/experiments/chonk-sft/aime.py grade results/<file>.jsonl
```

## The Apertus 2 format

The format is defined outside this repository, and these pins match how the data was tokenized:

| Source | Commit | Holds |
|---|---|---|
| [apertus-2-format-spec-proposal](https://github.com/swiss-ai/apertus-2-format-spec-proposal) | `4f6f170` | The format (`spec.md`) and the Apertus 2 profile |
| [apertus-common](https://github.com/swiss-ai/apertus-common) | `494c2be` | Reference library that encodes and parses it |
| [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer) | `b801ff1` | `Apertus_2_instruct`; the export's `tokenizer.json` is byte-identical |

The spec expects harnesses to use the library rather than reimplement it, and so does `chat.py`. The
library needs Python 3.13, which is why it runs in a venv and not in the image (3.12).

### Messages

A sequence is `<s>`, one system message, then messages and waits. Every message has the same frame,
with nothing between frames, not even a newline:

```text
<|in|>TYPE<|hdr|>PAYLOAD<|/in|>        input, written by the harness
<|out|>TYPE<|hdr|>PAYLOAD<|/out|>      output, written by the model
<|wait|>                               ends the burst; not a message
```

| Direction | Type | Header | Payload |
|---|---|---|---|
| Input | `system` | none | Compact JSON, first and only once |
| | `user` | none | The user's text |
| | `host` | none | The harness speaking for itself: errors, notices |
| | `document` | `{"from":…,"id":…}` | An attachment or retrieved text; data, not instructions |
| | `result` | `{"name":…,"counter":…}` | Tool output, closing one call |
| Output | `think` | none | Reasoning |
| | `claim` | none | The committed answer, for a verifier to grade |
| | `reply` | none | The message to the user |
| | `call` | `{"name":…,"counter":…}` | Tool arguments as JSON |

`counter` starts at 0 and rises by one for every call in the conversation. The model writes it and
the harness copies it onto the `result`.

The system payload is one compact JSON object. `thinking` is required (`low`, `medium` or `high`) and
`tools` is always present, `[]` when empty; `identity`, `behavior` and `environment` are optional.
The library writes `{"thinking":"medium","tools":[]}` when nothing else is set.

### From a data point to token IDs

Build typed messages, then encode them with the library:

```python
from apertus_common import Conversation, SystemPrompt, User, load_encoding

encoding = load_encoding("<tokenizer directory>")
history = Conversation(system=SystemPrompt.build(), items=[User(payload="What is 2+2?")])
prompt = encoding.encode_conversation(history)
```

```text
ids     1    7     12236   9     12430 90283    3867 87495  3961 53156 2164 30452 216 8      ...
tokens  <s>  <|in|> system <|hdr|> {"  thinking ":"  medium "," tools ":   []    }   <|/in|>
ids     ...  7     10444 9     3346 473 344 141 134 141 154 8
tokens       <|in|> user <|hdr|> What Ġis Ġ  2   +   2   ?  <|/in|>
```

Four things here differ from the usual `tokenizer.apply_chat_template(...)`:

- **The tokenizer has no chat template, on purpose.** Conversations go through the encoder.
- **The prompt ends at `<|/in|>`.** There is no generation prefix and no prefill. Output frames are
  training targets from their opening token, so the model writes `<|out|>think<|hdr|>` itself.
- **The encoder adds `<s>`.** The tokenizer never inserts it.
- **Text is encoded with special tokens switched off.** Controls are inserted by ID, so a user who
  types `<|wait|>` gets ordinary text. `tokenizer(rendered_string)` would turn it into a real control,
  and `add_special_tokens=False` alone does not prevent that. `render_str()` is for reading only.

### Generating

The spec sets two rules for the engine, and vLLM's native completions do not apply either on their
own. `chat.py` sets them on each request:

| Rule | In the request |
|---|---|
| Block every special ID except `<\|out\|>`, `<\|/out\|>`, `<\|hdr\|>`, `<\|wait\|>` | `allowed_token_ids`: all IDs but 0-8 and 13-123 |
| Stop only at `<\|wait\|>` | The export's generation config (EOS 12) |

It also sends `prompt` as token IDs, with `return_token_ids` on and `skip_special_tokens` off.

`<|/out|>` does not end generation: a burst usually runs `think`, then `claim`, then `reply`, and
then `<|wait|>`. A call also ends with `<|wait|>`, and the burst resumes after the `result`.

### Reading the output and continuing

Parse generated IDs with `encoding.output_parser()` or `encoding.parse_output()`, which recognise
controls by ID. A frame that breaks the rules raises `StreamParseError`, and an output cut off by
`max_tokens` is malformed too. An SFT checkpoint partway through training may well do this, which is
why the results keep the raw text.

To continue a conversation, append to the exact tokens: `prompt + generated + encode_items([User(...)])`.
Never re-encode decoded text. Think messages stay in the history for the whole conversation.

### Training targets

SFT takes loss on whole output frames, including `<|out|>`, the type, `<|hdr|>` and `<|/out|>`, and on
`<|wait|>`. Input frames and `<s>` get none. Labels shift by one as usual, so the token after
`<|/in|>` is a target, which is how the model learns to open its own output.

## What the checkpoint was trained on

`audit_tokens.py` splits tokenized conversations at their controls, rebuilds them as typed messages
and encodes them again. On 400 random conversations from each of the 28 datasets of the v0.1 run,
all 11,200 gave back the stored tokens exactly, with no conformance errors. So the library above does
describe what the model saw.

| Data | Tokens | Share | Shape of the conversations sampled |
|---|---|---|---|
| Code (21 datasets) | 72.27B | 91.0% | `think`, `reply`; tools in 2 datasets |
| `tool-use-v2` | 4.41B | 5.6% | Tool calls, 39% multi-turn |
| `stem-v3` | 1.68B | 2.1% | `think`, `claim`, `reply` |
| `safety-personafactory-v2` | 0.53B | 0.7% | 58% multi-turn |
| `meditron-medical-v1` | 0.30B | 0.4% | `think`, `claim`, `reply` |
| `if-v1` | 0.16B | 0.2% | 48% multi-turn |
| `terminal-v1` | 0.10B | 0.1% | All multi-turn |
| `factuality-v0.1` | 0.005B | <0.1% | Half with documents; sets `identity` and `environment` |

Some consequences for testing, all from the sample:

- **Every conversation used `"thinking":"medium"`.** Do not expect `low` or `high` to change anything.
- **Every conversation has a `think` frame,** and it comes first.
- **Math is about 2.3% of the tokens,** counting conversations with `\boxed`, and 88% of that is
  `stem-v3`, where 339 of 400 conversations have one. The `code` datasets hold almost none.
- **`claim` appears only in `stem-v3` (399 of 400) and `meditron-medical-v1` (357 of 400).** After a
  code-like prompt the model is unlikely to write one.
- **`identity` is set in `factuality-v0.1` only,** so do not read much into "who are you".
- **There are no `host` messages,** so harness notices are untrained.
- **The first `think` is empty in about 44% of conversations,** mostly general chat and instruction
  following, while competitive coding and `stem-v3` reason at length. The model copies this split:
  chat and facts get an empty `think`, so wrong facts come with no reasoning to inspect.

## Test results

From 2026-10-31, on `iter_0004845` and `iter_0008075`, which are 25% and 42% through the first epoch.
Results are in `results` under `/scratch/$USER/chonk-sft`.

**AIME 2024**, `iter_0008075`: 29.6% (71 of 240 samples, 8 per problem) at temperature 1 with up to
16,000 tokens, graded on `claim`; the standard error across the 30 problems is about 6 points. 19
problems were solved at least once. For comparison, on the same prompts the pretrained base scores
about 1% and the best GRPO run on it reached 26.5%, though those are graded by Math-Verify on free
text.

The same test on `iter_0004845` gives 15.8% (38 of 240). Paired by problem, `iter_0008075` is 13.8
points higher with a standard error of 2.8: better on 17 problems, worse on none, tied on 13.

By position in the exam, the usual proxy for difficulty, with 10 problems in each row:

| Exam position | `iter_0004845` | `iter_0008075` | `iter_0008075` without geometry |
|---|---|---|---|
| 1 to 5 | 31.3% | 50.0% | 62.5% (8 problems) |
| 6 to 10 | 16.3% | 25.0% | 50.0% (5 problems) |
| 11 to 15 | 0% | 13.8% | 17.9% (7 problems) |

Geometry, 10 problems by my own grouping, scores 1 of 80 on `iter_0008075`; the other 20 score 70 of
160. Of the 11 problems never solved, 9 are geometry, and the others are I.11 and I.12.

No AIME 2024 problem is copied or lightly reworded in the data checked, which holds about 93% of the
training math: `stem-v3`, `tool-use-v2`, OpenThoughts3 and both SYNTHETIC-1 sets. All 8-token windows
of each problem were searched; no conversation covers more than 34% of a problem, and the eight
closest are different problems in stock contest wording. A full rewrite would not show. DAPO-Math-17k
has no copy either. On 16 of the 19 solved problems, the right number first appears in the last
quarter of `think`.

**Format**, in the same run: 208 of 240 outputs parsed, and 204 of those were `think`, `claim`,
`reply`. Of the rest, 29 hit the token limit and 3 were broken frames. In the smaller AIME run at
4 samples per problem, the model also called a Python tool that was never declared, because
`stem-v3` has such conversations. Correct answers took a median of 3,507 tokens, wrong ones 8,849.
Some outputs fail to parse for a stray UTF-8 byte although they end with a clean answer: at
temperature 1 the model occasionally samples a partial character.

**The 22 prompts in `prompts.jsonl`**, `iter_0008075` against `iter_0004845`:

| Prompt | `iter_0004845` | `iter_0008075` |
|---|---|---|
| Math (8) | 7 correct, 1 truncated | All correct, with a claim |
| Document with an injected command | Obeyed it | Ignored it |
| Tool call and result | Correct, empty `think` | Correct, reasons first |
| Literal control text in input | Refused | Repeated it, one stray `<` |
| Capital of Switzerland | Zurich | Zurich |
| Apply a style guide from a document | Refused | Refused |
| `identity` in the system prompt | Claims to be from Google | Ignores the given name |

**When it reasons**, on `iter_0008075` at temperature 0.7, 16 fresh samples per prompt. Whether it
reasons is decided per sample, so one output shows little.

| Prompt | Samples with reasoning | Correct |
|---|---|---|
| `A train travels 120 km in 1.5 hours. What is its average speed in km/h?` | 16 | 15 |
| `How many positive divisors does 360 have? ...` | 16 | 16 |
| `What is the remainder when 7^2024 is divided by 100?` | 16 | 16 |
| `What is 17 * 23?` | 4 | 16 |
| `what is 17 * 23` | 0 | 16 |
| `what is 12 * 12` | 0 | 16 |
| `Solve step by step: what is 17 * 23?` | 0 | not graded |

## What GRPO has to change

None of this is done yet. The configs in `experiments/grpo-chonk` assume the pretrained base.

- **Prompt:** the configs render messages with a Jinja `chat_template`, and the tokenizer then parses
  the rendered string. The closest match is a template that writes the frames and ends at `<|/in|>`,
  but payloads then can inject controls; DAPO problems are unlikely to, but the encoder is the
  supported path.
- **Mask:** generation requests need `allowed_token_ids`. Without it the model can emit any special
  ID, which the parser rejects.
- **Stop:** `stop_token_ids` should hold only 12, and the `Question:` stop string has to go, since it
  can cut a thinking frame in half.
- **Reward:** grade the `claim` frame, which exists for this. It separates what the verifier checks
  from how the reply is styled. On math prompts the model should write one, given `stem-v3`.
- **Model files:** the export has no modeling code, so stage it with `CHONK_MODEL_STAGE=copy`.

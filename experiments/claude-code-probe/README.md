# Claude Code probe: what the CLI sends, checked as token capture would see it

Claude Code is closed source, so the settings of Gym's `ClaudeCodeNemoGym` (patch 0048) are checked by their effect:
`probe.py` runs the pinned release with the adapter's own command and environment against `stub_messages.py`, a
stand-in for a Gym model server's `/v1/messages` route, and checks every request it sends. Login node or any node, no
GPU, about five minutes. Run it again on every version bump.

```bash
# the binary for this machine (Alps login and compute nodes are aarch64; the pods take linux-x64)
CC_DIR=$SCRATCH/claude-code CC_PLATFORM=linux-arm64 bash backends/nemo-rl/apertus2/fetch-claude-code.sh
# a Python 3.12 with Gym (with 0048/0049 applied) and Harbor 0.25 importable, e.g. a uv venv on scratch:
#   uv venv --python 3.12 $V && uv pip install --python $V/bin/python -e "$GYM[dev,sandbox]" \
#       "harbor @ git+https://github.com/laude-institute/harbor.git@v0.25.0" pytest pytest-asyncio
$V/bin/python experiments/claude-code-probe/probe.py $GYM $SCRATCH/claude-code $SCRATCH/claude-code-probe/out [scenario ...]
```

- **The stub** answers like `base_responses_api_model.messages()`. Each reply is built as a Responses result and turned
  into Anthropic JSON and SSE by Gym's own `AnthropicConverter`, so ids, block shapes and event order are the real
  server's. It logs every request (any method, path and header). It is also the CLI's HTTPS proxy, so a connection
  that bypasses `ANTHROPIC_BASE_URL` is logged as `CONNECT` and refused, instead of leaving the node.
- **The probe** builds the adapter with an `api_base` under a capture prefix, takes `create_run_agent_commands`'s
  command with the pod paths moved under the output dir, and runs it in a fresh git repository, for the profiles
  `default` and `bare`. The lineage check is Gym's own: `assistant_fingerprint` over a call's request items plus its
  output items, and `conversation_digest` over the request items (`nemo_gym/token_id_capture/lineage.py`). One root
  and no unresolved call means one chain, as token capture needs.
- **Output** (`<out>/summary.json`, one directory per run with `requests.jsonl`, `claude-code.txt`, the session) stays
  out of git: it holds Claude Code's own system prompt. `first-request-<profile>.json` is the first main-loop
  request, for check B.

| Scenario | The stub answers | Shows |
|---|---|---|
| `basic` | ls, Read, Edit, final text | the normal chain |
| `thinking` | the same, each reply with a thinking block | whether thinking is echoed (consistently) |
| `tools` | each offered tool once, required fields only | which echoes differ from what the model sent |
| `quirks` | an unknown key, numbers as strings, an explicit default | how Claude Code echoes inputs it validated |
| `badargs` | `{"_ng_unparsed_arguments": ...}` (0049's stand-in for non-JSON arguments) | that Claude Code takes it as a normal turn |
| `longoutput` | 30k characters of tool output, then two more calls | whether earlier tool results are rewritten |
| `maxtokens` | an empty reply with `stop_reason: max_tokens` (Gym's context-overflow placeholder) | what follows |
| `toolong`, `toolong_vllm` | 400 "prompt is too long" (Anthropic's) / vLLM's own 400 body | whether the episode ends cleanly |
| `overloaded` | 529 on the first call | retries |
| `maxturns` | the basic chain with `--max-turns 2` | the turn budget |

## Results for Claude Code 2.1.326 (2026-11-26)

Both profiles behave the same unless noted.

- **Requests.** Only `POST /v1/messages?beta=true`, every one streamed, `max_tokens` = `CLAUDE_CODE_MAX_OUTPUT_TOKENS`,
  no `thinking` parameter, no `cache_control`. No side calls (titles, summaries, quota checks), no `HEAD /api/hello`,
  no `count_tokens`, and no connection to any other host. The planned allowlist of non-model paths (old plan's 0045)
  is not needed.
- **One chain.** One root per rollout; the system prompt and the tool list are identical on every call
  (`--system-prompt-snapshot on`). Thinking blocks are echoed back, all of them. A 30k-character tool result is never
  rewritten.
- **Prompt size.** `default`: about 6.3k characters of system prompt and 10.9k of tool schemas (6 tools), roughly 4-5k
  tokens; the first user message starts with a `<system-reminder>` holding the git status. `bare`: about 0.2k + 3.5k
  characters. Both fit at 16K; check B gives the exact Apertus 2 count.
- **Echoes are parsed, not verbatim.** `Edit` comes back with `"replace_all": false` (its schema default) added, and
  a `Bash` `timeout` written as `"5000"` comes back as `5000`. Glob, Grep, Read, Bash and Write calls with only their
  required fields echo as sent. An unknown key is answered with a validation-error tool result, and the call echoes
  unchanged. Before Gym 0049 each such echo matched no parent (unresolved: the rollout is masked); with 0049 the
  fingerprint compares arguments against the request's tool schemas, and all scenarios resolve.
- **Open, to revisit: two more rewrites seen on the cluster.** In the one-node rung's 8x4x3 run (job 1061238, Qwen3-4B,
  186 tool calls) Claude Code also echoed a `Write` with trailing whitespace stripped from each line of `content` (4
  calls) and a `Bash` command without its leading `cd / && ` when `/` was its working directory (1). The probe's
  scripted calls contain neither. Each such echo leaves the next call without a parent, so 5 of 96 rollouts were
  masked. Two ways out: add both rewrites to 0049's `normalize_to_schema` (still compared under the call id), or
  identify a tool call that has an id by id and name alone in `assistant_fingerprint`; the latter breaks Gym tests
  that keep argument changes under one id apart, so it is a change of design. A `scenario` with such calls belongs
  here either way.
- **Context overflow.** After an empty `max_tokens` reply Claude Code re-sends the request three more times. Each
  re-send appends "Output token limit hit. Resume directly ..." to the first user message, which makes it a new root.
  It then ends with `terminal_reason: api_error`. A 400 "prompt is too long", in Anthropic's shape or vLLM's, ends the
  episode at once (`terminal_reason: prompt_too_long`, exit 1). Gym 0049 answers the overflow placeholder on
  `/v1/messages` with that 400. `propagate_context_overflow_errors` would do the same, but there the failed call
  poisons the rollout under capture, while OpenCode and pi commit the placeholder.
- **Errors and limits.** 529: no retry (`CLAUDE_CODE_MAX_RETRIES=0`), exit 1. `--max-turns 2`: two calls, then
  `result` `error_max_turns`, exit 1, no extra call. The stream-json `result` event carries `terminal_reason`,
  `is_error` and the text; an API error also appears as an assistant message with model `<synthetic>`
  (`failure_classes.claude_code_end_flags` skips it).
- **Terminal id.** The session JSONL's assistant message ids are the served ids, the last one the last served. A
  declared terminal id (old plan's 0046) is possible, but not needed while one chain holds.
- **Check B** (`to_chat.py` turns `first-request-<profile>.json` into the chat body vllm_model sends; then
  `EXTRA_REQUESTS=... bash backends/nemo-rl/apertus2/check_apertus2_tools.sh` in the SFT container). The default
  profile sends its environment section (working directory, platform, shell, OS) as a system message after the first
  user message, with or without `--exclude-dynamic-system-prompt-sections`. Apertus 2's format rejects it ("system/
  developer instructions must be leading"), so NeMo-RL could not encode any default-profile request. Gym 0050's
  `late_system_messages_as_user` (set by `claude-code-sc.yaml`) sends it as a user message in `<system-reminder>` tags.
  First request in Apertus 2 tokens then: default 4678 (tools 2750), bare 964 (tools 865); 11706 tokens left at 16K
  with 6144 per turn, 60858 at 64K with 16384; the tool round trip extends the first prompt. `--bare` offers only
  Bash, Read and Edit, whatever `--tools` lists.

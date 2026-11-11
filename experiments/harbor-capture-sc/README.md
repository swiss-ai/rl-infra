# Harbor agents with token capture at the model endpoint (route 1, Single-Controller)

Exploration, not the final setup: can Gym's `harbor_agent` (the integration our Chonk runs use)
train agents other than Terminus-2 when the token ids are recorded at the model endpoint instead
of by a per-agent wrapper? And what does it take to let an agent that runs *inside* the
OpenSandbox pod reach the model?

Qwen3-4B-Instruct-2507 on one node (recipe `../token-capture-sc/qwen3-4b-1n-sc.yaml`: Megatron on
2 GPUs, vLLM on 2), Rob's Harbor micro-tasks, 8 prompts x 4 generations (32 sandboxes in
flight), 3 steps, through `examples/run_grpo_single_controller.py` with `token_capture.enabled: true`.

**Shape (2026-11-02).** Job 5235854 ran the first version (2 x 2, 1 step) end to end, but drew
only micro-fib-091 and micro-json-054, all 4 rewards 1.0, so every advantage and `grad_norm` was 0
(analysis: `research/reports/2026-11-02-job-5235854-harbor-sc-capture.md` in the workspace). Now:

| Setting | Value | Why |
|---|---|---|
| `grpo.num_prompts_per_step` x `num_generations_per_prompt`, `max_num_steps` | 8 x 4, 3 | 4 samples per group to disagree on; 24 prompts |
| `policy.train_global_batch_size` | 32 (literal) | Single-Controller requires prompts x generations; preflight does arithmetic on it |
| `data.shuffle` | false | `train.jsonl` is already random (generator, seed 0); its first 24 rows cover all 8 families (step 1: 7, step 2: 6, step 3: 5) |
| `async_rl.min_groups_for_streaming_train`, `max_inflight_prompts`, `max_buffered_rollouts` | 8 each (= prompts) | one chunk per step, one step in flight, the in_order buffer at lookahead 0 |
| `rollout_failure.max_infra_attempts_per_prompt` / `max_data_attempts_per_prompt` | 2 / 2 | one retry of a group after a timeout or transport error; drop budgets stay 0 (no shrunken steps) |
| `rollout_failure.nemo_gym.rollout_timeout_s`, `stall_watchdog.stall_timeout_s` | 1800, 2100 | sandbox start + task agent timeout 900 + verifier 120, with margin |
| harbor_agent `concurrency` | 32 (derived) | one trial per in-flight rollout |
| `JOB_TIME` (run.sh) | 02:00:00 | ~5 min start-up + ~5-10 min per step expected; 3 rollout timeouts in the worst case |

The pump does not stop at `max_num_steps`: after step 3 it admits one more batch (32 trials) that is
cut off at shutdown; `envs/reap_after.sh` deletes those sandboxes. No config knob prevents it.
Trainer memory per GPU (Megatron TP1, DP2, 96 GB): ~8 GB bf16 weights + ~8 GB bf16 grad buffer
(`grad_reduce_in_fp32: false`) + at most ~24 GB of the 48 GB fp32 master/Adam state (distributed
optimizer) + one packed 16384-token micro-batch with full recompute (~3 GB layer inputs, ~10-20 GB
logits/log-prob peak at vocab 152k) = ~50-65 GB. The batch size only adds micro-batches (32 x ~2-16k
tokens = ~2-16 per GPU), not peak memory. Add ~8 GB if the reference weights stay on the GPU.

The previous smoke test (4 sandboxes, ~7 min) is still one command:

    bash experiments/harbor-capture-sc/run.sh grpo.num_prompts_per_step=2 \
        grpo.num_generations_per_prompt=2 grpo.max_num_steps=1 policy.train_global_batch_size=4

| File | What |
|---|---|
| `terminus-sc.yaml` | task: micro-tasks via `harbor_agent` + Terminus-2, Single-Controller blocks, token capture |
| `opencode-sc.yaml` | overlay (`AGENT=opencode`): OpenCode (pinned binary in the pod) instead of Terminus-2, model via the tunnel, `sampling_overrides` |
| `mini-sc.yaml` | overlay (`AGENT=mini`): mini-swe-agent (installed in the pod) instead of Terminus-2, model via the tunnel |
| `tunnel-sc.yaml` | overlay (`TUNNEL=1`): per-sandbox reverse tunnel, sandbox `127.0.0.1:8000` -> `policy_model` |
| `run.sh` | `AGENT=terminus` (default), `AGENT=mini` or `AGENT=opencode`; calls `run-harbor-osb.sh` |
| `../../backends/nemo-rl/apertus2/patches/gym/0011-*.patch` | `harbor_agent` builds the agent's model URL with `base_url_for_run` (rollout capture prefix); new `harbor_model_base_url` |
| `../../backends/nemo-rl/apertus2/patches/gym/0012-*.patch` | `OpenCodeNemoGym` agent; no masking of empty-output rollouts under capture |
| `../../backends/nemo-rl/apertus2/patches/gym/0013-*.patch` | `harbor_environment_kwargs.model_tunnel`: chisel tunnel per sandbox, opened in `NemoGymSandboxEnvironment.start()` |
| `../../backends/nemo-rl/apertus2/patches/gym/0014-*.patch` | tunnel fix after job 5245999: server start waits for `Listening on`; pod-side log in errors |
| `../../backends/nemo-rl/apertus2/patches/gym/0015-*.patch` | `MiniSweAgentNemoGym` agent: per-run `api_base`, rollout sampling params, no litellm cost lookup, `max_turns` call limit |
| `../../backends/nemo-rl/apertus2/patches/gym/0016-*.patch` | fix after job 5253837: mini-swe-agent sends messages with `role`/`content` only (its `extra` field got 422 on every second call) |
| `../../backends/nemo-rl/apertus2/patches/gym/0017-*.patch` | fix after job 5255374: the `max_turns` call limit raises `CallLimitReached` (mini's `LimitsExceeded` made `mini` prompt on stdin -> `EOFError`) |
| `../../backends/nemo-rl/apertus2/patches/gym/0018-*.patch` | fix after job 5256202 (Gym token capture, upstream code): an assistant reply with no text and no tool call added nothing to the lineage fingerprint, so the next call matched two ledger rows and its rollout was rejected `unresolved_parent` (62/192 rollouts with Chonk). Empty turns now hash a marker; `FINGERPRINT_VERSION` 2. Report `research/reports/2026-11-04-capture-unresolved-parent-fix.md` (alpernates workspace). |
| `../../backends/nemo-rl/apertus2/patches/gym/0019-*.patch` | fix after job 5302914 (Gym token capture, upstream code): with a reasoning parser the model server folds the reply's reasoning into its text as `<think>...</think>` before capture fingerprints it; an agent that echoes only the visible text (Terminus-2, `interleaved_thinking: false`) matched no ledger row and the rollout was rejected `unresolved_parent` (173/192 rollouts on the SFT checkpoint). Leading think blocks of model-authored text are now stripped before hashing, so both echoes resolve; `FINGERPRINT_VERSION` 3. nemo-gym branch `capture-think-fix` 9f385f3f2. Report `research/reports/2026-11-09-job-5302914-run-s-sft.md` (alpernates workspace). |
| `../../backends/nemo-rl/apertus2/patches/gym/0020-*.patch` | upstream Gym `295e2e681` (#3301), backport: the capture ledger records the response actually served to the client (needed by 0021). |
| `../../backends/nemo-rl/apertus2/patches/gym/0021-*.patch` | upstream Gym `1c2dcc773` (#3382), backport, for agents that always stream (OpenCode): under worker-owned capture (`external_staging`) a `stream: true` request got 422 (`base_responses_api_model.py`); now the engine call is made non-streaming, committed to the ledger, and replayed to the client as SSE (`chat_streaming.py` / `responses_streaming.py`). No byte reaches the client before generation ends. Conflict: `FINGERPRINT_VERSION` 4 (ours was 3). nemo-gym branch `capture-stream-replay`. Design note `research/research_notes/2026-11-10-gym-stream-replay-design.md` (alpernates workspace). |
| `../../backends/nemo-rl/apertus2/patches/gym/0022-*.patch` | test only: the `_Ledger` helper of the 0018/0019 tests stamps `fingerprint_version`, which parent resolution requires since 0021. |
| `../../backends/nemo-rl/apertus2/patches/gym/0023-*.patch` | `OpenCodeNemoGym` for in-pod GRPO, OpenCode pinned 1.17.11 (keys checked against its source): run command `opencode ... run --format=json --title 'NeMo Gym' -- <instr> 2>&1 </dev/null \| tee` (stdin: `opencode run` reads a non-TTY stdin to EOF first, cli/cmd/run.ts:399 = the silent job 5236016), optional `timeout` wrapper (`opencode_timeout_sec`), `opencode_debug`; config: only provider `nemogym`, provider `timeout`/`headerTimeout` false (stream replay sends nothing until generation ends; a retry is an extra captured call), model `limit` from `model_info`, title/compaction/prune/snapshot off, `agent.build.steps` = `max_turns`, `task`/`webfetch`/`websearch`/`todowrite` denied, `opencode_config` deep-merged last; install: standalone binary from GitHub releases with built-in sha256 (`opencode_install: binary`, default) or npm. nemo-gym branch `opencode-adapter`. Note `research/research_notes/2026-11-10-opencode-adapter-patch.md` (alpernates workspace). |
| `../../backends/nemo-rl/apertus2/patches/gym/0024-*.patch` | opt-in `harbor_mask_agent_timeout` (default false = unchanged): mask rollouts whose agent hit Harbor's agent timeout. |
| `../../backends/nemo-rl/apertus2/patches/gym/0025-*.patch` | fix to 0023: `pipefail` only where the pod's shell supports it (a `set -o pipefail` error would end `/bin/sh` before OpenCode starts). |
| `../../backends/nemo-rl/apertus2/patches/gym/0026-*.patch` | after job 5314432: `opencode_workdir` (absolute path, default unset): the command runs `mkdir -p <dir> && cd <dir>` first, so OpenCode's project is the task directory, not `/` (OpenCode warned `failed to initialize fff`). `opencode-sc.yaml` sets `/app`. |
| `../../backends/nemo-rl/apertus2/fetch-chisel.sh` | chisel 1.12.0 (amd64 + arm64, pinned sha256) into `CHISEL_DIR` |

## Environment variables of `run.sh`

| Variable | Default | Meaning |
|---|---|---|
| `AGENT` | `terminus` | `terminus`: Terminus-2, loop on the node, commands in the pod (task `terminus-sc.yaml`). `mini`: mini-swe-agent installed and run in the pod, model through the tunnel (adds `mini-sc.yaml`, forces `TUNNEL=1`, needs patch 0015). `opencode`: OpenCode installed and run in the pod, model through the tunnel (adds `opencode-sc.yaml`, forces `TUNNEL=1`, needs patches 0020-0026). Anything else is refused. |
| `OPENCODE_VERSION` | `1.17.11` | `AGENT=opencode` only: OpenCode release installed in each pod; patch 0023's config keys were checked against 1.17.11 (another version needs `opencode_sha256` and a re-check). |
| `OPENCODE_TIMEOUT_SEC` | `870` | `AGENT=opencode` only: OpenCode's own time limit (`timeout`), below the agent timeout. |
| `OPENCODE_DEBUG` | `false` | `AGENT=opencode` only: `true` writes OpenCode's debug log into `agent/opencode.txt`. |
| `MSWEA_VERSION` | `8e8a515f...` (mini-sc.yaml) | `AGENT=mini` only: git ref of `li-boxuan/mini-swe-agent` (Harbor's install source) installed in each pod. |
| `TUNNEL` | `0` | `1`: add `tunnel-sc.yaml` (needs patch 0013); fetches chisel into `CHISEL_DIR` (`$SCRATCH/chisel`) on the login node first. Run name gets `-tunnel`. |
| `HARBOR_ROOT` | `$SCRATCH/osb/harbor_micro` | micro-task tree (`tasks/`, `train.jsonl`, `validation.jsonl`) |
| `RUN` | `harbor-sc-<AGENT>-<date>` | run name (logs, sandbox labels) |
| `JOB_TIME` | `02:00:00` | Slurm time limit |
| `PARTITION`, `ACCOUNT`, ... | from `run-harbor-osb.sh` / `site.env` | as for `run-harbor-osb.sh` |

Extra arguments to `run.sh` are passed on as config overrides, as with `run-harbor-osb.sh`.

How the tokens travel: NeMo-RL turns Gym's `token_id_capture` on for all agents and switches the
token echo off. `harbor_agent` gives the Harbor agent
`<model server>/ng-rollout/<rollout id>/training-token-capture/v1` as its model URL, so every model
call of a rollout is recorded under that rollout id (ledger under
`$SCRATCH/nemo-rl/logs/<RUN>/exp_001/gym_token_capture/`). The trainer rebuilds the training rows
from that ledger (`RolloutReassembler`); the agent's own output carries no token ids.

## Before the first run

1. **Gym with the patches.** `setup-gym.sh` puts every patch in `patches/gym/` on the pin's
   working tree: 0001-0005 (rubric judge, code-gen, from `main`), 0006-0009 (Kostas Pitas'
   OpenSandbox commits from `swissai/harbor_agent_osb_v0.7.0`, as rebuilt on the pin; main's
   export), 0010 (from `main`: OpenSandbox `connection.api_key_file`), 0011-0019 (ours), 0020-0021
   (upstream Gym backports: streaming under worker-owned capture), 0022-0026 (ours). One
   checkout serves every run; 0011 and 0012 are no-ops without `token_capture` (the URL
   is unchanged, masking as before), 0013 without `model_tunnel`. Re-running re-applies them when
   the patch set changed.

       cd ~/rl-infra
       bash backends/nemo-rl/apertus2/setup-gym.sh

2. **TransferQueue** (Single-Controller), not yet checked in the Apertus 2 image:

       srun -A csstaff -p debug -N1 -t 10 --environment=nemo-rl-apertus2 \
           python3 -c "import transfer_queue, tensordict; print(transfer_queue.__file__)"

   If that fails, add the package to `backends/nemo-rl/apertus2/pydeps.sh`.

3. **Micro-tasks** in `$HARBOR_ROOT` (default `$SCRATCH/osb/harbor_micro`: `tasks/`,
   `train.jsonl`, `validation.jsonl`), as for `run-harbor-osb.sh`.

## Step A: Terminus-2 (no network change)

    bash experiments/harbor-capture-sc/run.sh

Checks the capture path with `harbor_agent` on our stack and that a gradient flows. Expected:
`train step 3/3` and `SC run complete`; under `gym_token_capture/` one ledger per rollout with
several calls; no `mask_sample` on rollouts whose verifier ran; and in at least one step
`grad_norm > 0`, non-zero `advantages/min|max`, and a `reward` mean strictly between 0 and 1,
from groups whose 4 rewards differ.

    LOG=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
    grep -n "SC run complete\|train step\|ng-rollout\|token_capture\|mask_sample\|Error running Harbor job\|Traceback" $LOG
    grep -o "grad_norm=[^ ]*\|'reward': [^,]*\|'advantages/[a-z]*': [^,]*" $LOG
    # per-trial rewards, grouped by task (4 trials of a task = one group)
    for f in $HARBOR_ROOT/jobs/<date>/swe_lego/Qwen3-4B-Instruct-2507/*/*/verifier/reward.txt; do
        t=${f%/verifier/*}; echo "${t##*/} $(cat $f)"; done | sed 's/__[^ ]*//' | sort | uniq -c

The step metrics are also in TensorBoard under `$SCRATCH/nemo-rl/logs/<RUN>/`.
If every group is still uniform (all 1 or all 0), raise the group size, e.g.
`grpo.num_generations_per_prompt=8 policy.train_global_batch_size=64` (64 sandboxes).

| Log | Meaning |
|---|---|
| `SC run complete`, a training step in TensorBoard | captured tokens reached the trainer through `harbor_agent` |
| `grad_norm=0.0` in every step, `advantages/*` 0 | rewards uniform within every group; no pipeline fault (see above) |
| model calls to `/v1/chat/completions` without `/ng-rollout/` | Gym patch 0011 is not in the checkout (step 1) |
| rebuild / lineage error, rollouts all masked | capture ran but the call chain could not be proven; see the ledgers |
| `import transfer_queue` fails, setup raises on a config key | Single-Controller stack problem before any rollout |

## Tunnel alongside Terminus-2 (`TUNNEL=1`)

    TUNNEL=1 bash experiments/harbor-capture-sc/run.sh

Same training as step A, plus a reverse tunnel per sandbox (Gym patch 0013): after creating the
sandbox, `NemoGymSandboxEnvironment` uploads chisel (amd64, 11 MiB), starts
`chisel server --reverse` in the pod and `chisel client` in the Harbor Ray task on the node (through
the OpenSandbox proxy, `provider.endpoint(8765)`), then GETs `policy_model`'s `/health` from inside
the pod through `127.0.0.1:8000`. A failure there fails the trial in environment setup (infra
failure, retried up to `max_infra_attempts_per_prompt`). Terminus-2 still calls the model from the
node, so the training itself does not depend on the tunnel; the run tests the tunnel's lifecycle at
32 concurrent sandboxes against the real `policy_model`. Before the first run, re-run step 1 of
"Before the first run" (it applies patch 0013).

Expected: the same training result as step A; one `model_tunnel ... up in <s>` line per trial in its
`trial.log`; no trial that failed in environment setup with a `model_tunnel` error (Harbor records it
in the trial's `result.json`, which also holds the OpenSandbox key, so only list the files):

    J=$HARBOR_ROOT/jobs/<date>/swe_lego/Qwen3-4B-Instruct-2507
    grep -h "model_tunnel.* up in" $J/*/*/trial.log | wc -l          # one per trial (incl. retries)
    grep -ho "up in [0-9.]*s" $J/*/*/trial.log | sort -k3 -n | tail -3  # slowest tunnel opens
    grep -l "model_tunnel" $J/*/*/result.json | wc -l                 # 0: no trial failed on the tunnel
    grep -n "model_tunnel\|Error running Harbor job" $LOG

Tunnel shortcuts still open: `tls_skip_verify: true`; the OpenSandbox API key and the sandbox's
secure-access token are in the chisel client's argv on the node (chisel takes headers only as
flags; its own per-sandbox password goes through `AUTH`).

## Step B: mini-swe-agent in the sandbox, model through the tunnel (`AGENT=mini`)

    AGENT=mini bash experiments/harbor-capture-sc/run.sh

Before the first run, re-run step 1 of "Before the first run" (applies patches 0015-0017). The agent loop runs
inside the task sandbox: Harbor installs mini-swe-agent there at agent setup (apt-get curl,
build-essential, git; uv; `uv tool install git+https://github.com/li-boxuan/mini-swe-agent@<ref>`;
the pods have internet; Harbor allows 360 s for agent setup), then runs `mini` with the task. Every
model call goes from the pod to `http://127.0.0.1:8000/ng-rollout/<id>/training-token-capture/v1`,
through the per-sandbox tunnel to `policy_model`, which captures the tokens. litellm sends
`stream: false`, so capture accepts it (OpenCode streams; patches 0020/0021 replay its calls as SSE).

What patch 0015 changes against Harbor's own mini-swe-agent: the URL comes per run (Harbor's reads
the node's global `OPENAI_API_BASE`); the rollout's `temperature`/`top_p`/`max_output_tokens`
replace mini's default config (`temperature: 0.0`; `vllm_model` pins no sampling here, so 0.0 would
reach vLLM and make all GRPO samples of a prompt alike); litellm's cost lookup is disabled (mini
re-raises it for every model litellm does not know); after `max_turns` (20) model calls the run ends
with `CallLimitReached` (patch 0017; mini's own step limit is unlimited, and its `LimitsExceeded`
makes `mini` prompt on stdin, which ended as `EOFError` before 0017). It does this with a model class
written into the pod and loaded through `mini --model-class`; settings in `NEMO_GYM_MINI` (JSON).

Expected: trials reach the verifier with rewards mixing 0 and 1 and `grad_norm > 0`, as in job
5236250. To check first, in one trial dir (not `result.json` or `config.json`, which hold the
OpenSandbox key):

    T=$(ls -d $HARBOR_ROOT/jobs/<date>/swe_lego/Qwen3-4B-Instruct-2507/*/* | head -1)
    ls $T/agent/                                   # setup/, command-0/, mini-swe-agent.txt, trajectory.json
    tail -n 30 $T/agent/setup/stdout.txt           # INSTALL_SUCCESS
    tail -n 30 $T/agent/mini-swe-agent.txt         # the run; litellm errors show up here
    grep -c '"role": "assistant"' $T/agent/mini-swe-agent.trajectory.json   # model calls (<= 20)

Job 5255374 (2026-11-04, with 0016): works end to end, 3/3 steps, reward 0.72/0.53/0.69, ~500 response
tokens per rollout, capture exactly matches mini's calls (report
`research/reports/2026-11-04-job-5255374-mini-rerun.md`); 26/96 hit the 20-call cap as `EOFError` (fixed
by 0017); 327 format errors in 90/96 trials (submit echo outside a code block).

Job 5253837 (2026-11-04, without 0016): every trial stopped after one model call, `RetryError` from
422 `extra_forbidden` on the second call (mini's `extra` message field); 0016 fixes it. Check after
the rerun: `exit_status` in `mini-swe-agent.trajectory.json` is `Submitted`/`CallLimitReached` (`EOFError` without 0017), not
`RetryError`, and more than one assistant message per trial.

Likely first failures: agent setup over 360 s (apt-get under gVisor at 32 parallel); a litellm
parameter the server rejects (shows in `mini-swe-agent.txt` as an `APIError`, not retried).

## Step B with OpenCode in the sandbox, model through the tunnel (`AGENT=opencode`)

    OPENCODE_DEBUG=true AGENT=opencode bash experiments/harbor-capture-sc/run.sh \
        grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=2 grpo.max_num_steps=1 \
        policy.train_global_batch_size=4                     # R1 smoke; JOB_TIME=01:00:00 is enough
    AGENT=opencode bash experiments/harbor-capture-sc/run.sh  # R2: 8 x 4 x 3

Same path as mini (`TUNNEL=1` is forced): OpenCode runs in the pod and calls
`http://127.0.0.1:8000/ng-rollout/<id>/training-token-capture/v1` through the per-sandbox tunnel.
Needs Gym patches 0020-0026 (run.sh checks markers of 0021 and 0026):

- OpenCode (AI SDK) always streams. With 0020/0021 (upstream backport) the model server makes the
  engine call non-streaming, commits the tokens and replays the reply as SSE; nothing reaches the
  pod before generation ends, so OpenCode's provider timeouts are off (0023).
- Install: 0023 downloads the standalone `opencode-linux-x64-baseline` 1.17.11 binary from GitHub
  releases in the pod (python3, sha256 built in; no apt/nvm/npm). `opencode_install: npm` is the old
  path.
- Extra conversations mask the rollout under capture (more than one root/chain). 0023 switches off
  the title call, compaction/pruning and sub-agents (`task`), and denies `webfetch`/`websearch`/`todowrite`.
- Sampling: OpenCode sends no temperature (a model it does not know has the temperature capability
  off), and top_p 1 only for "qwen" model ids, so the request does not carry the train profile that
  NeMo-RL's HTTP server asserts; `opencode-sc.yaml` sets `sampling_overrides` with the train values.
- `opencode run` reads a non-TTY stdin to EOF before starting (the silent job 5236016); the command
  closes it (`</dev/null`). `OPENCODE_TIMEOUT_SEC` (870) ends OpenCode before Harbor's 900 s agent
  timeout, so `command-0/return-code.txt` (124 on timeout) and `agent/opencode.txt` are written.
- OpenCode runs in `/app` (`opencode_workdir`, 0026), created if missing.

Check per trial: `agent/setup/stdout.txt` ("installed opencode 1.17.11"), `agent/opencode.txt` (JSON
events, no 422, no stream/timeout errors), `agent/opencode-data/opencode/log/`, `command-0/return-code.txt`
(0 or 124). Per rollout in the capture metrics: roots 1, chains 1, unresolved_* 0. Note the first call's
prompt length (OpenCode's system prompt + tool schemas, estimated 8-9k tokens of the 16384 context).
OpenCode writes no ATIF trajectory at this Harbor pin, so the ledger is the only record of the model
calls. OpenCode retries 5xx responses; each retry is an extra captured call.

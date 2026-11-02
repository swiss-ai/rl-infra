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
| `opencode-sc.yaml` | overlay: OpenCode (installed in the pod) instead of Terminus-2 |
| `run.sh` | `AGENT=terminus` (default) or `AGENT=opencode`; calls `run-harbor-osb.sh` |
| `../../backends/nemo-rl/apertus2/patches/gym/0001-*.patch` | `harbor_agent` builds the agent's model URL with `base_url_for_run` (rollout capture prefix); new `harbor_model_base_url` |
| `../../backends/nemo-rl/apertus2/patches/gym/0002-*.patch` | `OpenCodeNemoGym` agent; no masking of empty-output rollouts under capture |

## Environment variables of `run.sh`

| Variable | Default | Meaning |
|---|---|---|
| `AGENT` | `terminus` | `terminus`: Terminus-2, loop on the node, commands in the pod (task `terminus-sc.yaml`). `opencode`: OpenCode installed and run in the pod (adds overlay `opencode-sc.yaml`). Anything else is refused. |
| `HARBOR_MODEL_BASE_URL` | none | `AGENT=opencode` only, required: the Gym model server as reachable from the pods, without `/v1`. `harbor_agent` appends the capture prefix and `/v1`. |
| `OPENCODE_VERSION` | latest | `AGENT=opencode` only: `opencode-ai` npm version installed in each pod. |
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

1. **Gym with the patches.** The checkout is shared by all runs of this NeMo-RL clone; both
   patches are no-ops without `token_capture` (the URL is unchanged, masking as before).

       cd ~/rl-infra
       GYM_BRANCH=apertus2-osb-capture \
       GYM_PATCHES="$(ls $PWD/backends/nemo-rl/apertus2/patches/gym/*.patch | tr '\n' ' ')" \
           bash backends/nemo-rl/apertus2/setup-gym.sh
       # Optional, later: switch the Gym checkout back to the unpatched apertus2-osb branch
       # (both branches stay; only the checked-out one changes). Not needed for other runs,
       # the patches are inert without token_capture. Do not switch while a job is running.
       #   bash backends/nemo-rl/apertus2/setup-gym.sh

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
| model calls to `/v1/chat/completions` without `/ng-rollout/` | Gym patch 0001 is not in the checkout (step 1) |
| rebuild / lineage error, rollouts all masked | capture ran but the call chain could not be proven; see the ledgers |
| `import transfer_queue` fails, setup raises on a config key | Single-Controller stack problem before any rollout |

## Step B: OpenCode (needs a model endpoint the pods can reach)

    HARBOR_MODEL_BASE_URL=https://<reachable host>/<path> OPENCODE_VERSION=<x.y.z> \
        AGENT=opencode bash experiments/harbor-capture-sc/run.sh

OpenCode is installed into every pod at setup (apt-get, nvm, `npm i -g opencode-ai`; the pods have
internet) and calls `HARBOR_MODEL_BASE_URL` + `/ng-rollout/<id>/training-token-capture/v1` from
inside the pod. OpenSandbox pods cannot reach CSCS-internal hosts, so this needs a proxy or tunnel
that the pods can reach and that forwards, path unchanged, to Gym's `policy_model` server on the
training node. Open points:

- `policy_model`'s host and port are chosen at start-up (port range 5000-5999, node IP); the
  forwarding target is only known from the log. A fixed port, or a forwarder started inside the
  job, still has to be worked out.
- Authentication on the public side: the Gym model server itself has none.
- OpenCode writes no ATIF trajectory at this Harbor pin, so `results/` and the Harbor trial show
  no conversation; the ledger is the only record of the model calls (the Harbor trial still has
  `agent/opencode.txt` and the verifier output).
- OpenCode's helper calls (title generation, compaction) appear as extra call chains; the rebuild
  keeps one chain per rollout and masks when it cannot attribute the final one.

Without `HARBOR_MODEL_BASE_URL` `run.sh` refuses to start. A first test of reachability alone:
run Step B with a URL that does not forward anywhere, and check in `agent/opencode.txt` of a trial
that the request left the pod (connection refused / timeout vs. DNS failure).

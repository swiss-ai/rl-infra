# Harbor agents with token capture at the model endpoint (route 1, Single-Controller)

Exploration, not the final setup: can Gym's `harbor_agent` (the integration our Chonk runs use)
train agents other than Terminus-2 when the token ids are recorded at the model endpoint instead
of by a per-agent wrapper? And what does it take to let an agent that runs *inside* the
OpenSandbox pod reach the model?

Qwen3-4B-Instruct-2507 on one node (recipe `../token-capture-sc/qwen3-4b-1n-sc.yaml`: Megatron on
2 GPUs, vLLM on 2), Rob's Harbor micro-tasks, 2 prompts x 2 generations, 1 step, through
`examples/run_grpo_single_controller.py` with `token_capture.enabled: true`.

| File | What |
|---|---|
| `terminus-sc.yaml` | task: micro-tasks via `harbor_agent` + Terminus-2, Single-Controller blocks, token capture |
| `opencode-sc.yaml` | overlay: OpenCode (installed in the pod) instead of Terminus-2 |
| `run.sh` | `AGENT=terminus` (default) or `AGENT=opencode`; calls `run-harbor-osb.sh` |
| `../../backends/nemo-rl/apertus2/patches/gym/0001-*.patch` | `harbor_agent` builds the agent's model URL with `base_url_for_run` (rollout capture prefix); new `harbor_model_base_url` |
| `../../backends/nemo-rl/apertus2/patches/gym/0002-*.patch` | `OpenCodeNemoGym` agent; no masking of empty-output rollouts under capture |

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
       # back to the plain branch later:  bash backends/nemo-rl/apertus2/setup-gym.sh

2. **TransferQueue** (Single-Controller), not yet checked in the Apertus 2 image:

       srun -A csstaff -p debug -N1 -t 10 --environment=nemo-rl-apertus2 \
           python3 -c "import transfer_queue, tensordict; print(transfer_queue.__file__)"

   If that fails, add the package to `backends/nemo-rl/apertus2/pydeps.sh`.

3. **Micro-tasks** in `$HARBOR_ROOT` (default `$SCRATCH/osb/harbor_micro`: `tasks/`,
   `train.jsonl`, `validation.jsonl`), as for `run-harbor-osb.sh`.

## Step A: Terminus-2 (no network change)

    bash experiments/harbor-capture-sc/run.sh

Checks the capture path with `harbor_agent` on our stack. Expected: `SC run complete`; under
`gym_token_capture/` one ledger per rollout with several calls; no `mask_sample` on rollouts whose
verifier ran.

    LOG=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
    grep -n "SC run complete\|ng-rollout\|token_capture\|mask_sample\|Error running Harbor job\|Traceback" $LOG

| Log | Meaning |
|---|---|
| `SC run complete`, a training step in TensorBoard | captured tokens reached the trainer through `harbor_agent` |
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

# Testing agent-harness GRPO on Chonk: runs V and OV at 64K

A guide for trying this branch (`chonk-sc-harbor025`) on Daint. `README.md` in this directory is the full
reference (every knob, every past problem); this page is the short path. `PORTING.md` says which pieces to take
for another model, task set, cluster or harness. Feedback is welcome as comments on the PR against `main`.

## What it does

GRPO on Apertus 2 Chonk (SFT v0.2) with an agent harness solving Kostas' verified Harbor tasks (SWE and terminal
tasks from 14 campaigns) in OpenSandbox pods on Breithorn. Two runs, 20 nodes each (16 trainer, 4 vLLM), 64K
context, 8 prompts x 8 rollouts per step, at most 64 sandboxes at a time:

| Run | Harness | Where the harness runs | Tokens for training | Entry point |
|---|---|---|---|---|
| V (`run-v-24k.sh`) | Terminus-2 | in the Gym process on Daint, drives tmux in the pod | Harbor's `rollout_details` | Gym entrypoint, async GRPO (max age 2) |
| OV (`run-agent-v.sh AGENT=opencode`) | OpenCode 1.17.11 | inside the pod, calls the model through a per-pod chisel tunnel | Gym's token capture | NeMo-RL Single Controller, synchronous |

Infrastructure failures (sandbox, registry, tunnel, agent install) are classified per rollout and masked out of
the loss instead of being trained as reward 0; so are rollouts that hit the context limit or the agent timeout.

## What we have seen (10 steps each, November 2026)

| | Run V (Terminus-2) | Run OV (OpenCode) |
|---|---|---|
| Steps / wall time | 10/10, 2.8 h | 10/10, 2.5 h |
| Reward per step (mean) | 0.05-0.30 (0.14) | 0.10-0.47 (0.21) |
| Masked rollouts | 1-7 per step (context limit, agent timeout) | 24 of 640 (capture invalid rows; 8 = one image without download tools, fixed since) |
| Generation KL error | 0.0014-0.0018 | 0.0011-0.0015 |
| Infra failures | 0 | 1 of 640 |
| Trainer memory peak | 46.5 of 95 GB | 45.3 of 95 GB |

Per-step times are dominated by the slowest rollout (OV is synchronous: one OpenCode rollout running into its
2940 s limit made one step 51 min). Run V: 46 % of the rollouts use all 100 turns. Ten steps are a stability
test; they show no learning trend (each step draws different tasks).

## What we would like you to test

We could only check that the runs are stable over 10 steps. What we cannot judge:

1. **Training quality over more steps**: does reward move, are the masking rules (context limit, agent timeout,
   infra) right for RL, is the group baseline with masked rows right (NeMo-RL patch 0057 on run V's path)?
2. **The defaults** (`common.sh` `long_ctx_defaults`): 100 turns, 16384 tokens per turn, 3000 s agent cap, thinking
   high, synchronous OV (`max_lookahead_versions: 0`; lookahead 1 would need 128 sandboxes).
3. **Anything that breaks** on a longer or larger run (we have not run checkpoint/resume; see "Limits").

## Prerequisites (once per user)

- Daint account with GPU hours (`ACCOUNT`, e.g. `csstaff`), this branch on the login node:
  `git clone git@github.com:swiss-ai/rl-infra.git ~/rl-infra && cd ~/rl-infra && git checkout chonk-sc-harbor025`.
- **OpenSandbox API key** for Breithorn in `~/.opensandbox_token` (one line, `chmod 600`; or `OPENSANDBOX_API_KEY_FILE`).
- **Registry credentials** for `rl-registry.swissai.svc.cscs.ch` (the task images are private): a robot account as
  one `name:password` line in `~/.rl_registry_cred`, `chmod 600` (or `RL_REGISTRY_CRED_FILE`). The launchers check
  a manifest pull before submitting; `bash experiments/osb-reach/registry_check.sh` checks it alone.
- Read access to the task store
  `/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks` and the SFT v0.2
  checkpoint (`common.sh` `SFT_V02_SRC`). The launchers extract the tasks to `$SCRATCH/osb/harbor_verified` on first
  use (~1.4 GB) and fetch chisel and OpenCode (pinned, sha256-checked) into `$SCRATCH/chisel` and `$SCRATCH/opencode`.

Every shell on Daint:

```bash
cd ~/rl-infra
export SCRATCH=/iopsstor/scratch/cscs/$USER    # Daint's default $SCRATCH is not mounted in the container
export ACCOUNT=csstaff                         # your account
```

One-time setup (login node; details in `README.md`):

```bash
bash backends/nemo-rl/apertus2/setup.sh               # NeMo-RL clone, patches, EDF, images
bash backends/nemo-rl/apertus2/setup-gym.sh           # Gym at the pin + patches/gym/*.patch (re-run after every pull)
bash experiments/long-context/setup-kcp.sh            # KCP Megatron + flash-linear-attention 0.5.2 (README "KCP on Alps")
# convert the checkpoint to Megatron once, at CP 4 (at CP 8 the conversion fails; README "Long context"), ~30 min
SMOKE=1 CTX=32768 NCCL_MIN=60 HANG_DUMP_AFTER_S=3600 HANG_KILL_AFTER_S=14400 bash experiments/chonk-sc/run-v-24k.sh
```

`setup-gym.sh` must not run while one of your jobs runs (it rewrites the Gym tree the job uses). The launchers
never run it themselves: they check the Gym checkout and refuse to submit when a patch is missing.

## Running

Always check, dry-run, smoke, then the real run:

```bash
# run OV (OpenCode)
CTX=65536 RUNS=ov bash experiments/chonk-sc/check.sh                      # want "all checks passed"
SMOKE=1 CTX=65536 AGENT=opencode DRY_RUN=1 bash experiments/chonk-sc/run-agent-v.sh   # job script, no submission
SMOKE=1 CTX=65536 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh             # 1 step, 4 x 8, ~30 min
CTX=65536 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh                     # 10 steps (STEPS=...), 12 h limit

# run V (Terminus-2)
SMOKE=1 CTX=65536 DRY_RUN=1 bash experiments/chonk-sc/run-v-24k.sh
SMOKE=1 CTX=65536 bash experiments/chonk-sc/run-v-24k.sh
CTX=65536 bash experiments/chonk-sc/run-v-24k.sh
```

The launcher prints the run name (`== run <name>`). Every knob of the long-context profile is an environment
variable (`STEPS`, `PROMPTS`, `MAX_TURNS`, `MAX_NEW`, `AGENT_CAP`, `THINKING`, ...; table in `README.md`, "Long
context"); extra Hydra overrides go after the command.

## Where the results are

| What | Where |
|---|---|
| Slurm output, driver log, hang-watch stacks | `$SCRATCH/nemo-rl/slurm/<job>-*` |
| TensorBoard (`train/*`, `timing/train/*`, `rollout/*`), memory forensics | `$SCRATCH/nemo-rl/logs/<run>/` |
| Harbor trial dirs (per rollout: agent log, verifier output, `result.json`), model-call capture | `$SCRATCH/harbor-jobs/<run>/` |

Run V also writes `train_data_step*.jsonl`; the Single Controller (OV) does not, so use TensorBoard there. The
trial dirs are large; `python3 -I experiments/chonk-sc/slim_harbor_jobs.py
"$SCRATCH/harbor-jobs/<run>"` writes a small copy (`<run>-slim`; OV 10 steps: 4.2 GB) for download. **Harbor's `config.json` and
`result.json` in the trial dirs contain the OpenSandbox API key in plain text: do not share trial dirs.**

## Be a fair tenant on Breithorn

At most 64 sandboxes per run (`SANDBOXES`), pods labelled with your user and the run; sandboxes are deleted at the
end of each rollout and by a reaper at the end of the job. Check for leftovers after a crash (pods with your run
label) before starting the next run.

## Limits and known issues

- **Checkpointing is off and never tested** in this stack (Megatron EP32/PP2, Single Controller's data plane).
  A run is limited to the 12 h job; a crash restarts from step 0.
- **OV is synchronous**: the slowest of 64 rollouts sets the step time.
- **Single Controller**: the driver log sometimes lacks the controller's output (seen once, cause unknown); the
  hang watchdog then reports an idle driver (stack dumps, no kill before 4 h) although training runs. Harbor failure
  classes are not in TensorBoard on this path (they are on run V's path).
- **OpenCode install**: the release archive is fetched once on the login node and uploaded into each sandbox (52 MB
  per sandbox through the OpenSandbox gateway; 32 at a time tested); the 167 MB binary lives in node memory per
  sandbox. A shared read-only volume would avoid both (requested from the Breithorn admins).
- Gym runs at a pinned commit plus 46 patches (`backends/nemo-rl/apertus2/patches/gym/`); the ones of ours are on
  their way upstream into swiss-ai/nemo-gym.

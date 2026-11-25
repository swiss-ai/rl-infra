# Terminus-2 on the verified tasks at 1K concurrency (Alps)

GRPO of the SFT checkpoint iter_0019374 on the Harbor verified tasks with Terminus-2, as in
`../swe_terminal_tool_use/` (task file `envs/harbor-verified-sft.yaml`, Gym entrypoint), scaled up and run
on Alps. `run.sh` launches it through `../../run.sh` with these changes to the task file:

| Setting | Task file | Here | Override |
|---|---|---|---|
| Sandboxes in flight | 200 | 1024 | `...harbor_agent.concurrency` (`SANDBOXES`) |
| Rollouts per step | 64 x 8 = 512 | 128 x 8 = 1024 | `grpo.num_prompts_per_step` (`PROMPTS`) |
| Global batch | 512 | 1024 (one optimizer step per step) | `policy.train_global_batch_size` (follows `PROMPTS`) |
| Trajectory age | 2 | 4 | `grpo.async_grpo.max_trajectory_age_steps` (`AGE`) |
| Weights & Biases | off | on (project `rl-swe-scalingtest`, entity `apertus`, run name = `RUN`) | `T2V1K_WANDB_PROJECT` / `WANDB_ENTITY` |

Everything else is the task file's and the recipe's: 12 nodes (8 trainer, 4 vLLM), 16384 tokens per
trajectory, 6144 per turn, 30 turns, 20 steps, no validation, no checkpoints, agent and verifier caps
1800 s / 1200 s. Recipe: `recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml` (the HANGANG one
links its export from `/appsdata`).

## Run

From the repository root on a Daint login node:

```bash
export SCRATCH=/iopsstor/scratch/cscs/$USER ACCOUNT=<account>
DRY_RUN=1 bash experiments/terminus2-verified-1k/run.sh     # setup + job script, no submission
bash experiments/terminus2-verified-1k/run.sh               # 20 steps, JOB_TIME 12:00:00
STEPS=1 bash experiments/terminus2-verified-1k/run.sh       # one step first
```

- W&B: the API key comes from `WANDB_API_KEY` in the environment (sbatch passes it on), or, when that is
  unset, from `WANDB_KEY_FILE` (default `~/.w_and_b_key`), which `driver.sh` reads at job start. `WANDB=0`
  runs without W&B.
- Also needed: `~/.opensandbox_token`, `~/.rl_registry_cred`, SSH access to `swiss-ai/apertus-common`
  (`APERTUS_COMMON_URL` defaults to SSH), and the task data in `$SCRATCH/osb/harbor_verified`
  (prepared from the capstor task store on first use).
- Left out of training by default, as in the earlier Alps runs: campaign `swe-rebench-v2.1-rl` (slow pod
  starts) and 5 tasks whose verifier timed out (`EXCLUDE_CAMPAIGNS`, `EXCLUDE_TASKS`; empty = none).
- Any other setting is one more `key=value` argument.

## Its own clone

This branch pins Gym and NeMo-RL differently from other branches that use the default clone
`${PROJECTS_DIR}/nemo-rl-apertus2`, and `setup.sh` / `setup-gym.sh` would reset that clone under their
running and queued jobs (a job mounts the clone its EDF names when it starts, not when it is submitted).
So `run.sh` here uses a clone (next to the shared one), Gym venvs, a pydeps overlay and an EDF of its own
(`*-t2v1k`). It ignores `NEMO_RL_DIR`, `GYM_VENV_DIR`, `PYDEPS_GYM_DIR`, `EDF_NAME` and `WANDB_PROJECT` from the
environment (a login shell that exports `NEMO_RL_DIR` would otherwise point it at the shared clone) and refuses
to run on the shared clone or a shared EDF; `T2V1K_NEMO_RL_DIR`, `T2V1K_GYM_VENV_DIR`, `T2V1K_PYDEPS_GYM_DIR`,
`T2V1K_EDF_NAME` and `T2V1K_WANDB_PROJECT` override. The first launch clones and sets it up; `DRY_RUN=1` runs
the setup too.

## Risks at this size (not tested)

- **Sandbox cluster.** 1024 task pods (plus verifier pods) on Breithorn, which other tenants share. The
  task file stays at 200 because sandboxes "queued to start and failed readiness mid-trial" at about 250
  pods; whether the cluster takes 1024 is not known. Watch the OpenSandbox Grafana dashboard (waiting
  sandboxes, image pulls) and lower `SANDBOXES` if pods queue.
- **Inference.** Up to 1024 agent sessions on 4 vLLM nodes; requests queue in vLLM, which slows each
  trial and so each step.
- **Age 4 vs concurrency 1024.** One step's rollouts fill all 1024 sandboxes, so later steps' rollouts can
  only start as earlier ones finish; age 4 adds overlap only when the pod limit allows it.
- **Gym at this branch's pin** has none of the Terminus-2 fixes of the chonk-sc branch: no limit on
  consecutive truncated replies (Terminus-2 re-asked up to 320 times there), no total deadline for a
  sandbox create (a registry outage held a group for ~6900 s), no failure classes or infra masking.
  Each of these hits more often with 1024 rollouts in flight.

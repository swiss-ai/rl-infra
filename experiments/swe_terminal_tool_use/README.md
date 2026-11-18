# SWE and terminal tool-use RL

GRPO on the Harbor verified tasks: SWE and terminal tasks from 15 validation campaigns, each checked
against its own solution. The agent is Terminus-2 through Gym's `harbor_agent`. Every trial runs in a
fresh OpenSandbox pod, and the task's tests, run in a second pod, give the reward. The task file is
`envs/harbor-verified-sft.yaml`, and the starting point is the SFT checkpoint in
`recipes/apertus2-chonk-sft18315-v0.2-vllm-adamw-tools.yaml`, the task file's recipe.

The commands below are for HANGANG and run from the repository root on a login node.

## 1. Prerequisites

Once per user:

- **OpenSandbox API key**, one line, in `~/.opensandbox_token`:
  ```bash
  install -m 600 /dev/null ~/.opensandbox_token && echo '<key>' > ~/.opensandbox_token
  ```
- **Registry credentials** for the private project holding the task images, one line
  `username:password`, in `~/.rl_registry_cred`:
  ```bash
  install -m 600 /dev/null ~/.rl_registry_cred && echo '<user>:<password>' > ~/.rl_registry_cred
  ```
- **Weights & Biases key**, for runs that log to W&B. See the header of `backends/nemo-rl/launch.sh`
  for where it is read from.
- **Access to `swiss-ai/apertus-common`**, a private repository that the setup clones for the SFT
  checkpoint. Either run `gh auth setup-git` once, or, if SSH to GitHub works, prefix the launch
  with:
  ```bash
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.git@github.com:swiss-ai/apertus-common.git.insteadOf \
      GIT_CONFIG_VALUE_0=https://github.com/swiss-ai/apertus-common.git
  ```
- **The task store**: a copy of `harbor_verified_tasks`, one directory per campaign with the tasks as
  tarballs under `<campaign>/tasks/`. On HANGANG it is at
  `~/projects/harbor_verified_tasks/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks`.

## 2. Prepare the data

```bash
bash envs/prepare_harbor_verified_tasks.sh \
    ~/projects/harbor_verified_tasks/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks \
    ~/projects/task_datasets/harbor_verified
```

This writes `tasks/<campaign>__<task>/`, `train.jsonl`, `validation.jsonl` and `task_names.txt` under
`~/projects/task_datasets/harbor_verified`. It runs on the login node and needs only the standard
library; nothing is pulled, because every `task.toml` names digest-pinned images.

- The collector takes only the tasks a campaign's `tasks.csv` marks `keep`, and skips campaigns
  without a `tasks.csv`.
- Other knobs: `LIMIT` (0 = every task), `NUM_VALIDATION` (16), `SEED` (0), and `COLLECT_ARGS=--force`
  to extract every task again.
- Reruns are incremental: unchanged tasks are left alone, and tasks that left the store are removed.

## 3. Launch

```bash
RUN=sft19374-harbor-verified-<name>
RUN=$RUN EXCLUDE_NODES=node1903 JOB_TIME=24:00:00 WANDB_PROJECT=grpo-dev WANDB_ENTITY=apertus \
bash run.sh envs/harbor-verified-sft.yaml \
    harbor_data_dir=$HOME/projects/task_datasets/harbor_verified
```

`DRY_RUN=1` shows the job without submitting it. On success, `run.sh` prints the job ID and the paths
of the training log, the metrics and the final weights. Any other setting can be overridden the same
way, e.g. `grpo.max_num_steps=50`.

## 4. Monitor

- **Training log**: `$SCRATCH/nemo-rl/slurm/<job id>-logs/ray-driver.log`. The driver's output is
  buffered, so step summaries can appear late. Grep it for `Total step time`, `Avg Reward` and
  `Collecting rollouts`.
- **Trial results**: one `result.json` per trial under `$SCRATCH/nemo-rl/results/$RUN/harbor-jobs`.
  `exception_info.exception_type` says why a trial failed (`AgentTimeoutError`,
  `SandboxApiException`, …).
- **W&B**: project `grpo-dev` under `apertus`, with the run named `$RUN`.
- **Sandbox cluster**: the OpenSandbox Grafana dashboard, linked from
  https://osb-ui.swissai.svc.cscs.ch/#metrics. It shows running and waiting sandboxes, image pulls
  and node memory. Its panels also explain most slowdowns: sandboxes waiting in `PodInitializing`
  are pulling images.

## 5. Clean up

A run's sandboxes live at most `sandbox_ttl_s` (1 h). To delete those a cancelled or finished run leaves
behind, by the run's label:

```bash
printf 'sandbox:\n  opensandbox:\n    connection:\n      domain: https://opensandbox-gvisor.breithorn.svc.cscs.ch\n      api_key_file: %s\n' \
    $HOME/.opensandbox_token > $SCRATCH/opensandbox-connection.yaml
$HOME/.cache/nemo-rl-apertus2/gym-venvs/resources_servers/code_gen/.venv/bin/python \
    $HOME/projects/nemo-rl-apertus2/3rdparty/Gym-workspace/Gym/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py \
    --connection-config $SCRATCH/opensandbox-connection.yaml --run-id $RUN --user rl-infra --reap
```

## Comments

### The launch arguments

Config overrides, after the task file:

| Argument | Why it is on the command line |
|---|---|
| `harbor_data_dir=$HOME/projects/task_datasets/harbor_verified` | Required, `???` in the task file. The tasks prepared in step 2. A per-user path, and the container's `HOME` is not yours. |

Environment variables, read by `run.sh` and `backends/nemo-rl/launch.sh` rather than the task file:

| Variable | Why |
|---|---|
| `RUN=sft19374-harbor-verified-<name>` | The run's name: the Slurm job, the W&B run, the output directories, and the `run` label on its pods. Without it, `run.sh` uses the task file's name and a timestamp. |
| `EXCLUDE_NODES=node1903` | Keeps a node with a faulty GPU out of the job. Drop it once the node is repaired. |
| `JOB_TIME=24:00:00` | The job's time limit. The default is 5 hours. |
| `WANDB_PROJECT=grpo-dev` | Logs the run to Weights & Biases, in this project. Without it, nothing goes to W&B. |
| `WANDB_ENTITY=apertus` | The W&B team the project belongs to. |

### Defaults in the task file

These used to be passed at launch and are now set in `envs/harbor-verified-sft.yaml`. Override them
for a full training run.

| Setting | Default | For a training run |
|---|---|---|
| Checkpointing | The weights every 20 steps, without the optimizer state, and only the latest kept: `enabled: true`, `save_period: 20`, `save_optimizer: false`, `keep_top_k: 1` | Keep. `checkpointing.enabled=false` for a test. A run resumed from these starts with a fresh optimizer. |
| Validation | `grpo.val_period: 0`, `val_at_start: false`, `val_at_end: false` | `grpo.val_period=10 grpo.val_at_start=true grpo.val_at_end=true` |
| Trajectory age | `grpo.async_grpo.max_trajectory_age_steps: 2` | Keep. A step's rollouts can start while the two previous steps' are still in flight, which keeps more sandboxes busy. |

The task file also sets the recipe and where your credentials are read from:

| Setting | Default | To change |
|---|---|---|
| Recipe | `defaults: ../recipes/apertus2-chonk-sft18315-v0.2-vllm-adamw-tools.yaml`, SFT checkpoint chonk-lc-16k-v0.2 `iter_0018315` | `RECIPE=recipes/<recipe>.yaml` at launch, e.g. `apertus2-chonk-sft19374-vllm-adamw-tools.yaml` for v0.1 `iter_0019374` |
| Registry credentials | `registry_cred_file: ${oc.env:RL_USER_HOME}/.rl_registry_cred` | `registry_cred_file=<path>` |
| OpenSandbox API key | `…connection.api_key_file: ${oc.env:RL_USER_HOME}/.opensandbox_token` | `env.nemo_gym.sandbox.opensandbox.connection.api_key_file=<path>` |

`HOME` inside the container is not yours, so `run.sh` exports your `HOME` as `RL_USER_HOME` for the
task file. Both files hold credentials and stay out of the repository.

The task file also sets three fixes that runs need:

- `harbor_agent.concurrency: 300`, the number of trials, and so pods, in flight at once. The previous
  default, one per rollout in the batch, would be 512. The cluster runs about 300 sandboxes at once
  whatever the cap: it starts about 1.2 pods per second. Caps of 512 and 1024 only left 300 to 750
  sandboxes waiting to start, which time out after 600 s (`POD_READY_TIMEOUT`) and are recreated.
- `sandbox.opensandbox.connection.use_server_proxy: false`. Gym's `opensandbox.yaml` sets it to
  `true`, which sends every command through the OpenSandbox server. At about 190 sandboxes that
  server answered "no available server"; `false` goes through the ingress gateway instead.
- `policy.megatron_cfg.env_vars.PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"`, against
  training running out of GPU memory from allocator fragmentation.

### Limits per trial

- **Context**: one budget for the whole trajectory, `policy.max_total_sequence_length` (16384 from
  the recipe), which becomes vLLM's `max_model_len` and Terminus-2's `max_input_tokens`. A trial that
  runs out ends with `ContextLengthExceededError` and keeps its reward, usually 0. The SWE campaigns
  hit it most. Each turn generates at most `policy.generation.max_new_tokens` (6144).
- **Time**: from each task's `task.toml`: agent timeout (capped at 1800 s), verifier timeout (capped at 1200 s), and
  environment start timeout, multiplied by `harbor_environment_build_timeout_multiplier` (5) to cover
  a pod's wait for its image.

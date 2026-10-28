# Chonk GRPO on Harbor tasks with OpenSandbox sandboxes

How to run the Apertus 2 Chonk blueprint (async GRPO, Megatron trainer + vLLM) with agentic rollouts
whose sandboxes are pods on the OpenSandbox cluster (Breithorn). Written so that anyone in the
swiss-ai organisation with Clariden access can reproduce it from a clean account.

**Status (2026-10-28): the full Chonk job runs end to end on KISTI** (job 1153929, 12 nodes: 5 GRPO steps,
302 of 320 rollouts real, multi-turn agent trials on OpenSandbox sandboxes, reward 0 as expected). **It also runs end to end on CSCS Alps (Daint)**
(job 5184052, 12 nodes: 5 GRPO steps, 411 trials, one trial with reward 1). It has not run on Clariden yet (queued).
What is proven and what is not is listed in [Status](#status) below.

Contents: [Changelog](#changelog) · [What it is](#what-it-is) · [Status](#status) · [Access](#access-you-need) ·
[Steps](#steps) · [Config reference](#config-reference) · [Errors we hit](#errors-we-hit) ·
[Queue etiquette](#queue-etiquette) · [Cleanup](#cleanup) · [KISTI HANGANG](#kisti-hangang) ·
[Open questions](#open-questions) · [Attempt log](#attempt-log)

## Changelog

Newest first. If you followed an earlier version, the entries marked **action** need you to change something.

- **2026-10-29**
  - Larger layouts, both validated on KISTI (5/5 steps): `recipes/apertus2-chonk-vllm-adamw-20n.yaml` (16 trainer
    nodes EP64 + 4 vLLM; runs 8192-token contexts) and `-24n.yaml` (8 trainer + 16 vLLM; 128 sandboxes per step).
    See [Larger layouts](#larger-layouts).
  - `OVERLAY=<file>` in both launchers merges a task overlay last; `envs/harbor-osb-proxy-sft.yaml` is an experiment
    (role-marked chat template, 8192 tokens) that did not improve the base model's JSON validity.
  - Sandboxes also outlive **successful** runs (the batch in flight at the end): reap them after every run, see [Cleanup](#cleanup).
- **2026-10-28 (evening)**
  - **Action:** OpenSandbox moved to **`https://opensandbox-gvisor.breithorn.svc.cscs.ch`**, with new API keys. The old
    `opensandbox.breithorn.svc.cscs.ch` has served Traefik's default self-signed certificate and 404 since 11:32 today.
    The launchers and `validate_micro_tasks.py` now default to the new endpoint; put the new key in `~/.opensandbox_token`.
  - Real SWE tasks: new `envs/prepare_swe_lego_tasks.py` builds a `HARBOR_ROOT` from Ambroise's 1,830 validated SWE-Lego
    tasks, see [Using real SWE tasks](#using-real-swe-tasks).
- **2026-10-28**
  - **Action:** pull the branch. `envs/harbor-osb.yaml` gained three settings without which every rollout is empty:
    `harbor_agent_environment_server: null`, `vllm_cfg.http_server_serving_chat_kwargs.chat_template`, and a
    `policy.tokenizer.chat_template` that ends assistant turns with EOS. See [Errors we hit](#errors-we-hit).
  - **Action:** `backends/nemo-rl/apertus2/pydeps.sh` installs Gym's dependencies that David's image lacks
    (`devtools`, `yappi`, `pydot`, a current `openai`), and `setup-gym.sh` relaxes Gym's `requires-python` from 3.13
    to 3.12 (the image's Python). Re-run `setup-gym.sh` on an existing clone; `pydeps.sh` re-installs by itself.
  - New `kisti/launch-harbor-osb.sh`: the GRPO job on KISTI. First complete Chonk run (KISTI job 1153929).
    The KISTI procedure moved to the private `swiss-ai/kisti-docs` (folder `rl/`); [KISTI HANGANG](#kisti-hangang) links to it.
  - First run on CSCS Alps (Daint, account `csstaff`, Philip Mueller): job 5184052 completed all 5 steps with one
    reward of 1; the 6-node layout (job 5184751) runs out of GPU memory in its first training step. [Status](#status),
    [Watch](#7-watch), [Errors we hit](#errors-we-hit), [Open questions](#open-questions) and [Attempt log](#attempt-log)
    updated. The per-trial globs were one level too short: trials are five directories below `$HARBOR_ROOT/jobs`.
- **2026-10-27**
  - **Action:** `envs/harbor-osb.yaml` now sets `enable_chunked_prefill: true`. With `false`, every vLLM worker
    fails at start-up on Chonk ("Chunked prefill is required for mamba cache mode 'align'", job 3753228). Pull the
    branch before submitting.
  - **Action:** the `harbor-osb` branch is on GitHub (`swiss-ai/rl-infra`), rebased on main. Earlier versions of
    this page referred to a bundle; use the branch instead.
  - New section [KISTI HANGANG](#kisti-hangang): Chonk loads and generates on KISTI, 2-node
    NCCL works, and the OpenSandbox API is reachable from KISTI. New scripts in `kisti/`.
  - GPU memory: GH200 "120GB" nodes report about **96 GB per GPU** (`nvidia-smi`: 97,871 MiB). Size memory with 96.
  - Attempt log and Errors table updated (job 3753228 result, queued jobs).
  - Removed references to `sandbox-instructions.md`, which is not in this repository; use the handbook page
    `engineering/post-training/opensandbox` instead.
- **2026-10-26:** first complete version (steps, config reference, errors, queue etiquette, attempt log).

## What it is

- **Model and trainer:** Apertus 2 Chonk (108.3B parameters, 42 layers, 256 experts, top 8; the
  long-context pretraining checkpoint at iteration 3576), Megatron trainer with AdamW and CPU optimizer
  offload, vLLM generation. This is David's Apertus 2 blueprint (`recipes/apertus2-chonk-vllm-adamw.yaml`,
  12 nodes = 8 trainer + 4 vLLM, EP32), unchanged apart from the overrides in `envs/harbor-osb.yaml`.
- **Rollouts:** NeMo Gym's `harbor_agent` runs Terminus-2 **on the training nodes**. It calls the policy
  through the vLLM HTTP server and drives `tmux` inside a remote sandbox over the OpenSandbox API.
  One trial = one sandbox. The reward is read once, at the end, from `/logs/verifier/reward.txt`
  (the Harbor verifier, `tests/test.sh`) and is one scalar for the whole trajectory. Only
  Clariden -> OpenSandbox traffic is needed; the sandbox never calls back.
- **Data:** by default 96 generated *micro-tasks* (8 families such as "write the sum of these numbers to a
  file"), all on the public `python:3.12-slim` image, with a reference solution each. They exist to test the
  plumbing without registry credentials. Real SWE-Lego tasks work through the same path (see
  [Using real SWE tasks](#using-real-swe-tasks)).
- **Expect reward near 0.** Chonk is a pretraining checkpoint with a passthrough chat template (turns are
  concatenated with no role markers) and may not produce Terminus-2's JSON actions. The first run is a
  plumbing run, not a learning run.

## Status

Proven:

| What | Evidence |
|---|---|
| Clariden compute nodes can drive OpenSandbox; the Harbor/Terminus-2 loop works end to end | 1B smoke run on `debug` (Kostas's stack, Qwen2.5-1.5B): 114 trials, no exceptions, every trial reached the verifier, one GRPO step completed. Rewards 0.0, as expected for 1.5B |
| The same path trains a 30B model with non-zero rewards | Kostas's Qwen3-30B-A3B run (9 nodes), see the handbook page `engineering/post-training/rl-nemorl` in `swiss-ai/apertus-program` |
| The micro-tasks are valid | `envs/validate_micro_tasks.py`: reward 0 on the untouched pod, 1 after the reference solution, all 8 families |
| Chonk builds in David's image with Gym and the OpenSandbox provider importable | `check_env.sh`: 108.3B parameters, no shape mismatches, 39 `e_score_correction_bias` tensors absent from the checkpoint and left at their init values, "Gym (Harbor + OpenSandbox)" section passes |
| The config gets through Ray start-up, data loading and config validation | 12-node attempts reached the entrypoint's config check and `grpo.setup()`; the 6-node job 3753228 reached vLLM worker start-up (Ray cluster ready 72 s after the job started) before the errors below ([attempt log](#attempt-log)) |
| **Chonk GRPO with Harbor + OpenSandbox runs end to end** (12 nodes, blueprint layout, async GRPO) | KISTI job 1153929: 5 steps in 20.5 min; 302 of 320 rollouts real (18 masked); 200 trials checked had no exception, mean 4.2 Terminus-2 turns (max 20); a sandbox per trial on Breithorn, verifier reward read for all 320; weight sync about 4.6 s; generation KL error 0.0006; step times 16 to 235 s. Reward 0.0 everywhere, so the loss is 0 and nothing is learned yet |
| Chonk serves from David's image with the blueprint's vLLM layout (TP1, DP4, expert parallel per node) | KISTI job 1152908: weights load in 113 s (58 GiB per GPU, 16.9 GiB left for KV cache at 0.85 utilisation), and a completion returns "Bern." for "The capital of Switzerland is". See [KISTI HANGANG](#kisti-hangang) |
| **The same on CSCS Alps (Daint)**, 12 nodes, branch as of 2026-10-28 | Daint job 5184052 (`csstaff`, `normal`): 5 steps, driver exit 0; start-up about 9.5 min (`ray.sub` start to step 1), step times 241 / 27 / 129 / 73 / 100 s; 411 trials over all 8 families, 16 masked; **one trial scored 1** in step 1 (mean 0.0156 over 64), 0 afterwards. Peak trainer memory per GPU 59.6 GB allocated, 68.2 GB reserved |

**Not proven** (nobody has run these):

- A model that learns. Daint job 5184052 had one reward of 1 (step 1, so one step with a non-zero gradient); otherwise
  Chonk, a pretraining checkpoint, produced no correct Terminus-2 answer in 320 KISTI trials and 410 of 411 Daint trials.
  359 of the 411 Daint trials had at least one "No valid JSON object found" parser warning (5034 in total).
  Next: a chat/SFT checkpoint, or easier tasks.
- Contexts longer than 4096 tokens: 661 agent calls were refused with HTTP 400 because the conversation outgrew
  `max_model_len` (the trial then ends); on Daint 615 calls, and 125 trials ended with `ContextLengthExceededError`.
  Memory allows more on 12 nodes (68.2 of 96 GB reserved at 4096). Raise `max_total_sequence_length`.
- The same run on Clariden (jobs queued).
- The 6-node layout (4 trainer EP16 + 2 vLLM) does not fit at 4096 tokens: out of GPU memory in the first training
  step (Daint job 5184751, see [Errors we hit](#errors-we-hit)).

## Access you need

1. **GitHub (swiss-ai org):** `swiss-ai/rl-infra`, `swiss-ai/Nemo-RL` (the fork), `swiss-ai/nemo-gym`,
   `swiss-ai/Megatron-LM-MoE`, `swiss-ai/Megatron-Bridge`. The Clariden login node can reach GitHub; private repos need a
   deploy key or token there.
2. **Clariden:** Slurm account `infra01`, partitions `debug` (max 4 nodes, 90 min, one running job per user),
   `normal`, `preemptable`, `highprio` (see [Queue etiquette](#queue-etiquette)).
3. **An OpenSandbox API key** from the platform administrators (the admin key is not for routine use).
   Your IP must be on the allowlist; Clariden compute nodes are.
   Put the key in `~/.opensandbox_token` on Clariden, **one line, only the token**, then `chmod 600`.
   Never paste it into chat or commit it. Check it with an authenticated call (`/health` needs no key and
   still returned 200 while the key was rejected with HTTP 401 on 2026-10-22):

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" -H "OPEN-SANDBOX-API-KEY: $(tr -d '\n' < ~/.opensandbox_token)" \
     "https://opensandbox-gvisor.breithorn.svc.cscs.ch/v1/sandboxes?pageSize=1"      # expect 200
   ```
4. **No registry credential** is needed for the micro-tasks or for the public Harbor project `swe_lego_v05_1`.
   The wrapper sends the placeholder `anonymous` / `unused`, which the public project accepts.
   Private projects (e.g. `swe_lego_v04_1`) need real credentials, and the registry is IP-whitelisted.
5. `python3` on the Clariden login node is 3.6. Use `python3.11` (it is in `~/.local/bin`) for the scripts below,
   or run them on your laptop.

## Steps

All commands run on the Clariden login node unless stated. `$SCRATCH` is `/iopsstor/scratch/cscs/$USER`.
On Daint `$SCRATCH` is `/ritom/...`, which the EDF does not mount (the job then exits with code 127);
`export SCRATCH=/iopsstor/scratch/cscs/$USER` before setup and submit (`site.env` only sets it when unset).

### 1. Get the repository

```bash
git clone https://github.com/swiss-ai/rl-infra.git ~/rl-infra && cd ~/rl-infra
git checkout harbor-osb            # branch with the files below (rebased on main)
cp site.env.example site.env       # then set ACCOUNT=infra01 in site.env
```

The branch adds to David's blueprint:

| File | What it is |
|---|---|
| `run-harbor-osb.sh` | Launcher. Reads the key, checks it with an authenticated call, sets the labels the reaper uses, writes the merged config, calls `launch.sh` |
| `envs/harbor-osb.yaml` | Task file merged over the recipe: Harbor agent, OpenSandbox, 64 sandboxes, 4096 tokens |
| `recipes/apertus2-chonk-vllm-adamw-6n.yaml` | The 12-node recipe cut to 6 nodes (4 trainer EP16 + 2 vLLM). Unvalidated |
| `envs/make_micro_tasks.py` | Generates the micro-tasks and the `train.jsonl` / `validation.jsonl` rows |
| `envs/validate_micro_tasks.py` | Checks the tasks on real sandboxes (reward 0 before, 1 after the reference solution) |
| `backends/nemo-rl/apertus2/setup-gym.sh` | Puts Gym with the OpenSandbox provider into the NeMo-RL clone |
| `backends/nemo-rl/apertus2/patches/nemo-rl/0008-...patch` | Ports `mask_degenerate_rollouts` into the NeMo-RL fork. `setup.sh` applies every patch in that directory |
| `backends/nemo-rl/apertus2/edf.toml.template` | `PYTHONPATH` gains the Gym checkout (`/opt/nemo-rl/3rdparty/Gym-workspace/Gym`) |
| `backends/nemo-rl/apertus2/check_env.sh` | New section that imports Gym, the OpenSandbox provider and the Harbor agent |

### 2. Blueprint setup (David's stack)

```bash
SKIP_VLLM_IMAGE=1 bash backends/nemo-rl/apertus2/setup.sh    # NeMo-RL fork + PRs + patches, Megatron, EDF, image
bash experiments/grpo-chonk/stage_model.sh                   # 202 GB Chonk checkpoint to $SCRATCH/models/apertus2-chonk-lc-iter3576
```

(`run.sh` does both for you; here we call them directly because `run-harbor-osb.sh` does not run setup.) The
first run copies about 200 GB, the 28 GB container image and clones several repositories, so it takes a while.
Problems we met in this step (all fixed by hand, none in the scripts yet):

- `setup.sh` merges **PR #2517** (NVIDIA-NeMo/RL, vLLM DP) by its head. The head moved on 2026-10-20 and the merge
  into `megatron_policy_worker.py` conflicted, which then broke the `ray.sub` patch. We pinned that PR at commit
  `05db6ab36` (2026-05-30). If you hit the same conflict: build the clone by hand, merging the pinned commits
  from `pin` in order, with that commit for #2517.
- The clone has no GitHub push access by design (`remote set-url --push origin no_push`).

### 3. Gym with OpenSandbox

```bash
bash backends/nemo-rl/apertus2/setup-gym.sh
```

This clones `swiss-ai/nemo-gym` into the NeMo-RL clone's `3rdparty/Gym-workspace/Gym`, creates the local branch
`apertus2-osb` at the blueprint's Gym pin `267305e2a` and cherry-picks Kostas Pitas' four commits from
`swissai/harbor_agent_osb_v0.7.0` (`810196ff6`, `0fe241c66`, `44c77151b`, `85f6def4a`). Reason: the fork pins a Gym
(2026-09-15) that is newer than the OpenSandbox provider (2026-06-22) and the Harbor agent (2026-03), so it has
neither the provider nor the training config. The four commits applied without conflicts when we tried it. The
branch is local; nothing is pushed.

### 4. Check the environment on a GPU node (about 3 minutes, 1 node)

Run `backends/nemo-rl/apertus2/check_env.sh` inside the EDF `nemo-rl-apertus2` with working directory `/opt/nemo-rl`,
on `debug`, for example:

```bash
srun -A infra01 -p debug -N1 -t 00:20:00 --environment=nemo-rl-apertus2 --container-workdir=/opt/nemo-rl \
  bash ~/rl-infra/backends/nemo-rl/apertus2/check_env.sh
```

Expect exit code 0, the "Gym (Harbor + OpenSandbox)" section passing, `Apertus2ForCausalLM 108.3B params`, no
dropped keys or shape mismatches, and the note that 39 `e_score_correction_bias` tensors stay at their init values.
"Container fingerprint file not found, skipping version check" is normal here.

The Gym section imports `nemo_gym` and the provider, but not `nemo_gym.cli`, which is what needs `devtools` and a
current `openai`. Importing `from nemo_gym.cli import RunHelper, GlobalConfigDictParserConfig` (and
`vllm.entrypoints.openai.api_server`) with the `pydeps.sh` packages installed would catch them on `debug`.

### 5. Data

```bash
export HARBOR_ROOT=$SCRATCH/osb/harbor_micro
python3.11 envs/make_micro_tasks.py --out $HARBOR_ROOT/tasks --count 96 --seed 0 --num-validation 8
OPENSANDBOX_API_KEY=$(tr -d '\n' < ~/.opensandbox_token) python3.11 envs/validate_micro_tasks.py $HARBOR_ROOT/tasks
```

The first command writes 96 task directories (88 train and 8 validation rows in `train.jsonl` /
`validation.jsonl`, `instance_id` = `swe_lego::<task>`). The second creates one sandbox per task family, and every line
must read `OK ... before=0 after=1` followed by `ALL OK`. The `swe_lego` alias is only a name: `envs/harbor-osb.yaml`
maps it to `$HARBOR_ROOT/tasks`.

#### Using real SWE tasks

Ambroise's SWE-Lego set (public rl-registry project `swe_lego_v05_1`, 1,830 tasks whose reference solution scores 1
and whose empty run scores 0) is on Alps:

```bash
python3.11 envs/prepare_swe_lego_tasks.py /capstor/store/cscs/swissai/infra01/users/aborbely/swe_lego_v05/tasks \
  $SCRATCH/osb/swe_lego_v05 [--limit N]
export HARBOR_ROOT=$SCRATCH/osb/swe_lego_v05
```

It flattens the layout and adds `environment/Dockerfile` (`FROM <docker_image>`): the Harbor version Gym installs ignores
task directories without `environment/`, and every trial then fails with "Either datasets or tasks must be provided".

Alternatively, use `examples/nemo_gym/fetch_harbor_osb_tasks.py` and `prepare_harbor_osb_dataset.py` from `swiss-ai/nemo-rl-development`
(branch `swissai/harbor_agent_osb_v0.8.0`; see their `--help`. We used `--project swe_lego_v05_1 --limit N`
for the public project), then point `HARBOR_ROOT` at the output. These images are about 1.0 to 1.4 GB each and create
in about 26 s the first time, against about 4 s for the public micro-task image. Kostas's 10-task set uses the
private `swe_lego_v04_1` project and needs registry credentials.

### 6. Submit

```bash
cd ~/rl-infra
export HARBOR_ROOT=$SCRATCH/osb/harbor_micro PARTITION=normal JOB_TIME=02:00:00
export RECIPE=recipes/apertus2-chonk-vllm-adamw-6n.yaml      # 6 nodes; omit for the 12-node blueprint layout
export RUN=chonk-harbor-micro-$(date +%m%d-%H%M)
bash run-harbor-osb.sh
```

The wrapper stops if the OpenSandbox key is rejected. The key is exported **before** `sbatch` and is baked into
the job, so replace a bad key by cancelling the job and resubmitting. The wrapper writes `$SCRATCH/nemo-rl/configs/$RUN.yaml`
as `defaults: [recipe, envs/harbor-osb.yaml]` with absolute paths, so **edits to those files made while the job is
still queued take effect when it starts**.
The warning "checkpoint_must_save_by ... is not before the job time" is harmless: checkpointing is off.

#### Larger layouts

| Recipe | Nodes | Trainer / vLLM | Use | Validated (KISTI) |
|---|---|---|---|---|
| `apertus2-chonk-vllm-adamw.yaml` | 12 | 8 (EP32) / 4 | the blueprint; up to 6144 tokens | 1153929, 1172833 |
| `apertus2-chonk-vllm-adamw-20n.yaml` | 20 | 16 (EP64) / 4 | 8192 tokens | 1172834 (4096), 1172935 (8192) |
| `apertus2-chonk-vllm-adamw-24n.yaml` | 24 | 8 (EP32) / 16 | more sandboxes per step | 1172832 (128 per step) |

```bash
# 8192 tokens, 2048 per agent turn
RECIPE=recipes/apertus2-chonk-vllm-adamw-20n.yaml bash run-harbor-osb.sh \
  policy.max_total_sequence_length=8192 policy.sequence_packing.train_mb_tokens=8192 \
  policy.sequence_packing.logprob_mb_tokens=8192 policy.generation.max_new_tokens=2048
# 128 sandboxes per step
RECIPE=recipes/apertus2-chonk-vllm-adamw-24n.yaml bash run-harbor-osb.sh \
  grpo.num_prompts_per_step=16 policy.train_global_batch_size=128
```

Not working yet: 32 nodes (16 EP64 + 16 vLLM) at 8192 with 128 per step died in the first log-probability pass
(job 1172936). With more sandboxes per step, the sandboxes, not the GPUs, set the step time: each SWE-Lego task has
its own 1 to 1.4 GB image, pulled cold on each sandbox node (median 38 s from create to running, p90 8.5 min, max
23 min), and the 24-node run masked 128 of its 640 rollouts.

### 7. Watch

```bash
squeue -j <id> -o "%i %P %q %T %R %S"
tail -f $SCRATCH/nemo-rl/slurm/<id>-logs/ray-driver.log
```

All attempts so far died in the first two minutes of `ray-driver.log`, so read it as soon as the job starts. What to look for:

- `Loaded training dataset ... with 88 samples` then `Setting grpo.max_val_samples and grpo.val_batch_size to the length of the validation dataset`: config and data are fine.
- Model load, then vLLM start, then the first rollouts. `kubectl` is not needed: list sandboxes with
  `GET /v1/sandboxes` (handbook page `engineering/post-training/opensandbox`) and filter on your run id.
- A healthy Gym run of a 30B model takes about 1340 s per step and about 25 min of startup (Kostas, handbook). If steps take seconds and
  `Masking a degenerate` keeps climbing, every rollout is failing: stop and look at the first trial's logs under `$HARBOR_ROOT/jobs`.
  One `Masking a degenerate` line is one rollout that came back with no output (its sandbox did not start in time,
  or the trial failed before the first model reply); it gets zero weight in the loss instead of aborting the step.
  A few per step are normal.
- Per-trial rewards: `cat $HARBOR_ROOT/jobs/*/*/*/*/*/verifier/reward.txt | sort | uniq -c`; the solved ones:
  `grep -l 1 $HARBOR_ROOT/jobs/*/*/*/*/*/verifier/reward.txt`.
- Conversations: each trial directory `$HARBOR_ROOT/jobs/<date>/swe_lego/<model>/<HHMMSS_hash>/<task>__<id>/` holds
  `agent/trajectory.json` (every turn: model message, tool calls, observation) and `verifier/`; the task text is
  `$HARBOR_ROOT/tasks/<task>/instruction.md`. With `should_log_nemo_gym_responses: true` NeMo-RL writes no
  `train_data_step*.jsonl`.
- "Async GRPO training complete!" is also printed on the error path; check the exit code and the step count.
  The end of a successful log (`srun: error ... Terminated`, a graceful-shutdown timeout, the NemoGym actor killed
  by `ray.kill`) is the normal teardown.

## Config reference

`envs/harbor-osb.yaml` is merged over the recipe. These are the settings that matter and why.

| Setting | Value | Why |
|---|---|---|
| `grpo.num_prompts_per_step` x `num_generations_per_prompt` | 8 x 8 | 64 trajectories = 64 concurrent sandboxes, one per rollout |
| `grpo.max_num_steps` | 5 | Enough to time a step. Raise for a real run |
| `grpo.max_val_samples` | `null` | **Required.** The Gym entrypoint refuses a non-null value; the NeMo-RL default is 256 |
| `grpo.val_batch_size` / `val_num_generations_per_prompt` | `null` / 1 | Same as Kostas's config; the entrypoint sets the batch size from the validation set |
| `grpo.val_period`, `val_at_start`, `val_at_end` | 0 / false / false | No validation in a plumbing run |
| `checkpointing.enabled` | false | Skip the 200 GB checkpoint write. Use a **per-run** results directory: the trainer looks for the latest checkpoint even when saving is off, and a shared directory left at `max_num_steps` makes the next run exit as already finished |
| `policy.max_total_sequence_length` and both `sequence_packing.*_mb_tokens` | 4096 | Memory headroom on the trainer. David validated 2048. A multi-turn trajectory is cut at this length |
| `policy.generation.max_new_tokens` | 1024 | Per agent turn, not per trajectory |
| `policy.generation.stop_strings` | `null` | **Required.** The GSM8K recipe stops generation at `Question:`, which would cut agent turns short |
| `policy.generation.vllm_kwargs.enable_chunked_prefill` | true | **Required for Chonk.** Its Mamba layers need it; with `false` vLLM refuses to start. (The Harbor README disables it for non-Mamba models) |
| `policy.generation.vllm_cfg.async_engine` / `expose_http_server` | true / true | NeMo Gym calls the policy over an OpenAI-compatible HTTP endpoint |
| `policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template` | `${policy.tokenizer.chat_template}` | **Required for Chonk.** Its tokenizer has no chat template and the HTTP server does not use the recipe's on its own |
| `policy.tokenizer.chat_template` | pass-through + EOS after assistant turns | **Required for multi-turn.** NeMo-RL's HTTP server splices each turn's generated tokens back in at the EOS that closes them |
| `env.nemo_gym.harbor_agent_environment_server` | `null` | Kostas's Harbor config also starts `environment_servers/legacy_agent`, which only exists on his Gym branch |
| `data` block with `_override_: true` | see file | Replaces the whole recipe `data` block, so it must carry **`use_multiple_dataloader: false`**: `grpo.setup()` indexes that key |
| `env.nemo_gym.mask_degenerate_rollouts` | true | A rollout whose sandbox never came up returns no output items; this turns it into a zero-weight sample instead of aborting the step (patch 0008; needs `should_mask_flagged_samples`) |
| `...harbor_agent.concurrency` | `num_prompts * num_generations` | One trial and one sandbox per in-flight rollout |
| `...harbor_agent_kwargs.max_turns` | 20 | Terminus-2 turns per trial |
| `...sandbox_provider_options.secure_access` | true | The cluster rejects every create without it (HTTP 422) |
| `...sandbox_provider_options.image_auth` | placeholder | Ignored by public images and projects |
| `...harbor_agent_max_timeout` / `harbor_verifier_max_timeout` | 1800 s / 1200 s | Caps on the tasks' own limits |
| `...sandbox.opensandbox.connection.transport_backend` | `httpx` | As in Kostas's config |
| `NEMO_GYM_VENV_DIR` (wrapper) | `$HARBOR_ROOT/venvs` | Harbor runs each rollout as a Ray task whose runtime env carries an absolute venv path, so the venvs must be on a filesystem every node sees. The first run builds them, which takes minutes |
| `OPENSANDBOX_DOMAIN`, `OPENSANDBOX_API_KEY` | exported before `sbatch` | Gym resolves them inside Ray workers on every node, which inherit the job's environment |
| `NEMO_GYM_{TEAM,USER,WORKLOAD,RUN_ID}` | wrapper | Labels on every sandbox; the run id is what the reaper filters on |

OpenSandbox facts that matter: gVisor is the default isolation (since 2026-10-21), so a per-sandbox `networkPolicy` is
rejected; sandboxes have open internet but cannot reach CSCS-internal hosts (including the registry); about 110
pods per node and about 40 creates per second cluster-wide. Full API guide: the handbook page
`engineering/post-training/opensandbox` (repository `swiss-ai/apertus-program`).

## Errors we hit

| Error | Cause | Fix |
|---|---|---|
| `ValueError: A non-null grpo.max_val_samples parameter is not supported.` (driver log, 40 s after start) | The recipe inherits `max_val_samples: 256`; the Gym entrypoint rejects any non-null value | `grpo.max_val_samples: null` |
| `KeyError: 'use_multiple_dataloader'` in `grpo.py setup()` | The `data:` block with `_override_: true` replaced the recipe's block and dropped this key | `data.use_multiple_dataloader: false` |
| `ValidationError ... Chunked prefill is required for mamba cache mode 'align'` in every `VllmAsyncGenerationWorker` (driver log, about 3 min after start) | `enable_chunked_prefill: false` | `enable_chunked_prefill: true` (now set) |
| `No module named 'devtools'` at Gym start-up (then `yappi`, `pydot`) | David's image lacks Gym's dependencies | `pydeps.sh` installs them (KISTI: `kisti/pydeps-kisti.sh`) |
| `ImportError: cannot import name 'Moderation'` from `openai` at Gym start-up | The image has `openai` 2.7.2 | `pydeps.sh` installs a current `openai` |
| Exit code 127 right after the job starts, on Daint | `$SCRATCH` is `/ritom/...` there, which the EDF does not mount | `export SCRATCH=/iopsstor/scratch/cscs/$USER` |
| `Missing pyproject.toml or requirements.txt ... environment_servers/legacy_agent` | Kostas's Harbor config references a server that is only on his Gym branch | `harbor_agent_environment_server: null` |
| Gym server venvs: `the current Python version (3.12.3) does not satisfy Python>=3.13.14` | Gym declares 3.13 since 2026-08-04; the image has 3.12 | `setup-gym.sh` relaxes `requires-python` |
| Every rollout masked, `ChatTemplateResolutionError ... default chat template is no longer allowed` behind HTTP 500 from `/v1/chat/completions` | Chonk's tokenizer has no chat template; the HTTP server ignores the recipe's | `http_server_serving_chat_kwargs.chat_template` |
| HTTP 500 on every second agent turn, `AssertionError: EOS token #0 not found in template_token_ids` | The pass-through template has no EOS, which NeMo-RL needs to find the end of the previous turn | Template with EOS after assistant messages |
| HTTP 400 `Prompt exceeds max_model_len` (not an error) | The conversation outgrew 4096 tokens; the trial ends | Raise `max_total_sequence_length` when memory allows |
| 6 nodes: `ncclUnhandledCudaError ... Failed to CUDA calloc async 4 bytes` while creating an NCCL communicator in `_log_microbatch_router_metrics` (`finalize_model_grads`), first `policy.train()` | GPU out of memory: EP16 puts twice the experts on each GPU. Per GPU (96 GB): 65.4 GB allocated before the forward/backward, peaks 77.5 GB allocated and 88.8 GB reserved (12 nodes: 59.6 / 68.2 GB) | Use the 12-node layout. Untested: 2048 tokens, or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reserved minus allocated was 4 to 11 GB) |
| Agent turns truncated at "Question:" (not an error) | `stop_strings: ['Question:']` inherited from the GSM8K recipe | `policy.generation.stop_strings: null` |
| HTTP 401 `INVALID_API_KEY` from every create while `/health` is 200 | Key expired or rotated | New key; the wrapper now fails at submit time |
| HTTP 422 `ValidatingAdmissionPolicy ... denied request` | `secureAccess` missing | `secure_access: true` (already set) |
| `uv` cannot resolve editable submodules (`Automodel` empty, then nested `Megatron-LM` missing) | Submodules were not initialised recursively | Initialise with `--recursive` and sync them to the cluster |
| 20 nodes (EP64): `Assertion error internode.cu:3106` in UCCL at the first expert dispatch | The EP32 UCCL buffer settings are too small for 16 trainer nodes | `UCCL_EP_*_CONFIG` with 1024 (set in the 20-node recipe) |
| 20 nodes (EP64): every trainer process dies without a Python error in the first log-probability pass | Optimizer state and log-probability activations together exceed GPU memory | `policy.offload_optimizer_for_logprob: true` (set in the 20-node recipe) |
| `Reason=QOSGrpNodeLimit` for days on `normal` | The account's group node cap is filled by other users' jobs | See [Queue etiquette](#queue-etiquette) |

## Queue etiquette

- **`normal`** is the default. It is also where we waited **about 69 hours** for one 12-node job (priority, then
  `QOSGrpNodeLimit`, with a start estimate that kept moving later). Slurm's estimates move; do not plan around them.
- **`highprio`** (QOS `highprio`, set with `QOS=highprio` in the wrapper) started our two jobs within minutes (25 min and about 2 min), but it competes
  with the pretraining jobs, and the pretraining team complained after two short failed 12-node jobs. Use it only
  by agreement with the leads, and only for a run that has a real chance of running (check the config on `debug` first).
- **`preemptable`** is sooner but a preempted job loses its progress. It is acceptable for the 2 h plumbing run.
- **`debug`** (4 nodes, 90 min, one job per user) fits the 1B smoke run, not Chonk.
- Check `squeue -A infra01` to see who holds the nodes. A bug that fails in the first 2 minutes of the driver should be
  found on a small allocation first, not by queueing 12 nodes again.

## Cleanup

Sandboxes have a TTL (6 h in this setup). Both a killed job and a successful one leave sandboxes behind (a
successful one, the batch in flight when training ends), so reap them by run id after every run:

```bash
cd ~/projects/nemo-rl-apertus2/3rdparty/Gym-workspace/Gym
uv run --extra nemo_gym python -m nemo_gym.sandbox.providers.opensandbox.cleanup_sandboxes \
  --domain https://opensandbox-gvisor.breithorn.svc.cscs.ch --api-key "$(tr -d '\n' < ~/.opensandbox_token)" \
  --run-id <RUN> --user "$USER" --reap
```

(the command mirrors Kostas's handbook; we reaped our smoke run by listing `GET /v1/sandboxes` and deleting those whose
run label matched). `https://osb-ui.swissai.svc.cscs.ch` also lists and deletes sandboxes.

## KISTI HANGANG

The same job runs on KISTI's HANGANG system (Korea, GH200 nodes like Clariden, pyxis/enroot instead of the CSCS
Container Engine). The procedure, the shared copies of the image and checkpoint, the site settings and the attempt log
are in the private repository [`swiss-ai/kisti-docs`](https://github.com/swiss-ai/kisti-docs), folder `rl/`. The scripts
it uses are in `kisti/` of this branch.

## Open questions

- Does Chonk fit 4096-token multi-turn sequences, and does it follow the Terminus-2 JSON format? A reward of 0 would be expected
  and would prove the plumbing but not that the model can use the tools.
  *Daint 5184052:* memory fits with headroom (68.2 of 96 GB reserved); the JSON format mostly not (359 of 411 trials
  with parser warnings, one trial solved).
- Does async GRPO work with Gym at this scale? If the first step fails in the async code, a synchronous run is the fallback
  (`grpo.async_grpo.enabled: false`; the 6-node recipe then needs a re-check of the trainer/inference split).
  *Yes:* KISTI 1153929 and Daint 5184052 both ran 5 async steps (`max_trajectory_age_steps: 1`); on Daint, steps 3 to 5
  waited on the trajectory buffer, as expected.
- Is the 6-node layout stable (EP16, UCCL dispatch settings)? The 12-node layout is the one David validated.
  *Daint 5184751:* not at 4096 tokens; out of GPU memory in the first training step (see [Errors we hit](#errors-we-hit)).
- Can runc be requested per sandbox? Is a sandbox that lives for hours supported? Is the 110-pods-per-node cap compatible
  with 1k+ sandboxes?
- For datasets of 100k images: sandbox creation waits on image pulls (about 26 s cold, about 4 s warm). Async age 1 hides one step. Options:
  prefetch by dataloader order, a pull-through cache, a curriculum slice, or pools.
- KISTI: see `swiss-ai/kisti-docs`, folder `rl/`.

## Attempt log

| Job | Where | Result |
|---|---|---|
| 3712222 | `debug`, 1B, Kostas's stack | Passed: 114 trials, one GRPO step, rewards 0.0 |
| 3712438 | `debug`, `check_env.sh` | Passed |
| 3712461 | `normal`, 12 nodes, micro-tasks | Waited about 69 h, started 2026-10-24 09:43, failed after 2m20s: `max_val_samples` |
| 3752878 | `highprio`, 12 nodes | Started after 25 min, failed after 2m22s: `use_multiple_dataloader` |
| 3753104 | `highprio`, 12 nodes | Cancelled by us after 2m26s, during startup |
| 3753228 | `normal`, 6 nodes, 4096 tokens, `stop_strings: null` | Submitted 2026-10-26 11:01, started 2026-10-27 06:17 (19 h wait), failed after 5m19s: `enable_chunked_prefill` |
| 3762540 | `normal`, 6 nodes, all the fixes above | Queued |
| 3762800 | `preemptable`, same | Queued |
| 3754080 | `normal`, 12 nodes, Philip's reconstruction of the branch | `enable_chunked_prefill`, as 3753228 |
| 5162077, 5162114 | Daint `normal` (`csstaff`), 6 nodes, 2026-10-27 branch | Exit code 127: `$SCRATCH` (`/ritom`) not mounted |
| 5162937 | Daint, 6 nodes | `No module named 'devtools'` |
| 5167352 | Daint, 6 nodes, plus `devtools`, `yappi`, `pydot` | `ImportError: Moderation` (`openai` 2.7.2) |
| 5184052 | Daint, 12 nodes, branch as of 2026-10-28 | **Passed:** 5 steps in 1198 s, one reward 1 |
| 5184751 | Daint, 6 nodes, branch as of 2026-10-28 | GPU out of memory in the first training step after 788 s |
| KISTI jobs | 1 to 12 nodes | See `swiss-ai/kisti-docs`, folder `rl/` |

## Credits and sources

- David and Elias: the Apertus 2 blueprint, `recipes/` and `experiments/grpo-chonk/`.
- Kostas Pitas (and Ambroise): the Harbor + OpenSandbox integration, branches `swissai/harbor_agent_osb_v0.7.0`
  (`swiss-ai/nemo-gym`) and `swissai/harbor_agent_osb_v0.8.0` (`swiss-ai/nemo-rl-development`), and the handbook page
  `engineering/post-training/rl-nemorl`.
- OpenSandbox on Breithorn: the CSCS platform team.

# Apertus 2 Chonk on NeMo-RL's Single-Controller path (step 0, run A, run B, run S, run OV, run PV, run CV)

**New here? Start with `TESTING.md`**: prerequisites, commands and expected results of runs V and OV at 64K.
**`PORTING.md`**: which pieces to take for another model, task set, cluster or harness.
This README is the full reference and the history of the runs.

Implements the run sequence of the feasibility analysis
`research/reports/2026-11-04-chonk-single-controller-feasibility.md` (workspace, section 6):
Chonk (12 nodes: 8 trainer nodes Megatron EP32 with CPU-offloaded AdamW + 4 vLLM nodes TP1 EP4,
i.e. vLLM DP4 per node through NVIDIA-NeMo/RL#2517) trained through
`examples/run_grpo_single_controller.py` instead of the legacy `run_grpo*.py`.

| File | What |
|---|---|
| `check.sh`, `check.py` | step 0: static checks on the login node (no GPU, no sbatch) |
| `run-a-gsm8k.yaml`, `run-a.sh` | run A: GSM8K, no Gym, no capture |
| `run-b-harbor-sc.yaml`, `run-b.sh` | run B: SC rerun of Daint job 5184052 (Harbor micro-tasks, Terminus-2, token capture) |
| `run-s-sft.yaml`, `run-s.sh` | run S: run B on the SFT checkpoint (16k layout, Apertus 2 message format) |
| `run-v.sh`, `run-v-24k.sh` | run V: Terminus-2 on Kostas' verified tasks, Gym entrypoint (not SC), 24k on 20 nodes; `CTX=65536`: long context |
| `run-agent-v.sh`, `run-agent-v.yaml`, `run-ov.yaml` | run OV: OpenCode in the pod on run V's verified tasks, SC + token capture, 24k on 20 nodes; `CTX=65536`: long context |
| `run-pv.yaml` | run PV (`run-agent-v.sh AGENT=pi`): the pi coding agent instead of OpenCode; not run on the cluster yet |
| `run-cv.yaml` | run CV (`run-agent-v.sh AGENT=claude-code`, Gym 0048-0052): Claude Code instead of OpenCode; probed against a stub (`../claude-code-probe/`), one-node rung run (Qwen3-4B), not on Chonk yet |
| `common.sh` | config composition, the long-context profile of runs V and OV, and the clone / Gym checks, shared by the scripts |

Runs A and B: 3 steps, `async_rl` sampler `in_order` with lookahead 0 (synchronous), one chunk per
step (`min_groups_for_streaming_train` = `max_inflight_prompts` = `max_buffered_rollouts` =
prompts per step), TransferQueue data plane with 24 storage units (`${mul:2, num_nodes}` from
`grpo_math_1B.yaml`), generation fleet health off (its rebuild path ignores vLLM DP,
`weight_sync/membership.py`), validation and checkpointing off, stall watchdog `abort`,
`DRIVER_RETRIES=0` (SC writes no `train_data_step*.jsonl`, so `driver_retry.sh` would not retry).

## Daint

```bash
cd ~/rl-infra                                  # this branch: chonk-sc-harbor025
export SCRATCH=/iopsstor/scratch/cscs/$USER    # Daint: $SCRATCH is /ritom/..., which the EDF does not mount
export ACCOUNT=csstaff                         # run.sh / run-harbor-osb.sh default to infra01
```
On Clariden neither export is needed (`ACCOUNT` as usual). Partition defaults to `normal` in both launchers.

## Step 0: `check.sh`

```bash
bash experiments/chonk-sc/check.sh             # RUNS="a" or RUNS="b" for one run only
```
- **(b) NeMo-RL clone** (`$NEMO_RL_DIR`, default `~/projects/nemo-rl-apertus2`): the pinned PRs
  (#5, #1, #2517@05db6ab36, #8) are ancestors of HEAD; #2517 removed the assert "vLLM async_engine
  has some issues when using DP inside vLLM" (plain f196 `vllm_generation.py:191-199`) and added
  `workers_per_vllm_instance` (DP-aware `rank_prefix` in `init_collective`) and
  `data_parallel_external_lb` (`vllm_worker.py`); f196's newer code survived the `-Xtheirs` merge:
  `_refit_leader_workers`, `get_dp_leader_worker_idx`, `refit_timeout_s`,
  `membership.shard_prefixes`, `setup_token_capture` in `vllm_generation.py`,
  `weight_sync/collective_weight_synchronizer.py`. Markers chosen from the diff of #2517 against its
  merge base d9506f7dc and of f196 against the same base. Reference: the merge reproduced from the
  local commits (f196 + PRs + `backends/nemo-rl/patches` + `apertus2/patches/nemo-rl`) changes
  `vllm_generation.py` / `vllm_worker.py` against f196 by +9 -19 / +33 -19 lines; check.sh prints a
  note if the cluster clone differs. In that merge #2517 also replaces f196's
  `resolve_distributed_executor_backend` (`uni` for EP > TP) with its inline choice (`mp`); the
  legacy Chonk runs (5184052) ran with the same code.
- **(c) Gym checkout** (run B): `setup-gym.sh` has applied the current `patches/gym` set (its stamp
  matches), Gym patch 0014 (`harbor_model_base_url` / `base_url_for_run`: without it capture records
  nothing and every row is rejected, job 5236016), 0015 (`_should_mask`, capture-aware masking),
  `nemo_gym/token_id_capture`, the OpenSandbox configs (patches 0006-0009). Fix:
  ```bash
  bash backends/nemo-rl/apertus2/setup-gym.sh
  ```
  One checkout serves every run of the clone (main's tasks, Rob's harbor runs, run B).
- **(a) configs**: writes `$SCRATCH/chonk-sc-check/run-{a,b}.yaml` exactly as the launchers do
  (A: `run.sh` → `[recipe, run-a-gsm8k.yaml]`; B: `run-harbor-osb.sh` → `[recipe,
  envs/harbor-osb.yaml, run-b-harbor-sc.yaml]`), then `check.py` runs the clone's own code:
  `load_config` (`/opt/nemo-rl/...` read from the clone), the overrides `launch.sh` appends
  (`cluster.num_nodes=12 cluster.gpus_per_node=4 checkpointing.checkpoint_dir=... logger.log_dir=...`,
  plus any you pass), `MasterConfig(**config)` (pydantic validators, e.g. stall timeout >
  rollout timeout), the entrypoint's `async_grpo: null` / `data_plane.enabled` checks,
  `validate_single_controller_config` (batch equality, min_groups, buffer capacity, checkpoint
  metric, capture vs. advantages, wrong-path rollout_failure block, ...), and the config-only checks
  of `setup_single_controller` (validation off, `use_multiple_dataloader`, capture needs Gym +
  vLLM + async_engine, non-colocated), replicated because that function needs Ray and the
  tokenizer. Then it prints the resolved values and checks them against the plan. With `yq` it also
  runs `preflight.sh` as `launch.sh` will.

Python: `CHECK_PYTHON`, else a venv `$SCRATCH/chonk-sc-check/venv` built on first use from a Python
>= 3.12 on PATH (on the login nodes `python3.14`, tried first) with `omegaconf==2.3.1 hydra-core==1.3.2 pydantic==2.13.5 pyyaml` (the clone's
`uv.lock` versions; pip needs PyPI). Packages the config modules import but the venv lacks (torch,
ray, transformers, vllm, ...) are replaced by inert stubs and listed (`note stubbed`), with
`__version__` from the clone's `uv.lock`; the validators themselves are pure pydantic/config code.
For a run without stubs, use the image (optional, not tried):
`srun -A $ACCOUNT -p debug -N1 -t 00:10:00 --environment=nemo-rl-apertus2 env CHECK_PYTHON=/usr/bin/python3.12 bash experiments/chonk-sc/check.sh`.
Exit code non-zero on any FAIL; `WARN` lines (model / Harbor tree missing) do not fail.

## Run A: Chonk SC on GSM8K (no Gym)

```bash
bash experiments/chonk-sc/run-a.sh             # JOB_TIME default 01:30:00, RUN chonk-sc-a-gsm8k-<date>
```
`run.sh` (setup.sh + model staging, as every legacy run) with the 12n recipe and task
`run-a-gsm8k.yaml` (= `envs/gsm8k.yaml` + the SC block), `SCRIPT=examples/run_grpo_single_controller.py`.
32 prompts x 16 generations = `train_global_batch_size` 512 (SC: one RL step = one optimizer step;
the legacy 128 x 16 with 4 optimizer steps per RL step is not expressible), `max_total_sequence_length`
2048 (the recipe's; below 4096), max_new_tokens 2048.

Tests (risks 1, 2, 5, 6 of the report, without OpenSandbox): the SC split train API
(`begin_train_step` / `train_microbatch` / `finish_train_step`, `finalize_model_grads` once per step)
with MoE EP32 + UCCL/DeepEP flex dispatcher + CPU-offloaded distributed AdamW; refit through
`CollectiveWeightSynchronizer` (300 s watchdog) into 16 vLLM DP workers; trainer memory on SC;
TransferQueue with 24 storage units over 12 nodes.

Expected / grep (`$OUT/slurm/<jobid>-logs/ray-driver.log`, `$OUT` = `$SCRATCH/nemo-rl`):
```bash
L=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
grep -E "train step [0-9]+/3|Total step time|SC run complete" $L          # 3 steps, then 'SC run complete'
grep -o "'grad_norm': [^,]*" $L                                           # finite, > 0
grep -o "'reward': [^,]*" $L                                            # mean GSM8K reward per step, > 0 expected
grep -E "_sync_weights: sync done|refit: watchdog armed|RefitAborted" $L  # refit time (legacy 7.6-10.2 s)
grep -o 'max_alloc=[0-9.]*' $L | sort -t= -k2 -g | tail -1                # trainer peak ([gpumem] at chunk_enter/exit, step_finish)
grep -o 'max_reserved=[0-9.]*' $L | sort -t= -k2 -g | tail -1
grep -E "train_pump: step [0-9]+ (chunk|closing)" $L                      # exactly 1 chunk per step
grep -iE "out of memory|NCCL.*(timeout|error)|RolloutStall|Traceback" $L
```
No in-workspace legacy GSM8K Chonk baseline exists; for a direct comparison on the same shape, the
legacy path (async age 1) is `bash run.sh envs/gsm8k.yaml grpo.num_prompts_per_step=32
grpo.max_num_steps=3 grpo.val_period=0 grpo.val_at_start=false grpo.val_at_end=false
checkpointing.enabled=false` (not prepared further). Memory at 2048 tokens should be below
run B's / 5184052's 4096-token numbers.

## Run B: SC rerun of 5184052 (Harbor micro-tasks, Terminus-2, token capture)

```bash
bash experiments/chonk-sc/run-b.sh             # JOB_TIME default 02:30:00, RUN chonk-sc-b-harbor-<date>
```
`run-harbor-osb.sh` (OpenSandbox key check against `~/.opensandbox_token`, Harbor paths under
`$HARBOR_ROOT` = `$SCRATCH/osb/harbor_micro`, sandbox reaper) with the 12n recipe,
`envs/harbor-osb.yaml` (Rob's, unchanged) and overlay `run-b-harbor-sc.yaml`. Same shape as 5184052:
8 x 8 = 64 sandboxes, gbs 64, 4096 tokens total, 1024 new per turn, max 20 turns; 3 steps instead
of 5; lookahead 0 instead of age 1. `rollout_failure` 2 infra / 2 data attempts, no drops,
`rollout_timeout_s` 1800 (whole prompt group), stall watchdog 2100 s abort, harbor concurrency 64
(derived) = `max_inflight_prompts` x generations. `token_capture.enabled`: rows rebuilt from the
ledger in `<log_dir>/gym_token_capture`; NeMo-RL patch 0031's masking is inert under capture.

Tests (risks 3, 4): token capture with 16 vLLM DP workers on 4 nodes (a TQ client + token sink per
worker), the capture ledger's `fcntl.flock` on scratch across nodes, ContextLengthExceeded /
malformed-JSON turns under capture.

5184052 for comparison (legacy, Daint, 12 nodes, 4096 tokens, 8x8, async age 1): start-up ~9.5 min
(`ray.sub` 08:21:14 → step 1 08:30:35), step times 241 / 27 / 129 / 73 / 100 s (driver 1198 s),
trainer peak `max_alloc` 59.6 GB / `max_reserved` 68.2 GB per GPU (of ~95), 411 trials, 125 ended
in `ContextLengthExceededError`, 615 vLLM HTTP 400 "Prompt exceeds max_model_len 4096", 359/411
trials with "No valid JSON object found", one reward 1.0 (step 1). Steps 2-5 in 5184052 overlapped
generation with training (age 1); at lookahead 0 SC step times include the whole rollout, so expect
them longer than 27-129 s.

```bash
L=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
grep -E "train step [0-9]+/3|Total step time|exposed_generation|SC run complete" $L
grep -o "'grad_norm': [^,]*" $L                                           # 0.0 if every group has equal rewards (likely: pretraining ckpt)
grep -oE "'finalize/(invalid_row_rate|capture_poisoned_rollouts|terminal_selection_heuristic_fraction)': [^,]*|'num_mask_sample_filtered': [^,]*" $L
                                                       # heuristic_fraction 1.0 expected (harbor_agent has no terminal_response_id)
grep -c "Prompt exceeds max_model_len" $L                                 # vs 615 in 5184052
grep -o 'max_alloc=[0-9.]*' $L | sort -t= -k2 -g | tail -1                # vs 59.6
grep -o 'max_reserved=[0-9.]*' $L | sort -t= -k2 -g | tail -1             # vs 68.2
grep -iE "no valid response tokens|rollout_failed|RolloutStall|RefitAborted|out of memory|Traceback" $L
grep -l ContextLengthExceededError $HARBOR_ROOT/jobs/*/*/*/*/*/trial.log | wc -l     # CLE trials, vs 125 in 5184052
grep -l 1 $HARBOR_ROOT/jobs/*/*/*/*/*/verifier/reward.txt                 # trials with reward 1
ls $SCRATCH/nemo-rl/logs/<run>/*/gym_token_capture | head                 # *.tokens.intents / .lock by design
```
Compare `invalid_row_rate` and `num_mask_sample_filtered` with the CLE trial count: whether a
rollout that ended in ContextLengthExceeded becomes an invalid (masked) row or keeps its earlier
turns is not settled statically. After step 3 the pump starts one more batch (64 trials), cut off at
shutdown and reaped by `envs/reap_after.sh`.

## Run S: run B on the SFT checkpoint

Run B with a checkpoint that can solve the tasks: Rob's HF export of the last SFT save,
`/capstor/store/cscs/swissai/infra01/hf-checkpoints/chonk-lc-16k-v0.1-ep1/iter_0019374` (`SFT_MODEL_SRC`).
The pretraining checkpoint of run B solved 2 of 192 and 0 of 192 trials (jobs 5256202, 5257203), so
its gradient was zero. main ran the same task with SFT checkpoints on the Gym entrypoint (HANGANG jobs
1210063-1210341: validation 39 -> 43 of 64, train reward 0.61 -> 0.66 over 20 steps).

Config, merged in this order (`common.sh`, `compose_cfg s`):

1. `recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml`: 12 nodes, 8 trainer (Megatron PP2 EP16, CPU
   AdamW) + 4 vLLM (TP1 EP4, FP8), 16384 tokens.
2. `envs/harbor-osb-sft.yaml` without its `defaults:` and `script:` lines: Terminus-2 in the Apertus 2
   message format (`output_format: apertus2`, system prompt, stop token 12, thinking budget 4096,
   6144 new tokens per turn, `interleaved_thinking: false`), no leave-one-out baseline, no overlong
   filtering.
3. `run-b-harbor-sc.yaml`: SC, token capture, 3 steps, as run B.
4. `run-s-sft.yaml`: the staged SFT model, run B's tasks and dataset alias, pod labels from
   `NEMO_GYM_RUN_ID`/`NEMO_GYM_USER` (the reaper's filter), the OpenSandbox key as a file path,
   validation and checkpointing off, `rollout_timeout_s` 3000, stall watchdog 3300 s.

`run-s.sh` runs, on the login node and before the queue wait:

- `setup.sh` with the apertus-common overlay (`PYDEPS_APERTUS2`, overlay `pydeps-apertus2`), as
  `../../run.sh` does for a recipe with `sft_export`, into an EDF of its own
  (`~/.edf/nemo-rl-apertus2-sft.toml`, `SFT_EDF_NAME`): setup.sh bakes the overlay into the EDF, and a
  shared EDF would change under queued jobs of the other runs. It clones `swiss-ai/apertus-common`,
  a private repository, over SSH (`git@github.com:...`, `APERTUS_COMMON_URL` overrides): your SSH key on
  Clariden needs access to it.
- `experiments/grpo-chonk/stage_model.sh` to `$SCRATCH/models/chonk-lc-16k-v0.1-ep1-iter_0019374`: shards
  linked to `/capstor`, small files copied, the modeling code from hfconverter added and
  `config.json` given its `auto_map`.
- run B's Gym and NeMo-RL clone checks.

```bash
RUNS=s bash experiments/chonk-sc/check.sh      # step 0 for run S (default RUNS="a b s")
bash experiments/chonk-sc/run-s.sh              # JOB_TIME default 03:00:00
```

What to look at: reward above 0 and non-zero `grad_norm` (the point of the run), then that capture
rejects nothing (`invalid_row_rate` 0) in the Apertus 2 format. Risks, none tested on SC: Megatron PP2
and the FP8 vLLM weight refit, memory at 16k on 12 GH200 nodes, a turn cut at `max_new_tokens`
without a closed reply frame (`extend_chat_prompt` asserts on the next call, HTTP 500 to Gym).
Research report `2026-11-05-sft-checkpoint-for-harbor-runs.md` (workspace), sections 4 and 5.

### Run S with OpenCode (`AGENT=opencode`)

OpenCode 1.17.11 runs inside each pod and calls the model through the per-pod tunnel, as in
`../harbor-capture-sc` (R2 job 5314745 on Qwen3-4B); run-s.sh adds `opencode-sc.yaml`, `tunnel-sc.yaml`
and `run-s-opencode.yaml` (Gym model-call capture: every request and response body under
`$HARBOR_ROOT/model-calls/<run>`). Needs Gym patches 0016-0017 and 0023-0029 (setup-gym.sh) and chisel
(fetched to `CHISEL_DIR` on first use). Check B (job 5315873) encoded OpenCode's first request in the
Apertus 2 format: 5587 tokens, so ~10.8k of 16384 remain for the trajectory.

```bash
RUNS=o bash experiments/chonk-sc/check.sh        # step 0 (without the smoke overrides)
# smoke: 16 samples, the smallest batch (trainer data-parallel size 16)
AGENT=opencode bash experiments/chonk-sc/run-s.sh grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=8 grpo.max_num_steps=1 policy.train_global_batch_size=16
```

What to look at: `train step 1/1`, 0 x 422 and 0 unresolved parents in ray-driver.log, masked rollouts,
per-turn prompt and generation lengths (model-call capture), context overflows (HTTP 400 from the
chat endpoint), and whether long generations survive the tunnel (OpenCode's `step_finish` ->
`step_start` gaps in `agent/opencode.txt`).

### Run S with OpenCode on SWE-Lego (`AGENT=opencode TASKS=swe-lego`)

Harder tasks: Ambroise's SWE-Lego v05 Harbor tasks (real Python repositories, graded by the task's
FAIL_TO_PASS + PASS_TO_PASS tests), instead of the micro tasks. `run-s-swe-lego.yaml` runs OpenCode in
`/testbed` (where the tasks keep the repository) and ends it at 1740 s, under the 1800 s agent cap
(the tasks allow 3000 s). `HARBOR_ROOT` defaults to `$SCRATCH/osb/swe_lego_v05`; `JOB_TIME` to 04:00:00.

**Pull credentials.** The image project `swe_lego_v05_1` is no longer public: since about 2026-10-29, every
anonymous pull gets a 401 (job 5317993: all pods Pending). run-s.sh therefore needs `RL_REGISTRY_CRED_FILE`
(default `~/.rl_registry_cred`), a Harbor robot account with pull access as a single `name:password` line, chmod 600.
Gym patch 0013 (`image_auth_file`) reads that file when it creates a sandbox, and 0031 (`image_auth_registry`) sends the credentials only to `rl-registry.swissai.svc.cscs.ch`.
So the password is neither in the config nor in the logs, and never goes to Docker Hub.
Check the account first: `bash experiments/osb-reach/registry_check.sh`. Its line `4. C ... manifest with token -> HTTP 200`
means the account can pull. The token request returns 200 even for wrong credentials.

```bash
# once: keep the tasks whose reference solution scores 1 and whose empty run scores 0
python3 envs/prepare_swe_lego_tasks.py /capstor/store/cscs/swissai/infra01/users/aborbely/swe_lego_v05/tasks \
    $SCRATCH/osb/swe_lego_v05
RUNS=w bash experiments/chonk-sc/check.sh
# probe: 8 tasks x 8, one step (solve rate, context overflows, time per trial)
AGENT=opencode TASKS=swe-lego bash experiments/chonk-sc/run-s.sh grpo.max_num_steps=1
```

Watch: the solve rate (the SFT model scored 0.93 on the micro tasks); `maximum context length` 400s and
`ContextLengthExceededError` trials (16384 tokens, first prompt about 5.6k); OpenCode exit 124 (time
limit); whether OpenCode's `bash` finds the repository's Python (the tasks' `test.sh` puts
`/opt/conda/envs/testbed/bin` on PATH itself; UNVERIFIED that the sandbox exec has it); sandbox start
(one 1-1.4 GB image per task).

## Run OV: OpenCode on the verified tasks (`run-agent-v.sh AGENT=opencode`)

Run V's tasks (`../../envs/harbor-verified-sft.yaml`: Kostas' verified tasks, separate verifier pods, images from
rl-registry with the credentials file, gateway, create deadline 1200 s, infra rollouts masked) with OpenCode in each
pod instead of Terminus-2. OpenCode only makes HTTP calls, so its tokens come from Gym's token capture, which only
the SC entrypoint turns on: the run goes through `run-harbor-osb.sh` with `examples/run_grpo_single_controller.py`
(as run O), not run V's Gym entrypoint. Layout of `run-v-24k.sh`: 24576 tokens, 20 nodes (16 trainer EP32 + 4
vLLM), packing micro-batches of 24576 tokens, `VLLM_ALLOW_LONG_MAX_MODEL_LEN`; `CTX=16384` takes the 12-node recipe.
8 prompts x 8 rollouts, at most 64 sandboxes (`SANDBOXES`), synchronous (lookahead 0, as runs S/O; run V's age 2
has no counterpart here). Config: task copy + `run-b-harbor-sc.yaml` `run-s-sft.yaml`
`../harbor-capture-sc/opencode-sc.yaml` `tunnel-sc.yaml` `run-agent-v.yaml` `run-ov.yaml` (`common.sh`
`agent_v_overlays`), plus the launcher's overrides (`agent_v_args`). OpenCode runs in the task image's WORKDIR
(`opencode_workdir` unset: the repository lives in `/testbed`, `/workspace/<repo>`, `/app/<repo>`, `/home/<repo>`, ...
depending on the task), ended at 1740 s under Harbor's 1800 s cap. OpenCode and chisel come from the read-only mounts
Breithorn puts into every sandbox (`/opt/rl-tools/<name>/<version>/`, Gym 0047): nothing is uploaded, setup only checks
that they are there and logs their `--version`. The chisel client on the node is `$CHISEL_DIR`'s linux arm64 binary. Harbor trial dirs and the model-call capture:
`$SCRATCH/harbor-jobs/<run>/` (`model-calls/`). Another in-pod agent is one more `AGENT` value (`common.sh`
`agent_v_tag`, `agent_v_overlays`, `check_gym_agents`; its `run-<tag>.yaml`; `check.py`).

```bash
RUNS=ov bash experiments/chonk-sc/check.sh                    # step 0 (SMOKE=1 to check the smoke values)
SMOKE=1 AGENT=opencode DRY_RUN=1 bash experiments/chonk-sc/run-agent-v.sh   # setup + job script, no submission
SMOKE=1 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh   # 1 step, 4 x 8 = 32 (trainer data-parallel size 32)
AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh           # 10 steps, 8 x 8, JOB_TIME 08:00:00
GATEWAY_PROXY=1 ...       # sandboxes and tunnel through the OpenSandbox server proxy (if the gymtunnel gateway test fails)
REGISTRY_CHECK=pod ...    # before setup also start one test pod (default 1: manifest pull check only; 0: skip)
```

What to look at: `train step N/N`; `failure_<class>` per step (OpenCode writes no `agent_error_flags.json`, so only
`ok`, `infra`, `agent_timeout`, `verifier_timeout` and `other` can occur; a context overflow ends OpenCode and is
`ok` with the verifier's reward); masked and rejected rollouts (capture: unresolved parents, poisoned rows);
HTTP 400 context-length answers in `model-calls/`; OpenCode exit 124 (1740 s); reward per campaign.

## Run PV: pi on the verified tasks (`run-agent-v.sh AGENT=pi`)

Run OV with the [pi coding agent](https://github.com/earendil-works/pi) (v1.5.2) instead of OpenCode: same tasks,
layout, tunnel, token capture and long-context profile; config `../harbor-capture-sc/pi-sc.yaml` and `run-pv.yaml`
in place of `opencode-sc.yaml` and `run-ov.yaml`. Gym patch 0048 (`PiNemoGym`): pi runs from the release tree Breithorn
mounts into every sandbox (`/opt/rl-tools/pi/1.5.2/pi/pi`, Gym 0049; setup logs its `--version`). pi runs with compaction, retries,
timeouts, extensions, MCP and skills off and only `read`, `bash`, `edit`, `write`; `max_turns` is a pi extension that
blocks the tool calls of the last allowed turn; the failure classes read pi's events in `agent/pi.txt` (in
`--mode json` pi exits 0 also after an error). pi is ended at `PI_TIMEOUT_SEC` (1740 s; at long context
`AGENT_CAP - 60`). **Untested on the cluster**: the settings come from reading the pi source
(`research/reports/2026-11-24-pi-v1.5.2-source-notes.md` in the workspace); a smoke run first.

```bash
CTX=65536 RUNS=pv bash experiments/chonk-sc/check.sh
SMOKE=1 CTX=65536 AGENT=pi DRY_RUN=1 bash experiments/chonk-sc/run-agent-v.sh
SMOKE=1 CTX=65536 AGENT=pi bash experiments/chonk-sc/run-agent-v.sh      # 1 step, 4 x 8
```

What to look at in the smoke: `agent/setup/` (install), `agent/command-0/` (return code), `agent/pi.txt` (events;
`stopReason` of the last assistant message), `agent/pi-agent/` (the written config), the capture's rejected rows
(each extra or rewritten call masks the rollout), and in `model-calls/` that the requests carry no `store`,
`reasoning_effort` or `developer` role.

## Run CV: Claude Code on the verified tasks (`run-agent-v.sh AGENT=claude-code`)

Run OV with Anthropic's Claude Code CLI (2.1.326) instead of OpenCode: same tasks, layout, tunnel, token capture and
long-context profile; config `../harbor-capture-sc/claude-code-sc.yaml` and `run-cv.yaml` in place of
`opencode-sc.yaml` and `run-ov.yaml`. Claude Code speaks Anthropic Messages to the Gym model server's `/v1/messages`.
Gym patch 0048 (`ClaudeCodeNemoGym`): the native release binary is fetched once and gzipped into `$CC_DIR`
(`$SCRATCH/claude-code`, `fetch-claude-code.sh`; about 170 MB per sandbox, the image needs only gzip) and uploaded
into each sandbox. Claude Code runs with compaction, retries, the non-streaming fallback, sub-agents, web tools, MCP
and the task repository's settings and hooks off, with `Bash`, `Read`, `Edit`, `Write`, `Glob`, `Grep`.
`max_turns` is its own `--max-turns`; the failure classes read its stream-json events in `agent/claude-code.txt`.
Gym patch 0049 is needed with it: the fingerprint compares tool calls against the request's tool schemas (Claude
Code echoes `Edit` with `replace_all: false` filled in), tool-call arguments that are not a JSON object reach it as
`{"_ng_unparsed_arguments": ...}`, and the context-overflow placeholder is answered with a 400 "prompt is too long"
on `/v1/messages`, which ends the episode. Gym patch 0050: Claude Code sends its environment section as a system
message after the first user message, which Apertus 2's format rejects; `vllm_model`'s `late_system_messages_as_user`
sends it as a user message in `<system-reminder>` tags. Check B: the first request is 4678 Apertus 2 tokens (tools
2750; `--bare`: 964), which leaves 11706 tokens at 16K with 6144 per turn. Claude Code is ended at `CC_TIMEOUT_SEC` (1740 s; at long context
`AGENT_CAP - 60`); `CC_PROFILE=bare` swaps its system prompt for `--bare`'s minimal one. Claude Code is closed
source, so the settings were checked by their effect with `../claude-code-probe/` (results in its README).
The one-node rung (`AGENT=claude-code bash ../harbor-capture-sc/run.sh`, Qwen3-4B, micro tasks, Santis) ran: 2x2x1
(job 1060870) and 8x4x3 (job 1061238, 13.7 min, grad norms 1.68, 0.026, 1.46), every model call `POST /v1/messages`
with 200. 5 of the 96 rollouts of the 8x4x3 run were masked, because Claude Code also rewrites `Write` and `Bash`
echoes in ways 0049 does not cover; Gym 0051 turns its rewriting off (below).
Smokes on the SFT models (Santis, 2026-11-26): 16K (job 1061266, v0.1, 2 x 8), 32K (job 1061268, v0.2 at CP 4, 4 x 8;
it also made the one-off v0.2 conversion, in 8 min) and 64K (job 1061276, CP 8 with KCP, 4 x 8, 1137 s per step).
Every install clean, every model call `POST /v1/messages`. None had a non-zero gradient, for two reasons:
- Claude Code refuses to send a prompt of context - max output - 3000 tokens or more and ends with a synthetic
  "Prompt is too long" (`terminal_reason: blocking_limit`, no request; `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` sets the
  limit). With its first prompt of about 4.8K tokens that leaves about 2.4K tokens at 16K and 8.5K at 32K: 8 of 16
  and 28 of 32 trials ended there (masked as context overflows at 32K). At 64K 1 of 32 did; 24 ended on their own
  (up to 97 calls, prompts up to 46K tokens) and 6 scored 1.
- The echo rewrites mask most long rollouts on these models, which start most commands with `cd /workspace && `:
  12 of 32 at 32K and 19 of 32 at 64K (29 commands echoed without the `cd`), all the rewarded ones at 64K among them.
  Gym patch 0051 sets the undocumented `CLAUDE_CODE_HUMBLE_HAMMOCK=1`, with which Claude Code echoes every tool call
  as the model sent it (probe scenario `rewrites`, `../claude-code-probe/README.md`).
With 0051 the 64K smoke (job 1061395, 1092 s per step) trained: grad norm 0.114, 21 of 32 sequences valid, mean
reward 0.14 among them (4 of 8 solved on one task). No echo differed from its call (947 tool calls). Masked: the 8
FACET trials (agent timeout) and 2 rollouts in which a reply stopped at `max_tokens` after 326 tokens (of 16384) and
Claude Code re-sent the request with "Output token limit hit. Resume directly ..." appended, which continues no call.
Those stops are not a token limit: NeMo-RL's Apertus 2 output parser finishes a reply with neither text nor a usable
call as `length` (`apply_apertus2_output_to_chat_response_choices`), about 30 per 64K smoke (3% of calls; the model
ended a turn inside its reasoning, or wrote nothing after it). Claude Code drops such a reply and re-sends the
request, so the reply becomes a dead branch. After an empty text reply the re-sent request resolved to the call
before; after a reply of reasoning alone it matched both calls (ambiguous), because reasoning is not hashed. Gym
patch 0052: a reasoning item opens a turn, so such a reply hashes the empty-turn marker like an empty one.
`run-agent-v.sh` warns when `AGENT=claude-code` runs below `CTX=65536`.
The v0.1 model often names Claude Code's tools in lowercase (`bash`, `write`), which Claude Code rejects as unknown
tools; v0.2 does not. The task with `timeout_sec = 120` in its `task.toml` (FACET) times out in all 8 trials.

```bash
CTX=65536 RUNS=cv bash experiments/chonk-sc/check.sh
SMOKE=1 CTX=65536 AGENT=claude-code DRY_RUN=1 bash experiments/chonk-sc/run-agent-v.sh
SMOKE=1 CTX=65536 AGENT=claude-code bash experiments/chonk-sc/run-agent-v.sh      # 1 step, 4 x 8
```

What to look at in the smoke: `agent/setup/` (install time and the in-pod sha256 check), `agent/command-0/` (return
code; 1 after an API error), `agent/claude-code.txt` (the `result` event's `terminal_reason`), the capture's
rejected rows (each extra or rewritten call masks the rollout; roots 1, chains 1 per rollout), and in
`model-calls/` that every request is `POST /v1/messages` with one system prompt per rollout.

## Long context: runs V and OV at 64K (`CTX=65536`)

Plan `research/reports/2026-11-20-long-context-agent-runs-plan.md` (workspace), user decisions 2026-11-20. Both
launchers take the same `CTX`-driven profile (`common.sh` `long_ctx_defaults` / `long_ctx_args`). Up to 24576 tokens
nothing changes; above it the defaults are:

| Knob | 24k (unchanged) | > 24576 (64K) | What it sets |
|---|---|---|---|
| `SFT_CKPT` | v0.1 iter_0019374 | v0.2 iter_0018315 | recipe (`recipes/apertus2-chonk-sftv02-alps-vllm-adamw-tools-20n.yaml`, 20 nodes), staged dir, `policy.model_name` |
| `THINKING` | medium (task file's frame) | high | `policy.generation.system_prompt_file=envs/apertus2-system-high.json`; no thinking budget |
| `CP` / `KCP` | 1 / 0 | CTX/8192 = 8 / 1 (CP > 4) | `context_parallel_size`, `make_sequence_length_divisible_by` 2 x CP, `linear_attention_cp_impl=kcp`; trainer DP 32/CP = 4 |
| `CAP` | none | 0.74 | `NRL_CUDA_MEMORY_FRACTION` (trainer) |
| `NCCL_MIN` | none | 60 (from 65536) | `NRL_NCCL_TIMEOUT_MINUTES` (NeMo-RL 0040) |
| `MAX_NEW` | 6144 | 16384 | `max_new_tokens` per turn (model_info, OpenCode's max_tokens follow) |
| `MAX_TURNS` | 30 | 100 | `harbor_agent_kwargs.max_turns` (OpenCode: steps) |
| `AGENT_CAP` | 1800 | 3000 | `harbor_agent_max_timeout`; OV: OpenCode `AGENT_CAP - 60` (2940), rollout/stall timeout `+4800` / `+5100` (7800/8100) |
| `TTL` | 3600 | 4800 | `harbor_environment_kwargs.sandbox_ttl_s` |
| `MASK_AGENT_TIMEOUT` | 0 | 1 | `harbor_mask_agent_timeout` (rule ii; on run V's Gym entrypoint NeMo-RL 0057 keeps masked rows out of the group baseline) |
| `MASK_OVERFLOW` | 0 | 1 | `harbor_mask_context_overflow` (rule i, Gym 0043); `check_gym_capture` requires Gym 0043 when 1 |
| `HANG_DUMP_AFTER_S` / `HANG_KILL_AFTER_S` | 900 / 2700 | 3600 / 14400 (from 65536) | hang watchdog (exported; reaching the job on Alps UNVERIFIED) |
| `STEPS` / `JOB_TIME` | 20 (V), 10 (OV) / 08:00:00 | 10 / 12:00:00; `SMOKE=1`: 1 step, 4 x 8, 03:00:00 | Daint's time limit for `normal`: UNVERIFIED |

Every knob can be set at any context (e.g. `THINKING=high` at 24k). At long context the job reads a copy of the recipe
(`$SCRATCH/chonk-sc/recipes/<run>.yaml`), and with `KCP=1` the launchers point setup and job at the KCP stack
(`kcp_setup`): pin copy with the `rl-kcp` commit and a container spec with fla 0.5.2 on `PYTHONPATH` in
`$SCRATCH/chonk-sc/kcp/`, and an EDF of their own (`nemo-rl-apertus2-kcp` for V, `nemo-rl-apertus2-sft-kcp` for OV).
The checkpoint export is linked whole when it carries `MODELING_SOURCE`, else its small files are copied (both
launchers, the same directory). The first run converts v0.2 to Megatron, and **that first conversion must not run
at CP 8**: NeMo-RL's `import_model_from_hf_name` does not apply `megatron_cfg.model_overrides`, so the conversion
builds the model without `linear_attention_cp_impl=kcp` and fails the a2a check `linear_num_key_heads // TP (28)
must be a multiple of CP (8)`. Convert once at CP 4 (`SMOKE=1 CTX=32768 NCCL_MIN=60 HANG_DUMP_AFTER_S=3600
HANG_KILL_AFTER_S=14400 bash experiments/chonk-sc/run-v-24k.sh`, ~30 min). The conversion directory
`$HF_HOME/nemo_rl/model_<model path with / as _>` depends only on the model path (not on CP, CTX or the overrides),
so every later run at CP 8 reuses it; the training model itself does get the overrides. Check it with
`ls "$SCRATCH/huggingface/nemo_rl/"*<checkpoint name>*` (`iter_0000000`, `latest_checkpointed_iteration.txt`).

```bash
CTX=65536 RUNS=ov bash experiments/chonk-sc/check.sh          # step 0 for both runs (SMOKE=1 for the smoke values)
SMOKE=1 CTX=32768 NCCL_MIN=60 HANG_DUMP_AFTER_S=3600 HANG_KILL_AFTER_S=14400 bash experiments/chonk-sc/run-v-24k.sh  # once: conversion at CP 4
SMOKE=1 CTX=65536 DRY_RUN=1 bash experiments/chonk-sc/run-v-24k.sh        # run V smoke: setup + job script only
SMOKE=1 CTX=65536 bash experiments/chonk-sc/run-v-24k.sh                  # run V smoke: 1 step, 4 x 8, 03:00:00
CTX=65536 bash experiments/chonk-sc/run-v-24k.sh                          # run V: 10 steps, 8 x 8, 12:00:00
SMOKE=1 CTX=65536 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh # run OV smoke
```

### KCP on Alps (`experiments/long-context/setup-kcp.sh`, once)

`setup-kcp.sh` clones the pinned Megatron into `$PROJECTS_DIR/megatron-lm-moe-apertus2-kcp`, checks out branch
`rl-kcp` at the pin's commit and applies `experiments/long-context/kcp/*.patch` (plain git, login node), then installs
flash-linear-attention 0.5.2 (`fla.ops.cp`) with `uv pip install --no-deps --target $PROJECTS_DIR/fla-0.5.2` for
python3.12 if present, else python3.14 (Alps login nodes have no python3.12). `$PROJECTS_DIR` comes from `sites/alps.env`
(`/users/$USER/projects`) unless `site.env` changes it; `/users` is mounted in the container.

```bash
cd ~/rl-infra && export PROJECTS_DIR="$(bash -c 'source sites/load.sh && rl_infra_load && echo "${PROJECTS_DIR}"')"; echo "${PROJECTS_DIR}"
command -v uv python3.14                                     # uv on the login node: UNVERIFIED
bash experiments/long-context/setup-kcp.sh
ls "${PROJECTS_DIR}/fla-0.5.2/fla/ops/cp" && git -C "${PROJECTS_DIR}/megatron-lm-moe-apertus2-kcp" log --oneline -2 rl-kcp
```

The container question: the job imports fla with the image's Python 3.12 from `PYTHONPATH`, while the wheels were
chosen for the login node's 3.14. With `--no-deps --target` only the wheel files of fla itself land in the directory,
so this is fine if flash-linear-attention 0.5.2 and fla-core 0.5.2 are pure-Python wheels (`py3-none-any`;
UNVERIFIED) and accept 3.14 in `Requires-Python` (UNVERIFIED). Check what was installed:

```bash
ls "${PROJECTS_DIR}"/fla-0.5.2/*.dist-info/ && grep -h '^Tag:' "${PROJECTS_DIR}"/fla-0.5.2/*.dist-info/WHEEL   # want py3-none-any
```

If the tag is not `py3-none-any`, or uv refuses 3.14 or is missing, delete `$PROJECTS_DIR/fla-0.5.2` and install for the
container's interpreter instead, either (a) from the login node, resolving for Python 3.12 on aarch64
(`uv pip install --python python3.14 --python-version 3.12 --python-platform aarch64-unknown-linux-gnu --no-deps
--target "${PROJECTS_DIR}/fla-0.5.2" flash-linear-attention==0.5.2 fla-core==0.5.2`; these uv flags with `--target`:
UNVERIFIED), or (b) inside the image on a compute node (pip in the image and internet from compute nodes: UNVERIFIED;
the EDF `nemo-rl-apertus2` exists after any earlier `setup.sh`):

```bash
srun -A "${ACCOUNT}" -p normal -N 1 -t 00:20:00 --environment=nemo-rl-apertus2 bash -c \
  "python3 -m pip install --no-deps --target ${PROJECTS_DIR}/fla-0.5.2 flash-linear-attention==0.5.2 fla-core==0.5.2"
```

Either way, an import check in the image costs one short allocation and catches a wrong install before a 20-node job:

```bash
srun -A "${ACCOUNT}" -p normal -N 1 -t 00:10:00 --environment=nemo-rl-apertus2 \
  python3 -c "import sys; sys.path.insert(0, '${PROJECTS_DIR}/fla-0.5.2'); import fla, fla.ops.cp; print(fla.__version__, fla.__file__)"
```

`CTX=65536 RUNS=ov bash experiments/chonk-sc/check.sh` then shows `ok` for the KCP Megatron and fla 0.5.2 (warnings
section), and the launchers stop before setup when either is missing.

## Open risks (from the feasibility report, ranked)

1. SC split-train path with Megatron EP32 + UCCL/DeepEP + CPU-offloaded distributed AdamW: first
   ever (no upstream SC test with MoE EP + CPU optimizer). Watch grad_norm, NCCL hangs, OOM. (A)
2. #2517 `-Xtheirs` merge vs f196 refit code: static markers in step 0; refit through
   `CollectiveWeightSynchronizer` over 32 + 16 ranks only shows in a run. (A)
3. Token capture with 16 vLLM DP workers on 4 nodes; cross-node `flock` on the scratch ledger dir
   unverified (1-node only so far, job 5236250). (B)
4. ContextLengthExceeded / malformed-JSON turns under capture (invalid-row rate; signal kept or lost). (B)
5. Trainer memory on SC at 4096 (should match 59.6 / 68.2 GB; TQ worker mixin overhead unknown);
   streaming would keep grad buffers resident during the next chunk's log-prob pass, hence one chunk
   per step. 8192 OOMs on 12 nodes (5186694): use the 20n recipe for that (run D).
6. TransferQueue simple backend at 12 nodes / 24 units; host RAM on the CPU-offload trainer nodes.
7. If all 64 rows of a step are invalid (e.g. an OpenSandbox outage), SC raises "no valid response
   tokens after filtering" where legacy + mask_degenerate_rollouts trained a zero-loss step; Harbor trials failing
   on infra are scored 0 and not retried by NeMo-RL (job 5245999).

Next (not prepared here): run C = run B with `async_rl.sampler.max_lookahead_versions=1`,
`max_inflight_prompts`/`max_buffered_rollouts` 16 and harbor concurrency 128, 5+ steps; run D =
8192 tokens on `recipes/apertus2-chonk-vllm-adamw-20n.yaml`.

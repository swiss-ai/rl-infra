# Apertus 2 Chonk on NeMo-RL's Single-Controller path (step 0, run A, run B)

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
| `common.sh` | config composition and the clone / Gym checks, shared by the three scripts |

Both runs: 3 steps, `async_rl` sampler `in_order` with lookahead 0 (synchronous), one chunk per
step (`min_groups_for_streaming_train` = `max_inflight_prompts` = `max_buffered_rollouts` =
prompts per step), TransferQueue data plane with 24 storage units (`${mul:2, num_nodes}` from
`grpo_math_1B.yaml`), generation fleet health off (its rebuild path ignores vLLM DP,
`weight_sync/membership.py`), validation and checkpointing off, stall watchdog `abort`,
`DRIVER_RETRIES=0` (SC writes no `train_data_step*.jsonl`, so `driver_retry.sh` would not retry).

## Daint

```bash
cd ~/rl-infra                                  # this branch: chonk-sc
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
  matches), Gym patch 0011 (`harbor_model_base_url` / `base_url_for_run`: without it capture records
  nothing and every row is rejected, job 5236016), 0012 (`_should_mask`, capture-aware masking),
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

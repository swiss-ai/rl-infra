# NeMo-RL GRPO: architecture map (sync colocated + async non-colocated)

This document describes how NeMo-RL runs GRPO, as a basis for drawing diagrams. It covers two setups:

- **Setup A, sync + colocated**: our working run (Qwen3-32B, 4 nodes x 4 GH200, job 3606854).
- **Setup B, async + non-colocated**: the planned setup, modelled on NVIDIA's recipe
  `examples/configs/recipes/llm/performance/grpo-qwen3-32b-8n4g-async-1off.yaml`.

Versions: NeMo-RL commit `19244a0` (branch `r0.7.0`), vLLM `0.20.0`, math-verify `0.9.0`.
All `nemo_rl/...` links are GitHub permalinks to that commit, so line numbers match exactly. Our local clone
(`~/projects/nemo-rl`, branch `alps`) is that commit plus one patch to `ray.sub` only.

How to use this for diagrams: each "Diagram" section lists **boxes** (id, title, what to write inside) and
**arrows** (from -> to, label, code link). Sections 7-11 are reference material for annotations.

---

## 1. Vocabulary

| term | meaning |
|---|---|
| **Driver** | The process running `examples/run_grpo.py` on the head node. Plain Python + Ray client, no GPU. Holds the config, the dataloader, the driver-side wrapper objects, and runs the training loop. Not a Ray actor. |
| **Ray actor** | A Python process started by Ray on some node, with a handle the driver calls with `.remote(...)`. Every model worker is one actor (one per GPU). |
| **Driver-side wrapper** | A plain Python object living in the driver that owns a `RayWorkerGroup` of actors and fans calls out to them: `Policy` (trainer side), `VllmGeneration` (inference side). |
| **RayWorkerGroup** | Creates actors into placement-group bundles (1 bundle = 1 GPU), sets `RANK/WORLD_SIZE/MASTER_ADDR`, and offers dispatch helpers (`run_all_workers_single_data`, `run_all_workers_sharded_data`, ...). |
| **RayVirtualCluster** | Ray placement groups with one bundle per GPU. Colocated: one cluster shared by trainer and vLLM. Non-colocated: separate train and inference clusters. |
| **Environment actor** | Reward computation. `MathEnvironment` is itself a Ray actor (the coordinator) owning CPU-only verify workers. The driver maps `task_name -> environment actor` in `task_to_env`. |
| **GCS / raylet** | Ray's control plane: GCS on the head node, one raylet per node. Actor creation goes driver -> GCS -> raylet of the chosen node, which starts the process. Ray resources (`num_gpus=0.5`) are logical counters only; GPU memory is not managed by Ray. |
| **Refit** | Copying the trainer's updated weights into vLLM. |

Code: driver entry [run_grpo.py](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/examples/run_grpo.py) ·
[`setup()` grpo.py:L254](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L254) ·
[`RayWorkerGroup` worker_groups.py:L316](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/distributed/worker_groups.py#L316) ·
[actor creation `_create_workers_from_bundle_indices` L425](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/distributed/worker_groups.py#L425) ·
[`num_gpus = 1/max_colocated_worker_groups` L557](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/distributed/worker_groups.py#L557)

---

## 2. Diagram 1: Setup A topology (sync, colocated, 4 nodes x 4 GPUs)

Run parameters: Qwen3-32B; trainer = DTensor v1 worker (HF transformers model, FSDP2 x TP2, so DP=8);
vLLM TP2 (8 engines), `gpu_memory_utilization=0.6`; 64 prompts x 32 generations per step; math environment.

### Boxes

| id | title | text inside |
|---|---|---|
| D | Driver (`run_grpo.py`, head node, CPU) | config; `StatefulDataLoader` (OpenMathInstruct-2, 64 prompts/batch); `Policy`; `VllmGeneration`; `task_to_env`; `ClippedPGLossFn`; runs `grpo_train()` |
| P | `Policy` (driver-side wrapper) | `RayWorkerGroup` of 16 trainer actors; sharding `[PP, DP, CP, TP] = [1, 8, 1, 2]`; sequence-packing settings |
| T | `DTensorPolicyWorker` x16 (one per GPU) | HF `Qwen3ForCausalLM`; FSDP2 (DP=8) x TP2; fp32 master weights + AdamW; reference-model CPU copy (KL on); ZMQ socket `ipc:///tmp/<gpu-id>.sock` |
| G | `VllmGeneration` (driver-side wrapper) | `RayWorkerGroup` of 8 engine actors (one per TP group of 2 GPUs) |
| E | `VllmGenerationWorker` x8 (engine actors) | owns a sync `vllm.LLM`; sleep mode on; starts 2 vLLM GPU workers through Ray |
| V | vLLM GPU worker x16 (`RayWorkerWrapper`) | model shard (TP2), KV cache; NeMo-RL extension `VllmInternalWorkerExtension` (receives refits) |
| M | `MathEnvironment` (Ray actor, CPU) | coordinator; `max_concurrency=1000`; splits responses over workers |
| W | `HFVerifyWorker` x8 (Ray actors, CPU) | math-verify: extract answer, compare symbolically to ground truth, return 1.0 / 0.0 |
| GPU | each of the 16 GPUs | hosts one T and one V (each `num_gpus=0.5`); memory time-shared via vLLM sleep/wake + trainer offload |

### Arrows

| from -> to | label | code |
|---|---|---|
| D -> P | `init_policy()` creates 16 trainer actors | [grpo.py:L858](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L858), [lm_policy.py RayWorkerGroup](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/lm_policy.py#L263) |
| D -> G | `init_vllm()` creates engines, then puts vLLM to sleep | [grpo.py:L873](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L873), [vllm_generation.py RayWorkerGroup](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_generation.py#L232) |
| E -> V | Ray executor starts 2 GPU workers per engine | [vllm_worker.py:L293](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker.py#L293) |
| D -> M | `create_env` during data setup; stored in `task_to_env` | [data/utils.py:L158](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/data/utils.py#L158), [L189](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/data/utils.py#L189) |
| M -> W | creates `num_workers=8` verify workers | [math_environment.py:L359-363](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/math_environment.py#L359-L363) |
| D -> G -> E -> V | `generate(...)` each step | section 4, phase 3 |
| D -> M -> W | `step(...)` / `verify(...)` each step | section 4, phase 4 |
| D -> P -> T | `get_logprobs`, `train`, offload/onload calls | section 4 |
| T(i) -> V(i), same GPU | refit: ZMQ + CUDA IPC, one-to-one per GPU | section 6a |
| T <-> T | FSDP all-gather / reduce-scatter, TP all-reduce (trainer's own process group) | [dtensor_policy_worker.py mesh L386-401](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L386-L401) |

Setup details: colocated cluster [grpo.py:L585-597](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L585-L597)
(`max_colocated_worker_groups=2` for vLLM generation). vLLM is created first, then the policy, sequentially
(`use_parallel_init = not colocated`). The refit table (`{param name: (shape, dtype)}`) is sent once:
[grpo.py:L1175-1177](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1175-L1177).

Optional box (not used by us): **NeMo-Gym**, one `NemoGym` Ray actor
([nemo_gym.py:L146](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/nemo_gym.py#L146))
that starts its head/agent/resource servers as OS subprocesses on its node and calls vLLM over HTTP
(`dp_openai_server_base_urls`). Requires the async vLLM engine. Not studied in depth yet.

---

## 3. Diagram 2: Setup B topology (async, non-colocated)

NVIDIA's 8n4g recipe: 4 nodes (16 GPUs) for training, 4 nodes (16 GPUs) for vLLM with TP1, so 16 engines,
each using vLLM's `AsyncLLM`. The recipe uses the Megatron trainer; with DTensor the topology is the same.

### Boxes (in addition to / replacing Diagram 1)

| id | title | text inside |
|---|---|---|
| D | Driver | runs `async_grpo_train()`; samples batches from the replay buffer; triggers train + refit |
| T | trainer actors x16 (train nodes only) | as Diagram 1, but always resident on GPU (no sleep/offload dance) |
| E/V | vLLM engines x16 (inference nodes only) | `AsyncLLM`; generate one request per sample; accept weight updates between decode steps |
| C | `AsyncTrajectoryCollector` (Ray actor) | pulls prompts from the dataloader; launches one thread per prompt group (32 samples); runs rollouts + rewards; pushes finished groups to the buffer; throttles by target step |
| R | `ReplayBuffer` (Ray actor) | stores per-prompt groups tagged with (generation weight version, target step); evicts too-old groups; `sample()` returns exactly 64 groups meant for the current step |
| M/W | MathEnvironment + verify workers | unchanged; called per sample from the collector |
| N | NCCL `model_update_group` | members: 16 trainer ranks (0..15) + 16 vLLM ranks (16..31); src = trainer rank 0 |

### Arrows

| from -> to | label | code |
|---|---|---|
| D -> R | create buffer, `max_size = num_prompts x max_age x 2` | [grpo.py:L3468-3474](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3468-L3474) |
| D -> C | create collector, `start_collection(dataloader)` | [grpo.py:L3523-3538](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3523-L3538) |
| C -> E/V | per-sample async generation | [rollouts.py `run_async_multi_turn_rollout` L1104](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/experience/rollouts.py#L1104), [vllm_worker_async.py `generate_async` L996](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker_async.py#L996) |
| C -> M | per-sample `calculate_rewards` (in a thread) | [rollouts.py:L1012-1014](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/experience/rollouts.py#L1012-L1014) |
| C -> R | `add(group, generation_version, target_step)`; backoff if full | [trajectory_collector.py:L796-828](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L796-L828) |
| D -> R | `sample(64 groups, current version, max_age)` | [grpo.py:L3667-3672](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3667-L3672) |
| D -> T | `policy.train(...)` | [grpo.py:L3966](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3966) |
| D -> C | `prepare_for_refit` / `set_weight_version` / `resume_after_refit` | [grpo.py:L3978](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3978), [L3998-4000](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3998-L4000) |
| T(rank 0) -> all of N | refit: packed NCCL broadcast | section 6b |

Requirements for async (all enforced in code): non-colocated
([grpo.py:L3421](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3421)),
async engine ([L3364](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3364)),
`use_importance_sampling_correction: true`
([L3371](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3371)),
no dynamic sampling / reward scaling / reward shaping / multiple dataloaders
([run_grpo.py:L159-186](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/examples/run_grpo.py#L159-L186)).

---

## 4. Diagram 3: one sync GRPO step (Setup A), as a timeline

Everything below is orchestrated by the driver inside
[`grpo_train` grpo.py:L2131](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2131).
Trainer workers only run `get_logprobs` and `train`; generation requests, rewards, baselines and advantages are
handled by the driver.

| # | phase | who / what | code | GPU memory after the phase (per GPU) |
|---|---|---|---|---|
| 1 | Batch | driver takes 64 prompts, `repeat_interleave(32)` -> 2048 rows | [L2264](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2264) | trainer weights + optimizer resident (from last step) |
| 2a | Refit start | trainer: optimizer -> CPU (`offload_before_refit`) | [L1876](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1876), [worker L1952](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1952) | trainer fp32 weight shard ~8 GB |
| 2b | Wake vLLM weights | vLLM re-maps memory, copies weights back from pinned CPU | [L1880](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1880) | + vLLM weights 30.6 GiB |
| 2c | Transfer | trainer -> vLLM on same GPU, ZMQ + CUDA IPC (section 6a) | [L1924-1927](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1924-L1927) | + staging buffer (30% of free memory) |
| 2d | Trainer out | trainer weights -> CPU (`offload_after_refit`) | [L1962](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1962), [worker L1963](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1963) | trainer 0 GB |
| 2e | Wake KV cache | empty KV cache allocated | [L1966](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1966) | vLLM = 0.6 x 95 GB (weights + 22.6 GiB KV) |
| 3 | Generate | `run_multi_turn_rollout` -> `generate_responses` -> `VllmGeneration.generate` (splits 2048 over 8 engines) -> worker `generate` -> `llm.generate` | [L2389](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2389), [rollouts.py:L231](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/experience/rollouts.py#L231), [vllm_generation.py:L623](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_generation.py#L623), [vllm_worker.py:L633](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker.py#L633), [L685](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker.py#L685) | same |
| 4 | Rewards | `calculate_rewards`: group by task -> `MathEnvironment.step` -> 8 x `HFVerifyWorker.verify` (256 samples each) | [rollouts.py:L461](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/experience/rollouts.py#L461), [L693](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/experience/rollouts.py#L693), [math_environment.py:L422](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/math_environment.py#L422), [L76](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/math_environment.py#L76) | same |
| 5 | vLLM sleep | `finish_generation` -> `sleep(level=1)`: weights copied to pinned CPU, KV discarded | [L2400](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2400), [vllm_worker.py:L1009](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker.py#L1009) | ~0 GB |
| 6 | Reward post-processing | reward scaling / shaping (off); per-prompt baseline + std for metrics and dynamic sampling (off) | [L2413-2473](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2413-L2473) | same |
| 7a | Logprob prep | trainer weights -> GPU (`prepare_for_lp_inference`) | [worker L1914](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1914) | trainer weight shard |
| 7b | `prev_logprobs` | forward pass over all 2048 rows with current weights (pi_old) | [L2628](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2628), [lm_policy.py:L498](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/lm_policy.py#L498) | + unsharded model during the pass |
| 7c | Reference logprobs (KL > 0 only) | swap in the CPU reference copy, forward, swap back | [L2642](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2642), [worker `use_reference_model` L1739](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1739) | same |
| 8 | Advantages | leave-one-out baseline / std per prompt (section 8) | [L2685-2721](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2685-L2721), [advantage_estimator.py:L43](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/advantage_estimator.py#L43) | same |
| 9a | Training prep | optimizer -> GPU (`prepare_for_training`) | [L2726](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2726) | ~33 GB persistent (see below) |
| 9b | Train | `policy.train`: shard 2048 rows into 8 DP shards of packed bins -> each worker: 4 minibatches (optimizer steps) x packed microbatches -> loss -> backward -> clip grad norm 1.0 -> `optimizer.step`; LR scheduler steps once per GRPO step | [L2731](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2731), [lm_policy.py:L702](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/lm_policy.py#L702), [worker `train` L579](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L579) | ~33 GB + ~5 GB transient |
| 10 | Validation / checkpoint / logging | every `val_period` / `save_period` steps | [L2758-3100](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2758) | |

Refit runs only when the vLLM weights are stale (`NEED_REFIT and POLICY_GENERATION_STALE`). Otherwise the
`else` branch offloads the trainer and wakes vLLM fully:
[L2324-2326](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2324-L2326).

Measured step 1 (job 3606854): total 602.75 s = training 289.94 s + generation 191.11 s (incl. rewards, at most
~9-13 s) + policy and reference logprobs 106.46 s + prepare_for_generation 12.61 s (of which weight transfer
5.22 s).

Trainer memory per GPU while training (2.05B params per GPU): fp32 master shard 8.2 GB + fp32 grads 8.2 GB +
AdamW moments 2 x 8.2 GB = ~33 GB persistent, plus a bf16 gathered layer, checkpointed activations and fp32
logits transient. About 40 of 95 GB used.

Host (CPU) memory: vLLM sleep backs up 30.6 GiB per GPU (~122 GiB per node); offloaded trainer weights and
optimizer; pinned reference-model copy (KL on).

---

## 5. Diagram 4: async step (Setup B, `max_trajectory_age_steps = 1`)

Draw two lanes (trainer GPUs, vLLM GPUs) over time.

Rules:
- **Weight version** = number of refits so far. Starts at `step`
  ([grpo.py:L3408](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3408)),
  +1 after every training step's refit
  ([L3998](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3998)).
- **Targets**: vLLM holding weights v only generates for steps v+1 ... v+max_age
  ([trajectory_collector.py:L149](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L149)).
  Exception: the initial version may also target itself, so step 1 is on-policy
  ([L140-147](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L140-L147)).
- **Age window**: training step k accepts groups whose generation version is in [k - max_age, k]
  ([replay_buffer.py:L128-150](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/replay_buffer.py#L128-L150)),
  and only groups whose target is k
  ([L166-185](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/replay_buffer.py#L166-L185)).
  It stalls until exactly 64 are ready. With age 1: step k trains on data generated by weights k-1.
- **Throttle**: at most `num_prompts x max_age` groups in flight
  ([L103-109](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L103-L109));
  the collector pauses when every allowed target is generated or in progress
  ([L202-230](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L202-L230)).
  So with age 1, vLLM works at most one batch ahead and then idles.

Event sequence for step k:
1. Trainer trains step k on batch B_k (generated by weights k-1). In parallel, vLLM (weights k) generates
   B_{k+1}: one thread per prompt group, 32 samples each, each sample scored by the environment as soon as it
   finishes; finished groups are pushed to the buffer.
2. Trainer finishes. `prepare_for_refit` pauses new generation starts
   ([trajectory_collector.py:L462-512](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L462-L512)).
   - without in-flight updates: waits for running generations to finish (drain)
   - with `in_flight_weight_updates: true` (async engine): does not wait
3. Refit over NCCL (section 6b). Trainer is idle during it. With in-flight updates, vLLM applies the update
   between two decode steps; running requests continue with the new weights, so one response can mix weights
   k and k+1.
4. Version -> k+1, collector resumes and starts B_{k+2}
   ([grpo.py:L3998-4000](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3998-L4000)).
5. Trainer samples B_{k+1} (waits for its stragglers if needed) and trains step k+1.

Why in-flight updates help: the refit no longer waits for the longest response, and B_{k+2} can start while
B_{k+1}'s tail is still finishing. They only help when generation is the bottleneck. `max_age > 1` only helps
together with in-flight updates
([grpo.py:L3383-3388](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3383-L3388)).

KV cache after an in-flight update: default keeps the stale KV ("Magistral-style");
`recompute_kv_cache_after_weight_updates: true` is meant to invalidate it ("AReaL-style",
[trajectory_collector.py:L519-536](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/async_utils/trajectory_collector.py#L519-L536)),
but see section 12.

Where vLLM checks for the update: vLLM's engine-core loop alternates "drain input queue (new requests, aborts,
utility calls such as our weight update)" and "one decode step for all running requests"
([vLLM core.py:L1164-1170](https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/v1/engine/core.py#L1164-L1170),
[L1280](https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/v1/engine/core.py#L1280)). NeMo-RL sends the update
with `collective_rpc("update_weights_from_collective")`
([vllm_worker_async.py:L1451](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker_async.py#L1451)).

---

## 6. Diagram 5: the two refit mechanisms

### 6a. Colocated (Setup A): ZMQ + CUDA IPC, one-to-one per GPU

- Each trainer worker binds a ZMQ socket `ipc:///tmp/<gpu-id>.sock`; the vLLM worker on the same GPU connects
  ([base_policy_worker.py:L74-90](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/base_policy_worker.py#L74-L90),
  [vllm_backend.py:L140-158](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_backend.py#L140-L158)).
- Sender ([dtensor_policy_worker.py:L1832-1873](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1832-L1873),
  [policy/utils.py `stream_weights_via_ipc_zmq_impl` L361](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/utils.py#L361)):
  for each parameter, `full_tensor()` all-gathers it over FSDP x TP (every trainer GPU gets the full tensor),
  casts to bf16, packs it into a staging buffer; when the buffer half is full, sends a CUDA IPC handle + key
  list, waits for the ACK. Ping-pong: two halves, fill one while vLLM loads the other.
- Buffer size: `policy.refit_buffer_size_gb`, else `NRL_REFIT_BUFFER_MEMORY_RATIO` (default 0.3) x min free GPU
  memory over trainer workers
  ([grpo.py refit_policy_generation L1853](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1853)).
  Ours: halves of ~7.5 GB, ~9 chunks.
- Receiver ([vllm_backend.py `update_weights_via_ipc_zmq` L357](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_backend.py#L357)):
  rebuild the buffer from the handle, slice it by the refit table, `_load_weights`
  ([L332-352](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_backend.py#L332-L352)):
  vLLM's model `load_weights` keeps its TP slice and fuses q/k/v and gate/up.

### 6b. Non-colocated (Setup B, also sync non-colocated): NCCL broadcast from trainer rank 0

Setup, once ([grpo.py:L1136-1173](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1136-L1173)):
- world = all trainer GPUs + all vLLM GPUs; rendezvous address from the train cluster.
- trainer side [base_policy_worker.py:L28-45](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/base_policy_worker.py#L28-L45) (rank = own rank);
  vLLM side [vllm_backend.py:L114-132](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_backend.py#L114-L132) (rank = train_world_size + ...).
- `StatelessProcessGroup`: rank 0 (trainer rank 0) hosts a TCPStore, creates the NCCL id, others read it and join
  ([stateless_process_group.py:L22-45](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/distributed/stateless_process_group.py#L22-L45)).
- Driver launches both sides' joins together and waits on `futures_train + futures_inference` (a rendezvous).
- Separate from the trainers' own FSDP/TP process group. The driver is not a member.

Each refit (driver: [grpo.py:L1942-1948](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1942-L1948)):
- All 16 trainer workers run `broadcast_weights_for_collective`
  ([dtensor_policy_worker.py:L1875-1906](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/workers/dtensor_policy_worker.py#L1875-L1906)):
  per parameter, `full_tensor()` (all-gather), bf16 cast.
- `packed_broadcast_producer`
  ([packed_tensor.py:L39-103](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/utils/packed_tensor.py#L39-L103)),
  two buffers A/B, each with its own CUDA stream:
  1. switch buffer (L66); `streams[i].synchronize()` waits until this buffer's previous broadcast completed (L68)
  2. fill: flatten parameters to bytes until > target size (L76-89); target = `NRL_REFIT_BUFFER_MEMORY_RATIO`
     (default here 0.02) x GPU memory, capped at 5 GB; `NRL_REFIT_NUM_BUFFERS` = 2 (L24-36)
  3. `group.broadcast(packed, src=0)` (L94): queued on the buffer's stream, returns immediately
- vLLM side `packed_broadcast_consumer`
  ([L105-200](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/utils/packed_tensor.py#L105-L200)),
  called from [vllm_backend.py:L446-460](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_backend.py#L446-L460):
  1. switch buffer (L162); synchronize waits until this buffer's previous receive + load are done (L164)
  2. compute the exact size from the refit table in the same order (L172-181)
  3. allocate, receive (L183-186)
  4. unpack by offsets and `_load_weights` (L188-192)
- Back-pressure comes from NCCL: a broadcast completes only when every member has joined it. The trainer can
  refill its buffer once vLLM has *received* it; the next send into that slot waits until vLLM has *loaded* it.
  At most two buffers in flight per side.

### 6c. Known refit inefficiencies (both paths)
- ~700 separate per-parameter all-gathers (tiny norm tensors included); bucketing would cut overhead.
- Every trainer GPU gathers every full tensor, although each vLLM worker keeps only its TP slice.
- NCCL path: trainer ranks 1-15 are group members, so they also receive the broadcast they already have; a
  group of {trainer rank 0 + vLLM ranks} would avoid it.
- Only ~5 s of a ~600 s step today; matters more with frequent refits (async).

---

## 7. Data and batch shapes (Setup A)

| level | size | notes |
|---|---|---|
| prompts per step | 64 | `grpo.num_prompts_per_step` |
| rollout batch | 2048 = 64 x 32 | `repeat_interleave`, one row per (prompt, sample) |
| minibatch | 512 | `train_global_batch_size`; one optimizer step; 4 per GRPO step |
| per DP rank per minibatch | 64 rows | 8 DP ranks |
| microbatch | one packed bin | rows packed into bins of `train_mb_tokens = 4096` tokens (`max_total_sequence_length x train_micro_batch_size`); logprob bins 8192; first-fit-decreasing packing, bin count rounded to a multiple of DP; dummy microbatches keep collectives in sync; samples never split across bins |

Packing uses `cu_seqlens` + per-sample `position_ids` with flash-attention varlen, so samples in one bin don't
attend to each other. Code: [batched_data_dict.py `shard_by_batch_size` L288](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/distributed/batched_data_dict.py#L288).

Per-row fields after generation and rewards: `input_ids` (prompt + response, padded),
`generation_logprobs` (vLLM, response tokens), `token_mask` (1 on response tokens), `sample_mask` /
`loss_multiplier`, `total_reward`, later `prev_logprobs`, `reference_policy_logprobs`, `advantages`.

---

## 8. GRPO math as implemented (Setup A config)

**Rewards**: 1.0 or 0.0 from the math verifier.

**Advantage** (leave-one-out, `use_leave_one_out_baseline: true`, `normalize_rewards: true`), for sample i of a
prompt with 32 samples:

    A_i = (r_i - mean_{j != i} r_j) / std_{j != i}(r_j)      (normalized only where std > 0)

The same scalar is used for every response token. Code:
[utils.py `calculate_baseline_and_std_per_prompt` L103](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/utils.py#L103),
[advantage_estimator.py:L43-82](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/advantage_estimator.py#L43-L82).

**Per-token loss** ([`ClippedPGLossFn.__call__` loss_functions.py:L310](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/loss/loss_functions.py#L310)):

    r_t   = exp(logp_curr - logp_prev)                     PPO ratio, carries the gradient   (L463)
    w_t   = exp(logp_prev - logp_gen)                      importance weight, constant       (L515-517)
                                                           (1 unless use_importance_sampling_correction)
    pg_t  = max(-A r_t, -A clip(r_t, 1 - 0.2, 1 + 0.2))    clipped surrogate (negated: we minimize)
    kl_t  = exp(logp_ref - logp_curr) - (logp_ref - logp_curr) - 1      k3 estimator, beta = 0.01

    loss  = sum_t (w_t pg_t) / T  +  beta * KL-term,   T = valid response tokens in the whole minibatch
                                                       (all-reduced over DP ranks)          (L615-621)

- `logp_gen` = vLLM at sampling time; `logp_prev` = trainer recompute before any update (pi_old, phase 7b);
  `logp_curr` = trainer during the update.
- Decoupled PPO: `w_t * r_t = pi_curr / pi_gen` gives the full correction, but only `r_t` is clipped, so the trust
  region is anchored at `pi_prev` and stale or noisy samples don't trigger clipping. Without `w_t` the gradient is
  biased when `pi_gen != pi_prev` (async, MoE, fp8); negligible in Setup A (gen-KL ~0.0005).
- Token-level loss is the default (`token_level_loss: True`, L105); `False` gives per-sequence means averaged over
  sequences.
- Logged diagnostics: `gen_kl_error`, `token_mult_prob_error`, `sampling_importance_ratio`, `is_oob_ratio`
  (truncation on), `approx_entropy`; async adds `avg_trajectory_age`.
- Optimizer: AdamW lr 3e-7, wd 0.01, grad-norm clip 1.0, LinearLR warmup 13 steps then constant.

---

## 9. Reward path detail (math environment)

- Data: `nvidia/OpenMathInstruct-2`, split `train_1M`, 5% held out as validation. `problem` -> prompt,
  `expected_answer` -> ground truth. Augmented problems' answers are a Llama-3.1-405B majority vote.
- Prompt: `examples/prompts/cot.txt` ("Think step-by-step ... Output your answer inside of \\boxed{} tags.:"),
  Qwen3 chat template, `enable_thinking: False`.
- `step()` joins all assistant text, splits responses + ground truths over 8 workers
  ([math_environment.py:L444-478](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/math_environment.py#L444-L478)).
- `HFVerifyWorker.verify`: wraps the ground truth as `\boxed{gt}`
  ([L107](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/environments/math_environment.py#L107)),
  math-verify extracts an answer from the WHOLE response by prioritized regexes (final-answer phrases and
  `\boxed{}` first, then "answer:", then any LaTeX / number; rightmost match within a priority), converts to sympy,
  compares (string, numeric to 6 decimals, `simplify(a - b) == 0`). Any exception -> 0.0.
- Debug tool: `experiments/grpo-math/debug_math_verify.py` in this repo.

---

## 10. Feature availability: GRPO / DAPO, sync vs async

There is no separate DAPO trainer; DAPO = GRPO with config flags.

| feature | config | sync | async |
|---|---|---|---|
| clip-higher | `loss_fn.ratio_clip_max` | yes | yes |
| token-level loss | `loss_fn.token_level_loss` | yes | yes |
| no KL | `loss_fn.reference_policy_kl_penalty: 0` | yes | yes |
| overlong filtering (truncated -> loss 0) | `grpo.overlong_filtering` | yes ([L2525](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2525)) | yes ([L3781](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L3781)) |
| dynamic sampling | `grpo.use_dynamic_sampling` | yes ([L1257](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1257)) | no |
| reward scaling | `grpo.reward_scaling` | yes ([L1386](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L1386)) | no |
| overlong reward penalty | `grpo.reward_shaping` | yes ([reward_functions.py:L53](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/reward_functions.py#L53)) | no |
| multiple dataloaders | `data.use_multiple_dataloader` | yes | no |
| importance-sampling correction | `loss_fn.use_importance_sampling_correction` | optional | required |

For comparison: the swiss-ai verl fork (branch `1p5-async-rl`) does support dynamic sampling in its fully async
trainer (drop zero-variance groups before the queue, free the slot, pull the next prompt); upstream verl does not.

---

## 11. Backends

| role | options | ours |
|---|---|---|
| trainer | DTensor v1 (HF transformers + FSDP2/TP), DTensor v2 (NeMo Automodel), Megatron (Megatron-Core via Megatron-Bridge) — [lm_policy.py:L110-140](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/policy/lm_policy.py#L110-L140) | DTensor v1 |
| generation | vLLM (sync `vllm.LLM` or async `AsyncLLM`), Megatron inference, SGLang | vLLM sync |
| checkpoints | DTensor: DCP shards (v2 also safetensors); Megatron: dist checkpoints; converters in `examples/converters/` | DCP |

Megatron trainer + colocated Megatron generation skips refit entirely (`NEED_REFIT = False`,
[grpo.py:L2156-2159](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/algorithms/grpo.py#L2156-L2159));
Megatron trainer + vLLM converts to HF names on every refit (`export_hf_weights`).

---

## 12. Open items and known issues

Not covered yet (planned deep dives):
- Exact TP=2 x FSDP=8 sharding of Qwen3-32B and the forward/backward collectives
  ([parallelize.py Qwen3 plan L308-322](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/dtensor/parallelize.py#L308-L322));
  trade-offs vs PP/CP/EP (reading: Narayanan et al. 2021, arXiv 2104.04473).
- Context parallelism; vocab-parallel log-softmax; the vLLM vs trainer logprob mismatch in depth.
- Megatron backend; NeMo-Gym; validation and checkpoint internals.

Known issues found by reading code (not verified at runtime):
- `recompute_kv_cache_after_weight_updates` probably does not recompute in-flight KV with vLLM 0.20: the worker
  calls `reset_prefix_cache()` with `reset_running_requests=False`, which vLLM refuses while requests run, and
  NeMo-RL drops the result and reports success
  ([vllm_worker_async.py:L1475-1489](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker_async.py#L1475-L1489),
  [vllm_generation.py:L1057-1076](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_generation.py#L1057-L1076)).
- vLLM sleep level 1 is hardcoded
  ([vllm_worker.py:L1009](https://github.com/NVIDIA-NeMo/RL/blob/19244a0949fb363326f71675223da75e09d14866/nemo_rl/models/generation/vllm/vllm_worker.py#L1009)):
  30.6 GiB per GPU copied to CPU and back every step, though the refit overwrites the weights anyway.
- The math verifier scans the whole response, so with thinking enabled an answer that appears only inside the
  reasoning still scores 1.

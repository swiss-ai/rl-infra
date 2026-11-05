# Run commands

The exact command of each run reported so far, from the driver command Slurm saved, without the run's own output
paths and logging settings. All of them work from this repository, because setup applies every patch. Sampling is
random, so a rerun gives close numbers, not the same ones. Later runs use later patches, and earlier runs differ
only by code that is off for them. Before a run on another machine, see `README.md` for models, data and images.

## Base checkpoint, GSM8K

Job 1173382, 12 nodes:

```bash
RUN=gsm8k-adamw-opt1 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.train_global_batch_size=2048
```

Job 1173403, 12 nodes:

```bash
RUN=gsm8k-adamw-opt2 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.train_global_batch_size=1024
```

Job 1173380, 12 nodes:

```bash
RUN=gsm8k-adamw-opt4 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml
```

Job 1180612, 12 nodes:

```bash
RUN=gsm8k-adamw-lr2e-6-opt1 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    policy.train_global_batch_size=2048 \
    checkpointing.enabled=false
```

Job 1180613, 12 nodes:

```bash
RUN=gsm8k-adamw-lr2e-6-opt2 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    policy.train_global_batch_size=1024 \
    checkpointing.enabled=false
```

Job 1180614, 12 nodes:

```bash
RUN=gsm8k-adamw-lr2e-6-opt4 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    policy.train_global_batch_size=512 \
    checkpointing.enabled=false
```

Job 1180783, 12 nodes:

```bash
RUN=gsm8k-adamw-lr4e-6-opt1 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=4.0e-6 \
    policy.train_global_batch_size=2048 \
    checkpointing.enabled=false
```

Job 1180789, 12 nodes:

```bash
RUN=gsm8k-adamw-lr4e-6-opt2 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=4.0e-6 \
    policy.train_global_batch_size=1024 \
    checkpointing.enabled=false
```

Job 1180793, 12 nodes:

```bash
RUN=gsm8k-adamw-lr4e-6-opt4 RECIPE=recipes/apertus2-chonk-vllm-adamw.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=4.0e-6 \
    policy.train_global_batch_size=512 \
    checkpointing.enabled=false
```

Job 1178777, 20 nodes:

```bash
RUN=gsm8k-muon-lr1e-6-matrix1e-5 RECIPE=recipes/apertus2-chonk-vllm-muon.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=1.0e-5 \
    checkpointing.enabled=false
```

Job 1178778, 20 nodes:

```bash
RUN=gsm8k-muon-lr1e-6-gains1e-6-matrix1e-5 RECIPE=recipes/apertus2-chonk-vllm-muon.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.gains_lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=1.0e-5 \
    checkpointing.enabled=false
```

Job 1180618, 20 nodes:

```bash
RUN=gsm8k-muon-lr1e-6-matrix2e-5 RECIPE=recipes/apertus2-chonk-vllm-muon.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=2.0e-5 \
    checkpointing.enabled=false
```

Job 1183563, 20 nodes:

```bash
RUN=gsm8k-muon-lr1e-6-gains1e-6-matrix4e-5 RECIPE=recipes/apertus2-chonk-vllm-muon.yaml JOB_TIME=05:00:00 bash run.sh envs/gsm8k.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.gains_lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=4.0e-5 \
    checkpointing.enabled=false
```

## Base checkpoint, DAPO

`hangang-dapo`, `hangang-dapo-lr2e-6`, `hangang-dapo-fp8-skip-trunc0` and `hangang-dapo-muon-fp8-skip-trunc0` ran with
vLLM memory at 0.8, the value then. To match them, add `policy.generation.vllm_cfg.gpu_memory_utilization=0.8`.

Job 1173197, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup.yaml 16 hangang-dapo
```

Job 1173598, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup.yaml 16 hangang-dapo-lr2e-6 \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

Job 1180617, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup-fp8-skip-trunc0.yaml 16 hangang-dapo-fp8-skip-trunc0
```

Job 1186409, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup-fp8-skip-trunc0.yaml 16 hangang-dapo-fp8-skip-trunc0-lr2e-6 \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

Job 1185235, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup-fp8-skip-trunc0.yaml 16 hangang-dapo-fp8-skip-trunc0-lr4e-6 \
    policy.megatron_cfg.optimizer.lr=4.0e-6
```

Job 1184046, 16 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-16n-megatron-128x16-16k-async-pp2-dedup-fp8-skip-trunc0.yaml 16 hangang-dapo-fp8-skip-trunc0-mem088
```

Job 1180620, 24 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-24n-megatron-128x16-16k-async-pp4-dedup-md-fp8-skip-trunc0.yaml 24 hangang-dapo-muon-fp8-skip-trunc0
```

Job 1184044, 24 nodes:

```bash
JOB_TIME=1-00:00:00 bash backends/nemo-rl/apertus2/launch.sh experiments/grpo-chonk/chonk-dapo-24n-megatron-128x16-16k-async-pp4-dedup-md-fp8-skip-trunc0.yaml 24 hangang-dapo-muon-fp8-skip-trunc0-lr1e-6-gains1e-6-matrix4e-5 \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.gains_lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=4.0e-5
```

## Base checkpoint, rubric

Job 1190668, resumed as job 1193455, 14 nodes:

```bash
RUN=rubric-adamw-lr1e-6 RECIPE=recipes/apertus2-chonk-vllm-adamw-rubric-2judge.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    env.judge_pool.num_nodes=2 \
    grpo.max_num_steps=431 \
    checkpointing.save_optimizer=true \
    checkpointing.save_period=220 \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

Job 1190666, 14 nodes:

```bash
RUN=rubric-adamw-lr2e-6 RECIPE=recipes/apertus2-chonk-vllm-adamw-rubric-2judge.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    env.judge_pool.num_nodes=2 \
    grpo.max_num_steps=431 \
    checkpointing.save_optimizer=true \
    checkpointing.save_period=220 \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

Job 1190667, 22 nodes:

```bash
RUN=rubric-muon-lr1e-6-gains1e-6-matrix4e-5 RECIPE=recipes/apertus2-chonk-vllm-muon-rubric-2judge.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric.yaml \
    policy.megatron_cfg.optimizer.lr=1.0e-6 \
    policy.megatron_cfg.optimizer.gains_lr=1.0e-6 \
    policy.megatron_cfg.optimizer.matrix_lr=4.0e-5 \
    env.judge_pool.num_nodes=2 \
    grpo.max_num_steps=431 \
    checkpointing.save_optimizer=true \
    checkpointing.save_period=220 \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

## SFT, DAPO

Job 1190614, 16 nodes:

```bash
RUN=sft-dapo-adamw-lr1e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-dapo.yaml JOB_TIME=1-00:00:00 bash run.sh envs/dapo-math-sft.yaml
```

Job 1190615, 16 nodes:

```bash
RUN=sft-dapo-adamw-lr2e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-dapo.yaml JOB_TIME=1-00:00:00 bash run.sh envs/dapo-math-sft.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

Job 1190617, 24 nodes:

```bash
RUN=sft-dapo-muon-lr1e-6-gains1e-6-matrix4e-5 RECIPE=recipes/apertus2-chonk-sft-vllm-muon-dapo.yaml JOB_TIME=1-00:00:00 bash run.sh envs/dapo-math-sft.yaml
```

## SFT, DAPO and GSM8K

Job 1202985, 16 nodes. Each step takes 90 prompts from DAPO and 38 from GSM8K, and validation reports AIME 2024
and the GSM8K test set apart:

```bash
RUN=sft-mix-dapo-gsm8k-lr2e-6 JOB_TIME=1-05:00:00 bash run.sh envs/dapo-gsm8k-sft.yaml \
    grpo.max_num_steps=196 \
    grpo.val_period=20 \
    checkpointing.checkpoint_must_save_by=01:04:30:00 \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

## Last SFT checkpoint, DAPO

Job 1210085, 16 nodes. The same run as 1190615, from iter_0019374, the last save of the SFT run. The recipe
stages hanint05's HF export of it by link:

```bash
RUN=sft19374-dapo-adamw-lr2e-6 RECIPE=recipes/apertus2-chonk-sft19374-vllm-adamw-dapo.yaml JOB_TIME=1-00:00:00 bash run.sh envs/dapo-math-sft.yaml \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

## SFT, rubric

Job 1192745, 14 nodes:

```bash
RUN=sft-rubric-full-adamw-lr1e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-rubric.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric-sft.yaml \
    grpo.max_num_steps=431 \
    checkpointing.save_period=220 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

Job 1192750, 14 nodes:

```bash
RUN=sft-rubric-full-adamw-lr2e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-rubric.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric-sft.yaml \
    grpo.max_num_steps=431 \
    checkpointing.save_period=220 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_must_save_by=00:15:40:00 \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

Job 1192752, 22 nodes:

```bash
RUN=sft-rubric-full-muon-lr1e-6-gains1e-6-matrix4e-5 RECIPE=recipes/apertus2-chonk-sft-vllm-muon-rubric.yaml JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric-sft.yaml \
    grpo.max_num_steps=431 \
    checkpointing.save_period=220 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

## SFT, rubric, three judges

Job 1203039, 16 nodes:

```bash
RUN=sft-rubric-3judges-40step JOB_TIME=02:30:00 bash run.sh envs/ultrafeedback-rubric-sft-3judges.yaml \
    grpo.max_num_steps=40 \
    checkpointing.enabled=false
```

Job 1203152, 16 nodes:

```bash
RUN=sft-rubric-full-3judges-adamw-lr2e-6 JOB_TIME=16:00:00 bash run.sh envs/ultrafeedback-rubric-sft-3judges.yaml \
    grpo.max_num_steps=431 \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    checkpointing.save_period=220 \
    checkpointing.save_optimizer=true \
    checkpointing.checkpoint_must_save_by=00:15:40:00
```

## SFT, tool calling

Job 1194398, 12 nodes:

```bash
RUN=sft-tools-nocap-adamw-lr2e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml JOB_TIME=06:00:00 bash run.sh envs/workplace-assistant-sft.yaml \
    grpo.max_num_steps=39 \
    grpo.val_period=10 \
    grpo.val_at_start=true \
    grpo.val_at_end=true \
    env.should_log_nemo_gym_responses=true \
    checkpointing.enabled=false \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

## SFT, coding

Job 1194832, 12 nodes:

```bash
RUN=sft-code-debug-1 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml JOB_TIME=01:30:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.num_prompts_per_step=32 \
    grpo.max_num_steps=3 \
    grpo.val_period=0 \
    grpo.val_at_end=false \
    checkpointing.enabled=false
```

Job 1195452, 16 nodes:

```bash
RUN=sft-code-40-budget12k RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-code.yaml JOB_TIME=05:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.num_prompts_per_step=32 \
    grpo.max_num_steps=40 \
    checkpointing.enabled=false
```

Job 1195564, 12 nodes:

```bash
RUN=sft-code-8k-lr2e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml JOB_TIME=04:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.num_prompts_per_step=32 \
    grpo.max_num_steps=40 \
    grpo.val_period=10 \
    checkpointing.enabled=false \
    policy.generation.max_new_tokens=8192 \
    policy.generation.thinking_token_budget=6144 \
    policy.megatron_cfg.optimizer.lr=2.0e-6
```

Job 1195905, 16 nodes:

```bash
RUN=sft-code-full-128x16-wandb RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-code.yaml JOB_TIME=1-05:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.max_num_steps=184 \
    grpo.val_period=20 \
    policy.generation.max_new_tokens=8192 \
    policy.generation.thinking_token_budget=6144 \
    checkpointing.save_period=50 \
    checkpointing.checkpoint_must_save_by=01:04:30:00 \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    env.nemo_gym.code_gen.resources_servers.code_gen.sandbox_endpoint_file=${SCRATCH}/code-sandbox/endpoint-overnight.json
```

Job 1195570, 12 nodes:

```bash
RUN=sft-mbpp-lr2e-6 RECIPE=recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml JOB_TIME=03:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.num_prompts_per_step=32 \
    grpo.max_num_steps=40 \
    grpo.val_period=10 \
    checkpointing.enabled=false \
    policy.generation.max_new_tokens=4096 \
    policy.generation.thinking_token_budget=3072 \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    data.train.data_path=${SCRATCH}/code-gen-mbpp/train.jsonl \
    data.validation.data_path=${SCRATCH}/code-gen-mbpp/validation.jsonl
```

Sandboxes on OpenSandbox pods at CSCS instead of our own node (Gym patch 0005). The key stays in a file of one's
own, `~/.opensandbox_token` here. Both jobs ran from a private copy of the clone with that code, before it was a patch.

Job 1202265, 16 nodes, 40 steps to compare with the first 40 of job 1195905. Every job above and this one ran with
`reward_mode: fraction`, which `envs/code-gen-sft.yaml` set then; to rerun them add `${CODE_GEN}.reward_mode=fraction`:

```bash
CODE_GEN=env.nemo_gym.code_gen.resources_servers.code_gen
RUN=sft-code-osb-128x16 JOB_TIME=07:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.max_num_steps=40 \
    grpo.val_period=20 \
    checkpointing.enabled=false \
    policy.generation.max_new_tokens=8192 \
    policy.generation.thinking_token_budget=6144 \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    ${CODE_GEN}.sandbox_endpoint_file=null \
    ++${CODE_GEN}.opensandbox.domain=https://opensandbox-gvisor.breithorn.svc.cscs.ch \
    ++${CODE_GEN}.opensandbox.api_key_file=$HOME/.opensandbox_token \
    ++${CODE_GEN}.opensandbox.harness_dir=${SCRATCH}/code-sandbox/env/harness \
    ++${CODE_GEN}.opensandbox.num_pods=64
```

Job 1202946, 16 nodes, Gym's setup: `reward_mode: all_tests` and LiveCodeBench validation 10 times. Training data is
the same file as above, without the 308 rows over 4 MB. Job 1202858 measured one such validation at 20 minutes:

```bash
CODE_GEN=env.nemo_gym.code_gen.resources_servers.code_gen
RUN=sft-code-gym-osb-128x16 JOB_TIME=1-05:00:00 bash run.sh envs/code-gen-sft.yaml \
    grpo.max_num_steps=184 \
    grpo.val_period=20 \
    grpo.val_num_generations_per_prompt=10 \
    checkpointing.save_period=50 \
    checkpointing.checkpoint_must_save_by=01:04:30:00 \
    policy.generation.max_new_tokens=8192 \
    policy.generation.thinking_token_budget=6144 \
    policy.megatron_cfg.optimizer.lr=2.0e-6 \
    ${CODE_GEN}.sandbox_endpoint_file=null \
    ++${CODE_GEN}.opensandbox.domain=https://opensandbox-gvisor.breithorn.svc.cscs.ch \
    ++${CODE_GEN}.opensandbox.api_key_file=$HOME/.opensandbox_token \
    ++${CODE_GEN}.opensandbox.harness_dir=${SCRATCH}/code-sandbox/env/harness \
    ++${CODE_GEN}.opensandbox.num_pods=64
```

It stopped at step 20: the step-20 validation, 10 times over, ran the head node out of host memory while training's
memory was on it too, and Slurm killed a trainer rank. `envs/code-gen-osb-sft.yaml` has this job's settings with
validation once per prompt, as in job 1202265.

### Trainer GPU memory

Robert's 3-step run of `envs/code-gen-osb-sft.yaml` (job 1216876) and the same config unchanged (job 1221319) died in
the first training step with `ncclUnhandledCudaError`. PyTorch's cache of freed blocks had grown to 83-85 GB of the
95 GB while about 40 GB were in use, and NCCL, which sets up its buffers at first use outside PyTorch, found the GPU
full; a Triton kernel load failed the same way in job 1221320. NeMo-RL patch 0033 adds `NRL_CUDA_MEMORY_FRACTION`,
which caps PyTorch's allocator, and the config sets 0.78 (74 GB): the other 21 GB cover NCCL at its default channels,
whose largest use was 19.5 GB, during checkpoint saves. The config also trains with one optimizer step per GRPO step.

3-step tests, 16 sandbox pods, all but the first with one optimizer step (`experiments/trainer-memory/launch.sh
<variant>`; "free" is the lowest free GPU memory in 10 s samples over all trainer ranks):

| Variant | Change | Result |
|---|---|---|
| base | none, four optimizer steps | NCCL error in step 1 |
| 1opt | one optimizer step | 3 steps, 0.28 GB free, training 353/348 s (steps 2/3) |
| 1opt-split512 | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` | Triton out of memory in step 1 |
| 1opt-nocumem | `NCCL_CUMEM_ENABLE=0` | NCCL error in step 1 |
| 1opt-eagerconnect | `NCCL_RUNTIME_CONNECT=0` (trainer and vLLM) | NCCL error in step 1 |
| 1opt-bf16grad | gradients reduced in bf16 | PyTorch out of memory in step 2 (66 GB live on one rank: expert routing imbalance) |
| 1opt-pp22-20 | 22/20 layers on the two pipeline stages | 3 steps, 1.0 GB free |
| 1opt-nccl16 | at most 16 NCCL channels (trainer and vLLM) | 3 steps, NCCL 10 GB instead of 16, 2.3 GB free |
| 1opt-cap80-ckpt | cap 0.80, checkpoint and validation after step 2 | 3 steps, 3.8 GB free, training 355/362 s |
| 1opt-cap78-ckpt | cap 0.78, checkpoint and validation after step 2 | 3 steps, 5.9 GB free, training 359/361 s |

The full pass with the cap, job 1223855 (it started from the clone `$HOME/projects/nemo-rl-apertus2-mem`, which had
patch 0033 before main did):

```bash
RUN=sft-code-osb-16k-cap78 JOB_TIME=1-05:00:00 bash run.sh envs/code-gen-osb-sft.yaml \
    env.nemo_gym.code_gen.resources_servers.code_gen.opensandbox.api_key_file=$HOME/.opensandbox_token \
    grpo.async_grpo.max_trajectory_age_steps=2 \
    checkpointing.checkpoint_must_save_by=01:04:30:00
```

## SFT, Harbor terminal tasks

Terminus-2 through Gym's `harbor_agent`, a fresh OpenSandbox pod per trial, Robert's micro-tasks (`envs/harbor-osb-sft.yaml`,
Gym patches 0006-0010, NeMo-RL patches 0031 and 0032). The runs below started from the clone
`$HOME/projects/nemo-rl-apertus2-harbor`, which had those patches before main did; from main, leave out `NEMO_RL_DIR`.
The tasks, and a check of one task per family on real pods:

```bash
python3.12 envs/make_micro_tasks.py --out $SCRATCH/harbor/micro/tasks
python3.12 envs/validate_micro_tasks.py $SCRATCH/harbor/micro/tasks --api-key-file ~/.opensandbox_token
```

Debug runs of 2 steps, 12 nodes, 8x8 trials:

```bash
RUN=sft-harbor-micro-debug3 JOB_TIME=02:00:00 NEMO_RL_DIR=$HOME/projects/nemo-rl-apertus2-harbor bash run.sh envs/harbor-osb-sft.yaml \
    grpo.max_num_steps=2 \
    checkpointing.enabled=false \
    env.nemo_gym.sandbox.opensandbox.connection.api_key_file=$HOME/.opensandbox_token
```

- Job 1209869: a turn after one with thinking failed with HTTP 422 (Gym refuses an assistant message with its
  thinking), so most trials stopped after one turn. Cancelled; `interleaved_thinking: false` since.
- Job 1209940: 2 to 20 turns per trial, no errors, but every advantage was 0: the first prompt shows each pod's
  hostname, so GRPO grouped each trial alone. Fixed by NeMo-RL patch 0032.
- Job 1210063: advantages from -2.47 to 2.47, loss 0.15 and 0.46. Validation 20/32 at step 0, 21/32 after step 2,
  each about 16 minutes; a step's 64 trials took about 16 minutes. Generation KL error 0.002.

A run leaves the pods of its last batch behind until `sandbox_ttl_s` (1 h). To delete them, by the run's label:

```bash
printf 'sandbox:\n  opensandbox:\n    connection:\n      domain: https://opensandbox-gvisor.breithorn.svc.cscs.ch\n      api_key_file: %s\n' \
    $HOME/.opensandbox_token > $SCRATCH/opensandbox-connection.yaml
$SCRATCH/nemo-rl-apertus2/gym-venvs/resources_servers/code_gen/.venv/bin/python \
    $NEMO_RL_DIR/3rdparty/Gym-workspace/Gym/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py \
    --connection-config $SCRATCH/opensandbox-connection.yaml --run-id <run name> --user rl-infra --reap
```

Proof-of-concept runs, 20 steps, 12 nodes each, validation every 5 steps on 64 trials, lr 2e-6, weights saved at
the end only. Harbor's trial directories go to `/home01` (one's own file quota) instead of `$SCRATCH`, whose group
quota was at 4.8M of 5M files. Job 1210340, all 8 families (`--count 192 --num-validation 32`, 4 validation tasks per
family, 2 trials each), and job 1210341, the reverse family only (`--families reverse --count 96 --num-validation
16`, 4 trials each):

```bash
python3.12 envs/make_micro_tasks.py --out $SCRATCH/harbor/micro-all/tasks --count 192 --num-validation 32
python3.12 envs/make_micro_tasks.py --out $SCRATCH/harbor/micro-reverse/tasks --families reverse --count 96 --num-validation 16
for spec in "sft-harbor-micro8-20 micro-all 2" "sft-harbor-reverse-20 micro-reverse 4"; do
    set -- $spec
    RUN=$1 JOB_TIME=09:00:00 NEMO_RL_DIR=$HOME/projects/nemo-rl-apertus2-harbor bash run.sh envs/harbor-osb-sft.yaml \
        grpo.max_num_steps=20 \
        grpo.val_period=5 \
        grpo.val_num_generations_per_prompt=$3 \
        policy.megatron_cfg.optimizer.lr=2.0e-6 \
        checkpointing.save_period=20 \
        checkpointing.save_optimizer=false \
        checkpointing.checkpoint_must_save_by=00:08:30:00 \
        harbor_data_dir=$SCRATCH/harbor/$2 \
        env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.harbor_jobs_dir=$HOME/harbor-jobs/$1 \
        env.nemo_gym.sandbox.opensandbox.connection.api_key_file=$HOME/.opensandbox_token
done
```

Results, from the rollout dumps (validation on 64 trials each time; "tool" = trials that typed `rev` or Python's
`[::-1]`/`reversed(`):

| Validation step | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| reverse only (1210341) | 21/64 | 23/64 | 23/64 | 35/64 | 35/64 |
| reverse only, tool use | 28/64 | 29/64 | 32/64 | 40/64 | 51/64 |
| all 8 families (1210340) | 39/64 | 41/64 | 44/64 | 40/64 | 43/64 |

The reverse run learned to use a tool: at step 0 trials with a tool solved 20/28 and those without 1/36. Its training
reward went from 0.32 (mean of the first 5 weight versions) to 0.61 (last 5). The all-families run moved from 0.61 to
0.66 in training reward and stayed within noise on validation: the easy families are near 8/8, and each hard family
is 1/8 of the data. Jobs took 3 h 52 and 4 h 10; weights of step 20 (no optimizer) are in `results/<run>/step_20`.

These runs had a thinking budget of 4096 per turn. Their `train/policy_kl_error` (1e2 to 4e9) comes entirely from the
2 to 14 turns per step that reached it: vLLM forces `<|/out|>` there and reports logprob 0, the trainer gives it e^-9
to e^-34 (from the per-token dumps `train_data_step*.jsonl`). The coding run with a budget of 6144,
`sft-code-full-128x16-wandb`, reads up to 5e11 for the same reason. The IS weight of such a token is about 0, so
training was not affected. The task file has no budget since.

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

A run leaves the pods of its last batch behind until `sandbox_ttl_s` (1 h). To delete them, by the run's label
(on HANGANG the Gym venvs are in `$HOME/.cache/nemo-rl-apertus2/gym-venvs` since `sites/hangang.env` moved them
there, so use that instead of `$SCRATCH/nemo-rl-apertus2/gym-venvs`, and any server's venv):

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

## SFT, Harbor verified tasks

The verified Harbor tasks of `harbor_verified_tasks` (982 SWE and terminal tasks from 15 campaigns, port of
swiss-ai/nemo-rl-development), Terminus-2 through Gym's `harbor_agent` on Harbor 0.25.0, which verifies most of them
in a second pod of their own (`envs/harbor-verified-sft.yaml`, Gym patches 0011 to 0013). Iter_0019374 on the tools
layout, 12 nodes (8 trainer, 4 vLLM). The HANGANG copy of the store has no `tasks.csv`, so the collector takes every
task (`--include-unverified`); the campaigns ship fewer than 100 tasks each, so the dropped ones look removed already:

```bash
COLLECT_ARGS=--include-unverified bash envs/prepare_harbor_verified_tasks.sh \
    ~/projects/harbor_verified_tasks/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks \
    ~/projects/task_datasets/harbor_verified
```

That writes 966 training and 16 validation rows, 1.4 GB and 79,501 files. Harbor 0.25.0 loads all 982 tasks. The 39
tasks of `sample100_HF_ML_Tasksmith` ask for `network_mode = "no-network"`, which OpenSandbox pods cannot enforce,
so Harbor rejects them before any pod starts and their rollouts are masked.

Debug run of 2 steps, 8x8 trials, no weights saved:

```bash
RUN=sft19374-harbor-verified-debug3 JOB_TIME=02:00:00 WANDB_PROJECT=grpo-dev WANDB_ENTITY=apertus \
RECIPE=recipes/apertus2-chonk-sft19374-vllm-adamw-tools.yaml bash run.sh envs/harbor-verified-sft.yaml \
    grpo.max_num_steps=2 \
    checkpointing.enabled=false \
    harbor_data_dir=$HOME/projects/task_datasets/harbor_verified \
    registry_cred_file=$HOME/.rl_registry_cred \
    env.nemo_gym.sandbox.opensandbox.connection.api_key_file=$HOME/.opensandbox_token \
    env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent.harbor_jobs_dir=$HOME/harbor-jobs/$RUN
```

- Jobs 1254871 and 1254945: the group's `/scratch` file quota (5M files, shared by in0138) ran out at start-up, first
  in vLLM's DeepGEMM JIT cache, then in the Gym venv build, which is about 150k files with its uv cache. The venvs and
  the job's uv cache are in home since (`sites/hangang.env`), and a half-built venv has to be deleted by hand:
  `skip_venv_if_present` would reuse it.
- Job 1254979: passed, 1 h 39 min. Validation 8/64 at step 0 and 3/64 after step 2 (16 tasks, 4 trials each, within
  noise); training reward 0.078 and 0.281, loss 0.030 and 0.013, generation KL error 0.0012 and 0.0013. 382 Harbor
  trials, 39 with reward 1, 373 without an exception (5 cancelled at shutdown, 2 agent timeouts, 2 runtime errors);
  15 rollouts masked. Steps took 413 s and 2885 s, validation 1102 s and 1746 s: the sandboxes set the pace, and vLLM
  stayed idle 50 to 80% of the time with at most 9 requests in flight per engine. Many trials end at the 16k
  context (about 570 `ContextLengthExceededError` log lines for 382 trials), so a longer context is the first thing
  to try.

## Long context: STEM mix at 32K, coding at 64K and 128K

Launchers, layouts and lessons: `experiments/long-context/README.md`. All on SFT v0.2 (iteration 18,315), 512 prompts ×
16 answers per step, one optimizer step, lr 2e-6, checkpoints with weights only.

Run 3's mix at 32K, job 1291708, 32 nodes (16 trainer + 16 vLLM), 100 steps, staleness 2, before Ray on Slingshot:

```bash
MIX=run3 CP=4 CAP=0.74 STALENESS=2 STEPS=100 VAL=1 CKPT=weights DEBUG=0 JOB_TIME=1-12:00:00 SAVE_BY=01:11:30:00 \
RUN=sftv02-stem-mix-run3-32k-32n bash experiments/long-context/launch-stem-ctx.sh 32768
```

- AIME 2024 / GPQA Diamond: 53.1 / 33.7 at step 0, 80.4 / 48.2 at step 20, 82.4 / 53.2 at 40, 83.4 / 52.5 at 60,
  83.0 / 51.8 at 80, 82.9 / 53.1 at 100. Flat from step 40.
- The driver froze in a TensorBoard plot write at step 47; the watchdog's stop request went unanswered for 3 hours,
  and the retry resumed from the step-40 weights with a fresh optimizer state. 22 h 18 min in all.

Run 5's mix at 32K, job 1299180, 49 nodes (16 trainer + 32 vLLM + 1 judge), 135 steps, staleness 4:

```bash
MIX=run5 GEN_NODES=32 CP=4 CAP=0.74 STALENESS=4 STEPS=135 VAL=1 CKPT=weights KEEP=10 DUMP=text NCCL_MIN=60 DEBUG=0 \
JOB_TIME=2-00:00:00 SAVE_BY=01:23:30:00 RUN=sftv02-stem-mix-run5-32k-49n \
bash experiments/long-context/launch-stem-ctx.sh 32768
```

- AIME 2024 / GPQA Diamond / chemistry: 50.5 / 33.0 / 6.1 at step 0, 80.5 / 52.3 / 47.2 at 20, 82.4 / 54.2 / 49.6 at
  40, 83.2 / 58.5 / 50.4 at 60, 84.0 / 59.5 / 51.6 at 80, 84.4 / 60.3 / 52.9 at 100, 84.3 / 60.1 / 58.5 at 120,
  84.3 / 61.3 / 56.5 at 135. The same mix at 16K with staleness 3 (job 1285729) ended at 79.1 / 56.4 / 50.0.
- 15 h 31 min, no restart. A step took 270 s early and about 350 s late in the run (answers grew from 6,500 to
  9,700 tokens), 263 s of it training; the trainer waited 17 s per step for answers (median).
- 1,105,920 rollouts kept as text with rewards (`train_data_step<N>.jsonl`, 31 GB), 46.4% with reward 1. The grouping
  columns of patch 0044 are missing from this run's files; `tools/rollout_index.py` rebuilds all of them but the
  generation weight version.

Coding at 64K, job 1298910, 32 nodes (16 + 16) plus a sandbox node, 46 steps (one pass), staleness 4, KCP with CP 8:

```bash
SANDBOX_ENDPOINT_FILE=$SCRATCH/code-sandbox/endpoint-64k.json SANDBOX_WORKERS=192 JOB_TIME=2-00:00:00 \
    bash backends/nemo-rl/sandbox/start.sh
NODES=32 KCP=1 CP=8 STALENESS=4 PROMPTS=512 STEPS=46 VAL=full CKPT=weights CKPT_EVERY=20 DEBUG=0 \
JOB_TIME=2-00:00:00 SAVE_BY=01:23:30:00 RUN=sftv02-code-64k-512x16-32n \
SANDBOX=$SCRATCH/code-sandbox/endpoint-64k.json \
bash experiments/long-context/launch-code-ctx.sh 65536 checkpointing.keep_top_k=3
```

- LiveCodeBench (322 problems × 10): 61.0% at step 0, 64.6% at step 20 (thinking medium), 64.6% at step 40 (thinking
  high; 474 of the 3,220 answers differ from step 20, the totals are equal). Batch reward 0.44 at step 1, about 0.51 by
  step 36. About 950 s per step on Slingshot (1,240-1,590 s on the management network, job 1293154).
- Job 1298910 failed in the validation of step 40 (Gym's agent server overloaded, truncated vLLM responses, HTTP 500)
  and its retry resumed from step 20, because the step-40 checkpoint is written after the validation. Since then a step
  that saves and validates saves first (patch 0047, `CKPT_FIRST=1`), and Gym and the sandbox are on Slingshot (0048).
- Resumed from step 20 at thinking high as job 1308624 (steps 1-20 ran at medium), about 1,000 s per step. Jobs
  1306028 and 1306032 failed at setup before it: patch 0048 gave vLLM's URL the address of `<node>-hsn0`, which is not
  the address Ray registers for a worker node, and `place_on_generation_node` found no node for it (fixed by 0049).

```bash
SANDBOX_ENDPOINT_FILE=$SCRATCH/code-sandbox/endpoint-64k-hsn.json SANDBOX_WORKERS=192 JOB_TIME=2-00:00:00 \
    bash backends/nemo-rl/sandbox/start.sh
NODES=32 KCP=1 CP=8 STALENESS=4 PROMPTS=512 STEPS=46 CKPT=weights CKPT_EVERY=20 THINKING=high DEBUG=0 \
JOB_TIME=2-00:00:00 SAVE_BY=01:23:30:00 RUN=sftv02-code-64k-512x16-32n \
SANDBOX=$SCRATCH/code-sandbox/endpoint-64k-hsn.json \
bash experiments/long-context/launch-code-ctx.sh 65536 checkpointing.keep_top_k=3
```

- `KCP_FLA_DIR` names the flash-linear-attention 0.5.2 copy when it is not `${PROJECTS_DIR}/fla-0.5.2`.
- Running at the time of writing (step 45 of 46).

Coding at 128K, job 1299941, the same with CP 16 and the trainer off the driver's node:

```bash
NODES=32 KCP=1 CP=16 OFFHEAD=1 STALENESS=4 PROMPTS=512 STEPS=46 VAL=full CKPT=weights CKPT_EVERY=20 DEBUG=0 \
JOB_TIME=2-00:00:00 SAVE_BY=01:23:30:00 RUN=sftv02-code-128k-512x16-32n \
SANDBOX=$SCRATCH/code-sandbox/endpoint-128k.json \
bash experiments/long-context/launch-code-ctx.sh 131072 checkpointing.keep_top_k=3
```

- LiveCodeBench: 61.2% at step 0, 64.1% at step 20 (thinking medium). About 1,090 s per step.
- Without `OFFHEAD=1` the driver's node ran out of host memory in the first training step (job 1299189); on the
  management network the first training step timed out twice (jobs 1293158, 1293163).
- Job 1299941 lost a Slingshot link at step 39 (NCCL `NET/OFI ... PTLTE_NOT_FOUND` on one trainer node). Its retries
  read the recipe again after an edit (16 -> 15 vLLM nodes), built 17 trainer nodes and stopped (world size 68); the
  launchers now copy the recipe at submission. The run resumed from step 20.
- Resumed from step 20 at thinking high as job 1308640, about 1,115 s per step:

```bash
SANDBOX_ENDPOINT_FILE=$SCRATCH/code-sandbox/endpoint-128k-hsn.json SANDBOX_WORKERS=192 JOB_TIME=2-00:00:00 \
    bash backends/nemo-rl/sandbox/start.sh
NODES=32 KCP=1 CP=16 OFFHEAD=1 STALENESS=4 PROMPTS=512 STEPS=46 CKPT=weights CKPT_EVERY=20 THINKING=high DEBUG=0 \
JOB_TIME=2-00:00:00 SAVE_BY=01:23:30:00 RUN=sftv02-code-128k-512x16-32n \
SANDBOX=$SCRATCH/code-sandbox/endpoint-128k-hsn.json \
bash experiments/long-context/launch-code-ctx.sh 131072 checkpointing.keep_top_k=3
```

- Running at the time of writing (step 36 of 46).
## SFT v0.2 (iter_0018315), Reasoning Gym in the math environment

Job 1276175, 10 nodes (HANGANG, account hanint09). It ran an earlier version of this branch, which differs from the
current one in two ways. Its rows had 84 tasks, coin_flip included (24 train rows, 1 or 2 validation rows); the current
script drops coin_flip and gives 83. Its reward kept Reasoning Gym's partial credit above 0.05; the current verifier
is binary, and refuses NaN answers on complex_arithmetic and number_sorting. Of its 392 rewarded training answers
none used those holes (no coin_flip or number_sorting answer was rewarded, the 12 complex_arithmetic ones were
numbers, and no reward was below 0.2), so the numbers below should carry over, but they are not a rerun.

```bash
python3.12 envs/prepare_reasoning_gym.py $SCRATCH/reasoning-gym/native
RUN=rg-native-v02-high-20st JOB_TIME=04:00:00 bash run.sh envs/reasoning-gym-sft.yaml
```

Validation on the same 128 prompts, reward on the claim frame (counts from the environment's frame metrics; "reply
correct" = the reply frame verified the same way, logged but not rewarded):

| Validation step | 0 | 10 | 20 |
|---|---|---|---|
| accuracy | 0.070 | 0.109 | 0.178 |
| answers with a claim frame | 20/128 | 26/128 | 44/128 |
| correct claims | 9 | 14 | 24 |
| reply correct | 59 | 57 | 53 |

Training reward rose from 0.016 (step 1) to 0.20 and 0.19 (steps 19 and 20), and the share of training answers with
a claim frame from 2.7% to 26.6%; rows trained per step (non-zero advantage) went from 16-96 to 104-112 of 256. The
think frame was never empty (thinking level high). So far the gain is the claim format, not the reasoning: the reply
is right about as often as at step 0, and most correct replies still carry no claim. 1 h 35 min, about 63 GPU-h.

Two settings that did not work, from the same session:
- Thinking level medium (the default system frame): v0.2 left the think frame empty on every one of these
  prompts (128/128 on the validation set, through Gym's reasoning_gym server with an "<answer> tags" line on each
  prompt, job 1276079).
- DAPO-Math's "Answer:" instruction around each prompt: the claim rate did not rise (20/128 at step 0), the model
  wrote "Answer: X" in the reply instead, and only 2 of the 20 claims were right (job 1276229, stopped at step 5).

## RDKit chemistry in the math environment (claim graded)

Rows from `envs/prepare_chem_rdkit.py`: 12,499 train candidates (not pass-rate filtered) and validation rows from
held-out molecules (no shared Bemis-Murcko scaffold with train) covering all 123 properties. HANGANG, account hanint09,
thinking level high, lr 2e-6.

### From SFT v0.2: the model does not claim

Job 1276738, 10 nodes, bare prompts, async trainer: at step 0, 110 of 128 validation answers had no claim frame, 1 had a
claim (wrong), and the reply was right in 37. Steps 1 and 2 had reward 0, then the async trainer waited for 32 groups
with a rewarded answer that never came ("not enough groups to form a full step", 503 times). Cancelled after 28 min.

As a diagnostic, job 1276869 graded the reply (`env.math.reward_frame=reply`): 0.273, 0.258, 0.398 at steps 0, 10, 20
on 128 prompts. 8 of its 24 newly correct answers had been cut at 16K at step 0, so part of that gain is truncation
repair.

### From the STEM-mix climb: it claims, with the answer-line instruction

`recipes/apertus2-chonk-sftv02-stemmix-run4-100-vllm-adamw.yaml` starts from step 100 of hanint06's
`sftv02-stem-mix-run4-33n`, converted to HF (`/scratch/hanint09/exports/sftv02-stem-mix-run4-step100`). That run
claims on 959 of its 960 DAPO-Math validation prompts. On chemistry, step 0 of
validation (128 prompts):

| Prompt | System frame | No claim | Correct (claim) | Correct (reply) | Mean length |
|---|---|---|---|---|---|
| bare | high | 125 | 0 | 42 | 5,801 |
| bare | medium | 127 | 0 | 31 | 522 (think frame empty in 125) |
| with the answer-line instruction | high | **0** | **60** | 61 | 4,909 |

The answer-line instruction is the text the math rows carry (as in DAPO-Math-17k), word for word: "Solve the following
math problem step by step. The last line of your response should be of the form Answer: $Answer …" before the
prompt and "Remember to put your answer on its own line after "Answer:"." after it. `prepare_chem_rdkit.py` always adds it.

### Training: async with 32 prompts is flat, dynamic sampling with 64 learns

| Job | Trainer | Prompts per step | Steps | Validation accuracy (claim) |
|---|---|---|---|---|
| 1280114 | async, `skip_zero_advantage_groups` | 32 | 20 | 0.469, 0.414, 0.461 (128 prompts, steps 0/10/20) |
| 1280378 | async, `skip_zero_advantage_groups` | 32 | 50 | 0.469, 0.438, 0.461 (128 prompts, steps 0/20/40); length 4,909 to 7,041 |
| **1280697** | **sync, `use_dynamic_sampling`** | **64 (of 128 generated)** | 45 | **0.420, 0.438, 0.484, 0.463, 0.492** (512 prompts, steps 0/10/20/30/40) |

The async runs trained on all 32 groups per step although 11 to 15 had equal rewards. `skip_zero_advantage_groups`
leaves them out of the passes but scales the rest so the update equals the full-batch one ("Training 160 of 256
rows"). In the async loop `use_dynamic_sampling` is only logged; the filtering is in the synchronous trainer.

Job 1280697 (`chem-rdkit-sft.yaml` as committed): 14 nodes, 8 trainer + 6 vLLM.
- **Dynamic sampling**: 77 to 83 of 128 generated groups had differing rewards, and all 512 trained rows had
  nonzero advantage. Steps took about 360 s, and about 680 s when a second generation batch was needed.
- **Validation (512 prompts)**:
  - steps 0 to 20: 79 newly correct (1 previously cut), 46 newly wrong;
  - fragments 147/322 to 174/322, counts 54/170 to 61/170;
  - mean length 4,979 to 5,372;
  - at most 14 of 1,024 rollouts cut at 16K per step.
- **Training reward per step**: about 0.43 over steps 1 to 25, about 0.48 over steps 26 to 45.
- **OOM**: trainer rank 19 ran out of memory in `train_prep` at steps 27 and 46. It requested 6.1 GiB with 68.7 GiB
  reserved and 53.2 GiB allocated, under the 0.78 cap (74.1 GiB). `driver_retry` resumed from the step-25 checkpoint
  (weights only, so Adam restarted). The job was stopped at step 46, before the step-50 save.
- Raising `NRL_CUDA_MEMORY_FRACTION` is the next thing to try; the task file now saves every 10 steps.

```bash
python3.12 envs/prepare_chem_rdkit.py /scratch/hanint09/chem/data/faithful-qwen-v1/rows $SCRATCH/chem-rdkit/native --val-rows 512
RUN=chem-rdkit-run4s100-dapo-dynsamp JOB_TIME=10:00:00 bash run.sh envs/chem-rdkit-sft.yaml
```

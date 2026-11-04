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

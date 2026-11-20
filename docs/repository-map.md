# Repository map

Every file of the repository and its role (336 files, branch `restructure`, 2026-11-20). Roles:
**run** = used by every training run, **setup** = prepares sources, images, models or data,
**check** = a test or probe before or after a run, **tool** = analysis of a finished run,
**history** = kept as the record of past runs, not used by new ones.

## How a run passes through the files

```
bash run.sh envs/stem/main.yaml
 └ run.sh ─ sites/load.sh (sites/hangang.env, site.env, backends/nemo-rl/apertus2/env, backends/nemo-rl/env)
   ├ backends/nemo-rl/config.sh       merges the task's defaults chain, copies every config file for the job
   ├ sizes/<tier>.yaml                node counts and batch scale
   ├ backends/nemo-rl/apertus2/contexts.tsv (+ kcp/use.sh above 32K)   CP, packing, memory fraction
   ├ backends/nemo-rl/apertus2/setup.sh         NeMo-RL clone at the pin + patches, Megatron, Bridge, vLLM files, image
   ├ experiments/grpo-chonk/stage_model.sh      model export staged on scratch
   └ backends/nemo-rl/apertus2/launch.sh → backends/nemo-rl/launch.sh
       ├ backends/nemo-rl/preflight.sh          config check, report next to the Slurm log
       ├ backends/nemo-rl/container.sh          container flags from apertus2/edf.toml.template
       └ sbatch <clone>/ray.sub → numa_watch.sh (hang_watch.sh) on every node
           └ driver_retry.sh → apertus2/driver.sh (pydeps.sh) → python examples/run_grpo.py
```

## Top level

| File | Role | Purpose |
|---|---|---|
| `run.sh` | run | The entry point: task file in, Slurm job out (rl-infra keys, tiers, context, config copies, setup, launch) |
| `setup.sh` | setup | Setup of the stock NeMo-RL stack (the Qwen experiments in `experiments/grpo-math`); the Apertus 2 setup is `backends/nemo-rl/apertus2/setup.sh` |
| `README.md` | doc | Quick start, task files and tiers, NeMo-RL development, cluster notes |
| `site.env.example` | doc | Template of the personal, gitignored `site.env` (account, scratch, W&B) |
| `.gitignore` | — | Keeps `site.env` and generated files out of git |
| `docs/nemo-rl-architecture.md` | doc | How NeMo-RL works inside (driver, workers, refit) |
| `docs/repository-map.md` | doc | This file |

## `sites/` (cluster profiles) and `sizes/` (node tiers)

| File | Role | Purpose |
|---|---|---|
| `sites/load.sh` | run | Loads the settings in order: environment, `site.env`, site profile, stack env files |
| `sites/hangang.env` | run | KISTI HANGANG: account, partition, Slurm flags, scratch, pyxis, NCCL/libfabric fixes, mounts, image, Slingshot suffix, `RL_DATA_DIR` |
| `sites/alps.env` | run | CSCS Alps: the same for Clariden (container engine EDF) |
| `sizes/cheap.yaml` | run | 8 trainer + 16 vLLM nodes, batch x1 |
| `sizes/expensive.yaml` | run | 16 + 32 (run 5), batch x1 |
| `sizes/ultra.yaml` | run | 32 + 64, batch x2 (to be confirmed by the benchmark jobs 1315243/45/47) |

## `envs/` (what to train on)

| File | Role | Purpose |
|---|---|---|
| `stem/base.yaml` | run | STEM: 8 training sets, 3 validation sets, all environments, judge node, thinking level per dataset (`frames`) |
| `stem/main.yaml` | run | STEM official run: run 5's mix, 135 steps, 32K, thinking high, checkpoints, text rollouts, expensive tier |
| `stem/dev/run5-32k-job1299180.yaml` | run | Reproduces job 1299180 (thinking medium except Reasoning Gym and chemistry) |
| `stem/dev/chem-rdkit-sft.yaml`, `stem/dev/reasoning-gym-sft.yaml` | run | Single-environment tasks: chemistry alone, Reasoning Gym alone (older layout: recipe in `defaults`) |
| `stem/archive/stem-mix-run1.yaml` … `run5.yaml`, `-run3-base`, `-run3-sciworld`, `-enigmata`, `-openqa-judge` | history | Task files of the STEM runs 1-5 and their single-environment trials, as they ran |
| `stem/prepare_chem_rdkit.py`, `stem/gen_chem_rdkit.py` | setup | Chemistry rows (from Antonio's generator; a template generator of our own) |
| `stem/prepare_reasoning_gym.py` | setup | Reasoning Gym rows from nvidia/Nemotron-RL-ReasoningGym-v1 |
| `stem/stem-mix-four-shot.txt` | history | Few-shot prompt of runs 1-2 (pretraining checkpoint) |
| `stem/dev/gsm8k.yaml`, `stem/dev/dapo-math.yaml` | run | GSM8K and DAPO-Math on the pretraining checkpoint (the quick start) |
| `stem/dev/gsm8k-sft.yaml`, `stem/dev/dapo-math-sft.yaml`, `stem/dev/dapo-gsm8k-sft.yaml` | run | The same on an SFT checkpoint; DAPO + GSM8K 70/30 mix |
| `stem/gsm8k-five-shot.txt` | run | Few-shot prompt of the pretraining-checkpoint GSM8K task |
| `code/code-gen-sft.yaml`, `code/code-gen-osb-sft.yaml` | run | Competitive coding on Gym's code_gen, tests on a sandbox node or on OpenSandbox pods |
| `code/harbor-osb-sft.yaml`, `code/harbor-verified-sft.yaml` | run | Harbor terminal/SWE tasks with Terminus-2 on OpenSandbox |
| `code/collect_harbor_verified_tasks.py`, `prepare_harbor_osb_dataset.py`, `prepare_harbor_verified_tasks.sh` | setup | Harbor task sets |
| `code/make_micro_tasks.py`, `code/validate_micro_tasks.py` | check | Harbor micro-tasks for plumbing runs; nop/oracle check of tasks on real pods |
| `tool-use/workplace-assistant-sft.yaml` | run | Tool calls on Gym's workplace assistant (27 simulated tools) |
| `chat/ultrafeedback-rubric.yaml`, `-sft.yaml`, `-sft-2judges.yaml`, `-sft-3judges.yaml` | run | General chat rewarded by one, two or three LLM judges on rubrics, AlpacaEval validation |
| `chat/prepare_ultrafeedback.py`, `chat/prepare_alpaca_eval.py`, `chat/ultrafeedback-few-shot.txt` | setup | Rows and few-shot prompt of the rubric tasks |
| `lean/lean-sft.yaml` | run | Lean 4 proving with Gym math_formal_lean |
| `frames/apertus2-system.json`, `-high.json` | run | System frame of the Apertus 2 format: thinking medium, high |
| `frames/apertus2-system-sciworld-act.json`, `-high.json` | run | The same with ScienceWorld's `act` tool declared |
| `common/stage_judge_model.sh` | setup | Downloads a judge model to scratch |
| `common/strip_few_shot.py` | setup | Removes the few-shot prompt from prepared Gym rows (for SFT checkpoints) |
| `common/score-text.yaml` | check | Teacher-forced log-probabilities of plain text (not a training task) |
| `README.md` | doc | The RL experts and their folders; medical, safety, knowledge and translation have none yet |

## `recipes/` (model and algorithm, 34 files)

| Files | Role | Purpose |
|---|---|---|
| `apertus2-chonk-sftv02-grpo.yaml` | run | Algorithm of the STEM runs on SFT v0.2 (512 x 16, one optimizer step, lr 2e-6, staleness 4) |
| `apertus2-chonk-sftv02-18315-*.yaml` | run | SFT v0.2 on 32 nodes, with 1, 2 or 4 judge nodes; Reasoning Gym variant |
| `apertus2-chonk-sft-vllm-*.yaml`, `apertus2-chonk-sft19374-*.yaml` | run | SFT checkpoints v0.1 / 19374 with AdamW or Muon, for DAPO, code, tools, rubric |
| `apertus2-chonk-vllm-*.yaml`, `apertus2-chonk-megatron-*.yaml` | run | Pretraining checkpoint, vLLM or Megatron inference, AdamW or Muon, with judges |
| `apertus2-megachonk-*.yaml` | run | MegaChonk (the larger model) recipes, DAPO and STEM at 16K on 36/64 nodes |
| `apertus2-chonk-sftv02-stemmix-run4-100-*.yaml` | run | Continuing from the run-4 checkpoint at step 100 (chemistry task) |

Most recipes are a chain of small files ending in `experiments/grpo-chonk/` and NeMo-RL's
`grpo_math_1B.yaml`; `DRY_RUN=1 bash run.sh <task>` (its preflight report) lists the files of a run's chain.

## `backends/nemo-rl/` (launching NeMo-RL)

| File | Role | Purpose |
|---|---|---|
| `launch.sh` | run | Config + node count → preflight, container flags, `sbatch ray.sub` with the driver command |
| `config.sh` | run | Merges a defaults chain as NeMo-RL does; freezes config copies for a job |
| `preflight.sh` | run | Prints the settings that matter and fails on known mistakes, before submission |
| `container.sh` | run | Renders the container spec into EDF (Alps) or pyxis flags (HANGANG) |
| `driver_retry.sh` | run | Restarts a failed driver in the same allocation from the latest checkpoint |
| `failure_report.sh` | run | Summary file of a failed driver attempt |
| `numa_watch.sh`, `hang_watch.sh` | run | Per node: host memory on GPU HBM, Ray log copies; stack dumps when the job looks stuck, stops a hung driver |
| `strtok_ts.c` | run | Thread-safe strtok preloaded into job processes (UCCL NIC parsing on HANGANG) |
| `patches/0001-0002` | run | `ray.sub` for the Alps container engine / pyxis; applied to the clone by setup |
| `pin`, `env`, `edf.toml.template`, `check_env.sh` | run (stock) | The stock NeMo-RL stack (NGC image) used by `experiments/grpo-math` |
| `strip_optimizer.py` | tool | Removes optimizer state from finished checkpoints (frees scratch) |
| `apertus2/env` | run | The Apertus 2 stack: clone paths, image, overlay, driver |
| `apertus2/pin` | run | NeMo-RL fork commit + PRs, Automodel, Megatron-LM-MoE, Bridge, vLLM files, Gym, apertus-common commits |
| `apertus2/setup.sh` | run | Builds or updates the clone (PRs, patches by subject), Megatron, Bridge, vLLM files, image |
| `apertus2/launch.sh` | run | `launch.sh` with the Apertus 2 stack variables |
| `apertus2/driver.sh` | run | Driver entry in the container: overlay, W&B key, unbuffered Python |
| `apertus2/pydeps.sh` | run | Python overlay the image lacks (nemo-lens, soundfile; Gym, Reasoning Gym, apertus-common when needed) |
| `apertus2/edf.toml.template` | run | Container image, mounts and environment of the Apertus 2 stack |
| `apertus2/contexts.tsv` | run | CP, scheme, packing, memory fraction per context length |
| `apertus2/kcp/setup.sh`, `use.sh`, `0001-*.patch` | run | KCP context parallelism: Megatron copy with PR 112, flash-linear-attention 0.5.2 |
| `apertus2/setup-gym.sh` | run | NeMo Gym checkout for Gym tasks (`script:`) |
| `apertus2/patches/nemo-rl/0001-0061` (53 files) | run | Our NeMo-RL changes on top of the pin, applied in order |
| `apertus2/patches/gym/` (15), `automodel/` (2), `megatron/` (6), `vllm/` (1), `uccl/` (2) | run / setup | Changes to Gym, Automodel, Megatron, the vLLM Apertus 2 model, UCCL |
| `apertus2/uccl-fix/build_image.sh`, `rebuild_uccl.sh` | setup | Builds the image jobs run (UCCL rebuilt with the two patches) |
| `apertus2/check_env.sh`, `check_env_job.sh` | check | Stack sanity check on one GPU node (also runs `download_model.sh`, `check_judge.sh`) |
| `apertus2/download_model.sh` | setup | Downloads an HF repository from a compute node |
| `apertus2/check_judge.sh`, `check_judge.py`, `judge_stats.py` | check / tool | Judge test before a run; judge load summary after |
| `apertus2/check_rg_verifier.sh` | check | Test of the Reasoning Gym verifier (patch 0034) |
| `sandbox/*` (9) | setup / run | Code sandbox node for code_gen: server, harness, data, start script, attack test |
| `lean/*` (11) | setup / run | Lean 4 sandbox (NeMo-Skills image) and its data |

## `experiments/` (records and experiment tooling)

| Folder | Role | Purpose |
|---|---|---|
| `runs.md`, `megachonk.md` | history | Every run with its command, job, result; MegaChonk notes |
| `grpo-chonk/` (29) | run | The config chain under every Chonk recipe (GSM8K → DAPO → FP8 → Apertus 2 format), `stage_model.sh` (used by every run), README of the vLLM/Megatron and AdamW/Muon comparison |
| `long-context/` | run / tool | `launch-code-ctx.sh` (coding at 32K-128K) and its node recipes, checkpoint export, `eval/` (vLLM-only validation), `tools/` (rollout index and statistics, step timings, network and pod probes), README with the context measurements |
| `enigmata/` (16) | setup / check / history | Puzzle pool generation and dataset build, pass-rate scoring, environment tests, its debug launcher |
| `scienceworld/` (18) | setup / check / history | ScienceWorld install, data build, probes and scoring, environment tests, its debug launcher |
| `chonk-sft/` (9) | check | Serving and chatting with SFT checkpoints, AIME check, SFT data audit, conversion |
| `trainer-memory/launch.sh` | check | Trainer memory tests on the 16K coding task |
| `stem-mix-run5/gen_chem.sbatch` | setup | Chemistry rows job for run 5 |
| `grpo-math/` (3) | history | Qwen3 GRPO on math with the stock stack |
| `vlm-geo3k/` (2) | history | Qwen VL configs, marked broken in NeMo-RL v0.7.0 |
| `swe_terminal_tool_use/README.md` | doc | Notes on the SWE/terminal tool-use tasks |

## Still out of place (next cleanup steps)

- `experiments/grpo-chonk/stage_model.sh` is part of every run but lives in an experiment folder.
- `experiments/long-context/launch-code-ctx.sh` duplicates `run.sh` for coding (sandbox job, KCP, node
  layouts); the code domain needs `envs/code/base.yaml`/`main.yaml` and tiers like STEM.
- `experiments/enigmata/launch.sh`, `scienceworld/launch.sh`, `debug16.sh`, `probe.sh`,
  `trainer-memory/launch.sh`: more launchers with their own variables.
- `experiments/long-context/tools/` are general run-analysis tools, not long-context ones.
- `backends/nemo-rl/sandbox/` and `lean/` are services of the code and Lean domains.
- Two stacks (stock NeMo-RL for Qwen, Apertus 2): the stock one is only used by `experiments/grpo-math`.
- The recipe chain under `experiments/grpo-chonk/` (13+ files deep) could be flattened into one recipe
  per model.

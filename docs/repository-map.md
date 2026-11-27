# Repository map

Every file of the repository and its role (364 files, branch `restructure`, 2026-11-20). Roles:
**run** = used by every training run, **setup** = prepares sources, images, models or data,
**check** = a test or probe before or after a run, **tool** = analysis of a finished run,
**history** = kept as the record of past runs, not used by new ones.

## How a run passes through the files

```
bash run.sh stem      (experts/main.yaml + experts/stem/base.yaml)
 └ run.sh ─ sites/load.sh (site.env, sites/alps.env, env/stack.env)
   ├ config/config.sh            merges the task's defaults chain, copies every config file for the job
   ├ the recipe's nodes: + the size tier             node counts (tiers in run.sh)
   ├ the recipe's context_parallel (KCP above CP 4)  CP, packing, memory fraction
   ├ setup/setup.sh              NeMo-RL clone at the pin + patches, Megatron, Bridge, vLLM files, image
   ├ setup/stage_model.sh        model export staged on scratch
   ├ setup/setup-gym.sh          Gym checkout (tasks with `script:`)
   └ launch.sh
       ├ config/preflight.sh     config check (settings, dataset files), report next to the Slurm log
       ├ config/container.sh     container flags from env/edf.toml.template
       └ sbatch <clone>/ray.sub → job/numa_watch.sh (job/hang_watch.sh) on every node
           └ job/driver_retry.sh → job/driver.sh (job/pydeps.sh) → python examples/run_grpo.py
(paths under backends/nemo-rl/ unless they start at the top)
```

## Top level

| File | Role | Purpose |
|---|---|---|
| `run.sh` | run | The entry point: task file in, Slurm job out (rl-infra keys, tiers, context, config copies, setup, launch) |
| `README.md` | doc | Quick start, task files and tiers, NeMo-RL development, cluster notes |
| `AGENTS.md`, `CLAUDE.md` | doc | Guidelines for agents and people: where to work, what needs the maintainers, patches, runs, commits (`CLAUDE.md` imports `AGENTS.md`) |
| `site.env.example` | doc | Template of the personal, gitignored `site.env` (account, scratch, W&B) |
| `.gitignore` | — | Keeps `site.env` and generated files out of git |
| `docs/nemo-rl-architecture.md` | doc | How NeMo-RL works inside (driver, workers, refit) |
| `docs/repository-map.md` | doc | This file |
| `docs/MIGRATION.md` | doc | What changed with the restructure and how to move a branch from the old layout |
| `docs/runs.md` | history | Every run with its command, job and result |
| `docs/megachonk.md` | history | The MegaChonk runs: checkpoint, layouts, results |
| `docs/long-context.md` | history | GRPO at 32K-128K: patches, measurements, what limits a step |

## `sites/` (cluster profiles)

| File | Role | Purpose |
|---|---|---|
| `README.md` | doc | The profiles, the load order, how to add a cluster |
| `sites/load.sh` | run | Loads the settings in order: environment, `site.env`, site profile, stack env files |
| `sites/hangang.env` | run | KISTI HANGANG: account, partition, Slurm flags, scratch, pyxis, NCCL/libfabric fixes, mounts, image, Slingshot suffix, `RL_DATA_DIR` |
| `sites/alps.env` | run | CSCS Alps: the same for Clariden (container engine EDF) |

## `experts/` (what to train on, one folder per RL expert)

Every expert folder has `base.yaml` (its official run), `dev/` (variants), `archive/` (task files of runs that
happened, as they ran) and `scripts/` (data builders and checks); empty folders hold a `.gitkeep`.

| File | Role | Purpose |
|---|---|---|
| `main.yaml` | run | The run every expert shares: tier, time limit and context (resources), checkpoints, rollout dump, validation period; run.sh puts it under an expert's files |
| `README.md` | doc | The layout and the experts with their status |
| `stem/base.yaml` | run | STEM official run: 8 training sets, 3 validation sets, all environments, thinking level per dataset (`frames`), 135 steps (resources from main.yaml) |
| `stem/dev/gsm8k.yaml`, `stem/dev/dapo-math.yaml` | run | GSM8K and DAPO-Math on the pretraining checkpoint (the quick start) |
| `stem/dev/gsm8k-sft.yaml`, `dapo-math-sft.yaml`, `dapo-gsm8k-sft.yaml` | run | The same on an SFT checkpoint; DAPO + GSM8K 70/30 mix |
| `stem/dev/chem-rdkit-sft.yaml`, `stem/dev/reasoning-gym-sft.yaml` | run | Single-environment tasks: chemistry alone, Reasoning Gym alone (older layout) |
| `stem/archive/run5-32k-job1299180.yaml` | history | Run 5 at 32K (job 1299180) in the expert layout; its merged config equals the run's |
| `stem/archive/stem-mix-run1.yaml` … `run5.yaml`, `-run3-base`, `-run3-sciworld`, `-enigmata`, `-openqa-judge` | history | Task files of the STEM runs 1-5 and their trials, as they ran |
| `stem/frames/apertus2-system-sciworld-act.json`, `-high.json` | run | ScienceWorld's system frames: thinking medium or high, with the `act` tool declared (STEM only) |
| `stem/scripts/prepare_chem_rdkit.py`, `gen_chem_rdkit.py`, `gen_chem_rdkit.sbatch`, `prepare_reasoning_gym.py` | setup | Chemistry and Reasoning Gym rows |
| `stem/scripts/enigmata/` (15) | setup / check / history | Enigmata puzzle pool and dataset build, pass-rate scoring (its one-node recipe and tasks), environment test, the launcher of the Enigmata trial |
| `stem/scripts/scienceworld/` (18) | setup / check / history | ScienceWorld install, data build, episode probes and scoring, environment test, the launchers of its trial and debug run |
| `stem/scripts/eval/` | tool | Validation of an HF export with vLLM only (the run's own `grpo.validate()`), benchmark sets (AIME, MATH-500, MMLU-Pro, held-out training sets) |
| `stem/README.md` | doc | The STEM expert's files |
| `code/base.yaml`, `chat/base.yaml`, `tool-use/base.yaml`, `lean/base.yaml` | template | Take the latest archived run through `defaults` until written |
| `code/scripts/sandbox/*` (9) | setup / run | Code sandbox node for code_gen: server, harness, data, start script, attack test |
| `lean/scripts/*` (11) | setup / run | Lean 4 sandbox (NeMo-Skills image) and its data |
| `code/archive/code-gen-sft.yaml`, `code-gen-osb-sft.yaml` | history | Competitive coding on Gym's code_gen, tests on a sandbox node or on OpenSandbox pods |
| `code/archive/harbor-osb-sft.yaml`, `harbor-verified-sft.yaml` | history | Harbor terminal/SWE tasks with Terminus-2 on OpenSandbox |
| `code/README.md` | doc | How to run the Harbor SWE/terminal tasks (prerequisites, credentials, commands) |
| `code/scripts/` (5 files) | setup / check | Harbor task sets; micro-tasks and their nop/oracle check on real pods |
| `code/scripts/long-context/launch-code-ctx.sh`, `recipes/` (4) | run / history | Coding on code_gen at 32K-128K (sandbox node or pods, CP, KCP, driver node) and its node layouts |
| `code/scripts/trainer-memory.sh` | check | Trainer memory tests on the 16K coding task |
| `chat/archive/ultrafeedback-rubric.yaml`, `-sft.yaml`, `-sft-2judges.yaml`, `-sft-3judges.yaml` | history | General chat rewarded by one, two or three LLM judges on rubrics, AlpacaEval validation |
| `chat/scripts/prepare_ultrafeedback.py`, `prepare_alpaca_eval.py`, `ultrafeedback-few-shot.txt` | setup | Rows and few-shot prompt of the rubric tasks |
| `chat/scripts/check_rubric_judge.sh`, `.py` | check | The rubric judge model on one GPU, scored with Gym's rubric_judge prompts, before a run |
| `tool-use/archive/workplace-assistant-sft.yaml` | history | Tool calls on Gym's workplace assistant (27 simulated tools) |
| `lean/archive/lean-sft.yaml` | history | Lean 4 proving with Gym math_formal_lean |
| `medical/`, `safety/`, `knowledge/`, `translation/` `base.yaml` | template | Placeholders: medical multiple choice; safety, knowledge and translation graded by the shared judge |

## `shared/` (pieces the task files take in; not experts)

| File | Role | Purpose |
|---|---|---|
| `README.md` | doc | What each folder holds |
| `frames/apertus2-system.json`, `-high.json` | run | System frame of the Apertus 2 format: thinking medium, high |

| `judges/llm-judge.yaml` | run | The LLM judge grader (`env.llm_judge`: verdict, scoring, frames), taken in by STEM and the judge-based templates |
| `judges/judge_stats.py` | tool | Judge load summary after a run (from a server's `stats_path` file) |
| `judges/servers/gemma-4-26B-A4B-it.yaml` | run | The judge server with gemma-4-26B-A4B-it (`env.extra_judge_pools.gemma-4-26B-A4B-it`, server `judge_gemma-4-26B-A4B-it`: one replica per GPU; the expert sets its `num_nodes`) |
| `prompts/gsm8k-five-shot.txt`, `stem-mix-four-shot.txt` | run / history | Few-shot prompts read at run time (GSM8K on the pretraining checkpoint; STEM runs 1-2) |
| `scripts/stage_judge_model.sh` | setup | Downloads a judge model to scratch |
| `scripts/strip_few_shot.py` | setup | Removes the few-shot prompt from prepared Gym rows (for SFT checkpoints) |
| `scripts/convert_rl_ckpt.sh` | tool | NeMo-RL checkpoint to an HF export |
| `scripts/analysis/` (5) | tool | Rollout index and statistics, per-step timings and rewards, vLLM load, rank start stagger |
| `scripts/network/` (3) | check | Network interfaces and Ray transfer speed between two nodes |
| `scripts/sandbox/` (4) | check | OpenSandbox pod health and capacity |
| `sft-checkpoint/` (8) | check | Serving and chatting with SFT checkpoints, AIME check, SFT data audit, conversion; README: the Apertus 2 message format |
| `debug/score-text.yaml` | check | Teacher-forced log-probabilities of plain text (not a training task) |

## `recipes/` (how to train: model and algorithm)

| File | Role | Purpose |
|---|---|---|
| `chonk-sftv02.yaml` | run | The active recipe, one file on NeMo-RL's defaults: Chonk SFT v0.2; Megatron trainer (TP1 PP2 EP16, Adam lr 2e-6 constant, optimizer on CPU); vLLM FP8 (EP4 per node); async GRPO (512 x 16, one optimizer step, staleness 4, TIS 2.0) |
| `megachonk.yaml` | template | MegaChonk pretraining checkpoint: TP4 PP8 EP16, vLLM TP4 EP32 replicas of 8 nodes, 4,096 tokens per GPU; no expert yet |
| `README.md` | doc | What a recipe is, the active ones, how to add a variant |
| `archive/*.yaml` (34) | history | Recipes of earlier runs as they were (pretraining checkpoint, SFT v0.1 / 19374 / v0.2, MegaChonk, AdamW or Muon, judge-node layouts); archived task files and launchers use them |
| `archive/chain/*.yaml` (27) | history | The experiment chain they build on (GSM8K -> DAPO -> FP8 -> Apertus 2 format), formerly `experiments/grpo-chonk/`; `README.md` there: the GSM8K -> DAPO comparison (vLLM against Megatron inference, AdamW against Muon) |

## `backends/nemo-rl/` (launching NeMo-RL)

| File | Role | Purpose |
|---|---|---|
| `README.md`, `patches/README.md` | doc | The folders by when they run; how patches are applied and added |
| `launch.sh` | run | Config + node count → preflight, container flags, `sbatch ray.sub` with the driver command |
| `env/stack.env` | run | The stack: clone paths, image, overlay, driver, outputs (`OUT`) |
| `env/pin` | run | NeMo-RL fork commit + PRs, Automodel, Megatron-LM-MoE, Bridge, vLLM files, Gym, apertus-common commits |
| `env/edf.toml.template` | run | Container image, mounts and environment |
| `setup/setup.sh` | run | Builds or updates the clone (PRs, patches by subject), Megatron, Bridge, vLLM files, image |
| `setup/stage_model.sh` | run | Stages the model export on scratch (copied small files, linked shards; modeling code check), every launch |
| `setup/setup-gym.sh` | run | NeMo Gym checkout for Gym tasks (`script:`) |
| `config/config.sh` | run | Merges a defaults chain as NeMo-RL does; freezes config copies for a job |
| `config/container.sh` | run | Renders the container spec into EDF (Alps) or pyxis flags |
| `config/preflight.sh` | run | Prints the settings that matter and fails on known mistakes and missing dataset files, before submission |
| `job/driver.sh` | run | Driver entry in the container: overlay, W&B key, unbuffered Python |
| `job/pydeps.sh` | run | Python overlay the image lacks (nemo-lens, soundfile; Gym, Reasoning Gym, apertus-common, KCP's flash-linear-attention when needed) |
| `job/driver_retry.sh` | run | Restarts a failed driver in the same allocation from the latest checkpoint |
| `job/failure_report.sh` | run | Summary file of a failed driver attempt |
| `job/numa_watch.sh`, `job/hang_watch.sh` | run | Per node: host memory on GPU HBM, Ray log copies; stack dumps when the job looks stuck, stops a hung driver |
| `diagnostics/check_env.sh`, `check_env_job.sh` | check | Stack sanity check on one GPU node; `check_env_job.sh` runs it, the others here or a repository script as a one-node job |
| `setup/download_model.sh` | setup | Downloads an HF repository from a compute node |
| `patches/ray-sub/` (2) | run | NeMo-RL's `ray.sub` for the Alps container engine / pyxis, applied first |
| `patches/nemo-rl/0001-0067` (59) | run | Our NeMo-RL changes on top of the pin, applied in order |
| `patches/megatron/` (7) | run | Megatron-LM-MoE: optimizer and checkpoint fixes, 0007 = KCP context parallelism for the KDA layers |
| `patches/gym/` (15), `automodel/` (2), `vllm/` (1) | run / setup | Changes to Gym, Automodel, the vLLM Apertus 2 model |

## Still out of place (next cleanup steps)

- `experts/code/scripts/long-context/launch-code-ctx.sh` duplicates `run.sh` for coding (sandbox job, KCP, node
  layouts); the code expert needs its `experts/code/base.yaml` written (now a template) and tiers like STEM.
- `experts/stem/scripts/enigmata/launch.sh`, `scienceworld/launch.sh`, `debug16.sh`, `probe.sh`,
  `experts/code/scripts/trainer-memory.sh`: launchers with their own overrides (they now run from this repository).

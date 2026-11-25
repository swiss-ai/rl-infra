# Migrating a branch to the restructured layout

The branch `restructure` (started from main at `965e15c`, 2026-11-19) reorganises where task files live, what
`run.sh` does, and how NeMo-RL changes reach a run. This file lists every change a branch based on the old main has
to follow, and the steps to move one over. `docs/repository-map.md` describes the new layout file by file.
A person or an agent session porting a branch follows section 3 with the tables of sections 5 and 6; section 7 is
the checklist to hand an agent session.

## 1. What changed

| Area | Before | Now |
|---|---|---|
| Task files | `envs/<task>.yaml`, all side by side | `experts/<expert>/`: one folder per RL expert (medical, safety, chat, stem, code, knowledge, tool-use, translation, lean); inside: `base.yaml` (the official run), `dev/`, `archive/` (runs as they ran), `scripts/` (data builders); `experts/main.yaml` holds the run settings all experts share (`experts/README.md`) |
| Launch command | `bash run.sh envs/<task>.yaml` or a launcher | `bash run.sh <expert>` (the expert's `base.yaml` with `experts/main.yaml` under it), `bash run.sh experts/<expert>/dev/<file>.yaml`, `bash run.sh experts/<expert>/archive/<file>.yaml` |
| Shared pieces | `envs/apertus2-system*.json`, `envs/stage_judge_model.sh`, ... | `shared/frames/`, `shared/judges/llm-judge.yaml`, `shared/prompts/`, `shared/scripts/`, `shared/debug/` |
| Launch settings | launcher variables (`CP`, `CAP`, `GEN_NODES`, `STEPS`, `SAVE_BY`, ...) in a launcher per experiment | rl-infra keys read by `run.sh`: `size`, `job_time`, `context` in `experts/main.yaml` (a dev file may change them), `recipe` in the expert's `base.yaml`; Hydra overrides on the command line |
| Recipes | 34 recipes, each a few keys on a chain of up to 14 files (`recipes/`, `experiments/grpo-chonk/`) | one flat file per active model, on NeMo-RL's defaults only: `recipes/chonk-sftv02.yaml`; the rest in `recipes/archive/` and `recipes/archive/chain/` (`recipes/README.md`) |
| Node layouts | one recipe per node split | `size:` a tier of the recipe's base layout (`nodes:`; Chonk 8+8): lite 8+4, cheap 8+8, expensive 16+16, ultra 16+32, or explicit counts such as `16+64`; judge nodes come from the task |
| Context length | `CP`, `KCP`, `CAP`, packing set by hand | `context:` in the task; the recipe's `context_parallel` gives CP (context / tokens_per_gpu, rounded up to a power of 2), KCP above CP 4, packing = context and the memory fraction (the next power of 2 up: 50K gets the 64K settings) |
| Run modes | `DRY_RUN=1` ran setup and printed the job; launchers had `DEBUG=1` by default | plain `run.sh` = full run (W&B required); `DEBUG=1` = debug run; `DRY_RUN=1` = everything shown, no setup, no job |
| Config at job start | the job read the recipe and task chain from their original paths when it started | the job reads copies made at launch (`$OUT/configs/<run>-<time>/`, with the setup log) |
| NeMo-RL code | experiment clones (`nemo-rl-apertus2-stem32k`, ...) carried commits that were not patches in rl-infra | every change is a patch in `backends/nemo-rl/patches/nemo-rl/`; 0034 and 0057-0066 added (Reasoning Gym verifier, claim_multichoice, llm_judge, Enigmata, ScienceWorld), after main's 0052-0056; 0067 (drop_long_prompts) added later; the next free number is 0068 |
| KCP | `experiments/long-context/setup-kcp.sh`, `kcp/`, `KCP=1` in the launchers | Megatron patch 0007 (`backends/nemo-rl/patches/megatron/`) and flash-linear-attention 0.5.2 in the overlay, used automatically for a context above 32K |
| Backend | `backends/nemo-rl/` with the stock NGC stack and the Apertus 2 stack in `apertus2/` | one stack, grouped by when its files run: `launch.sh`, `env/`, `setup/`, `config/`, `job/`, `diagnostics/`, `patches/` (section 6) |
| Data paths | absolute paths in task files (`/scratch/hanint06/data/...`) | `${oc.env:RL_DATA_DIR}/...` (set in `sites/<site>.env` or `site.env`); your own files under `${oc.env:RL_USER_HOME}` |

Removed: `experiments/long-context/launch-stem-ctx.sh` and its three node-split recipes, the launchers of
`experiments/stem-mix-run4`, `stem-mix-run5` and `openqa-judge`. Run commands in `docs/runs.md` (formerly `experiments/runs.md`) that name
them refer to the commit before the restructure (`965e15c` on main, `f1ec52e` on `stem5`).

## 2. Behaviour changes that can surprise you

- **A launch without variables is a full run.** `bash run.sh <task>` names the run without `-DEBUG` and refuses to
  start without `WANDB_PROJECT` (`site.env`). For a debug run pass `DEBUG=1`: the name ends in `-DEBUG`, and unless
  the command line says otherwise the run has 3 steps, no validation, no checkpoints and a 2-hour limit.
- **A launcher of your own that calls `run.sh`** and sets steps, validation and names itself must call it with
  `DEBUG=0` (as `experiments/long-context/launch-code-ctx.sh` now does); otherwise an exported `DEBUG=1` adds the debug
  defaults before your overrides, and validation would be switched off.
- **`DRY_RUN=1` no longer runs setup.** It needs an existing NeMo-RL clone; on a fresh account launch once or run
  `backends/nemo-rl/setup/setup.sh`.
- **The checkpoint deadline follows the time limit.** For a run with checkpoints, `run.sh` sets
  `checkpointing.checkpoint_must_save_by` to 30 minutes before `JOB_TIME` (or the task's `job_time`), over the value in
  the recipe. A different deadline goes on the command line, quoted: `checkpointing.checkpoint_must_save_by="'00:11:00:00'"`.
- **Default run names** are the task's path under `experts/` and the time (`stem-main-20261120-081500`); `RUN` still
  names a run.
- **`sft_export` and `cluster.num_nodes` are read from the merged config**, through `defaults`, not from the recipe
  file alone. A recipe that does not set `sft_export` itself now inherits its parent's.
- **Edits after launch do not reach the job.** To change a queued run, cancel and launch again. The code (the
  NeMo-RL clone, Megatron) is still read when the job starts: do not change the clone a queued job uses.
- **Environment names must be registered in NeMo-RL** (`nemo_rl/environments/utils.py`). The math environment is
  registered twice (`math`, `math_default`), so a task has two math configurations at most, plus `reasoning_gym`,
  `llm_judge`, `enigmata`, `scienceworld`, `nemo_gym` and the others listed there.
- **`policy.model_name` follows `sft_export`.** For a recipe with `sft_export`, run.sh sets `policy.model_name` to the
  staged export (`$SCRATCH/models/<name>`); `sft_export` may now be absolute. A recipe sets its context parallelism with
  `context_parallel: {tokens_per_gpu, memory_fraction}` (the context tables are gone); padding now includes TP with sequence parallelism (2 x CP x TP).
- **`driver_retry.sh`** now reads the quoted deadline and shortens it for a retry, which it never did before.

## 3. Steps to move a branch

1. **Rebase or merge onto `restructure`** (main once it is merged). Git follows the renames for files you edited;
   expect conflicts in `experiments/runs.md`, `README.md` and `run.sh`.
2. **Move the task files you added** into the expert they train (`experts/README.md`): a variant in `dev/`
   (`defaults: ../base.yaml` in the expert layout), a finished run's file in `archive/` (as it ran), data builders in
   `scripts/`. What becomes the expert's official run goes into its `base.yaml`; settings all experts share go into
   `experts/main.yaml`, not into each expert. Then fix the paths in the moved files:
   ```bash
   # recipe two levels deeper in dev/ and archive/
   sed -i 's#^defaults: \.\./recipes/#defaults: ../../../recipes/#' experts/<expert>/<dev or archive>/<task>.yaml
   # shared pieces
   sed -i -e 's#/envs/apertus2-system#/shared/frames/apertus2-system#g' \
          -e 's#envs/stage_judge_model\.sh#shared/scripts/stage_judge_model.sh#g' \
          -e 's#envs/strip_few_shot\.py#shared/scripts/strip_few_shot.py#g' <files>
   ```
   Any other `envs/<file>` you reference: see the table in section 5.
3. **Use the new keys where they replace launcher variables:** `recipe:` (instead of naming the recipe in
   `defaults`), `context:` (instead of CP, KCP, cap and packing), `size:` (instead of a recipe per node split),
   `job_time:` (instead of `JOB_TIME` and `SAVE_BY`). Settings that were fixed overrides of a launcher (batch, lr,
   staleness) belong in a recipe (example: `recipes/chonk-sftv02.yaml`). A task in the older layout
   (recipe in `defaults`, no keys) still runs as before.
4. **Delete your launcher** once its variables are keys or overrides. Keep a script only for what `run.sh` cannot
   do (for example starting a sandbox job first).
5. **Turn NeMo-RL commits into patches.** Every commit in a clone of your own that a run depends on becomes a patch
   with the next free number (0068 on), with a subject no other patch has (`setup.sh` matches patches by subject):
   ```bash
   git -C <your clone> format-patch --start-number 68 -o backends/nemo-rl/patches/nemo-rl/ <base>..<head>
   ```
   Check that the series is complete: in a fresh worktree at the pin (the PR merges of `backends/nemo-rl/env/pin`), apply every
   patch in order with `git am -3`, and compare the tree with the clone you tested on (`git diff --stat`). Patches
   numbered 0015-0017 and 0035-0039 on other branches are not on main: renumber them after 0067.
6. **Replace personal paths:** datasets under `${oc.env:RL_DATA_DIR}/...`, files in your home under
   `${oc.env:RL_USER_HOME}/...`, the judge model under `${oc.env:SCRATCH}/models/...`.
7. **Check before the first run:** `DRY_RUN=1 bash run.sh experts/<expert>/<task>.yaml` prints nodes, context, the
   overrides `run.sh` adds, the config copies and the preflight report. To confirm nothing changed, merge the old and
   the new config with `backends/nemo-rl/config/config.sh` (`config_resolve`) and compare them leaf by leaf
   (`yq r <file> --printMode pv '**' | sort`).
8. **A new expert** starts from its template in `experts/<expert>/` (or copies one): fill in the datasets marked
   TODO. `run.sh` submits no run while a `data_path` is missing (`backends/nemo-rl/config/preflight.sh`).

## 4. Branches on origin that are not migrated yet (2026-11-25)

A trial merge of each branch into `restructure` (`git merge-tree`), with the number of files in conflict. A branch
without conflicts can still need work: files it adds under `experiments/`, `envs/` or an old backend path stay there,
and a launcher that calls an old path no longer runs.

| Branch | Author | Commits not on main | Files in conflict |
|---|---|---|---|
| `chonk-sc-pi` | Philip Mueller | 162 | 57 |
| `chonk-sc-harbor025` | Philip Mueller | 159 | 53 |
| `apertus-v11` | Tommaso Bendinelli | 60 | 7 |
| `terminus2-verified-1k` | Philip Mueller | 6 | 6 (its NeMo-RL patches are main's 0052-0056) |
| `restructure-megachonk-sft` | davmels | 68 | 4 (NeMo-RL 0062 and Megatron 0007 taken twice) |
| `lean-kimina-verifier` | Chengshang Liu | 18 | 4 |
| `or-benign-balance`, `feat/oss-validator-or-single-turn` | Emma Anastassova | 6, 24 | 0 |
| `grpo_mt` | Javi GG | 4 | 0 |
| `reasoning-gym-native` | Antonio Lopardo | 3 | 0 |

## 5. Moved files (main `965e15c` and `stem5` → now)

| Before | Now |
|---|---|
| `envs/stem-mix-run1.yaml` … `stem-mix-run5.yaml`, `stem-mix-run3-base.yaml`, `stem-mix-run3-sciworld.yaml`, `stem-mix-enigmata.yaml`, `stem-mix-openqa-judge.yaml` | `experts/stem/archive/` |
| `envs/gsm8k.yaml`, `gsm8k-sft.yaml`, `dapo-math.yaml`, `dapo-math-sft.yaml`, `dapo-gsm8k-sft.yaml` | `experts/stem/dev/` |
| `envs/chem-rdkit-sft.yaml`, `reasoning-gym-sft.yaml` | `experts/stem/dev/` |
| `envs/prepare_chem_rdkit.py`, `gen_chem_rdkit.py`, `prepare_reasoning_gym.py` | `experts/stem/scripts/` |
| `envs/gsm8k-five-shot.txt`, `stem-mix-four-shot.txt` | `shared/prompts/` |
| `envs/code-gen-sft.yaml`, `code-gen-osb-sft.yaml`, `harbor-osb-sft.yaml`, `harbor-verified-sft.yaml` | `experts/code/archive/` |
| `envs/collect_harbor_verified_tasks.py`, `prepare_harbor_osb_dataset.py`, `prepare_harbor_verified_tasks.sh`, `make_micro_tasks.py`, `validate_micro_tasks.py` | `experts/code/scripts/` |
| `envs/ultrafeedback-rubric.yaml`, `-sft.yaml`, `-sft-2judges.yaml`, `-sft-3judges.yaml` | `experts/chat/archive/` |
| `envs/ultrafeedback-few-shot.txt`, `prepare_ultrafeedback.py`, `prepare_alpaca_eval.py` | `experts/chat/scripts/` |
| `envs/workplace-assistant-sft.yaml` | `experts/tool-use/archive/` |
| `envs/lean-sft.yaml` | `experts/lean/archive/` |
| `envs/apertus2-system.json`, `-high.json` | `shared/frames/` |
| `envs/apertus2-system-sciworld-act.json`, `-sciworld-act-high.json` | `experts/stem/frames/` (only ScienceWorld declares the `act` tool) |
| `envs/stage_judge_model.sh`, `envs/strip_few_shot.py` | `shared/scripts/` |
| `envs/score-text.yaml` | `shared/debug/` |
| the judge blocks of `envs/stem-mix-run4.yaml` / `run5.yaml` (`env.llm_judge`, `env.judge_pool`) | `shared/judges/llm-judge.yaml` (the grader) and `shared/judges/servers/gemma-4-26B-A4B-it.yaml` (the server), both in the expert's `defaults:`; the server's nodes (`env.extra_judge_pools.<model>.num_nodes`) and the grader's server (`env.llm_judge.judge_pool`) in its base.yaml |
| `experiments/long-context/setup-kcp.sh`, `experiments/long-context/kcp/` | Megatron patch `backends/nemo-rl/patches/megatron/0007-*` (flash-linear-attention 0.5.2 in the overlay) |
| `experiments/long-context/recipes/sftv02-stem-*.yaml` | `size:` tiers (`experts/main.yaml`) |
| `recipes/apertus2-*.yaml` (all 34) | `recipes/archive/`; the STEM recipe written out as `recipes/chonk-sftv02.yaml` |
| `experiments/grpo-chonk/*.yaml` (27) | `recipes/archive/chain/` |
| `experiments/grpo-chonk/stage_model.sh` | `backends/nemo-rl/setup/stage_model.sh` |
| `experiments/long-context/launch-stem-ctx.sh` | `run.sh stem` (run 5 again: `run.sh experts/stem/archive/run5-32k-job1299180.yaml`) |
| `sizes/cheap.yaml`, `expensive.yaml`, `ultra.yaml` | `size:` tiers computed from the recipe's `nodes:` (old expensive 16+32 = new `ultra`; old cheap 8+16 = `size: 8+16`); no batch scaling |
| `experiments/enigmata/`, `experiments/scienceworld/` | `experts/stem/scripts/enigmata/`, `scienceworld/` (launchers now run from this repository: `recipes/archive/`, the NeMo-RL clone of `site.env`) |
| `experiments/stem-mix-run5/gen_chem.sbatch` | `experts/stem/scripts/gen_chem_rdkit.sbatch` |
| `experiments/long-context/eval/` | `experts/stem/scripts/eval/` |
| `experiments/long-context/launch-code-ctx.sh`, `recipes/code-*.yaml` | `experts/code/scripts/long-context/` |
| `experiments/trainer-memory/launch.sh` | `experts/code/scripts/trainer-memory.sh` |
| `experiments/swe_terminal_tool_use/README.md` | `experts/code/README.md` |
| `experiments/long-context/convert_rl_ckpt.sh` | `shared/scripts/convert_rl_ckpt.sh` |
| `experiments/long-context/tools/` | `shared/scripts/analysis/` (rollouts, timings), `network/` (netcheck, Ray transfer), `sandbox/` (pod probes) |
| `experiments/chonk-sft/` | `shared/sft-checkpoint/` |
| `experiments/grpo-math/`, `experiments/vlm-geo3k/` | removed with the stock NeMo-RL stack (Qwen configs, 2026-11-24) |
| `experiments/grpo-chonk/README.md` | `recipes/archive/chain/README.md` |
| `experiments/runs.md`, `megachonk.md`, `long-context/README.md` | `docs/runs.md`, `docs/megachonk.md`, `docs/long-context.md` |

## 6. Backend files (main → now)

Paths under `backends/nemo-rl/` unless they start with `experts/` or `shared/`.

| Before | Now |
|---|---|
| `apertus2/env` | `env/stack.env` (it also sets `OUT`, which came from the stock `env`) |
| `apertus2/pin`, `apertus2/edf.toml.template` | `env/pin`, `env/edf.toml.template` |
| `apertus2/launch.sh`, `launch.sh` | `launch.sh` (one script; it exports the stack's variables itself) |
| `apertus2/setup.sh`, `stage_model.sh`, `setup-gym.sh`, `download_model.sh` | `setup/` |
| `config.sh`, `container.sh`, `preflight.sh` | `config/`; preflight also checks that every dataset file exists |
| `apertus2/driver.sh`, `apertus2/pydeps.sh`, `driver_retry.sh`, `failure_report.sh`, `hang_watch.sh`, `numa_watch.sh` | `job/` |
| `apertus2/check_env.sh`, `check_env_job.sh` | `diagnostics/`; `CHECK_SCRIPT` takes a repository path |
| `apertus2/check_judge.sh`, `check_judge.py` | `experts/chat/scripts/check_rubric_judge.sh`, `.py` |
| `apertus2/check_rg_verifier.sh` | `experts/stem/scripts/check_rg_verifier.sh` |
| `apertus2/judge_stats.py` | `shared/judges/judge_stats.py` |
| `apertus2/patches/<repository>/` | `patches/<repository>/` (`nemo-rl`, `megatron`, `gym`, `automodel`, `vllm`) |
| `patches/0001-*.patch`, `patches/0002-*.patch` (`ray.sub`) | `patches/ray-sub/` |
| `apertus2/kcp/` (patch, `setup.sh`, `use.sh`) | Megatron patch `patches/megatron/0007-*`; above CP 4 run.sh adds flash-linear-attention 0.5.2 to the overlay |
| `sandbox/` | `experts/code/scripts/sandbox/` |
| `lean/` | `experts/lean/scripts/` |
| `apertus2/uccl-fix/`, `apertus2/patches/uccl/` | outside the repository (the image they built exists) |
| the stock stack: `env`, `pin`, `edf.toml.template`, `check_env.sh`, `examples/`, the top-level `setup.sh` | removed |
| `strip_optimizer.py`, `strtok_ts.c` | removed |
| `APERTUS2_DIR` | `BACKEND_DIR` (= `backends/nemo-rl`) |
| `rl_infra_load "${APERTUS2_DIR}/env"` | `rl_infra_load` (it always loads `env/stack.env`) |
| NeMo-RL patches 0052-0062 of `restructure` | 0057-0067 (main's 0052-0056 come first) |
| `template: true` | removed: preflight refuses a run whose dataset files are missing |

## 7. Checklist for an agent session porting a branch

Give the session this file and `docs/repository-map.md`, then ask it to:

1. Merge `restructure` into the branch (`git merge origin/restructure`; main once it is merged), and resolve the
   conflicts once.
2. For every file the branch adds or changes at an old path, take the new path from sections 5 and 6, `git mv` it,
   and fix the paths inside it (`grep -rn "apertus2/\|envs/\|experiments/"`).
3. Keep every patch number of `restructure`. Renumber the branch's own new patches after the highest number there,
   without changing their subjects (`setup.sh` matches patches by subject).
4. Replace each launcher by a task file with the keys of section 3 (steps 3 and 4).
5. Run `DRY_RUN=1 bash run.sh <task>` for every task the branch runs, then one `DEBUG=1` run.
6. Report what it moved, what it changed inside files, and what it could not place, for the branch's author.

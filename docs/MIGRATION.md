# Migrating a branch to the restructured layout

The branch `restructure` (started from main at `965e15c`, 2026-11-19) reorganises where task files live, what
`run.sh` does, and how NeMo-RL changes reach a run. This file lists every change a branch based on the old main has
to follow, and the steps to move one over. `docs/repository-map.md` describes the new layout file by file.

## 1. What changed

| Area | Before | Now |
|---|---|---|
| Task files | `envs/<task>.yaml`, all side by side | `experts/<expert>/`: one folder per RL expert (medical, safety, chat, stem, code, knowledge, tool-use, translation, lean); inside: `base.yaml` (the official run), `dev/`, `archive/` (runs as they ran), `scripts/` (data builders); `experts/main.yaml` holds the run settings all experts share (`experts/README.md`) |
| Launch command | `bash run.sh envs/<task>.yaml` or a launcher | `bash run.sh <expert>` (the expert's `base.yaml` with `experts/main.yaml` under it), `bash run.sh experts/<expert>/dev/<file>.yaml`, `bash run.sh experts/<expert>/archive/<file>.yaml` |
| Shared pieces | `envs/apertus2-system*.json`, `envs/stage_judge_model.sh`, ... | `shared/frames/`, `shared/judges/llm-judge.yaml`, `shared/prompts/`, `shared/scripts/`, `shared/debug/` |
| Launch settings | launcher variables (`CP`, `CAP`, `GEN_NODES`, `STEPS`, `SAVE_BY`, ...) in a launcher per experiment | rl-infra keys read by `run.sh`: `size`, `job_time`, `context` in `experts/main.yaml` (a dev file may change them), `recipe` and `template` in the expert's `base.yaml`; Hydra overrides on the command line |
| Recipes | 34 recipes, each a few keys on a chain of up to 14 files (`recipes/`, `experiments/grpo-chonk/`) | one flat file per active model, on NeMo-RL's defaults only: `recipes/chonk-sftv02.yaml`; the rest in `recipes/archive/` and `recipes/archive/chain/` (`recipes/README.md`) |
| Node layouts | one recipe per node split | `size:` a tier of the recipe's base layout (`nodes:`; Chonk 8+8): lite 8+4, cheap 8+8, expensive 16+16, ultra 16+32, or explicit counts such as `16+64`; judge nodes come from the task |
| Context length | `CP`, `KCP`, `CAP`, packing set by hand | `context:` in the task; `backends/nemo-rl/apertus2/contexts.tsv` gives CP, KCP, packing and memory fraction (the next row up: 50K gets the 64K settings) |
| Run modes | `DRY_RUN=1` ran setup and printed the job; launchers had `DEBUG=1` by default | plain `run.sh` = full run (W&B required); `DEBUG=1` = debug run; `DRY_RUN=1` = everything shown, no setup, no job |
| Config at job start | the job read the recipe and task chain from their original paths when it started | the job reads copies made at launch (`$OUT/configs/<run>-<time>/`, with the setup log) |
| NeMo-RL code | experiment clones (`nemo-rl-apertus2-stem32k`, ...) carried commits that were not patches in rl-infra | every change is a patch in `backends/nemo-rl/apertus2/patches/nemo-rl/`; 0034 and 0052-0061 added (Reasoning Gym verifier, claim_multichoice, llm_judge, Enigmata, ScienceWorld); the next free number is 0062 |
| KCP | `experiments/long-context/setup-kcp.sh`, `kcp/`, `KCP=1` in the launchers | `backends/nemo-rl/apertus2/kcp/`, used automatically for a context above 32K |
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
  `backends/nemo-rl/apertus2/setup.sh`.
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
  staged export (`$SCRATCH/models/<name>`); `sft_export` may now be absolute. A recipe for another model names its
  context table with `contexts:`; padding now includes TP with sequence parallelism (2 x CP x TP).
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
   with the next free number (0062 on), with a subject no other patch has (`setup.sh` matches patches by subject):
   ```bash
   git -C <your clone> format-patch --start-number 62 -o backends/nemo-rl/apertus2/patches/nemo-rl/ <base>..<head>
   ```
   Check that the series is complete: in a fresh worktree at the pin (the PR merges of `apertus2/pin`), apply every
   patch in order with `git am -3`, and compare the tree with the clone you tested on (`git diff --stat`). Patches
   numbered 0015-0017 and 0035-0039 on other branches are not on main: renumber them after 0061.
6. **Replace personal paths:** datasets under `${oc.env:RL_DATA_DIR}/...`, files in your home under
   `${oc.env:RL_USER_HOME}/...`, the judge model under `${oc.env:SCRATCH}/models/...`.
7. **Check before the first run:** `DRY_RUN=1 bash run.sh experts/<expert>/<task>.yaml` prints nodes, context, the
   overrides `run.sh` adds, the config copies and the preflight report. To confirm nothing changed, merge the old and
   the new config with `backends/nemo-rl/config.sh` (`config_resolve`) and compare them leaf by leaf
   (`yq r <file> --printMode pv '**' | sort`).
8. **A new expert** starts from its template in `experts/<expert>/` (or copies one): fill in the datasets marked
   TODO and delete `template: true`, which `run.sh` refuses to submit.

## 4. Branches in this checkout that are not migrated yet (2026-11-20)

Commits not in `restructure`, per local branch of `~/projects/rl-infra`:

| Branch | Commits | Last commit | Note |
|---|---|---|---|
| `apertus-v11-pre-restructure` | 7 | 2026-11-20 | migrated: `apertus-v11` now starts from `restructure` (`recipes/apertus-v11*.yaml`, `backends/posttraining`, `docs/apertus-v11.md`) |
| `megachonk` | 5 | 2026-11-18 | |
| `pr6` | 3 | 2026-11-15 | |
| `chonk-debug` | 2 | 2026-10-28 | |
| `enigmata`, `rg-review` | 2 each | 2026-11-14 | merged into `stem5` in another form; check before migrating |
| `process-rubric` | 2 | 2026-11-11 | |
| `openqa-judge` | 1 | 2026-11-14 | as `enigmata` |
| `harbor-sft`, `lean`, `multi-judge` | 1 each | 2026-11-03 to 06 | |

The NeMo-RL clones of these branches may also hold commits that are not patches yet (step 5).

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
| `envs/apertus2-system.json`, `-high.json`, `-sciworld-act.json`, `-sciworld-act-high.json` | `shared/frames/` |
| `envs/stage_judge_model.sh`, `envs/strip_few_shot.py` | `shared/scripts/` |
| `envs/score-text.yaml` | `shared/debug/` |
| the judge blocks of `envs/stem-mix-run4.yaml` / `run5.yaml` (`env.llm_judge`, `env.judge_pool`) | `shared/judges/llm-judge.yaml` (taken in with `defaults:`) |
| `experiments/long-context/setup-kcp.sh`, `experiments/long-context/kcp/` | `backends/nemo-rl/apertus2/kcp/setup.sh`, `kcp/` |
| `experiments/long-context/recipes/sftv02-stem-*.yaml` | `size:` tiers (`experts/main.yaml`) |
| `recipes/apertus2-*.yaml` (all 34) | `recipes/archive/`; the STEM recipe written out as `recipes/chonk-sftv02.yaml` |
| `experiments/grpo-chonk/*.yaml` (27) | `recipes/archive/chain/` |
| `experiments/grpo-chonk/stage_model.sh` | `backends/nemo-rl/apertus2/stage_model.sh` |
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
| `experiments/grpo-math/`, `experiments/vlm-geo3k/` | `backends/nemo-rl/examples/` |
| `experiments/grpo-chonk/README.md` | `recipes/archive/chain/README.md` |
| `experiments/runs.md`, `megachonk.md`, `long-context/README.md` | `docs/runs.md`, `docs/megachonk.md`, `docs/long-context.md` |

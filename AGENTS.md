# Working in rl-infra (for agents and people)

rl-infra trains one RL expert per domain (STEM, chat, code, tool use, ...) on the Apertus 2 models with NeMo-RL.
`README.md` explains how a run is put together; `docs/repository-map.md` says what every file does.

## Where to work

Work in the folder of the expert you were given, `experts/<expert>/`:

- `dev/`: variants and experiments (`defaults: ../base.yaml` plus what they change). New work starts here.
- `base.yaml`: the expert's official run. Change it only when the expert's owner wants that change as the default.
- `archive/`: runs as they ran. Do not edit them; copy one into `dev/` to run it again with changes.
- `scripts/`: the expert's data builders, environment setup and tests.

Ask the maintainers before changing:

- `recipes/`: the training setup (model, optimizer, algorithm, batch, node layout), shared by every expert. A
  setting only your expert needs goes into its task file as an override.
- `experts/main.yaml`: the settings every expert shares.
- `run.sh`, `backends/nemo-rl/` (except new patches, below) and `sites/`.
- Files in `shared/` and in other experts' folders. A new piece only your expert uses belongs in
  `experts/<expert>/`; it goes into `shared/` once a second expert takes it in.

## Code changes go in as patches

If your expert needs a change in NeMo-RL, Gym or another repository a run uses (a new environment, a verifier, a
fix), add a patch; `backends/nemo-rl/patches/README.md` explains how.

- Number it after the highest existing number, with a subject no other patch has.
- Never edit, renumber or remove an existing patch; correct one with a new patch.
- Leave other experts' runs unchanged: add a new environment, or put new behaviour behind a config key that is off
  by default.
- Export it with `git format-patch` from your own clone, so it keeps its author.
- Check the series: in a fresh worktree at the pin, every patch applies in order with `git am -3`.

## Running

- `DRY_RUN=1 bash run.sh <task>` before every launch: nodes, context, overrides and the preflight report, no job.
- `DEBUG=1 bash run.sh <task>` to test: 3 steps, no validation, no checkpoints, 2 hours.
- `bash run.sh <task>` is a full run and logs to W&B.
- Datasets go under `${oc.env:RL_DATA_DIR}/...` and your own files under `${oc.env:RL_USER_HOME}/...`. Committed
  files hold no personal absolute paths and no secrets.
- Login nodes are shared: data preparation, downloads and model loading run in jobs.
- Do not cancel or change jobs, clones or files you did not start.

## Commits

- Work on a branch and open a pull request to `main`; do not push to `main`.
- A run worth keeping goes into `docs/runs.md` (command, job, result); a finished run's task file into `archive/`.
- A branch from before the restructure: `docs/MIGRATION.md`, section 7.

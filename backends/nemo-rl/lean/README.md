# Lean 4 theorem proving

GRPO on Lean 4 proofs with NeMo Gym `math_formal_lean` as it ships: the model gets a theorem statement and
answers with a ```` ```lean4 ```` block; the reward is 1 if header + statement + proof compiles against
Mathlib without `sorry`, else 0. The task file is `envs/lean-sft.yaml`.

Proofs compile in the NeMo-Skills sandbox image (`igitman/nemo-skills-sandbox:0.7.1-arm64`: Lean 4.12.0,
Mathlib v4.12.0, its own Flask server under uWSGI, `POST /execute`), on one intTest node of its own. About
5 s per proof (each compile loads Mathlib), 128 at once.

## Steps

    bash backends/nemo-rl/lean/import_image.sh      # once: the image, as $SCRATCH/lean/images/*.sqsh
    bash backends/nemo-rl/lean/setup-data.sh        # once: $SCRATCH/lean/data/{train,validation}.jsonl
    bash backends/nemo-rl/lean/start.sh             # the sandbox job; its node lands in $SCRATCH/lean/sandbox-endpoint
    bash run.sh envs/lean-sft.yaml \
        env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.sandbox_host=$(cat $SCRATCH/lean/sandbox-endpoint)

Stop the sandbox with `scancel` after the run.

## Data

- Training: Nemotron-Math-Proofs-v1 (CC BY-SA 4.0, pinned revision), selected by Gym's
  `prepare_nemotron_math_proofs.py` with its defaults: theorems with 1-3 of 4 first tries proved, not used
  in Nano v3 SFT. 9,605 theorems.
- Validation: miniF2F test as Gym ships it, 244 theorems.
- `fix_rows.py` fixes the Gym script's rows. It keeps the statement ending in `:= by sorry` and the header
  without a newline, and the server compiles header + statement + proof as one string, so every proof
  compiled as `...open Subgrouptheorem ... := by sorry<proof>`: 0 of 64 verified proofs from the data
  passed, 33 of 64 after the fix (`test_sandbox.py`). The rest fail with real tactic errors.
- `download.py` fetches in 32 MB ranges over 64 connections: one stream from a compute node got under
  1 MB/s from Hugging Face and Docker Hub.

## Isolation

Lean runs code at compile time (`#eval`, `IO`). In an enroot container this site mounts `/scratch` and
`/home01` whatever pyxis is told, so the sandbox runs in bubblewrap on the image unpacked to node-local disk
(`sandbox_job.sh`): compiled code sees neither (checked with `System.FilePath.pathExists`). It keeps the
network: the 0.7.1 image predates NeMo-Skills' network blocking.

## With a Kimina Lean Server instead

`backends/nemo-rl/lean-kimina` checks the proofs in a Kimina Lean Server (Lean v4.34.0, warm REPLs) instead of
this sandbox, with the task file `envs/lean-kimina-sft.yaml`; it also makes this task's rows on Alps.

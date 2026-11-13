# Lean 4 proofs checked by a Kimina Lean Server

The Lean task of `backends/nemo-rl/lean` (Gym `math_formal_lean`) checks every proof in the NeMo-Skills sandbox:
one cold `lake env lean` per proof, which imports all of Mathlib again (Lean and Mathlib v4.12). This directory
puts a Kimina Lean Server in its place: [swiss-ai/kimina-lean-server](https://github.com/swiss-ai/kimina-lean-server),
branch `clariden-performance`, a fork of project-numina's with settings for many-core nodes (see its
`docs/MANY_CORE_NODES.md`). Its REPLs import the header once and check each proof in a warm REPL, on Lean
v4.34.0 with current Mathlib.

Gym patch `0011` (`../apertus2/patches/gym`) adds `verifier: kimina` to `math_formal_lean`. It keeps the
sandbox's result format, so scoring and correction prompts are unchanged. The task file is
`envs/lean-kimina-sft.yaml`.

**Status (2026-11-13):** `setup.sh`, the data job and the server job (`start.sh`) have run on Clariden, with the
REPLs in bubblewrap; see [Tested](#tested). The task file has not yet run with a training job.

## Before the first run: Lean and Mathlib

The server needs a Lean toolchain and a Lake project with Mathlib compiled for it, both at the pin's `REPL_TAG`
(v4.34.0). They are only read, and nothing here builds them. Set both in `site.env` (gitignored; see
`site.env.example`):

    : "${LEAN_TOOLCHAIN:=/path/to/elan/toolchains/leanprover--lean4---v4.34.0}"
    : "${LEAN_PROJECT:=/path/to/a/lake/project}"     # with .lake/packages/mathlib/.lake/build

Any Lake project that depends on Mathlib at v4.34.0 will do, for example one made with
`lake +leanprover/lean4:v4.34.0 new <name> math`, then `lake exe cache get` in it (about 6 GB). Put the project on
scratch: the server copies what it needs into node memory at start, so the file system it lives on does not
matter for speed.

## Steps (Alps)

    bash backends/nemo-rl/lean-kimina/setup.sh                       # once, login node, ~2 min
    PARTITION=debug bash backends/nemo-rl/lean-kimina/setup-data.sh  # once, one node, < 30 min; prints the job
    bash backends/nemo-rl/lean-kimina/start.sh                       # the server job; prints the job
    bash run.sh envs/lean-kimina-sft.yaml \
        env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.kimina_url=$(cat $SCRATCH/lean-kimina/endpoint) \
        env.nemo_gym.math_formal_lean.resources_servers.math_formal_lean.kimina_api_key=$(cat $SCRATCH/lean-kimina/endpoint.key)

Stop the server job with `scancel` after the run.

- **`setup.sh`** clones the fork at `pin`, makes a Python 3.11 venv with `requirements.txt` (Clariden has no
  3.12 outside containers), and builds the Lean REPL at `REPL_TAG` with `LEAN_TOOLCHAIN`, all in
  `$SCRATCH/lean-kimina/install`. It is safe to re-run.
- **`setup-data.sh`** makes the rows of `backends/nemo-rl/lean/setup-data.sh` with the same scripts and the same
  pinned data (Nemotron-Math-Proofs-v1, 29.5 GB; Gym's `prepare_nemotron_math_proofs.py` at the Gym pin; miniF2F
  test).
  - The differences: it runs on the host with the venv rather than in a pyxis container (which Alps does not
    have), and needs no NeMo-RL clone (it fetches Gym's two files at its pin).
  - In the same job a Kimina server keeps the rows whose statement elaborates on v4.34:
    `$SCRATCH/lean/data/{train,validation}.v434.jsonl`.
  - The other rows go to `*.dropped.jsonl` with Lean's first error. The rows were made for Mathlib v4.12, and a
    statement that no longer elaborates would score 0 on every rollout.
  - The job also checks 400 of the dataset's verified proofs (Lean 4.12) the way Gym builds them.
  - A rerun skips every step whose output exists.
- **`start.sh`** runs `server_job.sh` on a node of its own. It does the following:
  - stages Lean into `/dev/shm`;
  - starts the server with the settings below;
  - imports the training rows' import header into every REPL at start-up;
  - checks four controls;
  - writes `$SCRATCH/lean-kimina/endpoint` and `endpoint.key` (an API key made per job, mode 600).

Lean runs code while it elaborates (`#eval`, `IO`), so proofs from a model must not see scratch or home. Both jobs
start their REPLs in bubblewrap (`repl-bwrap.sh`: no network, read-only view of the Lean copy) when it works on
the node. If the server's controls then fail, the data job falls back to unconfined REPLs, while `server_job.sh`
stops.

## Settings and why

Measured on one GH200 node (Clariden debug jobs 3923541 and 3923726, 2026-11-12). The workload was 2,048
proofs, one RL step: a 108B model's own ProofNet attempts and Goedel-Prover proofs. Details are in the fork's
`docs/MANY_CORE_NODES.md`.

| Setting | Value | Measured |
|---|---|---|
| `MONITOR_REPLS` | `false` | Upstream's two psutil samplers per REPL keep the server's single event loop at ~98% CPU from 64 REPLs on: ≈1 proof/s. Without them: 26–31 proofs/s |
| `LOG_SNIPPETS` | `false` | Per-proof logging on the same event loop |
| `DIRECT_EXEC`, `LEAN_PATH` in `/dev/shm` | | 32 REPLs import Mathlib in 13 s from `/dev/shm`, 149 s from Lustre; staging takes about 32 s |
| `MAX_REPLS` | 128 | About 2 GB per warm REPL, within a job's 450 GB; 256 REPLs were OOM-killed; 64 REPLs already check a step's proofs as fast |
| `HEADER_TIMEOUT`, `PREWARM` | 900 s, on | A cold import never counts against a proof's timeout; a REPL killed by a timeout is replaced in the background |
| `CHECK_HEADER` | on | A REPL whose import failed answers `{"env": 0}` silently and then fails every proof as if wrong |

For comparison, one cold `lean` per proof (the sandbox's way) checked 3.2–4.2 proofs/s on the same node, at
26–47 s per proof. 27% of the checks took over 30 s, the task's compilation timeout.

## Tested

On Clariden (debug partition, one GH200 node per job):
- **`setup.sh`** made a fresh install in 2 min 20 s (2026-11-12). On 2026-11-13 it moved that install, made from
  the earlier patch, to the fork at the pin in 8 s; a rerun took 2 s and changed nothing. The patched tree and the
  fork's commit had the same code (only `README.md`, `.env.template` and `docs/` differ).
- **Data job 3925556** (2026-11-12, before the fixes below):
  - The 29.5 GB download took 18 s; reduction read 1,376,663 rows.
  - Gym's selection gave **9,605 theorems**, as `backends/nemo-rl/lean` documents.
  - On Lean v4.34.0, **9,128 of the 9,605 training statements and 212 of the 244 miniF2F statements elaborate**.
    The 32 miniF2F failures use the `∑ x in s` notation that newer Mathlib replaced by `∑ x ∈ s`.
  - Of 400 Nemotron proofs verified on Lean 4.12 and built into code as Gym does, **115 pass on v4.34**; 272 fail
    with a Lean error, 11 time out (60 s) and 2 are `sorry`.
  - With the REPLs in bubblewrap every import failed silently: `--dev /dev` came after binding the Lean copy under
    `/dev/shm` and hid it. `CHECK_HEADER` caught it and the job went on unconfined, but only after waiting its
    full 900 s (see job 3931849).
- **Test job 3931849** (2026-11-13, the scripts of this commit):
  - One REPL through the fixed `repl-bwrap.sh` imported Mathlib and checked a proof. From inside, reading a file
    on scratch failed (no such file), writing into the Lean copy failed (read-only file system), and `curl` found
    no network.
  - A server with 128 REPLs in bubblewrap was up after 68 s and passed its controls. Validating all 9,849
    statements again kept exactly the rows of the unconfined data job, and the 400 sample proofs gave the same
    counts.
  - The 900 s wait: a server whose start-up fails does not exit. When every initial REPL fails, replacements are
    still being started; at exit Python cancels them and waits forever for those half-started processes. A
    ten-line asyncio script reproduces this on Clariden's Python 3.11.13 without any Kimina code. `kimina_wait`
    only checked that the process was alive; it now also stops when uvicorn logs `Application startup failed`:
    after 20 s instead of 900 s in this job.
- **Server job 3931932** (2026-11-13, `start.sh`): up 43 s after staging, with 128 REPLs in bubblewrap. It wrote
  `endpoint`; from a login node `/health` answered, a check without the key got HTTP 401, and the controls passed.
  Its `endpoint.key` was readable by two groups: the folder's default ACL overrides the umask. `server_job.sh` now
  makes the file private before the key goes in (checked on a login node: no one but the owner can read it).

Locally, 2026-11-12 and 2026-11-13:
- All 11 Gym patches apply in order to Gym at the pin. With 0011 applied, Gym's own `math_formal_lean` tests and
  the new ones pass against the real `nemo_gym` (54 tests).
- The fork's test suite passes on Python 3.11 with `requirements.txt` (50 tests). Upstream's `test_mathlib` only
  fails because the Mac imports Mathlib slowly.
- `setup.sh` was checked against a local copy of the fork's branch: it moves an install made from the earlier
  patch to the fork's commit, does nothing on a rerun, clones a fresh install, and stops with a clear error when
  `LEAN_TOOLCHAIN`/`LEAN_PROJECT` are unset.
- `setup-data-job.sh` ran with real Lean v4.34.0 + Mathlib on synthetic Nemotron rows, everything but the
  download:
  - Gym's selection kept the hard, medium and easy theorems and dropped the very-easy and nano_v3 ones.
  - The server passed its controls, and validation dropped the statement that does not elaborate, with Lean's
    error.
  - The sample check passed the correct proof and failed the wrong one, and the job exited 0.

Not yet run: a training run with `envs/lean-kimina-sft.yaml`, and a server job with the key-file change.

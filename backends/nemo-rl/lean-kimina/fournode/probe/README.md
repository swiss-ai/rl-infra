# Four-node Chonk GRPO: the probe job

One 4-node job checks whether the 4-node colocated recipe
(`recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-4n-colocated.yaml`, stage 2) works. It uses the 200-step run's
Lean setup (`run-200.sh`'s overrides: 128 × 8 rollouts, 16k tokens, ε 1e-14, a recovery checkpoint every step).
It answers four questions:

- does the recipe fit in GPU and host memory;
- does a resume restore ScratchAdamW's moments;
- how long each phase of a step takes;
- how long ScratchAdamW's step takes.

```bash
bash probe.sh setup          # private source copies with patches 0007 + 0034, the probe's EDF; no job
DRY_RUN=1 bash probe.sh      # preflight and the sbatch command; submits nothing
bash probe.sh                # submits (normal partition, 4 nodes, 4 h)
python3 report.py $SCRATCH/lean/fournode-probe/nemo-rl <job>
```

## What the job does

`probe_driver.sh` runs the NeMo-RL driver twice inside the allocation:

1. **Phase A** runs steps 1-3 from the SFT export. Every rank's ScratchAdamW starts with zero moments and prints
   `ScratchAdamW moments: reset`. The run saves `step_3`, with hard links to the moments in
   `step_3/policy/weights/optimizer-scratch/`.
2. **Phase B** starts a new driver, which brings new Ray workers, a new Megatron optimizer and a new ScratchAdamW.
   It resumes from `step_3` as a requeued job does and prints `restored` on every rank. Then it runs steps 4-5.

ScratchAdamW refuses a step whose moments are not at the parameters' step count, so steps 4-5 running at all shows
that the resume paired the moments correctly.

`probe_watch.sh` runs on every node, in place of `numa_watch.sh`, and records:

- every GPU's memory (`nvidia-smi`, every 2 s; trainer and vLLM together) in `gpumem-<node>.csv`;
- host memory every 5 s (MemAvailable and each CPU NUMA node's free memory) in `hostmem-<node>.tsv`.

It then hands over to `numa_watch.sh`, which records HBM spills, hangs and Xid errors as usual.

ScratchAdamW writes a line per step and rank to `results/<RUN>/scratch-adam-live/rank<r>/hdo<n>.stats.jsonl`.

`report.py` gives each part a verdict: GO, CHECK (works but tight, or data missing) or NO-GO:

| Part | GO when |
|---|---|
| GPU | no out-of-memory line, and every GPU had at least 2 GiB left at its peak |
| host | MemAvailable never dropped below 16 GiB, and no host memory spilled onto HBM |
| resume | 16 × `reset` in phase A, 16 × `restored` in phase B, steps 1-5 logged, phase B exit 0 |
| checkpoint | `step_3` holds a moment manifest for each rank |
| scratch | every rank reported every step |

It also prints the per-step phase times (`timing/train/*`).

## Nothing shared changes

The 200-step chain uses the shared NeMo-RL, Megatron and Bridge clones in `~/projects` and the EDF
`~/.edf/nemo-rl-apertus2.toml`. A launch through `run.sh` would patch those clones, so the probe never does that.
Instead:

- `probe.sh setup` copies the clones into `$PROBE/src` (rsync; the source side is only read).
- `apertus2/setup.sh` then applies Megatron 0007 and NeMo-RL 0034 to the copies.
- It writes the probe's own EDF to `$PROBE/edf/` (`EDF_DIR`), and jobs name that EDF by its absolute path.
- Setup checks that the shared clones and the shared EDF are unchanged.
- It checks that the probe's EDF is the shared one with only the clone paths changed.

Launches set `SKIP_SETUP=1`, so `run.sh` leaves out its setup. Its model staging would re-link the files of the
shared `$SCRATCH/models/<export>` folder that the running chain reads.

Setup only reads the staged model, the apertus-common checkout and the Python overlay, and checks they are complete.

The job writes under `$PROBE` and in `$SCRATCH/lean-kimina/runs/<RUN>` (the in-job Kimina server).

The probe has its own Triton cache in `$PROBE/triton`, seeded from the shared one.

## Settings

All are environment variables; `probe.sh` documents them:

- `PROBE`
- `RUN`
- `STEPS` (5) and `FIRST_STEPS` (3)
- `VLLM_UTIL` (the recipe's 0.75 when unset)
- `PARTITION` (normal)
- `JOB_TIME` (04:00:00)
- `EXTRA_OVERRIDES`

Driver retries are off, so a failure ends the probe instead of being hidden by a retry.

## Tests

`probe_test.py` (laptop, 11 tests) covers:

- the two phases with a stand-in driver: step counts, the new deadline for phase B, a skipped phase A,
  failures;
- the settings `probe.sh` hands the job;
- the EDF path;
- the report on made-up jobs (GO, CHECK, NO-GO).

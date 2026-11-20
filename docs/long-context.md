# Long context: GRPO at 32K, 64K and 128K tokens

GRPO on the SFT v0.2 checkpoint of Apertus 2 "Chonk" with a context limit of 32K, 64K and 128K tokens instead of 16K,
on KISTI HANGANG (November 2026). Two task families: the STEM mix (`bash run.sh stem`, which sets the
context with its `context:` key) and competitive coding on Gym's `code_gen` (`launch-code-ctx.sh`). Results and run
commands are in `../runs.md`; commands there that name `launch-stem-ctx.sh` refer to the commit before the
restructure, where that launcher still exists.

## What is needed

| Piece | Where |
|---|---|
| NeMo-RL patches 0040-0046 | `backends/nemo-rl/apertus2/patches/nemo-rl/`, applied by `setup.sh` like the others |
| Ray on Slingshot by default | `sites/hangang.env` (`RAY_NODE_NAME_SUFFIX=-hsn0`, read by patch 0041) |
| KCP context parallelism, for more than 4 context-parallel ranks | `backends/nemo-rl/apertus2/kcp/`: run.sh sets it up and uses it for a context above 32K (above CP 4: the recipe's `context_parallel`); `launch-code-ctx.sh` with `KCP=1` |
| STEM task files | `experts/stem/` (`base.yaml`, `dev/`, `archive/`, `scripts/`) |
| Code sandbox node, for coding | `backends/nemo-rl/sandbox/start.sh`, then `SANDBOX=<endpoint file>` |
| W&B key | `WANDB_KEY_FILE` (default `~/.wandb_api_key`); a full run (`DEBUG=0`) refuses to start without it |

The NeMo-RL patches are all opt-in or behaviour-preserving:

| Patch | What it does | Switch |
|---|---|---|
| 0040 | How long a collective of the trainer waits for its peers | `NRL_NCCL_TIMEOUT_MINUTES` (trainer env var) |
| 0041 | `ray.sub` resolves `<node><suffix>` instead of the node name | `RAY_NODE_NAME_SUFFIX` |
| 0042 | The trainer keeps off the node of the Ray head, where the driver runs | `env.train_off_head` |
| 0043 | The per-step rollout file without the per-token arrays (text, reward, length, sample mask) | `env.train_dump_text_only` |
| 0044 | Grouping columns in the rollout file: prompt group, task name, prompt index, prompt ordinal, generation weight version | always, when the file is written |
| 0045 | vLLM FP8: KV-cache scales without `Attention.calculate_kv_scales` (removed in vLLM 0.28) | always |
| 0046 | Load an HF export into vLLM through the refit path (FP8 quantization included) | used by `eval/run_val.py` |
| 0047 | A step that saves and validates saves first | `checkpointing.save_before_validation` (the launchers set it, `CKPT_FIRST=0` turns it off) |
| 0048 | Gym servers and other services advertise the address of `<node><RAY_NODE_NAME_SUFFIX>` | `RAY_NODE_NAME_SUFFIX` |

## Trainer layout per context length

The trainer stays pipeline parallel 2 × expert parallel 16. Context parallelism (CP) splits each packed micro-batch
over CP ranks; the measured limit is about 8,192 tokens per GPU.

| Context | CP | Scheme | Trainer nodes | PyTorch cap | Measured |
|---|---|---|---|---|---|
| 16K | 1 | | 16 | 0.84 | the earlier runs |
| 32K | 1 | | 16 | 0.84 | out of memory in the first backward (job 1284638) |
| 32K | 2 | pinned (heads split) | 16 | 0.84 | works, peak 60.1 GB live per GPU (job 1285312) |
| 32K | 4 | pinned | 16 | 0.74 | works, 8,192 tokens per GPU; used by the full STEM runs |
| 64K | 4 | pinned | 16 | 0.84 / 0.78 | NCCL out of memory at 0.84 (job 1285313), PyTorch out of memory at 0.78 (job 1285402) |
| 64K | 8 | KCP | 8 or 16 | 0.74 | works (jobs 1285787, 1289457); used by the coding run |
| 128K | 16 | KCP | 8 or 16 | 0.74 | works (job 1292002); used by the coding run |

The pinned Megatron splits the 28 heads of the KDA layers over the CP ranks, so it allows CP 1, 2 or 4 only. KCP
(Megatron-LM-MoE PR 112, ported onto the pinned commit in `backends/nemo-rl/apertus2/kcp/`) keeps the tokens split
instead and allows any CP that divides the ranks of a pipeline stage. It needs flash-linear-attention 0.5.2, newer than
the image's; `kcp/setup.sh` installs it and `kcp/use.sh` (run.sh) or the coding launcher put it first on `PYTHONPATH`.
The recipe's `context_parallel` block (8,192 tokens per GPU, memory fraction 0.74) reproduces the rows of this table that run.sh applies, apart from 16K, which now gets CP 2. A cap of 0.74 (70.3 GB) leaves room for NCCL and Triton
outside PyTorch, which reached 16 to 24 GB per GPU at long context.

vLLM refuses a limit above the export's `max_position_embeddings` (16,384) unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`;
the model has no position table, so the value is only that check. The launchers set it.

## Things that matter at long context

- **Ray on Slingshot.** A compute node's name resolves to the 1 Gbit management Ethernet, and Ray used it: the driver
  sends each training batch to the trainer nodes as Ray objects at 117 MB/s. The trainer nodes then started a step 5
  to 10 minutes apart at 64K and more than 30 minutes apart at 128K. On `hsn0` the same transfer runs at 4,178 MB/s
  (`tools/ray_nettest.py`, jobs 1297629 and 1297631) and the nodes start within seconds. A 64K coding step went from
  1,240-1,590 s to about 950 s. Check a run's driver log: worker lines show `ip=10.169.` on Slingshot, `ip=10.161.` on
  the management network.
- **Collective timeout.** A trainer collective waits 10 minutes for its peers by default (PyTorch and Megatron Bridge).
  The two 128K coding runs on the slow network died in their first training step with
  `Watchdog caught collective operation timeout` (jobs 1293158, 1293163). The launchers set
  `NRL_NCCL_TIMEOUT_MINUTES=60` from 64K on (`NCCL_MIN`); on Slingshot it is a safety margin.
- **Host memory of the driver's node at 128K.** The driver holds the whole batch, padded to the longest answer. When
  Ray placed four trainer ranks (with their CPU-offloaded optimizer state) on the driver's node, the node ran out of
  host memory in the first training step (job 1299189, `srun: error: ... Out Of Memory`). `OFFHEAD=1` (patch 0042)
  keeps the trainer off that node, so it hosts vLLM workers; in job 1299941 the driver used 92 GB and 147 GB stayed
  free. Placement varies from job to job without it.
- **Gym servers under load, and validation before saving.** In coding runs NeMo-RL sends every rollout of a step
  to Gym's agent server at once, and the servers of one node cannot accept the connections fast enough: the 64K run
  (job 1298910) counted 11.5 million connections dropped from full accept queues on the Gym node and 10.4 million
  retried requests in Gym. Validations, which add 3,220 rollouts to those of the next steps, slowed from 18 to 79
  minutes. In the validation of step 40 Gym's model server received truncated vLLM responses, the agent server
  answered HTTP 500, and the driver exited; the run resumed from the step-20 checkpoint, because the step-40
  checkpoint is written after the validation. Patch 0047 saves first. The Gym servers also listened on the management
  network (patch 0048 moves them to Slingshot), though in that run the traffic stayed within one node.
- **Hang watchdog.** A 64K answer can take close to an hour and a 128K answer about two, so from 64K on the launchers
  raise the watchdog limits to 4 h (6 h during a validation).
- **Unit tests for coding.** The OpenSandbox pods at CSCS stopped finishing commands on 17 Nov 2026; the 64K run then
  stalled for an hour with every check failing and pods being recreated about three times per second (job 1293154).
  The coding runs since use one sandbox node of their own (`SANDBOX=<endpoint file>`, 192 checks at once); LiveCodeBench
  validation of the start model is the same on both (61.0% against 60.7%). `tools/pod_probe.py` tests the pod service.
- **Generation or training as the limit.** With 16 trainer nodes the STEM mix at 32K needed 32 vLLM nodes to keep the
  trainer busy; generation then ran the full 4 steps of staleness ahead and the trainer waited 17 s per step (median).
  Coding at 64K and 128K is training-bound with 16 vLLM nodes.

## Files

| File | Purpose |
|---|---|
| `launch-code-ctx.sh <tokens>` | Coding on `code_gen` with pods or a sandbox node, CP, KCP, `OFFHEAD`, `HEAD` (a dedicated driver node), node counts 12, 16, 31, 32 |
| `recipes/` | Node layouts of the coding runs: 8+4, 8+8, 16+15 (with `HEAD=1`), 16+16 (STEM runs take a `size:` tier) |
| `convert_rl_ckpt.sh <run> <step> <name>` | NeMo-RL checkpoint to an HF export (one node, about 10 minutes; uses the shared converter of the SFT exports) |
| `eval/` | Evaluation with vLLM only, see below |
| `tools/rollout_index.py <exp dir>` | Per-step index of the rollout files: step, row, prompt group, dataset, row in the dataset file, reward, length (for files written before patch 0044) |
| `tools/rollout_stats.py`, `run_report_data.py`, `gen_stats.py` | Rollout statistics, per-step timings and rewards, vLLM load per step, from a run's files and driver log |
| `tools/stagger.py <driver log> <ranks>` | How far apart the trainer ranks start each step |
| `tools/netcheck.sbatch`, `ray_nettest.py`, `submit_ray_nettest.sh` | Network interfaces and Ray transfer speed between two nodes |
| `tools/pod_probe.*`, `pod_capacity.*` | OpenSandbox pod health and capacity |

## Evaluation with vLLM only

`eval/run_val.py` evaluates one checkpoint without a trainer. It starts vLLM with dummy weights, loads the HF export
through the refit path (patch 0046), and calls the run's own `grpo.validate()` with the run's saved config: the same
message format, environments, judge, sampling and limits as the periodic validation of the run, multi-turn
ScienceWorld and the open-QA judge included. `eval/submit_val.sh <export> <step> <name> [vLLM nodes]` submits it
(8 vLLM nodes by default, plus the judge node when the open-QA set is included).

`eval/prepare_benchmarks.py` builds the sets on the data mover: AIME 2025 and 2026 (30 × 32 answers each), MATH-500
(× 4), MMLU-Pro (12,032 × 1), and held-out sets of four training environments: Reasoning Gym (its held-out file, × 8),
ScienceWorld (dev variations, × 4), Enigmata (pool puzzles not in the training file, × 2) and Open QA (× 2). The Open-QA
gold pool lies inside the training file, so its held-out questions are training-file rows that run 5 at 32K never
drew; later runs must leave them out of training to keep them held out.

Status (18 Nov 2026): the first test of the job is queued; the scripts have not run yet.

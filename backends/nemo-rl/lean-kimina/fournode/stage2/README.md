# Four-node Chonk GRPO, stage 2: the training side on 4 nodes

Stage 2 makes the trainer fit on 4 GH200 nodes (16 GPUs) next to colocated vLLM. `budget.py` computes the memory per GPU
and per node for each option. It is calibrated against job 4008441: 35 GB static per GPU, 162 GB of Adam state and 54 GB
of gradient copies per trainer node.

```bash
python3 budget.py --nodes 4 --pp 2 4 --detail
```

## Memory at 4 nodes (budget.py)

```bash
python3 budget.py --nodes 4 --pp 2 4 --option bf16-scratch-mv scratch-mv --c2c --detail
python3 budget.py --nodes 4 --pp 4 --option bf16-scratch-mv --stage-layers 12 10 10 10 --detail
```

| Layout | Experts in training | GPU static / peak | Host per node | C2C per GPU per step |
|---|---|---|---|---|
| PP2 EP8, m, v on scratch | bf16 on the GPU | 54 / 83-97 GB, does not fit | 439 GB | 18 GB |
| PP2 EP8, m, v on scratch | fp8, offloaded | 44 / 73-87 GB | 491 GB, does not fit | 4.2 TB host to GPU |
| PP4 EP4, m, v on scratch | bf16 on the GPU | 46 / 75-89 GB | 429 GB | 15 GB |
| PP4 EP4, m, v on scratch | fp8, offloaded | 39 / 68-82 GB | 480 GB, does not fit | 8.4 TB host to GPU |
| **PP4 EP4 12/10/10/10, m, v on scratch (recipe)** | bf16 on the GPU | 47 / 76-90 GB | 435 GB | 15 GB |
| PP4 EP4 11/11/11/9, m, v on scratch | bf16 on the GPU | 51 / 80-94 GB | 462 GB, does not fit | 15 GB |

The rows without a layer split assume every stage holds the average 9.75 MoE layers; with `--stage-layers` they
describe the stage with the most MoE layers (Chonk's first 3 layers are dense). A PP4 stage is a whole node, so the
heaviest stage sets both the GPU and the host figure.

Scratch I/O is 433 GB per node per step in every row. Corrected 2026-11-25: the coarse-grained expert offload (the
only mode NeMo-RL supports) keeps pinned fp8 copies of every expert in both orientations next to the bf16 ones
(+51.5 GB per node), and its per-layer GPU buffers grow with PP. The host figure assumes the measured 115 GB of
overhead per trainer node; job 4008441's trainers used 274 GB in all, so the real overhead may be about half, which
would make the fp8 rows fit with little margin. GPU peaks add the activations and communication buffers measured on 8
nodes with EP16. Colocated generation frees the gradient buffers (Megatron's `offload_to_cpu` resizes them to 0).

## The pieces

**Expert offloading** exists in Megatron-LM-MoE (0536034), and Chonk's SFT ran with it (`sft-chonk.sh`:
`MOE_OFFLOAD=1`). It is not used: the coarse-grained mode NeMo-RL needs computes the experts in fp8 and keeps extra
pinned fp8 copies, which overflow host memory (table above). The recipe keeps today's bf16 training instead.

**ScratchAdamW** (`scratch_adam.py`) is a CPU AdamW whose m and v live in files on scratch.
- It streams them through a few aligned buffers, with readers, update and writers overlapped. A reader takes a buffer
  before it claims a chunk; the other order deadlocked (found by `test_pipeline_never_deadlocks`).
- Results are bit-identical to torch's AdamW: fused (Megatron's CPU optimizer uses `fused=True`) through
  `torch._fused_adamw_` per chunk, otherwise the single-tensor operations. Chunks start on 256-element boundaries.
- Every step writes a new generation into fresh files, then replaces `manifest.json` atomically. A failed step leaves
  the previous state intact. Since files are never rewritten, a checkpoint holds hard links to them: no copy, no time.
- The step counts in the optimizer state (Megatron's checkpoint has them) must equal the steps the files were written
  at; `step()` refuses otherwise, so moments are never paired with another step's weights.
- Files are checked lazily: Megatron first builds a throwaway optimizer over unsharded params in the same folder.
- `skip_next_step()` turns Megatron's `dummy_step()` (a zero-grad step before a load) into a no-op for the files.

**Megatron patch 0007** (`apertus2/patches/megatron/0007-cpu-optimizer-state-dir-scratch-adam.patch`):
- `OptimizerConfig.cpu_optimizer_state_dir`: each HybridDeviceOptimizer gets `bind(<dir>, "rank<r>/hdo<n>")`.
- `HybridDeviceOptimizer.dummy_step()` calls `skip_next_step()`.
- `DistributedOptimizer.state_dict()` reads the step count from ScratchAdamW as from torch's AdamW.
- It carries a copy of `scratch_adam.py`. After changing `scratch_adam.py`, run `python3 patch_tool.py`
  (`stage2_test.py` fails while they differ).

**NeMo-RL patch 0034** (`apertus2/patches/nemo-rl/0034-...`):
- `save_checkpoint` links each rank's current moments into `<weights>/optimizer-scratch/rank<r>/hdo<n>/`.
- After the load in `setup_model_and_optimizer`, `after_checkpoint_load` restores the moments saved with the loaded
  checkpoint when the load filled the optimizer state, and starts from zero otherwise (fresh run, weights-only resume).
- Both do nothing without patch 0007 or without `cpu_optimizer_state_dir`.

**Recipe** `recipes/apertus2-chonk-sftv02-18315-vllm-adamw-code-4n-colocated.yaml`: 4 nodes, colocated vLLM
(synchronous GRPO; NeMo-RL allows colocated vLLM only without async), PP4 EP4 with 12/10/10/10 layers per stage, bf16
experts on the GPU, ScratchAdamW with its live folder next to the run's `step_<N>` folders. vLLM's memory share (0.75) is a guess until stage 1's sleep/wake data.

**Resuming**: moments are tied to the parallel layout (4 nodes, PP4 12/10/10/10, EP4). The 200-step run's checkpoints are
`dp_reshardable` with 8 trainer nodes, so their optimizer state cannot move to 4 nodes at all, with or without
ScratchAdamW. A 4-node run starts from the SFT export, or from such a checkpoint's weights with `load_optim: false`
(fresh moments).

## Tests

- `stage2_test.py` (laptop, 23 tests): bit-identity with torch AdamW (single-tensor and fused), restart, layout and
  step mismatches, failed steps, checkpoint links that survive later steps, restore, reset, the dummy step, Megatron's
  nesting and param groups, deadlock stress, the patch copy, the budget's calibration and per-stage split.
- `gpu/hdo_gpu_test.py` via `gpu/hdo_job.sbatch` (one GH200 node, preemptable, 30 min): the patched
  HybridDeviceOptimizer with bf16 CUDA params, ScratchAdamW against torch's fused AdamW, a resume as Megatron does it
  (dummy step, `load_state_dict`, restore from the links), a fresh start, O_DIRECT and hard links on Lustre, and one
  step's speed over 2 B params. The job patches a copy of the pinned `hybrid_optimizer.py`; the shared clone is not
  touched. Code goes to `$SCRATCH/lean/fournode-stage2/code/` (with `megatron-patches/`), results to `out/<job>-hdo/`.

```bash
sbatch --output=$SCRATCH/lean/fournode-stage2/logs/%j-hdo.out $SCRATCH/lean/fournode-stage2/code/gpu/hdo_job.sbatch
```

## Still to do

1. Run `gpu/hdo_job.sbatch` (one node).
2. Apply the patches to the shared clones (`apertus2/setup.sh`: Megatron 0007 by `git apply`, NeMo-RL 0034 by
   `git am`); both are no-ops for runs without `cpu_optimizer_state_dir`.
3. `preflight.sh` on the 4-node recipe (login node), then a **probe job**: 4 nodes, 3-5 GRPO steps of 16k Lean
   rollouts, a checkpoint and a resume. It records GPU memory per rank, host memory per NUMA node, phase times and
   ScratchAdamW's step time.
4. If the probe shows the heaviest node short of host memory: the fp32 masters on scratch too (another 108 GB per node
   off the host, 650 GB of scratch I/O per node per step). The GPU side has about 5 GB to spare at the measured peak;
   the probe's per-rank numbers decide whether that holds.

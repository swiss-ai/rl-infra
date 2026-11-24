# Four-node Chonk GRPO, stage 2: the training side on 4 nodes

Stage 2 makes the trainer fit on 4 GH200 nodes (16 GPUs) next to colocated vLLM. `budget.py` computes the memory per GPU
and per node for each option. It is calibrated against job 4008441: 35 GB static per GPU, 162 GB of Adam state and 54 GB
of gradient copies per trainer node.

```bash
python3 budget.py --nodes 4 --pp 2 4 --detail
```

## Memory at 4 nodes (budget.py, PP2 EP8)

| Option | GPU static / peak | Host per node | Scratch I/O per step |
|---|---|---|---|
| today's settings | 54 / 83-97 GB | 648 GB, does not fit | 0 |
| + expert offloading | 42 / 71-85 GB | 648 GB, does not fit | 0 |
| + m, v on scratch (**target**) | 42 / 71-85 GB | 439 GB | 433 GB per node |
| + expert grads on host used in place, master on scratch | 17 / 46-60 GB | 335 GB | 650 GB per node |

The host figure assumes the measured 115 GB of overhead per trainer node. Job 4008441's trainers used 274 GB in all, so
the real overhead is probably lower. Colocated generation frees the gradient buffers (Megatron's `offload_to_cpu`
resizes them to 0) and copies only the dense weights to the host. Stage 1's `stream-adam` measures whether the scratch
I/O fits inside generation.

## The pieces

**Expert offloading** already exists in Megatron-LM-MoE (0536034), and Chonk's SFT ran with it (`sft-chonk.sh`:
`MOE_OFFLOAD=1`). NeMo-RL supports it through `policy.megatron_cfg.model_overrides`.

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
(synchronous GRPO; NeMo-RL allows colocated vLLM only without async), PP2 EP8, expert offloading, ScratchAdamW with its
live folder next to the run's `step_<N>` folders. vLLM's memory share (0.75) is a guess until stage 1's sleep/wake data.

**Resuming**: moments are tied to the parallel layout (4 nodes, PP2, EP8). The 200-step run's checkpoints are
`dp_reshardable` with 8 trainer nodes, so their optimizer state cannot move to 4 nodes at all, with or without
ScratchAdamW. A 4-node run starts from the SFT export, or from such a checkpoint's weights with `load_optim: false`
(fresh moments).

## Tests

- `stage2_test.py` (laptop, 22 tests): bit-identity with torch AdamW (single-tensor and fused), restart, layout and
  step mismatches, failed steps, checkpoint links that survive later steps, restore, reset, the dummy step, Megatron's
  nesting and param groups, deadlock stress, the patch copy, the budget's calibration.
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
4. Later (scratch-all): `moe_offload_main_grad` with the hybrid optimizer using the host-resident expert gradients in
   place, then the fp32 masters on scratch too.

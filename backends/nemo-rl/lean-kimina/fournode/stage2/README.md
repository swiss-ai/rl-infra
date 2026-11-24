# Four-node Chonk GRPO, stage 2: the training side on 4 nodes

Stage 2 makes the trainer fit on 4 GH200 nodes (16 GPUs) next to colocated vLLM. `budget.py` computes the memory per GPU
and per node for each option. It is calibrated against job 4008441: 35 GB static per GPU, 162 GB of Adam state and 54 GB
of gradient copies per trainer node.

```bash
python3 budget.py --nodes 4 --pp 2 4 --detail
```

## What already exists in our stack

Megatron-LM-MoE (`~/projects/megatron-lm-moe-apertus2`, 0536034) offloads experts, and Chonk's SFT ran with it
(`sft-chonk.sh`: `MOE_OFFLOAD=1`, PP4 EP16). NeMo-RL (2f7739e) supports it in its policy worker: fp8 copies, reference
swaps and exports. Recipes set it through `policy.megatron_cfg.model_overrides`.

```yaml
policy:
  megatron_cfg:
    pipeline_model_parallel_size: 2
    expert_model_parallel_size: 8          # 16 GPUs / PP2
    model_overrides:
      moe_use_offloading_experts: true     # bf16 experts in pinned host memory, fp8 copies streamed per layer
      moe_use_inplace_fp8_param: true
      moe_offloading_mode: coarse-grained  # NeMo-RL needs coarse-grained or extra fp8 storage
      moe_offloading_num_chunks: 8
      moe_offloading_num_stages: 2
      moe_offload_activations: [moe_input, moe_fc1_output]
      # moe_offload_main_grad: true        # later (scratch-all): expert fp32 grads on the host
    optimizer:
      optimizer_cpu_offload: true
      optimizer_offload_fraction: 1.0
      overlap_cpu_optimizer_d2h_h2d: false # one CPU optimizer over all parameters (needed by ScratchAdamW)
      cpu_optimizer_state_dir: ${oc.env:SCRATCH}/lean/fournode-opt/${RUN}   # new, see "Megatron patch" below
  generation:
    colocated:
      enabled: true
cluster:
  num_nodes: 4
```

## Memory at 4 nodes (budget.py, PP2 EP8)

| Option | GPU static / peak | Host per node | Scratch I/O per step |
|---|---|---|---|
| today's settings | 54 / 83-97 GB | 648 GB, does not fit | 0 |
| + expert offloading | 42 / 71-85 GB | 648 GB, does not fit | 0 |
| + m, v on scratch (**target**) | 42 / 71-85 GB | 439 GB | 433 GB per node |
| + expert grads on host used in place, master on scratch | 17 / 46-60 GB | 335 GB | 650 GB per node |

The host figure assumes the measured 115 GB of overhead per trainer node. Job 4008441's trainers used 274 GB in all, so
the real overhead is probably lower. Stage 1's `stream-adam` measures whether the scratch I/O fits inside generation.

## Built here

- `scratch_adam.py`: **ScratchAdamW**, a CPU AdamW whose m and v live in files on scratch.
  - It streams them through a few aligned buffers, with readers, update and writers overlapped.
  - Its results are bit-identical to `torch.optim.AdamW(foreach=False)` (tested, including learning-rate changes and
    parameters without a gradient).
  - State is double-buffered on scratch with an atomic `manifest.json`. A failed step leaves the previous step's state
    intact, and a restart from the manifest continues bit-identically.
  - Chunk sizes are rounded to 256 elements. Torch's CPU kernels vectorize from a tensor's start, and a chunk starting
    mid-block would differ by one ulp (found by the tests).
- `stage2_test.py`: 9 tests. They include the budget's calibration against job 4008441.

## Still to build

1. **Megatron patch 0007.** Add `OptimizerConfig.cpu_optimizer_state_dir`. When it is set, `get_megatron_optimizer` passes
   `functools.partial(ScratchAdamW, state_dir=<dir>/rank<r>/hdo<i>)` as `cpu_optimizer_cls`. There is one folder per
   hybrid optimizer instance, since the distributed optimizer builds several. NeMo-RL passes `megatron_cfg.optimizer` keys
   on to `OptimizerConfig`.
2. **Checkpoints.** The hybrid optimizer's state then holds only `step` for these parameters.
   - Save: write the Megatron checkpoint without m and v, and hard-link the last finished generation's files plus
     `manifest.json` into `step_N/optimizer-scratch/rank<r>/`. Lustre hard links are instant, so the every-step recovery
     checkpoint stays cheap.
   - Load: copy or link them back into the state folder before the optimizer is built.
   - Converting today's step_50 checkpoint (m and v inside the distributed checkpoint) into the files needs a one-off
     export, through `ScratchAdamW.moments` in reverse.
3. **Recipe** `apertus2-chonk-...-4n-colocated.yaml` with the block above, plus a 4-node dry run (preflight only).
4. **Probe job.** 4 nodes, 3-5 GRPO steps of 16k Lean rollouts. It records GPU memory per rank, host memory per NUMA node,
   step and phase times, and checks ScratchAdamW's I/O speed against stage 1.
5. **Later (scratch-all).** Turn on `moe_offload_main_grad` and let the hybrid optimizer use the host-resident expert
   gradients in place, instead of its own pinned copies. Then put the master weights on scratch too.

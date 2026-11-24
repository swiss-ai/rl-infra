#!/usr/bin/env python3
"""Memory budget of Chonk GRPO training per GPU and per node, for a node count and a set of options (stdlib only).

  python3 budget.py                 # the table for 4, 6 and 8 nodes and every option
  python3 budget.py --nodes 4 --pp 2 --option scratch-mv

Parameter counts come from Chonk's config (42 layers, 39 MoE layers x 256 experts of 3 x 1792 x 1920, 108.27 B in all);
measured overheads come from job 4008441 (8 trainer nodes, PP2, EP16): GPU activations/logits 17-24 GB and NCCL/DeepEP
12-19 GB per GPU; host overhead per trainer node (Ray, CUDA contexts, pinned pools, page cache) about 115 GB.

Options (cumulative left to right is not implied; each is one configuration):
  today          no expert offload; Adam state and fp32 gradient copies on the host (HybridDeviceOptimizer, fraction 1)
  expert-offload Megatron-LM-MoE's offloading experts (bf16 experts in pinned host memory, fp8 compute copies streamed
                 per layer, coarse-grained, no extra fp8 storage); expert fp32 main grads stay on the GPU
  scratch-mv     expert-offload + Adam's m and v on scratch, streamed during the step (master stays on the host)
  scratch-all    expert-offload + expert main grads on the host (moe_offload_main_grad), used in place by the CPU
                 optimizer (no second copy), and master, m and v on scratch
"""
import argparse

GB = 1e9
EXPERT_PARAMS = 39 * 256 * 3 * 1792 * 1920          # 103.05 B
TOTAL_PARAMS = 108.27e9
DENSE_PARAMS = TOTAL_PARAMS - EXPERT_PARAMS          # 5.22 B: embeddings, attention/KDA, shared experts, dense MLPs
GPU_ACT = (17, 24)        # GB per GPU, activations and logits (measured, 16k sequences)
GPU_COMM = (12, 19)       # GB per GPU, NCCL and DeepEP buffers (measured)
GPU_TOTAL = 95.0          # usable per GH200 GPU
HOST_TOTAL = 450.0        # Slurm's default job memory per node (the Grace LPDDR)
HOST_OVERHEAD = 115.0     # measured per trainer node
VLLM_FP8 = 1.0            # bytes per parameter of vLLM's fp8 weights kept on the host during level-1 sleep
OPTIONS = ("today", "expert-offload", "scratch-mv", "scratch-all")


def layout(nodes, pp):
    gpus = 4 * nodes
    ep = gpus // pp                                  # experts spread over each pipeline stage's GPUs
    if 256 % ep or gpus % pp:
        return None
    return {"gpus": gpus, "ep": ep, "dp": gpus // pp,
            "expert_per_gpu": EXPERT_PARAMS / gpus,  # each GPU holds 1/gpus of all experts, whatever PP is
            "dense_per_gpu": DENSE_PARAMS / pp,      # dense layers split over PP stages, replicated over DP
            "dense_opt_per_gpu": DENSE_PARAMS / pp / (gpus // pp)}  # the distributed optimizer shards over DP


def budget(nodes, pp, option):
    lay = layout(nodes, pp)
    if lay is None:
        return None
    e, d, dopt = lay["expert_per_gpu"], lay["dense_per_gpu"], lay["dense_opt_per_gpu"]
    opt_params = e + dopt                            # parameters whose Adam state this GPU's rank owns
    gpu = {"dense bf16 weights": 2 * d, "dense fp32 grads": 4 * d}
    host = {"overhead (measured)": HOST_OVERHEAD,
            "vLLM fp8 weights while asleep": VLLM_FP8 * TOTAL_PARAMS / nodes / GB}
    per_node = 4 / GB                                # x4 GPUs per node, in GB
    if option == "today":
        gpu["expert bf16 weights"] = 2 * e
        gpu["expert fp32 grads"] = 4 * e
        host["Adam master+m+v (fp32)"] = 12 * opt_params * per_node
        host["fp32 grad copies (optimizer)"] = 4 * opt_params * per_node
        host["Megatron weights while vLLM runs (bf16)"] = 2 * (e + d) * per_node
    else:
        gpu["expert fp8 staging (2 stages)"] = 2 * 2 * e / 39       # one MoE layer's experts, both orientations
        host["expert bf16 weights (pinned)"] = 2 * e * per_node
        host["Megatron dense weights while vLLM runs"] = 2 * d * per_node
        if option in ("expert-offload", "scratch-mv"):
            gpu["expert fp32 grads"] = 4 * e
            host["fp32 grad copies (optimizer)"] = 4 * opt_params * per_node
            host["Adam master (fp32)"] = 4 * opt_params * per_node
            if option == "expert-offload":
                host["Adam m+v (fp32)"] = 8 * opt_params * per_node
            else:
                host["m+v chunk buffers"] = 8.0
        else:  # scratch-all
            gpu["expert fp32 grad staging"] = 4 * e / 39
            host["expert fp32 grads (= optimizer grads)"] = 4 * e * per_node
            host["dense fp32 grad copies"] = 4 * dopt * per_node
            host["master+m+v chunk buffers"] = 12.0
    gpu = {k: v / GB if v > 1e3 else v for k, v in gpu.items()}
    static = sum(gpu.values())
    out = {"nodes": nodes, "pp": pp, "option": option, **lay, "gpu": gpu, "host": host,
           "gpu_static": static, "gpu_peak": (static + GPU_ACT[0] + GPU_COMM[0], static + GPU_ACT[1] + GPU_COMM[1]),
           "host_total": sum(host.values())}
    out["gpu_fits"] = out["gpu_peak"][1] <= GPU_TOTAL
    out["host_fits"] = out["host_total"] <= HOST_TOTAL
    # bytes to and from scratch per node per optimizer step
    if option == "scratch-mv":
        out["scratch_io_per_step"] = 2 * 8 * opt_params * per_node
    elif option == "scratch-all":
        out["scratch_io_per_step"] = 2 * 12 * opt_params * per_node
    else:
        out["scratch_io_per_step"] = 0.0
    return out


def table(rows):
    lines = ["| nodes | PP | EP | option | GPU static | GPU peak | fits | host per node | fits | scratch I/O per step |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['nodes']} | {r['pp']} | {r['ep']} | {r['option']} | {r['gpu_static']:.0f} GB | "
                     f"{r['gpu_peak'][0]:.0f}-{r['gpu_peak'][1]:.0f} GB | {'yes' if r['gpu_fits'] else 'NO'} | "
                     f"{r['host_total']:.0f} GB | {'yes' if r['host_fits'] else 'NO'} | "
                     f"{r['scratch_io_per_step']:.0f} GB |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, nargs="*", default=[4, 6, 8])
    ap.add_argument("--pp", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--option", nargs="*", default=list(OPTIONS))
    ap.add_argument("--detail", action="store_true", help="every item of every row")
    args = ap.parse_args(argv)
    rows = [b for n in args.nodes for p in args.pp for o in args.option if (b := budget(n, p, o))]
    print(table(rows))
    if args.detail:
        for r in rows:
            print(f"\n{r['nodes']} nodes PP{r['pp']} {r['option']}")
            for side in ("gpu", "host"):
                for k, v in r[side].items():
                    print(f"  {side:4} {k:42} {v:7.1f} GB")
    return rows


if __name__ == "__main__":
    main()

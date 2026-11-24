#!/usr/bin/env python3
"""Stage-1 report: the measured primitives and what they mean for Chonk GRPO on four nodes (stdlib only).

  python3 report.py OUT_ROOT [--md report.md]

OUT_ROOT is $STAGE1/out: <job>-host/host.json from host_job.sbatch and <job>-vllm-kv<KV>/r<N>/ from vllm_job.sbatch.
The four-node quantities come from the design (artifact "Chonk GRPO on Four Nodes") and job 4008441:
"""
import argparse
import json
import sys
from pathlib import Path

PARAMS = 108.27e9            # Chonk v2
NODES = 4
GPUS = 4 * NODES
PER_NODE = PARAMS / NODES    # parameters whose optimizer state each node holds
STATE_BYTES = 12             # fp32 master, m, v
GRAD_BF16_PER_GPU = 18e9     # bytes of bf16 gradients one GPU flushes per micro-batch (design)
GEN_TOKENS = 10.1e6          # generated tokens per GRPO step (J1 4008441)
BUSY = 0.90                  # share of generation time a GPU is busy with continuous admission
TODAY_TOK_S = 1780           # tokens/s per busy vLLM GPU in J1 4008441
GEN_PHASE = (310, 390)       # seconds of generation per step at 4 nodes (design), the window Adam must fit in


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def fmt(x, nd=1):
    return "–" if x is None else f"{x:,.{nd}f}"


def host_section(host, lines):
    lines.append("## Host side (one node)")
    f = host.get("facts") or {}
    lines.append(f"torch {f.get('torch')}, {f.get('cpus')} CPUs, NUMA nodes {len(f.get('numa', []))}, GPUs {len(f.get('gpus', []))}")
    c2c = host.get("c2c") or []
    if c2c:
        d2h = min(r["all_d2h_GBps"] for r in c2c)
        h2d = min(r["all_h2d_GBps"] for r in c2c)
        lines.append(f"- **CPU-GPU link**, all 4 GPUs at once (slowest GPU): host->GPU {fmt(h2d)} GB/s, GPU->host {fmt(d2h)} GB/s; "
                     f"both ways at once {fmt(min(r['all_both_GBps_each_way'] for r in c2c))} GB/s each way; "
                     f"one GPU alone {fmt(max(r['solo_d2h_GBps'] for r in c2c))} GB/s GPU->host; pageable host->GPU "
                     f"{fmt(min(r['solo_pageable_h2d_GBps'] for r in c2c))} GB/s.")
        lines.append(f"  - 18 GB of bf16 gradients per GPU leave in {fmt(GRAD_BF16_PER_GPU / d2h / 1e9 * 1e3, 0)} ms (design: 40 ms).")
    gs = host.get("gradsum") or []
    if gs:
        rate = min(r["flush_sum_Gparams_per_s"] for r in gs)
        lines.append(f"- **Gradient flush + fp32 sum on Grace**, per GPU/socket, all at once (slowest): {fmt(rate, 2)} Gparams/s "
                     f"(CPU add alone {fmt(min(r['cpu_add_Gparams_per_s'] for r in gs), 2)}).")
        lines.append(f"  - one micro-batch's 9 G bf16 gradients per GPU: {fmt(GRAD_BF16_PER_GPU / 2 / rate / 1e9, 2)} s (design: 0.18 s).")
    ad = host.get("adam") or []
    if ad:
        for v in ("fused", "foreach"):
            key = f"{v}_Gparams_per_s"
            if all(key in r for r in ad):
                node = sum(r[key] for r in ad)
                lines.append(f"- **AdamW in host memory, {v}**: {fmt(node, 2)} Gparams/s per node "
                             f"-> {fmt(PER_NODE / node / 1e9, 0)} s for a node's {PER_NODE / 1e9:.1f} G parameters.")
            elif any(f"{v}_error" in r for r in ad):
                lines.append(f"- AdamW {v}: failed ({next(r[f'{v}_error'] for r in ad if f'{v}_error' in r)[:120]})")
    for key in sorted(k for k in host if k.startswith("io-")):
        io = host[key]
        lines.append(f"- **Scratch, {key[3:]} layout** ({'O_DIRECT' if io.get('direct') else 'page cache'}):")
        for r in io.get("runs", []):
            mixed = r.get("mixed") or {}
            lines.append(f"  - {r['streams']} files: write {fmt(r['write']['GBps'], 2)} GB/s (1-s p10 {fmt(r['write']['p10_1s_GBps'], 2)}), "
                         f"read {fmt(r['read']['GBps'], 2)} GB/s (p10 {fmt(r['read']['p10_1s_GBps'], 2)}), "
                         f"read+write at once {fmt(mixed.get('read'), 2)} + {fmt(mixed.get('write'), 2)} GB/s")
    sa = host.get("stream-adam") or []
    if sa:
        steps = [r["steps"][-1] for r in sa if r.get("steps")]
        if steps:
            node_rate = sum(s["Gparams_per_s"] for s in steps)
            slowest = max(s["seconds"] for s in steps)
            params = sum(r["params"] for r in sa)
            io = sum(s["io_GBps_read_plus_write"] for s in steps)
            need = PER_NODE / node_rate / 1e9
            lines.append(f"- **Streamed Adam step from scratch** ({params / 1e9:.1f} G parameters, {len(sa)} sockets at once): "
                         f"{fmt(slowest)} s, {fmt(node_rate, 3)} Gparams/s per node, scratch read+write {fmt(io, 2)} GB/s; "
                         f"compute {fmt(sum(s['compute_seconds'] for s in steps) / len(steps))} s per socket.")
            verdict = "fits" if need <= GEN_PHASE[0] else "fits only the longer phases" if need <= GEN_PHASE[1] else "does NOT fit"
            lines.append(f"  - a node's {PER_NODE / 1e9:.1f} G parameters: **{fmt(need, 0)} s** against a generation phase of "
                         f"{GEN_PHASE[0]}-{GEN_PHASE[1]} s: {verdict}.")
            errs = [e for s in steps for e in s.get("errors", [])]
            if errs:
                lines.append(f"  - I/O errors: {errs[:3]}")


def vllm_section(runs, lines):
    lines.append("## vLLM on one node (continuous admission)")
    if not runs:
        lines.append("no results yet")
        return
    lines.append(f"Today (J1 4008441): {TODAY_TOK_S} tokens/s per busy GPU. At 4 nodes a step generates {GEN_TOKENS / 1e6:.1f} M tokens on "
                 f"{GPUS} GPUs, {BUSY:.0%} busy.")
    lines.append("")
    lines.append("| KV | weights | per GPU in flight | tok/s per GPU | vs today | running | KV use (peak) | preemptions | cut at 16k | 4-node generation |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, run in runs:
        server = run.get("server", "")
        weights = server.split("weights=")[1].split()[0] if "weights=" in server else "?"
        for lv in run.get("levels") or []:
            t = lv["gen_tok_per_s_per_gpu"]
            gen_s = GEN_TOKENS / (GPUS * t * BUSY) if t else None
            kv = lv.get("mean_kv_usage")
            pk = lv.get("peak_kv_usage")
            lines.append(f"| {run['kv']} | {weights} | {lv['level_per_gpu']} | {fmt(t, 0)} | {fmt(t / TODAY_TOK_S if t else None, 2)}x | "
                         f"{fmt(lv.get('mean_running'), 0)} | {fmt(kv * 100 if kv is not None else None, 0)} % ({fmt(pk * 100 if pk is not None else None, 0)} %) | "
                         f"{fmt(lv.get('preemptions'), 0)} | {fmt((lv.get('cut_share') or 0) * 100, 0)} % | {fmt(gen_s, 0)} s |")
    for name, run in runs:
        sw = run.get("sleepwake")
        if sw:
            parts = [f"{s['step']} {s['status']} {s['seconds']:.1f} s" for s in sw.get("steps", [])]
            lines.append(f"- sleep/wake ({run['kv']}): " + "; ".join(parts))
            mem = [(s["step"], s.get("gpu_mem_mib")) for s in sw.get("steps", []) if s["step"] in ("sleep1", "sleep2", "wake2")]
            lines.append(f"  - GPU memory (MiB) after: {mem}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_root")
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    lines = ["# Four-node design, stage 1 measurements", ""]
    hosts = sorted(root.glob("*-host/host.json"))
    if hosts:
        for h in hosts:
            lines.append(f"_{h.parent.name}_")
            host_section(load(h) or {}, lines)
            lines.append("")
    else:
        lines.append("## Host side: no results yet\n")
    runs = []
    for d in sorted(root.glob("*-vllm-kv*/r*")):
        levels = load(d / "levels.json")
        if levels is None:
            continue
        server = (d / "server.txt").read_text() if (d / "server.txt").exists() else ""
        runs.append((str(d.relative_to(root)), {"kv": d.parent.name.split("-kv")[-1], "levels": levels,
                                                "server": server, "sleepwake": load(d / "sleepwake.json")}))
    vllm_section(runs, lines)
    text = "\n".join(lines) + "\n"
    if args.md:
        Path(args.md).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()

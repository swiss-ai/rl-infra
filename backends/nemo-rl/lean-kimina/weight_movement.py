#!/usr/bin/env python3
"""How far each kind of weight moved between two HF checkpoints of Chonk (Apertus 2 MoE), read-only.
    python3 weight_movement.py <start HF dir> <trained HF dir> [--layers 3,12,21,30,41] [--dense 1] [--every 4]
Python 3.6 + numpy (Clariden's login nodes), no torch. Reads the sampled layers (every --every-th routed expert)
plus embeddings and lm_head, a few GB per checkpoint and a few minutes; big tensors in chunks, so about 1 GB of
memory. RUNBOOK.md, "Learning speed", has the result for H1 step 40.

Both checkpoints are bf16; training keeps fp32 master weights that start exactly at start's bf16 values. The
export rounds them to the nearest bf16 value, so a weight changes in the export only if it moved MORE THAN HALF A
bf16 STEP (ulp) from its start; smaller moves vanish. The bf16 step grows with |w|, so the fair comparison is
between weights of the same size: the script reports, per group and per |w| bin, the share of weights that changed,
i.e. moved more than the bin's threshold (half a step, printed in the header). The frozen routers (mlp.gate.*)
must show exactly 0. For reference, Adam moves a weight with a gradient well above its epsilon by up to the
learning rate per update, so 40 updates at 1e-5 move a weight at most 4e-4.
"""
import argparse
import collections
import json
import struct

import numpy as np

CHUNK = 1 << 24  # elements


class Ckpt:
    def __init__(self, d):
        self.d = d
        self.map = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
        self.hdr = {}

    def info(self, name):
        f = self.d + "/" + self.map[name]
        if f not in self.hdr:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdr[f] = (8 + n, json.loads(fh.read(n).decode()))
        base, h = self.hdr[f]
        t = h[name]
        return f, base + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0], t["dtype"], t["shape"]

    def chunks(self, name):
        """float32 chunks of the flattened tensor"""
        f, off, size, dt, _ = self.info(name)
        width = {"BF16": 2, "F32": 4}[dt]
        with open(f, "rb") as fh:
            fh.seek(off)
            left = size
            while left:
                raw = fh.read(min(left, CHUNK * width))
                left -= len(raw)
                if dt == "BF16":
                    yield (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    yield np.frombuffer(raw, dtype=np.float32)


def group(n):
    if ".experts." in n:
        return "routed experts"
    if ".shared_experts." in n:
        return "shared expert"
    if ".mlp.gate." in n:
        return "router (frozen)"
    if ".mlp.latent_" in n:
        return "latent proj (MoE in/out)"
    if ".self_attn." in n:
        return "attention"
    if "layernorm" in n or n.endswith("norm.weight"):
        return "norms"
    if ".mlp." in n:
        return "dense MLP"
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("trained")
    ap.add_argument("--layers", default="3,12,21,30,41", help="MoE layers to sample")
    ap.add_argument("--dense", default="1", help="dense layers to sample")
    ap.add_argument("--every", type=int, default=4, help="sample every n-th routed expert")
    args = ap.parse_args()
    a, b = Ckpt(args.start), Ckpt(args.trained)
    moe = [int(x) for x in args.layers.split(",") if x]
    dense = [int(x) for x in args.dense.split(",") if x]

    names = []
    for L in dense + moe:
        p = "model.layers.%d." % L
        names += [n for n in sorted(a.map) if n.startswith(p) and ".experts." not in n]
        if L in moe:
            experts = sorted({int(n.split(".")[5]) for n in a.map if n.startswith(p + "mlp.experts.")})
            for e in experts[::args.every]:
                names += [p + "mlp.experts.%d.%s.weight" % (e, w) for w in ("gate_proj", "up_proj", "down_proj")]
    names += ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]

    # Per |w| bin: frexp exponent e, |w| in [2^(e-1), 2^e), bf16 step 2^(e-8), change threshold 2^(e-9).
    E0, NB = -12, 16  # bins e = -12 .. 3
    G = collections.defaultdict(lambda: np.zeros((2, NB)))  # [weights, changed] per bin
    rms = collections.defaultdict(lambda: np.zeros(2))  # n, sum w^2
    per_layer = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((2, NB))))
    per_expert = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros((2, NB))))
    for n in names:
        ia, ib = a.info(n), b.info(n)
        assert ia[2:] == ib[2:], "%s: size/dtype/shape differ" % n
        acc = np.zeros((2, NB))
        for x, y in zip(a.chunks(n), b.chunks(n)):
            e = np.clip(np.frexp(np.abs(x))[1] - E0, 0, NB - 1)
            acc[0] += np.bincount(e, minlength=NB)
            acc[1] += np.bincount(e[x != y], minlength=NB)
            rms[group(n)] += [x.size, float((x.astype(np.float64) ** 2).sum())]
        g = group(n)
        G[g] += acc
        if n.startswith("model.layers."):
            L = int(n.split(".")[2])
            per_layer[g][L] += acc
            if g == "routed experts":
                per_expert[L][int(n.split(".")[5])] += acc

    def pct(v, b, min_n=1000):
        return "%5.1f%%" % (100 * v[1][b] / v[0][b]) if v[0][b] >= min_n else "     -"

    bins = [b for b in range(NB) if sum(v[0][b] for v in G.values()) > 0.02 * sum(v[0].sum() for v in G.values())]
    print("share of weights that moved more than half a bf16 step, by |w| (threshold in brackets):")
    print("%-26s %11s %8s %8s " % ("group", "weights", "rms w", "all") + " ".join(
        "%17s" % ("|w|<%.3g [%.2g]" % (2.0 ** (b + E0), 2.0 ** (b + E0 - 9))) for b in bins))
    for g in sorted(G, key=lambda g: -G[g][1].sum() / G[g][0].sum()):
        v = G[g]
        print("%-26s %11d %8.3g %7.1f%% " % (g, v[0].sum(), (rms[g][1] / rms[g][0]) ** .5, 100 * v[1].sum() / v[0].sum())
              + " ".join("%17s" % pct(v, b) for b in bins))
    ref = max(bins, key=lambda b: G["routed experts"][0][b]) if "routed experts" in G else bins[0]
    print("\nby layer, weights with |w| in [%.3g, %.3g) (changed if moved > %.2g):"
          % (2.0 ** (ref + E0 - 1), 2.0 ** (ref + E0), 2.0 ** (ref + E0 - 9)))
    for g in sorted(per_layer):
        print("  %-26s " % g + "  ".join("L%d %s" % (L, pct(v, ref, 100).strip()) for L, v in sorted(per_layer[g].items())))
    print("\nrouted experts, same |w| bin, share changed per sampled expert: quantiles across experts")
    for L in sorted(per_expert):
        m = np.array([v[1][ref] / v[0][ref] for v in per_expert[L].values() if v[0][ref] > 0])
        q = 100 * np.percentile(m, [0, 10, 50, 90, 100])
        print("  L%-2d min %.2f%%  p10 %.2f%%  median %.2f%%  p90 %.2f%%  max %.2f%%" % ((L,) + tuple(q)))

if __name__ == "__main__":
    main()

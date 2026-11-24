#!/usr/bin/env python3
"""Test of optimizer_stats.py on a small synthetic checkpoint in the on-disk format of this task's Megatron
checkpoints (torch_dist: torch.save'd chunks + DCP .metadata; dp_reshardable optimizer buckets).
    python3 optimizer_stats_test.py [--keep DIR]
Checks: every trainable chunk is found (fused, split across DP chunks, padded, fp32), the frozen router is reported
unmatched, and the moments and movement statistics equal the values planted in the data."""
import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (BytesStorageMetadata, ChunkStorageMetadata, Metadata,
                                                   MetadataIndex, TensorProperties, TensorStorageMetadata)

HERE = os.path.dirname(os.path.abspath(__file__))
LR, WD, B1, B2 = 1e-5, 0.01, 0.9, 0.999
STEP_A, STEP_B = 30, 40
N_UPD = STEP_B - STEP_A
WDF = (1 - LR * WD) ** N_UPD
SV = 1e-9  # planted sqrt(v_hat)
DELTA = 3e-5  # planted movement of the moving tensors (|delta| / (n lr) = 0.3)


class Writer(object):
    def __init__(self, d):
        self.d, self.f, self.sd, self.st = d, open(os.path.join(d, "__0_0.distcp"), "wb"), {}, {}

    def blob(self, obj):
        b = io.BytesIO()
        torch.save(obj, b)
        off = self.f.tell()
        self.f.write(b.getvalue())
        return _StorageInfo(relative_path="__0_0.distcp", offset=off, length=len(b.getvalue()))

    def tensor(self, fqn, t, chunks):
        """chunks: list of (offsets, sizes) covering t."""
        cm = []
        for off, size in chunks:
            sl = t[tuple(slice(o, o + s) for o, s in zip(off, size))].clone()
            self.st[MetadataIndex(fqn, torch.Size(off), len(cm))] = self.blob(sl)
            cm.append(ChunkStorageMetadata(offsets=torch.Size(off), sizes=torch.Size(size)))
        self.sd[fqn] = TensorStorageMetadata(properties=TensorProperties(dtype=t.dtype), size=t.size(), chunks=cm)

    def obj(self, fqn, o):
        self.st[MetadataIndex(fqn)] = self.blob(o)
        self.sd[fqn] = BytesStorageMetadata()

    def close(self):
        self.f.close()
        with open(os.path.join(self.d, ".metadata"), "wb") as f:
            import pickle
            pickle.dump(Metadata(state_dict_metadata=self.sd, storage_data=self.st), f)


def rne(x):
    return x.to(torch.bfloat16)


def make_run(root):
    g = torch.Generator().manual_seed(0)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02)  # noqa: E731
    # model tensors at checkpoint A, exactly bf16-representable (as the fp32 masters start on bf16 values)
    m = {
        "decoder.layers.0.self_attention.in_proj.weight.query": rnd(64, 32),
        "decoder.layers.0.self_attention.in_proj.weight.key": rnd(64, 32),
        "decoder.layers.0.self_attention.in_proj.weight.value": rnd(64, 32),
        "decoder.layers.0.mlp.linear_fc1.weight": rnd(96, 32),
        "decoder.layers.0.pre_mlp_layernorm.weight": 1 + rnd(40),
        "decoder.layers.0.mlp.router.weight": rnd(8, 32),
        "decoder.layers.0.mlp.experts.experts.linear_fc1.weight": rnd(4, 48, 32),
        "decoder.layers.0.mlp.experts.experts.linear_fc2.weight": rnd(4, 32, 24),
    }
    m = dict((k, rne(v).float()) for k, v in m.items())
    fp32 = {"decoder.layers.0.self_attention.A_log": torch.randn(28, generator=g),
            "decoder.layers.0.self_attention.dt_bias": torch.randn(100, generator=g)}
    moving = {"decoder.layers.0.mlp.linear_fc1.weight", "decoder.layers.0.mlp.experts.experts.linear_fc2.weight#0",
              "decoder.layers.0.self_attention.dt_bias"}

    # bucket layouts (reverse registration order, starts aligned to 64, fused projections back to back)
    E1 = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
    E2 = "decoder.layers.0.mlp.experts.experts.linear_fc2.weight"
    dense = [["decoder.layers.0.pre_mlp_layernorm.weight"], ["decoder.layers.0.mlp.linear_fc1.weight"],
             ["decoder.layers.0.self_attention.in_proj.weight.%s" % s for s in ("query", "key", "value")]]
    experts = {0: [[(E2, 1)], [(E2, 0)], [(E1, 1)], [(E1, 0)]], 1: [[(E2, 3)], [(E2, 2)], [(E1, 3)], [(E1, 2)]]}

    def flat(params, values):
        """params: list of fused groups of (key, expert or None); returns flat tensor with garbage padding and the
        (start, length) of every fused group."""
        parts, pos, spans = [], 0, []
        for grp in params:
            pad = (-pos) % 64
            if pad:
                parts.append(torch.full((pad,), 7.0))  # garbage, as torch.empty
                pos += pad
            start = pos
            for key, e in grp:
                t = values(key, e).reshape(-1)
                parts.append(t)
                pos += t.numel()
            spans.append((start, pos - start))
        return torch.cat(parts), spans

    def theta(stage, key, e):
        base = m[key] if e is None else m[key][e]
        if stage == "a":
            return base.clone()
        is_1d = base.dim() == 1
        b = base if is_1d else base * WDF
        name = key if e is None else "%s#%d" % (key, e)
        return b + DELTA if name in moving else b.clone()

    def theta32(stage, key):
        base = fp32[key]
        return base + DELTA if (stage == "b" and key in moving) else base.clone()

    bc1, bc2 = 1 - B1 ** STEP_B, 1 - B2 ** STEP_B
    for stage, step in (("a", STEP_A), ("b", STEP_B)):
        sd = os.path.join(root, "step_%d" % step)
        d = os.path.join(sd, "policy", "weights", "iter_0000000")
        os.makedirs(d)
        w = Writer(d)
        # model weights: bf16 of the fp32 masters; fc1 halves as two chunks per expert, like the real checkpoint
        for k, base in m.items():
            if ".experts.experts." in k:
                t = torch.stack([theta(stage, k, e) for e in range(base.shape[0])])
                if "fc1" in k:
                    ch = [((e, h, 0), (1, 24, 32)) for e in range(4) for h in (0, 24)]
                else:
                    ch = [((e, 0, 0), (1, 32, 24)) for e in range(4)]
                w.tensor(k, rne(t), ch)
            elif k.endswith("linear_fc1.weight"):
                w.tensor(k, rne(theta(stage, k, None)), [((0, 0), (48, 32)), ((48, 0), (48, 32))])
            else:
                w.tensor(k, rne(theta(stage, k, None)), [((0,) * base.dim(), tuple(base.shape))])
        for k in fp32:
            w.tensor(k, theta32(stage, k), [((0,), tuple(fp32[k].shape))])
        # dense bf16 bucket, split into 3 DP chunks at arbitrary boundaries
        groups = [[(k, None) for k in grp] for grp in dense]
        par, spans = flat(groups, lambda k, e: theta(stage, k, None))
        n = par.numel()
        cuts = [0, 1000, 5000, n]
        dp = [((cuts[i],), (cuts[i + 1] - cuts[i],)) for i in range(3)]
        sv = torch.full((n,), SV)
        pre = "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0"
        w.tensor(pre + ".param", par, dp)
        w.tensor(pre + ".exp_avg_sq", (sv * sv) * bc2, dp)
        w.tensor(pre + ".exp_avg", sv * bc1, dp)
        # fp32 bucket
        par, spans = flat([[(k, None)] for k in fp32], lambda k, e: theta32(stage, k))
        n = par.numel()
        pre = "chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_1.dtype_(torch.float32, torch.float32).bucket_idx_0"
        sv = torch.full((n,), SV)
        w.tensor(pre + ".param", par, [((0,), (n,))])
        w.tensor(pre + ".exp_avg_sq", (sv * sv) * bc2, [((0,), (n,))])
        w.tensor(pre + ".exp_avg", sv * bc1, [((0,), (n,))])
        # expert buckets, one chunk per parameter (and per padding run); expert 3 got no gradient at all
        for r, params in experts.items():
            par, spans = flat(params, lambda k, e: theta(stage, k, e))
            n = par.numel()
            sv = torch.full((n,), SV)
            for (start, ln), grp in zip(spans, params):
                if grp[0][1] == 3:
                    sv[start:start + ln] = 0
            ch, pos = [], 0
            for start, ln in spans:
                if start > pos:
                    ch.append(((pos,), (start - pos,)))
                ch.append(((start,), (ln,)))
                pos = start + ln
            pre = "chained_1.optimizer.distributed.dp_group_idx_%d.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32)" \
                  ".bucket_idx_0" % r
            w.tensor(pre + ".param", par, ch)
            w.tensor(pre + ".exp_avg_sq", (sv * sv) * bc2, ch)
            w.tensor(pre + ".exp_avg", sv * bc1, ch)
        for c in (0, 1):
            w.obj("chained_%d.optimizer.distributed.dp_group_idx_0.optimizer/shard_0_1" % c,
                  {"optimizer": {"param_groups": [{"lr": LR, "weight_decay": WD, "step": step, "params": [0]}]}})
        w.close()
        with open(os.path.join(sd, "training_info.json"), "w") as f:
            json.dump({"current_step": step}, f)
        with open(os.path.join(sd, "config.yaml"), "w") as f:
            f.write("grpo:\n  num_prompts_per_step: 32\n  num_generations_per_prompt: 16\npolicy:\n"
                    "  train_global_batch_size: 512\n  megatron_cfg:\n    freeze_moe_router: true\n"
                    "    optimizer:\n      adam_beta1: %g\n      adam_beta2: %g\n      adam_eps: 1.0e-08\n"
                    "      lr: %g\n      min_lr: 5.0e-07\n      weight_decay: %g\n    scheduler:\n"
                    "      lr_decay_style: constant\n" % (B1, B2, LR, WD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    a = ap.parse_args()
    root = a.keep or tempfile.mkdtemp(prefix="optstats-")
    if os.path.exists(os.path.join(root, "step_30")):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    make_run(root)
    out = os.path.join(root, "out")
    script = os.path.join(HERE, "optimizer_stats.py")
    rc = subprocess.call([sys.executable, script, "run", "--ckpt-b", os.path.join(root, "step_40"), "--ckpt-a",
                          os.path.join(root, "step_30"), "--out", out, "--workers", "2", "--index-workers", "2",
                          "--threads", "1"])
    assert rc == 0, "run exited %d" % rc
    lay = json.load(open(os.path.join(out, "layout.json")))
    meta = json.load(open(os.path.join(out, "meta.json")))
    rows = dict((r["group"], r) for r in (json.loads(x) for x in gzip.open(os.path.join(out, "groups.jsonl.gz"),
                                                                              "rt")))
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(lay["numel_unmatched"] == 0, "unmatched bucket elements: %s" % lay["numel_unmatched"])
    check(lay["unmatched_chunks_by_pattern"] == {"decoder.layers.N.mlp.router.weight": 1},
          "unmatched chunks: %s" % lay["unmatched_chunks_by_pattern"])
    check(not lay["matched_twice"], "matched twice: %s" % lay["matched_twice"])
    check(all(b["a_verified"][0] == b["a_verified"][1] for b in lay["buckets"]), "A not verified")
    check(all(b["mode"] in ("rne", None) for b in lay["buckets"]), "rounding mode %s" % [b["mode"] for b in lay["buckets"]])
    check(meta["adam_step"] == 40 and meta["updates_a_to_b"] == 10 and not meta["warnings"], "meta %s" % meta)
    check(len(rows) == 15, "groups: %d %s" % (len(rows), sorted(rows)))  # 8 expert + 7 dense/fp32 tensors
    E1 = "decoder.layers.0.mlp.experts.experts.linear_fc1.weight"
    E2 = "decoder.layers.0.mlp.experts.experts.linear_fc2.weight"
    for gname, r in rows.items():
        n = r["n"]
        moving = gname in (E2 + "#e0", "decoder.layers.0.mlp.linear_fc1.weight", "decoder.layers.0.self_attention.dt_bias")
        dead = gname.endswith("#e3")
        check(n == {E1: 48 * 32, E2: 32 * 24}.get(gname.split("#")[0], n), "%s n=%d" % (gname, n))
        if dead:
            check(r["n_v0"] == n and r["damp_e8"] == 0, "%s should have no gradient" % gname)
        else:
            check(r["n_v0"] == 0, "%s n_v0=%s" % (gname, r["n_v0"]))
            sv = r["sum_sv"] / n
            check(abs(sv / SV - 1) < 1e-3, "%s sqrt(v_hat)=%g" % (gname, sv))
            check(abs(r["damp_e8"] / n - SV / (SV + 1e-8)) < 1e-3, "%s damp_e8=%g" % (gname, r["damp_e8"] / n))
            check(abs(r["step_e14"] / n - 1) < 1e-3, "%s step_e14=%g" % (gname, r["step_e14"] / n))
            check(sum(r["hist_sv"]) == n, "%s hist_sv total" % gname)
        mv = r["sum_ad"] / n / (N_UPD * LR)
        if moving:
            check(abs(mv - DELTA / (N_UPD * LR)) < 2e-3, "%s movement %g, expected %g" % (gname, mv, DELTA / (N_UPD * LR)))
        else:
            check(mv < 1e-3, "%s should not move (weight decay removed): %g" % (gname, mv))
        check(sum(r["hist_d"]) + r["n_d0"] == n, "%s hist_d total" % gname)
        check(sum(r["joint"]) == n, "%s joint total" % gname)
    rc = subprocess.call([sys.executable, script, "report", "t1=" + out, "t2=" + out,
                          "--out", os.path.join(root, "report.md")])
    check(rc == 0, "report exited %d" % rc)
    if fails:
        print("FAILED:\n  " + "\n  ".join(fails))
        return 1
    print("OK: %d groups, all checks passed (%s)" % (len(rows), root))
    if not a.keep:
        shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())

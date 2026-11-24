#!/usr/bin/env python3
"""Adam moments and fp32 weight movement of a Megatron checkpoint of this task, per tensor and per routed expert.
Read-only on the checkpoints; writes only into --out.

Question it answers (RUNBOOK "Learning speed"): do routed experts move less under Adam eps 1e-8 because eps dominates
their tiny gradient scale (sqrt(v_hat) near or below 1e-8), or because they get (almost) no gradient (v exactly 0)?

    run     one run: moments of checkpoint B, fp32 movement from checkpoint A to B
            python3 optimizer_stats.py run --ckpt-b <run>/step_40 --ckpt-a <run>/step_30 --out <dir> [--workers 6]
    report  tables per tensor class, and a comparison of two runs (stdlib only, runs with the login node's python3)
            python3 optimizer_stats.py report eps=<dir> h1=<dir> [--out report.md]

How `run` finds the tensors. The distributed optimizer stores fp32 master weights (`param`), `exp_avg` (m) and
`exp_avg_sq` (v) as flat buckets without tensor names (Megatron `dp_reshardable` format: parameters in reverse
registration order, each start aligned to 64 elements, fused parameters such as KDA in_proj back to back). The
checkpoint also stores the bf16 model weights by name, and each one is the bf16 rounding of its fp32 master. So every
named weight chunk is located in the buckets by content: its first 16 bf16 values give a candidate position, and the
whole chunk must then match exactly (128-bit hash). Every match is exact; anything unmatched is reported in
layout.json. Checkpoint A must have the identical bucket layout (checked) and is verified on a sample of segments.

Output of `run` in --out: meta.json (settings, optimizer step, param_groups), layout.json (coverage per bucket,
unmatched model chunks), groups.jsonl.gz (one line per tensor, or per expert of a routed-expert tensor).
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import time

K_FP = 16  # values in a content fingerprint
ALIGN = 64  # Megatron pad_param_start
BLOCK = 1 << 25  # elements per block for the statistics
EPS = (("e8", 1e-8), ("e14", 1e-14))
SV_LO, SV_W, SV_NB = -20.0, 0.25, 80  # histogram of log10 sqrt(v_hat): [-20, 0)
D_LO, D_W, D_NB = -8.0, 0.25, 36  # histogram of log10 |delta| / (updates * lr): [-8, 1)
J_SV, J_D = 21, 10  # joint histogram: rows v==0 + 20 decades of sqrt(v_hat), cols delta==0 + 9 decades


# ----------------------------------------------------------------------------------------------- checkpoint reading
class Ckpt(object):
    """A torch_dist checkpoint read chunk by chunk (no process group, no Megatron)."""

    def __init__(self, step_dir):
        import torch  # noqa: F401  (the pickled metadata refers to torch classes)
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata
        self.step_dir = os.path.abspath(step_dir)
        self.dir = os.path.join(self.step_dir, "policy", "weights", "iter_0000000")
        with open(os.path.join(self.dir, ".metadata"), "rb") as f:
            md = pickle.load(f)
        self.sd = md.state_dict_metadata
        self.tensor_keys = set(k for k, v in self.sd.items() if isinstance(v, TensorStorageMetadata))
        self.chunks = collections.defaultdict(list)  # fqn -> [(offsets, sizes, storage info)] sorted by offsets
        self.objects = {}  # fqn -> storage info of a bytes object
        for idx, info in md.storage_data.items():
            if getattr(info, "transform_descriptors", None):
                raise SystemExit("ERROR: transformed (compressed) checkpoint items are not supported: %s" % idx.fqn)
            if idx.offset is None:
                self.objects[idx.fqn] = info
            else:
                self.chunks[idx.fqn].append([tuple(idx.offset), None, info])
        for fqn, lst in self.chunks.items():
            sizes = dict((tuple(c.offsets), tuple(c.sizes)) for c in self.sd[fqn].chunks)
            for c in lst:
                c[1] = sizes[c[0]]
            lst.sort(key=lambda c: c[0])
        with open(os.path.join(self.step_dir, "training_info.json")) as f:
            self.training_info = json.load(f)

    def read_bytes(self, info):
        with open(os.path.join(self.dir, info.relative_path), "rb") as f:
            f.seek(info.offset)
            b = f.read(info.length)
        if len(b) != info.length:
            raise IOError("short read %s @%d" % (info.relative_path, info.offset))
        return b

    def read_chunk(self, info):
        import torch
        return torch.load(io.BytesIO(self.read_bytes(info)), map_location="cpu", weights_only=True)

    def read_object(self, fqn):
        import torch
        return torch.load(io.BytesIO(self.read_bytes(self.objects[fqn])), map_location="cpu", weights_only=False)

    def read_flat(self, fqn):
        """A 1-D tensor key (an optimizer bucket) assembled from its chunks."""
        import torch
        md = self.sd[fqn]
        out = torch.empty(int(md.size[0]), dtype=md.properties.dtype)
        covered = 0
        for off, size, info in self.chunks[fqn]:
            t = self.read_chunk(info).reshape(-1)
            if t.numel() != size[0]:
                raise ValueError("%s chunk @%d: %d elements, expected %d" % (fqn, off[0], t.numel(), size[0]))
            out[off[0]:off[0] + size[0]] = t
            covered += size[0]
        if covered != out.numel():
            raise ValueError("%s: chunks cover %d of %d elements" % (fqn, covered, out.numel()))
        return out

    def layout(self, fqn):
        return [(c[0], c[1]) for c in self.chunks[fqn]]


def buckets(ck):
    """Optimizer bucket prefixes (without .param) and their parameter dtype tag."""
    out = []
    for k in ck.tensor_keys:
        if ".optimizer.distributed." in k and k.endswith(".param"):
            p = k[:-len(".param")]
            out.append((p, "bf16" if "dtype_(torch.bfloat16" in p else "fp32"))
    return sorted(out)


def model_pieces(ck, expert_filter=None, key_filter=None):
    """Every stored chunk of every named model tensor (bf16 or fp32), as picklable descriptors."""
    import torch
    out = []
    for fqn in sorted(ck.tensor_keys):
        if ".optimizer." in fqn or fqn.startswith("optimizer"):
            continue
        dt = ck.sd[fqn].properties.dtype
        tag = {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(dt)
        if tag is None or (key_filter and not re.search(key_filter, fqn)):
            continue
        expert = ".experts.experts." in fqn
        for off, size, info in ck.chunks[fqn]:
            if expert and expert_filter is not None and off[0] not in expert_filter:
                continue
            out.append({"id": len(out), "fqn": fqn, "off": off, "size": size, "tag": tag,
                        "ndim": len(ck.sd[fqn].size), "expert": off[0] if expert else None, "info": info})
    return out


# ------------------------------------------------------------------------------------------ content fingerprints
def tagged(x, tag, mode):
    """Integer view of the values as they are stored for this tag: bf16 by round-to-nearest-even ('rne', what
    Tensor.to(bfloat16) and copy_ do) or by truncation ('trunc'), fp32 as is."""
    import torch
    if tag == "fp32":
        return x.contiguous().view(torch.int32)
    if x.dtype == torch.bfloat16:
        return x.contiguous().view(torch.int16)
    if mode == "rne":
        return x.to(torch.bfloat16).view(torch.int16)
    return (x.contiguous().view(torch.int32) >> 16).to(torch.int16)


def digest(t):
    return hashlib.blake2b(t.numpy().tobytes(), digest_size=16).digest()


def _index_piece(p):
    t = CTX["B"].read_chunk(p["info"]).reshape(-1)
    n = t.numel()
    if n < K_FP:
        return p["id"], None, None, n
    v = tagged(t, p["tag"], None)
    return p["id"], v[:K_FP].numpy().tobytes(), digest(v), n


def build_index(pieces, workers):
    import multiprocessing as mp
    index = {}  # (tag, fingerprint) -> [piece id]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(2,)) as pool:
        for i, (pid, fp, h, n) in enumerate(pool.imap_unordered(_index_piece, pieces, chunksize=8)):
            p = pieces[pid]
            p["n"], p["hash"] = n, h
            if fp is not None:
                index.setdefault((p["tag"], fp), []).append(pid)
            if (i + 1) % 5000 == 0:
                log("index: %d/%d model chunks, %.0f s" % (i + 1, len(pieces), time.time() - t0))
    log("index: %d model chunks, %d fingerprints, %.0f s" % (len(pieces), len(index), time.time() - t0))
    return index


def walk(theta, tag, index, pieces, name):
    """Locate model chunks in a flat fp32 master-weight bucket. Returns segments [(start, piece id)], a report."""
    n_tot = theta.numel()
    segs, used, ambiguous = [], set(), 0
    rep = {"bucket": name, "numel": n_tot, "padding": 0, "unmatched": 0, "unmatched_runs": [], "mode": None}
    modes = ["rne", "trunc"] if tag == "bf16" else [None]

    def try_at(p):
        if n_tot - p < K_FP:
            return None
        for mode in ([rep["mode"]] if rep["mode"] else modes):
            fp = tagged(theta[p:p + K_FP], tag, mode).numpy().tobytes()
            cands = [c for c in index.get((tag, fp), ()) if c not in used and pieces[c]["n"] <= n_tot - p]
            hits = [c for c in cands if digest(tagged(theta[p:p + pieces[c]["n"]], tag, mode)) == pieces[c]["hash"]]
            if hits:
                rep["mode"] = mode
                return hits
        return None

    pos, max_scan = 0, 1 << 20
    while pos < n_tot:
        hits = try_at(pos)  # a fused continuation starts right after the previous chunk
        if not hits:
            p = pos + (-pos) % ALIGN
            if p == pos:
                p += ALIGN
            scans = 0
            while p + K_FP <= n_tot and scans < max_scan:
                hits = try_at(p)
                if hits:
                    break
                p += ALIGN
                scans += 1
            gap_end = p if hits else n_tot
            gap = gap_end - pos
            if hits and gap < ALIGN:
                rep["padding"] += gap
            elif not hits and gap < ALIGN * 4:  # bucket-end padding is excluded from the stored size
                rep["padding"] += gap
            else:
                rep["unmatched"] += gap
                if len(rep["unmatched_runs"]) < 20:
                    rep["unmatched_runs"].append([pos, gap])
                if not hits and scans >= max_scan:
                    log("WARNING %s: nothing matched in %d aligned positions after %d; giving up on the rest"
                        % (name, max_scan, pos))
            if not hits:
                break
            pos = p
        if len(hits) > 1:
            ambiguous += 1
        c = hits[0]
        segs.append((pos, c))
        used.add(c)
        pos += pieces[c]["n"]
    rep["segments"], rep["ambiguous"] = len(segs), ambiguous
    rep["matched"] = sum(pieces[c]["n"] for _, c in segs)
    return segs, rep


# ------------------------------------------------------------------------------------------------------ statistics
def new_acc():
    import torch
    a = collections.defaultdict(float)
    a["hist_sv"] = torch.zeros(SV_NB, dtype=torch.int64)
    a["hist_d"] = torch.zeros(D_NB, dtype=torch.int64)
    a["joint"] = torch.zeros(J_SV * J_D, dtype=torch.int64)
    return a


def bins(logx, lo, w, nb):
    import torch
    return torch.bincount(((logx - lo) / w).floor().clamp(0, nb - 1).long(), minlength=nb)


def moment_stats(a, v, m, bc1, bc2):
    import torch
    f64 = torch.float64
    sv = torch.sqrt(v / bc2)
    am = m.abs() / bc1
    z = v == 0
    nz = ~z
    a["n"] += v.numel()
    a["n_v0"] += int(z.sum())
    a["n_m0"] += int((m == 0).sum())
    a["sum_sv"] += float(sv.sum(dtype=f64))
    a["sum_am"] += float(am.sum(dtype=f64))
    pos = sv[nz]
    if pos.numel():
        a["hist_sv"] += bins(torch.log10(pos), SV_LO, SV_W, SV_NB)
        a["sum_snr"] += float((am[nz] / pos).sum(dtype=f64))
    for name, e in EPS:
        a["damp_" + name] += float((sv / (sv + e)).sum(dtype=f64))  # how much of the eps-free update survives
        a["step_" + name] += float((am / (sv + e)).sum(dtype=f64))  # |Adam direction| per unit learning rate


def delta_stats(a, ta, tb, v, bc2, wd_factor, norm):
    import torch
    f64 = torch.float64
    d = (tb.double() - ta.double() * wd_factor).abs()
    a["n_d0"] += int((d == 0).sum())  # the zero bin of hist_d
    a["n_raw_equal"] += int((tb == ta).sum())
    a["sum_ad"] += float(d.sum())
    a["sum_ad2"] += float((d * d).sum())
    a["sum_abs_w"] += float(ta.abs().sum(dtype=f64))
    r = d / norm
    rp = r[r > 0]
    if rp.numel():
        a["hist_d"] += bins(torch.log10(rp), D_LO, D_W, D_NB)
    # bf16 visibility: what an export-to-export comparison sees, and moves larger than half a bf16 step of the
    # starting value (a bf16 step is 2^(e-8) for |w| in [2^(e-1), 2^e), so half of it is 2^(e-9))
    a["n_bf16_vis"] += int((ta.bfloat16() != tb.bfloat16()).sum())
    _, ex = torch.frexp(ta)
    half = torch.ldexp(torch.ones_like(ta, dtype=f64), (ex - 9).to(torch.int64))
    a["n_gt_half"] += int(torch.where(ta == 0, d > 0, d > half).sum())
    sv = torch.sqrt(v / bc2)
    row = torch.where(v == 0, torch.zeros_like(sv, dtype=torch.int64),
                      1 + (torch.log10(sv.clamp_min(1e-30)).floor() + 20).clamp(0, 19).long())
    col = torch.where(r == 0, torch.zeros_like(r, dtype=torch.int64),
                      1 + (torch.log10(r.clamp_min(1e-30)).floor() + 8).clamp(0, 8).long())
    a["joint"] += torch.bincount(row * J_D + col, minlength=J_SV * J_D)


def group_of(p):
    return p["fqn"] if p["expert"] is None else "%s#e%d" % (p["fqn"], p["expert"])


def weight_decay_applies(p):
    """Megatron's default: no weight decay for biases and 1-D parameters (local shape; experts drop dim 0)."""
    local_ndim = p["ndim"] - (1 if p["expert"] is not None else 0)
    return local_ndim >= 2 and not p["fqn"].endswith(".bias")


# ------------------------------------------------------------------------------------------------- bucket worker
CTX = {}


def _worker_init(threads):
    import torch
    torch.set_num_threads(threads)


def log(msg):
    sys.stdout.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def _do_bucket(job):
    import torch
    prefix, tag = job
    B, A, pieces, index, s = CTX["B"], CTX.get("A"), CTX["pieces"], CTX["index"], CTX["settings"]
    t0 = time.time()
    short = re.sub(r"\.optimizer\.distributed\.", ".", prefix).replace("dtype_(torch.bfloat16, torch.float32)", "bf16") \
        .replace("dtype_(torch.float32, torch.float32)", "fp32")
    tb = B.read_flat(prefix + ".param")
    segs, rep = walk(tb, tag, index, pieces, short)
    accs = collections.defaultdict(new_acc)
    v = B.read_flat(prefix + ".exp_avg_sq")
    m = B.read_flat(prefix + ".exp_avg")
    for start, c in segs:
        n = pieces[c]["n"]
        acc = accs[group_of(pieces[c])]
        for i in range(start, start + n, BLOCK):
            j = min(i + BLOCK, start + n)
            moment_stats(acc, v[i:j], m[i:j], s["bc1"], s["bc2"])
    del m
    if A is not None:
        ta = A.read_flat(prefix + ".param")
        rep["a_verified"] = verify_a(ta, tag, segs, rep["mode"])
        for start, c in segs:
            p = pieces[c]
            wdf = (1.0 - s["lr"] * s["wd"]) ** s["updates"] if weight_decay_applies(p) else 1.0
            acc = accs[group_of(p)]
            for i in range(start, start + p["n"], BLOCK):
                j = min(i + BLOCK, start + p["n"])
                delta_stats(acc, ta[i:j], tb[i:j], v[i:j], s["bc2"], wdf, s["updates"] * s["lr"])
        del ta
    rep["seconds"] = round(time.time() - t0, 1)
    log("%s: %d segments, %d/%d matched, %d padding, %d unmatched, mode %s, %.0f s"
        % (short, rep["segments"], rep["matched"], rep["numel"], rep["padding"], rep["unmatched"], rep["mode"],
           rep["seconds"]))
    out = {}
    for g, a in accs.items():
        out[g] = dict((k, (val.tolist() if hasattr(val, "tolist") else val)) for k, val in a.items())
    return rep, out, [c for _, c in segs]


def verify_a(ta, tag, segs, mode, sample=4):
    """Checkpoint A: the same bucket position holds the same named chunk (compared with A's own bf16 weights)."""
    A, pieces = CTX["A"], CTX["pieces"]
    picks = segs[:sample // 2] + segs[-(sample - sample // 2):] if len(segs) > sample else segs
    ok = 0
    for start, c in picks:
        p = pieces[c]
        info = dict((tuple(o), i) for o, _, i in A.chunks[p["fqn"]]).get(tuple(p["off"]))
        if info is None:
            continue
        want = digest(tagged(A.read_chunk(info).reshape(-1), p["tag"], None))
        ok += digest(tagged(ta[start:start + p["n"]], tag, mode)) == want
    return [ok, len(picks)]


# ---------------------------------------------------------------------------------------------------------- run
def load_config(step_dir):
    import yaml
    with open(os.path.join(step_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def cmd_run(a):
    import multiprocessing as mp
    import torch
    os.makedirs(a.out, exist_ok=True)
    B = Ckpt(a.ckpt_b)
    A = Ckpt(a.ckpt_a) if a.ckpt_a else None
    cfg = load_config(B.step_dir)
    opt = cfg["policy"]["megatron_cfg"]["optimizer"]
    sched = cfg["policy"]["megatron_cfg"].get("scheduler", {})
    grpo = cfg["grpo"]
    rollouts = grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    gbs = cfg["policy"]["train_global_batch_size"]
    if rollouts % gbs:
        raise SystemExit("ERROR: %d rollouts per step is not a multiple of the global batch %d" % (rollouts, gbs))
    ups = rollouts // gbs
    step_b = int(B.training_info["current_step"])
    step_a = int(A.training_info["current_step"]) if A else None
    t = step_b * ups if a.adam_step is None else a.adam_step
    b1, b2 = float(opt["adam_beta1"]), float(opt["adam_beta2"])
    s = {"bc1": 1.0 - b1 ** t, "bc2": 1.0 - b2 ** t, "lr": float(opt["lr"]), "wd": float(opt["weight_decay"]),
         "updates": (step_b - step_a) * ups if A else 0}
    meta = {"ckpt_b": B.step_dir, "ckpt_a": A.step_dir if A else None, "step_b": step_b, "step_a": step_a,
            "updates_per_step": ups, "adam_step": t, "adam_eps": float(opt["adam_eps"]), "beta1": b1, "beta2": b2,
            "lr": s["lr"], "min_lr": opt.get("min_lr"), "lr_decay_style": sched.get("lr_decay_style"),
            "weight_decay": s["wd"], "updates_a_to_b": s["updates"], "freeze_moe_router":
            cfg["policy"]["megatron_cfg"].get("freeze_moe_router"), "warnings": []}
    if A and sched.get("lr_decay_style") != "constant":
        meta["warnings"].append("learning rate not constant: movement is normalized by the peak lr")
    if A and load_config(A.step_dir)["policy"]["megatron_cfg"]["optimizer"] != opt:
        meta["warnings"].append("checkpoint A has different optimizer settings")
    # The optimizer's own param_groups (step count, weight decay per group): a cross-check of the settings above.
    meta["param_groups"] = {}
    for fqn in sorted(B.objects):
        if fqn.endswith("optimizer/shard_0_1") and ".optimizer.distributed." in fqn:
            try:
                obj = B.read_object(fqn)
                groups = obj.get("optimizer", obj).get("param_groups", []) if isinstance(obj, dict) else []
                meta["param_groups"][fqn] = [dict((k, (v if isinstance(v, (int, float, str, bool, type(None)))
                                                        else str(v))) for k, v in g.items() if k != "params")
                                             for g in groups]
                for g in groups:
                    g_step = g.get("step")
                    if g_step is not None and int(g_step) != t:
                        meta["warnings"].append("%s: param_group step %s != adam_step %d" % (fqn, g_step, t))
            except Exception as e:  # informative only
                meta["param_groups"][fqn] = "unreadable: %s" % e
    log("B %s step %d, A %s, adam_step %d (%d update(s)/step), eps %g, lr %g, %d updates A->B"
        % (B.step_dir, step_b, A.step_dir if A else "-", t, ups, meta["adam_eps"], s["lr"], s["updates"]))
    for w in meta["warnings"]:
        log("WARNING " + w)

    bk = buckets(B)
    if a.buckets:
        bk = [b for b in bk if re.search(a.buckets, b[0])]
    if A:
        for p, _ in bk:
            for suffix in (".param",):
                if p + suffix not in A.tensor_keys or A.layout(p + suffix) != B.layout(p + suffix):
                    raise SystemExit("ERROR: checkpoint A has a different layout for %s%s" % (p, suffix))
    experts = None
    if a.experts:
        lo, hi = [int(x) for x in a.experts.split("-")]
        experts = set(range(lo, hi + 1))
    pieces = model_pieces(B, experts, a.index_filter)
    log("%d buckets, %d model chunks to index" % (len(bk), len(pieces)))
    CTX.update(B=B, A=A, pieces=pieces, settings=s)
    index = build_index(pieces, a.index_workers)
    CTX["index"] = index

    sizes = dict((p, int(B.sd[p + ".param"].size[0])) for p, _ in bk)
    bk.sort(key=lambda b: -sizes[b[0]])
    reports, groups, seen = [], {}, collections.Counter()
    with mp.get_context("fork").Pool(a.workers, initializer=_worker_init, initargs=(a.threads,),
                                     maxtasksperchild=1) as pool:
        for rep, out, used in pool.imap_unordered(_do_bucket, bk):
            reports.append(rep)
            seen.update(used)
            for g, st in out.items():
                if g in groups:  # a tensor split across buckets: add up
                    for k, v in st.items():
                        groups[g][k] = [x + y for x, y in zip(groups[g][k], v)] if isinstance(v, list) \
                            else groups[g][k] + v
                else:
                    groups[g] = st

    unmatched = [p for p in pieces if p["id"] not in seen]
    twice = [p["fqn"] for p in pieces if seen[p["id"]] > 1]
    by_pattern = collections.Counter(re.sub(r"\.\d+\.", ".N.", p["fqn"]) for p in unmatched)
    layout = {"buckets": sorted(reports, key=lambda r: r["bucket"]),
              "model_chunks": len(pieces), "matched_chunks": len(pieces) - len(unmatched),
              "unmatched_chunks_by_pattern": dict(by_pattern.most_common()), "matched_twice": twice[:50],
              "numel_buckets": sum(r["numel"] for r in reports), "numel_matched": sum(r["matched"] for r in reports),
              "numel_unmatched": sum(r["unmatched"] for r in reports)}
    with open(os.path.join(a.out, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pmeta = {}
    for p in pieces:
        pmeta.setdefault(group_of(p), {"fqn": p["fqn"], "expert": p["expert"], "wd": weight_decay_applies(p)})
    with gzip.open(os.path.join(a.out, "groups.jsonl.gz"), "wt") as f:
        for g in sorted(groups):
            row = dict(pmeta[g])
            row["group"] = g
            row.update(groups[g])
            f.write(json.dumps(row) + "\n")
    log("done: %d groups; matched %d of %d bucket elements (%d unmatched); %d of %d model chunks matched; "
        "unmatched chunks: %s" % (len(groups), layout["numel_matched"], layout["numel_buckets"],
                                   layout["numel_unmatched"], layout["matched_chunks"], len(pieces),
                                   dict(by_pattern.most_common(6))))
    bad = [r for r in reports if r["unmatched"] or r["ambiguous"] or
           (r.get("a_verified") and r["a_verified"][0] != r["a_verified"][1])]
    if bad or twice:
        log("ATTENTION: %d bucket(s) with unmatched/ambiguous/unverified segments, %d chunk(s) matched twice; "
            "see layout.json" % (len(bad), len(twice)))
        return 2
    return 0


# ------------------------------------------------------------------------------------------------------- report
CLASSES = [
    ("routed experts fc1", r"\.experts\.experts\.linear_fc1\."),
    ("routed experts fc2", r"\.experts\.experts\.linear_fc2\."),
    ("shared experts", r"\.shared_experts\."),
    ("latent projections", r"_latent_proj\."),
    ("norms, biases", r"(layernorm|layer_norm_weight|out_norm|\.bias$)"),
    ("dense MLP (layers 0-2)", r"\.mlp\.linear_fc[12]\.weight$"),
    ("softmax attention", r"\.self_attention\.(linear_qkv|linear_proj)\.weight$"),
    ("KDA fp32 (A_log, dt_bias)", r"\.self_attention\.(A_log|dt_bias)$"),
    ("KDA attention", r"\.self_attention\."),
    ("embedding", r"^embedding\."),
    ("output layer", r"^output_layer\."),
    ("router (should be frozen)", r"\.router\."),
]


def klass(fqn):
    for name, rx in CLASSES:
        if re.search(rx, fqn):
            return name
    return "other"


def load_run(d):
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    rows = []
    with gzip.open(os.path.join(d, "groups.jsonl.gz"), "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return meta, rows


def add(acc, row):
    for k, v in row.items():
        if isinstance(v, list):
            acc[k] = [x + y for x, y in zip(acc[k], v)] if k in acc else list(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and k != "expert":
            acc[k] = acc.get(k, 0) + v


def hist_quantile(hist, n_zero, n, q, lo, w):
    """Quantile from a log10 histogram with an extra zero count below it (bin centres; None if all zero)."""
    target, c = q * n, n_zero
    if c >= target:
        return 0.0
    for i, h in enumerate(hist):
        c += h
        if c >= target:
            return 10 ** (lo + (i + 0.5) * w)
    return 10 ** (lo + len(hist) * w)


def share_below(hist, n_zero, n, x, lo, w):
    k = int(math.floor((math.log10(x) - lo) / w))
    return (n_zero + sum(hist[:max(0, k)])) / float(n) if n else float("nan")


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if pct:
        return "%.1f %%" % (100 * x)
    return "%.2g" % x if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4)) else "%.3g" % x


def summarize(meta, rows):
    cls = collections.OrderedDict()
    experts = collections.defaultdict(list)
    for r in rows:
        k = klass(r["fqn"])
        cls.setdefault(k, {})
        add(cls[k], r)
        if r.get("expert") is not None and r.get("n"):
            experts[k].append(r)
    return cls, experts


def class_table(tag, meta, cls):
    own = "e8" if abs(meta["adam_eps"] - 1e-8) < 1e-12 else "e14"
    moved = meta.get("updates_a_to_b")
    lines = ["### %s: step %s (adam step %s, eps %g)%s" % (
        tag, meta["step_b"], meta["adam_step"], meta["adam_eps"],
        (", movement from step %s (%d updates, lr %g)" % (meta["step_a"], moved, meta["lr"])) if moved else ""), "",
        "| class | elements | v = 0 | median √v̂ | √v̂ < 1e-8 | kept at ε 1e-8 | kept at ε 1e-14 | "
        "\\|step\\|/lr at own ε | median \\|Δ\\|/(n·lr) | mean \\|Δ\\|/(n·lr) | moved > ½ bf16 step | bf16 changed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in cls.items():
        n = a.get("n", 0)
        if not n:
            continue
        nd = (moved or 0) * meta["lr"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k, fmt(float(n)), fmt(a["n_v0"] / n, True),
            fmt(hist_quantile(a["hist_sv"], a["n_v0"], n, 0.5, SV_LO, SV_W)),
            fmt(share_below(a["hist_sv"], a["n_v0"], n, 1e-8, SV_LO, SV_W), True),
            fmt(a["damp_e8"] / n, True), fmt(a["damp_e14"] / n, True), fmt(a["step_" + own] / n),
            fmt(hist_quantile(a["hist_d"], a.get("n_d0", 0), n, 0.5, D_LO, D_W)) if moved else "–",
            fmt(a["sum_ad"] / n / nd) if moved else "–",
            fmt(a["n_gt_half"] / n, True) if moved else "–", fmt(a["n_bf16_vis"] / n, True) if moved else "–"))
    return lines


def expert_table(tag, meta, experts):
    lines = ["", "Per routed expert (%s): experts with no gradient at all (v = 0 everywhere), and the spread over experts"
             " of the mean √v̂ and of the mean movement." % tag, "",
             "| class | experts | v = 0 everywhere | mean √v̂: p10 / p50 / p90 | mean \\|Δ\\|/(n·lr): p10 / p50 / p90 |",
             "|---|---|---|---|---|"]
    nd = (meta.get("updates_a_to_b") or 0) * meta["lr"]
    for k, rs in experts.items():
        sv = sorted(r["sum_sv"] / r["n"] for r in rs)
        dd = sorted(r["sum_ad"] / r["n"] / nd for r in rs) if nd else []
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None  # noqa: E731
        lines.append("| %s | %d | %d | %s / %s / %s | %s |" % (
            k, len(rs), sum(1 for r in rs if r["n_v0"] == r["n"]), fmt(q(sv, .1)), fmt(q(sv, .5)), fmt(q(sv, .9)),
            " / ".join(fmt(q(dd, p)) for p in (.1, .5, .9)) if dd else "–"))
    return lines


def compare_table(runs):
    """Observed movement ratio between two runs per class, and the ratio the moments predict for eps 1e-8 vs 1e-14."""
    (ta, ma, ca), (tb, mb, cb) = runs[0], runs[1]
    lines = ["", "### %s vs %s" % (ta, tb), "",
             "Observed: mean |Δ| of %s / mean |Δ| of %s over the same updates. Predicted: mean |Adam step| at ε 1e-8 / "
             "at ε 1e-14 from each run's own moments (the share of the update that ε 1e-8 would keep, weighted by "
             "|m̂|). If ε explains the difference, the observed ratio is near the predicted one." % (ta, tb), "",
             "| class | observed %s/%s | predicted from %s moments | predicted from %s moments |" % (ta, tb, ta, tb),
             "|---|---|---|---|"]
    for k in ca:
        a, b = ca[k], cb.get(k)
        if not b or not a.get("n") or not b.get("n"):
            continue
        obs = None
        if ma.get("updates_a_to_b") and mb.get("updates_a_to_b") and b.get("sum_ad"):
            obs = (a["sum_ad"] / a["n"] / ma["updates_a_to_b"] / ma["lr"]) / \
                  (b["sum_ad"] / b["n"] / mb["updates_a_to_b"] / mb["lr"])
        pa = a["step_e8"] / a["step_e14"] if a.get("step_e14") else None
        pb = b["step_e8"] / b["step_e14"] if b.get("step_e14") else None
        lines.append("| %s | %s | %s | %s |" % (k, fmt(obs), fmt(pa), fmt(pb)))
    return lines


def cmd_report(a):
    runs, out = [], ["# Adam moments and fp32 movement per tensor class", ""]
    for spec in a.runs:
        tag, d = spec.split("=", 1)
        meta, rows = load_run(d)
        cls, experts = summarize(meta, rows)
        runs.append((tag, meta, cls))
        out += class_table(tag, meta, cls) + expert_table(tag, meta, experts) + [""]
        for w in meta.get("warnings", []):
            out.append("- WARNING: " + w)
    if len(runs) >= 2:
        out += compare_table(runs[:2])
    out += ["", "Columns: v = 0: elements with no gradient in any update; √v̂ = sqrt(v / (1 - β2^t)); kept at ε: "
            "mean of √v̂/(√v̂+ε), the share of the ε-free update size that survives; |step|/lr: mean of |m̂|/(√v̂+ε); "
            "|Δ|: fp32 movement from A to B with the weight-decay shrink removed (tensors with ≥ 2 dimensions); "
            "n·lr: updates times learning rate (the largest typical Adam move); moved > ½ bf16 step: share of weights "
            "a bf16 export can show at all; bf16 changed: share whose bf16 value changed."]
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="statistics of one run (needs torch)")
    r.add_argument("--ckpt-b", required=True, help="step folder whose moments are read, e.g. <run>/step_40")
    r.add_argument("--ckpt-a", help="earlier step folder of the same run, for the fp32 movement A -> B")
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=6, help="buckets in parallel (each holds 3 x 14 GB for experts)")
    r.add_argument("--threads", type=int, default=8, help="torch threads per bucket worker")
    r.add_argument("--index-workers", type=int, default=48)
    r.add_argument("--adam-step", type=int, help="optimizer step for bias correction (default: step x updates/step)")
    r.add_argument("--buckets", help="regex: only these buckets (tests)")
    r.add_argument("--experts", help="lo-hi: only index these routed experts (tests)")
    r.add_argument("--index-filter", help="regex: only index model tensors matching it (tests)")
    p = sub.add_parser("report", help="tables from one or two runs (stdlib only)")
    p.add_argument("runs", nargs="+", metavar="TAG=DIR")
    p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "report":
        return cmd_report(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare two HF checkpoints by their safetensors headers only: tensor names, shapes and dtypes, and the index.
    python3 compare_safetensors.py <ckpt> <start>
Also checks every shard of the index exists and that each shard's data size matches its header."""
import glob
import json
import os
import struct
import sys


def headers(d):
    out, problems = {}, []
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        end = max((v["data_offsets"][1] for v in h.values()), default=0)
        if os.path.getsize(f) != 8 + n + end:
            problems.append(f"{os.path.basename(f)}: size {os.path.getsize(f)} != header {8 + n + end}")
        for k, v in h.items():
            out[k] = (v["dtype"], tuple(v["shape"]), os.path.basename(f))
    return out, problems


a, pa = headers(sys.argv[1])
b, pb = headers(sys.argv[2])
idx = json.load(open(os.path.join(sys.argv[1], "model.safetensors.index.json")))["weight_map"]
print(f"ckpt: {len(a)} tensors in {len(set(v[2] for v in a.values()))} shards; start: {len(b)} tensors")
print("shard size problems:", pa or "none")
print("index entries", len(idx), "| index names == tensors:", set(idx) == set(a),
      "| index shards exist:", all(os.path.exists(os.path.join(sys.argv[1], s)) for s in set(idx.values())))
only_a, only_b = set(a) - set(b), set(b) - set(a)
diff = [k for k in set(a) & set(b) if a[k][:2] != b[k][:2]]
print("names only in ckpt:", len(only_a), sorted(only_a)[:5])
print("names only in start:", len(only_b), sorted(only_b)[:5])
print("dtype/shape differences:", len(diff), [(k, a[k][:2], b[k][:2]) for k in sorted(diff)[:5]])
print("dtypes:", sorted({v[0] for v in a.values()}))
ok = not (pa or only_a or only_b or diff) and set(idx) == set(a)
print("RESULT:", "OK" if ok else "MISMATCH")

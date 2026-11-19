#!/usr/bin/env python3
"""Minimal TensorBoard event-file reader (scalars only), no TensorFlow needed (Python 3.6+).
    tfevents.py FILE tags                 -> tag, number of points, first/last step
    tfevents.py FILE dump > out.jsonl     -> {"tag", "step", "value"} per scalar
    tfevents.py FILE trend TAG [TAG ...]  -> per tag: mean of the first and the last quarter of the steps, min, max
The runs' event files are in $SCRATCH/nemo-rl/logs/<RUN>/ (RUNBOOK.md, "Learning speed" used train/gen_kl_error,
train/approx_entropy, train/probs_ratio_min/max, train/grad_norm and train/moe/*).
"""
import json
import struct
import sys
from collections import defaultdict


def varint(b, i):
    shift = result = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if not c & 0x80:
            return result, i
        shift += 7


def fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = varint(b, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        elif wt == 2:
            ln, i = varint(b, i)
            v = b[i:i + ln]; i += ln
        else:
            raise ValueError(f"wire type {wt}")
        yield num, wt, v


def tensor_value(b):
    dtype = None
    for num, wt, v in fields(b):
        if num == 1:
            dtype = v
        elif num == 5:  # float_val
            return struct.unpack("<f", v)[0] if wt == 5 else struct.unpack("<f", v[:4])[0]
        elif num == 6:  # double_val
            return struct.unpack("<d", v)[0] if wt == 1 else struct.unpack("<d", v[:8])[0]
        elif num == 4 and v:  # tensor_content
            return struct.unpack("<f", v[:4])[0] if dtype == 1 else struct.unpack("<d", v[:8])[0]
    return None


def records(path):
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i + 12 <= len(data):
        (ln,) = struct.unpack("<Q", data[i:i + 8])
        yield data[i + 12:i + 12 + ln]
        i += 12 + ln + 4


def scalars(path):
    for rec in records(path):
        step, summary = 0, None
        for num, wt, v in fields(rec):
            if num == 2:
                step = v
            elif num == 5:
                summary = v
        if summary is None:
            continue
        for num, wt, val in fields(summary):
            if num != 1:
                continue
            tag, value = None, None
            for n2, w2, v2 in fields(val):
                if n2 == 1:
                    tag = v2.decode()
                elif n2 == 2:
                    value = struct.unpack("<f", v2)[0]
                elif n2 == 8:
                    value = tensor_value(v2)
            if tag is not None and value is not None:
                yield tag, step, value


def main():
    path, mode = sys.argv[1], sys.argv[2]
    if mode == "dump":
        for tag, step, value in scalars(path):
            print(json.dumps({"tag": tag, "step": step, "value": value}))
        return
    if mode == "trend":
        want = set(sys.argv[3:])
        series = defaultdict(dict)
        for tag, step, value in scalars(path):
            if tag in want:
                series[tag][step] = value
        for tag in sys.argv[3:]:
            v = [series[tag][s] for s in sorted(series[tag])]
            if not v:
                print(f"{tag}\tno points")
                continue
            q = max(1, len(v) // 4)
            print(f"{tag}\tn={len(v)}\tfirst quarter {sum(v[:q]) / q:.4g}\tlast quarter {sum(v[-q:]) / q:.4g}"
                  f"\tmin {min(v):.4g}\tmax {max(v):.4g}")
        return
    info = defaultdict(list)
    for tag, step, value in scalars(path):
        info[tag].append(step)
    for tag in sorted(info):
        s = info[tag]
        print(f"{tag}\t{len(s)}\t{min(s)}..{max(s)}")


if __name__ == "__main__":
    main()

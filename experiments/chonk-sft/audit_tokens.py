#!/usr/bin/env python3
"""Checks tokenized SFT data against the format's reference encoder, and counts what it contains.

Each sampled conversation is split into frames at its control tokens, rebuilt as typed messages and
encoded again, which must give back the stored tokens exactly. Run it with the venv from setup.sh:
    venv/bin/python audit_tokens.py tokenizer /scratch/hanint05/data/v0_1_chonk_sft_tokenized report.json
"""
import argparse
import collections
import glob
import json
import random
import struct

import numpy as np
from apertus_common import (
    Call,
    Claim,
    Conversation,
    Document,
    Host,
    Reply,
    Result,
    SystemPrompt,
    Think,
    UnknownOutput,
    User,
    Wait,
    checkers_for,
    load_encoding,
)

DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}
INPUTS = {"user": User, "host": Host}
OUTPUTS = {"think": Think, "reply": Reply, "claim": Claim}


class Dataset:
    """Megatron's indexed format, read by seeking: the login nodes limit address space, so no mmap."""

    def __init__(self, prefix):
        with open(f"{prefix}.idx", "rb") as index:
            head = index.read(34)
        self.dtype = np.dtype(DTYPES[head[17]])
        count = struct.unpack("<Q", head[18:26])[0]
        self.lengths = np.fromfile(f"{prefix}.idx", dtype=np.int32, count=count, offset=34)
        self.pointers = np.fromfile(f"{prefix}.idx", dtype=np.int64, count=count, offset=34 + 4 * count)
        self.tokens = open(f"{prefix}.bin", "rb")

    def __getitem__(self, index):
        self.tokens.seek(int(self.pointers[index]))
        data = self.tokens.read(int(self.lengths[index]) * self.dtype.itemsize)
        return np.frombuffer(data, dtype=self.dtype).tolist()


def frames(ids, encoding):
    """Yields None for a wait, else (is_input, header, payload)."""
    controls = encoding.controls

    def text(part):
        return b"".join(encoding.token_bytes(token) for token in part).decode("utf-8")

    position = 1  # after BOS
    while position < len(ids):
        token = ids[position]
        if token == controls.wait:
            yield None
            position += 1
            continue
        if token not in (controls.input_start, controls.output_start):
            raise ValueError(f"stray token {token} at {position}")
        is_input = token == controls.input_start
        close = controls.input_end if is_input else controls.output_end
        separator = ids.index(controls.header_end, position)
        end = ids.index(close, separator)
        yield is_input, text(ids[position + 1 : separator]), text(ids[separator + 1 : end])
        position = end + 1


def rebuild(ids, encoding):
    system, items, types = None, [], []
    for frame in frames(ids, encoding):
        if frame is None:
            items.append(Wait())
            types.append("wait")
            continue
        is_input, header, payload = frame
        name, brace, rest = header.partition("{")
        fields = json.loads(brace + rest) if brace else {}
        if name == "system":
            system = SystemPrompt.model_validate(json.loads(payload))
            continue
        types.append(name)
        if is_input:
            if name in INPUTS:
                items.append(INPUTS[name](payload=payload))
            elif name == "result":
                items.append(Result(payload=payload, **fields))
            elif name == "document":
                items.append(Document.model_validate({"payload": payload, **fields}))
            else:
                raise ValueError(f"unknown input type {name}")
        elif name in OUTPUTS:
            items.append(OUTPUTS[name](payload=payload))
        elif name == "call":
            items.append(Call(payload=payload, **fields))
        else:
            items.append(UnknownOutput(output_type=name, headers=fields, payload=payload))
    return Conversation(system=system, items=items), types


def audit(dataset, sample, encoding, checks):
    counts, thinking, fields, types, rules = (collections.Counter() for _ in range(5))
    forbidden = encoding.special_token_ids - {
        encoding.controls.input_start,
        encoding.controls.input_end,
        encoding.controls.output_start,
        encoding.controls.output_end,
        encoding.controls.header_end,
        encoding.controls.wait,
    }
    for index in sample:
        ids = dataset[index]
        counts["conversations"] += 1
        counts["other_special"] += bool(forbidden.intersection(ids[1:]))
        counts["bos_first"] += ids[0] == encoding.bos_token_id
        try:
            conversation, kinds = rebuild(ids, encoding)
        except (ValueError, KeyError, UnicodeDecodeError):
            counts["unparsed"] += 1
            continue
        counts["exact"] += encoding.encode_conversation(conversation) == ids
        counts["system"] += conversation.system is not None
        counts["ends_with_wait"] += kinds[-1:] == ["wait"]
        counts["several_user_turns"] += kinds.count("user") > 1
        types.update(set(kinds))
        if conversation.system is not None:
            thinking[str(conversation.system.thinking.value)] += 1
            for field in ("identity", "behavior", "environment", "tools"):
                fields[field] += bool(getattr(conversation.system, field))
        for error in conversation.check(checkers=checks).errors:
            rules[error.rule] += 1
    return {**counts, "thinking": thinking, "system_fields": fields, "types": types, "rules": rules}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizer")
    parser.add_argument("data", help="directory with <name>/16k/tokens.{bin,idx} datasets, at any depth")
    parser.add_argument("report")
    parser.add_argument("--sample", type=int, default=400, help="conversations per dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    encoding = load_encoding(args.tokenizer)
    checks = checkers_for("conversation")
    rng = random.Random(args.seed)
    report = {}
    for path in sorted(glob.glob(f"{args.data}/**/tokens.bin", recursive=True)):
        prefix = path[: -len(".bin")]
        dataset = Dataset(prefix)
        sample = rng.sample(range(len(dataset.lengths)), min(args.sample, len(dataset.lengths)))
        name = prefix[len(args.data) + 1 :].removesuffix("/16k/tokens")
        report[name] = audit(dataset, sample, encoding, checks)
        result = report[name]
        print(f"{name:<52} {result['exact']}/{result['conversations']} exact", flush=True)
    with open(args.report, "w") as sink:
        json.dump(report, sink, indent=1)


if __name__ == "__main__":
    main()

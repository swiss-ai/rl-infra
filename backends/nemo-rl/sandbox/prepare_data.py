#!/usr/bin/env python3
"""Turns one file of the coding dataset into the JSONL that Gym's code_gen agent reads.

A row is a problem and its unit tests, and a few tests carry inputs of hundreds of MB, so rows above
--max-row-bytes are left out. The rollouts some rows carry are dropped too. Prints what it left out, so
that a cap never goes unnoticed.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-row-bytes", type=int, required=True)
    parser.add_argument("--agent-name", required=True)
    args = parser.parse_args()

    kept = kept_bytes = dropped = dropped_bytes = 0
    largest_kept = 0
    temporary = args.output + ".tmp"
    with open(args.input, "rb") as source, open(temporary, "wb") as output:
        for line in source:
            if len(line) > args.max_row_bytes:
                dropped += 1
                dropped_bytes += len(line)
                continue
            row = json.loads(line)
            row.pop("response", None)
            row.pop("reward", None)
            row.setdefault("agent_ref", {"type": "responses_api_agents", "name": args.agent_name})
            output.write(json.dumps(row).encode() + b"\n")
            kept += 1
            kept_bytes += len(line)
            largest_kept = max(largest_kept, len(line))

    total = kept + dropped
    if not kept:
        raise ValueError(f"--max-row-bytes {args.max_row_bytes} leaves no row of {total} in {args.input}.")
    os.replace(temporary, args.output)
    print(
        f"{args.output}: kept {kept} of {total} rows, {kept_bytes / 1e9:.2f} GB, largest "
        f"{largest_kept / 1e6:.1f} MB; left out {dropped} rows, {dropped_bytes / 1e9:.2f} GB, "
        f"over {args.max_row_bytes} bytes"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print the import header (as the Kimina server splits it) that most rows of the given Gym files use, so the
server can import it into every REPL at start-up (LEAN_SERVER_INIT_REPLS). Rows with another header still
work, but each change of header costs a REPL a fresh import. The counts go to stderr.
    python3 dominant_header.py <kimina-lean-server dir> train.jsonl [validation.jsonl ...]
"""
import collections
import json
import sys

sys.path.insert(0, sys.argv[1])
from server.split import split_snippet  # noqa: E402

counts: collections.Counter = collections.Counter()
for path in sys.argv[2:]:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            counts[split_snippet(row["header"] + row["formal_statement"]).header] += 1
total = sum(counts.values())
for header, n in counts.most_common(5):
    print(f"{n:7d} ({100 * n / total:.1f}%) {header!r}", file=sys.stderr)
print(counts.most_common(1)[0][0])

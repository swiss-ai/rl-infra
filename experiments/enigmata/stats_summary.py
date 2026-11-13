"""Summary of the enigmata environment's per-sample stats file (env.enigmata.stats_path).

  python3 -I stats_summary.py <enigmata_stats.jsonl> [<out.json>]

Prints reward, graded frame (claim / reply / none), verifier failures and verify time per task and overall,
and the number of answers per (task, difficulty).
"""

import collections
import json
import sys


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    def summarize(rs):
        n = len(rs)
        statuses = collections.Counter(
            s for r in rs for s in (r["primary_status"], r["side_status"]) if s not in ("ok", "empty")
        )
        return {
            "answers": n,
            "reward": round(sum(r["reward"] for r in rs) / n, 4),
            "graded_claim": sum(r["source"] == "claim" for r in rs),
            "graded_reply": sum(r["source"] == "reply" for r in rs),
            "graded_none": sum(r["source"] == "none" for r in rs),
            "truncated": sum(bool(r.get("truncated")) for r in rs),
            "parse_error": sum(bool(r.get("parse_error")) for r in rs),
            "failures": dict(statuses),
            "verify_ms_mean": round(sum(r["verify_ms"] for r in rs) / n, 2),
            "verify_ms_max": round(max(r["verify_ms"] for r in rs), 1),
        }

    out = {"overall": summarize(rows), "tasks": {t: summarize(rs) for t, rs in sorted(by_task.items())}}
    cells = collections.Counter((r["task"], r["difficulty"]) for r in rows)
    out["cells"] = {f"{t}/{d}": n for (t, d), n in sorted(cells.items())}
    text = json.dumps(out, indent=1)
    print(text)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

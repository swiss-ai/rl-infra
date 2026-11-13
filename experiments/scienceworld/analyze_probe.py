"""Summarize sciworld_score.py output (episodes.jsonl): per format and task, plus timing. Python 3.6+, stdlib only.

  python3 analyze_probe.py <episodes.jsonl> [--transcripts N]
"""
import argparse
import collections
import json
import statistics


def mean(v):
    return statistics.mean(v) if v else float("nan")


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100.0 * len(v)))] if v else float("nan")


def lstsq(rows, cols, target):
    """Least squares without numpy: normal equations, Gauss-Jordan."""
    n = len(cols)
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * t for r, t in zip(rows, target)) for i in range(n)]
    for i in range(n):
        p = a[i][i] or 1e-12
        for j in range(n):
            if j != i:
                f = a[j][i] / p
                a[j] = [x - f * y for x, y in zip(a[j], a[i])]
                b[j] -= f * b[i]
    return {c: b[i] / (a[i][i] or 1e-12) for i, c in enumerate(cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--transcripts", type=int, default=0)
    ap.add_argument("--format", default=None)
    ap.add_argument("--task", default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.path)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["format"], "ALL")].append(r)
        by[(r["format"], r["task"])].append(r)
    print("format task n reward success zero invalid_rate actions turns gen_tok think/turn gen/turn total_tok "
          "cut_off ends wall_mean wall_p95 wall_max")
    for (fmt, task), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "ALL", kv[0][1])):
        env = [r["env"] or {} for r in rs]
        ends = collections.Counter(r["end"] for r in rs)
        print("%s %-44s %3d %.3f %.2f %.2f %.3f %5.1f %5.1f %6.0f %5.0f %5.0f %6.0f %.2f %s %4.0f %4.0f %4.0f" % (
            fmt, task, len(rs), mean([r["reward"] for r in rs]), mean([e.get("sw_success", 0) for e in env]),
            mean([float(r["reward"] <= 0) for r in rs]), mean([e.get("sw_invalid_rate", 0) for e in env]),
            mean([e.get("sw_actions", 0) for e in env]), mean([r["turns"] for r in rs]),
            mean([r["gen_tokens"] for r in rs]), mean([t for r in rs for t in r["think_tokens"]]),
            mean([t for r in rs for t in r["turn_gen_tokens"]]), mean([r["total_tokens"] for r in rs]),
            mean([float(r["end"] == "cut_off") for r in rs]),
            ",".join("%s:%d" % kv for kv in ends.most_common()),
            mean([r["wall_s"] for r in rs]), pct([r["wall_s"] for r in rs], 95), max(r["wall_s"] for r in rs)))
    # Prompt groups (same task, variation, format): how many have reward variance.
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["format"], r["task"], r["variation"])].append(r["reward"])
    for fmt in sorted({k[0] for k in groups}):
        g = [v for k, v in groups.items() if k[0] == fmt]
        print(fmt, "groups", len(g), "all equal", sum(len(set(v)) == 1 for v in g),
              "all zero", sum(max(v) <= 0 for v in g), "all one", sum(min(v) >= 1 for v in g))
    # Timing model: wall = a * generated tokens + b * turns + c * prefill tokens (sum of turn inputs) + d.
    for fmt in sorted({r["format"] for r in rows}):
        rs = [r for r in rows if r["format"] == fmt]
        x = [[r["gen_tokens"], r["turns"], sum(r["turn_input_tokens"]) / 1000.0, 1.0] for r in rs]
        fit = lstsq(x, ["s_per_gen_token", "s_per_turn", "s_per_1k_prefill", "const"], [r["wall_s"] for r in rs])
        print(fmt, "timing fit", {k: round(v, 4) for k, v in fit.items()},
              "mean prefill tokens", round(mean([sum(r["turn_input_tokens"]) for r in rs])),
              "batch_s", rs[0]["batch_s"])
    shown = 0
    for r in rows:
        if shown >= args.transcripts:
            break
        if (args.format and r["format"] != args.format) or (args.task and r["task"] != args.task):
            continue
        shown += 1
        print("=" * 100)
        print(r["task"], r["variation"], r["format"], "reward", r["reward"], "end", r["end"], "turns", r["turns"])
        for m in r["transcript"][1:]:
            c = m["content"]
            print("[%s %d] %s" % (m["role"], m["tokens"], c if len(c) < 1500 else c[:700] + " ... " + c[-700:]))


if __name__ == "__main__":
    main()

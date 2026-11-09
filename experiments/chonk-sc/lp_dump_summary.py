#!/usr/bin/env python3
"""Summarise the logprob outlier dump of NeMo-RL patch 0034 (run-s-lpdump.yaml).

  python3 lp_dump_summary.py <log_dir>/logprob_dump [--tokenizer <model>/tokenizer.json] [--top 40]

Reads logprob-*.jsonl, splits the loss calls into training steps by time gaps, joins token records
across ranks by (seq_key, pos), and prints per step: tokens over 1/2/5/10 nats, the worst sequences,
where the outlier tokens sit (offset in their generated burst, last token of a burst, generation
logprob ~0 = forced/near-certain for the inference engine, trainer prev vs current), the most frequent
outlier token ids, and the worst tokens with their context. Stdlib only; with the tokenizers package
and --tokenizer, token ids are shown as text.
"""

import argparse
import collections
import glob
import json
import os
import sys

STEP_GAP_S = 60.0


def load(directory):
    calls, seqs, toks = [], [], []
    for path in sorted(glob.glob(os.path.join(directory, "logprob-*.jsonl"))):
        source = os.path.basename(path)
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # a line cut by the job's end
                record["_src"] = source
                {"call": calls, "seq": seqs, "tok": toks}.get(record.get("kind"), []).append(record)
    return calls, seqs, toks


def assign_steps(calls):
    """Map (source, call) -> step index by gaps between loss calls of all ranks."""
    step_of, step, last = {}, 0, None
    for call in sorted(calls, key=lambda c: c["time"]):
        if last is not None and call["time"] - last > STEP_GAP_S:
            step += 1
        last = call["time"]
        step_of[(call["_src"], call["call"])] = step
    return step_of


def tokenizer_from(path):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("(tokenizers not installed: showing token ids)", file=sys.stderr)
        return None
    return Tokenizer.from_file(path)


def text(tok, ids):
    if tok is None:
        return " ".join(str(i) for i in ids)
    return tok.decode(list(ids), skip_special_tokens=False).replace("\n", "\\n")


def piece(tok, token_id):
    if tok is None:
        return str(token_id)
    return f"{tok.id_to_token(token_id)!r}({token_id})"


def bucket(offset, length):
    if offset == length - 1:
        return "last"
    if offset == 0:
        return "0"
    if offset < 10:
        return "1-9"
    if offset < 100:
        return "10-99"
    return "100+"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    parser.add_argument("--tokenizer", help="tokenizer.json of the model")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    calls, seqs, toks = load(args.directory)
    if not calls:
        sys.exit(f"no logprob-*.jsonl records under {args.directory}")
    tok = tokenizer_from(args.tokenizer)
    step_of = assign_steps(calls)

    def step(record):
        return step_of.get((record["_src"], record["call"]), -1)

    # the same sequence can be scored on several ranks (context parallel) or passes: keep the worst
    unique = {}
    for record in toks:
        key = (step(record), record.get("seq_key"), record["pos"])
        if key not in unique or record["gap"] > unique[key]["gap"]:
            unique[key] = record
    toks = list(unique.values())
    unique_seqs = {}
    for record in seqs:
        key = (step(record), record.get("seq_key"))
        if key not in unique_seqs or record["seq_mult_prob_error"] > unique_seqs[key]["seq_mult_prob_error"]:
            unique_seqs[key] = record
    seqs = list(unique_seqs.values())

    steps = sorted(set(step_of.values()))
    print(f"{len(calls)} loss calls from {len({c['_src'] for c in calls})} files, {len(steps)} steps "
          f"(split at gaps > {STEP_GAP_S:.0f} s); {len(seqs)} sequences and {len(toks)} tokens over the threshold")
    for s in steps:
        cs = [c for c in calls if step_of[(c["_src"], c["call"])] == s]
        ss = sorted((r for r in seqs if step(r) == s), key=lambda r: -r["seq_mult_prob_error"])
        ts = sorted((r for r in toks if step(r) == s), key=lambda r: -r["gap"])
        valid = sum(c["valid_tokens"] for c in cs)
        over = collections.Counter()
        for c in cs:
            over.update(c["over"])
        errors = [e for c in cs for e in c["seq_mult_prob_error"]]
        print(f"\n=== step {s + 1}: {len(cs)} loss calls, {valid} valid tokens, {len(errors)} rows")
        print("  tokens with |gen - prev| over " + ", ".join(f"{k} nats: {over[k]}" for k in ("1", "2", "5", "10")))
        if errors:
            errors.sort()
            print(f"  seq_mult_prob_error: median {errors[len(errors) // 2]:.3f}, "
                  f"max {errors[-1]:.3g}, rows > 2: {sum(e > 2 for e in errors)}")
        if not ts:
            continue
        print(f"  worst sequences ({len(ss)} with a token over the threshold):")
        for r in ss[:5]:
            print(f"    seq {r.get('seq_key')} err {r['seq_mult_prob_error']:.3g} max_gap {r['max_gap']:.1f} "
                  f"valid {r['valid_tokens']} bursts {r['runs']} last_pos {r['last_pos']} over {r['over']}")
        n = len(ts)
        higher = sum(r["gen"] > r["prev"] for r in ts)
        forced = sum(r["gen"] > -1e-3 for r in ts)
        drift = sum(abs(r["curr"] - r["prev"]) > 0.5 for r in ts)
        print(f"  {n} outlier tokens: gen > prev {higher}, gen ~0 (engine near-certain) {forced}, "
              f"|curr - prev| > 0.5 {drift}")
        where = collections.Counter(bucket(r["run_offset"], r["run_len"]) for r in ts)
        print("  offset in burst: " + ", ".join(f"{k} {where[k]}" for k in ("0", "1-9", "10-99", "100+", "last")))
        burst = collections.Counter(min(r["run"], 9) for r in ts)
        print("  burst index: " + ", ".join(f"{k if k < 9 else '9+'}: {v}" for k, v in sorted(burst.items())))
        ids = collections.Counter(r.get("token") for r in ts)
        print("  most frequent tokens: " + ", ".join(
            f"{piece(tok, t) if t is not None else '?'} x{c}" for t, c in ids.most_common(12)))
        print(f"  worst {min(args.top, n)} tokens (gap gen prev curr | pos burst offset/len | context):")
        for r in ts[: args.top]:
            token = r.get("token")
            context = ""
            if token is not None:
                context = f"{text(tok, r['before'])} [[{text(tok, [token])}]] {text(tok, r['after'])}"
            print(f"    {r['gap']:6.2f} {r['gen']:8.3f} {r['prev']:8.3f} {r['curr']:8.3f} | "
                  f"{r['pos']} b{r['run']} {r['run_offset']}/{r['run_len']} | {context}")


if __name__ == "__main__":
    main()

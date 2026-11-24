#!/usr/bin/env python3
"""How much would model-free speculative decoding speed up Chonk's Lean rollouts? Replays real rollouts offline.

At temperature 1 with exact speculative sampling, a deterministic draft token d is accepted with probability p(d), the
target model's probability of d. A real rollout's next token was sampled from p, so P(real next token == d) = p(d): the
match rate of drafts against real rollouts is an unbiased estimate of the acceptance rate (no model needed).

For each rollout, from position 0: draft up to K tokens by n-gram lookup (the longest of N_MAX..N_MIN trailing tokens,
most recent occurrence), count how many leading draft tokens equal the real continuation (a), advance a + 1 (the
verification pass also yields one token), count one forward pass. Draft sources, cumulative:
  self      the prompt and the rollout's own tokens so far (prompt-lookup / n-gram decoding)
  group     + the other rollouts of the same prompt, up to the same position (they are generated alongside)
  history   + the other rollouts in full (as a previous epoch's rollouts of the same prompt would be)
Reports tokens per forward pass, overall and for rollouts cut at the length limit (loops).

  python spec_sim.py SAMPLES.jsonl [--prompts DATA.jsonl --tokenizer DIR --system-prompt-file F --sampler-dir D]
"""
import argparse
import bisect
import json
import sys
from collections import defaultdict


def build_index(seq, n_values):
    """n -> {ngram: sorted positions where the ngram ends (exclusive end index)}"""
    idx = {n: defaultdict(list) for n in n_values}
    for n in n_values:
        d = idx[n]
        for end in range(n, len(seq) + 1):
            d[tuple(seq[end - n:end])].append(end)
    return idx


def draft_from(seq, index, key_n, key, limit_end, k):
    """Latest occurrence of key (ending at < limit_end, with continuation inside seq[:limit_end]) -> up to k tokens."""
    ends = index[key_n].get(key)
    if not ends:
        return None
    j = bisect.bisect_left(ends, limit_end) - 1     # ends strictly before limit_end (exclusive of the current spot)
    while j >= 0:
        e = ends[j]
        if e < limit_end:
            return seq[e:min(e + k, limit_end)] or None
        j -= 1
    return None


def simulate(gen, prompt, siblings, mode, k, n_values, own_index=None, sib_index=None):
    ctx = list(prompt) + list(gen)          # own context: prompt then generated tokens
    own_index = own_index or build_index(ctx, n_values)
    if mode == "self":
        siblings, sib_index = [], []
    elif sib_index is None:
        sib_index = [build_index(s, n_values) for s in siblings]
    p0 = len(prompt)
    i = 0
    passes = 0
    accepted = 0
    while i < len(gen):
        pos = p0 + i                           # tokens known: ctx[:pos]
        draft = None
        for n in n_values:                     # longest match first
            if pos < n:
                continue
            key = tuple(ctx[pos - n:pos])
            draft = draft_from(ctx, own_index, n, key, pos, k)
            if draft is None:
                for s, si in zip(siblings, sib_index):
                    limit = len(s) if mode == "history" else min(len(s), i)
                    draft = draft_from(s, si, n, key, limit, k)
                    if draft:
                        break
            if draft:
                break
        a = 0
        if draft:
            while a < len(draft) and i + a < len(gen) and draft[a] == gen[i + a]:
                a += 1
        accepted += a
        passes += 1
        i += a + 1
    return passes, accepted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples")
    ap.add_argument("--k", type=int, nargs="*", default=[3, 5, 8])
    ap.add_argument("--n", type=int, nargs="*", default=[6, 4, 3], help="n-gram lengths, longest first")
    ap.add_argument("--max-rows", type=int, default=100)
    ap.add_argument("--prompts")
    ap.add_argument("--tokenizer")
    ap.add_argument("--system-prompt-file")
    ap.add_argument("--sampler-dir")
    args = ap.parse_args(argv)
    groups = defaultdict(list)
    for line in open(args.samples):
        r = json.loads(line)
        if r.get("token_ids"):
            groups[r["row"]].append(r)
    rows = sorted(groups)[: args.max_rows]
    prompts = {}
    if args.prompts:
        sys.path.insert(0, args.sampler_dir)
        import sampler  # noqa: E402

        fmt = sampler.Apertus2Format(args.tokenizer, args.system_prompt_file)
        data = sampler.load_rows(args.prompts)
        for r in rows:
            prompts[r] = fmt.encode(data[r]["responses_create_params"]["input"])
    own_idx, gen_idx = {}, {}
    for r in rows:
        for x in groups[r]:
            own_idx[id(x)] = build_index(list(prompts.get(r, [])) + list(x["token_ids"]), args.n)
            gen_idx[id(x)] = build_index(x["token_ids"], args.n)
    results = []
    for mode in ("self", "group", "history"):
        for k in args.k:
            tot = {"all": [0, 0, 0], "cut": [0, 0, 0], "done": [0, 0, 0]}   # tokens, passes, accepted
            for r in rows:
                rs = groups[r]
                for x in rs:
                    others = [y for y in rs if y is not x]
                    passes, acc = simulate(x["token_ids"], prompts.get(r, []), [y["token_ids"] for y in others], mode, k,
                                           args.n, own_idx[id(x)], [gen_idx[id(y)] for y in others])
                    for key in ("all", "cut" if x.get("finish") == "length" else "done"):
                        t = tot[key]
                        t[0] += len(x["token_ids"]); t[1] += passes; t[2] += acc
            res = {"mode": mode, "k": k}
            for key, (tokens, passes, acc) in tot.items():
                res[key] = {"tokens": tokens, "tokens_per_pass": tokens / passes if passes else None}
            results.append(res)
            print(json.dumps(res), flush=True)
    print(f"rows {len(rows)}, rollouts {sum(len(groups[r]) for r in rows)}, prompts {'yes' if prompts else 'no'}")
    return results


if __name__ == "__main__":
    main()

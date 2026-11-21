#!/usr/bin/env python3
"""Build a class- and length-balanced Safety Gym training view with over-refusal validation.

Why: the first run trained on 87.5% harmful rows, all of them long adversarially framed
prompts (5th percentile about 290 characters), and over-refused short direct benign
requests at evaluation time. This builder

* fixes the benign share (default 25%) inside EVERY prompt-length bin, so prompt length
  carries no information about the class;
* adds short direct prompts (WildGuardMix train, vanilla) for both classes and keeps a
  minimum share of short rows (default 30% below 400 characters);
* tags benign rows with a response policy, so the over-refusal judge rubric is selected
  by the data (default: refuse_then_redirect);
* writes a validation set with separate Gym agent names per subset. NeMo-RL logs
  validation/<agent>/reward per agent, so over-refusal = 1 - reward of the or_val_* agents.

Selection is deterministic (hash priority). Prompts whose normalised text matches an
evaluation prompt (--exclude-hashes) are dropped. Nothing in the source is modified.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

FORMAT = "safety-gym-balanced-v1"
TRAIN_ROUTES = {"harmful": "jailbreak_detection_simple_agent", "benign": "over_refusal_detection_simple_agent"}
VAL_ROUTES = {
    ("harmful", "adversarial"): "jailbreak_detection_simple_agent",
    ("harmful", "vanilla"): "jailbreak_val_wg_test_vanilla",
    ("benign", "adversarial"): "or_val_wg_test_adversarial",
    ("benign", "vanilla"): "or_val_wg_test_vanilla",
}
EDGES = (100, 200, 400, 800, 1600)


def bin_of(length):
    for index, edge in enumerate(EDGES):
        if length < edge:
            return index
    return len(EDGES)


def bin_name(index):
    low = 0 if index == 0 else EDGES[index - 1]
    return f"{low}-{EDGES[index] - 1}" if index < len(EDGES) else f"{EDGES[-1]}+"


def norm_hash(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()


def priority(salt, key):
    return hashlib.sha256(salt + b"\0" + key.encode()).digest()


def index_main(path, excluded, salt):
    """One streaming pass; keep only (priority, offset, class, bin, language) per row."""
    rows, dropped = [], 0
    with open(path, "rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            row = json.loads(raw)
            text = row["responses_create_params"]["input"][-1]["content"]
            if norm_hash(text) in excluded:
                dropped += 1
                continue
            rows.append((priority(salt, row["id"]), offset, row["class_type"], bin_of(len(text)), row["language"]))
    return rows, dropped


def load_vanilla(path, split, excluded, salt):
    rows, dropped, seen = [], 0, set()
    for raw in open(path, encoding="utf-8"):
        row = json.loads(raw)
        if row["split"] != split:
            continue
        digest = norm_hash(row["text"])
        if digest in excluded or digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        rows.append((priority(salt, row["id"]), row, row["class_type"], bin_of(len(row["text"])), row["language"]))
    return rows, dropped


def gym_row(row, agent, policy):
    """Vanilla rows get the same request shape as the msafetemp export."""
    out = dict(row)
    out.setdefault("responses_create_params", {"input": [{"role": "user", "content": row["text"]}]})
    out["agent_ref"] = {"type": "responses_api_agents", "name": agent}
    if policy and row["class_type"] == "benign":
        out["response_policy_mapped"] = policy
    return out


def read_at(handle, offset):
    handle.seek(offset)
    return json.loads(handle.readline())


def language_balanced(items, count):
    """Round-robin over languages in priority order."""
    by_lang = defaultdict(list)
    for item in sorted(items, key=lambda i: i[0]):
        by_lang[item[4]].append(item)
    chosen, langs = [], sorted(by_lang)
    while len(chosen) < count and any(by_lang[l] for l in langs):
        for lang in langs:
            if by_lang[lang] and len(chosen) < count:
                chosen.append(by_lang[lang].pop(0))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="msafetemp-safetygym export (train.jsonl, validation.jsonl)")
    p.add_argument("--vanilla", type=Path, required=True, help="WildGuardMix vanilla prompts JSONL (train and test splits)")
    p.add_argument("--exclude-hashes", type=Path, help="sha1 of normalised evaluation prompts, one per line")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benign-share", type=float, default=0.25)
    p.add_argument("--short-chars", type=int, default=400, choices=EDGES)
    p.add_argument("--short-fraction", type=float, default=0.30, help="minimum share of rows shorter than --short-chars")
    p.add_argument("--benign-policy", default="refuse_then_redirect", help="response_policy_mapped for benign rows; empty keeps the binary rubric")
    p.add_argument("--val-harmful", type=int, nargs=2, default=(200, 100), metavar=("ADVERSARIAL", "VANILLA"))
    p.add_argument("--val-benign", type=int, nargs=2, default=(250, 250), metavar=("ADVERSARIAL", "VANILLA"))
    p.add_argument("--plan", action="store_true", help="print the plan, write nothing")
    a = p.parse_args()
    if not 0 < a.benign_share < 1:
        p.error("--benign-share must be in (0, 1)")
    if a.output.exists() and not a.plan:
        p.error("output exists; choose another directory")
    excluded = set(a.exclude_hashes.read_text().split()) if a.exclude_hashes else set()
    salt = b"safety-gym-balanced-v1"

    main_rows, dropped_main = index_main(a.source / "train.jsonl", excluded, salt)
    van_rows, dropped_van = load_vanilla(a.vanilla, "train", excluded, salt)
    pool = defaultdict(list)  # (class, bin) -> items
    for item in main_rows + van_rows:
        pool[item[2], item[3]].append(item)
    for items in pool.values():
        items.sort(key=lambda i: i[0])

    ratio = (1 - a.benign_share) / a.benign_share  # harmful rows per benign row
    nbins = len(EDGES) + 1
    quota = {b: min(len(pool["benign", b]), int(len(pool["harmful", b]) / ratio)) for b in range(nbins)}
    short_bins = [b for b in range(nbins) if b < EDGES.index(a.short_chars) + 1]
    short = sum(quota[b] for b in short_bins)
    long_total = sum(quota[b] for b in range(nbins) if b not in short_bins)
    long_cap = int(short * (1 - a.short_fraction) / a.short_fraction) if a.short_fraction > 0 else long_total
    scale = min(1.0, long_cap / long_total) if long_total else 1.0
    for b in range(nbins):
        if b not in short_bins:
            quota[b] = int(quota[b] * scale)
    plan = {
        "format": FORMAT,
        "benign_share": a.benign_share,
        "short_chars": a.short_chars,
        "short_fraction_min": a.short_fraction,
        "benign_policy": a.benign_policy,
        "available": {bin_name(b): {c: len(pool[c, b]) for c in ("benign", "harmful")} for b in range(nbins)},
        "selected": {bin_name(b): {"benign": quota[b], "harmful": int(round(quota[b] * ratio))} for b in range(nbins)},
        "dropped_as_eval_overlap_or_duplicate": {"msafetemp_train": dropped_main, "vanilla_train": dropped_van},
    }
    total_b = sum(v["benign"] for v in plan["selected"].values())
    total_h = sum(v["harmful"] for v in plan["selected"].values())
    plan["train_rows"] = total_b + total_h
    plan["train_benign_share"] = round(total_b / (total_b + total_h), 4)
    plan["train_short_share"] = round(sum(sum(plan["selected"][bin_name(b)].values()) for b in short_bins) / (total_b + total_h), 4)
    plan["steps_per_epoch_at_128_prompts"] = (total_b + total_h) // 128

    # validation: msafetemp test split (adversarial, multilingual) + WildGuardTest vanilla (short, English)
    val_main, _ = index_main(a.source / "validation.jsonl", set(), salt)
    val_van, _ = load_vanilla(a.vanilla, "test", set(), salt)
    val_sel = {}
    for cls, counts in (("harmful", a.val_harmful), ("benign", a.val_benign)):
        val_sel[cls, "adversarial"] = language_balanced([i for i in val_main if i[2] == cls], counts[0])
        val_sel[cls, "vanilla"] = sorted([i for i in val_van if i[2] == cls], key=lambda i: i[0])[: counts[1]]
    plan["validation_rows"] = {VAL_ROUTES[k]: len(v) for k, v in val_sel.items()}
    plan["validation_note"] = (
        "validation uses the WildGuard test split (same source as the WG benchmark); it is a monitor, "
        "not an independent test. Over-refusal = 1 - validation/or_val_*/reward."
    )
    if a.plan:
        print(json.dumps(plan, indent=2))
        return

    a.output.mkdir(parents=True)
    lang_counts, digest = Counter(), hashlib.sha256()
    selected = []
    for b in range(nbins):
        selected += pool["benign", b][: quota[b]] + pool["harmful", b][: int(round(quota[b] * ratio))]
    selected.sort(key=lambda i: i[0])  # deterministic shuffle
    with open(a.source / "train.jsonl", "rb") as src, open(a.output / "train.jsonl", "wb") as out:
        for item in selected:
            row = read_at(src, item[1]) if isinstance(item[1], int) else item[1]
            line = (json.dumps(gym_row(row, TRAIN_ROUTES[item[2]], a.benign_policy), ensure_ascii=False) + "\n").encode()
            out.write(line); digest.update(line)
            lang_counts[item[2], item[4]] += 1
    plan["train_sha256"] = digest.hexdigest()
    plan["train_language_counts"] = {c: {l: n for (cc, l), n in sorted(lang_counts.items()) if cc == c} for c in ("benign", "harmful")}
    digest = hashlib.sha256()
    with open(a.source / "validation.jsonl", "rb") as src, open(a.output / "validation.jsonl", "wb") as out:
        for key, items in val_sel.items():
            for item in items:
                row = read_at(src, item[1]) if isinstance(item[1], int) else item[1]
                line = (json.dumps(gym_row(row, VAL_ROUTES[key], a.benign_policy), ensure_ascii=False) + "\n").encode()
                out.write(line); digest.update(line)
    plan["validation_sha256"] = digest.hexdigest()
    plan["source"] = str(a.source.absolute()); plan["vanilla"] = str(a.vanilla.absolute())
    (a.output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

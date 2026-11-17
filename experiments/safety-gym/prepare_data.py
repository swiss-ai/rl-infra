#!/usr/bin/env python3
"""Link all training rows and select 500/1000 held-out multilingual examples.

Only --plan is allowed outside Slurm. No training data is copied or rewritten.
"""

import argparse
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile


FORMAT = "safety-gym-validation-v1"
ROUTES = {
    "harmful": "jailbreak_detection_simple_agent",
    "benign": "over_refusal_detection_simple_agent",
}


def inspect(source, count):
    source = Path(source).absolute()
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    languages = manifest["selected_languages"]
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("source manifest must list unique selected_languages")
    files = {}
    for name in ("train.jsonl", "validation.jsonl"):
        stat = (source / name).stat()
        files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "format": FORMAT,
        "source": str(source),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_files": files,
        "source_rows_declared": manifest["source_row_counts"],
        "languages": languages,
        "validation_samples": count,
    }


def prepare(source, output, count, *, plan=False):
    if count not in (500, 1000):
        raise ValueError("validation_samples must be 500 or 1000")
    identity = inspect(source, count)
    if plan:
        return identity
    source = Path(identity["source"])
    output = Path(output).absolute()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the source directory")
    if output.exists() or output.is_symlink():
        previous = json.loads((output / "manifest.json").read_text())
        for key, value in identity.items():
            if previous.get(key) != value:
                raise ValueError(f"existing output differs: {key}; choose another output")
        train = output / "train.jsonl"
        if not train.is_symlink() or os.readlink(train) != str(source / "train.jsonl"):
            raise ValueError("existing training symlink differs")
        validation = (output / "validation.jsonl").read_bytes()
        if hashlib.sha256(validation).hexdigest() != previous["validation_sha256"]:
            raise ValueError("existing validation checksum differs")
        if len(validation.splitlines()) != count:
            raise ValueError("existing validation row count differs")
        return previous

    languages = identity["languages"]
    base, remainder = divmod(count, len(languages))
    quotas = {lang: base + (index < remainder) for index, lang in enumerate(languages)}
    heaps = {(lang, route): [] for lang in languages for route in ROUTES}
    digest = hashlib.sha256()
    scanned = 0
    with (source / "validation.jsonl").open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            language, kind = row["language"], row["class_type"]
            if row["split"] != "test" or language not in quotas or kind not in ROUTES:
                raise ValueError(f"invalid held-out split/language/class at row {number}")
            if row["agent_ref"]["name"] != ROUTES[kind]:
                raise ValueError(f"wrong Gym routing at row {number}")
            priority = int.from_bytes(hashlib.sha256(b"safety-gym-v1\0" + raw).digest(), "big")
            item = (-priority, -number, raw)
            heap = heaps[language, kind]
            if len(heap) < quotas[language]:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            scanned += 1
    source_manifest = json.loads((source / "manifest.json").read_text())
    if scanned != identity["source_rows_declared"]["test"]:
        raise ValueError("held-out row count differs from manifest")
    if digest.hexdigest() != source_manifest["output_sha256"]["validation.jsonl"]:
        raise ValueError("held-out source checksum differs from manifest")

    chosen = []
    for language, quota in quotas.items():
        groups = [sorted(heaps[language, kind], reverse=True) for kind in ROUTES]
        if any(not group for group in groups) or quota < 2:
            raise ValueError(f"both safety routes required for language {language}")
        selected = [group[0] for group in groups]
        pool = sorted([item for group in groups for item in group[1:]], reverse=True)
        selected.extend(pool[:quota - 2])
        if len(selected) != quota:
            raise ValueError(f"not enough held-out rows for language {language}")
        chosen.extend(selected)
    chosen.sort(key=lambda item: -item[1])
    content = b"".join(item[2].rstrip(b"\n") + b"\n" for item in chosen)
    identity.update({
        "validation_sha256": hashlib.sha256(content).hexdigest(),
        "validation_rows": len(chosen),
        "validation_language_counts": quotas,
        "validation_route_counts": dict(Counter(json.loads(item[2])["agent_ref"]["name"] for item in chosen)),
        "source_validation_rows_scanned": scanned,
        "training": "absolute symlink; all source training rows preserved",
        "selection": "deterministic hash sample balanced across languages, at least one example per native route per language; source test split only",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    # A job-level advisory lock avoids concurrent preparation of the shared view.
    import fcntl
    with (output.parent / (output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            return prepare(source, output, count)
        with tempfile.TemporaryDirectory(prefix=".validation-stage-", dir=output.parent) as stage_name:
            stage = Path(stage_name)
            (stage / "train.jsonl").symlink_to(source / "train.jsonl")
            (stage / "validation.jsonl").write_bytes(content)
            (stage / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
            os.rename(stage, output)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-samples", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.source, args.output, args.validation_samples, plan=args.plan), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

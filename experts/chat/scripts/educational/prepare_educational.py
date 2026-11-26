#!/usr/bin/env python3
"""Prepare English multi-turn educational learner simulation specs (v1: User Sim general_educational).

Pairs topics (math problems from ``dmacjam/eduardoRL-transfer-dataset`` + open subjects from
NVIDIA User Sim's ``general_educational``) with personas from ``nvidia/Nemotron-Personas-USA``
(or a local JSONL/Parquet file), computes OCEAN behavioral profiles and learner system prompts
(English-only in v1), and optionally decontaminates ``problem`` and ``transfer_problem`` against
``sft-data-apertus2``'s gathered evaluation benchmark prompt sets (``general`` and ``stem``).

Outputs ``{train,val}_specs.jsonl`` to ``out_dir``. Run ``make_openers.py`` next to generate
the gated opening user turn and write the final ``ResponseDataset`` JSONL files.

Example (login node):
    python3.12 experts/chat/scripts/educational/prepare_educational.py \
        ${RL_DATA_DIR}/chat-rl/educational \
        --sft-data-dir ../sft-data-apertus2 \
        --benchmarks-root /path/to/benchmarks_apertus2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from behavioral import (
    DEFAULT_LANGUAGE,
    DEFAULT_LOCALE,
    compute_behavioral_profile,
    compute_disclosure_style,
    compute_user_interaction_style,
    ensure_ocean_traits,
    format_behavioral_profile_for_prompt,
    format_disclosure_instructions,
    format_interaction_style_instructions,
    format_persona_for_prompt,
    language_instruction,
    persona_display_name,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"
SUBJECTS_PATH = SCRIPT_DIR / "subjects_en_US.json"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_table(path_or_repo: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path_or_repo)
    if p.exists():
        if p.suffix == ".jsonl":
            rows = _load_jsonl(p)
        elif p.suffix == ".json":
            rows = json.loads(p.read_text(encoding="utf-8"))
        elif p.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows = pq.read_table(p).to_pylist()
        else:
            raise ValueError(f"Unsupported local dataset format: {p}")
        return rows[:limit] if limit is not None else rows

    from datasets import load_dataset

    ds = load_dataset(path_or_repo, split=split, streaming=limit is not None)
    if limit is not None:
        return [dict(row) for _, row in zip(range(limit), ds)]
    return [dict(row) for row in ds]


def _maybe_decontaminate(
    problems: list[dict[str, Any]],
    *,
    sft_data_dir: Path | None,
    benchmarks_root: Path | None,
    benchmark_names: list[str],
    skip_decon: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Drop problems whose ``problem`` or ``transfer_problem`` matches Apertus 2 eval benchmarks."""
    if skip_decon:
        print("WARNING: --skip-decon set; skipping benchmark decontamination.", file=sys.stderr)
        return problems, 0
    if sft_data_dir is None or benchmarks_root is None:
        raise SystemExit(
            "ERROR: Pass --sft-data-dir and --benchmarks-root to decontaminate against "
            "benchmarks/{general,stem}.yaml (or pass --skip-decon only for local smoke tests)."
        )

    src = sft_data_dir / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from sft_data.decon import Index, load_prompt_sets  # type: ignore[import-not-found]

    prompt_sets = []
    for name in benchmark_names:
        candidates = sorted(benchmarks_root.glob(f"{name}_v*"))
        if not candidates:
            raise SystemExit(f"ERROR: No gathered prompt set matching {name}_v* in {benchmarks_root}")
        prompt_sets.append(candidates[-1])

    index = Index.from_prompt_sets(load_prompt_sets(prompt_sets))
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in problems:
        texts = [str(row.get("problem") or "")]
        if row.get("transfer_problem"):
            texts.append(str(row["transfer_problem"]))
        if any(index.match(t) is not None for t in texts if t.strip()):
            dropped += 1
        else:
            kept.append(row)
    print(f"Decontamination ({', '.join(p.name for p in prompt_sets)}): kept {len(kept)}, dropped {dropped}")
    return kept, dropped


def _row_seed(key: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_spec_row(
    *,
    kind: str,
    persona_raw: dict[str, Any],
    topic: dict[str, Any],
    prompt_problem: str,
    prompt_subject: str,
    incremental_ratio: float,
    seed: int,
    row_id: str,
) -> dict[str, Any]:
    persona = ensure_ocean_traits(persona_raw, seed=seed)
    profile = compute_behavioral_profile(persona)
    interaction_style = compute_user_interaction_style(profile)
    disclosure_style = compute_disclosure_style(
        persona_seed=_row_seed(row_id, seed),
        incremental_ratio=incremental_ratio,
    )
    persona_text = format_persona_for_prompt(persona, seed=seed)
    persona_name = persona_display_name(persona)

    fields = dict(
        language_instruction=language_instruction(),
        persona=persona_text,
        behavioral_instructions=format_behavioral_profile_for_prompt(profile),
        disclosure_instructions=format_disclosure_instructions(disclosure_style),
        interaction_style_instructions=format_interaction_style_instructions(interaction_style),
    )
    if kind == "problem":
        problem_text = str(topic["problem"]).strip()
        learner_system = prompt_problem.format(**fields, problem=problem_text)
        subject_text = None
    elif kind == "subject":
        subject_text = str(topic.get("description") or topic.get("type") or "").strip()
        learner_system = prompt_subject.format(**fields, subject=subject_text)
        problem_text = None
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    return {
        "row_id": row_id,
        "kind": kind,
        "locale": DEFAULT_LOCALE,
        "language": DEFAULT_LANGUAGE,
        "problem": problem_text,
        "answer": str(topic["answer"]).strip() if kind == "problem" and topic.get("answer") is not None else None,
        "transfer_problem": (
            str(topic["transfer_problem"]).strip()
            if kind == "problem" and topic.get("transfer_problem") is not None
            else None
        ),
        "transfer_answer": (
            str(topic["transfer_answer"]).strip()
            if kind == "problem" and topic.get("transfer_answer") is not None
            else None
        ),
        "pre_solve_rate": (
            float(topic["pre_solve_rate"])
            if kind == "problem" and topic.get("pre_solve_rate") is not None
            else None
        ),
        "pre_solve_rate_transfer_problem": (
            float(topic["pre_solve_rate_transfer_problem"])
            if kind == "problem" and topic.get("pre_solve_rate_transfer_problem") is not None
            else None
        ),
        "subject": subject_text,
        "subject_type": topic.get("type") if kind == "subject" else None,
        "persona_name": persona_name,
        "persona_text": persona_text,
        "learner_system": learner_system,
        "interaction_style": interaction_style,
        "disclosure_style": disclosure_style,
        "patience": profile["patience"],
    }


def _build_split(
    *,
    split: str,
    problems: list[dict[str, Any]],
    subjects: list[dict[str, Any]],
    personas: list[dict[str, Any]],
    num_rows: int,
    problem_ratio: float,
    prompt_problem: str,
    prompt_subject: str,
    incremental_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    n_problem = min(len(problems), int(round(num_rows * problem_ratio))) if problems else 0
    n_subject = max(0, num_rows - n_problem) if subjects else 0

    chosen_problems = rng.sample(problems, n_problem) if n_problem < len(problems) else list(problems)
    rows: list[dict[str, Any]] = []
    for i, prob in enumerate(chosen_problems):
        persona = personas[rng.randrange(len(personas))]
        rows.append(
            build_spec_row(
                kind="problem",
                persona_raw=persona,
                topic=prob,
                prompt_problem=prompt_problem,
                prompt_subject=prompt_subject,
                incremental_ratio=incremental_ratio,
                seed=seed + i,
                row_id=f"{split}-problem-{i:06d}",
            )
        )
    for i in range(n_subject):
        subj = subjects[i % len(subjects)]
        persona = personas[rng.randrange(len(personas))]
        rows.append(
            build_spec_row(
                kind="subject",
                persona_raw=persona,
                topic=subj,
                prompt_problem=prompt_problem,
                prompt_subject=prompt_subject,
                incremental_ratio=incremental_ratio,
                seed=seed + 100_000 + i,
                row_id=f"{split}-subject-{i:06d}",
            )
        )
    rng.shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="Output directory for {train,val}_specs.jsonl")
    parser.add_argument(
        "--problems-dataset",
        default="dmacjam/eduardoRL-transfer-dataset",
        help="HF repo or local JSONL/Parquet for math problems",
    )
    parser.add_argument(
        "--personas-dataset",
        default="nvidia/Nemotron-Personas-USA",
        help="HF repo or local JSONL/Parquet for personas",
    )
    parser.add_argument("--personas-limit", type=int, default=20_000, help="Max personas to stream/load")
    parser.add_argument("--subjects-file", type=Path, default=SUBJECTS_PATH)
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--train-rows", type=int, default=14_000, help="Total training rows (~70%% problems, 30%% subjects)")
    parser.add_argument("--val-rows", type=int, default=400, help="Total validation rows")
    parser.add_argument("--problem-ratio", type=float, default=0.7)
    parser.add_argument("--incremental-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-data-dir", type=Path, default=None, help="Path to sft-data-apertus2 repo for decon.py")
    parser.add_argument("--benchmarks-root", type=Path, default=None, help="Directory with gathered benchmarks_apertus2/*_v*")
    parser.add_argument("--decon-benchmarks", nargs="+", default=["general", "stem"])
    parser.add_argument("--skip-decon", action="store_true", help="Skip decontamination (local smoke tests only)")
    args = parser.parse_args()

    prompt_problem = (args.prompts_dir / "learner_system_problem.txt").read_text(encoding="utf-8").strip()
    prompt_subject = (args.prompts_dir / "learner_system_subject.txt").read_text(encoding="utf-8").strip()
    subjects = json.loads(args.subjects_file.read_text(encoding="utf-8"))

    train_problems = _load_table(args.problems_dataset, split="train")
    val_problems = _load_table(args.problems_dataset, split="test")
    train_problems, _ = _maybe_decontaminate(
        train_problems,
        sft_data_dir=args.sft_data_dir,
        benchmarks_root=args.benchmarks_root,
        benchmark_names=args.decon_benchmarks,
        skip_decon=args.skip_decon,
    )
    personas = _load_table(args.personas_dataset, split="train", limit=args.personas_limit)
    if not personas:
        raise SystemExit("ERROR: Loaded 0 personas.")

    # Hold out a disjoint slice of personas for validation so val tests unseen personas as well.
    rng = random.Random(args.seed)
    rng.shuffle(personas)
    val_persona_cut = max(1, min(len(personas) // 10, 1000))
    val_personas = personas[:val_persona_cut]
    train_personas = personas[val_persona_cut:] or personas

    train_rows = _build_split(
        split="train",
        problems=train_problems,
        subjects=subjects,
        personas=train_personas,
        num_rows=args.train_rows,
        problem_ratio=args.problem_ratio,
        prompt_problem=prompt_problem,
        prompt_subject=prompt_subject,
        incremental_ratio=args.incremental_ratio,
        seed=args.seed,
    )
    val_rows = _build_split(
        split="val",
        problems=val_problems,
        subjects=subjects,
        personas=val_personas,
        num_rows=args.val_rows,
        problem_ratio=args.problem_ratio,
        prompt_problem=prompt_problem,
        prompt_subject=prompt_subject,
        incremental_ratio=args.incremental_ratio,
        seed=args.seed + 1,
    )

    _write_jsonl(args.out_dir / "train_specs.jsonl", train_rows)
    _write_jsonl(args.out_dir / "val_specs.jsonl", val_rows)
    print(f"Wrote {len(train_rows)} train specs and {len(val_rows)} val specs to {args.out_dir} (English en_US only)")


if __name__ == "__main__":
    main()

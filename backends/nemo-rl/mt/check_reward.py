"""Check the complete production reward on documents before Ray/training."""

import argparse
import json
from pathlib import Path

from check_alignment import document_examples
from mt_reward.scoring import Example
from mt_reward.worker_core import RewardWorker


def reward_examples():
    source = (
        "El lunes, la biblioteca municipal inauguró una nueva sala de lectura. "
        "La sala ofrece cuarenta puestos de trabajo y también abre los fines de semana. "
        "Los visitantes pueden utilizar gratuitamente el wifi y solicitar libros del depósito. "
        "Todos los miércoles se ofrecen visitas guiadas para grupos escolares. "
        "La inscripción se realiza a través del sitio web de la biblioteca. "
        "Durante la primera semana habrá personal adicional para responder preguntas. "
        "La ciudad ha financiado la reforma para crear más espacio para estudiar. "
        "El próximo año también se habilitará una zona infantil."
    )
    examples = [{**row, "source": source, "source_lang": "es"} for row in document_examples()]
    for name, text in [("wrong_language", source), ("empty_output", "")]:
        examples.append({**examples[0], "name": name, "hypothesis": text})
    return examples


def score_example(worker, example, config):
    payload = {key: example[key] for key in Example.__dataclass_fields__}
    score = worker.score([payload])[0]
    passed = score["language_alignment"]
    alignment_bonus = config.get("alignment_weight", 20) * score["word_alignment"] if passed else 0.0
    chrf_bonus = config.get("chrf_weight", 20) * score["chrf_pp"] / 100 if passed else 0.0
    kiwi_bonus = config.get("cometkiwi_weight", 0) * score["cometkiwi"] \
        if passed and score.get("cometkiwi") is not None else 0.0
    return {**example, **score, "alignment_bonus": alignment_bonus,
            "chrf_bonus": chrf_bonus, "cometkiwi_bonus": kiwi_bonus}


def builtin_checks(results):
    rows = {row["name"]: row for row in results}
    good, identical = rows["good_paraphrase"], rows["identical_control"]
    checks = {
        "German documents pass language gate": all(rows[name]["language_alignment"] for name in
            ("identical_control", "good_paraphrase", "unrelated_document", "incorrect_facts")),
        "Reference copy gets chrF++ 100 and positive alignment":
            abs(identical["chrf_pp"] - 100) < 1e-6 and (identical["word_alignment"] or 0) > 0,
        "Wrong language and empty output get -25": all(
            not rows[name]["language_alignment"] and rows[name]["reward"] == -25
            for name in ("wrong_language", "empty_output")),
    }
    for name in ("unrelated_document", "incorrect_facts"):
        bad = rows[name]
        checks[f"MetricX prefers good paraphrase to {name}"] = (
            good["metricx"] is not None and bad["metricx"] is not None and good["metricx"] > bad["metricx"])
        checks[f"Total reward prefers good paraphrase to {name}"] = good["reward"] > bad["reward"]
    return checks


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", help="Optional JSONL with source_lang, target_lang, source, reference, hypothesis, name")
    parser.add_argument("--output", help="Write inputs, component scores, and weighted bonuses as JSONL")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a GPU allocation")
    config = json.loads(Path(args.config).read_text())
    examples = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()] \
        if args.input else reward_examples()
    if not examples:
        raise ValueError("Supply at least one example")
    worker = RewardWorker(config)
    print("Production reward models loaded", flush=True)
    print(f"{'example':26s} {'LID':>5s} {'MetricX':>10s} {'align F1':>10s} "
          f"{'align bonus':>12s} {'chrF++':>10s} {'chrF bonus':>12s} "
          f"{'Kiwi':>10s} {'Kiwi bonus':>12s} {'total':>10s}", flush=True)
    results = []
    stream = open(args.output, "w") if args.output else None
    try:
        for index, example in enumerate(examples):
            example = {"name": f"example_{index}", **example}
            row = score_example(worker, example, config)
            results.append(row)
            def fmt(value):
                return "n/a" if value is None else f"{value:.4f}"
            print(f"{row['name']:26s} {str(row['language_alignment']):>5s} "
                  f"{fmt(row['metricx']):>10s} {fmt(row['word_alignment']):>10s} "
                  f"{row['alignment_bonus']:12.4f} {row['chrf_pp']:10.4f} "
                  f"{row['chrf_bonus']:12.4f} {fmt(row.get('cometkiwi')):>10s} "
                  f"{row['cometkiwi_bonus']:12.4f} {row['reward']:10.4f}", flush=True)
            if stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
    finally:
        if stream:
            stream.close()
    if not args.input:
        checks = builtin_checks(results)
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}: {name}", flush=True)
        if not all(checks.values()):
            raise SystemExit("Document reward check failed; inspect component scores above. "
                             "A ranking failure is distinct from a model-loading failure.")
    print("Full MT reward check passed (real model inference; Ray scheduling not tested)")


if __name__ == "__main__":
    main()

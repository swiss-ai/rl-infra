"""Check production reference alignment on documents without MetricX, LID, or Ray."""

import argparse
import json
from pathlib import Path

from mt_reward.models import ReferenceAlignment
from mt_reward.scoring import bounded


def document_examples():
    reference = (
        "Am Montag eröffnete die Stadtbibliothek einen neuen Lesesaal. "
        "Der Raum bietet vierzig Arbeitsplätze und ist auch am Wochenende geöffnet. "
        "Besucher können dort kostenlos das WLAN nutzen und Bücher aus dem Magazin bestellen. "
        "Für Schulklassen werden jeden Mittwoch Führungen angeboten. "
        "Die Anmeldung erfolgt über die Website der Bibliothek. "
        "Während der ersten Woche helfen zusätzliche Mitarbeiter bei Fragen. "
        "Die Stadt hat den Umbau finanziert, um mehr Platz zum Lernen zu schaffen. "
        "Im kommenden Jahr soll außerdem ein Bereich für Kinder eingerichtet werden."
    )
    good = (
        "Die Stadtbibliothek hat am Montag einen neuen Saal zum Lesen eröffnet. "
        "Er verfügt über vierzig Arbeitsplätze und bleibt auch am Wochenende zugänglich. "
        "Gäste können kostenlos WLAN verwenden und Bücher aus dem Magazin anfordern. "
        "Jeden Mittwoch finden Führungen für Schulklassen statt. "
        "Anmelden kann man sich auf der Website der Bibliothek. "
        "In der ersten Woche stehen zusätzliche Mitarbeiter für Fragen bereit. "
        "Mit der Finanzierung des Umbaus wollte die Stadt mehr Lernplätze schaffen. "
        "Für das nächste Jahr ist zusätzlich ein Kinderbereich geplant."
    )
    unrelated = (
        "In einer Pfanne wird zuerst etwas Olivenöl erhitzt. "
        "Danach schneidet man Zwiebeln und Knoblauch klein und brät sie langsam an. "
        "Sobald die Zwiebeln weich sind, kommen gewürfelte Tomaten hinzu. "
        "Die Sauce sollte zwanzig Minuten bei geringer Hitze köcheln. "
        "In einem zweiten Topf wird gesalzenes Wasser für die Nudeln aufgekocht. "
        "Frisches Basilikum gibt dem Gericht einen kräftigen Geschmack. "
        "Zum Schluss werden die Nudeln mit der Sauce vermischt und auf Tellern angerichtet. "
        "Wer möchte, kann geriebenen Käse und schwarzen Pfeffer darüberstreuen."
    )
    wrong_facts = reference.replace("Montag", "Freitag").replace("vierzig", "zehn").replace(
        "kostenlos", "gegen Bezahlung").replace("Mittwoch", "Dienstag").replace(
        "Die Stadt hat", "Ein privates Unternehmen hat").replace("für Kinder", "für Erwachsene")
    return [
        {"name": name, "target_lang": "de", "reference": reference, "hypothesis": text}
        for name, text in [("identical_control", reference), ("good_paraphrase", good),
                           ("unrelated_document", unrelated), ("incorrect_facts", wrong_facts)]
    ]


def evaluate(scorer, example, weight):
    for key in ("reference", "hypothesis", "target_lang"):
        if not isinstance(example.get(key), str) or not example[key].strip():
            raise ValueError(f"Example requires nonempty {key}")
    language = example["target_lang"]
    details = scorer.details(example["reference"], example["hypothesis"], language, language)
    return {**example, **details,
            "alignment_bonus": weight * details["word_alignment"],
            "aligned_words": [[details["reference_words"][i], details["hypothesis_words"][j]]
                              for i, j in details["pairs"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Production scorer JSON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input", help="Optional JSONL: name, target_lang, reference, hypothesis")
    parser.add_argument("--output", help="Write complete scores and word pairs as JSONL")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["device"] = args.device
    examples = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()] \
        if args.input else document_examples()
    if not examples:
        raise ValueError("Supply at least one example")
    # Built-ins are German; preserve any additional configured language mappings.
    if not args.input:
        config["languages"] = {**config.get("languages", {}), "de": "deu_Latn"}
    weight = bounded(config.get("alignment_weight", 20), 0, float("inf"), "alignment_weight")
    scorer = ReferenceAlignment(config)
    print(json.dumps({"model": scorer.aligner.encoder.model_name,
                      "layer": scorer.aligner.encoder.layer, "device": args.device,
                      "method": "ctfalign", "alignment_weight": weight}))
    results = [evaluate(scorer, example, weight) for example in examples]
    print(f"{'example':26s} {'ref coverage':>13s} {'hyp coverage':>13s} {'F1':>9s} {'bonus':>10s}")
    for index, result in enumerate(results):
        name = result.get("name", f"example_{index}")
        print(f"{name:26s} {result['reference_coverage']:13.4f} "
              f"{result['hypothesis_coverage']:13.4f} {result['word_alignment']:9.4f} "
              f"{result['alignment_bonus']:10.4f}")
    if args.output:
        Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
        print(f"Word pairs and full scores: {args.output}")
    print("Bonus = alignment_weight * coverage F1, as in GRPO. "
          "This isolates alignment: inputs are treated as already target-language filtered; "
          "GlotLID/MaskLID, MetricX, and chrF++ are not run.")
    print("Coverage measures aligned words, not factual correctness. "
          "Compare the incorrect-facts and unrelated controls; a high score is not proof of a good translation.")


if __name__ == "__main__":
    main()

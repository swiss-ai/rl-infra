"""Check the complete production reward on a GPU before starting Ray/training."""

import argparse
import json

from mt_reward.worker_core import RewardWorker


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this check inside a GPU allocation")
    with open(args.config) as stream:
        config = json.load(stream)
    worker = RewardWorker(config)
    source = "El gato está durmiendo en el sofá junto a la ventana."
    reference = "Die Katze schläft auf dem Sofa neben dem Fenster."
    examples = [
        {"source_lang": "es", "target_lang": "de", "source": source,
         "hypothesis": reference, "reference": reference},
        {"source_lang": "es", "target_lang": "de", "source": source,
         "hypothesis": source, "reference": reference},
    ]
    scores = worker.score(examples)
    for example, score in zip(examples, scores):
        print(json.dumps({**example, **score}, ensure_ascii=False), flush=True)
    good, wrong_language = scores
    if not good["language_alignment"] or good["metricx"] is None:
        raise RuntimeError("Correct German translation did not pass the language gate")
    if good["word_alignment"] is None or good["word_alignment"] <= 0:
        raise RuntimeError("Correct translation has no word coverage; inspect MaskLID/alignment")
    if abs(good["chrf_pp"] - 100) > 1e-6:
        raise RuntimeError("Reference-matching translation did not get chrF++ 100")
    if wrong_language["language_alignment"] or wrong_language["reward"] != -25:
        raise RuntimeError("Spanish source copy did not receive the wrong-language penalty")
    print("Full MT reward check passed (model inference; Ray scheduling not tested)")


if __name__ == "__main__":
    main()

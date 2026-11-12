"""Run the production MetricX adapter with real weights, without Ray/BGE/LID."""

import argparse
import json
import platform

from mt_reward.metricx import MetricXReward


def main():
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Scorer JSON configuration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input", help="Optional JSONL with source, hypothesis, reference")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    config["device"] = args.device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("This check needs a GPU allocation and CUDA-enabled PyTorch")
    scorer = MetricXReward(config)
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "transformers": transformers.__version__, "device": args.device,
                      "loading_info": scorer.loading_info}))
    if args.input:
        with open(args.input) as stream:
            examples = [json.loads(line) for line in stream if line.strip()]
    else:
        examples = [
            {"source": "The cat is sleeping on the sofa.",
             "hypothesis": "Die Katze schläft auf dem Sofa.",
             "reference": "Die Katze schläft auf dem Sofa."},
            {"source": "The cat is sleeping on the sofa.",
             "hypothesis": "Morgen fliege ich nach Paris.",
             "reference": "Die Katze schläft auf dem Sofa."},
        ]
    if not examples:
        raise ValueError("Supply at least one example")
    for example in examples:
        reward = scorer.score(example["source"], example["hypothesis"], example["reference"])
        print(json.dumps({**example, "metricx_error": -reward, "metricx_reward": reward},
                         ensure_ascii=False))
    print("MetricX checkpoint load and reference-based inference passed")


if __name__ == "__main__":
    main()

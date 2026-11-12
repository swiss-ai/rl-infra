"""Validate parallel JSONL and preview translation prompts, without GPU packages."""

import argparse
from collections import Counter
import json

from mt_reward.prompts import format_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    directions, templates = Counter(), Counter()
    count = 0
    with open(args.input, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                example = format_example(json.loads(line), count, args.seed)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Line {line_number}: {error}") from error
            directions[f"{example['source_lang']}->{example['target_lang']}"] += 1
            templates[example["template_id"]] += 1
            if count < 2:
                print(json.dumps({"template_id": example["template_id"],
                                  "prompt": example["prompt"]}, ensure_ascii=False))
            count += 1
    if count < 35:
        raise ValueError("The smoke recipe needs at least 35 rows (32 held out + 3 training)")
    print(json.dumps({"rows": count, "directions": directions, "templates": templates}))
    print("MT JSONL validation passed (tokenization and training not tested)")


if __name__ == "__main__":
    main()

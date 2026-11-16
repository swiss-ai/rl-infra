"""Check MT chat formatting and causal training support without loading weights."""

import argparse
import json

from mt_reward.prompts import format_example


def check_template(tokenizer):
    if not tokenizer.chat_template:
        raise ValueError("The policy tokenizer has no chat template")
    example = format_example({"source_lang": "es", "target_lang": "de",
                              "source": "Buenos días.", "reference": "REFERENCE_MUST_STAY_PRIVATE"}, 0)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "Buenos días." not in rendered or "REFERENCE_MUST_STAY_PRIVATE" in rendered:
        raise ValueError("Invalid MT chat rendering or reference leakage")
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Empty tokenized prompt")
    return rendered, len(ids)


def main():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered, length = check_template(tokenizer)
    print(json.dumps({"model": args.model, "model_type": config.model_type,
                      "architectures": config.architectures, "prompt_tokens": length,
                      "rendered_prompt": rendered}, ensure_ascii=False), flush=True)
    if type(config) not in AutoModelForCausalLM._model_mapping:
        raise SystemExit(
            "This model config is not supported by AutoModelForCausalLM in the training runtime. "
            "Use a training runtime with support for this checkpoint architecture; "
            "changing the model ID or chat template alone is insufficient.")
    print("Chat-template and causal-model config checks passed. "
          "Weight loading, backward pass, and the separate vLLM runtime still need validation.")


if __name__ == "__main__":
    main()

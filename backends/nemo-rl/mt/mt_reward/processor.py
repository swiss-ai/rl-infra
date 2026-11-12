"""Translation prompt tokenization for NeMo-RL, with references kept in metadata."""

def translation_data_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    messages = [{"role": "user", "content": datum_dict["prompt"]}]
    if task_data_spec.system_prompt:
        messages.insert(0, {"role": "system", "content": task_data_spec.system_prompt})
    # Qwen3's non-thinking template starts generation directly in translation mode.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    length = len(tokens)
    if max_seq_length is not None and length >= max_seq_length:
        raise ValueError(f"MT prompt {idx} has {length} tokens, exceeding the input limit {max_seq_length}")
    metadata = {key: datum_dict[key] for key in
                ("source", "reference", "source_lang", "target_lang", "template_id")}
    return {"message_log": [{"role": "user", "content": text, "token_ids": tokens}],
            "length": length, "extra_env_info": metadata, "loss_multiplier": 1.0,
            "idx": idx, "task_name": datum_dict["task_name"]}

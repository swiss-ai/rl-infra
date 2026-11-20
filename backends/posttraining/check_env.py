"""Checks the SFT environment inside the container (check.sh runs it after build_env.sh and prepare_data.py):
package versions, one bitsandbytes AdEMAMix step on a GPU, the posttraining encoding of a few conversations
(apertus chat template, assistant-only loss mask) against the chat template the released Apertus-v1.1
Instruct models ship, and the token lengths of a random sample of the mixture.

Usage: python check_env.py <prepared dataset> <base model folder> <instruct model folder>
"""
import importlib
import sys

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from post_training.data_sft.tokenization import TokenizerConfig, get_tokenizer
from post_training.data_sft.utils_for_dataset import sft_tulu_tokenize_and_truncate


def versions():
    for name in ["torch", "transformers", "tokenizers", "huggingface_hub", "trl", "peft", "accelerate",
                 "datasets", "deepspeed", "bitsandbytes", "flash_attn", "hydra", "wandb"]:
        try:
            mod = importlib.import_module(name)
            print(f"   {name:16s} {getattr(mod, '__version__', '?'):24s} {getattr(mod, '__file__', '')}")
        except Exception as e:  # report and continue
            print(f"   {name:16s} IMPORT FAILED: {e!r}")


def ademamix_step():
    import bitsandbytes as bnb
    p = torch.nn.Parameter(torch.randn(256, 256, device="cuda"))
    opt = bnb.optim.AdEMAMix([p], lr=1e-3, betas=(0.9, 0.999, 0.999), alpha=8.0)
    before = p.detach().clone()
    (p ** 2).sum().backward()
    opt.step()
    delta = (p.detach() - before).abs().max().item()
    assert torch.isfinite(p).all() and delta > 0, delta
    print(f"== bitsandbytes AdEMAMix step on GPU: ok (max update {delta:.2e})")


def encoding(ds, base, instruct):
    tok = get_tokenizer(TokenizerConfig(model_name_or_path=base, padding_side="left", trust_remote_code=False,
                                        add_bos_to_chat_template=True, chat_template_name="apertus",
                                        model_pad_token_id=10, model_eos_token_id=68))
    ref = AutoTokenizer.from_pretrained(instruct)
    same = 0
    for i in range(20):
        row = ds[i]
        out = sft_tulu_tokenize_and_truncate(row, tok, max_seq_length=4096)
        ids = out["input_ids"].tolist()
        ref_ids = ref.apply_chat_template(row["messages"], tokenize=True, return_tensors="pt")[0].tolist()
        same += ids == ref_ids
        if i == 0:
            # The PLW trainer weights the loss with completion_mask (prompt weight 0); labels equal input_ids.
            trained = [t for t, m in zip(ids, out["completion_mask"].tolist()) if m == 1]
            print("== first conversation, full text:\n" + tok.decode(ids)[:1500])
            print("== tokens with loss (assistant only):\n" + tok.decode(trained)[:800])
            print(f"   {len(ids)} tokens, {len(trained)} with loss, last token id {ids[-1]} (eos 68)")
    print(f"== posttraining encoding equals the Instruct model's chat template on {same}/20 conversations")


def lengths(ds, base, n=20000):
    tok = get_tokenizer(TokenizerConfig(model_name_or_path=base, padding_side="left", trust_remote_code=False,
                                        add_bos_to_chat_template=True, chat_template_name="apertus",
                                        model_pad_token_id=10, model_eos_token_id=68))
    sample = ds.shuffle(seed=0).select(range(n))
    lens = sample.map(lambda r: {"n": len(tok.apply_chat_template(r["messages"], tokenize=True))},
                      num_proc=64, remove_columns=sample.column_names)["n"]
    lens = np.array(lens)
    print(f"== token lengths of {n} random conversations: mean {lens.mean():.0f}, median {np.median(lens):.0f}, "
          f"p90 {np.percentile(lens, 90):.0f}, p99 {np.percentile(lens, 99):.0f}, "
          f"over 4096: {(lens > 4096).mean() * 100:.1f}%")
    print(f"   tokens in the full mixture at this mean: {lens.mean() * len(ds) / 1e9:.2f}B (before truncation)")


if __name__ == "__main__":
    import os

    from accelerate.state import PartialState

    # posttraining's helpers log through accelerate, which needs its state; one process, no launcher.
    for key, value in {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1",
                       "MASTER_PORT": "29511"}.items():
        os.environ.setdefault(key, value)
    PartialState()
    data, base, instruct = sys.argv[1:4]
    print("== versions")
    versions()
    ademamix_step()
    ds = load_from_disk(data)
    encoding(ds, base, instruct)
    lengths(ds, base)

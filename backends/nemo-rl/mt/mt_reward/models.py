"""Model adapters for WALAR's publicly released components.

Loads upstream MetricX regression and MaskLID classes from a separate WALAR
checkout; no upstream code is copied into this repository.
"""

import importlib.util
from pathlib import Path
import sys

from .scoring import alignment_f1
from .metricx import MetricXReward


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WalarComponents:
    def __init__(self, config):
        import fasttext
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = config.get("device", "cuda:0")
        self.max_length = int(config.get("max_length", 1536))
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3")
        self.languages = config["languages"]
        if not self.languages or any(not isinstance(v, str) or "_" not in v
                                     for v in self.languages.values()):
            raise ValueError("languages must map dataset codes to GlotLID labels, e.g. en: eng_Latn")
        code = Path(config["walar_dir"]) / "code"
        masklid = load_module(code / "masklid.py", "mt_masklid_upstream")
        self.lid = fasttext.load_model(config["glotlid_model"])
        labels = ["__label__" + label for label in set(self.languages.values())]
        self.masklid = masklid.MaskLID(config["masklid_model"], languages=labels)
        # Do not silently ignore a misspelled or unavailable language label.
        for detector in (self.lid, self.masklid.model):
            missing = set(labels) - set(detector.get_labels())
            if missing:
                raise ValueError(f"Language labels absent from detector: {sorted(missing)}")
        self.metricx_scorer = MetricXReward(config)
        alignment_model = config.get("alignment_model", "BAAI/bge-m3")
        self.align_tokenizer = AutoTokenizer.from_pretrained(alignment_model, use_fast=True)
        if not self.align_tokenizer.is_fast:
            raise ValueError("BGE-M3 alignment requires a fast tokenizer for word mappings")
        self.align_model = AutoModel.from_pretrained(alignment_model).to(self.device).eval()
        self.han = None
        if any(label.startswith("cmn_") for label in self.languages.values()):
            import hanlp
            # Explicit path avoids an implicit tokenizer change for Chinese.
            self.han = hanlp.load(config["chinese_tokenizer"], devices=[])

    def label(self, language):
        if language not in self.languages:
            raise ValueError(f"Unsupported dataset language code: {language}")
        return self.languages[language]

    def language_matches(self, text, language):
        # The batch API avoids fastText 0.9.3's single-string NumPy-2
        # copy=False conversion while preserving the same top-label result.
        label = self.lid.predict([text.replace("\n", " ").strip()])[0][0][0]
        return label.removeprefix("__label__") == self.label(language)

    def mask(self, text, language):
        segments = self.masklid.predict_codeswitch(
            text.replace("\n", " "), beta=20, alpha=3, max_lambda=3,
            min_length=10, min_prob=0.90, max_retry=3,
            alpha_step_increase=3, beta_step_increase=5)
        return segments.get("__label__" + self.label(language), "")

    def metricx(self, source, hypothesis, reference):
        return self.metricx_scorer.score(source, hypothesis, reference)

    def words(self, text, language):
        if self.label(language).startswith("cmn_"):
            excluded = {",", '"', ".", "—", "(", ")", "/", "\\", "'", "：", "“", "”", "，", "。"}
            return [w for w in self.han(text) if w not in excluded]
        return text.split()

    def embeddings(self, words):
        # Preserve word-to-subword mapping, excluding special tokens even when
        # a long sentence is truncated. WALAR uses layer 24 and dot products.
        if not words:
            return None, []
        # The fast tokenizer's encoding API adds special tokens and preserves
        # word IDs. prepare_for_model cannot construct its special-token mask.
        encoded = self.align_tokenizer(
            words, is_split_into_words=True, add_special_tokens=True,
            truncation=True,
            max_length=min(self.max_length, self.align_tokenizer.model_max_length),
            return_tensors="pt")
        word_ids = encoded.word_ids()
        positions = [i for i, word_id in enumerate(word_ids) if word_id is not None]
        mapping = [word_ids[i] for i in positions]
        if not positions:
            return None, []
        inputs = dict(encoded)
        inputs = {k: v.reshape(1, -1).to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.align_model(**inputs, output_hidden_states=True).hidden_states[24][0]
        return hidden[positions], mapping

    def alignment(self, source, hypothesis, source_lang, target_lang):
        source_words, target_words = self.words(source, source_lang), self.words(hypothesis, target_lang)
        if not source_words or not target_words:
            return 0.0
        src, src_map = self.embeddings(source_words)
        tgt, tgt_map = self.embeddings(target_words)
        if src is None or tgt is None:
            return 0.0
        dot = src @ tgt.T
        pairs = ((dot.softmax(-1) > 1e-3) & (dot.softmax(-2) > 1e-3)).nonzero().tolist()
        # Unique covered words, not number of alignment edges (many-to-many).
        return alignment_f1(pairs, src_map, tgt_map, len(source_words), len(target_words))

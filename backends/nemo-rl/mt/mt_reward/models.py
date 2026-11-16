"""Model adapters for WALAR's publicly released components.

Uses CTFAlign for reference alignment and loads MetricX from a WALAR
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
        self.device = config.get("device", "cuda:0")
        self.max_length = int(config.get("max_length", 1536))
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3")
        self.languages = config["languages"]
        if not self.languages or any(not isinstance(v, str) or "_" not in v
                                     for v in self.languages.values()):
            raise ValueError("languages must map dataset codes to GlotLID labels, e.g. en: eng_Latn")
        self.lid = fasttext.load_model(config["glotlid_model"])
        labels = ["__label__" + label for label in set(self.languages.values())]
        # Do not silently ignore a misspelled or unavailable language label.
        missing = set(labels) - set(self.lid.get_labels())
        if missing:
            raise ValueError(f"Language labels absent from detector: {sorted(missing)}")
        self.cometkiwi_scorer = None
        if config.get("cometkiwi_weight", 0):
            from .cometkiwi import CometKiwiReward
            self.cometkiwi_scorer = CometKiwiReward(config)
        self.metricx_scorer = MetricXReward(config)
        self.alignment_scorer = ReferenceAlignment(config)

    def label(self, language):
        if language not in self.languages:
            raise ValueError(f"Unsupported dataset language code: {language}")
        return self.languages[language]

    def language_matches(self, text, language):
        # The batch API avoids fastText 0.9.3's single-string NumPy-2
        # copy=False conversion while preserving the same top-label result.
        label = self.lid.predict([text.replace("\n", " ").strip()])[0][0][0]
        return label.removeprefix("__label__") == self.label(language)

    def cometkiwi(self, source, hypothesis):
        if self.cometkiwi_scorer is None:
            raise RuntimeError("COMET-Kiwi was not enabled at initialization")
        return self.cometkiwi_scorer.score(source, hypothesis)

    def metricx(self, source, hypothesis, reference):
        return self.metricx_scorer.score(source, hypothesis, reference)

    def alignment(self, reference, hypothesis, reference_lang, target_lang):
        return self.alignment_scorer.alignment(reference, hypothesis, reference_lang, target_lang)


class ReferenceAlignment:
    """Production alignment component, usable without MetricX or language detectors."""

    def __init__(self, config):
        from ctfalign import WordAligner

        self.device = config.get("device", "cuda:0")
        self.languages = config["languages"]
        alignment_model = config.get("alignment_model", "BidirLM/BidirLM-1B-Embedding")
        is_bidirlm = alignment_model == "BidirLM/BidirLM-1B-Embedding"
        revision = config.get("alignment_revision", "transformers-v4" if is_bidirlm else None)
        if revision is not None:
            # CTFAlign's HFEncoder does not forward a revision to HF loaders.
            # Resolve a single snapshot for both tokenizer and custom model code.
            from huggingface_hub import snapshot_download
            alignment_model = snapshot_download(alignment_model, revision=revision)
        self.aligner = WordAligner.from_huggingface(
            alignment_model,
            method="ctfalign", mode="argmax", units="words",
            device=self.device,
            # Use the selected BidirLM hidden layer for word alignment.
            layer=config.get("alignment_layer", 15 if is_bidirlm else None),
            granularity="documents",
            trust_remote_code=config.get("alignment_trust_remote_code", is_bidirlm))
        encoder = self.aligner.encoder
        if getattr(encoder.model.config, "model_type", None) == "bidirlm":
            from transformers import AutoTokenizer
            from .bidirlm import BidirLMHiddenStates

            # Transformers 4.57.6 misidentifies this local custom config as
            # Mistral. BidirLM uses Gemma's Split(" ") tokenizer, not the
            # Mistral regex sequence; applying that patch changes tokenization
            # and crashes when indexing the standalone Split object.
            encoder.tokenizer = AutoTokenizer.from_pretrained(
                alignment_model, fix_mistral_regex=False)
            encoder.model = BidirLMHiddenStates(encoder.model, encoder.layer)
        self.han = None
        if any(label.startswith("cmn_") for label in self.languages.values()):
            import hanlp
            # Explicit path avoids an implicit tokenizer change for Chinese.
            self.han = hanlp.load(config["chinese_tokenizer"], devices=[])

    def label(self, language):
        if language not in self.languages:
            raise ValueError(f"Unsupported dataset language code: {language}")
        return self.languages[language]

    def words(self, text, language):
        if self.label(language).startswith("cmn_"):
            excluded = {",", '"', ".", "—", "(", ")", "/", "\\", "'", "：", "“", "”", "，", "。"}
            return [w for w in self.han(text) if w not in excluded]
        return text.split()

    def alignment(self, reference, hypothesis, reference_lang, target_lang):
        return self.details(reference, hypothesis, reference_lang, target_lang)["word_alignment"]

    def details(self, reference, hypothesis, reference_lang, target_lang):
        reference_words = self.words(reference, reference_lang)
        target_words = self.words(hypothesis, target_lang)
        # CTFAlign returns indices into these whitespace-delimited word lists.
        pairs = self.aligner.align(" ".join(reference_words), " ".join(target_words)) \
            if reference_words and target_words else []
        score = alignment_f1(
            pairs, range(len(reference_words)), range(len(target_words)),
            len(reference_words), len(target_words))
        return {"word_alignment": score,
                "reference_coverage": len({i for i, _ in pairs}) / len(reference_words) if reference_words else 0,
                "hypothesis_coverage": len({j for _, j in pairs}) / len(target_words) if target_words else 0,
                "reference_words": reference_words, "hypothesis_words": target_words,
                "pairs": pairs}

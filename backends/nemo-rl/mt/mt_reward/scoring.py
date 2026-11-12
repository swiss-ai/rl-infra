"""Adapted WALAR with reference-based MetricX and chrF++."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class Example:
    source_lang: str
    target_lang: str
    source: str
    hypothesis: str
    reference: str

    def __post_init__(self):
        for key in ("source_lang", "target_lang", "source", "reference"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        if not isinstance(self.hypothesis, str):
            raise ValueError("hypothesis must be a string")


class Components(Protocol):
    def language_matches(self, text: str, language: str) -> bool: ...
    def mask(self, text: str, language: str) -> str: ...
    def metricx(self, source: str, hypothesis: str, reference: str) -> float: ...
    def alignment(self, source: str, hypothesis: str,
                  source_lang: str, target_lang: str) -> float: ...


def alignment_f1(pairs, source_map, target_map, source_length, target_length):
    """Coverage F1 from subword edges; repeated edges cannot inflate coverage."""
    if not source_length or not target_length:
        return 0.0
    recall = len({source_map[i] for i, _ in pairs}) / source_length
    precision = len({target_map[j] for _, j in pairs}) / target_length
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@dataclass(frozen=True)
class Score:
    reward: float
    language_alignment: bool
    metricx: float | None
    word_alignment: float | None
    chrf_pp: float

    def asdict(self):
        return asdict(self)


def bounded(value: float, low: float, high: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid {name}: {value}; expected [{low}, {high}]")
    return value


class TranslationReward:
    def __init__(self, components: Components, *, alignment_weight=20.0,
                 chrf_weight=20.0, chrf=None):
        self.components = components
        self.alignment_weight = bounded(alignment_weight, 0, math.inf, "alignment_weight")
        self.chrf_weight = bounded(chrf_weight, 0, math.inf, "chrf_weight")
        if chrf is None:
            from sacrebleu.metrics import CHRF
            chrf = CHRF(char_order=6, word_order=2, beta=2, lowercase=False,
                        whitespace=False, eps_smoothing=False)
        self.chrf = chrf

    def score(self, example: Example) -> Score:
        # chrF++ always uses the original output, never the masked translation.
        chrf = bounded(self.chrf.sentence_score(
            example.hypothesis, [example.reference]).score, 0, 100, "chrF++")
        if not example.hypothesis.strip() or not self.components.language_matches(
                example.hypothesis, example.target_lang):
            return Score(-25.0, False, None, None, chrf)
        masked = self.components.mask(example.hypothesis, example.target_lang)
        metricx = bounded(self.components.metricx(
            example.source, example.hypothesis, example.reference), -25, 0, "MetricX")
        alignment = bounded(self.components.alignment(
            example.source, masked, example.source_lang, example.target_lang),
            0, 1, "word alignment") if masked.strip() else 0.0
        reward = metricx + self.alignment_weight * alignment + self.chrf_weight * chrf / 100
        return Score(reward, True, metricx, alignment, chrf)

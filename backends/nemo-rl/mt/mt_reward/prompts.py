"""Reproducible translation prompts and normalization of parallel JSONL rows."""

import hashlib


LANGUAGE_NAMES = {
    "es": "Spanish", "de": "German", "en": "English", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ru": "Russian",
}

TEMPLATES = (
    "Translate the following {source_language} text into {target_language}. "
    "Return only the translation.\n\n{source}",
    "Provide a {target_language} translation of this {source_language} passage. "
    "Output only the translated text.\n\n{source}",
    "Translate from {source_language} to {target_language}, preserving the meaning. "
    "Respond with the translation only.\n\n{source}",
    "How would this {source_language} text read in {target_language}? "
    "Give only the translation, without explanation.\n\n{source}",
)


def format_example(row, index, seed=42, language_names=None):
    names = LANGUAGE_NAMES if language_names is None else language_names
    fields = {"source": ("source", "src"), "reference": ("reference", "tgt"),
              "source_lang": ("source_lang", "source_language"),
              "target_lang": ("target_lang", "target_language")}
    normalized = {}
    for key, aliases in fields.items():
        value = next((row[name] for name in aliases if name in row), None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Row {index}: {key} must be a nonempty string")
        normalized[key] = value
    src, tgt = normalized["source_lang"], normalized["target_lang"]
    for language in (src, tgt):
        if language not in names:
            raise ValueError(f"Row {index}: no language name mapping for {language!r}")
    # Independent of worker count, Python hash randomization, and rollout order.
    digest = hashlib.sha256(f"{seed}:{index}:{src}:{tgt}".encode()).digest()
    template_id = int.from_bytes(digest[:8], "big") % len(TEMPLATES)
    prompt = TEMPLATES[template_id].format(
        source_language=names[src], target_language=names[tgt], source=normalized["source"])
    return {**normalized, "prompt": prompt, "template_id": template_id}

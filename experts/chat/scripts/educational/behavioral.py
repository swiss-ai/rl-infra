# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""English-only behavioral augmentation and persona formatting adapted from NVIDIA NeMo UserSim.

Source: https://github.com/NVIDIA-NeMo/UserSim
  - src/usersim/engine/core/behavioral.py
  - src/usersim/engine/core/persona.py
  - src/usersim/engine/core/user_language.py

Kept self-contained (standard library only) so data preparation runs on login nodes
without installing the full UserSim stack. Non-English locales are disabled in v1.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

DEFAULT_LOCALE = "en_US"
DEFAULT_LANGUAGE = "English"

_OCEAN_TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

_EDUCATION_ORDINAL_EN_US: dict[str, float] = {
    "less_than_9th": 0.0,
    "9th_12th_no_diploma": 0.15,
    "high_school": 0.3,
    "some_college": 0.45,
    "associates": 0.5,
    "bachelors": 0.7,
    "graduate": 1.0,
}

_ED_HIGH_KEYWORDS = ("master", "doctor", "phd", "bachelor", "graduate", "university")
_ED_LOW_KEYWORDS = ("illiterate", "no_formal", "less_than", "below_primary")

_TECH_OCCUPATIONS_EXACT = {
    "software_developer",
    "software_quality_assurance_analyst_or_tester",
    "computer_programmer",
    "computer_or_information_research_scientist",
    "computer_hardware_engineer",
    "computer_or_information_systems_manager",
    "computer_systems_analyst",
    "computer_support_specialist",
    "computer_network_architect",
    "computer_occupation",
    "web_developer",
    "web_or_digital_interface_designer",
    "database_administrator_or_architect",
    "network_or_computer_systems_administrator",
    "information_security_analyst",
}
_TECH_KEYWORDS = {"software", "computer", "programmer", "developer", "database"}

# Fallback OCEAN descriptions when a HuggingFace persona row (e.g. nvidia/Nemotron-Personas-USA)
# does not ship DataDesigner's OCEAN columns.
_OCEAN_FALLBACK_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "openness": {
        "low": "Prefers familiar routines and concrete examples over abstract theory.",
        "average": "Balanced between practical thinking and curiosity for new ideas.",
        "high": "Curious, imaginative, and eager to explore abstract ideas and connections.",
    },
    "conscientiousness": {
        "low": "Spontaneous and informal; may skip details or jump ahead.",
        "average": "Moderately organized and attentive to details when needed.",
        "high": "Methodical, thorough, and careful to verify each step.",
    },
    "extraversion": {
        "low": "Reserved and concise; keeps messages brief and focused.",
        "average": "Conversational and moderately expressive.",
        "high": "Outgoing, chatty, and expressive in conversation.",
    },
    "agreeableness": {
        "low": "Skeptical, direct, and quick to question unclear explanations.",
        "average": "Polite yet willing to push back if something feels off.",
        "high": "Warm, patient, and collaborative.",
    },
    "neuroticism": {
        "low": "Calm and steady even when stuck on a hard point.",
        "average": "Occasionally uneasy when confused, but recovers quickly.",
        "high": "Easily stressed or self-doubting when explanations feel overwhelming.",
    },
}

USER_QUERY_INSTRUCTION_GROUNDED = (
    "You are chatting with an AI assistant right now. "
    "Ask your question or make your request — a short, natural message "
    "in plain text, no formatting. "
    "This is your first conversation with this assistant; it knows "
    "nothing about you yet. Your question should reflect who you are — "
    "make it a question only YOU would ask, not something anyone could "
    "ask. Do not vaguely reference personal attributes without "
    "specifying them. "
    "Stay in character as the user."
)

DISCLOSURE_INSTRUCTIONS = {
    "incremental": (
        "<INFORMATION_DISCLOSURE>\n"
        "IMPORTANT — Realistic information sharing:\n"
        "- Do NOT provide all relevant details in your first message.\n"
        "- State your general need first, without a comprehensive brief.\n"
        "- Reveal specific details (names, IDs, numbers, dates) only when "
        "asked or when it becomes naturally relevant.\n"
        "- If you have identifying information, wait for the assistant to ask.\n"
        "- Keep your initial message short and conversational.\n"
        "- When the assistant DOES ask for something you would know — your own "
        "details, or those of someone you brought up yourself — answer it. Only "
        "say you are unsure about things you plausibly would not know offhand "
        "(an exact transaction date, a reference number, a precise balance).\n"
        "</INFORMATION_DISCLOSURE>"
    ),
    "upfront": "",
}

INTERACTION_STYLES = {
    "confrontational": {
        "base_frustration": 2,
        "escalation_rate": 2.0,
        "description": (
            "You have a confrontational personality. You are direct, blunt, "
            "and do not tolerate inefficiency. You speak your mind plainly "
            "and do not sugarcoat your reactions."
        ),
    },
    "frustrated": {
        "base_frustration": 1,
        "escalation_rate": 1.5,
        "description": (
            "You are easily frustrated and have little patience. "
            "When things feel slow or unclear, you become terse and "
            "let your annoyance show through your tone."
        ),
    },
    "impatient": {
        "base_frustration": 0,
        "escalation_rate": 1.5,
        "description": (
            "You are impatient and value your time. You give short, "
            "direct replies — a sentence or two at most. You prefer "
            "concise answers and do not want lengthy explanations."
        ),
    },
    "neutral": {
        "base_frustration": 0,
        "escalation_rate": 1.0,
        "description": "",
    },
    "cooperative": {
        "base_frustration": 0,
        "escalation_rate": 0.3,
        "description": (
            "You are patient and cooperative. You give the assistant the "
            "benefit of the doubt and respond politely even when things "
            "take longer than expected."
        ),
    },
}

_CHAT_REALISM_STYLE_EN = (
    "- Write like you talk: use contractions (don't, I'm, can't), fragments "
    "('Makes sense', 'Got it'), and hedging ('I think', 'maybe', 'not sure').\n"
    "- React first, then ask: 'Oh interesting — but what about X?' or "
    "'Hmm ok, so does that mean Y?'\n"
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def education_ordinal(education_level: str) -> float:
    if education_level in _EDUCATION_ORDINAL_EN_US:
        return _EDUCATION_ORDINAL_EN_US[education_level]
    ed_lower = education_level.lower()
    if any(kw in ed_lower for kw in _ED_HIGH_KEYWORDS):
        return 0.8
    if any(kw in ed_lower for kw in _ED_LOW_KEYWORDS):
        return 0.1
    return 0.5


def _is_tech_occupation(occupation: str) -> bool:
    occ_lower = occupation.lower().strip()
    if occ_lower in _TECH_OCCUPATIONS_EXACT:
        return True
    words = set(re.split(r"[_\s,/]+", occ_lower))
    return bool(words & _TECH_KEYWORDS)


def _stable_seed(key: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def ensure_ocean_traits(persona: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Return a copy of ``persona`` with OCEAN trait dicts populated if absent.

    HuggingFace ``nvidia/Nemotron-Personas-USA`` omits OCEAN columns, whereas
    NVIDIA DataDesigner's managed persona tables include ``{label, t_score, description}``.
    When absent, we deterministically draw Gaussian T-scores (mean=50, sd=10, clipped 20..80)
    seeded by the persona's uuid/text so each persona gets a stable OCEAN profile.
    """
    if all(persona.get(t) is not None for t in _OCEAN_TRAITS):
        return dict(persona)
    out = dict(persona)
    key = str(persona.get("uuid") or persona.get("persona") or persona)
    rng = random.Random(_stable_seed(key, seed))
    for trait in _OCEAN_TRAITS:
        if out.get(trait) is not None:
            continue
        t_score = round(max(20.0, min(80.0, rng.gauss(50.0, 10.0))), 1)
        label = "low" if t_score < 45 else "high" if t_score > 55 else "average"
        out[trait] = {
            "t_score": t_score,
            "label": label,
            "description": _OCEAN_FALLBACK_DESCRIPTIONS[trait][label],
        }
    return out


def _parse_ocean_value(raw: Any) -> float:
    if isinstance(raw, dict):
        return _clamp((float(raw.get("t_score", 50)) - 20.0) / 60.0)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return _clamp((float(parsed.get("t_score", 50)) - 20.0) / 60.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        return _clamp(float(raw))
    except (ValueError, TypeError):
        return 0.5


def _extract_ocean_metadata(persona: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    labels: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for trait in _OCEAN_TRAITS:
        raw = persona.get(trait)
        d: dict[str, Any] | None = None
        if isinstance(raw, dict):
            d = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    d = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        if d is not None:
            labels[trait] = str(d.get("label", "?"))
            descriptions[trait] = str(d.get("description", ""))
        else:
            labels[trait] = "?"
            descriptions[trait] = ""
    return labels, descriptions


def _infer_tech_literacy(persona: dict[str, Any]) -> float:
    score = 0.5
    ed_ord = education_ordinal(str(persona.get("education_level", "")))
    if ed_ord >= 0.7:
        score += 0.15
    elif ed_ord <= 0.15:
        score -= 0.2
    if _is_tech_occupation(str(persona.get("occupation", ""))):
        score += 0.25
    age = persona.get("age")
    if age is not None and str(age).strip():
        try:
            age_i = int(float(age))
            if age_i > 75:
                score -= 0.25
            elif age_i > 60:
                score -= 0.15
            elif age_i < 30:
                score += 0.1
        except (ValueError, TypeError):
            pass
    return _clamp(score)


def compute_behavioral_profile(persona: dict[str, Any]) -> dict[str, Any]:
    ocean = {t: _parse_ocean_value(persona.get(t, 0.5)) for t in _OCEAN_TRAITS}
    patience = _clamp(0.5 + 0.3 * ocean["agreeableness"] - 0.3 * ocean["neuroticism"])
    verbosity = _clamp(0.3 + 0.35 * ocean["extraversion"] + 0.2 * ocean["openness"])
    cooperativeness = _clamp(0.3 + 0.5 * ocean["agreeableness"] + 0.1 * ocean["conscientiousness"])
    tech_literacy = _infer_tech_literacy(persona)

    ed_ord = education_ordinal(str(persona.get("education_level", "")))
    error_proneness = 0.3
    if ed_ord <= 0.15:
        error_proneness += 0.3
    if tech_literacy < 0.3:
        error_proneness += 0.2
    error_proneness = _clamp(error_proneness)

    ocean_labels, ocean_descriptions = _extract_ocean_metadata(persona)
    return {
        "patience": round(patience, 2),
        "verbosity": round(verbosity, 2),
        "cooperativeness": round(cooperativeness, 2),
        "tech_literacy": round(tech_literacy, 2),
        "error_proneness": round(error_proneness, 2),
        "ocean": {k: round(v, 2) for k, v in ocean.items()},
        "ocean_labels": ocean_labels,
        "ocean_descriptions": ocean_descriptions,
    }


def format_behavioral_profile_for_prompt(profile: dict[str, Any]) -> str:
    lines: list[str] = []
    ocean_descs = profile.get("ocean_descriptions", {})
    if any(ocean_descs.values()):
        lines.append("Personality profile:")
        for trait in _OCEAN_TRAITS:
            desc = ocean_descs.get(trait, "")
            if desc:
                lines.append(f"- {desc}")

    tech_desc = (
        "tech-savvy"
        if profile["tech_literacy"] > 0.7
        else "not very tech literate"
        if profile["tech_literacy"] < 0.4
        else "moderately tech literate"
    )
    error_desc = (
        "prone to typos and ambiguous phrasing"
        if profile["error_proneness"] > 0.6
        else "generally clear in communication"
    )
    lines.extend(["", "Additional traits:", f"- {tech_desc}", f"- {error_desc}", ""])
    lines.append(
        "<CHAT_REALISM>\n"
        "CRITICAL — You are typing in a chat window, not writing a document:\n"
        "- Use plain text ONLY. Absolutely NO markdown: no headers (#), no bold (**), "
        "no tables, no horizontal rules (---), no bullet lists with dashes.\n"
        "- Keep each reply SHORT — 1 to 4 sentences is typical for a chat message.\n"
        "- Focus on ONE thing per message: one question, one reaction, one concern.\n"
        + _CHAT_REALISM_STYLE_EN
        + "- You are the person ASKING for help. You do NOT have the answers. "
        "Never provide information, data, analysis, or explanations — that is "
        "the assistant's job.\n"
        "- When the assistant asks for details (names, dates, numbers, preferences), "
        "provide ONLY what was asked and then stop. Do NOT continue to do the task "
        "yourself — let the assistant use your input to produce the result.\n"
        "- Never echo, restate, or restructure what the assistant said. "
        "If you understood, say so briefly and move on.\n"
        "- Do NOT number your points or create organized lists. Real people "
        "in chat do not structure replies that way.\n"
        "- Never offer to help the assistant — you are the one receiving help, "
        "not giving it.\n"
        "- Never refer to the conversation itself, your role in it, or how "
        "messages are being written.\n"
        "</CHAT_REALISM>"
    )
    return "\n".join(lines)


def compute_disclosure_style(persona_seed: int | None = None, incremental_ratio: float = 0.6) -> str:
    rng = random.Random(persona_seed)
    return "incremental" if rng.random() < incremental_ratio else "upfront"


def format_disclosure_instructions(style: str) -> str:
    return DISCLOSURE_INSTRUCTIONS.get(style, "")


def compute_user_interaction_style(profile: dict[str, Any]) -> str:
    ocean = profile.get("ocean", {})
    neuroticism = ocean.get("neuroticism", 0.5)
    agreeableness = ocean.get("agreeableness", 0.5)
    patience = profile.get("patience", 0.5)
    if neuroticism > 0.6 and agreeableness < 0.45:
        return "confrontational"
    if neuroticism > 0.5 and patience < 0.45:
        return "frustrated"
    if agreeableness < 0.45 or patience < 0.4:
        return "impatient"
    if agreeableness > 0.6 and neuroticism < 0.4:
        return "cooperative"
    return "neutral"


def format_interaction_style_instructions(style: str) -> str:
    info = INTERACTION_STYLES.get(style, INTERACTION_STYLES["neutral"])
    desc = info["description"]
    if not desc:
        return ""
    return f"<INTERACTION_STYLE>\n{desc}\n</INTERACTION_STYLE>"


def language_instruction() -> str:
    return f"IMPORTANT: You MUST write your messages in {DEFAULT_LANGUAGE}."


def persona_display_name(persona: dict[str, Any]) -> str:
    for key in ("name", "full_name"):
        val = str(persona.get(key) or "").strip()
        if val:
            return val
    first = str(persona.get("first_name") or "").strip()
    last = str(persona.get("last_name") or "").strip()
    if first or last:
        return f"{first} {last}".strip()
    text = str(persona.get("persona") or persona.get("professional_persona") or "").strip()
    m = re.match(r"^([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){0,2})\s+(?:is|was|has|works|lives)\b", text)
    if m:
        return m.group(1)
    return "Learner"


def format_persona_for_prompt(persona: dict[str, Any], seed: int = 0) -> str:
    """Format a Nemotron-Personas dict into a compact prompt block."""
    name = persona_display_name(persona)
    demo: list[str] = [f"Name: {name}"]
    for label, key in (
        ("Age", "age"),
        ("Sex", "sex"),
        ("Education", "education_level"),
        ("Occupation", "occupation"),
        ("Marital Status", "marital_status"),
    ):
        val = str(persona.get(key) or "").strip()
        if val:
            demo.append(f"{label}: {val}")
    loc_parts = [str(persona.get(k) or "").strip() for k in ("city", "state", "country")]
    loc = ", ".join(p for p in loc_parts if p)
    if loc:
        demo.append(f"Location: {loc}")

    facets: list[tuple[str, str]] = []
    for label, key in (
        ("Background", "persona"),
        ("Professional", "professional_persona"),
        ("Cultural Background", "cultural_background"),
        ("Skills", "skills_and_expertise"),
        ("Hobbies", "hobbies_and_interests"),
        ("Goals", "career_goals_and_ambitions"),
    ):
        val = str(persona.get(key) or "").strip()
        if val:
            facets.append((label, val))
    rng = random.Random(_stable_seed(name + str(persona.get("uuid", "")), seed))
    rng.shuffle(facets)
    lines = [" | ".join(demo)]
    for label, val in facets[:4]:
        lines.append(f"{label}: {val}")
    return "\n".join(lines)

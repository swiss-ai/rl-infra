#!/usr/bin/env python3
"""Precompute gated opening user messages for educational specs and emit NeMo-RL JSONL.

Reads ``train_specs.jsonl`` / ``val_specs.jsonl`` from ``prepare_educational.py``, calls any
OpenAI-compatible endpoint serving Gemma 4 (with ``USER_QUERY_INSTRUCTION_GROUNDED``,
fourth-wall / refusal-echo checks, and the ``learner_gate.txt`` judge with up to
``--gate-retries`` retries), and writes ``train.jsonl`` / ``val.jsonl`` in ``ResponseDataset``
format:

    {"messages": [{"role": "user", "content": "<opener>"}], "tutoring": {...}}

Precomputing the opener once keeps all 8 GRPO rollouts of a prompt on the exact same
initial user turn and keeps generation out of ``reset()``.

Example:
    python3.12 experts/chat/scripts/educational/make_openers.py \
        ${RL_DATA_DIR}/chat-rl/educational/train_specs.jsonl \
        ${RL_DATA_DIR}/chat-rl/educational/train.jsonl \
        --base-url http://localhost:8000/v1 \
        --model ${SCRATCH}/models/gemma-4-26B-A4B-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp

from behavioral import USER_QUERY_INSTRUCTION_GROUNDED

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "prompts"

_FOURTH_WALL_PHRASES = (
    "as an ai",
    "as a language model",
    "in this simulation",
    "roleplay",
    "role-play",
    "stay in character",
    "first message",
    "start the conversation",
)

_REFUSAL_ECHO_PHRASES = (
    "i'm sorry, but i can't",
    "sorry, but i can't",
    "i can't help with that",
    "i can't assist with that",
    "i cannot help with that",
    "i cannot assist with that",
    "i'm unable to help",
)

_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string", "maxLength": 400},
        "rating": {"type": "string", "enum": ["success", "warn", "failure"]},
    },
    "required": ["explanation", "rating"],
    "additionalProperties": False,
}


def _quick_reject_reason(text: str) -> str | None:
    if not text or not text.strip():
        return "empty message"
    lower = " ".join(text.lower().split())
    if any(p in lower for p in _FOURTH_WALL_PHRASES):
        return "fourth-wall break or meta-reference to the simulation"
    if any(p in lower for p in _REFUSAL_ECHO_PHRASES):
        return "sounds like an assistant refusal instead of a learner"
    return None


def _parse_gate(text: str) -> tuple[str, str, bool]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("rating") in ("success", "warn", "failure"):
            return str(obj.get("explanation", "")).strip(), str(obj["rating"]), True
    except (json.JSONDecodeError, TypeError):
        pass
    m_rat = re.search(r"<rating>\s*(success|warn|failure)\s*</rating>", text, re.I)
    m_exp = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.S | re.I)
    if m_rat:
        return (m_exp.group(1).strip() if m_exp else ""), m_rat.group(1).lower(), True
    return text.strip()[:300], "failure", False


async def _chat(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    body: dict[str, Any],
    retries: int,
) -> str | None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(retries + 1):
        try:
            async with session.post(url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                print(f"[make_openers] request failed after {retries + 1} tries: {exc}", file=sys.stderr)
                return None
            await asyncio.sleep(min(2.0 * (attempt + 1), 10.0))
    return None


async def _generate_one(
    row: dict[str, Any],
    *,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    base_url: str,
    api_key: str,
    model: str,
    gate_template: str,
    retry_template: str,
    gate_retries: int,
    use_gate: bool,
    structured_gate: bool,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool,
    http_retries: int,
) -> dict[str, Any] | None:
    extra_kwargs: dict[str, Any] = {}
    if disable_thinking:
        extra_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    messages: list[dict[str, str]] = [
        {"role": "system", "content": row["learner_system"]},
        {"role": "user", "content": USER_QUERY_INSTRUCTION_GROUNDED},
    ]

    async with semaphore:
        for attempt in range(gate_retries + 1):
            draft = await _chat(
                session,
                base_url,
                api_key,
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "max_tokens": max_tokens,
                    **extra_kwargs,
                },
                retries=http_retries,
            )
            if not draft:
                return None

            reason = _quick_reject_reason(draft)
            passed = reason is None
            if passed and use_gate:
                gate_body: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an expert judge."},
                        {
                            "role": "user",
                            "content": gate_template.format(
                                conversation_history="(Opening message — no previous turns.)",
                                user_turn_to_evaluate=draft,
                            ),
                        },
                    ],
                    "temperature": temperature,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    **extra_kwargs,
                }
                if structured_gate:
                    gate_body["structured_outputs"] = {"json": _GATE_SCHEMA}
                gate_raw = await _chat(session, base_url, api_key, gate_body, retries=http_retries)
                if gate_raw is None:
                    return None
                expl, rating, parsed_ok = _parse_gate(gate_raw)
                passed = parsed_ok and rating in ("success", "warn")
                reason = expl or "failed learner gate"

            if passed:
                spec = dict(row)
                spec["opener"] = draft
                return {
                    "messages": [{"role": "user", "content": draft}],
                    "tutoring": spec,
                }

            if attempt < gate_retries:
                messages.append({"role": "assistant", "content": draft})
                messages.append({"role": "user", "content": retry_template.format(reason=(reason or "")[:400])})

    return None


async def _run(args: argparse.Namespace) -> None:
    gate_template = (args.prompts_dir / "learner_gate.txt").read_text(encoding="utf-8")
    retry_template = (args.prompts_dir / "learner_retry.txt").read_text(encoding="utf-8").strip()

    specs = []
    with args.specs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    if args.limit is not None:
        specs = specs[: args.limit]

    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    semaphore = asyncio.Semaphore(args.concurrency)

    kept: list[dict[str, Any]] = []
    dropped = 0
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            _generate_one(
                row,
                session=session,
                semaphore=semaphore,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                gate_template=gate_template,
                retry_template=retry_template,
                gate_retries=args.gate_retries,
                use_gate=not args.no_gate,
                structured_gate=not args.no_structured_gate,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                disable_thinking=not args.allow_thinking,
                http_retries=args.http_retries,
            )
            for row in specs
        ]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            result = await fut
            if result is None:
                dropped += 1
            else:
                kept.append(result)
            if i % 250 == 0 or i == len(tasks):
                print(f"[make_openers] {i}/{len(tasks)} done (kept={len(kept)}, dropped={dropped})", flush=True)

    # Restore deterministic order by row_id.
    kept.sort(key=lambda r: r["tutoring"]["row_id"])
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} rows to {args.out_path} (dropped {dropped})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs_path", type=Path, help="Input *_specs.jsonl from prepare_educational.py")
    parser.add_argument("out_path", type=Path, help="Output train.jsonl or val.jsonl")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL (e.g. http://host:8000/v1)")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--gate-retries", type=int, default=2)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-structured-gate", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-thinking", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--http-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

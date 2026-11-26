#!/usr/bin/env python3
"""Unit tests for prepare_educational.py, make_openers.py, and behavioral.py."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import behavioral
import make_openers
import prepare_educational


def test_behavioral_and_prepare_spec_rows() -> None:
    persona = {
        "uuid": "abc123",
        "persona": "Mary Alberti is a routine-obsessed food service specialist.",
        "education_level": "high_school",
        "occupation": "fast_food_or_counter_worker",
        "age": 28,
        "sex": "Female",
        "city": "Madison",
        "state": "WI",
        "country": "USA",
    }
    enriched = behavioral.ensure_ocean_traits(persona, seed=42)
    assert all(t in enriched for t in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"))
    profile = behavioral.compute_behavioral_profile(enriched)
    assert 0.0 <= profile["patience"] <= 1.0
    assert behavioral.persona_display_name(enriched) == "Mary Alberti"

    prompt_prob = (SCRIPT_DIR / "prompts" / "learner_system_problem.txt").read_text(encoding="utf-8")
    prompt_subj = (SCRIPT_DIR / "prompts" / "learner_system_subject.txt").read_text(encoding="utf-8")
    row_prob = prepare_educational.build_spec_row(
        kind="problem",
        persona_raw=persona,
        topic={
            "problem": "Solve 3x = 12.",
            "answer": "4",
            "transfer_problem": "Solve 5x = 20.",
            "transfer_answer": "4",
            "pre_solve_rate": 0.25,
            "pre_solve_rate_transfer_problem": 0.3,
        },
        prompt_problem=prompt_prob,
        prompt_subject=prompt_subj,
        incremental_ratio=0.6,
        seed=7,
        row_id="train-problem-000000",
    )
    assert row_prob["kind"] == "problem"
    assert row_prob["locale"] == "en_US"
    assert row_prob["language"] == "English"
    assert "Solve 3x = 12." in row_prob["learner_system"]
    assert row_prob["answer"] == "4"
    assert row_prob["persona_name"] == "Mary Alberti"

    row_subj = prepare_educational.build_spec_row(
        kind="subject",
        persona_raw=persona,
        topic={"type": "algebra_functions", "description": "Algebra, functions, and graph interpretation"},
        prompt_problem=prompt_prob,
        prompt_subject=prompt_subj,
        incremental_ratio=0.6,
        seed=8,
        row_id="train-subject-000000",
    )
    assert row_subj["kind"] == "subject"
    assert row_subj["locale"] == "en_US"
    assert "Algebra, functions, and graph interpretation" in row_subj["learner_system"]
    assert row_subj["answer"] is None


def test_make_openers_end_to_end(tmp_path: Path) -> None:
    specs_path = tmp_path / "specs.jsonl"
    out_path = tmp_path / "train.jsonl"
    specs_path.write_text(
        json.dumps(
            {
                "row_id": "train-problem-000000",
                "kind": "problem",
                "locale": "en_US",
                "language": "English",
                "problem": "Solve 3x = 12.",
                "answer": "4",
                "learner_system": "You are Alex, a learner.",
                "persona_name": "Alex",
                "persona_text": "Alex",
                "interaction_style": "neutral",
                "disclosure_style": "incremental",
                "patience": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    attempts = {"learner": 0}

    async def run() -> None:
        app = web.Application()

        async def handler(request: web.Request) -> web.Response:
            body: dict[str, Any] = await request.json()
            if "expert judge" in body["messages"][0]["content"]:
                return web.json_response(
                    {"choices": [{"message": {"content": json.dumps({"explanation": "good", "rating": "success"})}}]}
                )
            attempts["learner"] += 1
            if attempts["learner"] == 1:
                text = "As an AI language model in this simulation, here is my question."
            else:
                text = "Hey, I'm stuck on 3x = 12 — how do I figure out x?"
            return web.json_response({"choices": [{"message": {"content": text}}]})

        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        try:
            args = type(
                "Args",
                (),
                {
                    "specs_path": specs_path,
                    "out_path": out_path,
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "model": "mock-gemma",
                    "api_key": "",
                    "prompts_dir": SCRIPT_DIR / "prompts",
                    "concurrency": 4,
                    "gate_retries": 2,
                    "no_gate": False,
                    "no_structured_gate": False,
                    "max_tokens": 128,
                    "temperature": 1.0,
                    "allow_thinking": False,
                    "timeout_s": 30.0,
                    "http_retries": 0,
                    "limit": None,
                },
            )()
            await make_openers._run(args)
        finally:
            await runner.cleanup()

    asyncio.run(run())
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert attempts["learner"] == 2
    assert rows[0]["messages"] == [{"role": "user", "content": "Hey, I'm stuck on 3x = 12 — how do I figure out x?"}]
    assert rows[0]["tutoring"]["opener"] == rows[0]["messages"][0]["content"]

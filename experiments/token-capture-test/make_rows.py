#!/usr/bin/env python3
"""Build train.jsonl and validation.jsonl for mini-swe-osb.yaml from Gym's SWE-bench Verified sample.

    python3 make_rows.py --gym <NeMo-RL clone>/3rdparty/Gym-workspace/Gym --out <HARBOR_ROOT>

Each row is a SWE-bench instance (mini_swe_agent_2 reads the instance from the /run body), plus the
agent to route it to and the on-policy sampling parameters Gym's vLLM server asserts on.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--gym", required=True, type=Path, help="Gym checkout")
ap.add_argument("--out", required=True, type=Path, help="directory for train.jsonl and validation.jsonl")
ap.add_argument("--train", type=int, default=2, help="training rows (= grpo.num_prompts_per_step)")
ap.add_argument("--max-output-tokens", type=int, default=2048, help="per model call")
a = ap.parse_args()

src = a.gym / "responses_api_agents/mini_swe_agent_2/data/example.jsonl"
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
if len(rows) < a.train + 1:
    raise SystemExit(f"{src} has {len(rows)} rows, need {a.train + 1}")
for row in rows:
    row["agent_ref"] = {"type": "responses_api_agents", "name": "mini_swe_agent_2"}
    row["responses_create_params"] = {
        "input": [],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_output_tokens": a.max_output_tokens,
    }

a.out.mkdir(parents=True, exist_ok=True)
for name, part in (("train", rows[: a.train]), ("validation", rows[a.train : a.train + 1])):
    (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    print(f"{a.out / name}.jsonl: {', '.join(r['instance_id'] for r in part)}")

# RL experts

One folder per RL expert (the domain teachers trained in parallel). Inside a folder: `base.yaml` defines the
expert's datasets, environments and validation once, `main.yaml` is its official run, `dev/` holds development
variants and single-environment tasks, `archive/` the run files of earlier runs as they ran. Only `stem/` has the
base/main layout so far; the other experts still hold one task file per run.

| Expert | Folder | Task files |
|---|---|---|
| Medical | — | none yet |
| Safety | — | none yet |
| General chat, helpfulness, instruction following | `chat/` | UltraFeedback prompts rewarded by one, two or three LLM judges on rubrics, AlpacaEval validation |
| STEM: math, reasoning, physics, chemistry | `stem/` | `main.yaml` (8-dataset mix of run 5), GSM8K and DAPO-Math, Reasoning Gym and chemistry alone (`dev/`), runs 1-5 (`archive/`) |
| Code, SWE, terminal, agentic | `code/` | competitive coding on Gym's code_gen (sandbox node or OpenSandbox pods), Harbor terminal/SWE tasks with Terminus-2 |
| Knowledge, factuality, QA | — | none yet (the STEM mix trains science multiple choice and open science QA) |
| Tool use, web search | `tool-use/` | Gym's workplace assistant (27 simulated tools) |
| Translation | — | none yet |
| Lean (formal proofs) | `lean/` | Lean 4 proving with Gym math_formal_lean |

`frames/` holds the system frames of the Apertus 2 format (thinking level, declared tools) and `common/` helpers
several experts use (judge model download, few-shot removal, a log-probability debug task).

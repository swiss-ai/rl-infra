# RL experts

One folder per RL expert (the domain teachers trained in parallel). Inside a folder: `base.yaml` defines the
expert's datasets, environments and validation once, `main.yaml` is its official run, `dev/` holds development
variants and single-environment tasks, `archive/` the run files of earlier runs as they ran. `stem/` and the
templates of the new experts (medical, safety, knowledge, translation) have this layout; chat, code, tool-use and
lean still hold one task file per run. A template carries `template: true`, which run.sh refuses to submit; fill in
its datasets, then delete that key.

| Expert | Folder | Task files |
|---|---|---|
| Medical | `medical/` | template: medical multiple choice (`base.yaml`, `main.yaml`) |
| Safety | `safety/` | template: prompts graded by the shared LLM judge |
| General chat, helpfulness, instruction following | `chat/` | UltraFeedback prompts rewarded by one, two or three LLM judges on rubrics, AlpacaEval validation |
| STEM: math, reasoning, physics, chemistry | `stem/` | `main.yaml` (8-dataset mix of run 5), GSM8K and DAPO-Math, Reasoning Gym and chemistry alone (`dev/`), runs 1-5 (`archive/`) |
| Code, SWE, terminal, agentic | `code/` | competitive coding on Gym's code_gen (sandbox node or OpenSandbox pods), Harbor terminal/SWE tasks with Terminus-2 |
| Knowledge, factuality, QA | `knowledge/` | template: open QA (judge) and multiple choice; the STEM mix already trains science multiple choice and open science QA |
| Tool use, web search | `tool-use/` | Gym's workplace assistant (27 simulated tools) |
| Translation | `translation/` | template: translations graded by the shared LLM judge |
| Lean (formal proofs) | `lean/` | Lean 4 proving with Gym math_formal_lean |

Two folders are not experts:
- `frames/`: the system frame every prompt of the Apertus 2 message format starts with, as small JSON files: the
  thinking level (`{"thinking":"high","tools":[]}`) and the tools the model may call (ScienceWorld's `act`). A dataset
  picks one with `system_prompt_file`; in the base/main layout through `frames.<key>`.
- `common/`: pieces several experts use: `llm-judge.yaml` (the judge environment and its node, taken in with
  `defaults: ../common/llm-judge.yaml`), `stage_judge_model.sh` (downloads the judge model), `strip_few_shot.py`
  (removes the few-shot prompt from prepared Gym rows) and `score-text.yaml` (a log-probability debug task).

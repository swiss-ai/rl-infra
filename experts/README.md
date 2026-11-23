# RL experts

One folder per RL expert (the domain teachers trained in parallel), all with the same layout:

```
experts/
  main.yaml          the run every expert shares: resources (tier, time limit, context), checkpoints, rollout
                     dump, validation period
  <expert>/
    base.yaml        the expert's official run: its datasets, environments, validation, recipe and steps;
                     tier, time limit and context come from main.yaml
    dev/             variants in progress (`defaults: ../base.yaml` plus what they change)
    archive/         the task files of runs that happened, as they ran: the record to reproduce a run
    scripts/         data builders and checks of the expert (not read by a run)
```

```bash
bash run.sh stem                           # the official STEM run: recipe, main.yaml, stem/base.yaml
bash run.sh experts/stem/dev/<variant>.yaml   # a variant: recipe, main.yaml, stem/base.yaml, the variant
bash run.sh experts/stem/archive/<run>.yaml   # a past run again
```

`run.sh` puts `main.yaml` between the recipe and the expert's files, so `base.yaml` and dev files override any
of it. Files in `archive/` written in the older layout (recipe in `defaults`) do not take `main.yaml` and run as
they ran. A variant that wins moves its changes into `base.yaml`; when a run is done, its task file goes to
`archive/` with the job and the command in its header.

| Expert | Folder | Status |
|---|---|---|
| Medical | `medical/` | template: medical multiple choice |
| Safety | `safety/` | template: prompts graded by the shared LLM judge |
| General chat, helpfulness, instruction following | `chat/` | template on the archived rubric-judge runs (UltraFeedback, one to three judges, AlpacaEval validation) |
| STEM: math, reasoning, physics, chemistry | `stem/` | `base.yaml`: the 8-dataset mix of run 5; GSM8K, DAPO-Math, Reasoning Gym and chemistry alone in `dev/`; runs 1-5 in `archive/` |
| Code, SWE, terminal, agentic | `code/` | template on the archived runs: competitive coding on Gym's code_gen (sandbox node or OpenSandbox pods), Harbor terminal/SWE tasks |
| Knowledge, factuality, QA | `knowledge/` | template: open QA (judge) and multiple choice; the STEM mix already trains science multiple choice and open science QA |
| Tool use, web search | `tool-use/` | template on the archived workplace-assistant run (27 simulated tools) |
| Translation | `translation/` | template: translations graded by the shared LLM judge |
| Lean (formal proofs) | `lean/` | template on the archived Lean 4 run (Gym math_formal_lean) |

A template carries `template: true`, which `run.sh` refuses to submit (`DRY_RUN=1` shows it). The templates of
chat, code, tool-use and lean take their latest archived run through `defaults` until their `base.yaml` is written.

What several experts take in lives in `shared/` at the top of the repository:
- `shared/frames/`: the system frame every prompt of the Apertus 2 message format starts with, as small JSON files:
  the thinking level (`{"thinking":"high","tools":[]}`) and the tools the model may call (ScienceWorld's `act`). A
  dataset picks one with `system_prompt_file`; in `base.yaml` through `frames.<key>`.
- `shared/judges/llm-judge.yaml` (the grader) and `shared/judges/servers/<model>.yaml` (the judge model's server):
  an expert takes both in `defaults:`, gives the server its nodes (`env.extra_judge_pools.<model>.num_nodes`) and points the grader at it (`env.llm_judge.judge_pool: judge_<model>`) in its own base.yaml. The server file lays out one vLLM replica (TP x PP GPUs); run.sh checks that the nodes hold whole replicas.
- `shared/prompts/`: few-shot prompts read at run time (GSM8K five-shot, STEM four-shot).
- `shared/scripts/`: `stage_judge_model.sh` (downloads the judge model), `strip_few_shot.py` (removes the few-shot
  prompt from prepared Gym rows); `shared/debug/score-text.yaml`, a log-probability debug task.

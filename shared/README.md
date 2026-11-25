# shared

Pieces that several experts take in. A piece only one expert uses goes into that expert's folder instead
(for example `experts/stem/frames/`).

| Folder | Contents |
|---|---|
| `frames/` | Apertus 2 system frames (thinking level, tools); a dataset names one as its `system_prompt_file` |
| `judges/` | The LLM judge: `llm-judge.yaml` (the grader), `servers/<model>.yaml` (the judge model's vLLM server), `judge_stats.py` (load of each replica, after a run) |
| `prompts/` | Few-shot prompts (GSM8K, the STEM mix) |
| `scripts/` | Judge model staging, checkpoint conversion, few-shot stripping; `analysis/` (rollouts, timings), `network/` (network tests), `sandbox/` (OpenSandbox pod probes) |
| `debug/` | `score-text.yaml`: the trainer's log-probabilities of plain text, not a training task |
| `sft-checkpoint/` | How the SFT checkpoint export was made (its README) |

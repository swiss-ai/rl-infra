# STEM expert

Math, science multiple choice and open answers, Reasoning Gym, Enigmata puzzles, ScienceWorld and chemistry, trained
as one mix. `bash run.sh stem` runs `base.yaml`.

| Path | Contents |
|---|---|
| `base.yaml` | The official run: each dataset with its environment and prompts per step, validation, the LLM judge for open answers, the frames |
| `dev/` | Variants and single-environment tasks (GSM8K, DAPO math, chemistry, Reasoning Gym) |
| `archive/` | The STEM mix runs 1 to 5 as they ran |
| `frames/` | ScienceWorld's system frames (they declare its `act` tool) |
| `scripts/` | Data builders (Reasoning Gym, chemistry), `enigmata/` and `scienceworld/` (install, data, tests), `eval/` (validation of an export), `check_rg_verifier.sh` (Reasoning Gym verifier test) |

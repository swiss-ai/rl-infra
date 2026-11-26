# Multi-turn educational learner simulation (`experts/chat/scripts/educational/`)

Data builders and prompt templates for multi-turn GRPO training of the Apertus 2 chat expert
against simulated English-speaking learners (v1, adapted from NVIDIA NeMo UserSim's
`general_educational` probe).

## Overview

Each training episode simulates an up-to-8-turn conversation between the policy model (with no
special tutoring system prompt) and a persona-grounded learner role-played by Gemma 4:
- **Topics:** math problems from [`dmacjam/eduardoRL-transfer-dataset`](https://huggingface.co/datasets/dmacjam/eduardoRL-transfer-dataset)
  (~70%) and open educational subjects from UserSim (`subjects_en_US.json`, ~30%), paired with
  personas from `nvidia/Nemotron-Personas-USA` and OCEAN behavioral profiles (`behavioral.py`).
- **Environment:** native NeMo-RL multi-turn environment (`env_name: tutoring`, patch `0068`),
  with learner quality gating, early stop when the learner is satisfied (from turn 3),
  context-margin protection, and whole-dialogue judging on `helpfulness`, `accuracy`,
  `coherence`, and `pedagogical_quality`.
- **Language:** English (`en_US`) only in v1.

## Quickstart

1. **Build decontaminated specs:**
   ```bash
   python3.12 experts/chat/scripts/educational/prepare_educational.py \
       ${RL_DATA_DIR}/chat-rl/educational \
       --sft-data-dir ../sft-data-apertus2 \
       --benchmarks-root /path/to/benchmarks_apertus2
   ```
2. **Precompute gated opening user messages** (against a served Gemma 4 endpoint):
   ```bash
   python3.12 experts/chat/scripts/educational/make_openers.py \
       ${RL_DATA_DIR}/chat-rl/educational/train_specs.jsonl \
       ${RL_DATA_DIR}/chat-rl/educational/train.jsonl \
       --base-url http://<host>:8000/v1 \
       --model ${SCRATCH}/models/gemma-4-26B-A4B-it
   python3.12 experts/chat/scripts/educational/make_openers.py \
       ${RL_DATA_DIR}/chat-rl/educational/val_specs.jsonl \
       ${RL_DATA_DIR}/chat-rl/educational/val.jsonl \
       --base-url http://<host>:8000/v1 \
       --model ${SCRATCH}/models/gemma-4-26B-A4B-it
   ```
3. **Run training:**
   ```bash
   DRY_RUN=1 bash run.sh experts/chat/dev/educational-v1.yaml
   DEBUG=1   bash run.sh experts/chat/dev/educational-v1.yaml
   bash run.sh experts/chat/dev/educational-v1.yaml
   ```

## Attribution

`behavioral.py`, `subjects_en_US.json`, and `prompts/` are adapted from
[NVIDIA NeMo UserSim](https://github.com/NVIDIA-NeMo/UserSim) (`general_educational` probe,
`engine/core/behavioral.py`, `engine/core/simulation.py`, `engine/evaluator/axes.py`,
`engine/evaluator/prompts.py`), licensed under **Apache-2.0**.

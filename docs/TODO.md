# Ideas for later

Not planned or supported yet; written down so they are not lost.

- **Thinking level per prompt instead of per dataset** (2026-11-23). A system frame is a file per thinking level and
  tool set (`shared/frames/apertus2-system*.json`, `experts/stem/frames/`), because NeMo-RL takes a system prompt
  only as a file (`system_prompt_file`). Varying the reasoning level dynamically (per prompt, by a schedule, or as a
  curriculum) cannot be expressed with files. It would need a NeMo-RL patch to the Apertus 2 data processor: the task
  states `thinking: [low, medium, high]` (with a rule for choosing) and the tools, and the processor builds each
  prompt's frame; the frame files would then go away.
- **Adam weight decay on a Muon-trained model**: see "To consider" in `recipes/README.md`.

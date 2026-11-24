# Model-free speculative decoding for Chonk's Lean rollouts: offline estimate

`spec_sim.py` replays real rollouts. At temperature 1 with exact speculative sampling, a fixed draft token is accepted
with the target's probability of that token. That is also the chance that a real rollout's next token equals the draft,
so the match rate on real rollouts estimates the acceptance rate without bias, and without a model.

The run used lean-sampling `goedel-v02-t10` shard 0: 100 Goedel statements × 8 rollouts, T 1.0, 16k, 7.6 M generated
tokens, prompts rendered as in training. It ran on the Clariden login node on 2026-11-24. N-grams of 6, 4 and 3 tokens,
longest first, most recent occurrence.

| draft source | k=3 | k=5 | k=8 |
|---|---|---|---|
| prompt + own text (vLLM's `ngram` method) | 1.75 | 1.88 | 1.98 |
| + the group's other rollouts up to the same position | 1.94 | 2.12 | 2.26 |
| + full earlier rollouts of the same prompt (a revisit) | 2.04 | 2.26 | 2.42 |

Numbers are tokens per model forward pass. Rollouts cut at 16k (27 % of tokens) behave the same: 1.79 / 1.94 / 2.04 with
`self`. Raw results are in `results-goedel-t10.jsonl`.

This is an upper bound on the speed-up. A verification pass carries up to k+1 tokens per sequence, and costs more than a
plain decode step by an amount stage 1's concurrency sweep measures.

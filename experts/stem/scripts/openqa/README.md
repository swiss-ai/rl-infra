# Nemotron-RL-knowledge-openqa with an LLM judge: integration and judge scaling

Running log (newest last). Work dir: /scratch/hanint06/rl-analysis/openqa-judge. Code: NeMo-RL copy
~/projects/nemo-rl-apertus2-openqa (branch apertus2-openqa), rl-infra worktree ~/projects/rl-infra-openqa (branch
openqa-judge).

## 1. Data (job 1275473, 2026-11-14)

- prep_openqa.py (NeMo-RL container) -> /scratch/hanint06/data/stem-rl/openqa/meta.jsonl (122,389 rows: prompt,
  question, answer, answer kind, rule-checkable flag, Qwen3-235B and gpt-oss-120b pass rates).
- label_openqa.py: gemma-4-26B-A4B-it domain labels, same prompt and labels as the knowledge-mcqa labelling,
  4 GPUs, 150-165 s per quarter.
- Domains over all 122,389 rows: **law 62,495 (51%)**, physics 17,522, biology 13,962, math 8,258, chemistry 7,313,
  medicine 6,596, engineering 2,526, social science 1,955, earth science 746, economics 546, CS 295, humanities 141.
- finalize_openqa.py (hangang-dm): drop Qwen3-235B pass rate 1.0 (45,186 rows), keep science + CS (no medicine,
  law, humanities, social science): **38,351 rows** in openqa_science.jsonl (physics 14,214, biology 9,628, math 6,256,
  chemistry 5,614, engineering 1,908, earth 509, CS 222). Answer kinds: sentence 16,654, expression 10,318,
  short phrase 6,710, number 4,669. Rule-checkable gold (number/expression math-verify parses): **14,987** (openqa_gold.jsonl).
- gen_sample.jsonl: 1,000 non-gold + 500 gold questions for the candidate answers.

## 2. Integration path (decided 2026-11-14)

Native environment, not Gym. In NeMo-RL, `env.should_use_nemo_gym` is global: with Gym on, every rollout of the run
goes through Gym (`trajectory_collector`), so the Gym `equivalence_llm_judge` server cannot run next to the native
`math` and `math_default` environments of the run-3 mix, and moving math and MCQ to Gym would change their grading
(Gym grades the reply, the native environments the claim). The judge pool (patch 0019) already serves the judge
inside the job; it only lacked a native consumer. Commit 0d753f83d (branch apertus2-openqa):
- `nemo_rl/environments/llm_judge_environment.py`: Ray async actor `LLMJudgeEnvironment`, env name `llm_judge`.
  Grades the claim frame (or its last `\boxed{}` content) against the reference; modes `verdict` (one token,
  reward P(Yes) or P(Yes) >= 0.5) and `justify` (NVIDIA Gym equivalence prompt, first `[[A=B]]`/`[[A!=B]]` decides);
  aiohttp calls round-robin over the pool replicas, `max_concurrent_calls`, retries, 0 for failures,
  `[llm_judge]` log lines (calls, failures, latency, in flight).
- `judge_pool.py`: publishes each pool's URLs and model in Ray internal KV (`judge_pool_address(name)`); no Gym
  config needed. `grpo.py`: native runs start the judge pools after the trainer and inference clusters.
- rl-infra (branch openqa-judge): `envs/stem-mix-openqa-judge.yaml`, recipes `...-32n-judge{1,2,4}.yaml`,
  `experiments/openqa-judge/launch.sh` (variant-B 512-prompt settings, 8 steps, validation and checkpoints off).

## 3. Standalone judge throughput (judge_bench.py; one node; same judge pool code and calls as llm_judge)

Synthetic sanity set (triples_synth.jsonl: reference vs itself, reference vs another row's reference), job 1275490,
gemma-4-26B-A4B-it, 4 replicas (TP1), pool start 204 s:

| mode | concurrency | calls/s per node | p50 / p95 latency (s) | prompt / output tokens | identical -> yes | unrelated -> yes |
|---|---|---|---|---|---|---|
| verdict | 64 | 269 | 0.08 / 0.94 | 359 / 1 | 750/750 | 0/748 |
| verdict | 1024 | 1,149 | 0.64 / 1.57 | 359 / 1 | 1500/1500 | 2/1497 |
| justify (greedy) | 64 | 91 | 0.53 / 1.03 | 636 / 45 | 400/400 | 0/399 |
| justify (greedy) | 1024 | 277 | 2.04 / 3.82 | 636 / 45 | 1500/1500 | 1/1497 |

Calls of one prompt share the system text, question and reference, so prefix caching helps as it will in a run.
A run step needs 2,048 calls (128 openqa prompts x 16): about 2-8 s of one gemma node in verdict mode, 7-10 s in
justify mode, against a step of about 300 s.

## 4. Candidate answers (job 1275487)

SFT v0.2, thinking medium (as run 3), BF16 vLLM on one node, 1,500 gen_sample questions x 2 = 3,000 answers to the
dataset's own prompt ("... Put your answer inside \boxed{}"): mean 6,329 answer tokens (median 3,773), 16.0% cut
off at 16k, **69.2% without a claim frame** (57% with an empty think frame; the model answers in the reply and
boxes the answer there). The math environment's claim-frame reward is therefore 0 for most answers (accuracy 1.6%).
Answer for the judge (build_triples.py): last `\boxed{}` of the claim, else the claim text; without a claim, the
last `\boxed{}` of the reply. Sources: claim 924, reply box 1,192, none 884 (cut off or no box): 2,116 triples with
an answer, 666 of them rule-checkable (math-verify against the reference gives the truth).
This changed the environment: `fallback_frame: reply` (commit 32aeba548). The first debug run (job 1275526,
claim only) was cancelled after step 1 (627 s, the first step includes compilation; judge calls in flight 3-5,
p95 latency 2.7 s, 4.5% of calls without a verdict label).

## 5. Judges and modes on real answers (jobs 1275730 gemma, 1275731 Qwen, 1275765 Nemotron; one node each)

2,066 triples (2,116 with an answer; the benchmark's 3,000-call cap and skip of empty answers leave 2,066 in
`triples_used.jsonl`), 666 of them rule-checkable. Throughput at 1,024 calls in flight (calls/s per node; mean
judge output tokens in brackets for justify):

| judge | replicas per node | verdict calls/s (p95 latency) | justify calls/s (p95 latency, output tokens) | justify without a label |
|---|---|---|---|---|
| gemma-4-26B-A4B-it (BF16, TP1) | 4 | **622** (1.7 s) | 144 (8.7 s, 132) | 3.3% |
| Qwen3.8-27B-FP8 (TP1) | 4 | 160 (6.9 s) | 34 (52 s, 271) | **24.5%** (all hit max_tokens 512) |
| Nemotron-3-Super-120B-A12B-FP8 (TP4; TP2 ran out of memory at start, job 1275732) | 1 | 121 (8.3 s) | 17 (72 s, 177) | 6.1% |

Quality (hard verdict at 0.5; `quality.json`, `disagreements.txt`):

| judge-mode | verdict coverage | yes rate | agreement with math-verify on rule-checkable (n) | agreement with the majority of the 6 |
|---|---|---|---|---|
| gemma-verdict | 95.0% (min mass 0.5) | 22.3% | 90.9% (594) | **99.0%** |
| gemma-justify | 96.7% | 25.5% | 89.8% (608) | 95.6% |
| Qwen-verdict | 100% | 17.2% | 90.1% (666) | 96.2% |
| Qwen-justify | 75.5% | 27.0% | 86.0% (350) | 98.0% |
| Nemotron-verdict | 100% | 16.6% | **92.2%** (666) | 96.0% |
| Nemotron-justify | 93.9% | 26.8% | 91.1% (561) | 94.1% |

math-verify is not a clean gold here. Hand-read: of 22 random "gemma yes, math-verify no" rule-checkable cases,
about 18 are correct answers math-verify rejects because of units or notation (`20` vs `20 m s⁻¹`, `\dfrac14` vs `25%`,
`57` vs `57°C`, `B_{13}=\overline{B_{31}}` vs `B_{13}=B_{31}^*`); judge errors were `88` vs `88.8 mg` and rounding
cases. All 11 "gemma no, math-verify yes" cases are math-verify false positives: it accepts `2.10\times10^{-20}` for
`2.10\times10^{-19}` and `3.3\times10^{-22}` for `3.3\times10^{-21}` s (tiny numbers compare equal), a swapped pair
(`s(1,0)=1, s(0,1)=0` vs `s(1,0)=0, s(0,1)=1`), sign-flipped and incomplete expressions. (This math-verify behaviour
also affects the math environment: see open issues.)
Hand-read of 30 random disagreements among the six (sentence and phrase answers mostly): gemma-verdict was right on
about 21 of 24 clear cases, gemma-justify on about 13 of 25 (it accepts loosely related answers: method vs method,
"free ER membrane" vs "ER membrane with the signal sequence embedded"), Qwen-verdict on about 10 of 24 (it rejects
correct paraphrases). Justify modes say yes more often (25-27% vs 17-22%) and are 4-7x slower.
Gemma's 5% verdict calls without enough Yes/No mass start to explain ("To", "The") with No usually in its top 3;
`min_verdict_mass: 0` (commit 0014ea786) reads the verdict from the Yes/No tokens among the top 20.

## 6. In the training run (16 trainer + 16 vLLM + judge nodes, 512 prompts per step, variant B)

Task `envs/stem-mix-openqa-judge.yaml` (128 DAPO + 128 NuminaMath + 128 science MCQ + 128 openqa per step),
SFT v0.2, thinking medium, gemma-4-26B-A4B-it on one judge node (4 replicas), `analyze_debug.py`.

**Justify mode, reply fallback (job 1275648, 8 steps, 1 h 19 min):**
- Step time 652 s (step 1, with compilation), then 295, 310, 300, 304, 315, 303, 276 s (median 303 s). Run 3
  (same settings, no openqa, no judge) ran 200-330 s per step on the same nodes, so the judge adds no visible time.
- Judge pool: requests waiting never above 0 (3,400 samples of the replicas' vLLM stats); at most 12 requests
  running per replica; KV cache at most 3% used; requests running in 51% of samples. llm_judge in flight at most 35.
- Judge calls: 13,824 in 8 steps (about 1,730 per step: 84% of the 2,048 openqa answers have an answer to grade),
  latency p50 0.6-1.0 s, p95 2.1-4.4 s, against a step of about 300 s. 533 calls (3.9%) had no [[A=B]]/[[A!=B]]
  label (scored 0; old code retried them identically); no HTTP failures.
- Buffer starvation only at steps 6 and 7 (36 s, 18 s), with exposed generation 54 and 38 s there: generation
  tail, not the judge (whose calls take seconds).
- Training reward per source over steps 1-8: DAPO 0.68 -> 0.79, NuminaMath 0.22 -> 0.32, MCQ 0.37 -> 0.40,
  **openqa 0.18 -> 0.16** (justify, flat over 8 steps). Cut off at 16k: 11-12%.

**Verdict mode, unconstrained (job 1276181, 6 steps, commit 0014ea786: min_verdict_mass 0, no retry):**
- Step time 641 s (step 1), then 316, 317, 305, 310, 274 s (median 310 s): the same as justify and as run 3.
- Judge calls: 10,240 in 6 steps, latency p50 0.02 s, p95 0.03-0.06 s (one prompt token batch, one output token);
  the pool's vLLM stats, sampled every 5 s, never caught a request running.
- **Calls without a verdict: 1,335 of 10,240 (13.0%)**, all `no_verdict` (no Yes or No among gemma's top 20
  first tokens; no HTTP error or timeout). Offline on the benchmark triples the same prompt gave 5% below 0.5 mass;
  in the run these answers scored reward 0, depressing the openqa reward.
- Training reward: DAPO 0.67 -> 0.79, NuminaMath 0.22 -> 0.29, MCQ 0.37 -> 0.37, **openqa 0.10-0.12** (stricter
  than justify's 0.13-0.18, as offline: yes rate 22% vs 26%).

**Verdict failures in 1276181, by cause.** All 1,335 counted failures were judge calls whose answer had neither
Yes nor No among gemma's top 20 first tokens (`no_verdict`): 0 `[llm_judge] call failed` lines (no HTTP error,
timeout or parse error). Answers with nothing to judge (no claim and no boxed reply) were not sent and not counted
as failures in that version; they scored 0 as intended. Offline the same prompt left 5% of calls below 0.5 Yes/No
mass, so the run's answers (more unboxed claim text) push gemma towards explaining first.

**Format-safe verdicts and zero weight for judge failures (commit cc7c17d2b, task commit 097ccce):**
- `structured: true`: vLLM structured outputs (`"structured_outputs": {"choice": ["Yes", "No"]}`, max_tokens 4)
  for verdict; for justify a JSON schema `{"reasoning": string, maxLength max_reasoning_chars (400), "verdict":
  "A=B" | "A!=B"}` with its own prompt (`JUSTIFY_JSON_PROMPT`).
- Logprobs under structured outputs are the **masked** distribution: in the image's vLLM 0.28
  (`v1/worker/gpu_model_runner.py`) `apply_grammar_bitmask` runs on the logits before `self.sampler`, which computes
  raw and processed logprobs from those logits. Only Yes and No stay finite, so P(Yes) is the raw
  P(Yes) / (P(Yes) + P(No)): the same number the unconstrained reading gives when both tokens are in the top 20.
  Confirmed on job 1276557 (`[llm_judge] sample:` lines): the top logprobs hold only grammar-allowed tokens (Yes,
  No and their prefixes N, Y, Ye, e.g. Yes -0.0016, No -6.47, N -11.4, Y -14.6); every other token is -9999.
- Failure causes are counted separately in every `[llm_judge]` line and in the environment metrics: `ok`,
  `no_verdict`, `http`, `timeout`, `other`, and `no_answer` (answers never sent; reward 0, not a failure); the first
  five answers without a verdict are printed.
- `mask_failures: true`: a failed judge call (`no_verdict`, `http`, `timeout`, `other`) sets `mask_sample` in the
  environment metadata. Native rollouts now always carry `mask_sample` (`nemo_rl/experience/rollouts.py`), so GRPO's
  existing `_apply_mask_sample_filter` zeroes their `loss_multiplier` (counted in `num_mask_sample_filtered`, which
  gives the rate) and the advantage estimator gets `valid_mask` without them, keeping them out of their group's
  baseline (`grpo.py`, both GRPO paths). Empty answers stay reward 0 and are trained on.
- The comparison run (job 1276415, same as 1276181 plus these settings) failed in vLLM engine start-up after 3
  minutes with `[Errno 122] Disk quota exceeded` writing the FlashInfer autotune cache under
  `/scratch/hanint06/vllm_cache_*`; a relaunch failed in setup (`ln`/`curl`: `Disk quota exceeded`). The group's
  **file** quota is full: 4,981,691 of 5,000,000 files on /scratch for in0138 (space 66 of 80 TB). Our user holds
  333k files; this task's directories hold under 2,000. The staged v0.2 model directory was checked and is intact
  (51 of 51 shard links, config.json from 07:21). Nothing can start until the group frees files.

**Structured verdicts in the run (job 1276557, 2026-11-14, same shape and steps as 1276181; group files freed to
4.86M of 5M):**

| | 1276181 unconstrained | 1276557 structured + mask_failures |
|---|---|---|
| judge calls in 6 steps | 10,240 | 10,752 |
| failed calls | **1,335 (13.0%)**, all no_verdict | **0** (no_verdict 0, http 0, timeout 0, other 0) |
| answers with nothing to judge (no_answer, reward 0) | not counted | 4,298 of 15,050 openqa answers (28.6%) |
| num_mask_sample_filtered per step | (no masking) | 0, 0, 0, 0, 0, 0 |
| judge latency p50 / p95 | 0.02 / 0.03-0.06 s | 0.03 / 0.03 s (0.15 s once) |
| step time, steps 1-6 | 641, 316, 317, 305, 310, 274 s | 651, 313, 319, 296, 308, 322 s |
| openqa training reward, steps 1-6 | 0.117, 0.104, 0.115, 0.097, 0.094, 0.100 | 0.120, 0.101, 0.125, 0.101, 0.086, 0.109 |
| other sources at step 6 | DAPO 0.79, NuminaMath 0.29, MCQ 0.37 | DAPO 0.78, NuminaMath 0.31, MCQ 0.35 |

Choice decoding removes the judge failures at no cost in latency or step time. The openqa reward is unchanged: the
failed calls in 1276181 were mostly "No" verdicts in disguise (No among gemma's top 3 first tokens). The masking
path ran (the `mask_sample` field flowed through rollouts, the filter and the advantage estimator without errors)
but had nothing to mask; a failure of a judge call would now give its sample zero weight.

## 7. Recommendation

- **Judge: gemma-4-26B-A4B-it, verdict mode, one judge node**, hard score (reward 1 if P(Yes) >= 0.5),
  `structured: true` (0 failed calls in 10,752, job 1276557) and `mask_failures: true`. It is the fastest judge (622
  verdict calls/s per node), agrees best with the majority of the six judge-mode combinations (99.0%), and its
  apparent errors against math-verify are mostly math-verify's. Nemotron-3-Super (verdict) is the stricter,
  5x slower alternative with slightly higher agreement with math-verify (92.2% vs 90.9%). Avoid Qwen3.8 justify
  (24% of answers cut at 512 tokens without a label) and gemma justify (too lenient on sentence answers).
- **Judge nodes: one is enough.** With 128 openqa prompts of 512 per step (about 1,700 calls per step, one every
  0.2 s on average), the pool's queue stayed empty, latency was 0.02 s (verdict) and 1 s (justify) against a 300 s
  step, and step time matched run 3 without a judge. One gemma node serves 144 justify or 622 verdict calls/s, so
  the 2- and 4-node runs were not needed; Nemotron justify (17 calls/s per node, 72 s p95 latency at full load) is
  the only setup here that could need a second node.
- **Answer extraction:** grade the claim frame's last `\boxed{}`, else the reply's last `\boxed{}`
  (`fallback_frame: reply`); SFT v0.2 at thinking medium leaves out the claim frame on 69% of these answers.

## 8. Open issues

1. The zero-weight path has run but not yet masked anything (no judge failures with structured outputs); its first
   real use will show in `train/num_mask_sample_filtered`.
2. math-verify (also the math environment's checker) treats very small numbers as equal: it accepted
   `2.10\times10^{-20}` for `2.10\times10^{-19}` and `3.3\times10^{-22}` for `3.3\times10^{-21}`; it also accepted a
   swapped pair and incomplete expressions. Worth a look in the math environment (NuminaMath has such answers).
3. openqa training reward is low (0.10-0.18) and flat over 8 steps; 16% of answers are cut at 16k and 29% have no
   gradable answer at thinking medium. Thinking high (v0.2 GPQA 48 vs 33) may fit these questions better.
4. The openqa references are one of several correct answers for sentence questions; judge disagreements cluster
   there (hand-read in section 5).
5. The validation set has no openqa part; a held-out science openqa slice graded by the same judge would show
   progress.

## Commits (not pushed)

- NeMo-RL `~/projects/nemo-rl-apertus2-openqa`, branch `apertus2-openqa` (from `apertus2-stem` 2795b7423):
  0d753f83d llm_judge environment + native judge pool; 32aeba548 reply fallback; 0014ea786 min_verdict_mass, no
  futile retries; cc7c17d2b structured verdicts, failure causes, zero weight for failed calls.
- rl-infra `~/projects/rl-infra-openqa`, branch `openqa-judge` (from `dapo-dynamic-sampling` e748364): 84f701b run-3
  recipe and task; 930a1c5 openqa task, judge-node recipes, launcher; b05b98a fallback and min mass; 097ccce structured
  and masking settings.
- Scripts and data here and in /scratch/hanint06/data/stem-rl/openqa (prep_openqa.py, label_openqa.py,
  finalize_openqa.py, gen_score.py, build_triples.py, judge_bench.py, analyze_quality.py, analyze_debug.py, cfg/).

Jobs: 1275473 data, 1275487 candidates, 1275490 gemma synthetic, 1275730/1275731/1275765 (1275732 OOM at TP2)
benchmarks, 1275526 (cancelled), 1275648 justify, 1276181 verdict, 1276415 structured (failed: file quota),
1276557 structured verdict (completed, 1 h 12 min).

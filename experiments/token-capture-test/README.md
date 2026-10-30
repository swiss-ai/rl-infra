# Token-capture test: can legacy GRPO train on a harness without a token wrapper?

NeMo-RL's legacy GRPO entrypoint (`examples/nemo_gym/run_grpo_nemo_gym.py`, used by
`run-harbor-osb.sh`) takes the training tokens only from the output items an agent's `/run`
returns: `_postprocess_nemo_gym_to_nemo_rl_result` in `nemo_rl/environments/nemo_gym.py`. Harbor's
Terminus-2 wrapper puts them there; most other Gym agents do not. Gym can record them at the
model server instead (`nemo_gym/token_id_capture/`), but in NeMo-RL only the Single-Controller
path consumes that record. This run checks the first half on the cluster: one GRPO step with
Gym's `mini_swe_agent_2`, whose wrapper drops the token ids
(`responses_api_agents/mini_swe_agent_2/app.py`, `_split_trajectory_for_responses`).

mini-swe-agent's loop runs in the Gym process on the training node and runs every bash action
in an OpenSandbox pod (`sandbox_environment.py`), so, like Terminus-2, it needs only node ->
OpenSandbox traffic.

| File | What |
|---|---|
| `qwen2.5-1.5b-1n.yaml` | recipe: Qwen2.5-1.5B-Instruct, 1 node, Megatron + colocated vLLM, sync GRPO, Hermes tool parser |
| `mini-swe-osb.yaml` | task file: mini_swe_agent_2 on OpenSandbox, 2 prompts x 2 generations, 1 step, `step_limit: 10` |
| `make_rows.py` | 2 train + 1 validation rows from Gym's `mini_swe_agent_2/data/example.jsonl` (SWE-bench Verified) |
| `run.sh` | builds the rows in `$HARBOR_ROOT` (default `$SCRATCH/osb/mini_swe`) and calls `run-harbor-osb.sh` |

## Run

Same prerequisites as `run-harbor-osb.sh` (setup done, Gym at `apertus2-osb`,
`~/.opensandbox_token`). The model is downloaded to `$SCRATCH/huggingface` on first use; the task
images are public `docker.io/swebench/sweb.eval.x86_64.*` images.

    cd ~/rl-infra
    bash experiments/token-capture-test/run.sh            # PARTITION, JOB_TIME (default 01:00:00), RUN

## Reading the result

    LOG=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
    grep -n "no generation data\|response.output item types\|Masking a degenerate\|Error running mini-swe-agent\|\[EVAL\]" $LOG

| What the log shows | Meaning |
|---|---|
| `ValueError: NeMo Gym returned a result with no generation data` with a non-empty `response.output item types (N items): [...]` | **Expected.** The agent ran (model calls, tool calls), but the result carries no token ids, and legacy GRPO cannot train on it. A harness without a token wrapper needs Single-Controller with token capture, or a wrapper of its own. |
| The step finishes and the training metrics show tokens for these samples | Legacy GRPO got token ids from this agent; no switch is needed for it. Not expected from the code. |
| `0 items`, `Error running mini-swe-agent`, or no `[EVAL]` lines | Inconclusive: the rollout failed before any model call (sandbox create, image pull, agent venv). Fix that and rerun. |

Per-rollout files: mini_swe_agent_2 writes them under the Gym server's working directory, which
is the clone mount `/opt/nemo-rl` (= `$NEMO_RL_DIR`, gitignored `results/`), in
`results/verified/Qwen/Qwen2.5-1.5B-Instruct/<instance_id>/`: the mini-swe-agent trajectory
(`<instance_id>_<run_id>.traj.json`, every model reply with the full response in `extra.response`),
the patch, the test output, the SWE-bench report, and `<instance_id>.json` (the `/run` result; the
two generations of an instance overwrite each other's). A token id field in `extra.response`
(e.g. `prompt_token_ids`) shows the ids reached the agent and were dropped by its wrapper.

Sandboxes carry `nemo-gym.nvidia.com/workload=token-capture-test` and the run id; reap them as in
`docs/harbor-osb-chonk-run.md`.

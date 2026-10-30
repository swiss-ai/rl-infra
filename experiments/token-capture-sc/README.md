# Token-capture test, Single-Controller variant

Can NeMo-RL train on a harness whose Gym wrapper carries no token ids, when the token ids are
captured at the model endpoint instead? Same rollouts as `../token-capture-test` (Gym's
`mini_swe_agent_2` on SWE-bench Verified, every task container an OpenSandbox pod, Qwen3-4B,
2 prompts x 2 generations, 1 step), but through `examples/run_grpo_single_controller.py` with
`token_capture.enabled: true`.

The legacy test (job 5212512, 2026-10-30) showed that legacy GRPO cannot: Gym's model server
returned the token ids on the assistant message, litellm moved them into
`provider_specific_fields`, and on the second turn the model server rejected the resent message
(HTTP 422), so no rollout finished. With capture on, NeMo-RL switches that echo off
(`return_token_id_information: false`) and the vLLM worker's HTTP server records every call per
rollout instead; the trajectory is rebuilt from those records. Nothing in the harness changes.

| File | What |
|---|---|
| `qwen3-4b-1n-sc.yaml` | recipe: `../token-capture-test/qwen3-4b-1n.yaml` with vLLM on 2 of the 4 GPUs (Single-Controller rejects colocated vLLM) |
| `mini-swe-osb-sc.yaml` | task: `../token-capture-test/mini-swe-osb.yaml` plus data plane, `async_rl` (synchronous), token capture |
| `run.sh` | sets TASK, RECIPE, SCRIPT and calls `../token-capture-test/run.sh` (rows, litellm pin, `run-harbor-osb.sh`) |

## Before the first run

Single-Controller needs the TransferQueue package (NeMo-RL pins it in `pyproject.toml`). Whether
the Apertus 2 image has it is not checked yet:

    srun -A csstaff -p debug -N1 -t 10 --environment=nemo-rl-apertus2 \
        python3 -c "import transfer_queue, tensordict; print(transfer_queue.__file__)"

If that fails, the package has to go into the runtime overlay (`backends/nemo-rl/apertus2/pydeps.sh`).

## Run

    cd ~/rl-infra
    bash experiments/token-capture-sc/run.sh        # PARTITION, JOB_TIME (default 01:00:00), RUN

## Reading the result

    LOG=$SCRATCH/nemo-rl/slurm/<jobid>-logs/ray-driver.log
    grep -n "SC run complete\|Error running mini-swe-agent\|422 Unprocessable\|token_capture\|Traceback" $LOG
    ls $SCRATCH/nemo-rl/logs/<RUN>/exp_001/gym_token_capture/      # per-rollout capture ledgers

| What the log shows | Meaning |
|---|---|
| `SC run complete`, a training step with `train/finalize/total_ms > 0` in TensorBoard | **Yes**: captured tokens reached the trainer; endpoint capture works for a harness without a token wrapper. |
| `422 Unprocessable Entity` again | The token echo was not switched off; check that the policy_model override from `token_capture` reached Gym. |
| An error in `RolloutReassembler` / finalize, or a lineage error | Capture ran but the rebuild failed; the ledgers under `gym_token_capture/` show which call. |
| `import transfer_queue` fails, or setup raises on a config key | Stack problem before any rollout; see above. |

Per-rollout files of the agent: as in `../token-capture-test/README.md` (under
`.../mini_swe_agent_2/results/verified/Qwen/Qwen3-4B-Instruct-2507/`).

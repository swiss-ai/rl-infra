# Safety Gym Chonk training

Edit the settings at the top of `resume.sh`, then run it with no arguments from
a GPU login node:

```bash
bash /home01/hanint14/safety-gym/rl-infra/experiments/safety-gym/resume.sh
```

It refuses to submit while a job with the same run name is active. Otherwise it checks the selected judge and native config, then submits one 12-node job
(8 trainer nodes and 4 rollout nodes). The prior 16k training attempt was Slurm job `1292775`; its OSS judge is job
`1292749` at `http://node1878:8080/v1` with a 32k context. The defaults use
that judge and run
`msafetemp-chonk-16k`, validate on 500 held-out multilingual examples, and request
72 hours with a checkpoint deadline 30 minutes before the wall limit. The judge and training chunks each request 72 hours. Keep the judge available
through queue wait and the full training allocation. The readiness check
confirms the endpoint now, not its future lifetime.

Each invocation submits one job. If it ends before training reaches step 4,553,
make sure the latest checkpoint finished saving, then run the same script again
with the same `RUN`, `OUT`, dataset source, and validation size. The native
launcher uses `$OUT/results/$RUN`; NeMo-RL restores the latest checkpoint,
including model optimizer state, training step, and StatefulDataLoader position.
Do not resubmit after step 4,553 is complete. For a separate Nemotron run, edit
`JUDGE_KIND`, `JUDGE_URL`, and `RUN` at the top and keep those values fixed on
later resumes.

W&B logging targets [the existing safety-gym project](https://forge.coreweave.com/wandb/nutemmax-epfl/safety-gym?nw=nwusernutemmax) under `nutemmax-epfl`. The wrapper reads the private key from
`$OUT/private/wandb.key`; the native launcher keeps it out of command lines
and logs.

## Routes and validation

Dataset rows carry native `agent_ref` routes: harmful rows use
`jailbreak_detection_simple_agent`; benign rows use
`over_refusal_detection_simple_agent`. The selected judge is configured behind
both Gym judge aliases. Validation preparation checks both routes and links the
original training file. Keep `$OUT/data/validation-500` unchanged across
resumes; set `VALIDATION_SAMPLES=1000` at the top for 1,000 validation examples.

The job prepares the fixed validation view before model initialization. Each
training sample has a 16,384-token total context; the response cap is 2,048
tokens and generation can use only the space remaining after its prompt. Longer
responses take more rollout compute.
There is no full-dataset prompt audit, so an overlength prompt can still fail at
generation time. The wrapper's `--check` only checks judge readiness and the
native static config.

## Recent failures

Job `1292654` failed during its first rollout. Its audit reported a maximum of
2 tokens because it counted the tokenizer result fields, not the token IDs. The
trainer's vLLM instance rejected a prompt of 2,048 tokens plus 1 requested
output token against its 2,048-token limit; NeMo Gym then reported an empty
generation. The broken full-dataset audit hook was removed. The current run
uses a separate checkpoint name and a 16k total context; its 32k OSS judge
leaves room for the policy prompt, safety rubric, and verdict output.

Earlier job `1292474` reached all 48 Ray workers, then exited because its local
driver required `SLURM_JOB_ID`, which was absent inside the container. That
redundant check was removed from the internal driver and data-preparation helper.
Keep Hangang's `--propagate=NONE`: it controls compute
node process limits and fixes Ray worker thread startup; it did not cause either
failure.

## Storage and Slurm

The config checkpoints every 50 steps, retains the latest two, and saves the
optimizer. Rollout JSONL dumps are retained every 50 steps. Metrics and
TensorBoard logging remain enabled. Hangang jobs use the dedicated `intTest`
partition; its time limit is unlimited under the site policy. This launcher
requests a 72-hour chunk as an operator choice and stops training early enough
to save a checkpoint.

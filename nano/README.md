# Nano tasks: GRPO on the Apertus-v1.1 models

Task files for the small dense Apertus-v1.1 models (0.5B, 1.5B, 4B) on one node, kept apart from the Apertus 2 RL
experts in `experts/`. The layout is the experts' one:

```
nano/
  main.yaml          the run every nano task shares: one node, 4K context, 2.5-hour limit, validation every 10 steps,
                     checkpoints
  <task>/
    base.yaml        the task's run: dataset, environment, validation set, recipe and steps
    dev/             variants in progress (`defaults: ../base.yaml` plus what they change)
    prompt.txt, ...  files the task reads
```

```bash
bash run.sh nano/gsm8k policy.model_name=<SFT model folder>        # recipe, nano/main.yaml, nano/gsm8k/base.yaml
bash run.sh nano/gsm8k/dev/<variant>.yaml                          # a variant
```

| Task | Folder | Status |
|---|---|---|
| GSM8K, chain of thought in the chat template, `\boxed{}` answer | `gsm8k/` | runs; results in `docs/apertus-v11.md` |

The recipes are `recipes/apertus-v11.yaml` (synchronous) and `recipes/apertus-v11-async.yaml`; SFT before GRPO is
`sft.sh` (`backends/posttraining`). Development of the nano models stays in this folder, these recipes,
`backends/posttraining/` and `docs/apertus-v11.md`; `experts/` and `shared/` belong to the Apertus 2 experts.

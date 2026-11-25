# backends/nemo-rl

How rl-infra runs a task on NeMo-RL: the swiss-ai NeMo-RL fork at a pinned commit, our patches, and the Apertus 2
container under Slurm. `run.sh` calls everything here; a normal run needs nothing from this folder by hand.

| Path | Runs | Contents |
|---|---|---|
| `launch.sh` | every launch, last | Preflight, container flags, `sbatch ray.sub` with the driver command |
| `env/` | read by every script | `stack.env` (clone paths, image, overlay, outputs), `pin` (commits of NeMo-RL, Megatron, Bridge, Gym, vLLM files), `edf.toml.template` (container image, mounts, environment) |
| `setup/` | every launch, login node | `setup.sh` (clones at the pin, patches), `stage_model.sh` (model on scratch), `setup-gym.sh` (Gym tasks); `download_model.sh` by hand |
| `config/` | every launch, login node | `config.sh` (merge and freeze the yaml chain), `container.sh` (EDF or pyxis flags), `preflight.sh` (checks the merged config and the dataset files) |
| `job/` | inside the job | `driver.sh`, `pydeps.sh` (training with the package overlay); `driver_retry.sh`, `failure_report.sh` (restart from the latest checkpoint, failure summary); `hang_watch.sh`, `numa_watch.sh` (stack dumps on a hang, host memory on GPU memory) |
| `diagnostics/` | by hand | `check_env.sh` (stack test on one GPU node); `check_env_job.sh` submits it, or any `CHECK_SCRIPT`, as a one-node job |
| `patches/` | by `setup/setup.sh` | Our changes to NeMo-RL, Megatron, Gym, Automodel and vLLM (`patches/README.md`) |

To change the code a run uses, add a patch. To change a commit or the image, edit `env/`.

# Patches

Our changes to the repositories a run uses, applied by `../setup/setup.sh` on top of the commits in `../env/pin`.

| Folder | Repository | How |
|---|---|---|
| `ray-sub/` | NeMo-RL's `ray.sub`: Alps container engine, one srun step, memory binding, pyxis flags | `git am`, first |
| `nemo-rl/` | NeMo-RL (swiss-ai fork) | `git am` in number order; an existing clone gets the patches whose subject it lacks |
| `megatron/` | Megatron-LM-MoE; 0007 is KCP context parallelism | `git apply` on the working tree |
| `gym/` | NeMo Gym, for Gym tasks (`../setup/setup-gym.sh`) | `git apply` on the working tree |
| `automodel/` | Automodel | `git apply` on the working tree |
| `vllm/` | the vLLM Apertus 2 model file | `patch` on a copy of the file |

A new NeMo-RL change: commit it in a clone of your own, then export it with the next free number and a subject no
other patch has (`setup.sh` matches patches by subject):

    git -C <clone> format-patch --start-number <next> -o backends/nemo-rl/patches/nemo-rl/ <base>..HEAD

A patch from `format-patch` keeps the commit's author, so an upstream PR made from it is attributed to whoever wrote
it. Some patches in the other folders are plain diffs without an author; new ones should be in git format.

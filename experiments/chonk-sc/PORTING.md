# Using the agent-RL pieces in another setup

`TESTING.md` runs runs V (Terminus-2) and OV (OpenCode) as they are. This page is for changing one dimension:
another model or recipe, another task set, another cluster or sandbox server, or another harness. It says which
pieces of this branch you need and which settings are specific to Apertus 2 Chonk, Daint or Breithorn. We have run
only the combination in `TESTING.md`; everything else here is untested.

The Gym side of in-sandbox agents (adapter options, the per-sandbox model tunnel, token capture) is documented in
Gym's `harbor_agent` README, sections "Agents inside the sandbox (OpenCode, pi, mini-swe-agent)" and "Failure
classes". After `setup-gym.sh` it is at `$NEMO_RL_DIR/3rdparty/Gym-workspace/Gym/responses_api_agents/harbor_agent/README.md`;
it comes from Gym patches 0037, 0041-0043 and 0046-0048.

## The stack

| Layer | Where | How it gets there |
|---|---|---|
| NeMo-RL | clone at `NEMO_RL_COMMIT` (`backends/nemo-rl/apertus2/pin`) + `patches/nemo-rl/*.patch` | `setup.sh` |
| NeMo Gym | clone at `GYM_COMMIT` (same pin, swiss-ai/nemo-gym) + `patches/gym/*.patch` on the working tree | `setup-gym.sh` (`git apply`, in order) |
| Harbor 0.25.0 | Gym's `harbor_agent` requirements | Gym builds the venv at job start |
| Launchers, configs | `run.sh`, `run-harbor-osb.sh`, `experiments/chonk-sc/`, `experiments/harbor-capture-sc/`, `envs/`, `recipes/` | this branch |

Without our launchers, the Gym tree is: `git checkout <GYM_COMMIT>`, then `git apply` every
`backends/nemo-rl/apertus2/patches/gym/*.patch` in order.

### Gym patches by feature

| Feature | Patches | Config keys |
|---|---|---|
| Rubric judge, code-gen (other recipes) | 0001-0005 | |
| OpenSandbox provider: API key file, registry credentials file and host, total create deadline | 0006-0010, 0013, 0031, 0040 | `sandbox.opensandbox.connection.api_key_file`, `sandbox_provider_options.image_auth_file` / `image_auth_registry`, `create.deadline_s` |
| harbor_agent on Harbor 0.25.0 (separate verifier pods, job options, exec user, preflight), local task by name, build timeout multiplier | 0011, 0012, 0032-0034, 0038, 0039 | `harbor_environment_build_timeout_multiplier` |
| Token capture for in-sandbox agents (per-rollout URL, fingerprints, stream replay) | 0014, 0021-0025, 0030 | `harbor_model_base_url` |
| OpenCode adapter | 0015, 0026-0029, 0044, 0045 | `harbor_agent_kwargs.opencode_*` |
| Per-sandbox model tunnel (chisel) | 0016, 0017 | `harbor_environment_kwargs.model_tunnel` |
| mini-swe-agent adapter | 0018-0020 | |
| Terminus-2 truncation handling | 0035, 0036, 0042 | `max_consecutive_truncations`, `truncation_retry`, `context_limit_truncation` |
| Failure class per rollout and masking | 0037, 0041, 0043 (+ 0027) | `harbor_mask_infra`, `harbor_mask_context_overflow`, `harbor_mask_agent_timeout` |
| Documentation of the above | 0046 | |
| pi adapter (uploaded release, capture-safe settings, turn budget, failure classes from `agent/pi.txt`) | 0048 | `harbor_agent_kwargs.pi_*` |

Several are open PRs against swiss-ai/nemo-gym; rl-infra drops a patch once its PR is merged.

### NeMo-RL patches

Most of `patches/nemo-rl/` is Apertus 2 and Chonk support (output format, tool calls, thinking budget, Megatron
and vLLM fixes, memory and NCCL knobs, debugging dumps). Two matter for agent RL with any model:

- **0057**: masked rows stay out of the GRPO group baseline (Gym entrypoint, sync and async). The Single Controller
  path already leaves them out.
- `env.should_mask_flagged_samples: true` (a config key, not a patch): NeMo-RL drops the rows Gym flags with
  `mask_sample` only with it. Without it, masked rollouts train as reward 0.

## How runs V and OV are put together

Configs are OmegaConf files merged in order, later ones win; command-line overrides come last.
`DRY_RUN=1` shows the job without submitting it, and the launchers copy the configs into `$SCRATCH/nemo-rl/configs/`.

**Run V** (`run-v-24k.sh` -> `run-v.sh` -> `run.sh`, NeMo-RL's Gym entrypoint `examples/nemo_gym/run_grpo_nemo_gym.py`):

1. the recipe (`RECIPE`, e.g. `recipes/apertus2-chonk-sftv02-alps-vllm-adamw-tools-20n.yaml`)
2. the task file `envs/harbor-verified-sft.yaml`: tasks, OpenSandbox connection, Terminus-2 options, timeouts,
   masking, registry credentials
3. command-line overrides from `run-v.sh` and `common.sh` (`long_ctx_args`: context, steps, batch, limits)

**Run OV** (`run-agent-v.sh` -> `run-harbor-osb.sh`, `examples/run_grpo_single_controller.py`):

1. the recipe
2. a copy of the task file `envs/harbor-verified-sft.yaml` without its `defaults`/`script` lines
3. `run-b-harbor-sc.yaml`, `run-s-sft.yaml`: what the Single Controller needs (`token_capture.enabled: true`),
   the SFT model, pod labels
4. `../harbor-capture-sc/opencode-sc.yaml`: the OpenCode adapter, `harbor_model_base_url`, `sampling_overrides`
5. `../harbor-capture-sc/tunnel-sc.yaml`: the per-sandbox tunnel
6. `run-agent-v.yaml`: rollout and stall timeouts, model-call capture, the Apertus 2 reasoning workaround
7. `run-ov.yaml`: OpenCode on these tasks (working directory, mounted or upload install, timeout)
8. command-line overrides from `common.sh` (`agent_v_args`, with the long-context profile)

An in-sandbox agent needs the Single Controller path: only it turns on Gym's token capture. On the Gym entrypoint
every OpenCode rollout would be masked.

## Another model or recipe

`RECIPE=recipes/<name>.yaml` picks another recipe (layout, model, optimizer). Settings specific to Apertus 2 or
Chonk, to change for another model:

| Setting | Where | Why |
|---|---|---|
| `policy.generation.output_format: apertus2`, `stop_token_ids: [12]`, `system_prompt_file` | `envs/harbor-verified-sft.yaml` | Apertus 2 chat format and stop token |
| `uses_interleaved_reasoning: false` | `run-agent-v.yaml` | workaround: Apertus 2's encoder refuses OpenCode's echoed thinking |
| `THINKING` (high/medium), `SFT_CKPT` (v0.1/v0.2) | `common.sh long_ctx_defaults` | Apertus 2 system prompts and Chonk checkpoints |
| `CP`, `KCP`, `CAP`, `NCCL_MIN` | `common.sh long_ctx_defaults` | Chonk's memory at 64K; `KCP` is needed for CP above 4 (Chonk's KDA layers) |
| Megatron conversion at CP 4 | `README.md` "Long context" | Chonk-specific failure at CP 8 |

What follows the recipe on its own: `model_info.max_input_tokens` / `max_output_tokens` come from
`policy.max_total_sequence_length` and `policy.generation.max_new_tokens`, and so do the context-limit failure
classes and OpenCode's limits. The batch (`PROMPTS` x 8) must be a multiple of the trainer's data-parallel size.

## Another task set

harbor_agent takes any Harbor task tree: one directory per task with `task.toml`, `instruction.md` and `tests/`.
For OpenSandbox, `[environment].docker_image` must name an image the sandbox server can pull; Dockerfile builds
are not supported. Our tasks also set `[verifier] environment_mode = "separate"` (verifier in its own pod, Harbor 0.25).

1. Rows: `python3 envs/prepare_harbor_osb_dataset.py --tasks-dir <tree> --output-dir <dir> --alias <alias>`
   writes `train.jsonl` / `validation.jsonl`; each row is `instance_id: <alias>::<task>`
   (`envs/prepare_harbor_verified_tasks.sh` does the same for the verified-task store).
2. Config: `harbor_datasets.<alias>.local_dataset_path: <tree>` and `harbor_data_dir` (`data.train.data_path`).
3. Limits: `harbor_agent_max_timeout` and `harbor_verifier_max_timeout` cap the per-task `timeout_sec`;
   `harbor_environment_build_timeout_multiplier` scales the setup limit. With OpenCode, keep `opencode_timeout_sec`
   below the agent cap so its log and exit status are recorded and the verifier runs.
4. Images: private images need `image_auth_file` (a `name:password` line) and `image_auth_registry` (the host the
   credentials are sent to). With OpenCode's `upload` install the image needs only `tar` (`mounted`: nothing); the tunnel needs `uname`,
   `setsid` or `nohup`, and `bash`. OpenCode runs in the image's working directory unless `opencode_workdir` is set.
5. Before a run, check that the reference solution scores 1 and an empty run scores 0 on a sample of tasks
   (`README.md`, "Run S with OpenCode on SWE-Lego", shows how we did it).

## Another cluster or sandbox server

- **Slurm cluster:** a `sites/<name>.env` profile (account, filesystems, container runtime, image paths); see the
  top-level `README.md`, "Clusters". The Apertus 2 images and the EDF template (`backends/nemo-rl/apertus2/`) are
  Alps-specific.
- **OpenSandbox server:** `OPENSANDBOX_DOMAIN` (default: the Breithorn gVisor server), the API key in
  `~/.opensandbox_token` or `OPENSANDBOX_API_KEY_FILE`, `connection.use_server_proxy` (`GATEWAY_PROXY=0` = ingress
  gateway, `1` = the server's proxy). `SANDBOXES` (64) is our share of Breithorn, `create.deadline_s` (1200) is tuned to its start times,
  `tls_skip_verify: true` in `tunnel-sc.yaml` to its certificates.
  Pods are labelled through `attribution` (`NEMO_GYM_USER`, run name).
- **Network:** Breithorn's pods cannot connect to Daint's nodes, hence the tunnel. Where sandboxes can reach the
  model server directly, leave out `tunnel-sc.yaml` and set `harbor_model_base_url` to the address the sandbox
  can reach (untested).
- **Binaries:** on Breithorn the sandbox platform mounts chisel, OpenCode and pi read-only into every sandbox
  (image volumes built from `images/agent-bin/` on branch `agent-image-volumes`, at `/opt/rl-tools/<name>/<version>/`); `AGENT_BINARIES=mounted`
  (the default, Gym 0047, `common.sh agent_binaries_env`) uses them and uploads nothing. Elsewhere set
  `AGENT_BINARIES=upload` (or mount the same images at `AGENT_BIN_ROOT`): `fetch-chisel.sh` and `fetch-opencode.sh`
  fetch pinned, sha256-checked releases on a machine with internet into `CHISEL_DIR` / `OPENCODE_DIR`, from where
  every sandbox gets them by upload. The node's chisel client always comes from `CHISEL_DIR`.
- **Another sandbox backend:** harbor_agent's sandbox environment uses Gym's sandbox providers; the tunnel needs
  the provider's `sandbox.endpoint(port)` to carry WebSockets. Only OpenSandbox has been tried.

## Another harness

- **In the sandbox:** a Gym adapter like `OpenCodeNemoGym` (point the agent at `api_base`; switch off every model
  call that is not the main loop, or capture masks the rollout; record its log for failure classes). In rl-infra,
  a new `AGENT` value for `run-agent-v.sh`: `common.sh` `agent_v_tag`, `agent_v_agent`, `agent_v_overlays`,
  `check_gym_agents`, an overlay `run-<tag>.yaml`, and the expected values in `check.py`. `AGENT=pi` (run PV, Gym 0048,
  `pi-sc.yaml`, `run-pv.yaml`, `fetch-pi.sh`) is the complete set of these changes for one agent; for a new agent,
  check its source for every model call besides the main loop (titles, summaries, retries) and every rewrite of
  earlier messages, since each one masks the rollout under capture.
- **In the Harbor process** (like Terminus-2): the adapter returns token IDs and logprobs itself (Harbor's
  `rollout_details`), no tunnel or capture needed, and the Gym entrypoint works (run V).

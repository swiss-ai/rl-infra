# Shared by check.sh, run-a.sh, run-b.sh, run-s.sh, run-agent-v.sh and run-v-24k.sh (sourced, not run). Defines how runs
# A, B, S and OV compose their configs (the same files, in the same order, as ../../run.sh and ../../run-harbor-osb.sh
# write them), the long-context profile of runs V and OV, and the static checks of the NeMo-RL clone and the Gym checkout.
# Needs HERE (this directory) and ROOT (the rl-infra checkout) set by the caller.

APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
# 12 nodes: 8 trainer (Megatron EP32, CPU AdamW) + 4 vLLM (TP1 EP4); the recipe sets num_nodes
CHONK_RECIPE="${ROOT}/recipes/apertus2-chonk-vllm-adamw.yaml"
SC_SCRIPT=examples/run_grpo_single_controller.py
# Run S (run B on the SFT checkpoint): main's SFT tool-call recipe (12 nodes: 8 trainer Megatron PP2
# EP16 + 4 vLLM TP1 EP4 FP8, 16384 tokens) and SFT Harbor task, Rob's HF export of the last SFT save
# (SFT_MODEL_SRC, staged to SFT_MODEL_DIR by run-s.sh; run-s-sft.yaml names the same directory).
SFT_RECIPE="${ROOT}/recipes/apertus2-chonk-sft-vllm-adamw-tools.yaml"
SFT_TASK="${ROOT}/envs/harbor-osb-sft.yaml"
# The SFT exports on /capstor and the names they are staged under in $SCRATCH/models (run.sh: sft_stage_as).
# v0.1 iter_0019374 (Rob's export, no modeling code): runs S/O/W and the 24k runs V/OV. v0.2 iter_0018315: main's
# long-context runs; path from origin/lean-kimina-verifier. Daint 2026-11-20: 51 shards, 202G, config.json identical to
# v0.1's, no MODELING_SOURCE (so staged like v0.1: small files copied, shards linked, modeling code from hfconverter).
SFT_V01_SRC=/capstor/store/cscs/swissai/infra01/hf-checkpoints/chonk-lc-16k-v0.1-ep1/iter_0019374
SFT_V01_NAME=chonk-lc-16k-v0.1-ep1-iter_0019374
SFT_V02_SRC=/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/posttraining/chonk-lc-16k-v0.2-iter_0018315
SFT_V02_NAME=chonk-lc-16k-v0.2-iter_0018315
SFT_MODEL_SRC_ENV="${SFT_MODEL_SRC:-}"   # a source given by the caller wins for either checkpoint
SFT_MODEL_SRC="${SFT_MODEL_SRC:-${SFT_V01_SRC}}"
SFT_MODEL_DIR="${SCRATCH}/models/${SFT_V01_NAME}"
# Its own EDF: run S needs the apertus-common overlay in the container spec, which setup.sh bakes into
# ~/.edf/<EDF_NAME>.toml; a shared EDF would change under jobs of the other runs waiting in the queue.
SFT_EDF_NAME="${SFT_EDF_NAME:-nemo-rl-apertus2-sft}"
# NeMo-RL 0057 (valid_mask in the group baseline on the Gym entrypoint, rule (ii)): a string only 0057 adds to
# nemo_rl/algorithms/grpo.py (twice: sync and async compute_advantage); check_nemo_rl_clone requires it. Empty = skip.
: "${NRL_0057_MARKER=valid_mask=sample_mask}"

# --- Long context: runs V (run-v-24k.sh) and OV (run-agent-v.sh) above 24576 tokens --------------------------------
# Plan research/reports/2026-11-20-long-context-agent-runs-plan.md (workspace), user decisions 2026-11-20. The profile
# follows CTX. Up to 24576 nothing changes: SFT v0.1, the task file's frame (thinking medium), its 6144 tokens per turn,
# 30 turns, agent cap 1800 s, TTL 3600 s, no context parallelism, no memory cap. Above 24576:
#   SFT_CKPT   v0.2 (else v0.1): the checkpoint, its recipe (20 nodes) and staged directory
#   THINKING   high (else medium): policy.generation.system_prompt_file=envs/apertus2-system-high.json, the frame
#              NeMo-RL's vLLM HTTP server builds for every Gym call (Terminus-2 and OpenCode alike); no budget
#   CP         CTX/8192 (8 at 64K, 16 at 128K): trainer context parallelism, 8192 tokens per GPU as main measured;
#              policy.make_sequence_length_divisible_by = 2 x CP. The trainer's data-parallel size is 32/CP (20 nodes)
#   KCP        1 when CP > 4 (else 0): the KCP scheme for the KDA layers (the pinned Megatron splits their 28 heads:
#              CP 1, 2 or 4 only); needs experiments/long-context/setup-kcp.sh once (kcp_setup, README.md)
#   CAP        0.74: NRL_CUDA_MEMORY_FRACTION of each trainer GPU (NeMo-RL 0033), room for NCCL/Triton outside PyTorch
#   NCCL_MIN   60 from 65536 on: NRL_NCCL_TIMEOUT_MINUTES (NeMo-RL 0040)
#   MAX_NEW    16384 tokens per turn (the task file: 6144); model_info.max_output_tokens and OpenCode's max_tokens follow
#   MAX_TURNS  100 (Terminus-2 turns, OpenCode steps)
#   AGENT_CAP  3000 s Harbor agent cap (the tasks' own); run OV: OpenCode ends at AGENT_CAP - 60, rollout timeout
#              AGENT_CAP + 4800 (two pod creates <= 1800 each, verifier 1200), stall watchdog 300 s more
#   TTL        4800 s sandbox_ttl_s (agent cap + create deadline 1200 + margin)
#   MASK_AGENT_TIMEOUT  1: harbor_mask_agent_timeout (rule (ii)); NeMo-RL 0057 keeps masked rows out of the group
#              baseline on the Gym entrypoint (run V)
#   MASK_OVERFLOW  1: harbor_mask_context_overflow (rule (i)), a key of Gym 0043 (check_gym_capture requires 0043)
#   HANG_DUMP_AFTER_S / HANG_KILL_AFTER_S  3600 / 14400 from 65536 on (backends/nemo-rl/hang_watch.sh: 900 / 2700)
# Every knob can be set at any CTX; empty = the task file's value. JOB_TIME, STEPS, PROMPTS per launcher.
LONG_CTX_KNOBS="CTX LONG_CTX SFT_CKPT THINKING CP KCP CAP NCCL_MIN MAX_NEW MAX_TURNS AGENT_CAP TTL MASK_AGENT_TIMEOUT MASK_OVERFLOW"

# long_ctx_defaults: the knobs above from CTX (shell variables, not exported; check.sh exports LONG_CTX_KNOBS).
long_ctx_defaults() {
    : "${CTX:=24576}"
    [[ "${CTX}" =~ ^[0-9]+$ ]] || { echo "long_ctx_defaults: CTX must be a number of tokens, not '${CTX}'" >&2; return 1; }
    if (( CTX > 24576 )); then
        LONG_CTX=1
        : "${SFT_CKPT:=v0.2}" "${THINKING:=high}" "${CP:=$(( CTX / 8192 ))}" "${CAP:=0.74}" "${MAX_NEW:=16384}"
        : "${MAX_TURNS:=100}" "${AGENT_CAP:=3000}" "${TTL:=4800}" "${MASK_AGENT_TIMEOUT:=1}"
        # Rule (i), user 2026-11-17: a rollout that hits the context limit is masked whole (Gym 0043,
        # harbor_mask_context_overflow). Gym's config is extra="allow", so without 0043 the key would be taken silently
        # and mask nothing: check_gym_capture requires 0043 when this is 1.
        : "${MASK_OVERFLOW:=1}"
        if (( CTX >= 65536 )); then
            : "${NCCL_MIN:=60}" "${HANG_DUMP_AFTER_S:=3600}" "${HANG_KILL_AFTER_S:=14400}"
        fi
    else
        LONG_CTX=0
        : "${SFT_CKPT:=v0.1}" "${THINKING:=medium}" "${CP:=1}" "${MASK_AGENT_TIMEOUT:=0}" "${MASK_OVERFLOW:=0}"
    fi
    : "${KCP:=$(( CP > 4 ? 1 : 0 ))}" "${CAP:=}" "${NCCL_MIN:=}" "${MAX_NEW:=}" "${MAX_TURNS:=}" "${AGENT_CAP:=}" "${TTL:=}"
    local v
    case "${SFT_CKPT}" in v0.1|v0.2) ;; *) echo "long_ctx_defaults: SFT_CKPT must be v0.1 or v0.2, not '${SFT_CKPT}'" >&2; return 1 ;; esac
    case "${THINKING}" in high|medium) ;; *) echo "long_ctx_defaults: THINKING must be high or medium, not '${THINKING}'" >&2; return 1 ;; esac
    for v in KCP MASK_AGENT_TIMEOUT MASK_OVERFLOW; do
        [[ "${!v}" == [01] ]] || { echo "long_ctx_defaults: ${v} must be 0 or 1, not '${!v}'" >&2; return 1; }
    done
    for v in CP NCCL_MIN MAX_NEW MAX_TURNS AGENT_CAP TTL; do
        [[ -z "${!v}" || "${!v}" =~ ^[1-9][0-9]*$ ]] || { echo "long_ctx_defaults: ${v} must be a positive number, not '${!v}'" >&2; return 1; }
    done
    (( CP == 1 || CTX % (2 * CP) == 0 )) || { echo "long_ctx_defaults: CTX ${CTX} is not a multiple of 2 x CP (${CP})" >&2; return 1; }
    [[ "${KCP}" == 1 ]] || (( CP == 1 || CP == 2 || CP == 4 )) || {
        echo "long_ctx_defaults: without KCP=1 the KDA layers split their 28 heads over the CP ranks: CP must be 1, 2 or 4, not ${CP}" >&2; return 1; }
    [[ -z "${AGENT_CAP}" ]] || (( AGENT_CAP > 60 )) || { echo "long_ctx_defaults: AGENT_CAP ${AGENT_CAP} s is too short" >&2; return 1; }
    # the checkpoint: source (SFT_MODEL_SRC from the caller wins), staged directory, 20-node recipe
    case "${SFT_CKPT}" in
        v0.1) SFT_MODEL_SRC="${SFT_MODEL_SRC_ENV:-${SFT_V01_SRC}}" SFT_MODEL_DIR="${SCRATCH}/models/${SFT_V01_NAME}"
              SFT_RECIPE_20N="${ROOT}/recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools-20n.yaml" ;;
        v0.2) SFT_MODEL_SRC="${SFT_MODEL_SRC_ENV:-${SFT_V02_SRC}}" SFT_MODEL_DIR="${SCRATCH}/models/${SFT_V02_NAME}"
              SFT_RECIPE_20N="${ROOT}/recipes/apertus2-chonk-sftv02-alps-vllm-adamw-tools-20n.yaml" ;;
    esac
}

# long_ctx_args: fills LONG_CTX_ARGS with the overrides of the knobs (after long_ctx_defaults), the same for the Gym
# entrypoint (run V) and the Single Controller (run OV). Empty at the 24k defaults.
long_ctx_args() {
    local p="env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent"
    LONG_CTX_ARGS=()
    (( CP == 1 )) || LONG_CTX_ARGS+=(policy.megatron_cfg.context_parallel_size="${CP}"
                                     policy.make_sequence_length_divisible_by="$(( 2 * CP ))")
    [[ "${KCP}" != 1 ]] || LONG_CTX_ARGS+=(++policy.megatron_cfg.model_overrides.linear_attention_cp_impl=kcp)
    [[ -z "${CAP}" ]] || LONG_CTX_ARGS+=("++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'${CAP}'\"")
    [[ -z "${NCCL_MIN}" ]] || LONG_CTX_ARGS+=("++policy.megatron_cfg.env_vars.NRL_NCCL_TIMEOUT_MINUTES=\"'${NCCL_MIN}'\"")
    # medium: the task file's frame (envs/apertus2-system.json)
    [[ "${THINKING}" != high ]] || LONG_CTX_ARGS+=(policy.generation.system_prompt_file="${ROOT}/envs/apertus2-system-high.json")
    [[ -z "${MAX_NEW}" ]] || LONG_CTX_ARGS+=(policy.generation.max_new_tokens="${MAX_NEW}")
    [[ -z "${MAX_TURNS}" ]] || LONG_CTX_ARGS+=("${p}.harbor_agent_kwargs.max_turns=${MAX_TURNS}")
    [[ -z "${AGENT_CAP}" ]] || LONG_CTX_ARGS+=("${p}.harbor_agent_max_timeout=${AGENT_CAP}")
    [[ -z "${TTL}" ]] || LONG_CTX_ARGS+=("${p}.harbor_environment_kwargs.sandbox_ttl_s=${TTL}")
    [[ "${MASK_AGENT_TIMEOUT}" != 1 ]] || LONG_CTX_ARGS+=("++${p}.harbor_mask_agent_timeout=true")
    [[ "${MASK_OVERFLOW}" != 1 ]] || LONG_CTX_ARGS+=("++${p}.harbor_mask_context_overflow=true")
}

# long_ctx_summary: one line with the profile in effect (after agent_v_defaults), so a CTX that did not reach the
# script (e.g. set without export) shows at the top of check.sh's output instead of only as 24k values further down.
long_ctx_summary() {
    local p="24k profile"; [[ "${LONG_CTX}" != 1 ]] || p="long context"
    echo "== CTX ${CTX} (${p}): SFT ${SFT_CKPT}, thinking ${THINKING}, CP ${CP} KCP ${KCP}, mask overflow ${MASK_OVERFLOW}," \
        "mask agent timeout ${MASK_AGENT_TIMEOUT}, recipe ${AGENT_V_RECIPE##*/}"
}

# check_python_usable <python>: the config code's imports work. Not just "import omegaconf": after a scratch purge a
# venv keeps the package directories without their files, and Python imports an empty directory as a namespace
# package, so only the names (OmegaConf, compose, VERSION) tell a purged venv from a complete one.
check_python_usable() {
    [[ -x "$1" ]] && "$1" -c 'from omegaconf import OmegaConf; from hydra import compose; import pydantic; pydantic.VERSION' \
        >/dev/null 2>&1
}

# long_ctx_job_env: exports what the job itself reads (sbatch passes the environment on; that HANG_* reach
# hang_watch.sh inside the job on Alps is UNVERIFIED, launch.sh passes NUMA_WATCH the same way).
long_ctx_job_env() {
    [[ -z "${HANG_DUMP_AFTER_S:-}" ]] || export HANG_DUMP_AFTER_S
    [[ -z "${HANG_KILL_AFTER_S:-}" ]] || export HANG_KILL_AFTER_S
}

# kcp_setup <EDF name>: points setup.sh and launch.sh at the KCP stack, as main's launch-code-ctx.sh does: the Megatron
# checkout with branch rl-kcp (KCP_MEGATRON_DIR, default $PROJECTS_DIR/megatron-lm-moe-apertus2-kcp) through a copy of
# the pin with its commit (setup.sh checks out the pin's commit), flash-linear-attention 0.5.2 (KCP_FLA_DIR, default
# $PROJECTS_DIR/fla-0.5.2) ahead of the image's on PYTHONPATH through a copy of the container spec, and an EDF of its
# own (<EDF name>-kcp, KCP_EDF_NAME overrides): setup.sh writes the spec to ~/.edf/<EDF_NAME>.toml, and the shared
# one would change under queued jobs. Both copies go to $SCRATCH/chonk-sc/kcp. Set up once with
# experiments/long-context/setup-kcp.sh (README.md, "Long context").
kcp_setup() {
    local dir="${KCP_MEGATRON_DIR:-${PROJECTS_DIR}/megatron-lm-moe-apertus2-kcp}" fla="${KCP_FLA_DIR:-${PROJECTS_DIR}/fla-0.5.2}"
    local out="${SCRATCH}/chonk-sc/kcp" commit
    commit="$(git -C "${dir}" rev-parse --verify -q rl-kcp 2>/dev/null)" \
        || { echo "kcp_setup: ${dir} has no branch rl-kcp; run experiments/long-context/setup-kcp.sh (README.md, long context)" >&2; return 1; }
    # a missing directory would only show inside the job, as the image's older fla without fla.ops.cp
    [[ -d "${fla}/fla/ops/cp" ]] \
        || { echo "kcp_setup: ${fla} has no flash-linear-attention 0.5.2 (fla/ops/cp); run setup-kcp.sh or set KCP_FLA_DIR" >&2; return 1; }
    mkdir -p "${out}"
    sed "s/^MEGATRON_COMMIT=.*/MEGATRON_COMMIT=${commit}/" "${APERTUS2_DIR}/pin" > "${out}/pin-kcp.tmp"
    grep -qx "MEGATRON_COMMIT=${commit}" "${out}/pin-kcp.tmp" || { echo "kcp_setup: no MEGATRON_COMMIT line in ${APERTUS2_DIR}/pin" >&2; return 1; }
    mv -f "${out}/pin-kcp.tmp" "${out}/pin-kcp"
    sed "s|:@@MEGATRON_DIR@@:|:@@MEGATRON_DIR@@:${fla}:|" "${APERTUS2_DIR}/edf.toml.template" > "${out}/edf-kcp.toml.template"
    grep -qF ":@@MEGATRON_DIR@@:${fla}:" "${out}/edf-kcp.toml.template" \
        || { echo "kcp_setup: could not put ${fla} on the container's PYTHONPATH (edf.toml.template changed?)" >&2; return 1; }
    export MEGATRON_DIR="${dir}" PIN_FILE="${out}/pin-kcp" CONTAINER_SPEC="${out}/edf-kcp.toml.template"
    export EDF_NAME="${KCP_EDF_NAME:-$1-kcp}"
    echo "== KCP: Megatron ${dir} at ${commit:0:9} (rl-kcp), fla ${fla}, EDF ${EDF_NAME}"
}

# recipe_copy <recipe> <run>: prints the path of a copy of the recipe in $SCRATCH/chonk-sc/recipes with its one-line
# defaults made absolute, which the job reads instead of the worktree's file (main's launch-code-ctx.sh: the driver
# loads the config anew at its start, after any edit or pull made while the job waited). Its defaults stay files.
recipe_copy() {
    local src d out="${SCRATCH}/chonk-sc/recipes/$2.yaml"
    src="$(realpath "$1")" || return 1
    d="$(sed -n 's/^defaults: *//p' "${src}")"
    [[ -n "${d}" ]] || { echo "recipe_copy: ${src} has no one-line defaults: entry" >&2; return 1; }
    [[ "${d}" == /* ]] || d="$(realpath -m "$(dirname "${src}")/${d}")"
    [[ -f "${d}" ]] || { echo "recipe_copy: ${d} (defaults of ${src}) does not exist" >&2; return 1; }
    mkdir -p "$(dirname "${out}")"
    sed "s|^defaults: .*|defaults: ${d}|" "${src}" > "${out}" && echo "${out}"
}

# sft_task_copy <out.yaml>: the SFT task file without its `defaults:` (it would merge the 8075 recipe
# over the one the launcher puts first) and `script:` (not a config key) lines, as ../../run.sh does
# with yq; both are single top-level lines in that file.
sft_task_copy() {
    grep -vE '^(defaults|script):' "${SFT_TASK}" > "$1"
    grep -q '^env:' "$1" && ! grep -qE '^(defaults|script):' "$1" || { echo "sft_task_copy: unexpected ${SFT_TASK}" >&2; return 1; }
}

# Run OV (run-agent-v.sh AGENT=opencode): OpenCode inside each pod on Kostas' verified tasks (the task file of
# run V), on the SC path; run PV (AGENT=pi) the same with the pi coding agent: an in-pod agent only makes OpenAI-compatible HTTP calls, so its token IDs come from Gym's
# token capture at the model endpoint, which only the SC entrypoint turns on (NeMo-RL
# single_controller_utils/setup.py token_capture; the Gym entrypoint of run V has none). Another in-pod agent is
# one more AGENT value: a case in agent_v_tag, agent_v_overlays and check_gym_agents (and check.py).
VERIFIED_TASK="${ROOT}/envs/harbor-verified-sft.yaml"
CAPTURE_SC="${ROOT}/experiments/harbor-capture-sc"

# verified_task_copy <out.yaml>: the verified task file without its `defaults:` and `script:` lines (as
# sft_task_copy; run-agent-v.sh puts the recipe first itself).
verified_task_copy() {
    grep -vE '^(defaults|script):' "${VERIFIED_TASK}" > "$1"
    grep -q '^env:' "$1" && ! grep -qE '^(defaults|script):' "$1" || { echo "verified_task_copy: unexpected ${VERIFIED_TASK}" >&2; return 1; }
}

# agent_v_tag <agent>: the run's short name (run OV = opencode, run PV = pi); unknown agents fail here first.
agent_v_tag() {
    case "$1" in
        opencode) echo ov ;;
        pi) echo pv ;;
        *) echo "agent_v_tag: AGENT must be opencode (run OV) or pi (run PV), not '$1'" >&2; return 1 ;;
    esac
}

# agent_v_agent <tag>: the inverse (check.sh RUNS=ov / RUNS=pv).
agent_v_agent() {
    case "$1" in
        ov) echo opencode ;;
        pv) echo pi ;;
        *) echo "agent_v_agent: no in-pod agent run '$1'" >&2; return 1 ;;
    esac
}

# agent_v_defaults: run-agent-v.sh's knobs with their defaults (check.sh RUNS=ov checks the same values).
# CTX 24576 = run V's 24k layout (20 nodes: 16 trainer EP32 + 4 vLLM, trainer data-parallel size 32);
# CTX 16384 = the 12-node layout (8 trainer EP16, data-parallel size 16); CTX > 24576 = the long-context profile
# (long_ctx_defaults: 20 nodes, SFT v0.2, CP8 KCP at 64K, data-parallel size 32/CP, ...). RECIPE overrides (then the
# batch is not checked against the data-parallel size). SMOKE=1: one step, one sample per data-parallel rank (long
# context: 4 x 8, JOB_TIME 03:00:00; 10 steps: 12:00:00, both UNVERIFIED against Daint's limit for normal).
agent_v_defaults() {
    long_ctx_defaults || return 1
    if [[ -n "${RECIPE:-}" ]]; then
        AGENT_V_RECIPE="$(realpath "${RECIPE}")" AGENT_V_DP=""
    elif (( CTX > 16384 )); then
        (( 32 % CP == 0 )) || { echo "agent_v_defaults: CP ${CP} does not divide the 32 ranks of a pipeline stage" >&2; return 1; }
        AGENT_V_RECIPE="${SFT_RECIPE_20N}" AGENT_V_DP=$(( 32 / CP ))
    else
        [[ "${SFT_CKPT}" == v0.1 ]] || { echo "agent_v_defaults: no 12-node recipe for SFT ${SFT_CKPT}; set RECIPE" >&2; return 1; }
        (( 16 % CP == 0 )) || { echo "agent_v_defaults: CP ${CP} does not divide the 16 ranks of a pipeline stage" >&2; return 1; }
        AGENT_V_RECIPE="${ROOT}/recipes/apertus2-chonk-sft19374-alps-vllm-adamw-tools.yaml" AGENT_V_DP=$(( 16 / CP ))
    fi
    if [[ "${SMOKE:-0}" == 1 && "${LONG_CTX}" == 1 ]]; then
        : "${STEPS:=1}" "${PROMPTS:=4}" "${JOB_TIME:=03:00:00}"
    elif [[ "${SMOKE:-0}" == 1 ]]; then
        : "${STEPS:=1}" "${PROMPTS:=$(( ${AGENT_V_DP:-32} / 8 ))}" "${JOB_TIME:=02:30:00}"
    fi
    [[ "${LONG_CTX}" != 1 ]] || : "${JOB_TIME:=12:00:00}"
    : "${STEPS:=10}" "${PROMPTS:=8}" "${SANDBOXES:=64}" "${GATEWAY_PROXY:=0}" "${MODEL_CALLS:=1}"
    : "${HARBOR_ROOT:=${SCRATCH}/osb/harbor_verified}"
    [[ "${GATEWAY_PROXY}" == [01] ]] || { echo "agent_v_defaults: GATEWAY_PROXY must be 0 or 1, not ${GATEWAY_PROXY}" >&2; return 1; }
    # OpenCode / pi end 60 s before Harbor's agent cap, so their exit status and log are recorded (run-ov.yaml,
    # run-pv.yaml)
    if [[ -n "${AGENT_CAP}" ]]; then
        : "${OPENCODE_TIMEOUT_SEC:=$(( AGENT_CAP - 60 ))}" "${PI_TIMEOUT_SEC:=$(( AGENT_CAP - 60 ))}"
        export OPENCODE_TIMEOUT_SEC PI_TIMEOUT_SEC
    fi
    export CTX STEPS PROMPTS SANDBOXES GATEWAY_PROXY MODEL_CALLS HARBOR_ROOT AGENT_V_RECIPE AGENT_V_DP
}

# agent_v_overlays <agent>: the overlays after the task copy, in merge order, one per line.
#   run-b-harbor-sc.yaml run-s-sft.yaml     SC, token capture, the SFT model, pod labels, credentials by file
#   <agent overlay>, tunnel-sc.yaml          the agent in the pod (model URL = the tunnel's pod end), the tunnel
#   run-agent-v.yaml                        what the verified tasks need on SC for any in-pod agent
#   run-<tag>.yaml                          the agent's settings for these tasks (run-ov.yaml, run-pv.yaml: WORKDIR, 1740 s)
agent_v_overlays() {
    printf '%s\n' "${HERE}/run-b-harbor-sc.yaml" "${HERE}/run-s-sft.yaml"
    case "$1" in
        opencode) printf '%s\n' "${CAPTURE_SC}/opencode-sc.yaml" "${CAPTURE_SC}/tunnel-sc.yaml" \
                      "${HERE}/run-agent-v.yaml" "${HERE}/run-ov.yaml" ;;
        pi) printf '%s\n' "${CAPTURE_SC}/pi-sc.yaml" "${CAPTURE_SC}/tunnel-sc.yaml" \
                "${HERE}/run-agent-v.yaml" "${HERE}/run-pv.yaml" ;;
        *) echo "agent_v_overlays: AGENT must be opencode or pi, not $1" >&2; return 1 ;;
    esac
}

# agent_v_args <train.jsonl> <cred file> <harbor jobs dir>: fills AGENT_V_ARGS with the command-line overrides
# run-agent-v.sh passes (after agent_v_defaults), as run-v.sh / run-v-24k.sh pass theirs.
agent_v_args() {
    local p="env.nemo_gym.harbor_agent.responses_api_agents.harbor_agent"
    AGENT_V_ARGS=(
        grpo.max_num_steps="${STEPS}"
        grpo.num_prompts_per_step="${PROMPTS}" policy.train_global_batch_size="$(( PROMPTS * 8 ))"
        policy.max_total_sequence_length="${CTX}"
        policy.sequence_packing.train_mb_tokens="${CTX}" policy.sequence_packing.logprob_mb_tokens="${CTX}"
        data.train.data_path="$1"
        registry_cred_file="$2"
        "${p}.harbor_jobs_dir=$3"
        "${p}.concurrency=${SANDBOXES}"
        # the checkpoint the launcher staged (SFT_CKPT); run-s-sft.yaml, merged after the recipe, names v0.1
        policy.model_name="${SFT_MODEL_DIR}"
    )
    # as run-v-24k.sh: vLLM refuses max_model_len > max_position_embeddings (16384) without it
    (( CTX <= 16384 )) || AGENT_V_ARGS+=("++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"")
    # the long-context knobs (empty at the 24k defaults); with an agent cap, the SC limits that follow from it
    # (run-agent-v.yaml: 6600 / 6900 s for the task file's 1800 s cap)
    long_ctx_args
    AGENT_V_ARGS+=(${LONG_CTX_ARGS[@]+"${LONG_CTX_ARGS[@]}"})
    if [[ -n "${AGENT_CAP}" ]]; then
        AGENT_V_ARGS+=(async_rl.rollout_failure.nemo_gym.rollout_timeout_s="$(( AGENT_CAP + 4800 ))"
                       async_rl.stall_watchdog.stall_timeout_s="$(( AGENT_CAP + 5100 ))")
    fi
    # GATEWAY_PROXY as in experiments/osb-reach: 0 = through the ingress gateway (use_server_proxy false, the task
    # file's value, as run V), 1 = through the OpenSandbox server proxy (runs S/O, the tunnel's tested path)
    case "${GATEWAY_PROXY}" in
        0) AGENT_V_ARGS+=(env.nemo_gym.sandbox.opensandbox.connection.use_server_proxy=false) ;;
        1) AGENT_V_ARGS+=(env.nemo_gym.sandbox.opensandbox.connection.use_server_proxy=true) ;;
        *) echo "agent_v_args: GATEWAY_PROXY must be 0 or 1, not ${GATEWAY_PROXY}" >&2; return 1 ;;
    esac
    [[ "${MODEL_CALLS}" == 1 ]] || AGENT_V_ARGS+=(env.nemo_gym.observability_enabled=false)
}

# compose_cfg <a|b> <out.yaml>: the top-level config the launcher writes.
#   A: ../../run.sh writes   defaults: [RECIPE, TASK]           with TASK = run-a-gsm8k.yaml
#   B: ../../run-harbor-osb.sh writes defaults: [RECIPE, TASK, OVERLAY...]
#      with TASK = envs/harbor-osb.yaml, OVERLAY = run-b-harbor-sc.yaml
#   S: the same with RECIPE = SFT_RECIPE, TASK = sft_task_copy (written next to out.yaml as
#      <out>-task.yaml), OVERLAY = run-b-harbor-sc.yaml run-s-sft.yaml
#   O: S + ../harbor-capture-sc/opencode-sc.yaml tunnel-sc.yaml run-s-opencode.yaml (run-s.sh AGENT=opencode)
compose_cfg() {
    case "$1" in
        a) printf 'defaults:\n  - %s\n  - %s\n' "$(realpath "${CHONK_RECIPE}")" "$(realpath "${HERE}/run-a-gsm8k.yaml")" > "$2" ;;
        b) printf 'defaults:\n  - %s\n  - %s\n  - %s\n' "$(realpath "${CHONK_RECIPE}")" \
               "$(realpath "${ROOT}/envs/harbor-osb.yaml")" "$(realpath "${HERE}/run-b-harbor-sc.yaml")" > "$2" ;;
        s) sft_task_copy "${2%.yaml}-task.yaml" || return 1
           printf 'defaults:\n  - %s\n  - %s\n  - %s\n  - %s\n' "$(realpath "${SFT_RECIPE}")" \
               "$(realpath "${2%.yaml}-task.yaml")" "$(realpath "${HERE}/run-b-harbor-sc.yaml")" \
               "$(realpath "${HERE}/run-s-sft.yaml")" > "$2" ;;
        # O: run S with AGENT=opencode (run-s.sh): + opencode-sc.yaml tunnel-sc.yaml run-s-opencode.yaml
        o) compose_cfg s "$2" || return 1
           printf '  - %s\n' "$(realpath "${ROOT}/experiments/harbor-capture-sc/opencode-sc.yaml")" \
               "$(realpath "${ROOT}/experiments/harbor-capture-sc/tunnel-sc.yaml")" \
               "$(realpath "${HERE}/run-s-opencode.yaml")" >> "$2" ;;
        # W: run O on SWE-Lego (run-s.sh AGENT=opencode TASKS=swe-lego): + run-s-swe-lego.yaml
        w) compose_cfg o "$2" || return 1
           printf '  - %s\n' "$(realpath "${HERE}/run-s-swe-lego.yaml")" >> "$2" ;;
        # OV / PV: run-agent-v.sh AGENT=opencode / pi (run-harbor-osb.sh writes RECIPE, TASK, OVERLAY...); needs
        # agent_v_defaults first (the recipe follows CTX)
        ov|pv) local o agent
           agent="$(agent_v_agent "$1")" || return 1
           verified_task_copy "${2%.yaml}-task.yaml" || return 1
           printf 'defaults:\n  - %s\n  - %s\n' "${AGENT_V_RECIPE}" "$(realpath "${2%.yaml}-task.yaml")" > "$2"
           while read -r o; do printf '  - %s\n' "$(realpath "${o}")" >> "$2"; done < <(agent_v_overlays "${agent}") ;;
        *) echo "compose_cfg: run must be a, b, s, o, w, ov or pv" >&2; return 1 ;;
    esac
}

# check_nemo_rl_clone: the merged NeMo-RL clone (NEMO_RL_DIR) has the pinned PRs, NVIDIA-NeMo/RL#2517
# removed the async_engine assert for vLLM EP != TP, and f196's refit / membership / token-capture
# code in vllm_generation.py survived the -Xtheirs merge. Prints one line per check; returns the
# number of failures.
check_nemo_rl_clone() {
    local d="${NEMO_RL_DIR}" fails=0 repo pr sha n
    local gen="${d}/nemo_rl/models/generation/vllm/vllm_generation.py"
    local wrk="${d}/nemo_rl/models/generation/vllm/vllm_worker.py"
    ok()  { echo "  ok    $*"; }
    bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
    note() { echo "  note  $*"; }
    [[ -e "${d}/.git" && -f "${gen}" ]] || { bad "no NeMo-RL clone at ${d}"; return 1; }
    echo "  clone ${d}: $(git -C "${d}" branch --show-current 2>/dev/null) @ $(git -C "${d}" rev-parse --short HEAD)"
    # shellcheck disable=SC1090
    ( source "${PIN_FILE:-${APERTUS2_DIR}/pin}"
      git -C "${d}" merge-base --is-ancestor "${NEMO_RL_COMMIT}" HEAD ) \
        && ok "based on the pinned f196 commit" || bad "not based on the pinned NeMo-RL commit (backends/nemo-rl/apertus2/pin)"
    while read -r repo pr sha; do
        [[ -n "${sha}" ]] || continue
        git -C "${d}" merge-base --is-ancestor "${sha}" HEAD 2>/dev/null \
            && ok "${repo} PR #${pr} merged (${sha:0:9})" || bad "${repo} PR #${pr} (${sha:0:9}) not merged; run backends/nemo-rl/apertus2/setup.sh"
    done <<< "$(source "${PIN_FILE:-${APERTUS2_DIR}/pin}"; echo "${NEMO_RL_PRS}")"
    # NVIDIA-NeMo/RL#2517 @05db6ab36
    grep -q "vLLM async_engine has some issues when using DP inside vLLM" "${gen}" \
        && bad "vllm_generation.py still asserts async_engine off for vLLM EP != TP (plain f196 line ~191-199): #2517 not applied" \
        || ok "#2517: async_engine assert for EP != TP is gone"
    grep -q "workers_per_vllm_instance" "${gen}" \
        && ok "#2517: DP-aware rank_prefix in init_collective (workers_per_vllm_instance)" \
        || bad "#2517: init_collective has no DP-aware rank_prefix (workers_per_vllm_instance missing)"
    grep -q "data_parallel_external_lb" "${wrk}" \
        && ok "#2517: external-LB vLLM DP kwargs for async_engine in vllm_worker.py" \
        || bad "#2517: vllm_worker.py lacks data_parallel_external_lb (async_engine + vLLM DP)"
    # f196 code the SC path uses (refit through CollectiveWeightSynchronizer, membership, capture)
    for m in "def _refit_leader_workers" "get_dp_leader_worker_idx" "refit_timeout_s" \
             "membership.shard_prefixes" "def setup_token_capture"; do
        grep -qF "${m}" "${gen}" && ok "f196 marker in vllm_generation.py: ${m}" \
            || bad "f196 marker missing from vllm_generation.py: ${m} (the #2517 -Xtheirs merge reverted newer code)"
    done
    [[ -f "${d}/nemo_rl/weight_sync/collective_weight_synchronizer.py" ]] \
        && ok "nemo_rl/weight_sync/collective_weight_synchronizer.py present" \
        || bad "nemo_rl/weight_sync/collective_weight_synchronizer.py missing"
    # NeMo-RL patch 0052: a continuation after a truncated Apertus 2 burst gets the context-length 400
    # (without it: assert -> HTTP 500 retried by Gym, job 5303503); run S needs it
    grep -q "class TruncatedBurstError" "${d}/nemo_rl/data/apertus2_format.py" 2>/dev/null \
        && ok "NeMo-RL patch 0052: TruncatedBurstError in apertus2_format.py" \
        || bad "NeMo-RL patch 0052 missing (TruncatedBurstError in nemo_rl/data/apertus2_format.py); run backends/nemo-rl/apertus2/setup.sh"
    # NeMo-RL patch 0053: logprob outlier dump (off unless run-s-lpdump.yaml sets NRL_LOGPROB_DUMP_DIR)
    grep -q "NRL_LOGPROB_DUMP_DIR" "${d}/nemo_rl/algorithms/loss/logprob_dump.py" 2>/dev/null \
        && ok "NeMo-RL patch 0053: nemo_rl/algorithms/loss/logprob_dump.py" \
        || bad "NeMo-RL patch 0053 missing (nemo_rl/algorithms/loss/logprob_dump.py); run backends/nemo-rl/apertus2/setup.sh"
    grep -q "def _invalid_apertus2_history" "${d}/nemo_rl/models/generation/vllm/vllm_worker_async.py" 2>/dev/null \
        && ok "NeMo-RL patch 0054: an unencodable Apertus 2 chat history gets HTTP 400" \
        || bad "NeMo-RL patch 0054 missing (_invalid_apertus2_history in vllm_worker_async.py); run backends/nemo-rl/apertus2/setup.sh"
    # NeMo-RL patch 0057: masked rows leave the group baseline on the Gym entrypoint (rule (ii)); checked while
    # NRL_0057_MARKER (top of this file) is non-empty
    if [[ -n "${NRL_0057_MARKER}" ]]; then
        grep -qF -- "${NRL_0057_MARKER}" "${d}/nemo_rl/algorithms/grpo.py" 2>/dev/null \
            && ok "NeMo-RL patch 0057: masked rows leave the group baseline (${NRL_0057_MARKER})" \
            || bad "NeMo-RL patch 0057 missing ('${NRL_0057_MARKER}' in nemo_rl/algorithms/grpo.py); run backends/nemo-rl/apertus2/setup.sh"
    fi
    for f in "${SC_SCRIPT}" nemo_rl/algorithms/single_controller.py nemo_rl/algorithms/single_controller_utils/config.py; do
        [[ -f "${d}/${f}" ]] && ok "${f} present" || bad "${f} missing"
    done
    # Reference: the merge reproduced from local commits (f196 + #5 + #1 + #2517 -Xtheirs + #8 +
    # patches) changes these two files against f196 by exactly these line counts. #2517 replaced
    # f196's resolve_distributed_executor_backend ("uni" for EP > TP) with its inline choice ("mp"),
    # which is what the legacy Chonk runs (5184052) used too.
    n="$(git -C "${d}" diff --numstat "$(source "${PIN_FILE:-${APERTUS2_DIR}/pin}"; echo "${NEMO_RL_COMMIT}")" HEAD \
        -- nemo_rl/models/generation/vllm/vllm_generation.py nemo_rl/models/generation/vllm/vllm_worker.py \
        | awk '{printf "%s +%s -%s; ", $3, $1, $2}')"
    if [[ "${n}" == "nemo_rl/models/generation/vllm/vllm_generation.py +9 -19; nemo_rl/models/generation/vllm/vllm_worker.py +33 -19; " ]]; then
        ok "vllm_generation.py / vllm_worker.py differ from f196 exactly as the reference merge (+9 -19, +33 -19)"
    else
        note "vllm_generation.py / vllm_worker.py vs f196: ${n:-no diff }(reference merge: +9 -19, +33 -19); inspect with git -C ${d} diff ${NEMO_RL_COMMIT:-<pin>} HEAD -- nemo_rl/models/generation/vllm/"
    fi
    grep -q "distributed_executor_backend\"\] = \"mp\"" "${wrk}" \
        && note "vLLM executor backend for EP > TP at TP1: mp (#2517, as in the legacy Chonk runs)" \
        || note "vLLM executor backend for EP > TP: not #2517's 'mp' (f196 resolve_distributed_executor_backend -> 'uni'?)"
    return "${fails}"
}

# check_gym_capture: the shared Gym checkout of this clone has what run B needs: setup-gym.sh
# applied the current patches/gym set (its stamp, a hash of all patches, matches), with rl-infra
# Gym patch 0014 (harbor_agent hands the agent the rollout-prefixed model URL, base_url_for_run;
# without it capture records nothing and every row is rejected, job 5236016) and 0015
# (_should_mask: under capture empty output is not masked, failed trials still are). Same marker
# as ../harbor-capture-sc/run.sh; 0022 (capture ignores think-wrapped reasoning, needed by run S); and
# 0024 (streaming under capture, needed by OpenCode).
# Returns the number of failures.
check_gym_capture() {
    local g="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym" fails=0 stamp
    local app="${g}/responses_api_agents/harbor_agent/app.py"
    ok()  { echo "  ok    $*"; }
    bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
    [[ -f "${g}/nemo_gym/__init__.py" ]] || { bad "no Gym checkout in ${g}"; return 1; }
    echo "  Gym ${g} @ $(git -C "${g}" rev-parse --short HEAD 2>/dev/null)"
    stamp="$(cat "${APERTUS2_DIR}"/patches/gym/*.patch | git hash-object --stdin)"
    [[ "$(cat "${g}/.git/rl-infra-patches" 2>/dev/null)" == "${stamp}" ]] \
        && ok "Gym patches: the current patches/gym set is applied" \
        || bad "Gym patches: the checkout does not carry the current patches/gym set (setup-gym.sh not re-run)"
    grep -q harbor_model_base_url "${app}" 2>/dev/null && grep -q "base_url_for_run(" "${app}" \
        && ok "Gym patch 0014: harbor_agent routes model calls through the rollout URL" \
        || bad "Gym patch 0014 missing (harbor_model_base_url / base_url_for_run in harbor_agent/app.py)"
    grep -q "def _should_mask" "${app}" 2>/dev/null \
        && ok "Gym patch 0015: capture-aware _should_mask" \
        || bad "Gym patch 0015 missing (_should_mask in harbor_agent/app.py)"
    [[ -d "${g}/nemo_gym/token_id_capture" ]] && ok "nemo_gym/token_id_capture present" \
        || bad "nemo_gym/token_id_capture missing from the Gym checkout"
    # 0022: without it every rollout whose model reasons is rejected unresolved_parent (job 5302914)
    grep -q "def _strip_leading_reasoning" "${g}/nemo_gym/token_id_capture/fingerprint.py" 2>/dev/null \
        && ok "Gym patch 0022: capture fingerprints ignore think-wrapped reasoning" \
        || bad "Gym patch 0022 missing (_strip_leading_reasoning in token_id_capture/fingerprint.py)"
    # 0030: without it a tool call with unparseable arguments (OpenCode echoes it as `invalid`) poisons the
    # rest of the rollout (job 5316800)
    grep -q "def _repaired_tool_name" "${g}/nemo_gym/token_id_capture/fingerprint.py" 2>/dev/null \
        && ok "Gym patch 0030: capture lineage survives a tool call with unparseable arguments" \
        || bad "Gym patch 0030 missing (_repaired_tool_name in token_id_capture/fingerprint.py)"
    # 0032: Harbor 0.25's Job.create preflight accepts the OpenCode/mini adapters' own kwargs; without it no
    # trial runs and every rollout is empty (job 5329860)
    grep -q "def kwargs_for_harbor_options" "${g}/responses_api_agents/harbor_agent/custom_agents/installed_compat.py" 2>/dev/null \
        && ok "Gym patch 0032: Harbor 0.25 preflight accepts our agents' kwargs" \
        || bad "Gym patch 0032 missing (kwargs_for_harbor_options in harbor_agent/custom_agents/installed_compat.py)"
    # 0033: OpenCode/mini drop the Terminus-2 kwargs of Gym's training yaml (Harbor 0.25 rejects them, job 5330618)
    grep -q '"nemo_model_server_timeout_sec",' "${g}/responses_api_agents/harbor_agent/custom_agents/installed_compat.py" 2>/dev/null \
        && ok "Gym patch 0033: OpenCode/mini drop the training yaml's Terminus-2 kwargs" \
        || bad "Gym patch 0033 missing (nemo_model_server_timeout_sec in SHARED_AGENT_KWARGS, custom_agents/installed_compat.py)"
    # 0035: Terminus-2 ends the episode after max_consecutive_truncations replies in a row that hit max_tokens
    # (it looped 320x, jobs 5350472/5355345); envs/harbor-verified-sft.yaml sets it, which Harbor 0.25 rejects
    # without the patch (no trial runs)
    grep -q "class TruncationLoopError" "${g}/responses_api_agents/harbor_agent/custom_agents/terminus_2_nemo_gym.py" 2>/dev/null \
        && ok "Gym patch 0035: Terminus-2 stops after max_consecutive_truncations" \
        || bad "Gym patch 0035 missing (TruncationLoopError in harbor_agent/custom_agents/terminus_2_nemo_gym.py)"
    # 0036: after such a reply Terminus-2 asks again with the same prompt; its own re-ask dropped the observation of
    # the commands that did run (job 5356101); envs/harbor-verified-sft.yaml sets truncation_retry, which Harbor 0.25
    # rejects without the patch (no trial runs)
    grep -q "def _take_back_truncated_exchange" "${g}/responses_api_agents/harbor_agent/custom_agents/terminus_2_nemo_gym.py" 2>/dev/null \
        && ok "Gym patch 0036: Terminus-2 resamples after a truncated reply" \
        || bad "Gym patch 0036 missing (_take_back_truncated_exchange in harbor_agent/custom_agents/terminus_2_nemo_gym.py)"
    # 0037: failure class per rollout (train/harbor_agent/failure_<class>/mean) and a reply cut by the context ends
    # the episode as an overflow; envs/harbor-verified-sft.yaml sets context_limit_truncation, which Harbor 0.25
    # rejects without the patch (no trial runs)
    grep -q "def classify_rollout" "${g}/responses_api_agents/harbor_agent/failure_classes.py" 2>/dev/null \
        && ok "Gym patch 0037: failure classes per rollout, context-limit truncation ends the episode" \
        || bad "Gym patch 0037 missing (classify_rollout in harbor_agent/failure_classes.py)"
    # 0038, 0039 (Kostas, from main's 0014/0015): a local task is named directly (no dataset scan per trial);
    # harbor_environment_build_timeout_multiplier, which envs/harbor-verified-sft.yaml sets
    grep -q "harbor_environment_build_timeout_multiplier" "${g}/responses_api_agents/harbor_agent/app.py" 2>/dev/null \
        && ok "Gym patches 0038/0039: local task named directly, environment build timeout multiplier" \
        || bad "Gym patch 0039 missing (harbor_environment_build_timeout_multiplier in harbor_agent/app.py)"
    # 0040: create.deadline_s, a total deadline for a sandbox create (job 5365870: 10 retries x 600 s pod-ready
    # wait held one create ~6900 s); envs/harbor-verified-sft.yaml sets it, which the provider rejects without
    # the patch (unexpected keyword)
    grep -q "def _create_deadline_stop" "${g}/nemo_gym/sandbox/providers/opensandbox/provider.py" 2>/dev/null \
        && ok "Gym patch 0040: total deadline for a sandbox create (create.deadline_s)" \
        || bad "Gym patch 0040 missing (_create_deadline_stop in sandbox/providers/opensandbox/provider.py)"
    # 0041: infra rollouts are masked also without token capture (job 5365870: 45 verifier-pod failures would
    # have trained as reward 0)
    grep -q "harbor_mask_infra" "${g}/responses_api_agents/harbor_agent/app.py" 2>/dev/null \
        && ok "Gym patch 0041: infra rollouts masked (harbor_mask_infra)" \
        || bad "Gym patch 0041 missing (harbor_mask_infra in harbor_agent/app.py)"
    # 0042: a length reply is cut by the context only within max(256, 1%) of the limit (job 5365870: 22+ of 73
    # "cuts" had room left); "prompt does not fit" only for a rejected first prompt
    grep -q "def context_cut_margin" "${g}/responses_api_agents/harbor_agent/custom_agents/llms/nemo_gym_llm.py" 2>/dev/null \
        && ok "Gym patch 0042: context cut only near the context limit" \
        || bad "Gym patch 0042 missing (context_cut_margin in harbor_agent/custom_agents/llms/nemo_gym_llm.py)"
    # 0043: harbor_mask_context_overflow (rule (i)); required when the launchers set it (MASK_OVERFLOW=1): without the
    # patch Gym's config takes the key silently (extra="allow") and masks nothing
    if [[ "${MASK_OVERFLOW:-0}" == 1 ]]; then
        grep -q "harbor_mask_context_overflow" "${app}" 2>/dev/null \
            && ok "Gym patch 0043: context_overflow rollouts masked (harbor_mask_context_overflow)" \
            || bad "Gym patch 0043 missing (harbor_mask_context_overflow in harbor_agent/app.py), but MASK_OVERFLOW=1 sets it"
    fi
    # 0031: image_auth (from image_auth_file, Kostas' 0013) only for one registry, not Docker Hub
    grep -q "def _image_registry" "${g}/nemo_gym/sandbox/providers/opensandbox/provider.py" 2>/dev/null \
        && ok "Gym patch 0031: image_auth_registry" \
        || bad "Gym patch 0031 missing (_image_registry in sandbox/providers/opensandbox/provider.py)"
    # 0024: streaming clients (OpenCode) under worker-owned capture; without it they get 422
    grep -q "def _stream_served_response" "${g}/nemo_gym/base_responses_api_model.py" 2>/dev/null \
        && ok "Gym patch 0024: streaming requests are replayed as SSE under capture" \
        || bad "Gym patch 0024 missing (_stream_served_response in nemo_gym/base_responses_api_model.py)"
    for f in nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
             responses_api_agents/harbor_agent/configs/harbor_agent_opensandbox_training.yaml; do
        [[ -f "${g}/${f}" ]] && ok "${f} present" || bad "${f} missing (OpenSandbox commits, setup-gym.sh)"
    done
    if (( fails )); then
        echo "  fix: bash ${APERTUS2_DIR}/setup-gym.sh"
    fi
    return "${fails}"
}

# check_gym_agents <agent>: the in-pod agent (opencode: runs O/OV; pi: run PV) and the per-pod model tunnel are in the Gym
# checkout (after check_gym_capture). Returns the number of failures.
check_gym_agents() {
    local g="${NEMO_RL_DIR}/3rdparty/Gym-workspace/Gym" fails=0
    local ca="${g}/responses_api_agents/harbor_agent/custom_agents"
    ok()  { echo "  ok    $*"; }
    bad() { echo "  FAIL  $*"; fails=$((fails + 1)); }
    grep -q server_start_script "${g}/responses_api_agents/harbor_agent/custom_envs/nemo_gym_sandbox/model_tunnel.py" 2>/dev/null \
        && ok "Gym patches 0016/0017: per-pod model tunnel (chisel), server started through exec" \
        || bad "Gym patches 0016/0017 missing (server_start_script in custom_envs/nemo_gym_sandbox/model_tunnel.py)"
    case "$1" in
        opencode)
            [[ -f "${ca}/opencode_pod/install-opencode-binary.sh.j2" ]] && grep -q "class OpenCodeNemoGym" "${ca}/opencode_nemo_gym.py" 2>/dev/null \
                && ok "Gym patch 0026: OpenCodeNemoGym with the pinned binary install" \
                || bad "Gym patch 0026 missing (OpenCodeNemoGym, opencode_pod/install-opencode-binary.sh.j2)"
            grep -q opencode_workdir "${ca}/opencode_nemo_gym.py" 2>/dev/null \
                && ok "Gym patch 0029: opencode_workdir (unset: the task's WORKDIR)" \
                || bad "Gym patch 0029 missing (opencode_workdir in custom_agents/opencode_nemo_gym.py)"
            # 0044 (job 5425513: 64 of 640 trials failed the install on images with python3 < 3.8)
            grep -q download_wget "${ca}/opencode_pod/install-opencode-binary.sh.j2" 2>/dev/null \
                && ok "Gym patch 0044: OpenCode install with any python3, curl or wget; setup failures are infra" \
                || bad "Gym patch 0044 missing (download_wget in opencode_pod/install-opencode-binary.sh.j2)"
            # 0045: run-ov.yaml sets opencode_install: upload (the archive from OPENCODE_DIR, fetch-opencode.sh)
            grep -q POD_ARCHIVE "${ca}/opencode_nemo_gym.py" 2>/dev/null \
                && ok "Gym patch 0045: OpenCode archive uploaded by the adapter (opencode_install: upload)" \
                || bad "Gym patch 0045 missing (POD_ARCHIVE in custom_agents/opencode_nemo_gym.py)" ;;
        pi)
            [[ -f "${ca}/pi_pod/install-pi.sh.j2" ]] && grep -q "class PiNemoGym" "${ca}/pi_nemo_gym.py" 2>/dev/null \
                && ok "Gym patch 0047: PiNemoGym (uploaded release archive, capture-safe settings)" \
                || bad "Gym patch 0047 missing (PiNemoGym, custom_agents/pi_pod/install-pi.sh.j2)"
            # without it a pi rollout that ran into the context limit reads as ok (pi exits 0 in --mode json)
            grep -q "def pi_end_flags" "${g}/responses_api_agents/harbor_agent/failure_classes.py" 2>/dev/null \
                && ok "Gym patch 0047: failure classes from pi's events (agent/pi.txt)" \
                || bad "Gym patch 0047 missing (pi_end_flags in harbor_agent/failure_classes.py)" ;;
        *) bad "check_gym_agents: no checks for agent '$1'" ;;
    esac
    return "${fails}"
}

#!/bin/bash
# In-pod agent runs on run V's verified Harbor tasks (../../envs/harbor-verified-sft.yaml, Kostas' tasks from
# several campaigns, separate verifier pods, images from rl-registry) with an agent inside each pod instead of
# Terminus-2:
#   AGENT=opencode  run OV: OpenCode 1.17.11 (as runs O / R2), the binary Breithorn mounts into each pod (Gym 0047)
#   AGENT=pi        run PV: the pi coding agent 1.5.2 (Gym 0048), the release tree Breithorn mounts into each pod
#                   (Gym 0049); untested on the cluster
#   AGENT=claude-code  run CV: Claude Code 2.1.326 (Gym 0050-0055; probed with experiments/claude-code-probe), the
#                   release binary fetched once (fetch-claude-code.sh) and uploaded gzipped into each pod; 64K
#                   smokes trained (jobs 1061395, 1061605), a full run is still pending
# (another agent = a new AGENT value: common.sh agent_v_tag / agent_v_agent / agent_v_overlays / check_gym_agents,
# its overlay run-<tag>.yaml, check.py)
# The agent calls the model through the per-pod chisel tunnel (pod 127.0.0.1:8000 -> policy_model), and the token
# IDs come from Gym's token capture at the model endpoint, so these runs go through the Single Controller
# (examples/run_grpo_single_controller.py via ../../run-harbor-osb.sh, as run-s.sh), not run V's Gym entrypoint,
# which has no token capture. Layout as run-v-24k.sh: 24576 tokens on 20 nodes (16 trainer EP32 + 4 vLLM).
# Long context (CTX > 24576, e.g. CTX=65536): run-v-24k.sh's long-context profile (common.sh long_ctx_defaults:
# SFT v0.2, thinking high, CP8 KCP at 64K, 16384 tokens per turn, 100 turns, agent cap 3000 s, OpenCode 2940 s,
# rollout/stall timeouts 7800/8100 s, TTL 4800 s, harbor_mask_agent_timeout, memory cap 0.74, NCCL 60 min, hang
# watchdog 3600/14400 s); SMOKE=1 = 1 step 4 x 8 JOB_TIME 03:00:00, else 10 steps 12:00:00 (Daint limit UNVERIFIED).
#   SMOKE=1 CTX=65536 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh
#   AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh [overrides...]
#   DRY_RUN=1 ...   setup and the job script, no submission (launch.sh)
# Smoke (one step, one sample per trainer data-parallel rank: 32 at 24k = 4 prompts x 8, 2 x 8 at 16k):
#   SMOKE=1 AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh
#   = STEPS=1 PROMPTS=4 JOB_TIME=02:30:00
# 10 steps (the defaults): AGENT=opencode bash experiments/chonk-sc/run-agent-v.sh
# Check first: RUNS=ov (AGENT=opencode), RUNS=pv (AGENT=pi) or RUNS=cv (AGENT=claude-code) bash
# experiments/chonk-sc/check.sh (same env as here).
# Env: AGENT (required), STEPS (10), PROMPTS (8; x 8 rollouts, train_global_batch_size PROMPTS x 8, a multiple
# of the data-parallel size), SANDBOXES (64: harbor_agent concurrency = at most 64 pods on Breithorn; one pod per
# trial, the agent pod is stopped before the verifier pod starts), CTX (24576; 16384 = the 12-node recipe),
# RECIPE (overrides the one CTX picks), JOB_TIME (08:00:00; SMOKE 02:30:00), RUN, GATEWAY_PROXY (0 = sandboxes and
# the tunnel's WebSocket through the ingress gateway, use_server_proxy false as run V; 1 = through the
# OpenSandbox server proxy, the path the tunnel was tested on in runs S/O: use it if the gymtunnel gateway test
# fails), MODEL_CALLS (1: Gym model-call capture under $SCRATCH/harbor-jobs/<run>/model-calls; 0 off),
# OPENCODE_TIMEOUT_SEC (1740; AGENT_CAP - 60 when AGENT_CAP is set), OPENCODE_VERSION, OPENCODE_DEBUG, CHISEL_DIR,
# (chisel and the agent in the pods: Breithorn's read-only mounts under /opt/rl-tools, Gym 0047/0049); for
# AGENT=pi: PI_TIMEOUT_SEC (as OPENCODE_TIMEOUT_SEC), PI_VERSION; for AGENT=claude-code: CC_TIMEOUT_SEC (as
# OPENCODE_TIMEOUT_SEC), CC_VERSION, CC_PROFILE (default or bare), CC_DIR ($SCRATCH/claude-code)
# ($SCRATCH/chisel), the long-context knobs (common.sh: SFT_CKPT, THINKING, CP, KCP, CAP, NCCL_MIN, MAX_NEW, MAX_TURNS,
# AGENT_CAP, TTL, MASK_AGENT_TIMEOUT, MASK_OVERFLOW, KCP_MEGATRON_DIR, KCP_FLA_DIR, KCP_EDF_NAME, HANG_*),
# and as run-v.sh: VERIFIED_STORE, HARBOR_ROOT ($SCRATCH/osb/harbor_verified), COLLECT_ARGS, EXCLUDE_CAMPAIGNS,
# EXCLUDE_TASKS, RL_REGISTRY_CRED_FILE (~/.rl_registry_cred), OPENSANDBOX_API_KEY_FILE (~/.opensandbox_token);
# as run-s.sh: SFT_MODEL_SRC, SFT_EDF_NAME, ACCOUNT, PARTITION (normal), DRIVER_RETRIES (0), OVERLAY;
# REGISTRY_CHECK (1: before setup, the robot account must be able to pull the first task's images, else no
# submission; pod: also one test pod with the task's resources, up to 600 s; 0: skip; ../osb-reach/registry_precheck.sh).
# Harbor trial dirs: $SCRATCH/harbor-jobs/<RUN> (as run V). Pods carry run=<RUN> (reaper filter).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
APERTUS2_DIR="${ROOT}/backends/nemo-rl/apertus2"
source "${ROOT}/sites/load.sh" && rl_infra_load "${APERTUS2_DIR}/env"
source "${HERE}/common.sh"
# AGENTS=opencode was silently ignored once (job 5316090 ran Terminus-2)
[[ -z "${AGENTS:-}" ]] || { echo "ERROR: AGENTS is set; the variable is AGENT" >&2; exit 1; }
TAG="$(agent_v_tag "${AGENT:-}")" || exit 1
agent_v_defaults || exit 1
if [[ -n "${AGENT_V_DP}" ]] && (( (PROMPTS * 8) % AGENT_V_DP )); then
    echo "ERROR: PROMPTS=${PROMPTS} gives a batch of $(( PROMPTS * 8 )), not a multiple of the trainer's data-parallel size ${AGENT_V_DP}" >&2
    exit 1
fi
(( SANDBOXES >= PROMPTS * 8 )) || {
    echo "ERROR: SANDBOXES=${SANDBOXES} < PROMPTS x 8 = $(( PROMPTS * 8 )): rollouts would queue inside Gym and eat rollout_timeout_s" >&2
    exit 1; }

# --- inputs (files are tested, never read) ---------------------------------------------------------------------
[[ -r "${SFT_MODEL_SRC}/config.json" && -r "${SFT_MODEL_SRC}/model.safetensors.index.json" ]] \
    || { echo "ERROR: ${SFT_MODEL_SRC} has no readable config.json + model.safetensors.index.json (an HF export)" >&2; exit 1; }
# absolute paths: HOME inside the container is not yours
CRED="$(realpath -m "${RL_REGISTRY_CRED_FILE:-${HOME}/.rl_registry_cred}")"
KEY="$(realpath -m "${OPENSANDBOX_API_KEY_FILE:-${HOME}/.opensandbox_token}")"
[[ -r "${CRED}" && -s "${CRED}" ]] || { echo "ERROR: no registry credentials file ${CRED} (username:password, chmod 600)" >&2; exit 1; }
[[ -r "${KEY}" ]] || { echo "ERROR: no OpenSandbox key file ${KEY}" >&2; exit 1; }
export RL_REGISTRY_CRED_FILE="${CRED}" OPENSANDBOX_API_KEY_FILE="${KEY}"

# --- the verified tasks, as run-v.sh (same store, same filters, same defaults: keep them in step) -------------
TASK_STORE="${VERIFIED_STORE:-/capstor/store/cscs/swissai/infra01/datasets/swe_and_agentic_gym_datasets/harbor_verified_tasks}"
if [[ ! -s "${HARBOR_ROOT}/train.jsonl" ]]; then
    [[ -d "${TASK_STORE}" ]] || { echo "ERROR: no task store ${TASK_STORE}; set VERIFIED_STORE" >&2; exit 1; }
    if [[ -z "${COLLECT_ARGS+x}" ]] && ! compgen -G "${TASK_STORE}/*/tasks.csv" > /dev/null; then
        echo "note: no <campaign>/tasks.csv in ${TASK_STORE}: taking every task (--include-unverified), as on HANGANG"
        export COLLECT_ARGS=--include-unverified
    fi
    echo "== preparing the verified tasks in ${HARBOR_ROOT}"
    bash "${ROOT}/envs/prepare_harbor_verified_tasks.sh" "${TASK_STORE}" "${HARBOR_ROOT}"
fi
echo "== tasks: $(wc -l < "${HARBOR_ROOT}/train.jsonl") train, $(wc -l < "${HARBOR_ROOT}/validation.jsonl") validation rows in ${HARBOR_ROOT}"
EXCLUDE_CAMPAIGNS="${EXCLUDE_CAMPAIGNS-swe-rebench-v2\.1-rl}"
EXCLUDE_TASKS="${EXCLUDE_TASKS-gui-cs__terminal-gui-4860|ml-explore__mlx-3604|floci-io__floci-1932|modin-project__modin-(6369|7193)}"
TRAIN="${HARBOR_ROOT}/train.jsonl"
DROP=()
[[ -z "${EXCLUDE_CAMPAIGNS}" ]] || DROP+=(-e "\"instance_id\": \"verified::[^\"]*(${EXCLUDE_CAMPAIGNS})__")
[[ -z "${EXCLUDE_TASKS}" ]] || DROP+=(-e "\"instance_id\": \"verified::[^\"]*__(${EXCLUDE_TASKS})\"")
if (( ${#DROP[@]} )); then
    # a file of its own: run V (train-filtered.jsonl) may be using the same HARBOR_ROOT with other filters
    TRAIN="${HARBOR_ROOT}/train-filtered-agent-v.jsonl"
    grep -vE "${DROP[@]}" "${HARBOR_ROOT}/train.jsonl" > "${TRAIN}.tmp" || true
    [[ -s "${TRAIN}.tmp" ]] || {
        echo "ERROR: EXCLUDE_CAMPAIGNS='${EXCLUDE_CAMPAIGNS}' EXCLUDE_TASKS='${EXCLUDE_TASKS}' leave no training rows" >&2
        rm -f "${TRAIN}.tmp"; exit 1; }
    mv -f "${TRAIN}.tmp" "${TRAIN}"
    echo "== left out $(( $(wc -l < "${HARBOR_ROOT}/train.jsonl") - $(wc -l < "${TRAIN}") )) rows" \
        "(campaigns /${EXCLUDE_CAMPAIGNS}/, tasks /${EXCLUDE_TASKS}/); training on ${TRAIN}"
fi

# fail before setup and the queue wait when Breithorn cannot pull the task images (job 5374114: every pod
# Pending, 10.7 node-hours for nothing); REGISTRY_CHECK=0 skips it, =pod also starts one test pod
bash "${ROOT}/experiments/osb-reach/registry_precheck.sh" "${TRAIN}" "${HARBOR_ROOT}" || exit 1

# exported before use (run-harbor-osb.sh reads RUN; the pods' run label is NEMO_GYM_RUN_ID = RUN)
export RUN="${RUN:-chonk-sc-${TAG}-${AGENT}-ctx${CTX}-$(date +%Y%m%d-%H%M%S)}"
JOBS="${SCRATCH}/harbor-jobs/${RUN}"
mkdir -p "${JOBS}"
export JOB_TIME="${JOB_TIME:-08:00:00}"
export PARTITION="${PARTITION:-normal}" DRIVER_RETRIES="${DRIVER_RETRIES:-0}"
# the task file names ${oc.env:RL_USER_HOME} (run.sh exports it; run-harbor-osb.sh does not); the overrides
# below replace both uses, this keeps the config resolvable either way
export RL_USER_HOME="${HOME}"

# --- setup as run-s.sh: apertus-common overlay in an EDF of its own, the SFT export staged -----------------------
export APERTUS_COMMON_URL="${APERTUS_COMMON_URL:-git@github.com:swiss-ai/apertus-common.git}"
source "${PIN_FILE}"
export EDF_NAME="${SFT_EDF_NAME}"
# KCP (long context): the Megatron checkout with the KCP path, fla 0.5.2 and an EDF of their own (common.sh kcp_setup)
if [[ "${KCP}" == 1 ]]; then kcp_setup "${SFT_EDF_NAME}" || exit 1; fi
long_ctx_job_env
export PYDEPS_APERTUS2="${SCRATCH}/nemo-rl-apertus2/apertus-common-${APERTUS_COMMON_COMMIT:0:9}"
export PYDEPS="${PYDEPS}-apertus2"
mkdir -p "${OUT}/configs"
LOG="${OUT}/setup-${RUN}.log"
echo "== setup for run ${TAG^^} (EDF ${EDF_NAME}, overlay ${PYDEPS}; log ${LOG})"
# as run.sh: an export that carries the modeling code (MODELING_SOURCE) is linked whole, one without (v0.1) gets a copy
# of its small files, linked shards and the code; run V stages the same export to the same directory the same way
STAGE=(CHONK_MODEL_STAGE=copy STAGE_SHARDS=link)
[[ ! -f "${SFT_MODEL_SRC}/MODELING_SOURCE" ]] || STAGE=(CHONK_MODEL_STAGE=link)
{ SKIP_VLLM_IMAGE=1 bash "${APERTUS2_DIR}/setup.sh" \
    && env SRC="${SFT_MODEL_SRC}" DST="${SFT_MODEL_DIR}" "${STAGE[@]}" \
       bash "${ROOT}/experiments/grpo-chonk/stage_model.sh"; } > "${LOG}" 2>&1 \
    || { tail -n 20 "${LOG}"; echo "ERROR: setup or model staging failed, see ${LOG}" >&2; exit 1; }
grep -E '^== (EDF|done|image)|^WARNING' "${LOG}" || true

# fail before the queue wait: capture patches incl. 0035-0042, the agent and the tunnel, the NeMo-RL clone
for chk in check_gym_capture "check_gym_agents ${AGENT}" check_nemo_rl_clone; do
    if ! out="$(${chk})"; then
        grep -v '^  ok' <<< "${out}" >&2
        echo "ERROR: ${chk% *} failed (bash ${APERTUS2_DIR}/setup-gym.sh; bash ${HERE}/check.sh)" >&2
        exit 1
    fi
done
for v in OPENCODE_VERSION OPENCODE_TIMEOUT_SEC OPENCODE_DEBUG PI_VERSION PI_TIMEOUT_SEC CC_VERSION CC_TIMEOUT_SEC \
        CC_PROFILE; do
    [[ -z "${!v:-}" ]] || export "${v}"
done
# chisel: the client runs on the node (linux arm64, from CHISEL_DIR); the server in each pod is Breithorn's mounted
# linux amd64 binary (tunnel-sc.yaml). The agent is mounted too: nothing to fetch for the pods.
export CHISEL_VERSION="${CHISEL_VERSION:-1.12.0}" CHISEL_DIR="${CHISEL_DIR:-${SCRATCH}/chisel}"
bash "${APERTUS2_DIR}/fetch-chisel.sh"
if [[ "${AGENT}" == claude-code ]]; then
    export CC_DIR="${CC_DIR:-${SCRATCH}/claude-code}"
    bash "${APERTUS2_DIR}/fetch-claude-code.sh"
fi
if [[ "${AGENT}" == claude-code ]]; then
    echo "== Claude Code in the pods: uploaded gzipped from ${CC_DIR} (not mounted)"
else
    echo "== chisel client on the node: ${CHISEL_DIR}/chisel_${CHISEL_VERSION}_linux_arm64; in the pods (mounted):" \
         "/opt/rl-tools/chisel/${CHISEL_VERSION}/chisel, ${AGENT} $([[ "${AGENT}" == pi ]] \
         && echo "/opt/rl-tools/pi/${PI_VERSION:-1.5.2}/pi/pi" || echo "/opt/rl-tools/opencode/${OPENCODE_VERSION:-1.17.11}/opencode")"
fi

TASK_COPY="${OUT}/configs/${RUN}-task.yaml"
verified_task_copy "${TASK_COPY}"
mapfile -t AGENT_OVERLAYS < <(agent_v_overlays "${AGENT}")
# long context: the job reads a copy of the recipe made now, not the worktree's file (common.sh recipe_copy)
if [[ "${LONG_CTX}" == 1 ]]; then AGENT_V_RECIPE="$(recipe_copy "${AGENT_V_RECIPE}" "${RUN}")" || exit 1; fi
export SCRIPT="${SC_SCRIPT}" RECIPE="${AGENT_V_RECIPE}" TASK="${TASK_COPY}"
export OVERLAY="${AGENT_OVERLAYS[*]}${OVERLAY:+ ${OVERLAY}}"
export NEMO_GYM_WORKLOAD="${NEMO_GYM_WORKLOAD:-chonk-${AGENT}-verified-sc}"
agent_v_args "${TRAIN}" "${CRED}" "${JOBS}"
echo "== run ${RUN} (${AGENT}): ${STEPS} steps, ${PROMPTS} x 8 rollouts, at most ${SANDBOXES} sandboxes, ${CTX} tokens," \
     "recipe $(basename "${AGENT_V_RECIPE}"), GATEWAY_PROXY=${GATEWAY_PROXY}, model calls ${MODEL_CALLS}; Harbor trial dirs in ${JOBS}"
AGENT_END="${OPENCODE_TIMEOUT_SEC:-1740}"
[[ "${AGENT}" != pi ]] || AGENT_END="${PI_TIMEOUT_SEC:-1740}"
[[ "${AGENT}" != claude-code ]] || AGENT_END="${CC_TIMEOUT_SEC:-1740}"
[[ "${LONG_CTX}" != 1 ]] || echo "== long context: SFT ${SFT_CKPT}, thinking ${THINKING}, CP ${CP} KCP ${KCP}, cap ${CAP:-none}," \
     "${MAX_NEW:-6144} tokens per turn, ${MAX_TURNS:-30} turns, agent cap ${AGENT_CAP:-1800} s, ${AGENT} ends at ${AGENT_END} s," \
     "TTL ${TTL:-3600} s, mask agent timeout ${MASK_AGENT_TIMEOUT}, mask overflow ${MASK_OVERFLOW}, recipe copy ${AGENT_V_RECIPE}"
# Claude Code sends no prompt of CTX - MAX_NEW - 3000 tokens or more (its blocking limit, terminal_reason
# blocking_limit), and its own first prompt is about 4.8K Apertus 2 tokens: below 64K most episodes end there
# (smokes on Santis: 8 of 16 trials at 16K, 28 of 32 at 32K, 1 of 32 at 64K).
if [[ "${AGENT}" == claude-code ]] && (( CTX < 65536 )); then
    echo "== WARNING: Claude Code stops before a prompt of ${CTX} - ${MAX_NEW:-6144} - 3000 tokens; with its first" \
         "prompt of about 4.8K tokens that leaves about $(( CTX - ${MAX_NEW:-6144} - 3000 - 4800 )) tokens per episode." \
         "Run CV trains at CTX=65536 (README.md, run CV)." >&2
fi
exec bash "${ROOT}/run-harbor-osb.sh" "${AGENT_V_ARGS[@]}" "$@"

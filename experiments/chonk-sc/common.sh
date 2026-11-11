# Shared by check.sh, run-a.sh, run-b.sh and run-s.sh (sourced, not run). Defines how runs A, B and S compose
# their configs (the same files, in the same order, as ../../run.sh and ../../run-harbor-osb.sh
# write them) and the static checks of the NeMo-RL clone and the Gym checkout.
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
SFT_MODEL_SRC="${SFT_MODEL_SRC:-/capstor/store/cscs/swissai/infra01/hf-checkpoints/chonk-lc-16k-v0.1-ep1/iter_0019374}"
SFT_MODEL_DIR="${SCRATCH}/models/chonk-lc-16k-v0.1-ep1-iter_0019374"
# Its own EDF: run S needs the apertus-common overlay in the container spec, which setup.sh bakes into
# ~/.edf/<EDF_NAME>.toml; a shared EDF would change under jobs of the other runs waiting in the queue.
SFT_EDF_NAME="${SFT_EDF_NAME:-nemo-rl-apertus2-sft}"

# sft_task_copy <out.yaml>: the SFT task file without its `defaults:` (it would merge the 8075 recipe
# over the one the launcher puts first) and `script:` (not a config key) lines, as ../../run.sh does
# with yq; both are single top-level lines in that file.
sft_task_copy() {
    grep -vE '^(defaults|script):' "${SFT_TASK}" > "$1"
    grep -q '^env:' "$1" && ! grep -qE '^(defaults|script):' "$1" || { echo "sft_task_copy: unexpected ${SFT_TASK}" >&2; return 1; }
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
        *) echo "compose_cfg: run must be a, b, s, o or w" >&2; return 1 ;;
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
    # NeMo-RL patch 0034: a continuation after a truncated Apertus 2 burst gets the context-length 400
    # (without it: assert -> HTTP 500 retried by Gym, job 5303503); run S needs it
    grep -q "class TruncatedBurstError" "${d}/nemo_rl/data/apertus2_format.py" 2>/dev/null \
        && ok "NeMo-RL patch 0034: TruncatedBurstError in apertus2_format.py" \
        || bad "NeMo-RL patch 0034 missing (TruncatedBurstError in nemo_rl/data/apertus2_format.py); run backends/nemo-rl/apertus2/setup.sh"
    # NeMo-RL patch 0035: logprob outlier dump (off unless run-s-lpdump.yaml sets NRL_LOGPROB_DUMP_DIR)
    grep -q "NRL_LOGPROB_DUMP_DIR" "${d}/nemo_rl/algorithms/loss/logprob_dump.py" 2>/dev/null \
        && ok "NeMo-RL patch 0035: nemo_rl/algorithms/loss/logprob_dump.py" \
        || bad "NeMo-RL patch 0035 missing (nemo_rl/algorithms/loss/logprob_dump.py); run backends/nemo-rl/apertus2/setup.sh"
    grep -q "def _invalid_apertus2_history" "${d}/nemo_rl/models/generation/vllm/vllm_worker_async.py" 2>/dev/null \
        && ok "NeMo-RL patch 0036: an unencodable Apertus 2 chat history gets HTTP 400" \
        || bad "NeMo-RL patch 0036 missing (_invalid_apertus2_history in vllm_worker_async.py); run backends/nemo-rl/apertus2/setup.sh"
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

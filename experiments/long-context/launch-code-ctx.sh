#!/bin/bash
# bash launch-code-ctx.sh <context tokens> [extra hydra overrides...]
# Competitive coding on Gym's code_gen with the unit tests in OpenSandbox pods (rl-infra
# experts/code/archive/code-gen-osb-sft.yaml: PROMPTS x 16 answers, one optimizer step, lr 2e-6, no thinking budget, reward 1 only
# if every test passes, LiveCodeBench validation 10 times over) on the SFT v0.2 checkpoint, with a longer context:
# prompt plus answer, vLLM's limit and the trainer's packed micro-batch are <context tokens>.
#
#   NODES      32 (default; 16 trainer + 16 vLLM), 16 (8 + 8) or 12 (8 + 4, for debug runs when few nodes are free).
#   HSN        1 (default, the rule for every run): Ray on the first Slingshot interface (400 Gbit) of each node.
#              0: on the address the node name resolves to, the 1 Gbit management Ethernet (36 times slower; the
#              batch reaches the trainer nodes through Ray).
#   PROMPTS    prompts per step (default 128, the task file's); 16 answers each, one optimizer step, so the global
#              batch is PROMPTS x 16.
#   CP         context-parallel size of the trainer (default 4): each GPU holds <context tokens>/CP tokens of a
#              micro-batch.
#   KCP        1: the KCP scheme for the KDA layers (Megatron copy ~/projects/megatron-lm-moe-apertus2-kcp, branch
#              rl-kcp, with flash-linear-attention 0.5.2 from fla-0.5.2/ ahead of the image's); any CP that divides
#              the ranks of a pipeline stage (16 with 8 trainer nodes, 32 with 16).
#              0 (default): the pinned Megatron's scheme (heads split over the CP ranks): CP 1, 2 or 4.
#   THINKING   high (default; the user's rule since 2026-11-19: high for training, its validations and every
#              evaluation) or medium (the task file's frame): the system frame of every prompt, in training and in
#              validation alike (policy.generation.system_prompt_file, shared/frames/apertus2-system[-high].json).
#   CAP        PyTorch's share of each trainer GPU (default 0.74; the task file says 0.78, set for 16K per GPU).
#   STALENESS  grpo.async_grpo.max_trajectory_age_steps (default 2, as the earlier coding runs).
#   STEPS      default: the task file's 184 (one pass over the training file).
#   VAL        full (default; as the task file: start, every 20 steps and end, 10 answers per problem),
#              end (only at the end), none.   VAL_REPEATS overrides the 10.
#   CKPT       file (default; the task file's settings: every 50 steps, with optimizer state),
#              weights (the same without optimizer state), none.
#   SAVE_BY    checkpointing.checkpoint_must_save_by (D:HH:MM:SS); set it with JOB_TIME.
#   DEBUG      1 (default): a debug run, its name gets the suffix -DEBUG. 0: a full run. Both log to W&B
#              (entity apertus, project chonk-rl), a full run refuses to start without the key file.
#   RUN, JOB_TIME (default 03:00:00), SBATCH_MORE and DRY_RUN pass through.
#
# Departures from the task file besides the context: loss_fn.force_on_policy_ratio=true (with one optimizer step
# the separate log-prob pass changes nothing and is skipped, as in the STEM runs), TORCHINDUCTOR_COMPILE_THREADS=1
# (the kernel-compile hang of job 1263111), vLLM max_num_seqs 256.
# vLLM refuses a limit above the export's max_position_embeddings (16,384) unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is
# set; the model has no position table, so the value is only that check.
set -euo pipefail
SCRATCH=${SCRATCH:-/scratch/${USER}}   # generated pin, container spec and task file go to ${SCRATCH}/long-context
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA="$(cd "${HERE}/../.." && pwd)"   # the rl-infra worktree
CTX=${1:?context length in tokens, e.g. 32768}
shift
NODES=${NODES:-32}
CP=${CP:-4}
PAD=$(( CP > 1 ? 2 * CP : 2 ))
(( CTX % PAD == 0 )) || { echo "ERROR: ${CTX} is not a multiple of 2 x CP (${PAD})" >&2; exit 1; }
KCP=${KCP:-0}
PROMPTS=${PROMPTS:-128}
case "${NODES}" in
    32) export RECIPE=${HERE}/recipes/code-sftv02-32n.yaml; TRAIN_NODES=16 ;;
    31) export RECIPE=${HERE}/recipes/code-sftv02-31n.yaml; TRAIN_NODES=16 ;;   # 16 + 15, for HEAD=1 on 32 nodes
    16) export RECIPE=${HERE}/recipes/code-sftv02-16n.yaml; TRAIN_NODES=8 ;;
    12) export RECIPE=${HERE}/recipes/code-sftv02-12n.yaml; TRAIN_NODES=8 ;;
    *)  echo "ERROR: NODES must be 12, 16 or 32" >&2; exit 1 ;;
esac

RANKS=$(( TRAIN_NODES * 4 / 2 ))   # ranks of one pipeline stage: 4 GPUs per trainer node, PP2
(( RANKS % CP == 0 )) || { echo "ERROR: CP ${CP} does not divide the ${RANKS} ranks of a pipeline stage" >&2; exit 1; }

EXTRA=()
if [[ "${KCP}" == 1 ]]; then
    # The Megatron copy with the KCP path, at its own commit: setup.sh checks out the pin's commit otherwise.
    export MEGATRON_DIR=${KCP_MEGATRON_DIR:-${PROJECTS_DIR:-${HOME}/projects}/megatron-lm-moe-apertus2-kcp}
    KCP_COMMIT=$(git -C "${MEGATRON_DIR}" rev-parse rl-kcp)
    mkdir -p ${SCRATCH}/long-context/launch
    export PIN_FILE=${SCRATCH}/long-context/launch/pin-kcp-main
    sed "s/^MEGATRON_COMMIT=.*/MEGATRON_COMMIT=${KCP_COMMIT}/" "${INFRA}/backends/nemo-rl/apertus2/pin" > "${PIN_FILE}"
    # The stack's container spec with flash-linear-attention 0.5.2 (fla.ops.cp) ahead of the image's copy.
    export CONTAINER_SPEC=${SCRATCH}/long-context/launch/edf-kcp-main.toml.template
    # A missing directory would only show inside the job, as the image's older flash-linear-attention without fla.ops.cp.
    FLA_DIR=${KCP_FLA_DIR:-${PROJECTS_DIR:-${HOME}/projects}/fla-0.5.2}
    [[ -d "${FLA_DIR}/fla/ops/cp" ]] || {
        echo "ERROR: ${FLA_DIR} has no flash-linear-attention 0.5.2 (fla/ops/cp); set KCP_FLA_DIR" >&2; exit 1; }
    sed "s|:@@MEGATRON_DIR@@:|:@@MEGATRON_DIR@@:${FLA_DIR}:|" \
        "${INFRA}/backends/nemo-rl/apertus2/edf.toml.template" > "${CONTAINER_SPEC}"
    grep -q "fla-0.5.2" "${CONTAINER_SPEC}" || { echo "ERROR: could not add fla-0.5.2 to PYTHONPATH" >&2; exit 1; }
    EXTRA+=(++policy.megatron_cfg.model_overrides.linear_attention_cp_impl=kcp)
elif (( CP != 1 && CP != 2 && CP != 4 )); then
    echo "ERROR: without KCP=1 the KDA layers split their 28 heads over the CP ranks: CP must be 1, 2 or 4" >&2; exit 1
fi
case "${THINKING:-high}" in
    high)   EXTRA+=(policy.generation.system_prompt_file=${INFRA}/shared/frames/apertus2-system-high.json) ;;
    medium) EXTRA+=(policy.generation.system_prompt_file=${INFRA}/shared/frames/apertus2-system.json) ;;
    *)      echo "ERROR: THINKING must be high or medium" >&2; exit 1 ;;
esac
echo "Thinking ${THINKING:-high} in training and validation"
if [[ -n "${STEPS:-}" ]]; then EXTRA+=(grpo.max_num_steps=${STEPS}); fi
case "${VAL:-full}" in
    full) ;;
    end)  EXTRA+=(grpo.val_at_start=false grpo.val_at_end=true grpo.val_period=100000) ;;
    none) EXTRA+=(grpo.val_at_start=false grpo.val_at_end=false grpo.val_period=100000) ;;
    *)    echo "ERROR: VAL must be full, end or none" >&2; exit 1 ;;
esac
if [[ -n "${VAL_REPEATS:-}" ]]; then EXTRA+=(grpo.val_num_generations_per_prompt=${VAL_REPEATS}); fi
case "${CKPT:-file}" in
    file)    ;;
    weights) EXTRA+=(checkpointing.save_optimizer=false) ;;
    none)    EXTRA+=(checkpointing.enabled=false) ;;
    *)       echo "ERROR: CKPT must be file, weights or none" >&2; exit 1 ;;
esac
# CKPT_EVERY: steps between checkpoints (the task file says 50; a 512-prompt pass has only 46 steps).
if [[ -n "${CKPT_EVERY:-}" ]]; then EXTRA+=(checkpointing.save_period=${CKPT_EVERY}); fi
# CKPT_FIRST=1 (default): a step that saves and validates saves first (NeMo-RL patch 0047). A failure during the
# validation then loses nothing: job 1298910 failed in the validation of step 40 and resumed from step 20.
if [[ "${CKPT_FIRST:-1}" == 1 ]]; then EXTRA+=(++checkpointing.save_before_validation=true); fi
if [[ -n "${SAVE_BY:-}" ]]; then EXTRA+=("checkpointing.checkpoint_must_save_by=\"'${SAVE_BY}'\""); fi
# PODS: OpenSandbox pods for the unit tests (the task file says 64, set for 128 prompts x 16); twice as many
# submissions are kept in flight, so that no pod waits for work. 256 pods started cleanly and scaled linearly on
# 2026-11-17 (pod_capacity.py, jobs 1292888 and 1292987).
CODE_GEN=env.nemo_gym.code_gen.resources_servers.code_gen
# SANDBOX=<endpoint file>: the unit tests run in rl-infra's sandbox service on a node of our own instead of the
# pods at CSCS (no Kubernetes). Start the service first, as a job of its own that outlives the run:
#   SANDBOX_ENDPOINT_FILE=<endpoint file> SANDBOX_WORKERS=192 JOB_TIME=2-00:00:00 \
#       bash <rl-infra>/backends/nemo-rl/sandbox/start.sh
# The task file is then experts/code/archive/code-gen-osb-sft.yaml without its pod pool (written to launch/), every other setting
# unchanged; the code_gen server takes the sandbox service when there is no pod pool. SANDBOX_WORKERS (default 192)
# must be the service's --workers: that many submissions are in flight.
if [[ -n "${SANDBOX:-}" ]]; then
    [[ -z "${PODS:-}" ]] || { echo "ERROR: PODS and SANDBOX are two different test runners, give one" >&2; exit 1; }
    # SANDBOX_JOB=<job id of start.sh>: the sandbox job is still queued; the run then starts after it (the server
    # writes the endpoint file seconds after it starts, Gym reads it minutes into the run).
    if [[ -n "${SANDBOX_JOB:-}" ]]; then
        SBATCH_MORE="${SBATCH_MORE:-} --dependency=after:${SANDBOX_JOB}"
    else
        [[ -s "${SANDBOX}" ]] || { echo "ERROR: ${SANDBOX} is missing: the sandbox service is not up (start.sh), or give SANDBOX_JOB" >&2; exit 1; }
    fi
    mkdir -p ${SCRATCH}/long-context/launch
    TASK=${SCRATCH}/long-context/launch/code-gen-sandbox-sft.yaml
    sed '/^          opensandbox:/,/^            num_pods:/d' "${INFRA}/experts/code/archive/code-gen-osb-sft.yaml" > "${TASK}"
    if grep -q "^ *opensandbox:" "${TASK}" || ! grep -q "^          reward_mode: all_tests" "${TASK}"; then
        echo "ERROR: could not take the pod pool out of the task file (${TASK})" >&2; exit 1
    fi
    EXTRA+=(${CODE_GEN}.sandbox_endpoint_file=${SANDBOX} ${CODE_GEN}.num_processes=${SANDBOX_WORKERS:-192})
else
    TASK=${INFRA}/experts/code/archive/code-gen-osb-sft.yaml
    EXTRA+=(${CODE_GEN}.opensandbox.api_key_file=${OPENSANDBOX_TOKEN_FILE:-${HOME}/.opensandbox_token})
    if [[ -n "${PODS:-}" ]]; then
        EXTRA+=(${CODE_GEN}.opensandbox.num_pods=${PODS} ${CODE_GEN}.num_processes=$(( 2 * PODS )))
    fi
fi
# Every run logs to W&B (entity apertus, project chonk-rl). The key is in ~/.wandb-api-key (hyphens: the user's
# file; rl-infra's default name is ~/.wandb_api_key, so WANDB_KEY_FILE names it) and is read at job start.
# DEBUG=1 (default) marks a debug run: its name ends with -DEBUG. DEBUG=0 is a full run and needs the key.
# launch.sh only checks that the key file is readable; an empty one would fail inside the job.
DEBUG=${DEBUG:-1}
WANDB_KEY=${WANDB_KEY_FILE:-${HOME}/.wandb_api_key}
if [[ -s "${WANDB_KEY}" ]]; then
    export WANDB_PROJECT=chonk-rl WANDB_ENTITY=apertus WANDB_KEY_FILE="${WANDB_KEY}"
elif [[ "${DEBUG}" == 0 && "${NO_WANDB_OK:-0}" != 1 ]]; then
    # NO_WANDB_OK=1: start a full run without W&B anyway (the user's go-ahead with the key file still empty); its
    # TensorBoard scalars are uploaded afterwards with submit_wandb_sync.sh.
    echo "ERROR: a full run must log to W&B, but ~/.wandb-api-key is missing or empty (NO_WANDB_OK=1 overrides)" >&2; exit 1
else
    echo "WARNING: ~/.wandb-api-key is missing or empty: this debug run logs to TensorBoard only" >&2
fi

export RUN=${RUN:-sftv02-code-osb-$(( CTX / 1024 ))k-cp${CP}-${NODES}n}
if [[ "${DEBUG}" == 1 && "${RUN}" != *-DEBUG ]]; then export RUN="${RUN}-DEBUG"; fi
# The run reads a copy of the recipe made now (its defaults made absolute), not the file in this worktree: the job,
# its driver_retry attempts and a requeue load the config anew each time they start, and the 128K run 1299941 read
# code-sftv02-31n.yaml again after an edit (16 -> 15 vLLM nodes), built 17 trainer nodes and failed every retry.
mkdir -p ${SCRATCH}/long-context/launch/recipes
RECIPE_COPY=${SCRATCH}/long-context/launch/recipes/${RUN}-$(date +%Y%m%d-%H%M%S).yaml
RECIPE_DEFAULTS=$(sed -n 's/^defaults: *//p' "${RECIPE}")
[[ -n "${RECIPE_DEFAULTS}" ]] || { echo "ERROR: ${RECIPE} has no one-line defaults: entry to copy" >&2; exit 1; }
[[ "${RECIPE_DEFAULTS}" == /* ]] || RECIPE_DEFAULTS=$(realpath -m "$(dirname "${RECIPE}")/${RECIPE_DEFAULTS}")
[[ -f "${RECIPE_DEFAULTS}" ]] || { echo "ERROR: ${RECIPE_DEFAULTS} (defaults of ${RECIPE}) does not exist" >&2; exit 1; }
sed "s|^defaults: .*|defaults: ${RECIPE_DEFAULTS}|" "${RECIPE}" > "${RECIPE_COPY}"
export RECIPE=${RECIPE_COPY}
echo "Recipe: ${RECIPE}"
export JOB_TIME=${JOB_TIME:-03:00:00}
# The hang watchdog (backends/nemo-rl/hang_watch.sh) stops a run whose driver log has not changed for 45 minutes,
# or for 2 hours during a validation. A 64K answer takes close to an hour to generate and a 128K one about two, so
# from 64K on the limits are raised (4 h, and 6 h during a validation); a real hang then costs that long.
if (( CTX >= 65536 )); then
    export HANG_DUMP_AFTER_S=${HANG_DUMP_AFTER_S:-3600} HANG_KILL_AFTER_S=${HANG_KILL_AFTER_S:-14400}
    export HANG_VAL_DUMP_AFTER_S=${HANG_VAL_DUMP_AFTER_S:-7200} HANG_VAL_KILL_AFTER_S=${HANG_VAL_KILL_AFTER_S:-21600}
fi
# NCCL_MIN: minutes a collective of the trainer waits for its peers (PyTorch and Megatron Bridge: 10). The trainer
# nodes receive their shard of the batch one after another: 8192 answers padded to 64K arrive 5 to 10 minutes apart,
# padded to 128K more than 24 minutes apart, and the ranks that start first then time out in the first collective
# of the step (the 128K full runs 1293158 and 1293163). From 64K on the limit is 60 minutes.
# Needs NeMo-RL patch 0040 (NRL_NCCL_TIMEOUT_MINUTES).
if (( CTX >= 65536 )); then NCCL_MIN=${NCCL_MIN:-60}; fi
if [[ -n "${NCCL_MIN:-}" ]]; then
    EXTRA+=("++policy.megatron_cfg.env_vars.NRL_NCCL_TIMEOUT_MINUTES=\"'${NCCL_MIN}'\"")
fi
# HSN=1: ray.sub resolves "<node>-hsn0" instead of the node name (RAY_NODE_NAME_SUFFIX, NeMo-RL patch 0041), so
# Ray and everything that takes its address from Ray (object transfers, vLLM and Gym servers, the trainer's
# rendezvous) use Slingshot. Measured between two nodes: 4,178 MB/s against 117 MB/s for an 8 GiB Ray object
# (jobs 1297629 and 1297631). sbatch passes the variable to the job with the rest of the environment.
if [[ "${HSN:-1}" == 1 ]]; then
    # A ray.sub without the commit ignores the variable and the run is on the slow network without a word.
    RAY_SUB=${NEMO_RL_DIR:-${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2}/ray.sub
    grep -q RAY_NODE_NAME_SUFFIX "${RAY_SUB}" || {
        echo "ERROR: ${RAY_SUB} does not read RAY_NODE_NAME_SUFFIX: rerun setup.sh (NeMo-RL patch 0041)" >&2; exit 1; }
    export RAY_NODE_NAME_SUFFIX=-hsn0
    echo "Ray on Slingshot: RAY_NODE_NAME_SUFFIX=${RAY_NODE_NAME_SUFFIX}"
else
    unset RAY_NODE_NAME_SUFFIX
    echo "WARNING: HSN=0, Ray on the 1 Gbit management Ethernet" >&2
fi
# HEAD=1: one node more than NODES, and the first node runs only Ray's control plane and the driver (ray.sub
# DEDICATED_RAY_HEAD=1; the later --nodes replaces launch.sh's, the driver stays sized for NODES). Without it the
# driver shares a node with whatever Ray places there: vLLM workers in jobs 1298910 and 1299180, trainer ranks in
# the 128K job 1299189, where the driver's batch (8,192 answers padded to 131,072 tokens), four trainer ranks with
# their copies of it and the CPU-offloaded optimizer state ran the node out of host memory in the first training
# step (a node has about 480 GB of CPU memory; the 870 GB that Slurm counts include the GPUs' 4 x 96 GB).
# OFFHEAD=1 (the user's choice over HEAD=1, no extra node): the trainer stays off the driver's node, which then
# hosts vLLM workers (env.train_off_head and ray.sub's off_head resource, NeMo-RL patch 0042, applied by setup.sh to the clone
# NeMo-RL patch 0042).
if [[ "${OFFHEAD:-0}" == 1 ]]; then
    NEMO_RL_DIR=${NEMO_RL_DIR:-${PROJECTS_DIR:-${HOME}/projects}/nemo-rl-apertus2}
    grep -q off_head "${NEMO_RL_DIR}/ray.sub" && grep -q train_off_head "${NEMO_RL_DIR}/nemo_rl/algorithms/grpo.py" || {
        echo "ERROR: OFFHEAD=1 needs NeMo-RL patch 0042, which ${NEMO_RL_DIR} does not have" >&2; exit 1; }
    EXTRA+=(++env.train_off_head=true)
    echo "Trainer off the driver's node: env.train_off_head, clone ${NEMO_RL_DIR}"
fi
if [[ "${HEAD:-0}" == 1 ]]; then
    export DEDICATED_RAY_HEAD=1
    SBATCH_MORE="${SBATCH_MORE:-} --nodes=$(( NODES + 1 ))"
    echo "Dedicated head node: DEDICATED_RAY_HEAD=1, $(( NODES + 1 )) nodes"
else
    unset DEDICATED_RAY_HEAD
fi
export SBATCH_EXTRA="--gres=gpu:4 --exclusive --comment=field=nlp;appl=pytorch --propagate=NONE --job-name=nrl-${RUN} ${SBATCH_MORE:-}"
cd "${INFRA}"
# This launcher names debug runs and sets steps, validation and checkpoints itself; DEBUG=0 keeps run.sh's debug
# defaults (3 steps, no validation, no checkpoints) out of it.
DEBUG=0 bash run.sh "${TASK}" \
    grpo.num_prompts_per_step=${PROMPTS} policy.train_global_batch_size=$(( PROMPTS * 16 )) \
    grpo.async_grpo.max_trajectory_age_steps=${STALENESS:-2} loss_fn.force_on_policy_ratio=true \
    "++policy.megatron_cfg.env_vars.NRL_CUDA_MEMORY_FRACTION=\"'${CAP:-0.74}'\"" \
    "++policy.megatron_cfg.env_vars.TORCHINDUCTOR_COMPILE_THREADS=\"'1'\"" \
    ++policy.generation.vllm_kwargs.max_num_seqs=256 \
    policy.max_total_sequence_length=${CTX} policy.generation.max_new_tokens=${CTX} \
    policy.sequence_packing.train_mb_tokens=${CTX} policy.sequence_packing.logprob_mb_tokens=${CTX} \
    policy.megatron_cfg.context_parallel_size=${CP} policy.make_sequence_length_divisible_by=${PAD} \
    "++policy.generation.vllm_cfg.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=\"'1'\"" \
    ${EXTRA[@]+"${EXTRA[@]}"} "$@"

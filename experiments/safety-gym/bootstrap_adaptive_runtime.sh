#!/usr/bin/env bash
# Assemble pinned native sources below a feature-only scratch root. Deliberately
# omits container/image setup from the site's setup.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
BASE="${ROOT}/backends/nemo-rl/apertus2"
FEATURE_ROOT="${SAFETY_ADAPTIVE_ROOT:-/scratch/${USER}/safetygym-adaptive}"
SRC="${FEATURE_ROOT}/src"
NEMO_RL_DIR="${SRC}/nemo-rl-apertus2"
MEGATRON_DIR="${SRC}/Megatron-LM-MoE"
BRIDGE_DIR="${SRC}/Megatron-Bridge"
APERTUS_COMMON_DIR="${SRC}/apertus-common"
TOOL_DIR="${FEATURE_ROOT}/tools"
mkdir -p "${SRC}" "${FEATURE_ROOT}/cache" "${FEATURE_ROOT}/private" "${TOOL_DIR}"
source "${BASE}/pin"
GIT=(git -c user.name=adaptive-runtime -c user.email=adaptive-runtime@localhost -c commit.gpgsign=false)

if [[ -e "${NEMO_RL_DIR}" ]]; then
  echo "Refusing to reuse existing feature source path: ${NEMO_RL_DIR}" >&2
  echo "Move it aside and inspect it before re-running bootstrap." >&2
  exit 2
fi
git clone --filter=blob:none "${NEMO_RL_URL}" "${NEMO_RL_DIR}"
git -C "${NEMO_RL_DIR}" switch -c safety-gym-adaptive "${NEMO_RL_COMMIT}"
while read -r repo pr sha; do
  git -C "${NEMO_RL_DIR}" fetch "https://github.com/${repo}.git" "pull/${pr}/head"
  if [[ "${repo}" == NVIDIA-NeMo/RL && "${pr}" == 2517 ]]; then
    "${GIT[@]}" -C "${NEMO_RL_DIR}" merge --no-commit --no-ff -Xtheirs "${sha:-FETCH_HEAD}" || true
    git -C "${NEMO_RL_DIR}" restore --source=HEAD -- tests/unit/test_recipes_and_test_suites.py
    git -C "${NEMO_RL_DIR}" add tests/unit/test_recipes_and_test_suites.py
    "${GIT[@]}" -C "${NEMO_RL_DIR}" commit -m "Merge ${repo} PR #${pr}"
  else
    "${GIT[@]}" -C "${NEMO_RL_DIR}" merge --no-edit -m "Merge ${repo} PR #${pr}" "${sha:-FETCH_HEAD}"
  fi
done <<< "${NEMO_RL_PRS}"
"${GIT[@]}" -C "${NEMO_RL_DIR}" am -3 "${ROOT}"/backends/nemo-rl/patches/*.patch "${BASE}"/patches/nemo-rl/*.patch
git -C "${NEMO_RL_DIR}" remote set-url --push origin no_push

AM="${NEMO_RL_DIR}/3rdparty/Automodel-workspace/Automodel"
git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Automodel.git "${AM}"
git -C "${AM}" checkout --detach "${AUTOMODEL_COMMIT}"
for p in "${BASE}"/patches/automodel/*.patch; do git -C "${AM}" apply "$p"; done

checkout_at() {
  local url="$1" dir="$2" commit="$3" ref="$4"
  git clone --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null || git -C "$dir" fetch origin "$ref"
  git -C "$dir" checkout --detach "$commit"
}
checkout_at "${MEGATRON_URL}" "${MEGATRON_DIR}" "${MEGATRON_COMMIT}" "${MEGATRON_REF}"
checkout_at "${BRIDGE_URL}" "${BRIDGE_DIR}" "${BRIDGE_COMMIT}" "${BRIDGE_REF}"
checkout_at "${APERTUS_COMMON_URL}" "${APERTUS_COMMON_DIR}" "${APERTUS_COMMON_COMMIT}" "${APERTUS_COMMON_REF}"
for p in "${BASE}"/patches/megatron/*.patch; do git -C "${MEGATRON_DIR}" apply "$p"; done

# Exact runner/input files from the pin, staged for read-only mounts. No image rebuild.
curl -fsSL -o "${TOOL_DIR}/gpu_model_runner.py" "${VLLM_RUNNER_URL}"
[[ "$(git hash-object "${TOOL_DIR}/gpu_model_runner.py")" == "${VLLM_RUNNER_BLOB}" ]]
curl -fsSL -o "${TOOL_DIR}/apertus2.py.orig" "${VLLM_APERTUS2_URL}"
[[ "$(git hash-object "${TOOL_DIR}/apertus2.py.orig")" == "${VLLM_APERTUS2_BLOB}" ]]
cp "${TOOL_DIR}/apertus2.py.orig" "${TOOL_DIR}/apertus2.py"
for p in "${BASE}"/patches/vllm/*.patch; do patch -s --fuzz=0 --no-backup-if-mismatch "${TOOL_DIR}/apertus2.py" < "$p"; done

python3 - "${FEATURE_ROOT}" "${NEMO_RL_DIR}" "${MEGATRON_DIR}" "${BRIDGE_DIR}" "${AM}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root=pathlib.Path(sys.argv[1]); paths={"nemo_rl":pathlib.Path(sys.argv[2]),"megatron":pathlib.Path(sys.argv[3]),"bridge":pathlib.Path(sys.argv[4]),"automodel":pathlib.Path(sys.argv[5])}
def git(path,*args): return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
files=["nemo_rl/algorithms/grpo.py","nemo_rl/algorithms/async_utils/trajectory_collector.py","nemo_rl/environments/nemo_gym.py","nemo_rl/models/policy/workers/megatron_policy_worker.py"]
manifest={"schema_version":"adaptive-native-runtime-v1","repositories":{name:{"path":str(path),"head":git(path,"rev-parse","HEAD")} for name,path in paths.items()},"native_files":{}}
for rel in files:
 path=paths["nemo_rl"]/rel; manifest["native_files"][rel]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"git_blob":git(paths["nemo_rl"],"hash-object",rel)}
manifest["manifest_sha256"]=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(root/"native-runtime-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
PY
echo "Feature native source assembled under ${FEATURE_ROOT}; no image or shared runtime was touched."

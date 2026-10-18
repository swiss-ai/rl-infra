#!/bin/bash
# Runs inside a writable copy of the NeMo-RL image (enroot start --root --rw), called by
# build_image.sh. This is the image's own UCCL build step (alps-extended-images 12288ce9,
# Alps-Images/apps/nemo-rl/Containerfile) with the patches in $1 applied after the
# checkout. The step's apt packages are already in the image, so that part is left out.
set -euxo pipefail

PATCH_DIR="$1"
UCCL_REPO=https://github.com/uccl-project/uccl.git
UCCL_REF=540cc775ba8ab231122f9a98a6d832699f239f70
NANOBIND_VERSION=3.0.1
UCCL_NVL_PEERS=4
UCCL_CUDA_ARCH=9.0
UCCL_MAX_JOBS=8
export UV_BREAK_SYSTEM_PACKAGES=1 UV_PYTHON_DOWNLOADS=never

site_dir="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
old_sum="$(md5sum "${site_dir}/uccl/ep.abi3.so" | cut -d' ' -f1)"

hard="$(ulimit -Hn)"
if [[ "${hard}" == unlimited ]]; then hard=65536; fi
ulimit -n "${hard}"

uv pip uninstall --system deep_ep uccl
uv pip install --system --no-cache -c /opt/alps/base-pins.txt \
  auditwheel build intervaltree "nanobind==${NANOBIND_VERSION}" \
  patchelf pybind11 wheel
rm -rf /io
git clone --recursive "${UCCL_REPO}" /io
git -C /io checkout -q "${UCCL_REF}"
git -C /io submodule update --init --recursive

shopt -s nullglob
patch_files=("${PATCH_DIR}"/*.patch)
shopt -u nullglob
if [[ ${#patch_files[@]} -eq 0 ]]; then
  echo "ERROR: no patches in ${PATCH_DIR}" >&2
  exit 1
fi
for patch_file in "${patch_files[@]}"; do
  git -C /io apply --check "${patch_file}"
  git -C /io apply --verbose "${patch_file}"
done
git -C /io diff --stat
if grep -n cudaMallocManaged /io/ep/src/uccl_ep.cc; then
  echo "ERROR: uccl_ep.cc still allocates managed memory" >&2
  exit 1
fi

mkdir -p /tmp/gpustub /opt/wheels/uccl
printf '#!/usr/bin/python3\nimport sys\na = " ".join(sys.argv[1:])\nif "query-gpu=name" in a:\n    print("NVIDIA GH200 120GB")\nelif "query-gpu=compute_cap" in a:\n    print("9.0")\nelse:\n    sys.exit(1)\n' > /tmp/gpustub/nvidia-smi
chmod +x /tmp/gpustub/nvidia-smi
libfabric_prefix="$(pkg-config --variable=prefix libfabric)"
libfabric_libdir="$(pkg-config --variable=libdir libfabric)"
test -n "${libfabric_prefix}"
test -f "${libfabric_libdir}/libfabric.so"
printf '%s\n' "${libfabric_libdir}" > /etc/ld.so.conf.d/98-alps-libfabric.conf
ldconfig

pushd /io
sed -i 's|-Wl,-rpath=/usr/lib/x86_64-linux-gnu||g' p2p/Makefile
export PATH="/tmp/gpustub:${PATH}" \
  CUDA_HOME=/usr/local/cuda \
  LIBFABRIC_HOME="${libfabric_prefix}" \
  LIBRARY_PATH="${libfabric_libdir}:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
  LDFLAGS="-L${libfabric_libdir} -Wl,-rpath,${libfabric_libdir} ${LDFLAGS:-}" \
  TORCH_CUDA_ARCH_LIST="${UCCL_CUDA_ARCH}" \
  NUM_MAX_NVL_PEERS="${UCCL_NVL_PEERS}" \
  MAX_JOBS="${UCCL_MAX_JOBS}" \
  USE_LIBFABRIC_CXI=1 PER_EXPERT_BATCHING=1 \
  TARGET=cu13 PY_VER=3.12 ARCH=aarch64 BUILD_TYPE=all IS_EFA="" \
  WHEEL_DIR=wheelhouse-cu13 \
  HOST_GLIBC_VER="$(python3 -c 'import platform; print(platform.libc_ver()[1])')" \
  UCCL_RETAG_TO_HOST_GLIBC=0 UCCL_LOCAL_VERSION=cu13
bash ./build_inner.sh
cp wheelhouse-cu13/uccl*.whl /opt/wheels/uccl/
test "$(find /opt/wheels/uccl -maxdepth 1 -name 'uccl-*.whl' | wc -l)" -eq 1
uv pip install --system --no-cache -c /opt/alps/base-pins.txt --no-deps /opt/wheels/uccl/uccl-*.whl
make -C p2p -j"${UCCL_MAX_JOBS}" install \
  PYTHON=python3 LIBFABRIC_HOME="${libfabric_prefix}"
printf '%s\n%s\n' /usr/local/lib "${site_dir}/uccl/lib" > /etc/ld.so.conf.d/98-uccl.conf
ldconfig
cd ep/deep_ep_wrapper
uv build --wheel --no-build-isolation --no-cache --python python3 --out-dir /opt/wheels/uccl .
test "$(find /opt/wheels/uccl -maxdepth 1 -name 'deep_ep-*.whl' | wc -l)" -eq 1
uv pip install --system --no-cache -c /opt/alps/base-pins.txt --no-deps /opt/wheels/uccl/deep_ep-*.whl
popd
rm -rf /tmp/gpustub /io /opt/wheels/uccl

new_sum="$(md5sum "${site_dir}/uccl/ep.abi3.so" | cut -d' ' -f1)"
if [[ "${new_sum}" == "${old_sum}" ]]; then
  echo "ERROR: ep.abi3.so unchanged" >&2
  exit 1
fi
uv pip show uccl deep_ep | grep -E '^(Name|Version|Location)'
echo "UCCL rebuilt: ep.abi3.so ${old_sum} -> ${new_sum}"

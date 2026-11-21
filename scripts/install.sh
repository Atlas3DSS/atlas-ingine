#!/usr/bin/env bash
# Install the Python environment, the private CUDA toolkit and ingine's prebuilt kernels, then the model.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  cat <<'HELP'
Usage: bash scripts/install.sh [--download-only | --update-from COMMIT]

Without options: check the system, install uv (into .tools/ when missing), the Python 3.12
environment in .venv/ (requirements-engine.txt), the CUDA 13.4 toolkit used to compile ingine's
kernels in .cuda-toolkit/, prebuild the kernels into engine/build/, then download and verify
the pinned model (Inkling-2-Mini weights and the ablation, about 20.2 GB) into models/Inkling-2-Mini.

--download-only    only what downloading needs (uv, .venv with huggingface_hub), then
                   download or resume the model; menu option 2.
--update-from REV  used by bash setup.sh --update: reinstall the software steps and download
                   the model only when the manifest's model revision differs from the one at
                   installer commit REV (or the files are incomplete).

INGINE_MODEL_DIR (environment or ~/.config/ingine/env) selects an existing download; a complete
one is reused as is. When it comes from the environment it is saved to ~/.config/ingine/env so
the ingine command serves from it.
CUDAHOSTCXX selects the host compiler nvcc uses; by default the first of $NVCC_CCBIN, c++, g++,
g++-16, g++-15, g++-14, g++-13 and g++-12 that the pinned nvcc accepts. A set CC must be one too:
PyTorch passes it to nvcc.
Requires Linux x86_64, an NVIDIA driver for CUDA 13 (580 or newer), git, curl and a C++20
compiler. The installer does not install or change GPU drivers.
HELP
  exit 0
fi

MODE=full
UPDATE_FROM=""
if [[ $# == 1 && "$1" == --download-only ]]; then
  MODE=download
elif [[ $# == 2 && "$1" == --update-from ]]; then
  MODE=update
  UPDATE_FROM="$2"
elif [[ $# != 0 ]]; then
  fail 'Unexpected arguments; use --help for usage.'
fi

# The pinned toolkit: nvcc 13.4.92, the version ingine's kernels are developed with. PyTorch's cu130
# build accepts any CUDA 13 nvcc. cuBLAS/cuSPARSE/cuSOLVER headers come from PyTorch's own CUDA
# wheels in .venv (CUDA_INC_PATH), matching the libraries PyTorch loads.
CUDA_VERSION=13.4.92
CUDA_PACKAGES=(
  nvidia-cuda-nvcc==13.4.92
  nvidia-cuda-crt==13.4.92
  nvidia-nvvm==13.4.92
  nvidia-cuda-runtime==13.4.92
  nvidia-cuda-cccl==13.3.4.3.1
  nvidia-nvtx==13.4.92
)
EXTENSIONS=(dense attn moe prefill)
MIN_DRIVER=580
MIN_GPU_MIB=31744
CONFIG="$HOME/.config/ingine/env"

confirm() {
  local answer
  printf 'Warning: %s\n' "$1" >&2
  [[ -t 0 ]] || fail 'Run the installer in a terminal to confirm this warning (bash setup.sh).'
  read -r -p 'Continue anyway? [y/N] ' answer || answer=""
  case "$answer" in
    y|Y|yes|YES) ;;
    *) fail 'Stopped at your request.' ;;
  esac
}

# Nearest existing folder of a path that may not exist yet.
existing_parent() {
  local path="$1"
  while [[ ! -e "$path" ]]; do
    path="$(dirname -- "$path")"
  done
  printf '%s\n' "$path"
}

# ---- checks before any change ----------------------------------------------------------------------

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'ingine requires Linux x86_64 with an NVIDIA RTX 5090.'
for command in git curl; do
  command -v "$command" >/dev/null || fail "Missing $command. Install it, or run bash setup.sh and choose 1."
done
[[ -r "$ROOT/runtime-manifest.json" ]] || fail "Missing $ROOT/runtime-manifest.json; this folder is not a complete installer clone."

if [[ "$MODE" != download ]]; then
  CXX_COMMAND="${CXX:-c++}"
  command -v "$CXX_COMMAND" >/dev/null || fail "Missing a C++20 compiler ($CXX_COMMAND). Run bash setup.sh and choose 1."
  command -v nvidia-smi >/dev/null || fail 'Install your NVIDIA driver first. This installer does not install or change drivers.'
  gpus="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader,nounits)" \
    || fail 'The NVIDIA driver is not working (nvidia-smi failed). Fix it before continuing.'
  printf 'Detected GPU(s):\n%s\n' "$gpus"
  driver="$(head -n 1 <<< "$gpus" | awk -F', ' '{print $5}')"
  driver_major="${driver%%.*}"
  [[ "$driver_major" =~ ^[0-9]+$ ]] || fail "Cannot read the NVIDIA driver version from nvidia-smi ($driver)."
  (( driver_major >= MIN_DRIVER )) || fail "NVIDIA driver $driver is too old: PyTorch for CUDA 13 needs driver $MIN_DRIVER or newer. Update the driver, then rerun."
  if ! awk -F', ' -v min="$MIN_GPU_MIB" '$3 == "12.0" && $4 + 0 >= min {found = 1} END {exit !found}' <<< "$gpus"; then
    confirm 'ingine targets a 32 GB RTX 5090 (compute capability 12.0, sm_120a); its kernels do not run on other GPUs, and the served 262,144-token context needs about 31 GiB of GPU memory.'
  fi
fi

# INGINE_MODEL_DIR: environment, then ~/.config/ingine/env, then models/<directory>; relative to this
# folder, as in scripts/ingine.
env_model_dir="${INGINE_MODEL_DIR:-}"
config_model_dir=""
if [[ -r "$CONFIG" ]]; then
  # shellcheck source=/dev/null
  config_model_dir="$(unset INGINE_MODEL_DIR; set +eu; source "$CONFIG" >/dev/null 2>&1; printf '%s' "${INGINE_MODEL_DIR:-}")" || config_model_dir=""
fi
model_setting="${env_model_dir:-$config_model_dir}"

# ---- uv and the Python environment ------------------------------------------------------------------

if [[ -x "$ROOT/.tools/bin/uv" ]]; then
  UV="$ROOT/.tools/bin/uv"
elif command -v uv >/dev/null; then
  UV="$(command -v uv)"
else
  echo "Installing uv from https://astral.sh/uv/install.sh into .tools/bin (no shell profile changes)."
  installer="$(mktemp)"
  if ! curl --fail --show-error --location --proto '=https' --tlsv1.2 https://astral.sh/uv/install.sh --output "$installer"; then
    rm -f -- "$installer"
    fail 'Could not download the uv installer. Check your connection and rerun.'
  fi
  status=0
  UV_UNMANAGED_INSTALL="$ROOT/.tools/bin" sh "$installer" || status=$?
  rm -f -- "$installer"
  [[ $status == 0 && -x "$ROOT/.tools/bin/uv" ]] || fail 'Installing uv failed; see the error above.'
  UV="$ROOT/.tools/bin/uv"
fi

[[ ! -L "$ROOT/.venv" ]] || fail "Refusing to modify a symlinked installation directory: $ROOT/.venv. Use a repository-local installation."
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  [[ ! -e "$ROOT/.venv" ]] || fail 'An incomplete .venv already exists. Move it aside before installing; no files were removed.'
  # uv's own Python builds include the headers the kernel build needs.
  managed=(--python-preference only-managed)
  if [[ "$("$UV" venv --help)" == *--managed-python* ]]; then
    managed=(--managed-python)
  fi
  "$UV" venv "${managed[@]}" --python 3.12 "$ROOT/.venv"
fi
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "ingine requires Python 3.12; move the existing .venv aside and rerun setup.")'
if [[ "$MODE" != download ]]; then
  "$PYTHON" -c 'import os, sys, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else "This Python has no development headers, which the kernel build needs; move .venv aside and rerun setup to get a uv-managed Python.")'
fi

mapfile -t pin < <("$PYTHON" - "$ROOT/runtime-manifest.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
model, serving = manifest["model"], manifest["serving"]
if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
    sys.exit("The model revision must be a full commit in runtime-manifest.json")
if not re.fullmatch(r"[A-Za-z0-9_.-]+", model["directory"]):
    sys.exit("Invalid model directory in runtime-manifest.json")
size_gb = sum(meta["size"] for meta in model["files"].values()) / 1e9
for value in (model["revision"], model["directory"], f"{size_gb:.1f}", serving["host"], serving["port"], serving["model_id"]):
    print(value)
PY
)
[[ ${#pin[@]} == 6 ]] || fail 'Cannot read runtime-manifest.json.'
REVISION="${pin[0]}"
MODEL_DIR="${model_setting:-models/${pin[1]}}"
[[ "$MODEL_DIR" == /* ]] || MODEL_DIR="$ROOT/$MODEL_DIR"

# Bytes of the pinned files that are missing or have the wrong size in MODEL_DIR.
model_missing_bytes() {
  "$PYTHON" - "$ROOT/runtime-manifest.json" "$MODEL_DIR" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    files = json.load(stream)["model"]["files"]
missing = 0
for name, meta in files.items():
    try:
        size = os.path.getsize(os.path.join(sys.argv[2], name))
    except OSError:
        size = 0
    if size != meta["size"]:
        missing += meta["size"] - size if size < meta["size"] else meta["size"]
print(missing)
PY
}

# The model step runs download_model.py, which resumes, verifies against the pinned revision and moves
# mismatching files aside (never deletes). A complete INGINE_MODEL_DIR, or an unchanged model during an
# update, is kept without rehashing.
missing_bytes="$(model_missing_bytes)"
download=1
if [[ "$MODE" != download && "$missing_bytes" == 0 ]]; then
  if [[ "$MODE" == update ]]; then
    previous_revision=""
    if [[ -n "$UPDATE_FROM" ]]; then
      previous_revision="$(git -C "$ROOT" show "$UPDATE_FROM:runtime-manifest.json" 2>/dev/null \
        | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["model"]["revision"])' 2>/dev/null)" || previous_revision=""
    fi
    [[ "$previous_revision" != "$REVISION" ]] || download=0
  elif [[ -n "$model_setting" ]]; then
    download=0
  fi
fi

# ---- free space --------------------------------------------------------------------------------------

declare -A need_kib=() where=()
add_need() {  # path, KiB
  local parent device
  parent="$(existing_parent "$1")"
  device="$(stat -c %d -- "$parent")"
  need_kib[$device]=$(( ${need_kib[$device]:-0} + $2 ))
  where[$device]="$parent"
}
if [[ "$MODE" != download ]]; then
  [[ -d "$ROOT/.venv/lib/python3.12/site-packages/torch" ]] || add_need "$ROOT" $(( 8 * 1048576 ))
  [[ -x "$ROOT/.cuda-toolkit/nvidia/cu13/bin/nvcc" ]] || add_need "$ROOT" 1048576
fi
if [[ "$download" == 1 && "$missing_bytes" != 0 ]]; then
  add_need "$MODEL_DIR" $(( missing_bytes / 1024 + 1048576 ))
fi
for device in "${!need_kib[@]}"; do
  free_kib="$(df -Pk -- "${where[$device]}" | awk 'NR == 2 {print $4}')"
  if (( free_kib < need_kib[$device] )); then
    confirm "$(printf '%s has %d GiB free; this step needs about %d GiB there.' \
      "${where[$device]}" $(( free_kib / 1048576 )) $(( (need_kib[$device] + 1048575) / 1048576 )))"
  fi
done

# ---- software ----------------------------------------------------------------------------------------

if [[ "$MODE" == download ]]; then
  if ! "$PYTHON" -c 'import huggingface_hub' 2>/dev/null; then
    "$UV" pip install --python "$PYTHON" 'huggingface_hub[hf_xet]'
  fi
else
  printf '\nInstalling the Python environment (PyTorch 2.14.0 for CUDA 13, about 5 GB on first install)...\n'
  "$UV" pip install --python "$PYTHON" -r "$ROOT/requirements-engine.txt"

  printf '\nInstalling the CUDA %s toolkit for the kernel build into .cuda-toolkit/...\n' "$CUDA_VERSION"
  [[ ! -L "$ROOT/.cuda-toolkit" ]] || fail 'Refusing to modify a symlinked .cuda-toolkit directory.'
  "$UV" pip install --python "$PYTHON" --target "$ROOT/.cuda-toolkit" --no-deps "${CUDA_PACKAGES[@]}"
  CUDA_HOME="$ROOT/.cuda-toolkit/nvidia/cu13"
  CUDA_INC_PATH="$("$PYTHON" -c 'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13", "include"))')"

  # NVIDIA's wheels omit the layout aliases PyTorch links with (-L$CUDA_HOME/lib64 -lcudart). Keep
  # existing correct aliases, and refuse to overwrite unrelated files or symlinks.
  ensure_link() {
    local target="$1" link="$2"
    if [[ -e "$link" || -L "$link" ]]; then
      [[ "$link" -ef "$(dirname -- "$link")/$target" ]] || fail "Unexpected CUDA layout at $link. Move it aside and rerun; it was not overwritten."
    else
      ln -s "$target" "$link"
    fi
  }
  ensure_link lib "$CUDA_HOME/lib64"
  ensure_link libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
  for file in bin/nvcc include/cuda_runtime.h include/cuda_fp8.h include/cooperative_groups.h include/crt/host_config.h \
    include/cccl/cuda/std/version nvvm/libdevice/libdevice.10.bc lib/libcudart.so.13; do
    [[ -f "$CUDA_HOME/$file" ]] || fail "Incomplete CUDA toolkit: $file is missing. Rerun the installation."
  done
  for file in cublas_v2.h cublasLt.h cusparse.h cusolverDn.h; do
    [[ -f "$CUDA_INC_PATH/$file" ]] || fail "PyTorch's CUDA headers are incomplete: $CUDA_INC_PATH/$file is missing. Rerun the installation."
  done
  [[ "$("$CUDA_HOME/bin/nvcc" --version)" == *"V$CUDA_VERSION"* ]] || fail "The private nvcc is not CUDA $CUDA_VERSION."

  # nvcc accepts only some host compilers (CUDA 13.4: GCC up to 16). Ask the pinned nvcc itself.
  # PyTorch passes a set CC to nvcc as -ccbin, so CC has to be one it accepts.
  probe="$(mktemp -d)"
  printf '#include <concepts>\n#include <span>\n__global__ void k() {}\nint main() { k<<<1, 1>>>(); return 0; }\n' > "$probe/probe.cu"
  if [[ -n "${CC:-}" ]]; then
    candidates=("$CC")
  elif [[ -n "${CUDAHOSTCXX:-}" ]]; then
    candidates=("$CUDAHOSTCXX")
  else
    candidates=(${NVCC_CCBIN:+"$NVCC_CCBIN"} c++ g++ g++-16 g++-15 g++-14 g++-13 g++-12)
  fi
  NVCC_HOST=""
  for candidate in "${candidates[@]}"; do
    path="$(command -v "$candidate")" || continue
    if "$CUDA_HOME/bin/nvcc" -ccbin "$path" -std=c++20 -gencode=arch=compute_120a,code=sm_120a \
      -c "$probe/probe.cu" -o "$probe/probe.o" >/dev/null 2>&1; then
      NVCC_HOST="$path"
      break
    fi
  done
  rm -rf -- "$probe"
  if [[ -z "$NVCC_HOST" ]]; then
    [[ -z "${CC:-}" ]] || fail "CUDA $CUDA_VERSION cannot compile C++20 with CC=$CC, which PyTorch passes to nvcc. Unset CC or set it to a GCC up to 16, then rerun."
    [[ -z "${CUDAHOSTCXX:-}" ]] || fail "CUDA $CUDA_VERSION cannot compile C++20 with CUDAHOSTCXX=$CUDAHOSTCXX."
    fail "No C++20 host compiler that CUDA $CUDA_VERSION accepts (GCC up to 16). Install one (for example g++-14 on Debian/Ubuntu) or set CUDAHOSTCXX, then rerun."
  fi
  printf 'CUDA host compiler: %s\n' "$NVCC_HOST"

  # scripts/ingine sources this file: the same values reach PyTorch's build flags at serving time, so the
  # prebuilt kernels are reused instead of rebuilt.
  env_file="$ROOT/.cuda-toolkit/ingine.env"
  {
    echo '# Written by scripts/install.sh and sourced by scripts/ingine: environment of the JIT kernel build.'
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export CUDA_INC_PATH=%q\n' "$CUDA_INC_PATH"
    printf 'export NVCC_CCBIN=%q\n' "$NVCC_HOST"
    echo 'export TORCH_CUDA_ARCH_LIST=12.0a'
  } > "$env_file.tmp"
  mv -f -- "$env_file.tmp" "$env_file"

  # PyTorch serializes builds with engine/build/<name>/lock and waits forever on one left by a killed
  # build. Remove only locks that no running process holds open.
  for lock in "$ROOT"/engine/build/*/lock; do
    [[ -e "$lock" ]] || continue
    if [[ -z "$(find /proc/[0-9]*/fd -maxdepth 1 -lname "$lock" -print -quit 2>/dev/null)" ]]; then
      rm -f -- "$lock"
      printf 'Removed a stale build lock from an interrupted build: %s\n' "$lock"
    fi
  done

  printf '\nBuilding the CUDA kernels (%s); the first build takes a few minutes...\n' "${EXTENSIONS[*]}"
  (
    # shellcheck source=/dev/null
    source "$env_file"
    export PATH="$ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
    cd "$ROOT/engine"
    # SIGTERM (setup.sh's interrupt) unwinds like Ctrl-C, so PyTorch releases its build lock.
    "$PYTHON" - "${EXTENSIONS[@]}" <<'PY'
import signal
import sys


def stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)
from ingine import ext

for name in sys.argv[1:]:
    print(f"  {name}", flush=True)
    ext.load(name)
PY
  ) || fail 'Building the CUDA kernels failed; see the error above. Rerun to retry; finished kernels are kept.'
  echo 'Kernels are built.'
fi

# ---- model -------------------------------------------------------------------------------------------

if [[ "$download" == 1 ]]; then
  printf '\nDownloading and verifying the model at revision %s (about %s GB) into %s...\n' "${REVISION:0:12}" "${pin[2]}" "$MODEL_DIR"
  "$PYTHON" "$ROOT/scripts/download_model.py" --model-dir "$MODEL_DIR" \
    || fail 'The model download did not finish. Run it again to resume (bash setup.sh, option 2).'
elif [[ "$MODE" == update ]]; then
  printf '\nThe model is unchanged (revision %s) and complete in %s.\n' "${REVISION:0:12}" "$MODEL_DIR"
else
  printf '\nUsing the complete existing download in %s (file sizes match revision %s).\n' "$MODEL_DIR" "${REVISION:0:12}"
  printf 'Check its hashes with: %q %q --model-dir %q --verify-only\n' "$PYTHON" "$ROOT/scripts/download_model.py" "$MODEL_DIR"
fi

# A model folder given only in the environment is remembered for the ingine command.
if [[ -n "$env_model_dir" && "$env_model_dir" != "$config_model_dir" ]]; then
  if [[ -n "$config_model_dir" ]]; then
    printf 'Note: %s sets INGINE_MODEL_DIR=%s; the ingine command serves from that folder unless INGINE_MODEL_DIR is set.\n' "$CONFIG" "$config_model_dir"
  elif [[ ! -e "$CONFIG" ]] || ! grep -q 'INGINE_MODEL_DIR' "$CONFIG"; then
    mkdir -p -- "$(dirname -- "$CONFIG")"
    printf 'INGINE_MODEL_DIR=%q\n' "$MODEL_DIR" >> "$CONFIG"
    printf 'Saved INGINE_MODEL_DIR=%s to %s for the ingine command.\n' "$MODEL_DIR" "$CONFIG"
  fi
fi

if [[ "$MODE" == download ]]; then
  printf '\nThe model is ready. Choose 1 in bash setup.sh to install the engine if you have not yet.\n'
else
  printf '\nReady. Start the server with bash setup.sh, option 3 (foreground), or with ingine start\n'
  printf '(background; option 5 installs the ingine command). API: http://%s:%s/v1, model %s\n' "${pin[3]}" "${pin[4]}" "${pin[5]}"
fi

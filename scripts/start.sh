#!/usr/bin/env bash
# One-command build-and-serve launcher for atlas-ingine (DGX Spark GB10 sm_121a & discrete GPUs).
# Note: sm_121a (DGX Spark GB10 aarch64) preset and recipe are compile-verified against CUDA toolkits;
# physical GB10 runtime performance is unmeasured until hardware access is available.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
NATIVE="$ROOT/native"

PRESET="${INGINE_PRESET:-auto}"
MODEL_DIR="${INGINE_MODEL_DIR:-$ROOT/models/Inkling-2-Mini}"
DRAFTER="${INGINE_DRAFTER:-dspark}"
DFLASH2_DIR="${INGINE_DFLASH2_DIR:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8003}"
MAX_CTX="${INGINE_CTX:-}"
SPEC_THRESHOLD="${INGINE_SPEC_THRESHOLD:-0.8}"
ABLATION="${INGINE_ABLATION:-}"
DENSE_FP8=1
KV_FP8=1
FORCE_BUILD=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'HELP'
Usage: bash scripts/start.sh [options] [-- <extra ingine-serve args>]

Builds ingine-serve for the detected (or selected) CMake preset if missing, then launches
ingine-serve exposing both OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) APIs.

Options:
  --preset <name>          CMake preset: auto (default), sm121, release, sm70, install, cpu-dev
                           auto selects sm121 on aarch64 or compute capability 12.1 (DGX Spark GB10),
                           sm70 on 7.0 (Tesla V100), release on 12.0 (RTX 5090 / PRO 6000), else install.
  --model-dir <path>       Checkpoint directory (default: $INGINE_MODEL_DIR or ./models/Inkling-2-Mini)
  --drafter <mode>         Speculative drafter: dspark (default), dflash2, none
  --dflash2-dir <path>     Path to incoai/Inkling-2-Mini-DFlash2 (rev 4f9088ea); implies --drafter dflash2
  --ablation <path>        Optional ablation safetensors file
  --host <addr>            Bind address (default: 127.0.0.1)
  --port <port>            Bind port (default: 8003)
  --max-ctx <tokens>       Context length (default: 131072 for sm121, 65536 otherwise)
  --spec-threshold <f>     Speculative confidence threshold (default: 0.8)
  --no-dense-fp8           Disable --dense-fp8
  --no-kv-fp8              Disable --kv-fp8
  --build                  Force reconfigure/rebuild before serving
  --dry-run                Print resolved preset, build command, and serve command without executing
  -h, --help               Show this help

Claude Code / Omarchy client example (Anthropic /v1/messages):
  ANTHROPIC_BASE_URL=http://127.0.0.1:8003 ANTHROPIC_API_KEY=local claude

Note: The sm121 (DGX Spark GB10 aarch64) preset has not been measured on physical GB10 hardware.
HELP
}

fail() {
  printf 'start.sh: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --drafter) DRAFTER="${2:-}"; shift 2 ;;
    --dflash2-dir) DFLASH2_DIR="${2:-}"; DRAFTER="dflash2"; shift 2 ;;
    --ablation) ABLATION="${2:-}"; shift 2 ;;
    --host) HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --max-ctx) MAX_CTX="${2:-}"; shift 2 ;;
    --spec-threshold) SPEC_THRESHOLD="${2:-}"; shift 2 ;;
    --no-dense-fp8) DENSE_FP8=0; shift ;;
    --no-kv-fp8) KV_FP8=0; shift ;;
    --build) FORCE_BUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$DRAFTER" in
  dspark|dflash2|none) ;;
  *) fail "unsupported --drafter '$DRAFTER' (expected dspark, dflash2, or none)" ;;
esac

detect_preset() {
  local arch cc=""
  arch="$(uname -m)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
  fi
  if [[ "$cc" == "12.1" || ( "$arch" == "aarch64" && -z "$cc" ) ]]; then
    printf 'sm121\n'
  elif [[ "$cc" == "12.0" ]]; then
    printf 'release\n'
  elif [[ "$cc" == "7.0" ]]; then
    printf 'sm70\n'
  elif [[ "$cc" == "6.0" || "$cc" == "6.1" || "$cc" == "7.5" ]]; then
    printf 'legacy-fp16\n'
  elif [[ -n "$cc" ]]; then
    printf 'install\n'
  else
    printf 'cpu-dev\n'
  fi
}

if [[ "$PRESET" == "auto" ]]; then
  PRESET="$(detect_preset)"
fi

case "$PRESET" in
  sm121|sm121-dev) BUILD_DIR="$NATIVE/build-sm121" ;;
  release|dev) BUILD_DIR="$NATIVE/build" ;;
  sm70|sm70-dev) BUILD_DIR="$NATIVE/build-sm70" ;;
  legacy-fp16) BUILD_DIR="$NATIVE/build-legacy-fp16" ;;
  install) BUILD_DIR="$NATIVE/build-install" ;;
  multiarch-dev) BUILD_DIR="$NATIVE/build-multiarch" ;;
  cpu-dev) BUILD_DIR="$NATIVE/build-cpu" ;;
  *) fail "unknown preset '$PRESET'" ;;
esac

if [[ -z "$MAX_CTX" ]]; then
  if [[ "$PRESET" == sm121* ]]; then
    MAX_CTX=131072
  else
    MAX_CTX=65536
  fi
fi

SERVER="$BUILD_DIR/apps/serve/ingine-serve"

SERVE_CMD=(
  "$SERVER"
  --model-dir "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --max-ctx "$MAX_CTX"
)

if [[ "$PRESET" == "cpu-dev" ]]; then
  SERVE_CMD+=(--fake)
else
  SERVE_CMD+=(--engine)
  if (( DENSE_FP8 )); then SERVE_CMD+=(--dense-fp8); fi
  if (( KV_FP8 )); then SERVE_CMD+=(--kv-fp8); fi
  case "$DRAFTER" in
    dspark)
      SERVE_CMD+=(--spec --spec-threshold "$SPEC_THRESHOLD")
      ;;
    dflash2)
      [[ -n "$DFLASH2_DIR" ]] || fail "--drafter dflash2 requires --dflash2-dir <path> (or INGINE_DFLASH2_DIR)"
      SERVE_CMD+=(--spec --dflash2-dir "$DFLASH2_DIR" --spec-threshold "$SPEC_THRESHOLD")
      ;;
    none)
      ;;
  esac
fi

if [[ -n "$ABLATION" ]]; then
  SERVE_CMD+=(--ablation "$ABLATION")
fi
if (( ${#EXTRA_ARGS[@]} > 0 )); then
  SERVE_CMD+=("${EXTRA_ARGS[@]}")
fi

if (( DRY_RUN )); then
  printf 'preset=%s\n' "$PRESET"
  printf 'build_dir=%s\n' "$BUILD_DIR"
  printf 'build_cmd=cmake -S %q --preset %s && cmake --build %q --target ingine-serve\n' "$NATIVE" "$PRESET" "$BUILD_DIR"
  printf 'serve_cmd='
  printf '%q ' "${SERVE_CMD[@]}"
  printf '\n'
  exit 0
fi

if (( FORCE_BUILD )) || [[ ! -x "$SERVER" ]]; then
  printf 'start.sh: building ingine-serve with preset %s (%s)\n' "$PRESET" "$BUILD_DIR" >&2
  cmake -S "$NATIVE" --preset "$PRESET"
  cmake --build "$BUILD_DIR" --target ingine-serve
fi

printf 'start.sh: serving OpenAI (/v1/chat/completions) + Anthropic (/v1/messages) at http://%s:%s\n' "$HOST" "$PORT" >&2
exec "${SERVE_CMD[@]}"

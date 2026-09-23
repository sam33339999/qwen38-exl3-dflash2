#!/usr/bin/env bash
# One-time host setup for scripts/serve-host.sh. No Docker.
#
# Creates:
#   exllamav3/   community pin 355c6ee (native DFlash2, sm_86)
#   env/         Python 3.13 venv
#   models/qwen38-27b-exl3
#   models/dflash2-exl3
#
# A second PyTorch CUDA wheel is about 6 GB. This disk cannot hold that next
# to the 16.5 GB target, so the venv reuses an existing torch 2.x+cu130 via
# TORCH_SITE (default: the openwebui venv on this machine). The extension is
# compiled against that torch. Set TORCH_SITE empty and free ~8 GB if you
# want a private torch==2.10.0+cu130 instead — that is the version the
# published image was measured with.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENGINE_URL="${ENGINE_URL:-https://github.com/r0b0tlab/exllamav3.git}"
ENGINE_COMMIT="${ENGINE_COMMIT:-355c6ee10fbd25b79070316a81ea0708cc18155a}"
TARGET_REPO="${TARGET_REPO:-r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw}"
DRAFT_REPO="${DRAFT_REPO:-r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw}"
TARGET_DIR="${TARGET_DIR:-$ROOT/models/qwen38-27b-exl3}"
DRAFT_DIR="${DRAFT_DIR:-$ROOT/models/dflash2-exl3}"
TORCH_SITE="${TORCH_SITE:-/home/sam/openwebui/.venv/lib/python3.13/site-packages}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"

if [[ ! -d "$ROOT/exllamav3/.git" ]]; then
  git clone --depth 1 --branch community "$ENGINE_URL" "$ROOT/exllamav3"
fi
if [[ "$(git -C "$ROOT/exllamav3" rev-parse HEAD)" != "$ENGINE_COMMIT" ]]; then
  echo "exllamav3 HEAD is $(git -C "$ROOT/exllamav3" rev-parse HEAD), want $ENGINE_COMMIT" >&2
  exit 1
fi

if [[ ! -x "$ROOT/env/bin/python" ]]; then
  uv venv --python 3.13 "$ROOT/env"
fi
PY="$ROOT/env/bin/python"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

if [[ -n "$TORCH_SITE" ]]; then
  if [[ ! -f "$TORCH_SITE/torch/__init__.py" ]]; then
    echo "TORCH_SITE has no torch: $TORCH_SITE" >&2
    exit 1
  fi
  echo "$TORCH_SITE" > "$SITE/reuse-torch.pth"
fi
"$PY" - <<'PY'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
PY

uv pip install --python "$PY" setuptools wheel ninja marisa-trie llguidance

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
# nvcc -O3 is memory-heavy; cap jobs so a 32 GB host does not get OOM-killed.
export MAX_JOBS="${MAX_JOBS:-4}"
export TMPDIR="${TMPDIR:-/tmp}"

cd "$ROOT/exllamav3"
# Object files go to tmpfs. The .so is written in-place next to the package,
# which is what `import exllamav3_ext` finds via PYTHONPATH.
"$PY" setup.py build_ext --inplace --build-temp "${BUILD_TEMP:-/tmp/exl3-build}"

# Xet stalls on this network; plain HTTPS resumes cleanly into --local-dir.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ ! -f "$DRAFT_DIR/config.json" || ! -f "$DRAFT_DIR/model.safetensors" ]]; then
  mkdir -p "$DRAFT_DIR"
  hf download "$DRAFT_REPO" --local-dir "$DRAFT_DIR"
fi
if [[ ! -f "$TARGET_DIR/config.json" || ! -f "$TARGET_DIR/model.safetensors.index.json" ]]; then
  mkdir -p "$TARGET_DIR"
fi
# Shards are the bulk. Download whenever either is missing.
if ! compgen -G "$TARGET_DIR/model-*.safetensors" > /dev/null; then
  mkdir -p "$TARGET_DIR"
  hf download "$TARGET_REPO" --local-dir "$TARGET_DIR"
fi

echo "ready: bash scripts/serve-host.sh"

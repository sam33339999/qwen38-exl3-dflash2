#!/usr/bin/env bash
# Serve Qwen3.8-27B EXL3 + DFlash2 on the host RTX 3090. No Docker.
#
# One-time setup (from the repo root):
#   bash scripts/bootstrap-host.sh
# Then:
#   bash scripts/serve-host.sh
#
# Listens on 0.0.0.0:8889 by default.
#   GET  /health
#   GET  /v1/models
#   POST /v1/chat/completions
#
# Overrides: HOST, PORT, TARGET, DRAFT, CTX, CQ, MAX_TOKENS,
#            DRY_MULTIPLIER, DRY_RANGE, REP_PENALTY, FREQ_PENALTY, THINK_BUDGET,
#            LOOP_WINDOW, LOOP_REPS, PYTHON
#   bash scripts/serve-host.sh --help
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/serve-host.sh
  bash scripts/serve-host.sh --help

在本機 RTX 3090 上啟動 Qwen3.8-27B EXL3 + DFlash2，不使用 Docker。
預設聽 http://0.0.0.0:8889
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions          （stream=true 為 SSE）
  POST /v1/completions

環境變數（寫在指令前面）。客戶端請求裡若帶了同名欄位，以客戶端為準。

  HOST=0.0.0.0              綁定位址
  PORT=8889                 埠
  CTX=262144                上下文長度
  CQ=3                      KV cache 量化 bit
  MAX_TOKENS=32768          客戶端沒帶 max_tokens 時，聊天最多新生成的 token。
                            思考和回答都算在裡面。prompt + 這個值要放得進 CTX。
  DRY_MULTIPLIER=0.8        DRY 重複片段懲罰。0 關掉。
                            要接出剛才出現過的片段時，那些 token 的機率會被壓低，
                            生成繼續，不會因為打轉把這次請求結束。
  DRY_BASE=1.75             重複越長，懲罰上升越快
  DRY_ALLOWED_LENGTH=2      短於等於這個長度的重複不罰
  DRY_RANGE=4096            往回看幾個 token。0 表示整段上下文
  REP_PENALTY=1.0           單字重複懲罰。1.0 是關掉，大於 1 才罰
  FREQ_PENALTY=0.3          最近 FREQ_RANGE 個 token 裡，出現越多次的字再被選中就越不利。
                            用來壓「換個詞再列一條」這種句型。0 關掉。
  FREQ_RANGE=512
  THINK_BUDGET=4096         思考超過這麼多個新 token 後，</think> 的分數開始上升，
                            模型會轉去寫答案。請求不會被掐掉。0 關掉。
  THINK_RAMP=1536           從開始加分到加滿要再多少 token
  THINK_BIAS=16             </think> 最多加多少 logit
  LOOP_WINDOW=0             大於 0 時，token 完全重複才會硬停生成。
                            預設 0，打轉只靠懲罰和 think budget，不中斷請求
  LOOP_REPS=3               搭配 LOOP_WINDOW 的重複次數
  PYTHON=...                指定 python

單次請求也可以蓋過預設，例如：
  "dry_multiplier": 0
  "repetition_penalty": 1.1
  "frequency_penalty": 0.2
  "presence_penalty": 0.2
  "max_tokens": 8192

前端按停止會關掉這條 HTTP 連線，服務會在下一個解碼步驟取消該次生成。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/env/bin/python}"
TARGET="${TARGET:-$ROOT/models/qwen38-27b-exl3}"
DRAFT="${DRAFT:-$ROOT/models/dflash2-exl3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8889}"
CTX="${CTX:-262144}"
CQ="${CQ:-3}"
# Chat completions only. Clients that send max_tokens still win.
# 32768 leaves room for a long think plus the answer; prompt + this must fit in CTX.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# DRY down-weights tokens that would extend a repeated phrase. It does not end the request.
# rep penalty 1.0 is off. LOOP_WINDOW 0 means a detected loop does not abort generation.
DRY_MULTIPLIER="${DRY_MULTIPLIER:-0.8}"
DRY_BASE="${DRY_BASE:-1.75}"
DRY_ALLOWED_LENGTH="${DRY_ALLOWED_LENGTH:-2}"
DRY_RANGE="${DRY_RANGE:-4096}"
REP_PENALTY="${REP_PENALTY:-1.0}"
FREQ_PENALTY="${FREQ_PENALTY:-0.3}"
FREQ_RANGE="${FREQ_RANGE:-512}"
THINK_BUDGET="${THINK_BUDGET:-4096}"
THINK_RAMP="${THINK_RAMP:-1536}"
THINK_BIAS="${THINK_BIAS:-16}"
LOOP_WINDOW="${LOOP_WINDOW:-0}"
LOOP_REPS="${LOOP_REPS:-3}"
# cache_tokens is the KV allocation; the server default (270336) covers 262144
# plus the DFlash2 verify window. Override only if you also lower CTX.
CACHE_TOKENS="${CACHE_TOKENS:-270336}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing $PYTHON — run: bash scripts/bootstrap-host.sh" >&2
  exit 1
fi
if [[ ! -f "$TARGET/config.json" || ! -f "$DRAFT/config.json" ]]; then
  echo "model dirs are incomplete:" >&2
  echo "  target $TARGET" >&2
  echo "  draft  $DRAFT" >&2
  echo "run: bash scripts/bootstrap-host.sh" >&2
  exit 1
fi

# 3090 defaults. Measured, do not override unless A/B testing.
export EXL3_HGEMM_F16ACC="${EXL3_HGEMM_F16ACC:-auto}"
export EXL3_INT8_GEMV="${EXL3_INT8_GEMV:-2}"
export EXL3_INT8_GEMV_MAX_K="${EXL3_INT8_GEMV_MAX_K:-5}"
export EXL3_QC_STAGING="${EXL3_QC_STAGING:-1}"

# Torch's bundled CUDA 13 libs must come before /usr/local/cuda (nvcc 12.4 is
# also on PATH; the extension was built against CUDA 13).
NVIDIA_LIB="$("$PYTHON" - <<'PY'
import os, pathlib, torch
base = pathlib.Path(torch.__file__).resolve().parent.parent / "nvidia"
libs = sorted(p for p in base.glob("*/lib") if p.is_dir())
print(":".join(str(p) for p in libs))
PY
)"
export LD_LIBRARY_PATH="${NVIDIA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"
# serve_openai.py inserts ../exllamav3 on sys.path. Keep the source tree first
# so the in-place sm_86 extension is the one that loads.
export PYTHONPATH="$ROOT/exllamav3${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" "$ROOT/scripts/serve_openai.py" \
  --target "$TARGET" \
  --draft "$DRAFT" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$CTX" \
  --cache-tokens "$CACHE_TOKENS" \
  --cq "$CQ" \
  --default-max-tokens "$MAX_TOKENS" \
  --loop-window "$LOOP_WINDOW" \
  --loop-min-reps "$LOOP_REPS" \
  --rep-penalty "$REP_PENALTY" \
  --freq-penalty "$FREQ_PENALTY" \
  --freq-range "$FREQ_RANGE" \
  --think-budget "$THINK_BUDGET" \
  --think-ramp "$THINK_RAMP" \
  --think-bias "$THINK_BIAS" \
  --dry-multiplier "$DRY_MULTIPLIER" \
  --dry-base "$DRY_BASE" \
  --dry-allowed-length "$DRY_ALLOWED_LENGTH" \
  --dry-range "$DRY_RANGE"

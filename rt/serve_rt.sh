#!/bin/bash
# Blackfrost RTI — reference launch script (sglang, TP=4).
#
# Serves STOCK MiniMax-M3.1-Flash with the runtime intervention hook installed
# via PYTHONPATH. Nothing is written to the checkpoint.
#
# Usage:
#   MODEL_PATH=/path/to/MiniMax-M3.1-Flash ./serve_rt.sh
#
# Override any of: MODEL_PATH, TP, PORT, CTX, GPUS, SERVED_NAME.
set -uo pipefail

RTI_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/models/MiniMax-M3.1-Flash}"
TP="${TP:-4}"
PORT="${PORT:-30000}"
CTX="${CTX:-65536}"
GPUS="${GPUS:-0,1,2,3}"
SERVED_NAME="${SERVED_NAME:-MiniMax-3.1-Flash-RTI}"

# --- runtime intervention wiring ---
export PYTHONPATH="$RTI_HOME/rt:${PYTHONPATH:-}"
export BLACKFROST_RT_INTERVENTION="$RTI_HOME/rt_config.json"

# --- environment required by this image/checkpoint (see docs/notes) ---
export FMHA_SM100_ALLOW_JIT=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MSA_V1_AOT_CACHE="${MSA_V1_AOT_CACHE:-/workspace/msa_cache}"
# MegaMoE prefill budget must cover chunked_prefill_size (16384) or startup
# aborts in validate_mega_moe_token_budget.
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-32768}"

echo "=== BLACKFROST RTI SERVE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "rti_home      = $RTI_HOME"
echo "model         = $MODEL_PATH   (STOCK — never modified)"
echo "served_name   = $SERVED_NAME"
echo "rt_config     = $BLACKFROST_RT_INTERVENTION"
echo "toggle        = $(cat "$RTI_HOME/rt/toggle.json" 2>/dev/null || echo '(none)')"
echo "tp=$TP  ctx=$CTX  port=$PORT  gpus=${CUDA_VISIBLE_DEVICES:-$GPUS}"
echo

exec sglang serve \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --reasoning-parser minimax-m3 \
  --tool-call-parser minimax-m3 \
  --tp "$TP" \
  --mem-fraction-static 0.8 \
  --max-running-requests 8 \
  --context-length "$CTX" \
  --host 0.0.0.0 \
  --port "$PORT"
#!/bin/bash
# Blackfrost RTI — container launch wrapper for b300-test.
#
# sglang runs INSIDE the `m31-rt` Docker container (image
# lmsysorg/sglang:dev-minimax-m31-flash), which owns GPUs 0-3 and shares the
# host network. The plain rt/serve_rt.sh cannot work on the host because
# `sglang` is not on the host PATH — it must be exec'd into the container.
#
# Usage:
#   ./serve_container.sh              # 1M context (default)
#   CTX=65536 ./serve_container.sh    # short-context smoke
#
# Override any of: CTX, TP, PORT, MAX_RUNNING, MEM_FRAC, CONTAINER, LOG.
set -uo pipefail

RTI_HOME="/workspace/rti-pkg"
CONTAINER="${CONTAINER:-m31-rt}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/MiniMax-M3.1-Flash}"
TP="${TP:-4}"
PORT="${PORT:-30000}"
CTX="${CTX:-1048576}"
MAX_RUNNING="${MAX_RUNNING:-8}"
MEM_FRAC="${MEM_FRAC:-0.8}"
LOG="${LOG:-$RTI_HOME/serve.log}"

if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "FATAL: container '$CONTAINER' not found" >&2
  exit 1
fi

echo "=== BLACKFROST RTI SERVE (container) $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "container     = $CONTAINER"
echo "model         = $MODEL_PATH   (STOCK — never modified)"
echo "served_name   = MiniMax-3.1-Flash-RTI"
echo "toggle        = $(cat "$RTI_HOME/rt/toggle.json" 2>/dev/null || echo '(none)')"
echo "tp=$TP  ctx=$CTX  port=$PORT  max_running=$MAX_RUNNING  mem_frac=$MEM_FRAC"
echo

# --- runtime intervention wiring (paths are container-visible via /workspace) ---
exec docker exec \
  -e PYTHONPATH="$RTI_HOME/rt" \
  -e BLACKFROST_RT_INTERVENTION="$RTI_HOME/rt_config.json" \
  -e FMHA_SM100_ALLOW_JIT=1 \
  -e CUDA_HOME=/usr/local/cuda-13.4 \
  -e HF_HOME=/workspace/hf \
  -e HF_HUB_OFFLINE=1 \
  -e MSA_V1_AOT_CACHE=/workspace/msa_cache \
  -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768 \
  "$CONTAINER" \
  /opt/sglang/bin/python3 /opt/sglang/bin/sglang serve \
    --model-path "$MODEL_PATH" \
    --served-model-name MiniMax-3.1-Flash-RTI \
    --trust-remote-code \
    --reasoning-parser minimax-m3 \
    --tool-call-parser minimax-m3 \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRAC" \
    --max-running-requests "$MAX_RUNNING" \
    --context-length "$CTX" \
    --host 0.0.0.0 \
    --port "$PORT"
#!/usr/bin/env bash
set -euo pipefail

cd /opt/train

BASE_DIR="/opt/train/data/nanogpt/MLA"
LOG_DIR="${BASE_DIR}/log"
TRAIN_FILE="${BASE_DIR}/train_gpt2_MLA.py"

echo "[nightly] $(date)"
echo "[nightly] BASE_DIR=${BASE_DIR}"
echo "[nightly] LOG_DIR=${LOG_DIR}"

# ------------------------------------------------
# 1. 自动探测 GPU 数量
# ------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l)"
else
  GPU_COUNT=0
fi

if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "[nightly][ERROR] No GPU detected!"
  exit 1
fi

echo "[nightly] Detected GPU_COUNT=${GPU_COUNT}"

# ------------------------------------------------
# 2. 找最新 checkpoint
# ------------------------------------------------
latest_ckpt="$(ls -1t "${LOG_DIR}"/*.pt 2>/dev/null | head -n 1 || true)"

echo "[nightly] latest_ckpt=${latest_ckpt:-<none>}"

# ------------------------------------------------
# 3. 启动 torchrun（自动多卡）
# ------------------------------------------------
TORCHRUN_ARGS=(
  --standalone
  --nproc_per_node="${GPU_COUNT}"
)

TRAIN_ARGS=()

if [[ -n "${latest_ckpt}" ]]; then
  TRAIN_ARGS+=(--resume "${latest_ckpt}")
fi

echo "[nightly] torchrun ${TORCHRUN_ARGS[*]} ${TRAIN_FILE} ${TRAIN_ARGS[*]}"

exec torchrun \
  "${TORCHRUN_ARGS[@]}" \
  "${TRAIN_FILE}" \
  "${TRAIN_ARGS[@]}"
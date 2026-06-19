#!/bin/bash
# Run title LoRA inference and tee output to logs/
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
JSONL="${JSONL:-${REPO}/data/llm_sft_sample/title/test.jsonl}"
ROW="${ROW:-0}"
ADAPTER="${ADAPTER:-/root/lora/tibetan-title-pilot}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"

mkdir -p "${LOG_DIR}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/title_lora_inference_${STAMP}.log"
DETAIL_LOG="${LOG_DIR}/title_lora_inference_${STAMP}_detail.log"

cd "${REPO}"
echo "log=${LOG}" | tee "${LOG}"
echo "detail_log=${DETAIL_LOG}" | tee -a "${LOG}"
export DETAIL_LOG
python3 -u scripts/run_title_lora_inference.py "${JSONL}" "${ROW}" "${ADAPTER}" "${DETAIL_LOG}" 2>&1 | tee -a "${LOG}"
echo "SAVED_LOG=${LOG}" | tee -a "${LOG}"
echo "SAVED_DETAIL_LOG=${DETAIL_LOG}" | tee -a "${LOG}"
echo "SAVED_DETAIL_JSON=${DETAIL_LOG%.log}.json" | tee -a "${LOG}"

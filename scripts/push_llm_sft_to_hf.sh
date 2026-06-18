#!/bin/bash
# Upload LLM SFT JSONL dataset to Hugging Face (resumable large-folder upload).
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET_ID="${DATASET_ID:-ganga4364/tibetan-metadata-llm-sft}"
SOURCE_DIR="${SOURCE_DIR:-data/llm_sft}"
LOG="${LOG:-data/llm_sft/logs/hf_upload.log}"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Missing /root/.hf_token — run: bash scripts/setup_hf_token.sh"
  exit 1
fi

if [[ ! -f "${SOURCE_DIR}/title/train.jsonl" ]]; then
  echo "Missing ${SOURCE_DIR}/title/train.jsonl"
  exit 1
fi

mkdir -p "$(dirname "${LOG}")"
hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true

echo "=== HF upload started at $(date) ===" | tee "${LOG}"
echo "Repo: ${DATASET_ID}" | tee -a "${LOG}"
echo "Source: ${SOURCE_DIR} ($(du -sh "${SOURCE_DIR}" | cut -f1))" | tee -a "${LOG}"

hf upload-large-folder "${DATASET_ID}" "${SOURCE_DIR}" \
  --type dataset \
  2>&1 | tee -a "${LOG}"

echo "=== HF upload finished at $(date) ===" | tee -a "${LOG}"

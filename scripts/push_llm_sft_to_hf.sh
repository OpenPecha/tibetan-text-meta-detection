#!/bin/bash
# Upload LLM SFT JSONL to Hugging Face (resumable large-folder upload).
#
# Examples:
#   bash scripts/push_llm_sft_to_hf.sh pilot
#   bash scripts/push_llm_sft_to_hf.sh full
#   DATASET_ID=user/repo SOURCE_DIR=data/custom bash scripts/push_llm_sft_to_hf.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-pilot}"

case "${MODE}" in
  pilot)
    DATASET_ID="${DATASET_ID:-ganga4364/tibetan-metadata-llm-sft}"
    SOURCE_DIR="${SOURCE_DIR:-data/llm_sft_pilot_10pct}"
    README_SRC="${README_SRC:-hub/llm_sft_pilot_README.md}"
    LOG="${LOG:-data/llm_sft_pilot_10pct/logs/hf_upload.log}"
    ;;
  full)
    DATASET_ID="${DATASET_ID:-ganga4364/tibetan-metadata-llm-sft-full}"
    SOURCE_DIR="${SOURCE_DIR:-data/llm_sft}"
    README_SRC="${README_SRC:-hub/llm_sft_full_README.md}"
    LOG="${LOG:-data/llm_sft/logs/hf_upload.log}"
    ;;
  *)
    echo "Usage: $0 {pilot|full}"
    echo "  pilot -> ganga4364/tibetan-metadata-llm-sft (10% subset)"
    echo "  full  -> ganga4364/tibetan-metadata-llm-sft-full"
    exit 1
    ;;
esac

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || cat "${HOME}/.hf_token" 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Missing HF token — run: bash scripts/setup_hf_token.sh"
  exit 1
fi

if [[ ! -f "${SOURCE_DIR}/title/train.jsonl" ]]; then
  echo "Missing ${SOURCE_DIR}/title/train.jsonl"
  exit 1
fi

# Ensure dataset card + LLaMA-Factory registry are in the upload tree.
if [[ -f "${README_SRC}" ]]; then
  cp "${README_SRC}" "${SOURCE_DIR}/README.md"
fi
if [[ ! -f "${SOURCE_DIR}/dataset_info.json" && -f data/llm_sft/dataset_info.json ]]; then
  cp data/llm_sft/dataset_info.json "${SOURCE_DIR}/dataset_info.json"
fi

mkdir -p "$(dirname "${LOG}")"
hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true

echo "=== HF upload (${MODE}) started at $(date) ===" | tee "${LOG}"
echo "Repo: ${DATASET_ID}" | tee -a "${LOG}"
echo "Source: ${SOURCE_DIR} ($(du -sh "${SOURCE_DIR}" | cut -f1))" | tee -a "${LOG}"
wc -l "${SOURCE_DIR}"/title/*.jsonl "${SOURCE_DIR}"/author/*.jsonl 2>/dev/null | tee -a "${LOG}" || true

hf upload-large-folder "${DATASET_ID}" "${SOURCE_DIR}" \
  --repo-type dataset \
  --exclude "logs/*" \
  2>&1 | tee -a "${LOG}"

echo "=== HF upload (${MODE}) finished at $(date) ===" | tee -a "${LOG}"
echo "Dataset: https://huggingface.co/datasets/${DATASET_ID}" | tee -a "${LOG}"

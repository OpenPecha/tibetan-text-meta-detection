#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRACTED_DIR="${EXTRACTED_DIR:-data/extracted}"
OUTPUT_DIR="${OUTPUT_DIR:-data/llm_sft}"
TOKENIZER="${TOKENIZER:-YoLo2000/TiLamb-7B}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-3584}"
CROPS_PER_POSITIVE="${CROPS_PER_POSITIVE:-3}"
SEED="${SEED:-42}"

if [[ ! -f "${EXTRACTED_DIR}/index.jsonl" ]]; then
  echo "Missing ${EXTRACTED_DIR}/index.jsonl — run: bash scripts/run_pull_extracted.sh"
  exit 1
fi

echo "Building LLM SFT data from ${EXTRACTED_DIR} -> ${OUTPUT_DIR}"
python3 -u -m llm_sft.build_dataset \
  --extracted-dir "${EXTRACTED_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
  --crops-per-positive "${CROPS_PER_POSITIVE}" \
  --seed "${SEED}"

echo "Done. Stats: ${OUTPUT_DIR}/reports/crop_stats.json"
wc -l "${OUTPUT_DIR}"/title/*.jsonl "${OUTPUT_DIR}"/author/*.jsonl

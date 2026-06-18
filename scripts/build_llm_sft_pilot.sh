#!/bin/bash
# Build 10% pilot subset from full LLM SFT JSONL.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE_DIR="${SOURCE_DIR:-data/llm_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-data/llm_sft_pilot_10pct}"
FRACTION="${FRACTION:-0.10}"
SEED="${SEED:-42}"

if [[ ! -f "${SOURCE_DIR}/title/train.jsonl" ]]; then
  echo "Missing ${SOURCE_DIR}/title/train.jsonl"
  exit 1
fi

python3 -u scripts/subsample_llm_sft.py \
  --source-dir "${SOURCE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fraction "${FRACTION}" \
  --seed "${SEED}"

echo "Pilot subset: ${OUTPUT_DIR}"
wc -l "${OUTPUT_DIR}"/title/*.jsonl "${OUTPUT_DIR}"/author/*.jsonl

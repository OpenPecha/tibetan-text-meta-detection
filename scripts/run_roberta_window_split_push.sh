#!/bin/bash
# Window extracted docs -> train/val/test splits (89/1/10) -> push windows Parquet to HF.
set -euo pipefail
cd /root/tibetan-metadata-detector

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

EXTRACTED_DIR="${EXTRACTED_DIR:-data/extracted}"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"
TRAIN_RATIO="${TRAIN_RATIO:-0.89}"
VAL_RATIO="${VAL_RATIO:-0.01}"
TEST_RATIO="${TEST_RATIO:-0.10}"
LOG="${LOG:-data/roberta_window_split_push.log}"

mkdir -p "$(dirname "$LOG")" "$PROCESSED_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "=== RoBERTa window + split + HF push started at $(date) ==="
echo "  extracted: $EXTRACTED_DIR"
echo "  processed: $PROCESSED_DIR"
echo "  ratios: train=$TRAIN_RATIO val=$VAL_RATIO test=$TEST_RATIO"

echo "=== Step 1: roberta-process ==="
python3 -u prepare_data.py roberta-process \
  --extracted-dir "$EXTRACTED_DIR" \
  --processed-dir "$PROCESSED_DIR"

echo "=== Step 2: balance windows ==="
python3 -u prepare_data.py balance-windows \
  --processed-dir "$PROCESSED_DIR"

echo "=== Step 3: stratified split ==="
python3 -u prepare_data.py split \
  --processed-dir "$PROCESSED_DIR" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO"

echo "=== Step 4: push windows Parquet to HF ==="
pip install -U datasets huggingface_hub pyarrow -q
python3 scripts/push_dataset_parquet.py \
  --windows-only \
  --splits-dir "$PROCESSED_DIR/splits"

echo "=== Done at $(date) ==="

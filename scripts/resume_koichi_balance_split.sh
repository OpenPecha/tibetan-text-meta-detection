#!/bin/bash
# Merge (if needed), balance, split for Koichi tokenizer run — no HF push.
set -euo pipefail
cd /root/tibetan-metadata-detector
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_koichi}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG="${LOG:-data/roberta_koichi_pipeline.log}"

exec >>"$LOG" 2>&1
echo "=== Koichi merge/balance/split at $(date) ==="

if ls "$PROCESSED_DIR"/roberta_all_examples.worker*.jsonl >/dev/null 2>&1; then
  echo "=== merge-roberta-shards ==="
  python3 -u prepare_data.py merge-roberta-shards \
    --processed-dir "$PROCESSED_DIR" \
    --num-workers "$NUM_WORKERS"
fi

echo "=== balance-windows ==="
python3 -u prepare_data.py balance-windows --processed-dir "$PROCESSED_DIR"

echo "=== split 89/1/10 ==="
python3 -u prepare_data.py split \
  --processed-dir "$PROCESSED_DIR" \
  --train-ratio 0.89 \
  --val-ratio 0.01 \
  --test-ratio 0.10

echo "=== Done at $(date) ==="

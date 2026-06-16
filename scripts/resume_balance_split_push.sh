#!/bin/bash
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"
LOG="${LOG:-data/roberta_merge_balance_push.log}"

exec >>"$LOG" 2>&1
echo "=== resume balance/split/push at $(date) ==="

echo "=== balance-windows ==="
python3 -u prepare_data.py balance-windows --processed-dir "$PROCESSED_DIR"

echo "=== split 89/1/10 ==="
python3 -u prepare_data.py split \
  --processed-dir "$PROCESSED_DIR" \
  --train-ratio 0.89 \
  --val-ratio 0.01 \
  --test-ratio 0.10

echo "=== push Parquet to HF ==="
bash scripts/push_balanced_splits.sh

echo "=== Done at $(date) ==="

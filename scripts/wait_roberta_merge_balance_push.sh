#!/bin/bash
# Wait for all roberta-process workers, merge shards, balance, split, push HF.
set -euo pipefail
cd /root/tibetan-metadata-detector

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

NUM_WORKERS="${NUM_WORKERS:-6}"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"
EXTRACTED_DIR="${EXTRACTED_DIR:-data/extracted}"
LOG="${LOG:-data/roberta_merge_balance_push.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== wait_roberta_merge_balance_push started at $(date) ==="

echo "Waiting for $NUM_WORKERS roberta-process workers..."
while pgrep -f "prepare_data.py roberta-process" >/dev/null; do
  sleep 120
done
echo "All workers finished at $(date)"

echo "=== merge-roberta-shards ==="
python3 -u prepare_data.py merge-roberta-shards \
  --processed-dir "$PROCESSED_DIR" \
  --num-workers "$NUM_WORKERS"

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

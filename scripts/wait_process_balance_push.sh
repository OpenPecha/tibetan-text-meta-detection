#!/bin/bash
# Wait for roberta-process to finish, then balance + split + push (no training).
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

LOG="${LOG:-data/wait_balance_push.log}"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"

exec > >(tee -a "$LOG") 2>&1
echo "=== wait_process_balance_push started at $(date) ==="

echo "Waiting for roberta-process..."
while pgrep -f "prepare_data.py roberta-process" >/dev/null; do
  sleep 120
done
echo "roberta-process finished at $(date)"

if ! test -f "$PROCESSED_DIR/roberta_all_examples.jsonl"; then
  echo "ERROR: missing roberta_all_examples.jsonl"
  exit 1
fi

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

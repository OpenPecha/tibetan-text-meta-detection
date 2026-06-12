#!/bin/bash
# Push balanced window splits (train/val/test JSONL) as Parquet to Hugging Face.
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

SPLITS_DIR="${SPLITS_DIR:-data/roberta_full/splits}"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"
REPO_ID="${REPO_ID:-ganga4364/tibetan-metadata-detector}"

if ! test -f "$SPLITS_DIR/train.jsonl"; then
  echo "Missing splits under $SPLITS_DIR"
  echo "Run: prepare_data.py balance-windows && prepare_data.py split"
  exit 1
fi

echo "=== Split counts ==="
wc -l "$SPLITS_DIR"/*.jsonl

pip install -U datasets huggingface_hub pyarrow -q
python3 scripts/push_dataset_parquet.py \
  --windows-only \
  --splits-dir "$SPLITS_DIR" \
  --repo-id "$REPO_ID"

echo "Done: https://huggingface.co/datasets/$REPO_ID"

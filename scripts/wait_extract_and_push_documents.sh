#!/bin/bash
# Wait for multi-worker extraction, rebuild index, push documents Parquet.
set -euo pipefail

cd /root/tibetan-metadata-detector
NUM_WORKERS="${NUM_WORKERS:-6}"
TARGET="${TARGET:-3794}"
LOG_DIR="${LOG_DIR:-data/extract_logs}"
POLL_SECS="${POLL_SECS:-60}"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

set -a
# shellcheck disable=SC1091
source .env
set +a

pip install -U datasets huggingface_hub pyarrow psycopg2-binary -q

echo "=== Waiting for $NUM_WORKERS workers (target ~$TARGET docs) ==="
while true; do
  running="$(pgrep -fc "extract_data.py" || true)"
  done_count="$(ls data/extracted/texts 2>/dev/null | wc -l)"
  echo "$(date -Is) running_workers=$running extracted=$done_count/$TARGET"
  if [[ "$running" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SECS"
done

echo "=== Rebuilding index.jsonl ==="
python3 extract_data.py --rebuild-index

echo "=== Pushing documents config as Parquet ==="
python3 -u scripts/push_dataset_parquet.py --documents-only \
  | tee data/extract_logs/push_documents.log

echo "=== Uploading dataset card ==="
hf upload ganga4364/tibetan-metadata-detector hub/dataset_README.md README.md \
  --type dataset --commit-message=DocumentsParquetCard

echo "=== Done ==="

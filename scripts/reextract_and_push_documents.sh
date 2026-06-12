#!/bin/bash
# Re-extract raw documents from DB, then push documents config as Parquet.
set -euo pipefail

cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

set -a
# shellcheck disable=SC1091
source .env
set +a

pip install -U datasets huggingface_hub pyarrow psycopg2-binary -q

NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_DIR="data/extract_logs"
mkdir -p "$LOG_DIR"

echo "=== Re-extracting documents ($NUM_WORKERS workers) ==="
for ((wid=0; wid<NUM_WORKERS; wid++)); do
  python3 -u extract_data.py --all \
    --num-workers "$NUM_WORKERS" \
    --worker-id "$wid" \
    >"$LOG_DIR/extract_worker${wid}.log" 2>&1 &
  echo "  worker $wid PID=$!"
done
wait
python3 extract_data.py --rebuild-index

echo "=== Pushing documents config as Parquet ==="
python3 scripts/push_dataset_parquet.py --documents-only

echo "=== Done ==="

#!/bin/bash
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

pip install -U huggingface_hub pyarrow -q

echo "Clearing HF cache to free disk for streaming upload…"
rm -rf /root/.cache/huggingface/datasets /root/.cache/huggingface/hub/datasets--*

if [[ ! -f data/extracted/index.jsonl ]]; then
  echo "Rebuilding index.jsonl…"
  python3 extract_data.py --rebuild-index
fi

python3 -u scripts/push_extracted_parquet.py "$@"

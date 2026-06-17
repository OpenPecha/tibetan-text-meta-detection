#!/bin/bash
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -U datasets huggingface_hub pyarrow -q
python3 scripts/push_dataset_parquet.py --windows-only "$@"

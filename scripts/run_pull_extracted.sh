#!/bin/bash
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p data
python3 -u scripts/pull_extracted_from_hf.py --output-dir data/extracted

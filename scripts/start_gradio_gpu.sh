#!/bin/bash
set -euo pipefail
cd /root/tibetan-metadata-detector/space
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install -q -r requirements.txt
exec python3 -u app.py 2>&1 | tee /root/tibetan-metadata-detector/data/gradio.log

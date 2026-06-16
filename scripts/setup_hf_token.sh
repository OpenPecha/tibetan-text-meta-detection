#!/bin/bash
set -euo pipefail
chmod 600 /root/.hf_token
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python3 <<'PY'
from huggingface_hub import login
login(token=open("/root/.hf_token").read().strip())
print("HF login ok")
PY
grep -q 'HF_TOKEN' /root/.bashrc || cat >> /root/.bashrc <<'EOF'

export HF_TOKEN=$(cat /root/.hf_token 2>/dev/null)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
EOF

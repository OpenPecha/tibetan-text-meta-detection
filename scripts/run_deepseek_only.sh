#!/bin/bash
set -euo pipefail
cd /root/tibetan-text-meta-detection
source /root/infer-venv/bin/activate
export HF_TOKEN="$(cat /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE=0 MODEL=deepseek_r1_14b bash scripts/run_benchmark_suite.sh

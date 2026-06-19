#!/bin/bash
# PEFT title LoRA inference (no Unsloth).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
JSONL="${JSONL:-${REPO}/data/llm_sft_sample/title/test.jsonl}"
ROW="${ROW:-0}"
ADAPTER="${ADAPTER:-ganga4364/tibetan-metadata-title-tilamb-lora-pilot}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3.11 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip wheel
pip install \
  "torch==2.6.0+cu124" \
  "torchvision==0.21.0+cu124" \
  "torchaudio==2.6.0+cu124" \
  --index-url https://download.pytorch.org/whl/cu124
pip install \
  "transformers==4.56.2" \
  "peft==0.18.1" \
  "accelerate==1.10.1" \
  "bitsandbytes>=0.45.0" \
  "sentencepiece" \
  "huggingface_hub"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

cd "${REPO}"
python3 -u scripts/run_title_lora_inference.py "${JSONL}" "${ROW}" "${ADAPTER}"

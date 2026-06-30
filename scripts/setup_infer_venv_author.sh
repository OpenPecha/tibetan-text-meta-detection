#!/bin/bash
# Minimal inference venv for the author LoRA benchmark (LoRA-only, no Koichi/title).
set -euo pipefail
VENV="${VENV:-/root/infer-venv}"
REPO="${REPO:-/root/tibetan-text-meta-detection}"
cd "${REPO}"

if [ ! -x "${VENV}/bin/python" ]; then
  python3.11 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip setuptools wheel packaging
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.56" accelerate peft bitsandbytes sentencepiece datasets \
  python-dotenv huggingface_hub
if [ -f /root/.hf_token ]; then
  bash scripts/setup_hf_token.sh || true
fi
echo INFER_SETUP_DONE

#!/bin/bash
# Run one-row TiLamb title LoRA inference on vastai7 (PEFT, no Unsloth Studio).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/unsloth-venv}"
JSONL="${JSONL:-${REPO}/data/llm_sft_sample/title/test.jsonl}"
ROW="${ROW:-0}"
ADAPTER="${ADAPTER:-/root/lora/tibetan-title-pilot}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"

# Match LLaMA-Factory training stack; avoid transformers 5.x + torchao breakage.
pip install -q "transformers==4.56.2" "peft>=0.18.0" "accelerate>=1.0.0" bitsandbytes

python3 -u -m llm_sft.inference \
  --jsonl "${JSONL}" \
  --row "${ROW}" \
  --task title \
  --adapter "${ADAPTER}"

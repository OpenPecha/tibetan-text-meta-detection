#!/bin/bash
# Fix Unsloth venv deps and run title LoRA inference on one test row.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/unsloth-venv}"
JSONL="${JSONL:-${REPO}/data/llm_sft_sample/title/test.jsonl}"
ROW="${ROW:-0}"
ADAPTER="${ADAPTER:-/root/lora/tibetan-title-pilot}"
LOG="${LOG:-/tmp/unsloth_title_inference.log}"

exec > >(tee -a "${LOG}") 2>&1
echo "=== Unsloth title inference started $(date) ==="

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Unsloth bootstrap pulled transformers 5.x + torchao; pin LLaMA-Factory-compatible stack.
pip uninstall -y torchao 2>/dev/null || true
pip install -q --force-reinstall \
  "transformers==4.56.2" \
  "peft==0.18.1" \
  "accelerate==1.10.1"

python3 -c "import unsloth; from unsloth import FastLanguageModel; print('unsloth_ok')"
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

cd "${REPO}"
python3 -u scripts/run_unsloth_title_inference.py "${JSONL}" "${ROW}" "${ADAPTER}"

echo "=== Done $(date) ==="

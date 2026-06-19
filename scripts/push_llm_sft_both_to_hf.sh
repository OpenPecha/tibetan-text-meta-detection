#!/bin/bash
# Upload pilot (10%) then full LLM SFT datasets to separate HF repos.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Step 1/2: pilot 10% -> ganga4364/tibetan-metadata-llm-sft ==="
bash scripts/push_llm_sft_to_hf.sh pilot

echo "=== Step 2/2: full -> ganga4364/tibetan-metadata-llm-sft-full ==="
bash scripts/push_llm_sft_to_hf.sh full

echo "=== Both uploads complete ==="

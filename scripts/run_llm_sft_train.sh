#!/bin/bash
set -euo pipefail
source /root/llama-venv/bin/activate
cd /root/LLaMA-Factory
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
REPO=/root/tibetan-text-meta-detection
LOG_DIR="${REPO}/data/llm_sft/logs"
mkdir -p "${LOG_DIR}"

echo "=== Title LoRA started at $(date) ===" | tee "${LOG_DIR}/title_train.log"
llamafactory-cli train "${REPO}/configs/llama_factory/title_lora_sft.yaml" 2>&1 | tee -a "${LOG_DIR}/title_train.log"
echo "=== Title LoRA finished at $(date) ===" | tee -a "${LOG_DIR}/title_train.log"

echo "=== Author LoRA started at $(date) ===" | tee "${LOG_DIR}/author_train.log"
llamafactory-cli train "${REPO}/configs/llama_factory/author_lora_sft.yaml" 2>&1 | tee -a "${LOG_DIR}/author_train.log"
echo "=== Author LoRA finished at $(date) ===" | tee -a "${LOG_DIR}/author_train.log"

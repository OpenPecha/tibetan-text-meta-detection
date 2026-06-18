#!/bin/bash
# Upload LLaMA-Factory LoRA adapters to Hugging Face Hub.
# Usage: bash scripts/push_lora_to_hf.sh [title|author|all]
set -euo pipefail

TASK="${1:-all}"
HF_USER="${HF_USER:-ganga4364}"
SAVES_ROOT="${SAVES_ROOT:-/root/LLaMA-Factory/saves}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/hf-lora-upload}"
LOG_DIR="${LOG_DIR:-/tmp/hf-lora-upload/logs}"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Missing /root/.hf_token"
  exit 1
fi

hf auth login --token "${HF_TOKEN}" >/dev/null 2>&1 || true
mkdir -p "${LOG_DIR}"

stage_and_upload() {
  local task="$1"
  local src="${SAVES_ROOT}/tibetan-${task}-lora-pilot"
  local repo="${HF_USER}/tibetan-metadata-${task}-tilamb-lora-pilot"
  local stage="${STAGE_ROOT}/tibetan-${task}-lora-pilot"
  local log="${LOG_DIR}/${task}.log"

  if [[ ! -f "${src}/adapter_model.safetensors" ]]; then
    echo "SKIP ${task}: missing ${src}/adapter_model.safetensors"
    return 1
  fi

  rm -rf "${stage}"
  mkdir -p "${stage}"

  cp "${src}/adapter_config.json" \
     "${src}/adapter_model.safetensors" \
     "${src}/special_tokens_map.json" \
     "${src}/tokenizer.json" \
     "${src}/tokenizer.model" \
     "${src}/tokenizer_config.json" \
     "${stage}/"

  [[ -f "${src}/chat_template.jinja" ]] && cp "${src}/chat_template.jinja" "${stage}/"
  [[ -f "${src}/eval_results.json" ]] && cp "${src}/eval_results.json" "${stage}/"
  [[ -f "${src}/train_results.json" ]] && cp "${src}/train_results.json" "${stage}/"

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  python3 "${SCRIPT_DIR}/write_lora_readme.py" "${task}" "${stage}/README.md" --repo "${repo}"

  echo "=== Uploading ${repo} from ${src} ===" | tee "${log}"
  hf upload "${repo}" "${stage}" \
    --repo-type model \
    --commit-message "Upload TiLamb ${task} LoRA pilot adapter" \
    2>&1 | tee -a "${log}"
  echo "=== Done: https://huggingface.co/${repo} ===" | tee -a "${log}"
}

case "${TASK}" in
  title) stage_and_upload title ;;
  author) stage_and_upload author ;;
  all)
    stage_and_upload title
    stage_and_upload author || true
    ;;
  *)
    echo "Usage: $0 [title|author|all]"
    exit 1
    ;;
esac

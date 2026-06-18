#!/bin/bash
# Train title LoRA, then author LoRA, in a tmux session (TiLamb-7B via LLaMA-Factory).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/root/LLaMA-Factory}"
SESSION="${SESSION:-llm_sft_train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/data/llm_sft/logs}"

mkdir -p "${LOG_DIR}"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

bash "${REPO_ROOT}/scripts/setup_llama_factory.sh"

TITLE_CFG="${REPO_ROOT}/configs/llama_factory/title_lora_sft.yaml"
AUTHOR_CFG="${REPO_ROOT}/configs/llama_factory/author_lora_sft.yaml"
TITLE_LOG="${LOG_DIR}/title_train.log"
AUTHOR_LOG="${LOG_DIR}/author_train.log"

tmux kill-session -t "${SESSION}" 2>/dev/null || true

tmux new-session -d -s "${SESSION}" bash -lc "
  set -euo pipefail
  cd '${LLAMA_DIR}'
  export HF_TOKEN=\$(cat /root/.hf_token 2>/dev/null || true)
  export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN

  echo '=== Title LoRA started at \$(date) ===' | tee '${TITLE_LOG}'
  llamafactory-cli train '${TITLE_CFG}' 2>&1 | tee -a '${TITLE_LOG}'
  echo '=== Title LoRA finished at \$(date) ===' | tee -a '${TITLE_LOG}'

  echo '=== Author LoRA started at \$(date) ===' | tee '${AUTHOR_LOG}'
  llamafactory-cli train '${AUTHOR_CFG}' 2>&1 | tee -a '${AUTHOR_LOG}'
  echo '=== Author LoRA finished at \$(date) ===' | tee -a '${AUTHOR_LOG}'
"

echo "LLM SFT training started in tmux session: ${SESSION}"
echo "  Attach:    tmux attach -t ${SESSION}"
echo "  Title log: tail -f ${TITLE_LOG}"
echo "  Author log: tail -f ${AUTHOR_LOG}"

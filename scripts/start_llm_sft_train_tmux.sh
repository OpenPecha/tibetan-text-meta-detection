#!/bin/bash
# Train title LoRA, then author LoRA, in a tmux session (TiLamb-7B via LLaMA-Factory).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-llm_sft_train}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/data/llm_sft/logs}"
VENV="${VENV:-/root/llama-venv}"
RUNNER="${REPO_ROOT}/scripts/run_llm_sft_train.sh"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${VENV}/bin/llamafactory-cli" ]]; then
  echo "Missing ${VENV}/bin/llamafactory-cli — run setup_llama_factory.sh first"
  exit 1
fi

bash "${REPO_ROOT}/scripts/setup_llama_factory.sh"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" bash -lc "bash '${RUNNER}'"

echo "LLM SFT training started in tmux session: ${SESSION}"
echo "  Attach:     tmux attach -t ${SESSION}"
echo "  Title log:  tail -f ${LOG_DIR}/title_train.log"
echo "  Author log: tail -f ${LOG_DIR}/author_train.log"

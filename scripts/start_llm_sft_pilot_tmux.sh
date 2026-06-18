#!/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-llm_sft_pilot}"
RUNNER="${REPO_ROOT}/scripts/run_llm_sft_pilot_train.sh"
LOG_DIR="${REPO_ROOT}/data/llm_sft/logs"

bash "${REPO_ROOT}/scripts/build_llm_sft_pilot.sh"
bash "${REPO_ROOT}/scripts/setup_llama_factory.sh"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" bash -lc "bash '${RUNNER}'"

echo "Pilot LLM SFT training started in tmux: ${SESSION}"
echo "  Attach:     tmux attach -t ${SESSION}"
echo "  Title log:  tail -f ${LOG_DIR}/title_pilot_train.log"
echo "  Author log: tail -f ${LOG_DIR}/author_pilot_train.log"

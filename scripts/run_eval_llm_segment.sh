#!/bin/bash
# Run TiLamb title LoRA segment eval — one segment at a time with logging.
#   bash scripts/run_eval_llm_segment.sh          # next 1 segment (--resume)
#   LIMIT=1 bash scripts/run_eval_llm_segment.sh    # explicit 1 segment
#   FULL=1 bash scripts/run_eval_llm_segment.sh     # all segments in tmux
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"
LIMIT="${LIMIT:-1}"
FULL="${FULL:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/llm_title_segment_eval_${STAMP}.log"

mkdir -p "${LOG_DIR}"
cd "${REPO}"

if [[ "${FULL}" == "1" ]]; then
  echo "=== Full segment eval in tmux (one-by-one, session: llm_eval) ==="
  tmux kill-session -t llm_eval 2>/dev/null || true
  tmux new-session -d -s llm_eval \
    "cd ${REPO} && FULL=1 GENERATE_REPORT=1 bash scripts/run_eval_llm_segment_inner.sh >> ${RUN_LOG} 2>&1"
  echo "Started tmux session llm_eval - attach: tmux attach -t llm_eval"
  echo "Session log: ${RUN_LOG}"
  echo "Per-segment log: logs/llm_segment_eval_run.log"
  exit 0
fi

echo "=== Preflight GPU ===" | tee "${RUN_LOG}"
if ! nvidia-smi >> "${RUN_LOG}" 2>&1; then
  echo "ERROR: nvidia-smi failed - reboot instance to fix driver mismatch" | tee -a "${RUN_LOG}"
  exit 1
fi

bash scripts/pull_segment_test_split.sh 2>&1 | tee -a "${RUN_LOG}"

echo "=== Segment eval: 1 segment at a time (limit=${LIMIT}) ===" | tee -a "${RUN_LOG}"
LIMIT="${LIMIT}" FULL=0 bash scripts/run_eval_llm_segment_inner.sh 2>&1 | tee -a "${RUN_LOG}"

echo "" | tee -a "${RUN_LOG}"
echo "Next segment: bash scripts/run_eval_llm_segment.sh" | tee -a "${RUN_LOG}"
echo "Full eval:      FULL=1 bash scripts/run_eval_llm_segment.sh" | tee -a "${RUN_LOG}"
echo "Run log:        logs/llm_segment_eval_run.log" | tee -a "${RUN_LOG}"

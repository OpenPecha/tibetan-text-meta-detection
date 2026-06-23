#!/bin/bash
# Run full pilot title benchmark across all model families (sequential, resumable).
#
# Usage:
#   bash scripts/run_benchmark_suite.sh              # smoke: 5 rows per model
#   SMOKE=0 bash scripts/run_benchmark_suite.sh      # full eval in tmux
#   MODEL=koichi bash scripts/run_benchmark_suite.sh   # single model only
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
PILOT_DIR="${PILOT_DIR:-data/llm_sft_pilot_10pct}"
KOICHI_DIR="${KOICHI_DIR:-models/koichi-ner}"
LORA_DIR="${LORA_DIR:-/root/lora/tibetan-title-pilot}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"
SMOKE="${SMOKE:-1}"
LIMIT="${LIMIT:-5}"
TMUX_SESSION="${TMUX_SESSION:-benchmark_pilot}"

cd "${REPO}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing venv at ${VENV} — run: bash scripts/bootstrap_vastai.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

TEST_JSONL="${PILOT_DIR}/title/test.jsonl"
META_JSONL="${PILOT_DIR}/title/test_meta.jsonl"

if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "ERROR: missing ${TEST_JSONL} — run bootstrap or hf download"
  exit 1
fi

run_one() {
  local kind="$1"
  shift
  local extra_args=("$@")
  local predictions="${LOG_DIR}/benchmark_${kind}_predictions.jsonl"
  local metrics="${LOG_DIR}/benchmark_${kind}_metrics.json"
  local limit_args=()
  if [[ "${SMOKE}" == "1" ]]; then
    limit_args=(--limit "${LIMIT}")
  fi
  echo "=== Benchmark ${kind} (smoke=${SMOKE}) ==="
  python3 -u eval_benchmark_rows.py \
    --model-kind "${kind}" \
    --test-jsonl "${TEST_JSONL}" \
    --meta-jsonl "${META_JSONL}" \
    --predictions "${predictions}" \
    --metrics-out "${metrics}" \
    --resume \
    "${limit_args[@]}" \
    "${extra_args[@]}"
}

run_all() {
  mkdir -p "${LOG_DIR}"
  if [[ -n "${MODEL:-}" ]]; then
    case "${MODEL}" in
      koichi) run_one koichi --checkpoint "${KOICHI_DIR}" ;;
      tilamb) run_one tilamb ;;
      tilamb_lora) run_one tilamb_lora --adapter "${LORA_DIR}" ;;
      alpaca) run_one alpaca ;;
      qwen) run_one qwen ;;
      gemma4) run_one gemma4 ;;
      qwen36_27b) run_one qwen36_27b ;;
      deepseek_r1_14b) run_one deepseek_r1_14b ;;
      *) echo "Unknown MODEL=${MODEL}"; exit 1 ;;
    esac
  else
    local failed=0
    run_one koichi --checkpoint "${KOICHI_DIR}" || failed=1
    run_one tilamb || failed=1
    run_one tilamb_lora --adapter "${LORA_DIR}" || failed=1
    run_one alpaca || failed=1
    run_one qwen || failed=1
    run_one gemma4 || failed=1
    run_one qwen36_27b || failed=1
    run_one deepseek_r1_14b || failed=1
    if [[ "${failed}" -ne 0 ]]; then
      echo "WARN: one or more models failed; check logs above"
    fi
  fi
  python3 scripts/compare_benchmark.py \
    --metrics-dir "${LOG_DIR}" \
    --output-md docs/metrics/benchmark_pilot_title.md \
    --output-json docs/metrics/benchmark_pilot_title.json
  echo "Leaderboard: docs/metrics/benchmark_pilot_title.md"
}

if [[ "${SMOKE}" == "0" ]] && [[ -z "${MODEL:-}" ]] && [[ -t 1 ]] && command -v tmux >/dev/null; then
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "Attach: tmux attach -t ${TMUX_SESSION}"
    exit 0
  fi
  tmux new-session -d -s "${TMUX_SESSION}" \
    "cd ${REPO} && SMOKE=0 bash scripts/run_benchmark_suite.sh 2>&1 | tee ${LOG_DIR}/benchmark_suite.log"
  echo "Started full benchmark in tmux session: ${TMUX_SESSION}"
  echo "  tmux attach -t ${TMUX_SESSION}"
  echo "  tail -f ${LOG_DIR}/benchmark_suite.log"
else
  run_all
fi

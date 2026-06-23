#!/bin/bash
# Wait for Qwen benchmark to finish, then run Alpaca + Gemma 4 and refresh leaderboard.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
LOG="${LOG:-${REPO}/logs/benchmark_followup.log}"
QWEN_PRED="${REPO}/logs/benchmark_qwen_predictions.jsonl"
QWEN_METRICS="${REPO}/logs/benchmark_qwen_metrics.json"
EXPECTED_ROWS=769

exec >>"${LOG}" 2>&1
echo "=== benchmark_followup started at $(date) ==="

cd "${REPO}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

while true; do
  if [[ -f "${QWEN_METRICS}" ]]; then
    echo "Qwen metrics found at $(date)"
    break
  fi
  if [[ -f "${QWEN_PRED}" ]]; then
    n="$(wc -l < "${QWEN_PRED}")"
    if [[ "${n}" -ge "${EXPECTED_ROWS}" ]]; then
      echo "Qwen predictions complete (${n} rows) at $(date)"
      break
    fi
    echo "Waiting for Qwen: ${n}/${EXPECTED_ROWS} rows..."
  else
    echo "Waiting for Qwen predictions file..."
  fi
  sleep 60
done

# Wait for benchmark_resume tmux to release GPU if still running
while tmux has-session -t benchmark_resume 2>/dev/null; do
  echo "Waiting for benchmark_resume session to finish..."
  sleep 30
done

echo "Installing deps for Alpaca (tiktoken)..."
pip install -q tiktoken

run_model() {
  local model="$1"
  local metrics="${REPO}/logs/benchmark_${model}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "Skip ${model}: metrics already exist"
    return 0
  fi
  echo "=== Running ${model} at $(date) ==="
  SMOKE=0 MODEL="${model}" bash scripts/run_benchmark_suite.sh || echo "WARN: ${model} failed"
}

run_model alpaca
run_model gemma4

python3 scripts/compare_benchmark.py \
  --metrics-dir "${REPO}/logs" \
  --output-md "${REPO}/docs/metrics/benchmark_pilot_title.md" \
  --output-json "${REPO}/docs/metrics/benchmark_pilot_title.json"

echo "=== benchmark_followup done at $(date) ==="
echo "Leaderboard: ${REPO}/docs/metrics/benchmark_pilot_title.md"

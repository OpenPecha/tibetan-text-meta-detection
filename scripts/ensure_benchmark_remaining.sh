#!/bin/bash
# Wait for the main benchmark tmux job, then run any missing models and refresh leaderboard.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
MAIN_SESSION="${MAIN_SESSION:-benchmark_pilot}"
LOG="${LOG:-${REPO}/logs/benchmark_ensure.log}"

exec >>"${LOG}" 2>&1
echo "=== ensure_benchmark_remaining started at $(date) ==="

while tmux has-session -t "${MAIN_SESSION}" 2>/dev/null; do
  echo "Waiting for ${MAIN_SESSION}... ($(date))"
  sleep 120
done

echo "Main session finished at $(date)"

cd "${REPO}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

MODELS=(koichi tilamb tilamb_lora alpaca qwen gemma4 qwen36_27b deepseek_r1_14b)
for m in "${MODELS[@]}"; do
  metrics="${REPO}/logs/benchmark_${m}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "OK: ${m} metrics present"
    continue
  fi
  echo "=== Running missing model: ${m} at $(date) ==="
  SMOKE=0 MODEL="${m}" bash scripts/run_benchmark_suite.sh || {
    echo "WARN: ${m} failed at $(date)" >&2
  }
done

python3 scripts/compare_benchmark.py \
  --metrics-dir "${REPO}/logs" \
  --output-md "${REPO}/docs/metrics/benchmark_pilot_title.md" \
  --output-json "${REPO}/docs/metrics/benchmark_pilot_title.json"

echo "=== ensure_benchmark_remaining done at $(date) ==="
echo "Leaderboard: ${REPO}/docs/metrics/benchmark_pilot_title.md"

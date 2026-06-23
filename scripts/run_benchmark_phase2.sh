#!/bin/bash
# Phase 2: after Gemma 4 (+ Alpaca) finish, run Qwen3.6-27B and DeepSeek-R1-14B.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
LOG="${LOG:-${REPO}/logs/benchmark_phase2.log}"
EXPECTED_ROWS=769

exec >>"${LOG}" 2>&1
echo "=== benchmark_phase2 started at $(date) ==="

cd "${REPO}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

wait_for_model() {
  local model="$1"
  local metrics="${REPO}/logs/benchmark_${model}_metrics.json"
  local pred="${REPO}/logs/benchmark_${model}_predictions.jsonl"
  while true; do
    if [[ -f "${metrics}" ]]; then
      echo "${model}: metrics ready"
      return 0
    fi
    if [[ -f "${pred}" ]]; then
      local n
      n="$(wc -l < "${pred}")"
      if [[ "${n}" -ge "${EXPECTED_ROWS}" ]]; then
        echo "${model}: predictions complete (${n})"
        return 0
      fi
      echo "Waiting for ${model}: ${n}/${EXPECTED_ROWS}"
    else
      echo "Waiting for ${model} to start..."
    fi
    sleep 120
  done
}

# Let current followup jobs finish first
for session in benchmark_followup alpaca_retry; do
  while tmux has-session -t "${session}" 2>/dev/null; do
    echo "Waiting for tmux session ${session}..."
    sleep 60
  done
done

wait_for_model gemma4

run_model() {
  local model="$1"
  local metrics="${REPO}/logs/benchmark_${model}_metrics.json"
  if [[ -f "${metrics}" ]]; then
    echo "Skip ${model}: already done"
    return 0
  fi
  echo "=== Running ${model} at $(date) ==="
  SMOKE=0 MODEL="${model}" bash scripts/run_benchmark_suite.sh || echo "WARN: ${model} failed"
}

# Alpaca may still be missing if earlier retry failed
if [[ ! -f "${REPO}/logs/benchmark_alpaca_metrics.json" ]]; then
  pip install -q tiktoken sentencepiece
  run_model alpaca
fi

run_model qwen36_27b
run_model deepseek_r1_14b

python3 scripts/compare_benchmark.py \
  --metrics-dir "${REPO}/logs" \
  --output-md "${REPO}/docs/metrics/benchmark_pilot_title.md" \
  --output-json "${REPO}/docs/metrics/benchmark_pilot_title.json"

echo "=== benchmark_phase2 done at $(date) ==="

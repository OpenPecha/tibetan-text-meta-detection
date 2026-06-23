#!/bin/bash
# Retry Alpaca after Gemma 4 finishes (GPU free).
set -euo pipefail
REPO=/root/tibetan-text-meta-detection
LOG=$REPO/logs/benchmark_alpaca_retry.log
exec >>"$LOG" 2>&1
echo "=== alpaca retry watcher started $(date) ==="
source /root/infer-venv/bin/activate
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
cd "$REPO"
while tmux has-session -t benchmark_followup 2>/dev/null; do
  n=$(wc -l < logs/benchmark_gemma4_predictions.jsonl 2>/dev/null || echo 0)
  echo "Waiting for gemma4/followup: gemma rows=$n $(date)"
  sleep 120
done
if [[ -f logs/benchmark_alpaca_metrics.json ]]; then
  echo "Alpaca metrics already exist, skip"
  exit 0
fi
echo "=== Running alpaca retry $(date) ==="
SMOKE=0 MODEL=alpaca bash scripts/run_benchmark_suite.sh || echo "WARN alpaca retry failed"
python3 scripts/compare_benchmark.py --metrics-dir logs \
  --output-md docs/metrics/benchmark_pilot_title.md \
  --output-json docs/metrics/benchmark_pilot_title.json
echo "=== alpaca retry done $(date) ==="

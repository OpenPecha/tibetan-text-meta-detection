#!/bin/bash
# Wait for window/split/push pipeline, then train with class weights + segment eval.
set -euo pipefail
cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

LOG="${LOG:-data/wait_train.log}"
exec > >(tee -a "$LOG") 2>&1

echo "=== Waiting for roberta_window tmux session at $(date) ==="
while tmux has-session -t roberta_window 2>/dev/null; do
  sleep 120
done

echo "=== Window pipeline finished at $(date) ==="
grep -E "Done at|Error|Traceback" /root/tibetan-metadata-detector/data/roberta_window_split_push.log | tail -5

if ! test -f data/roberta_full/splits/train.jsonl; then
  echo "ERROR: splits missing, aborting train"
  exit 1
fi

echo "=== Training with class weights ==="
python3 -u train_roberta.py \
  --splits-dir data/roberta_full/splits \
  --output-dir data/roberta_full/model \
  --batch-size 64 \
  --epochs 3 \
  --entity-weight 10 \
  --max-val-samples 10000

echo "=== Segment-level evaluation ==="
python3 -u eval_segment.py \
  --splits-dir data/roberta_full/splits \
  --model data/roberta_full/model/best \
  --output data/roberta_full/model/segment_test_metrics.json

echo "=== All done at $(date) ==="

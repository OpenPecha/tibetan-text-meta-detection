#!/bin/bash
set -e
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
cd /root/tibetan-metadata-detector
echo "Waiting for roberta-process..."
while pgrep -f "prepare_data.py roberta-process" > /dev/null; do
  sleep 60
done
echo "roberta-process finished at $(date)"
echo "=== Balance windows ==="
python3 -u prepare_data.py balance-windows --processed-dir data/roberta_full
echo "=== Split (89/1/10) ==="
python3 -u prepare_data.py split \
  --processed-dir data/roberta_full \
  --train-ratio 0.89 \
  --val-ratio 0.01 \
  --test-ratio 0.10
echo "=== Train ==="
python3 -u train_roberta.py \
  --splits-dir data/roberta_full/splits \
  --output-dir data/roberta_full/model \
  --batch-size 16 \
  --epochs 3 \
  --entity-weight 10
echo "=== Segment eval ==="
python3 -u eval_segment.py \
  --splits-dir data/roberta_full/splits \
  --model data/roberta_full/model/best \
  --output data/roberta_full/model/segment_test_metrics.json
echo "=== Done at $(date) ==="

#!/bin/bash
# Run parallel roberta-process workers (one shard JSONL per worker).
set -euo pipefail

cd /root/tibetan-metadata-detector

NUM_WORKERS="${NUM_WORKERS:-6}"
PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_full}"
EXTRACTED_DIR="${EXTRACTED_DIR:-data/extracted}"
LOG_DIR="${LOG_DIR:-data/roberta_process_logs}"
ROBERTA_MODEL="${ROBERTA_MODEL:-spsither/tibetan_RoBERTa_S_e3}"

mkdir -p "$LOG_DIR" "$PROCESSED_DIR"

echo "Starting $NUM_WORKERS roberta-process workers (logs in $LOG_DIR/)"
echo "  model=$ROBERTA_MODEL"
echo "  processed-dir=$PROCESSED_DIR"

for ((wid=0; wid<NUM_WORKERS; wid++)); do
  logfile="$LOG_DIR/roberta_worker${wid}.log"
  if pgrep -f "prepare_data.py roberta-process.*--worker-id ${wid} " >/dev/null 2>&1; then
    echo "  worker $wid already running -> $logfile"
    continue
  fi
  nohup python3 -u prepare_data.py roberta-process \
    --extracted-dir "$EXTRACTED_DIR" \
    --processed-dir "$PROCESSED_DIR" \
    --model "$ROBERTA_MODEL" \
    --num-workers "$NUM_WORKERS" \
    --worker-id "$wid" \
    >"$logfile" 2>&1 &
  echo "  worker $wid PID=$! -> $logfile"
done

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/roberta_worker0.log"
echo "  ls $PROCESSED_DIR/roberta_all_examples.worker*.jsonl | wc -l"
echo ""
echo "When all workers finish:"
echo "  python3 prepare_data.py merge-roberta-shards --processed-dir $PROCESSED_DIR --num-workers $NUM_WORKERS"
echo "  python3 prepare_data.py balance-windows --processed-dir $PROCESSED_DIR"
echo "  python3 prepare_data.py split --processed-dir $PROCESSED_DIR --train-ratio 0.89 --val-ratio 0.01 --test-ratio 0.10"

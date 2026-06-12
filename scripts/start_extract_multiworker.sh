#!/bin/bash
# Run parallel extraction workers with per-worker logs, then rebuild index.
set -euo pipefail

cd /root/tibetan-metadata-detector
NUM_WORKERS="${NUM_WORKERS:-6}"
LOG_DIR="${LOG_DIR:-data/extract_logs}"

set -a
# shellcheck disable=SC1091
source .env
set +a

mkdir -p "$LOG_DIR"

echo "Starting $NUM_WORKERS extraction workers (logs in $LOG_DIR/)"

for ((wid=0; wid<NUM_WORKERS; wid++)); do
  logfile="$LOG_DIR/extract_worker${wid}.log"
  if pgrep -f "extract_data.py --all.*--worker-id ${wid} " >/dev/null 2>&1; then
    echo "  worker $wid already running -> $logfile"
    continue
  fi
  nohup python3 -u extract_data.py --all \
    --num-workers "$NUM_WORKERS" \
    --worker-id "$wid" \
    >"$logfile" 2>&1 &
  echo "  worker $wid PID=$! -> $logfile"
done

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/extract_worker0.log"
echo "  ls data/extracted/texts | wc -l"
echo ""
echo "When all workers finish, rebuild index:"
echo "  python3 extract_data.py --rebuild-index"

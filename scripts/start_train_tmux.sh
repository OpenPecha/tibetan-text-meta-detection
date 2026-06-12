#!/bin/bash
# Start RoBERTa fine-tuning in tmux with live terminal + log file output.
set -euo pipefail

cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

LOG="data/train.log"
SPLITS="data/roberta_full/splits"
OUT="data/roberta_full/model"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EPOCHS="${EPOCHS:-3}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10000}"

mkdir -p data/roberta_full/model
: > "${LOG}"

tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s train bash -lc "
  cd /root/tibetan-metadata-detector
  export HF_TOKEN=\$(cat /root/.hf_token 2>/dev/null)
  export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
  echo '=== Training started at \$(date) ===' | tee -a ${LOG}
  python3 -u train_roberta.py \
    --splits-dir ${SPLITS} \
    --output-dir ${OUT} \
    --batch-size ${BATCH_SIZE} \
    --eval-batch-size ${EVAL_BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --eval-steps ${EVAL_STEPS} \
    --save-steps ${EVAL_STEPS} \
    --max-val-samples ${MAX_VAL_SAMPLES} \
  2>&1 | tee -a ${LOG}
  echo '=== Training finished at \$(date) ===' | tee -a ${LOG}
"

echo "Training started in tmux session: train"
echo "  Attach (live output): tmux attach -t train"
echo "  Detach:               Ctrl-b then d"
echo "  Log file:             tail -f ${LOG}"

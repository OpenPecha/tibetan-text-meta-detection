#!/bin/bash
# Train KoichiYasuoka/roberta-base-tibetan on local Koichi splits (comparison run).
set -euo pipefail

cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

LOG="data/roberta_koichi_train.log"
SPLITS="data/roberta_koichi/splits"
OUT="data/roberta_koichi/model"
MODEL="${MODEL:-KoichiYasuoka/roberta-base-tibetan}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EPOCHS="${EPOCHS:-3}"
EVAL_STEPS="${EVAL_STEPS:-2000}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-10000}"
ENTITY_WEIGHT="${ENTITY_WEIGHT:-10}"

mkdir -p data/roberta_koichi/model
: > "${LOG}"

tmux kill-session -t train_koichi 2>/dev/null || true

tmux new-session -d -s train_koichi bash -lc "
  cd /root/tibetan-metadata-detector
  export HF_TOKEN=\$(cat /root/.hf_token 2>/dev/null)
  export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
  echo '=== Koichi training started at \$(date) ===' | tee -a ${LOG}
  python3 -u train_roberta.py \
    --model ${MODEL} \
    --splits-dir ${SPLITS} \
    --output-dir ${OUT} \
    --batch-size ${BATCH_SIZE} \
    --eval-batch-size ${EVAL_BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --eval-steps ${EVAL_STEPS} \
    --save-steps ${EVAL_STEPS} \
    --max-val-samples ${MAX_VAL_SAMPLES} \
    --entity-weight ${ENTITY_WEIGHT} \
  2>&1 | tee -a ${LOG}
  echo '=== Koichi training finished at \$(date) ===' | tee -a ${LOG}
"

echo "Training started in tmux session: train_koichi"
echo "  Attach: tmux attach -t train_koichi"
echo "  Log:    tail -f ${LOG}"

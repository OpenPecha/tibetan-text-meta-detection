#!/bin/bash
# Core segment eval runner — one segment per invocation by default (--limit 1 --resume).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
ADAPTER="${ADAPTER:-/root/lora/tibetan-title-pilot}"
BASE_MODEL="${BASE_MODEL:-YoLo2000/TiLamb-7B}"
SPLITS_DIR="${SPLITS_DIR:-${REPO}/data/roberta_full/splits}"
EXTRACTED_DIR="${EXTRACTED_DIR:-${REPO}/data/extracted}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"
LIMIT="${LIMIT:-1}"
FULL="${FULL:-0}"
METRICS="${LOG_DIR}/llm_title_segment_metrics.json"
PREDICTIONS="${LOG_DIR}/llm_title_segment_predictions.jsonl"
DETAIL_DIR="${LOG_DIR}/llm_title_segment_details"
RUN_LOG="${LOG_DIR}/llm_segment_eval_run.log"

cd "${REPO}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: missing infer venv at ${VENV}"
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -q python-dotenv datasets 2>/dev/null || true

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"

if [[ ! -f "${SPLITS_DIR}/test.jsonl" ]]; then
  bash scripts/pull_segment_test_split.sh
fi

extra_args=(--run-log "${RUN_LOG}")
if [[ -f "${PREDICTIONS}" ]]; then
  extra_args+=(--resume)
fi
if [[ "${FULL}" != "1" ]]; then
  extra_args+=(--limit "${LIMIT}")
fi

if [[ "${FULL}" == "1" ]]; then
  echo "=== Full segment eval (all test segments, first-window-only, resume=${PREDICTIONS}) ==="
else
  echo "=== Eval 1 segment (limit=${LIMIT}, resume if predictions exist) ==="
fi
echo "Run log: ${RUN_LOG}"

python3 -u eval_llm_segment.py \
  --splits-dir "${SPLITS_DIR}" \
  --extracted-dir "${EXTRACTED_DIR}" \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --task title \
  --output "${METRICS}" \
  --predictions "${PREDICTIONS}" \
  --detail-dir "${DETAIL_DIR}" \
  "${extra_args[@]}"

if [[ "${FULL}" == "1" ]] || [[ "${GENERATE_REPORT:-0}" == "1" ]]; then
  python3 scripts/compare_segment_eval.py \
    --current "${METRICS}" \
    --output-json docs/metrics/tilamb_title_lora_segment.json \
    --output-md logs/llm_segment_eval_report.md
  echo "Report: logs/llm_segment_eval_report.md"
fi

echo "Metrics: ${METRICS}"
echo "Predictions: ${PREDICTIONS}"
echo "Per-segment detail: ${DETAIL_DIR}/"
echo "Run again for next segment: bash scripts/run_eval_llm_segment_inner.sh"

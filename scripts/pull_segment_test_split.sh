#!/bin/bash
# Fetch test.jsonl for segment eval (Spsither split on HF - same as EXPERIMENT_REPORT eval_segment.py).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
SPLITS_DIR="${SPLITS_DIR:-${REPO}/data/roberta_full/splits}"
TEST_JSONL="${SPLITS_DIR}/test.jsonl"
VENV="${VENV:-/root/infer-venv}"

cd "${REPO}"
mkdir -p "${SPLITS_DIR}"

if [[ -f "${TEST_JSONL}" ]]; then
  echo "OK: ${TEST_JSONL} already exists ($(wc -l < "${TEST_JSONL}") window rows)"
  exit 0
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

pip install -q datasets 2>/dev/null || true
python3 scripts/pull_segment_test_split.py --output "${TEST_JSONL}"

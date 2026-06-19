#!/bin/bash
# Fetch Koichi test split manifest for segment eval (doc_id + segment_id rows only).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
SPLITS_DIR="${SPLITS_DIR:-${REPO}/data/roberta_koichi/splits}"
TEST_JSONL="${SPLITS_DIR}/test.jsonl"
FALLBACK_SPLITS="${FALLBACK_SPLITS:-${REPO}/data/roberta_full/splits}"

cd "${REPO}"
mkdir -p "${SPLITS_DIR}"

if [[ -f "${TEST_JSONL}" ]]; then
  echo "OK: ${TEST_JSONL} already exists ($(wc -l < "${TEST_JSONL}") window rows)"
  exit 0
fi

export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== Pulling Koichi test split manifest ==="

# 1) Koichi splits on instance from prior training run
if [[ -f /root/data/roberta_koichi/splits/test.jsonl ]]; then
  cp /root/data/roberta_koichi/splits/test.jsonl "${TEST_JSONL}"
  echo "Copied from /root/data/roberta_koichi/splits/test.jsonl"
  exit 0
fi

# 2) HF dataset (if koichi splits were uploaded)
if command -v hf >/dev/null 2>&1; then
  if hf download ganga4364/tibetan-metadata-detector \
      --repo-type dataset \
      --include "roberta_koichi/splits/test.jsonl" \
      --local-dir /tmp/hf_koichi_splits 2>/dev/null; then
    if [[ -f /tmp/hf_koichi_splits/roberta_koichi/splits/test.jsonl ]]; then
      cp /tmp/hf_koichi_splits/roberta_koichi/splits/test.jsonl "${TEST_JSONL}"
      echo "Downloaded Koichi test.jsonl from HF dataset"
      exit 0
    fi
  fi
fi

# 3) Fallback: roberta_full test.jsonl (different segment set than Koichi baseline)
if [[ -f "${FALLBACK_SPLITS}/test.jsonl" ]]; then
  cp "${FALLBACK_SPLITS}/test.jsonl" "${TEST_JSONL}"
  echo "WARN: Using roberta_full test.jsonl fallback - metrics won't match Koichi baseline exactly"
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  mkdir -p "${FALLBACK_SPLITS}"
  hf download ganga4364/tibetan-metadata-detector \
    --repo-type dataset \
    --include "splits/test.jsonl" \
    --local-dir /tmp/hf_roberta_splits || true
  if [[ -f /tmp/hf_roberta_splits/splits/test.jsonl ]]; then
    cp /tmp/hf_roberta_splits/splits/test.jsonl "${TEST_JSONL}"
    echo "WARN: Downloaded roberta_full test.jsonl fallback from HF"
    exit 0
  fi
fi

echo "ERROR: Could not obtain test.jsonl"
echo "  Place Koichi splits at: ${TEST_JSONL}"
echo "  Or scp from training instance: data/roberta_koichi/splits/test.jsonl"
exit 1

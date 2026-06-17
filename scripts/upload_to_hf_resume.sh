#!/bin/bash
# Resume HF upload from extracted/ (dataset card already uploaded).
set -euo pipefail

cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

DATASET_ID="ganga4364/tibetan-metadata-detector"
MODEL_ID="ganga4364/tibetan-metadata-roberta-ner"
SPACE_ID="ganga4364/tibetan-metadata-highlight"

hf auth login --token "$HF_TOKEN"

_bind_stage() {
  local stage="$1" subpath="$2" source="$3"
  umount "$stage/$subpath" 2>/dev/null || true
  rm -rf "$stage"
  mkdir -p "$stage/$subpath"
  mount --bind "$source" "$stage/$subpath"
}

_unbind_stage() {
  local stage="$1" subpath="$2"
  umount "$stage/$subpath" 2>/dev/null || true
  rm -rf "$stage"
}

echo "=== Uploading extracted/ (~11 GB) ==="
STAGE_EXTRACTED="/tmp/hf_stage_extracted"
_bind_stage "$STAGE_EXTRACTED" extracted "$(pwd)/data/extracted"
hf upload-large-folder "$DATASET_ID" "$STAGE_EXTRACTED" --type dataset
_unbind_stage "$STAGE_EXTRACTED" extracted

echo "=== Uploading splits/ (~13 GB) ==="
STAGE_SPLITS="/tmp/hf_stage_splits"
_bind_stage "$STAGE_SPLITS" splits "$(pwd)/data/roberta_full/splits"
hf upload-large-folder "$DATASET_ID" "$STAGE_SPLITS" --type dataset
_unbind_stage "$STAGE_SPLITS" splits

echo "=== Creating model repo ==="
hf repos create "$MODEL_ID" --type model --public --exist-ok

echo "=== Uploading model card ==="
hf upload "$MODEL_ID" hub/model_README.md README.md --type model \
  --commit-message "Add model card"

echo "=== Uploading model weights ==="
hf upload "$MODEL_ID" data/roberta_full/model/best --type model \
  --commit-message "RoBERTa token classifier for Tibetan title/author NER"

hf upload "$MODEL_ID" data/roberta_full/model/test_metrics.json --type model \
  --commit-message "Add full test split metrics"

echo "=== Creating Space repo ==="
hf repos create "$SPACE_ID" --type space --space-sdk gradio \
  --flavor cpu-basic --public --exist-ok

echo "=== Uploading Space app ==="
hf upload "$SPACE_ID" space/ --type space \
  --commit-message "Gradio demo: sliding-window title/author highlight"

echo "=== Done ==="

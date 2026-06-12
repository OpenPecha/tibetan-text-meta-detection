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

echo "=== Uploading extracted/ (~11 GB) ==="
STAGE_EXTRACTED="/tmp/hf_stage_extracted"
rm -rf "$STAGE_EXTRACTED"
mkdir -p "$STAGE_EXTRACTED"
ln -sfn "$(pwd)/data/extracted" "$STAGE_EXTRACTED/extracted"
hf upload-large-folder "$DATASET_ID" "$STAGE_EXTRACTED" --type dataset

echo "=== Uploading splits/ (~13 GB) ==="
STAGE_SPLITS="/tmp/hf_stage_splits"
rm -rf "$STAGE_SPLITS"
mkdir -p "$STAGE_SPLITS"
ln -sfn "$(pwd)/data/roberta_full/splits" "$STAGE_SPLITS/splits"
hf upload-large-folder "$DATASET_ID" "$STAGE_SPLITS" --type dataset

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

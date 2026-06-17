#!/bin/bash
# Upload dataset folders only (bind-mount staging for upload-large-folder).
set -euo pipefail

cd /root/tibetan-metadata-detector
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

DATASET_ID="ganga4364/tibetan-metadata-detector"

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
find "$STAGE_EXTRACTED" -type f | head -3
hf upload-large-folder "$DATASET_ID" "$STAGE_EXTRACTED" --type dataset
_unbind_stage "$STAGE_EXTRACTED" extracted

echo "=== Uploading splits/ (~13 GB) ==="
STAGE_SPLITS="/tmp/hf_stage_splits"
_bind_stage "$STAGE_SPLITS" splits "$(pwd)/data/roberta_full/splits"
find "$STAGE_SPLITS" -type f | head -3
hf upload-large-folder "$DATASET_ID" "$STAGE_SPLITS" --type dataset
_unbind_stage "$STAGE_SPLITS" splits

echo "=== Dataset upload complete ==="

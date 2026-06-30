#!/bin/bash
set -euo pipefail
source /root/llama-venv/bin/activate
cd /root/LLaMA-Factory
export HF_TOKEN="$(cat /root/.hf_token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
REPO=/root/tibetan-text-meta-detection
PILOT_DATA="${PILOT_DATA:-${REPO}/data/llm_sft_pilot_10pct}"
LOG_DIR="${REPO}/data/llm_sft/logs"
mkdir -p "${LOG_DIR}"

# Merge pilot registry into LLaMA-Factory dataset_info.json
python3 <<PY
import json
from pathlib import Path
repo = Path("${REPO}")
data_dir = Path("/root/LLaMA-Factory/data")
info_path = data_dir / "dataset_info.json"
pilot = json.loads((repo / "configs/llama_factory/dataset_info_pilot.json").read_text())
base = json.loads(info_path.read_text()) if info_path.exists() else {}
base.update(pilot)
info_path.write_text(json.dumps(base, indent=2))
PY

ln -sfn "${PILOT_DATA}/title" /root/LLaMA-Factory/data/tibetan_title_sft_pilot
ln -sfn "${PILOT_DATA}/author" /root/LLaMA-Factory/data/tibetan_author_sft_pilot

if [[ "${SKIP_TITLE:-0}" == "1" ]]; then
  echo "SKIP_TITLE=1 — skipping title pilot training" | tee "${LOG_DIR}/title_pilot_train.log"
else
  echo "=== Pilot Title LoRA started at $(date) ===" | tee "${LOG_DIR}/title_pilot_train.log"
  llamafactory-cli train "${REPO}/configs/llama_factory/title_lora_sft_pilot.yaml" 2>&1 | tee -a "${LOG_DIR}/title_pilot_train.log"
  echo "=== Pilot Title LoRA finished at $(date) ===" | tee -a "${LOG_DIR}/title_pilot_train.log"
fi

if [[ "${SKIP_AUTHOR:-0}" == "1" ]]; then
  echo "SKIP_AUTHOR=1 — skipping author pilot training" | tee -a "${LOG_DIR}/author_pilot_train.log"
  exit 0
fi

echo "=== Pilot Author LoRA started at $(date) ===" | tee "${LOG_DIR}/author_pilot_train.log"
llamafactory-cli train "${REPO}/configs/llama_factory/author_lora_sft_pilot.yaml" 2>&1 | tee -a "${LOG_DIR}/author_pilot_train.log"
echo "=== Pilot Author LoRA finished at $(date) ===" | tee -a "${LOG_DIR}/author_pilot_train.log"

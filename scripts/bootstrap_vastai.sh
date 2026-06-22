#!/bin/bash
# One-shot bootstrap for VastAI benchmark instance (default: vastai-benchmark port 10671).
# Sets up repo, CUDA venv, pilot SFT data, Koichi NER, and TiLamb pilot LoRA.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/infer-venv}"
GIT_URL="${GIT_URL:-https://github.com/OpenPecha/tibetan-text-meta-detection.git}"
PILOT_DIR="${PILOT_DIR:-data/llm_sft_pilot_10pct}"
KOICHI_DIR="${KOICHI_DIR:-models/koichi-ner}"
LORA_DIR="${LORA_DIR:-/root/lora/tibetan-title-pilot}"
LOG="${LOG:-/tmp/bootstrap-vastai.log}"

exec > >(tee -a "${LOG}") 2>&1

echo "=== Bootstrap vastai benchmark started at $(date) ==="
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
echo "=== Disk ==="
df -h / | tail -1

apt-get update -qq
apt-get install -y -qq git tmux python3.11 python3.11-venv python3.11-dev build-essential

if [[ ! -d "${REPO}/.git" ]]; then
  git clone --depth 1 "${GIT_URL}" "${REPO}"
fi
ln -sfn "${REPO}" /root/tibetan-metadata-detector
cd "${REPO}"
git pull origin main || true
chmod +x scripts/*.sh 2>/dev/null || true

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3.11 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -U pip setuptools wheel packaging
pip install huggingface_hub
if [[ -f /root/.hf_token ]]; then
  bash scripts/setup_hf_token.sh
else
  echo "WARN: /root/.hf_token missing — set HF token before pulling gated models"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if echo "${GPU_NAME}" | grep -qiE '5090|5080|5070|Blackwell'; then
  echo "=== Installing PyTorch cu128 for ${GPU_NAME} ==="
  pip uninstall -y torch torchvision torchaudio torchao 2>/dev/null || true
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
  echo "=== Installing PyTorch cu124 for ${GPU_NAME} ==="
  pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
fi

pip install transformers accelerate peft bitsandbytes sentencepiece datasets python-dotenv

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [[ -f /root/.hf_token ]]; then
  if [[ ! -f "${PILOT_DIR}/title/test.jsonl" ]]; then
    echo "=== Downloading pilot SFT dataset ==="
    hf download ganga4364/tibetan-metadata-llm-sft \
      --repo-type dataset --local-dir "${PILOT_DIR}"
  fi
  if [[ ! -f "${KOICHI_DIR}/config.json" ]]; then
    echo "=== Downloading Koichi NER checkpoint ==="
    mkdir -p "${KOICHI_DIR}"
    hf download ganga4364/tibetan-metadata-koichi-ner --local-dir "${KOICHI_DIR}"
  fi
  mkdir -p "$(dirname "${LORA_DIR}")"
  if [[ ! -f "${LORA_DIR}/adapter_model.safetensors" ]]; then
    echo "=== Downloading TiLamb pilot title LoRA ==="
    hf download ganga4364/tibetan-metadata-title-tilamb-lora-pilot \
      --local-dir "${LORA_DIR}"
  fi
fi

mkdir -p logs docs/metrics

echo "=== Bootstrap complete at $(date) ==="
echo "  Repo: ${REPO}"
echo "  Venv: ${VENV}"
echo "  Pilot data: ${PILOT_DIR}"
echo "  Koichi: ${KOICHI_DIR}"
echo "  LoRA: ${LORA_DIR}"
echo ""
echo "Smoke test (5 rows, Koichi):"
echo "  source ${VENV}/bin/activate"
echo "  cd ${REPO}"
echo "  python eval_benchmark_rows.py --model-kind koichi --limit 5 --resume"
echo ""
echo "Full benchmark suite:"
echo "  bash scripts/run_benchmark_suite.sh"
wc -l "${PILOT_DIR}"/title/test.jsonl 2>/dev/null || true

#!/bin/bash
# One-shot bootstrap for VastAI instance (vastai7): repo + Unsloth + data.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/unsloth-venv}"
GIT_URL="${GIT_URL:-https://github.com/OpenPecha/tibetan-text-meta-detection.git}"
LOG="${LOG:-/tmp/bootstrap-vastai7.log}"

exec > >(tee -a "${LOG}") 2>&1

echo "=== Bootstrap vastai7 started at $(date) ==="
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
  echo "WARN: /root/.hf_token missing — set HF token before pulling private/gated data"
fi

# RTX 5090 (sm_120) needs cu128; older GPUs use cu124.
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

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets sentencepiece transformers accelerate peft

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [[ -f /root/.hf_token ]]; then
  if [[ ! -f data/extracted/index.jsonl ]]; then
    bash scripts/run_pull_extracted.sh
  fi
  if [[ ! -f data/llm_sft/title/train.jsonl ]]; then
    mkdir -p data/llm_sft
    hf download ganga4364/tibetan-metadata-llm-sft --local-dir data/llm_sft --repo-type dataset
  fi
  if [[ -f scripts/build_llm_sft_pilot.sh ]] && [[ ! -f data/llm_sft_pilot_10pct/title/train.jsonl ]]; then
    bash scripts/build_llm_sft_pilot.sh
  fi
  mkdir -p /root/lora
  if [[ ! -f /root/lora/tibetan-title-pilot/adapter_model.safetensors ]]; then
    hf download ganga4364/tibetan-metadata-title-tilamb-lora-pilot \
      --local-dir /root/lora/tibetan-title-pilot
  fi
fi

echo "=== Bootstrap complete at $(date) ==="
echo "  Repo: ${REPO}"
echo "  Venv: ${VENV}"
echo "  LoRA: /root/lora/tibetan-title-pilot"
echo ""
echo "Start Unsloth Studio:"
echo "  source ${VENV}/bin/activate && unsloth studio -H 0.0.0.0 -p 8888"
echo "  Local: ssh vastai7   then open http://127.0.0.1:8888"
wc -l data/extracted/index.jsonl data/llm_sft/title/train.jsonl data/llm_sft_pilot_10pct/title/train.jsonl 2>/dev/null || true

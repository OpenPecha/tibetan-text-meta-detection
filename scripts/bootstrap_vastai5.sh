#!/bin/bash
# One-shot bootstrap for VastAI GPU instance (vastai5 / generic).
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/unsloth-venv}"
GIT_URL="${GIT_URL:-https://github.com/OpenPecha/tibetan-text-meta-detection.git}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
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
fi
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets huggingface_hub sentencepiece transformers accelerate

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

echo "=== Bootstrap complete ==="
echo "  Repo: ${REPO}"
echo "  Venv: ${VENV}"
wc -l data/extracted/index.jsonl data/llm_sft/title/train.jsonl data/llm_sft_pilot_10pct/title/train.jsonl 2>/dev/null || true

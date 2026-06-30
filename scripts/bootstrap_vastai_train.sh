#!/bin/bash
# One-shot bootstrap for a VastAI *training* instance.
# Sets up the repo, a CUDA training venv (llama-venv) with LLaMA-Factory, and the
# 10% pilot SFT dataset pulled straight from Hugging Face (no local data build).
#
# Unlike scripts/setup_llama_factory.sh, this does NOT require a pre-built
# data/llm_sft tree — it downloads ganga4364/tibetan-metadata-llm-sft instead.
set -euo pipefail

REPO="${REPO:-/root/tibetan-text-meta-detection}"
VENV="${VENV:-/root/llama-venv}"
LLAMA_DIR="${LLAMA_DIR:-/root/LLaMA-Factory}"
GIT_URL="${GIT_URL:-https://github.com/OpenPecha/tibetan-text-meta-detection.git}"
PILOT_DIR="${PILOT_DIR:-data/llm_sft_pilot_10pct}"
DATASET_REPO="${DATASET_REPO:-ganga4364/tibetan-metadata-llm-sft}"
LOG="${LOG:-/tmp/bootstrap-vastai-train.log}"

exec > >(tee -a "${LOG}") 2>&1

echo "=== Bootstrap vastai TRAIN started at $(date) ==="
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
pip install -q huggingface_hub
if [[ -f /root/.hf_token ]]; then
  bash scripts/setup_hf_token.sh
else
  echo "WARN: /root/.hf_token missing — set HF token before training (gated base model)"
fi

# Torch first so LLaMA-Factory's editable install does not pull a CPU wheel.
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_DIR}"
fi
pushd "${LLAMA_DIR}" >/dev/null
pip install -q -e ".[torch,metrics]"
pip install -q 'transformers>=4.55,<=4.56.2' 'huggingface-hub>=0.34,<1.0'
pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
popd >/dev/null

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

if [[ -f /root/.hf_token ]]; then
  if [[ ! -f "${PILOT_DIR}/author/train.jsonl" ]]; then
    echo "=== Downloading pilot SFT dataset (${DATASET_REPO}) ==="
    hf download "${DATASET_REPO}" --repo-type dataset --local-dir "${PILOT_DIR}"
  fi
fi

echo "=== Bootstrap TRAIN complete at $(date) ==="
echo "  Repo: ${REPO}"
echo "  Train venv: ${VENV}"
echo "  LLaMA-Factory: ${LLAMA_DIR}"
echo "  Pilot data: ${PILOT_DIR}"
echo ""
echo "Train author LoRA only (skip title) in tmux:"
echo "  tmux new -s author_train 'cd ${REPO} && SKIP_TITLE=1 bash scripts/run_llm_sft_pilot_train.sh'"
wc -l "${PILOT_DIR}"/author/train.jsonl 2>/dev/null || true
wc -l "${PILOT_DIR}"/author/test.jsonl 2>/dev/null || true

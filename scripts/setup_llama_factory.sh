#!/bin/bash
# Clone LLaMA-Factory, install deps, wire Tibetan SFT JSONL datasets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/root/LLaMA-Factory}"
SFT_DATA="${SFT_DATA:-${REPO_ROOT}/data/llm_sft}"
VENV="${VENV:-/root/llama-venv}"

if [[ ! -f "${SFT_DATA}/title/train.jsonl" ]]; then
  echo "Missing ${SFT_DATA}/title/train.jsonl — run: bash scripts/build_llm_sft.sh"
  exit 1
fi

if [[ -f /root/.hf_token ]]; then
  bash "${REPO_ROOT}/scripts/setup_hf_token.sh"
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3.11 -m venv "${VENV}"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install -U pip setuptools wheel packaging
  pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
else
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_DIR}"
fi

cd "${LLAMA_DIR}"
pip install -q -e ".[torch,metrics]"
pip install -q 'transformers>=4.55,<=4.56.2' 'huggingface-hub>=0.34,<1.0'
pip install -q torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

mkdir -p data
cp "${REPO_ROOT}/configs/llama_factory/dataset_info.json" data/dataset_info.json
ln -sfn "${SFT_DATA}/title" data/tibetan_title_sft
ln -sfn "${SFT_DATA}/author" data/tibetan_author_sft

echo "LLaMA-Factory ready at ${LLAMA_DIR} (venv: ${VENV})"
echo "  title train: $(wc -l < "${SFT_DATA}/title/train.jsonl") rows"
echo "  author train: $(wc -l < "${SFT_DATA}/author/train.jsonl") rows"

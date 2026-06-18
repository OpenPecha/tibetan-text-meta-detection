#!/bin/bash
# Clone LLaMA-Factory, install deps, wire Tibetan SFT JSONL datasets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/root/LLaMA-Factory}"
SFT_DATA="${SFT_DATA:-${REPO_ROOT}/data/llm_sft}"

if [[ ! -f "${SFT_DATA}/title/train.jsonl" ]]; then
  echo "Missing ${SFT_DATA}/title/train.jsonl — run: bash scripts/build_llm_sft.sh"
  exit 1
fi

if [[ -f /root/.hf_token ]]; then
  bash "${REPO_ROOT}/scripts/setup_hf_token.sh"
fi

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_DIR}"
fi

cd "${LLAMA_DIR}"
pip install -q -e ".[torch,metrics]" 2>/dev/null || pip install -q -e .

mkdir -p data
cp "${REPO_ROOT}/configs/llama_factory/dataset_info.json" data/dataset_info.json
ln -sfn "${SFT_DATA}/title" data/tibetan_title_sft
ln -sfn "${SFT_DATA}/author" data/tibetan_author_sft

echo "LLaMA-Factory ready at ${LLAMA_DIR}"
echo "  title train: $(wc -l < "${SFT_DATA}/title/train.jsonl") rows"
echo "  author train: $(wc -l < "${SFT_DATA}/author/train.jsonl") rows"

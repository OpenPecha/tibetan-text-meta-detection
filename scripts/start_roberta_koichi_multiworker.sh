#!/bin/bash
# Window extracted docs with KoichiYasuoka/roberta-base-tibetan tokenizer (comparison run).
set -euo pipefail

export ROBERTA_MODEL="${ROBERTA_MODEL:-KoichiYasuoka/roberta-base-tibetan}"
export PROCESSED_DIR="${PROCESSED_DIR:-data/roberta_koichi}"
export LOG_DIR="${LOG_DIR:-data/roberta_koichi_process_logs}"

exec bash scripts/start_roberta_process_multiworker.sh

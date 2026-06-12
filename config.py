"""Configuration for tibetan-metadata-detector data pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
TEXTS_DIR = EXTRACTED_DIR / "texts"
ANNOTATIONS_DIR = EXTRACTED_DIR / "annotations"
INDEX_PATH = EXTRACTED_DIR / "index.jsonl"
STATS_PATH = EXTRACTED_DIR / "stats.json"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_WINDOWS_DIR = PROCESSED_DIR / "windows"
PROCESSED_REPORTS_DIR = PROCESSED_DIR / "reports"
SPLITS_DIR = PROCESSED_DIR / "splits"

DEFAULT_WINDOW_SIZE = 200
WINDOW_SIZE_CANDIDATES = [100, 200, 300, 500]

# RoBERTa subword sliding-window pipeline
ROBERTA_MODEL = "spsither/tibetan_RoBERTa_S_e3"
ROBERTA_WINDOW_SIZE = 512
ROBERTA_STRIDE = 256
ROBERTA_MAX_BEGIN_SLIDES = 15
ROBERTA_MAX_END_SLIDES = 15
ROBERTA_MEDIUM_TOKEN_THRESHOLD = 8192
TRAIN_RATIO = 0.89
VAL_RATIO = 0.01
TEST_RATIO = 0.10
RANDOM_SEED = 42

# Window balancing (after roberta-process, before split)
O_ONLY_CAP_RATIO = 2.0
AUTHOR_OVERSAMPLE = 2
BALANCED_EXAMPLES_FILENAME = "roberta_balanced_examples.jsonl"

# Legacy single-file path (deprecated)
EXTRACTED_PATH = DATA_DIR / "extracted_annotations.jsonl"

DB_CONFIG = {
    "host": os.environ.get("BENCHMARK_DB_HOST", ""),
    "user": os.environ.get("BENCHMARK_DB_USER", ""),
    "password": os.environ.get("BENCHMARK_DB_PASSWORD", ""),
    "port": int(os.environ.get("BENCHMARK_DB_PORT", "5432")),
    "database": os.environ.get("BENCHMARK_DB_NAME", "postgres"),
}

# Tibetan Metadata Detector

Extract training data for detecting **title** and **author** spans in digitized Tibetan text from the outliner PostgreSQL database.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # fill in BENCHMARK_DB_PASSWORD
```

## Explore database

```bash
python explore_db.py
```

## Extract data

```bash
python extract_data.py --dry-run
python extract_data.py --all
python extract_data.py --num-samples 10
python extract_data.py --output-dir data/my_run
```

Output layout under `data/extracted/` (or `--output-dir`):

```
data/extracted/
  index.jsonl              # lightweight manifest (one line per document)
  stats.json               # run summary
  texts/{doc_id}.txt       # full document text
  annotations/{doc_id}.json # title/author spans for that document
```

## Epic 1: Prepare training data

After extraction, run the full data pipeline (analyze, window, BIO tags, splits):

```bash
# Full pipeline: extract from DB + analyze + window report + BIO + split
python prepare_data.py all

# Or step by step
python prepare_data.py extract --num-samples 10
python prepare_data.py analyze
python prepare_data.py window-report
python prepare_data.py process --window-size 200
python prepare_data.py split
```

Processed output:

```
data/processed/
  reports/
    export_stats.json       # label/span/text length distribution
    window_coverage.json    # document begin/end capture rates (segment training used by default)
  windows/{doc_id}_{segment_id}.json
  all_examples.jsonl        # BIO-tagged segment examples (one per annotated segment)
  process_summary.json
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
    split_stats.json
```

Training examples are built **per annotated segment** (title/author spans are segment-local). Document begin/end windows are optional via `--include-doc-windows`.

## RoBERTa sliding-window pipeline

```bash
# Single worker
python prepare_data.py roberta-process --processed-dir data/roberta_full

# Multi-worker (recommended for ~3.8k docs)
NUM_WORKERS=6 bash scripts/start_roberta_process_multiworker.sh
# after all workers finish:
python prepare_data.py merge-roberta-shards --processed-dir data/roberta_full --num-workers 6

# Or run merge + balance + split + HF push in one waiter:
NUM_WORKERS=6 bash scripts/wait_roberta_merge_balance_push.sh

# Balance O-only windows + oversample author windows
python prepare_data.py balance-windows --processed-dir data/roberta_full

# Stratified split (default 89% / 1% / 10%)
python prepare_data.py split --processed-dir data/roberta_full

# Push splits to Hugging Face (Parquet)
bash scripts/push_balanced_splits.sh
```

## Train from Hugging Face (recommended on a new GPU instance)

No local `data/` required — load balanced splits directly:

```bash
pip install -r requirements.txt
python train_roberta.py \
  --hf-dataset ganga4364/tibetan-metadata-detector \
  --hf-config windows \
  --output-dir data/roberta_full/model \
  --batch-size 64 \
  --epochs 3 \
  --entity-weight 10
```

Or train from local JSONL splits under `data/roberta_full/splits/`.

## Tests

```bash
pytest tests/test_label_window.py -v -m "not slow"
```

## Scope

- **In scope:** title, author (from `title_span_*` / `author_span_*` on `outliner_segments`)
- **Out of scope:** translator

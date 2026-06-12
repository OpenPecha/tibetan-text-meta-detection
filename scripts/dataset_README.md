---
license: cc-by-4.0
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- ner
- metadata
- bdrc
- roberta
size_categories:
- 1M<n<10M
---

# Tibetan Metadata Detector Dataset

Training data for Tibetan **title** and **author** span detection from BDRC outliner exports.

## Contents

| Split | Rows | Description |
|-------|------|-------------|
| `train` | 819,620 | RoBERTa sliding-window training examples (Parquet) |
| `validation` | 92,907 | Validation windows |
| `test` | 108,954 | Test windows |

| `documents` | 3,665 | Raw per-document text + annotation JSON (Parquet) |

## Windowing (train = infer)

- Tokenizer: [`spsither/tibetan_RoBERTa_S_e3`](https://huggingface.co/spsither/tibetan_RoBERTa_S_e3)
- Window size: **512** subword tokens, stride **256**
- Short segments (≤512 tok): 1 window
- Long segments: up to **15 begin + 15 end** slides with overlap-aware deduplication

## Split sizes (windows)

| Split | Examples |
|-------|----------|
| train | 819,620 |
| val | 92,907 |
| test | 108,954 |

## JSONL fields

Each row in the `windows` config:

- `input_ids`, `attention_mask`, `labels` — HuggingFace-ready tensors (512 len)
- `offset_mapping` — char offsets per token (segment-relative)
- `window_name`, `window_side`, `slide_index` — window metadata
- `doc_id`, `segment_id`, `segment_tier`, `has_title`, `has_author`

## Model

Fine-tuned classifier: [ganga4364/tibetan-metadata-roberta-ner](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner)

## Demo

Interactive highlight demo: [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight)

## Usage

```python
from datasets import load_dataset

# RoBERTa window splits (default config)
windows = load_dataset("ganga4364/tibetan-metadata-detector")

# Raw extracted documents
docs = load_dataset("ganga4364/tibetan-metadata-detector", "documents")
```

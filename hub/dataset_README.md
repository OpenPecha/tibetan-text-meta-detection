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
- 100K<n<1M
---

# Tibetan Metadata Detector Dataset

RoBERTa sliding-window training data for Tibetan **title** and **author** span detection from BDRC outliner exports.

## Contents

Balanced window splits (fixed window-relative BIO labeling, O-only subsampling, author oversampling):

| Split | Windows | Documents (approx.) |
|-------|---------|---------------------|
| `train` | 241,377 | 3,355 |
| `validation` | 2,688 | 40 |
| `test` | 30,357 | 375 |

Load with the **`default`** config (Parquet upload):

```python
from datasets import load_dataset

ds = load_dataset("ganga4364/tibetan-metadata-detector")
train = ds["train"]
val = ds["validation"]
test = ds["test"]
```

**Balancing applied before split:**
- O-only windows capped at **2×** entity-bearing windows per segment
- Author-bearing windows duplicated **2×**
- Document-level stratified split **89% / 1% / 10%**
- Raw input: 1,061,770 windows → 274,422 after balance

Raw extracted documents: [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted)

## Windowing (train = infer)

- Tokenizer: [`spsither/tibetan_RoBERTa_S_e3`](https://huggingface.co/spsither/tibetan_RoBERTa_S_e3)
- Window size: **512** subword tokens, stride **256**
- Short segments (≤512 tok): 1 window
- Long segments: up to **15 begin + 15 end** slides with overlap-aware deduplication

## Fields

Each row:

- `input_ids`, `attention_mask`, `labels` — HuggingFace-ready tensors (512 len)
- `offset_mapping` — char offsets per token (**segment-relative**)
- `window_name`, `window_side`, `slide_index`, `window_annotations`
- `doc_id`, `segment_id`, `segment_tier`, `has_title`, `has_author`

## Train on a new GPU instance

```bash
pip install -r requirements.txt
python train_roberta.py \
  --hf-dataset ganga4364/tibetan-metadata-detector \
  --hf-config default \
  --batch-size 64 \
  --epochs 3 \
  --entity-weight 10
```

See [GPU instance playbook](https://github.com/OpenPecha/tibetan-text-meta-detection/blob/main/docs/GPU_INSTANCE.md) on GitHub.

## Model & demo

- Model: [ganga4364/tibetan-metadata-roberta-ner](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner)
- Demo: [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight)

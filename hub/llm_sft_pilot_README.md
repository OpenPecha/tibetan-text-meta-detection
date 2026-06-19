---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - bo
tags:
  - tibetan
  - metadata
  - title
  - author
  - llm-sft
  - tilamb
  - pilot
size_categories:
  - 10K<n<100K
---

# Tibetan metadata LLM SFT dataset (10% pilot)

**10% stratified random subsample** of the full TiLamb SFT JSONL, for pilot LoRA training and smoke tests.

Supervised fine-tuning data for **title** and **author** span extraction from BDRC outliner segments, built for [TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B).

## Full dataset

See [ganga4364/tibetan-metadata-llm-sft-full](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft-full) for the complete train/val/test JSONL.

## Layout

```
title/{train,val,test}.jsonl       # Alpaca format for LLaMA-Factory
title/{train,val,test}_meta.jsonl
author/{train,val,test}.jsonl
author/{train,val,test}_meta.jsonl
dataset_info.json                  # LLaMA-Factory registry snippet
reports/crop_stats.json            # if present
```

Each training row:

- `instruction` — fixed task prompt (title or author)
- `input` — cropped segment text (token-budget ≤3584 via TiLamb tokenizer)
- `output` — JSON `{"spans":[{"text","start","end"}]}` (crop-relative offsets)

## Subsampling

- Source: full `data/llm_sft` built from extracted annotations
- Method: 10% random row sample per split file (seed 42)
- Script: `scripts/subsample_llm_sft.py`

## Cropping

| Kind | When |
|------|------|
| `full` | Whole segment fits token budget |
| `positive` | Random window containing gold span (anti position-bias) |
| `negative` | Random window, empty spans for that task |

Source: [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted).

Built with [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection) `llm_sft` package.

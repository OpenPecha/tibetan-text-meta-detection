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
size_categories:
  - 100K<n<1M
---

# Tibetan metadata LLM SFT dataset (full)

Complete supervised fine-tuning JSONL for **title** and **author** span extraction from BDRC outliner segments, built for [TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B).

## Pilot subset

See [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) for a **10% pilot** subsample (same schema, smaller for quick experiments).

## Layout

```
title/{train,val,test}.jsonl       # Alpaca format for LLaMA-Factory
title/{train,val,test}_meta.jsonl
author/{train,val,test}.jsonl
author/{train,val,test}_meta.jsonl
dataset_info.json                  # LLaMA-Factory registry snippet
reports/crop_stats.json
```

Each training row:

- `instruction` — fixed task prompt (title or author)
- `input` — cropped segment text (token-budget ≤3584 via TiLamb tokenizer)
- `output` — JSON `{"spans":[{"text","start","end"}]}` (crop-relative offsets)

## Cropping

| Kind | When |
|------|------|
| `full` | Whole segment fits token budget |
| `positive` | Random window containing gold span (anti position-bias) |
| `negative` | Random window, empty spans for that task |

Tokenizer budgets use **TiLamb** (`YoLo2000/TiLamb-7B`) — enforced in tokens, not characters.

Source: [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted) (3794 docs).

Built with [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection) `llm_sft` package.

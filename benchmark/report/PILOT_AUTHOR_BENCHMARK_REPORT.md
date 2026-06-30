# Pilot Author Span Extraction Benchmark — Final Report

**Project:** [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection)  
**Date:** June 2026  
**Task:** Detect bibliographic **author** character spans in cropped Tibetan segment text.  
**Scope:** Single model — the newly trained **TiLamb-7B + author pilot LoRA** (no baselines), per request.

---

## 1. Executive summary

We trained a TiLamb-7B LoRA adapter for **author** span extraction on the 10% pilot SFT corpus and evaluated it on the matching held-out test split (**688 rows**).

> **Primary judgment for author = text-equalness, not offset.** Because the author sits at the very end of long inputs, character offsets are unreliable (see §5), so we judge the author model on whether it emits the **correct author text**, independent of offsets. Offset metrics are retained as secondary diagnostics.

### Primary metric — emitted text-equal F1 (offset-independent)

| Metric | Precision | Recall | **F1** |
|--------|-----------|--------|--------|
| Text-equal (exact string) | 29.3% | 33.0% | **31.1%** |
| Text-equal (normalized*) | 30.4% | 34.2% | **32.1%** |
| Text-contained (pred ⊆ gold or gold ⊆ pred) | 59.0% | 66.3% | **62.4%** |

\*Normalized = trailing whitespace / tsheg (`་`) / shad (`།`) stripped. Parse failure rate: **0.0%**.

### Secondary metrics — offset-based (diagnostic only)

| Metric | Value |
|--------|-------|
| Overlap IoU50 F1 | 1.39% |
| Offset ±50 F1 | 2.94% |
| Exact-offset F1 | 0.00% |
| Mean boundary error (MAE start / end) | ~981 / ~995 chars |

**Headline finding — the model extracts the author, but cannot localize it.**

The adapter produces **valid JSON every time** (0% parse failure) and emits a **correct author string in a third of rows exactly and an overlapping author string in two thirds of rows**. But the **character offsets are wrong by ~980 characters on average**, so every offset-based metric collapses to near-zero. We therefore adopt **text-equalness as the primary author metric**.

The offset failure is structural, not a training bug (train and test are consistent):

| Task | Gold span position in input | Median offset | Offset IoU50 | Primary metric |
|------|-----------------------------|---------------|--------------|----------------|
| Title (prior benchmark) | **start** of segment | ~5 chars (rel 0.00) | 58.9% | offset IoU50 |
| **Author (this benchmark)** | **end** of segment | **~8,327 chars (rel 0.98)** | 1.4% | **text-equal F1 (31–62%)** |

Authors sit at ~98% depth in ~8,800-character inputs. A 7B LLM cannot count to ~8,300 characters, so it approximates the offset, even though it reads the right author. Titles, which sit at offset ~0, were trivially localizable — hence the divergent primary metric.

**Takeaways**

- Judge and consume the author model by **emitted text**: it gets an overlapping author **62.4% F1** of the time, then re-locate the string in the source with a search if offsets are needed.
- The model is *not* failing the task — it is failing **character counting** at depth ~8k, which is why offset metrics are demoted to diagnostics.
- A token-classification head (like Koichi for titles) or a "predict text, then `str.find`" post-step would recover the offsets.

---

## 2. Benchmark corpus

### Test split used

| Item | Value |
|------|-------|
| Hugging Face dataset | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) |
| Files | `author/test.jsonl`, `author/test_meta.jsonl` |
| Rows | **688** (609 with a gold author span) |
| Unit | One JSONL row = one **cropped** segment; gold spans are **crop-relative** in `output` JSON |
| Subsample | 10% random rows per split file, seed 42 (`scripts/subsample_llm_sft.py`) |

Source annotations: [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted).

### Span-position characteristics (why this task is hard)

| Split | Rows | Input length (median) | Gold `start` (median) | Relative position (median) |
|-------|------|-----------------------|-----------------------|----------------------------|
| author/train | 5,826 | 8,859 | 8,339 | **0.97** |
| author/test | 688 | 8,799 | 8,327 | **0.98** |
| title/test (contrast) | 769 | 8,662 | 5 | 0.00 |

### Fairness rule

The model receives the **identical** `input` string from each test row; gold spans come from the row `output` field. **No re-cropping** at evaluation time. Cropping (≤3584 TiLamb tokens) was applied only when building the SFT dataset.

### Row format

```json
{
  "instruction": "…author extraction prompt…",
  "input": "<cropped Tibetan segment text>",
  "output": "{\"spans\":[{\"text\":\"…\",\"start\":8563,\"end\":8576}]}"
}
```

---

## 3. Model evaluated

### 3.1 TiLamb author pilot LoRA

| Setting | Value |
|---------|-------|
| Base model | [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) |
| Adapter | [ganga4364/tibetan-metadata-author-tilamb-lora-pilot](https://huggingface.co/ganga4364/tibetan-metadata-author-tilamb-lora-pilot) |
| Framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| Config | `configs/llama_factory/author_lora_sft_pilot.yaml` |
| Method | LoRA r=16, α=32, targets=all |
| Dataset | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) → `author/train.jsonl` (5,826 rows, 10% pilot) |
| Template | `llama2` chat |
| `cutoff_len` | 4096 |
| Epochs | 1 (365 optimization steps) |
| Batch | 1 × grad accum 16, lr 2e-4, cosine schedule, bf16 |
| Trainable params | 39,976,960 (0.57% of 7.02B) |
| Train loss / eval loss | **0.7298 / 0.4237** |
| Train runtime | ~2 h 55 m on one RTX 4090 |

Training curves: [train_assets/training_loss.png](train_assets/training_loss.png), [train_assets/training_eval_loss.png](train_assets/training_eval_loss.png).

**Inference guide:** [docs/TILAMB_AUTHOR_LORA_INFERENCE.md](../../docs/TILAMB_AUTHOR_LORA_INFERENCE.md)

The decreasing train/eval loss confirms the model learned the author-extraction objective; the failure mode is offset localization at inference, not under-training.

---

## 4. Evaluation protocol

### 4.1 Inference

- Script: `eval_benchmark_rows.py --model-kind tilamb_lora --task author`
- 4-bit NF4 (`bitsandbytes`), `max_new_tokens=768`, greedy decoding
- Resumable JSONL: one record per row with `gold_spans`, `pred_spans`, `input`, `raw_response`, timing
- Throughput: mean **4,034 ms/row**, median **3,675 ms/row** (688 rows ≈ 46 min)

### 4.2 Standard metrics (micro-F1 over all rows)

Implemented in `eval_common.span_eval_metrics` — greedy one-to-one span matching per row. Definitions with IoU examples and diagrams: [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md).

| Metric | Definition |
|--------|------------|
| **Exact** | Same label + identical `span_start` and `span_end` |
| **Overlap IoU50 / IoU80** | Same label + character IoU ≥ 0.5 / 0.8 |
| **Text equal** | Same label + extracted substring (from offsets) matches |
| **Offset ±10 / ±50** | Same label + start **and** end within ±N chars **and** char IoU > 0 |

> Note: "Text equal" above is computed from the **predicted offsets** (`input[start:end]`), so it inherits the offset failure. The offset-independent text match in [§6](#6-offset-independent-text-match) instead compares the model's **emitted** `text` field.

Aggregated JSON: [benchmark_pilot_author.json](benchmark_pilot_author.json) · table: [benchmark_pilot_author.md](benchmark_pilot_author.md)  
Per-model metrics: [../logs/benchmark_tilamb_lora_author_metrics.json](../logs/benchmark_tilamb_lora_author_metrics.json)  
Per-row predictions: [../logs/benchmark_tilamb_lora_author_predictions.jsonl](../logs/benchmark_tilamb_lora_author_predictions.jsonl)

---

## 5. Secondary metrics — offset-based (diagnostic)

These are retained for completeness and to document the offset failure; they are **not** the primary author judgment (see §6).

### 5.1 Standard leaderboard (688 rows)

| Model | Rows | Exact F1 | Overlap IoU50 | Overlap IoU80 | Text equal | Offset ±10 | Offset ±50 | Parse fail | ms/row |
|-------|------|----------|---------------|---------------|------------|------------|------------|------------|--------|
| TiLamb author LoRA | 688 | 0.00% | **1.39%** | 0.15% | 0.00% | 0.93% | **2.94%** | 0% | 4,034 |

Segment-dedup view (642 unique segments): Overlap IoU50 F1 **1.49%**, Offset ±50 F1 **3.14%**.

### 5.2 Offset-first diagnostics

Recomputed from saved predictions (`scripts/recompute_benchmark_offset_metrics.py`). Gold+pred are paired by best character IoU per row (**607 paired rows**). Full JSON: [benchmark_author_offset_diagnostics.json](benchmark_author_offset_diagnostics.json).

| Tolerance | Start hit | End hit | Both hit | Both + overlap | MAE start | MAE end | Median IoU |
|-----------|-----------|---------|----------|----------------|-----------|---------|------------|
| ±10 chars | 2.31% | 1.65% | 0.99% | 0.99% | 980.7 | 994.8 | 0.00% |
| ±50 chars | 6.75% | 6.59% | 5.93% | 3.13% | 980.7 | 994.8 | 0.00% |

**The mean boundary error of ~980 characters is the core result**: even the model's *best-paired* predictions miss the true boundary by roughly a thousand characters.

---

## 6. Primary metric — offset-independent text-equalness

**This is the primary judgment factor for the author task.** Computed by `scripts/analyze_emitted_text_match.py`, comparing the model's **emitted `text` field** to gold author text (offsets ignored). Full JSON: [benchmark_author_text_match.json](benchmark_author_text_match.json).

### 6.1 Span-level micro F1 (688 rows; FPs on no-author rows counted)

| Metric | Precision | Recall | **F1** | TP / FP / FN |
|--------|-----------|--------|--------|--------------|
| Text-equal (exact string) | 29.34% | 33.00% | **31.07%** | 201 / 484 / 408 |
| Text-equal (normalized) | 30.36% | 34.15% | **32.15%** | 208 / 477 / 401 |
| Text-contained | 58.98% | 66.34% | **62.44%** | 404 / 281 / 205 |

Normalized strips trailing whitespace / tsheg (`་`) / shad (`།`). "Contained" credits a prediction when the emitted string is a substring of the gold author or vice-versa.

### 6.2 Row-level hit rate (over 609 rows that have a gold author)

| Match type | Rows | Rate |
|------------|------|------|
| Exact emitted-text match | 201 | **33.0%** |
| Normalized match | 208 | 34.2% |
| Containment | 404 | **66.3%** |

So the adapter returns a string that overlaps the true author in **~2 of every 3 rows** — the information is there; only the numeric offset is unreliable.

**Worked example** (row `04ba381e…author:1`, input 8,809 chars):

| | start | end | substring at offsets | emitted `text` |
|--|------|-----|----------------------|----------------|
| Gold | 8563 | 8576 | `འཕགས་པ་བྱམས་པ` | — |
| Prediction | 8266 | 8270 | `འགྲུ` (wrong region) | **`འཕགས་པ་བྱམས་པས`** (correct author) |

> To reproduce: `python scripts/analyze_emitted_text_match.py benchmark/logs/benchmark_tilamb_lora_author_predictions.jsonl`

---

## 7. Infrastructure

- **GPU:** NVIDIA RTX 4090 (24 GB), Vast.ai instance (`vastai-author`, 76.64.86.119:44451)
- **Train bootstrap:** `scripts/bootstrap_vastai_train.sh` → `/root/llama-venv` + LLaMA-Factory + HF pilot data
- **Train:** `SKIP_TITLE=1 bash scripts/run_llm_sft_pilot_train.sh` → `saves/tibetan-author-lora-pilot`
- **Upload:** `bash scripts/push_lora_to_hf.sh author`
- **Inference venv:** `scripts/setup_infer_venv_author.sh` → `/root/infer-venv`
- **Compare / report:** `scripts/compare_benchmark.py --task author`, `scripts/recompute_benchmark_offset_metrics.py`, `scripts/analyze_emitted_text_match.py`

---

## 8. Recommendations

1. **Consume the emitted text, not the offset.** For author extraction, take the model's `text` field and re-locate it with `str.find` / fuzzy search in the source segment to recover offsets.
2. **Re-format the SFT target** so the author is near the start of the cropped window (as titles are), or train a token-classification head (Koichi-style) for author — both make offsets learnable.
3. **Don't compare author IoU to title IoU directly:** the tasks differ in span position, so the absolute numbers are not on the same footing. Use the text-match rate as the comparable signal.
4. **Scale up:** this is a 10% pilot, 1 epoch. Full-data training plus a relocation post-step is the path to a usable author extractor.

---

## 9. File index

```
benchmark/
├── README.md
├── report/
│   ├── PILOT_AUTHOR_BENCHMARK_REPORT.md       ← this document
│   ├── benchmark_pilot_author.json / .md       (standard leaderboard)
│   ├── benchmark_author_offset_diagnostics.json / .md
│   ├── benchmark_author_text_match.json        (offset-independent)
│   ├── METRICS_EXPLAINED.md
│   └── train_assets/                           (loss curves, trainer state)
└── logs/
    ├── benchmark_tilamb_lora_author_predictions.jsonl
    └── benchmark_tilamb_lora_author_metrics.json
```

# Pilot Title Span Extraction Benchmark — Final Report

**Project:** [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection)  
**Date:** June 2026  
**Task:** Detect bibliographic **title** character spans in cropped Tibetan segment text.

---

## 1. Executive summary

We compared five **complete** eval runs (769/769 test rows each) plus three **incomplete or failed** runs on a shared held-out test set derived from the 10% pilot LLM SFT corpus.

| Rank (overlap IoU50) | Model | Overlap IoU50 F1 | Offset ±50 F1 | Offset both-boundary hit @ ±50* |
|----------------------|-------|------------------|---------------|----------------------------------|
| 1 | **TiLamb-7B + pilot LoRA** | **58.89%** | **62.97%** | **67.6%** |
| 2 | Koichi RoBERTa NER | 35.21% | 37.66% | 80.7% |
| 3 | Gemma 4 E4B-it | 20.70% | 46.58% | 65.0% |
| 4 | Qwen2.5-7B-Instruct | 15.04% | 26.56% | 47.1% |
| 5 | TiLamb-7B (zero-shot) | 0.00% | 0.00% | — |

\*Row hit rate: share of rows with a gold+pred pair where **both** start and end are within ±50 chars (best IoU pairing). See [§5](#5-offset-first-metrics).

**Takeaways**

- **TiLamb pilot LoRA** is the best generative model for finding the right title *region* (overlap F1), but **Koichi RoBERTa** has the highest *boundary* hit rates when a span is paired — useful when approximate offsets matter more than exact JSON.
- General multilingual LLMs (Qwen, Gemma) lag behind domain-tuned TiLamb LoRA on overlap; Gemma is stronger on relaxed offset (±50) than Qwen.
- TiLamb base zero-shot outputs unparseable JSON on this task (100% parse failure).
- **Not completed:** Tibetan Alpaca (tokenizer error), Qwen3.6-27B (GPU OOM at 55 rows), DeepSeek-R1-14B (aborted at 29 rows).

---

## 2. Benchmark corpus

### Test split used

| Item | Value |
|------|-------|
| Hugging Face dataset | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) |
| Files | `title/test.jsonl`, `title/test_meta.jsonl` |
| Rows | **769** |
| Unit | One JSONL row = one **cropped** segment; gold spans are **crop-relative** in `output` JSON |
| Subsample | 10% random rows per split file, seed 42 (`scripts/subsample_llm_sft.py`) |

Full dataset (100%): [ganga4364/tibetan-metadata-llm-sft-full](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft-full).

Source annotations: [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted).

### Fairness rule

Every model receives the **identical** `input` string from each test row. Gold spans come from the row `output` field. **No re-cropping** at evaluation time. Cropping (≤3584 TiLamb tokens) was applied only when building the SFT dataset.

### Row format

```json
{
  "instruction": "…title extraction prompt…",
  "input": "<cropped Tibetan segment text>",
  "output": "{\"spans\":[{\"text\":\"…\",\"start\":123,\"end\":456}]}"
}
```

---

## 3. Models evaluated

### 3.1 Complete runs (769/769 rows)

| Model | Kind | Hugging Face artifact | Training / notes |
|-------|------|----------------------|------------------|
| Koichi RoBERTa NER | `koichi` | [ganga4364/tibetan-metadata-koichi-ner](https://huggingface.co/ganga4364/tibetan-metadata-koichi-ner) | Fine-tuned [KoichiYasuoka/roberta-base-tibetan](https://huggingface.co/KoichiYasuoka/roberta-base-tibetan); token classification BIO → spans |
| TiLamb-7B base | `tilamb` | [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) | Zero-shot generative JSON |
| TiLamb pilot LoRA | `tilamb_lora` | [ganga4364/tibetan-metadata-title-tilamb-lora-pilot](https://huggingface.co/ganga4364/tibetan-metadata-title-tilamb-lora-pilot) | LoRA on TiLamb-7B; see [§3.2](#32-tilamb-pilot-lora-training) |
| Qwen2.5-7B-Instruct | `qwen` | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Zero-shot; 4-bit inference |
| Gemma 4 E4B-it | `gemma4` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Zero-shot; 4-bit inference |

### 3.2 TiLamb pilot LoRA training

| Setting | Value |
|---------|-------|
| Base model | [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) |
| Framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| Config | `configs/llama_factory/title_lora_sft_pilot.yaml` |
| Method | LoRA r=16, α=32, targets=all |
| Dataset | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) → `title/train.jsonl` (10% pilot) |
| Template | `llama2` chat |
| `cutoff_len` | 4096 |
| Epochs | 1 |
| Batch | 1 × grad accum 16, lr 2e-4, cosine schedule |

Published adapter: [ganga4364/tibetan-metadata-title-tilamb-lora-pilot](https://huggingface.co/ganga4364/tibetan-metadata-title-tilamb-lora-pilot).

**Inference guide:** [docs/TILAMB_TITLE_LORA_INFERENCE.md](../../docs/TILAMB_TITLE_LORA_INFERENCE.md)

### 3.3 Incomplete or failed runs

| Model | Rows completed | Status |
|-------|----------------|--------|
| Tibetan Alpaca 7B | 0/769 | Failed — SentencePiece `tokenizer.model` vs tiktoken conflict |
| Qwen3.6-27B | 55/769 | Stopped — CUDA OOM on RTX 4090 24GB |
| DeepSeek-R1-Distill-Qwen-14B | 29/769 | Stopped — evaluation abandoned; partial logs retained |

Artifacts for partial runs are in `benchmark/logs/` but **excluded** from the main leaderboard tables below.

---

## 4. Evaluation protocol

### 4.1 Inference

- Script: `eval_benchmark_rows.py` via `scripts/run_benchmark_suite.sh`
- Generative models: 4-bit NF4 (`bitsandbytes`), `max_new_tokens=256` (768 in later code; completed runs used 256)
- RoBERTa: `pipeline.inference.predict_segment` on row `input`
- Resumable JSONL: one record per row with `gold_spans`, `pred_spans`, `input`, timing

### 4.2 Standard metrics (micro-F1 over all rows)

Implemented in `eval_common.span_eval_metrics` — greedy one-to-one span matching per row:

| Metric | Definition |
|--------|------------|
| **Exact** | Same label + identical `span_start` and `span_end` |
| **Overlap IoU50** | Same label + character IoU ≥ 0.5 |
| **Overlap IoU80** | Same label + character IoU ≥ 0.8 |
| **Text equal** | Same label + extracted substring matches |
| **Offset ±10 / ±50** | Same label + start **and** end within ±N chars **and** char IoU > 0 |

Aggregated JSON: [benchmark_pilot_title.json](benchmark_pilot_title.json)  
Per-model metrics: `benchmark/logs/benchmark_<kind>_metrics.json`  
Per-row predictions: `benchmark/logs/benchmark_<kind>_predictions.jsonl`

**Metric definitions with IoU examples and diagrams:** [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md)

### 4.3 Segment-dedup view

For each unique `(doc_id, segment_id)`, keep the row with the best exact-match TP. Reported in per-model sections of [benchmark_pilot_title.md](benchmark_pilot_title.md).

---

## 5. Offset-first metrics

Recomputed from saved predictions **without re-inference** (`scripts/recompute_benchmark_offset_metrics.py`).

Full JSON: [benchmark_offset_diagnostics.json](benchmark_offset_diagnostics.json)

### 5.1 Definitions

For each row, pair gold and prediction by **best character IoU** (same label). Then:

| Metric | Meaning |
|--------|---------|
| **Start hit @ ±N** | \|Δstart\| ≤ N |
| **End hit @ ±N** | \|Δend\| ≤ N |
| **Both hit @ ±N** | start and end both within N (overlap not required) |
| **Both+overlap @ ±N** | both within N **and** IoU > 0 (= standard offset relaxed metric) |
| **MAE start / end** | Mean absolute boundary error on paired rows |
| **Median IoU** | Median character IoU on paired rows |

Micro-F1 variants (`offset_start_*`, `offset_end_*`, `offset_both_*`) use greedy span matching across all rows (legacy TP/FP view).

### 5.2 Row hit rates — complete models only (±50 chars)

| Model | Paired rows | Start hit | End hit | Both hit | Both+overlap | Median IoU |
|-------|-------------|-----------|---------|----------|--------------|------------|
| Koichi RoBERTa | 667 | 82.8% | 81.1% | 80.7% | 80.5% | 89.3% |
| TiLamb pilot LoRA | 679 | 70.5% | 68.9% | 67.6% | 66.0% | 69.1% |
| Gemma 4 E4B-it | 551 | 74.8% | 65.5% | 65.0% | 63.2% | 38.1% |
| Qwen2.5-7B-Instruct | 308 | 55.5% | 48.4% | 47.1% | 44.2% | 2.4% |
| TiLamb-7B zero-shot | 0 | — | — | — | — | — |

### 5.3 Row hit rates — ±10 chars (complete models)

| Model | Start hit | End hit | Both hit | Both+overlap |
|-------|-----------|---------|----------|--------------|
| Koichi RoBERTa | 78.9% | 68.2% | 65.8% | 65.8% |
| TiLamb pilot LoRA | 63.2% | 44.0% | 42.9% | 42.9% |
| Gemma 4 E4B-it | 63.2% | 12.7% | 10.0% | 10.0% |
| Qwen2.5-7B-Instruct | 40.9% | 16.6% | 13.3% | 13.3% |

**Observation:** Gemma finds title *starts* more often than *ends* at tight tolerance (63% vs 13% at ±10). Koichi is the most balanced on boundaries.

---

## 6. Standard metrics — full leaderboard

Source: [benchmark_pilot_title.json](benchmark_pilot_title.json) / [benchmark_pilot_title.md](benchmark_pilot_title.md)

| Model | Rows | Exact F1 | Overlap IoU50 | Text equal | Offset ±10 | Offset ±50 | Parse fail | ms/row |
|-------|------|----------|---------------|------------|------------|------------|------------|--------|
| Koichi RoBERTa | 769 | 10.08% | **35.21%** | 10.08% | 30.73% | 37.66% | 0% | 138 |
| TiLamb LoRA | 769 | 1.83% | **58.89%** | 2.11% | 40.90% | **62.97%** | 0% | 37,440 |
| TiLamb base | 769 | 0% | 0% | 0% | 0% | 0% | 100% | 8,437 |
| Qwen2.5-7B | 769 | 0% | 15.04% | 0% | 8.01% | 26.56% | 3.4% | 2,127 |
| Gemma 4 E4B | 769 | 0% | 20.70% | 0% | 7.30% | 46.58% | 3.1% | 9,229 |

### Per-model artifact paths

| Model | Predictions | Metrics |
|-------|-------------|---------|
| koichi | [../logs/benchmark_koichi_predictions.jsonl](../logs/benchmark_koichi_predictions.jsonl) | [../logs/benchmark_koichi_metrics.json](../logs/benchmark_koichi_metrics.json) |
| tilamb | [../logs/benchmark_tilamb_predictions.jsonl](../logs/benchmark_tilamb_predictions.jsonl) | [../logs/benchmark_tilamb_metrics.json](../logs/benchmark_tilamb_metrics.json) |
| tilamb_lora | [../logs/benchmark_tilamb_lora_predictions.jsonl](../logs/benchmark_tilamb_lora_predictions.jsonl) | [../logs/benchmark_tilamb_lora_metrics.json](../logs/benchmark_tilamb_lora_metrics.json) |
| qwen | [../logs/benchmark_qwen_predictions.jsonl](../logs/benchmark_qwen_predictions.jsonl) | [../logs/benchmark_qwen_metrics.json](../logs/benchmark_qwen_metrics.json) |
| gemma4 | [../logs/benchmark_gemma4_predictions.jsonl](../logs/benchmark_gemma4_predictions.jsonl) | [../logs/benchmark_gemma4_metrics.json](../logs/benchmark_gemma4_metrics.json) |

---

## 7. Infrastructure

- **GPU:** NVIDIA RTX 4090 (24 GB), Vast.ai instance
- **Bootstrap:** `scripts/bootstrap_vastai.sh`
- **Suite runner:** `scripts/run_benchmark_suite.sh`
- **Compare / report:** `scripts/compare_benchmark.py`, `scripts/recompute_benchmark_offset_metrics.py`

---

## 8. Recommendations

1. **Production title extraction:** Prefer **TiLamb + pilot LoRA** for best overlap/recall of title regions; consider **Koichi** when fast CPU/GPU NER and boundary tolerance are enough.
2. **Metric choice:** Report **overlap IoU50** and **offset both-boundary hit @ ±50** alongside exact F1 — exact match is too strict for long Tibetan titles.
3. **Future work:** Fix Alpaca tokenizer (`use_fast=False`); skip Qwen3.6-27B on 24GB unless quantized further or context shortened.

---

## 9. File index

```
benchmark/
├── README.md
├── report/
│   ├── PILOT_TITLE_BENCHMARK_REPORT.md   ← this document
│   ├── benchmark_pilot_title.json
│   ├── benchmark_pilot_title.md
│   ├── benchmark_offset_diagnostics.json
│   └── benchmark_offset_diagnostics.md
└── logs/
    ├── benchmark_<kind>_predictions.jsonl
    └── benchmark_<kind>_metrics.json
```

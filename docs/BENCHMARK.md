# Tibetan Metadata Benchmark — Pilot Title Test

Row-level benchmark for **title span extraction** on the 10% pilot SFT held-out test split.

## Corpus

| Item | Value |
|------|-------|
| Dataset | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) |
| Split | `title/test.jsonl` + `title/test_meta.jsonl` |
| Unit | One JSONL row = one cropped segment with crop-relative gold spans |
| Subsample | 10% random rows per split file (seed 42); see `scripts/subsample_llm_sft.py` |

This is **not** the RoBERTa segment test (`data/roberta_full/splits/test.jsonl`, 6,492 unique segments). The pilot test is smaller and matches the TiLamb pilot training distribution.

## Models (v1)

| Model | Kind | HF artifact |
|-------|------|-------------|
| Koichi RoBERTa NER | `koichi` | `ganga4364/tibetan-metadata-koichi-ner` |
| TiLamb-7B base | `tilamb` | `YoLo2000/TiLamb-7B` (zero-shot) |
| TiLamb pilot LoRA | `tilamb_lora` | `ganga4364/tibetan-metadata-title-tilamb-lora-pilot` |
| Tibetan Alpaca 7B | `alpaca` | `ymaoj/Tibetan-Alpaca-7B` (zero-shot) |
| Qwen2.5-7B-Instruct | `qwen` | `Qwen/Qwen2.5-7B-Instruct` (zero-shot) |

**Fairness rule:** every model receives the same `input` string from the test row. Gold spans come from the row `output` JSON (crop-relative offsets). No re-cropping at eval time.

## Metrics

Title-only micro-F1 over all test rows (from `eval_common.span_eval_metrics`):

- **Exact** — `(label, span_start, span_end)` must match
- **Overlap IoU50** — same label, character IoU ≥ 0.5
- **Text equal** — extracted title string matches
- **Offset ±10 / ±50** — start/end within tolerance and spans overlap

Secondary: **segment-dedup** summary (best row per `(doc_id, segment_id)` by exact match) in the comparison report.

## VastAI setup

SSH (example):

```
Host vastai-benchmark
  HostName 82.141.118.42
  Port 10671
  User root
  IdentityFile ~/.ssh/<your-vastai-crf-private-key>
  LocalForward 8080 localhost:8080
```

Bootstrap:

```bash
ssh vastai-benchmark
echo "hf_..." > /root/.hf_token && chmod 600 /root/.hf_token
git clone https://github.com/OpenPecha/tibetan-text-meta-detection.git
cd tibetan-text-meta-detection
bash scripts/bootstrap_vastai.sh
```

## Run benchmark

Smoke test (5 rows per model):

```bash
source /root/infer-venv/bin/activate
cd /root/tibetan-text-meta-detection
bash scripts/run_benchmark_suite.sh
```

Full eval (all test rows, tmux):

```bash
SMOKE=0 bash scripts/run_benchmark_suite.sh
tmux attach -t benchmark_pilot
```

Single model:

```bash
MODEL=koichi bash scripts/run_benchmark_suite.sh
MODEL=tilamb_lora SMOKE=0 bash scripts/run_benchmark_suite.sh
```

Manual per-model:

```bash
python eval_benchmark_rows.py --model-kind koichi \
  --checkpoint models/koichi-ner \
  --resume

python eval_benchmark_rows.py --model-kind tilamb --resume

python eval_benchmark_rows.py --model-kind tilamb_lora \
  --adapter /root/lora/tibetan-title-pilot --resume

python eval_benchmark_rows.py --model-kind alpaca --resume
python eval_benchmark_rows.py --model-kind qwen --resume
```

Leaderboard:

```bash
python scripts/compare_benchmark.py
```

Outputs:

- `logs/benchmark_<kind>_predictions.jsonl` — resumable per-row predictions
- `logs/benchmark_<kind>_metrics.json` — aggregated metrics
- `docs/metrics/benchmark_pilot_title.md` — human-readable leaderboard
- `docs/metrics/benchmark_pilot_title.json` — machine-readable summary

## Qwen2.5 and Tibetan

Qwen2.5 handles Tibetan via multilingual pretraining but its default tokenizer is **less efficient** than TiLamb’s Tibetan-extended vocabulary (more tokens per syllable → slower inference). Use Qwen as a general multilingual zero-shot baseline, not a Tibetan-optimized model.

## Implementation files

| File | Role |
|------|------|
| `eval_benchmark_rows.py` | Unified row-level evaluator |
| `llm_sft/model_backends.py` | TiLamb / Alpaca / Qwen generative backends |
| `scripts/bootstrap_vastai.sh` | Instance bootstrap + HF downloads |
| `scripts/run_benchmark_suite.sh` | Sequential suite runner |
| `scripts/compare_benchmark.py` | Leaderboard aggregation |
| `tests/test_benchmark_rows.py` | Unit tests for loading/aggregation |

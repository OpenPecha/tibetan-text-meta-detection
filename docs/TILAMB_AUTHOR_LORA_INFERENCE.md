# TiLamb author LoRA — inference guide

Run inference with the fine-tuned author adapter:

**[ganga4364/tibetan-metadata-author-tilamb-lora-pilot](https://huggingface.co/ganga4364/tibetan-metadata-author-tilamb-lora-pilot)**

Base model: **[YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B)**

---

## 1. What the model expects

| Input | Description |
|-------|-------------|
| **Segment text** | A Tibetan string (usually one BDRC outliner segment, possibly pre-cropped) |
| **Instruction** | Fixed author-extraction prompt (same as SFT training) |
| **Chat template** | Llama-2 style via `tokenizer.apply_chat_template` |

The model returns **JSON only** with author span(s):

```json
{
  "spans": [
    {"text": "exact substring from input", "start": 84, "end": 97}
  ]
}
```

Offsets are **0-based**, **crop-relative** to the segment string you provide, `end` exclusive.

Default instruction (`llm_sft/prompts.py`):

```text
Extract the bibliographic AUTHOR from the Tibetan segment below.
Reply with JSON only using the key "spans".
Each span must include "text", "start", and "end" (0-based character
offsets relative to the segment text, inclusive at end).
Every span text must be an exact substring of the input.
If there is no author, reply: {"spans": []}.
```

---

## 2. Install dependencies

```bash
pip install torch transformers peft accelerate bitsandbytes
```

Tested stack: `transformers>=4.56`, `peft>=0.18`, CUDA GPU with **~8 GB VRAM** (4-bit).

```bash
huggingface-cli login   # if adapter is private or for faster downloads
```

---

## 3. Minimal Python example

Copy-paste runnable script:

```python
import json
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "YoLo2000/TiLamb-7B"
ADAPTER = "ganga4364/tibetan-metadata-author-tilamb-lora-pilot"

INSTRUCTION = (
    'Extract the bibliographic AUTHOR from the Tibetan segment below. '
    'Reply with JSON only using the key "spans". '
    'Each span must include "text", "start", and "end" (0-based character '
    'offsets relative to the segment text, inclusive at end). '
    'Every span text must be an exact substring of the input. '
    'If there is no author, reply: {"spans": []}.'
)

def parse_spans(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"No JSON in model output: {raw[:200]!r}")
    return json.loads(m.group(0))


def predict_author(segment: str, max_new_tokens: int = 256) -> dict:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    user = f"{INSTRUCTION}\n\n{segment}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_spans(raw)


if __name__ == "__main__":
    segment = "…paste Tibetan segment here…"
    print(json.dumps(predict_author(segment), ensure_ascii=False, indent=2))
```

### CPU / full precision

Omit `BitsAndBytesConfig` and load in `torch.bfloat16` or `float16` if you have enough RAM (~14 GB+ for 7B).

---

## 4. Repo helpers

### 4.1 Single row from JSONL (`llm_sft.inference`)

Uses sliding windows for segments longer than the token budget (same logic as training):

```bash
cd tibetan-text-meta-detection
pip install torch transformers peft accelerate bitsandbytes

python -m llm_sft.inference \
  --jsonl data/llm_sft_pilot_10pct/author/test.jsonl \
  --row 0 \
  --task author \
  --adapter ganga4364/tibetan-metadata-author-tilamb-lora-pilot \
  --max-input-tokens 3584 \
  --max-new-tokens 256
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--jsonl` | (required) | Alpaca-format JSONL with `input` / `output` |
| `--row` | `0` | Row index |
| `--task` | `title` | `title` or `author` (use `author` here) |
| `--adapter` | local path | HF repo ID or directory |
| `--max-input-tokens` | `3584` | Tokens per window |
| `--stride-tokens` | `3000` | Window stride |
| `--no-4bit` | off | Full-precision load |

### 4.2 Benchmark evaluator

Same backend as the pilot benchmark (`llm_sft/model_backends.py`):

```bash
python eval_benchmark_rows.py \
  --model-kind tilamb_lora \
  --task author \
  --adapter ganga4364/tibetan-metadata-author-tilamb-lora-pilot \
  --test-jsonl data/llm_sft_pilot_10pct/author/test.jsonl \
  --meta-jsonl data/llm_sft_pilot_10pct/author/test_meta.jsonl \
  --predictions logs/benchmark_tilamb_lora_author_predictions.jsonl \
  --metrics-out logs/benchmark_tilamb_lora_author_metrics.json \
  --resume
```

---

## 5. Get test data

```bash
hf download ganga4364/tibetan-metadata-llm-sft \
  --include "author/test.jsonl" "author/test_meta.jsonl" \
  --local-dir data/llm_sft_pilot_10pct
```

Dataset card: [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft)

---

## 6. Training recap (how this adapter was made)

| Item | Value |
|------|-------|
| Config | `configs/llama_factory/author_lora_sft_pilot.yaml` |
| LoRA | r=16, α=32 |
| Train script | `SKIP_TITLE=1 bash scripts/run_llm_sft_pilot_train.sh` |
| Upload script | `scripts/push_lora_to_hf.sh author` |

---

## 7. Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on 8 GB GPU | Use 4-bit (`bitsandbytes`); shorten segment or use windowed `llm_sft.inference` |
| Invalid JSON | Increase `max_new_tokens`; check raw output; model may emit extra text before `{` |
| Wrong offsets | Offsets are for the **cropped** string you sent, not the full document |
| Slow on long segments | Use `--max-input-tokens 3584` windowing in `llm_sft.inference` |

---

## 8. Related docs

- [benchmark/report/PILOT_AUTHOR_BENCHMARK_REPORT.md](../benchmark/report/PILOT_AUTHOR_BENCHMARK_REPORT.md) — evaluation results
- [benchmark/report/METRICS_EXPLAINED.md](../benchmark/report/METRICS_EXPLAINED.md) — IoU and offset metrics
- [docs/TILAMB_TITLE_LORA_INFERENCE.md](TILAMB_TITLE_LORA_INFERENCE.md) — title adapter guide
- [docs/BENCHMARK.md](BENCHMARK.md) — full benchmark protocol
- [hub/author_tilamb_lora_pilot_README.md](../hub/author_tilamb_lora_pilot_README.md) — Hugging Face model card source

---
license: apache-2.0
base_model: YoLo2000/TiLamb-7B
tags:
  - lora
  - tibetan
  - metadata-extraction
  - author
library_name: peft
pipeline_tag: text-generation
---

# Tibetan bibliographic author extraction — TiLamb-7B LoRA (pilot)

LoRA adapter on [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) for **author** span extraction from Tibetan text segments.

| Item | Link |
|------|------|
| Base model | [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) |
| Training data | [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) (10% pilot) |
| Benchmark report | [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection/blob/main/benchmark/report/PILOT_AUTHOR_BENCHMARK_REPORT.md) |

**Pilot benchmark (author test split):** see the [benchmark report](https://github.com/OpenPecha/tibetan-text-meta-detection/blob/main/benchmark/report/PILOT_AUTHOR_BENCHMARK_REPORT.md) for Overlap IoU50 / Offset ±50 / start-end hit rates.

## Training

| Setting | Value |
|---------|-------|
| Framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| Method | LoRA r=16, α=32, targets=all |
| Template | `llama2` chat |
| `cutoff_len` | 4096 |
| Epochs | 1 |
| Dataset | `author/train.jsonl` from pilot SFT split |

## Output format

JSON with key `spans`. Each span has `text`, `start`, `end` (0-based character offsets in the **input segment**; `end` is exclusive, matching Python slicing).

```json
{"spans": [{"text": "མི་ཕམ་རྒྱ་མཚོ", "start": 84, "end": 97}]}
```

If no author: `{"spans": []}`

## Install

```bash
pip install torch transformers peft accelerate bitsandbytes
```

Recommended: `transformers>=4.56`, `peft>=0.18`. A CUDA GPU with ~8 GB VRAM is enough for 4-bit inference.

## Quick inference (Python)

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

segment = "…your Tibetan segment text…"

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
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

# Parse JSON (model may wrap in markdown)
m = re.search(r"\{[\s\S]*\}", raw)
result = json.loads(m.group(0) if m else raw)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

## Long segments (sliding windows)

If the segment exceeds ~3584 TiLamb tokens, use the repo’s windowed inference (same as training crops):

```bash
git clone https://github.com/OpenPecha/tibetan-text-meta-detection.git
cd tibetan-text-meta-detection
pip install -r requirements.txt  # torch, transformers, peft, bitsandbytes

python -m llm_sft.inference \
  --jsonl path/to/segment.jsonl \
  --row 0 \
  --task author \
  --adapter ganga4364/tibetan-metadata-author-tilamb-lora-pilot
```

See [docs/TILAMB_AUTHOR_LORA_INFERENCE.md](https://github.com/OpenPecha/tibetan-text-meta-detection/blob/main/docs/TILAMB_AUTHOR_LORA_INFERENCE.md) for full options.

## Benchmark-style batch eval

```bash
python eval_benchmark_rows.py \
  --model-kind tilamb_lora \
  --task author \
  --adapter ganga4364/tibetan-metadata-author-tilamb-lora-pilot \
  --test-jsonl data/llm_sft_pilot_10pct/author/test.jsonl \
  --meta-jsonl data/llm_sft_pilot_10pct/author/test_meta.jsonl \
  --predictions logs/my_tilamb_lora_author_predictions.jsonl \
  --metrics-out logs/my_tilamb_lora_author_metrics.json \
  --resume
```

Download the test split from [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) (`author/test.jsonl`).

## Limitations

- Trained on **cropped** BDRC outliner segments (≤3584 tokens); very long raw documents should be windowed first.
- Pilot LoRA: 10% training subsample — use for experiments; retrain on full split for production.
- Offsets are relative to the **string you pass in**, not the original document.

## Citation / project

[OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection)

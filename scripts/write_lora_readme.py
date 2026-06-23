#!/usr/bin/env python3
"""Write Hugging Face model card for TiLamb LoRA adapters."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE_CARD = ROOT / "hub" / "title_tilamb_lora_pilot_README.md"

AUTHOR_CARD_TEMPLATE = """---
license: apache-2.0
base_model: YoLo2000/TiLamb-7B
tags:
- lora
- tibetan
- metadata-extraction
library_name: peft
pipeline_tag: text-generation
---

# Tibetan bibliographic AUTHOR extraction — TiLamb-7B LoRA pilot

LoRA adapter on [YoLo2000/TiLamb-7B](https://huggingface.co/YoLo2000/TiLamb-7B) for bibliographic **author** span extraction from Tibetan text segments.

## Training

- Framework: LLaMA-Factory
- Method: LoRA r=16, alpha=32
- Dataset: [ganga4364/tibetan-metadata-llm-sft](https://huggingface.co/datasets/ganga4364/tibetan-metadata-llm-sft) — 10% pilot split
- Epochs: 1
- Context: 4096
- Chat template: llama2

## Output format

JSON with key `spans`. Each span has `text`, `start`, `end` (0-based character offsets in the input segment). Use `{{"spans": []}}` when none found.

## Inference

See the title adapter guide: [docs/TILAMB_TITLE_LORA_INFERENCE.md](https://github.com/OpenPecha/tibetan-text-meta-detection/blob/main/docs/TILAMB_TITLE_LORA_INFERENCE.md) — use `--task author` and adapter `{repo}`.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "YoLo2000/TiLamb-7B"
adapter = "{repo}"

tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["title", "author"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    if args.task == "title":
        if not TITLE_CARD.is_file():
            raise SystemExit(f"Missing {TITLE_CARD}")
        text = TITLE_CARD.read_text(encoding="utf-8")
    else:
        text = AUTHOR_CARD_TEMPLATE.format(repo=args.repo)

    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

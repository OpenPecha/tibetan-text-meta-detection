#!/usr/bin/env python3
"""Unsloth title LoRA inference on one JSONL row (LLaMA-Factory llama2 prompt)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON in output: {text[:400]!r}")
    return json.loads(m.group(0))


def main() -> None:
    jsonl = Path(sys.argv[1] if len(sys.argv) > 1 else "data/llm_sft_sample/title/test.jsonl")
    row_idx = int(sys.argv[2] if len(sys.argv) > 2 else 0)
    adapter = sys.argv[3] if len(sys.argv) > 3 else "/root/lora/tibetan-title-pilot"
    base = "YoLo2000/TiLamb-7B"
    max_input_tokens = 3584

    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[row_idx])
    instruction = row["instruction"]
    segment = row["input"]
    gold = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]

    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"Loading base {base} + adapter {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    FastLanguageModel.for_inference(model)

    enc = tokenizer(segment, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(s, e) for s, e in enc["offset_mapping"] if e > s]
    n_tok = len(offsets)
    if n_tok > max_input_tokens:
        crop_text = segment[offsets[0][0] : offsets[max_input_tokens - 1][1]]
        print(f"NOTE: segment {n_tok} tokens -> using first {max_input_tokens} token window")
    else:
        crop_text = segment
        print(f"segment tokens: {n_tok} (full segment)")

    user = f"{instruction}\n\n{crop_text}"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
        use_cache=True,
    )
    new_ids = outputs[0, inputs["input_ids"].shape[1] :]
    pred_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = parse_json(pred_text)

    print("\n=== GOLD ===")
    print(json.dumps(gold, ensure_ascii=False, indent=2))
    print("\n=== PRED ===")
    print(json.dumps(pred, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

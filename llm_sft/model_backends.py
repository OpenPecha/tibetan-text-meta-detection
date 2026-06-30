"""Generative model backends for benchmark row evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_sft.inference import JSON_RE, parse_spans_json

ModelFamily = Literal["llama", "qwen", "qwen36", "gemma"]

DEFAULT_MAX_NEW_TOKENS = 768  # was 256; title JSON rarely needs more

DEFAULT_BASES: dict[str, str] = {
    "tilamb": "YoLo2000/TiLamb-7B",
    "tilamb_lora": "YoLo2000/TiLamb-7B",
    "alpaca": "ymaoj/Tibetan-Alpaca-7B",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma4": "google/gemma-4-E4B-it",
    "qwen36_27b": "Qwen/Qwen3.6-27B",
    "deepseek_r1_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}


@dataclass
class GenerativeSpec:
    kind: str
    base_model: str
    adapter_path: str | None = None
    family: ModelFamily = "llama"
    load_in_4bit: bool = True


def spec_for_kind(
    kind: str,
    *,
    base_model: str | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
) -> GenerativeSpec:
    if kind == "tilamb":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["tilamb"],
            adapter_path=None,
            family="llama",
            load_in_4bit=load_in_4bit,
        )
    if kind == "tilamb_lora":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["tilamb_lora"],
            adapter_path=adapter_path or "/root/lora/tibetan-title-pilot",
            family="llama",
            load_in_4bit=load_in_4bit,
        )
    if kind == "alpaca":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["alpaca"],
            adapter_path=None,
            family="llama",
            load_in_4bit=load_in_4bit,
        )
    if kind == "qwen":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["qwen"],
            adapter_path=None,
            family="qwen",
            load_in_4bit=load_in_4bit,
        )
    if kind == "gemma4":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["gemma4"],
            adapter_path=None,
            family="gemma",
            load_in_4bit=load_in_4bit,
        )
    if kind == "qwen36_27b":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["qwen36_27b"],
            adapter_path=None,
            family="qwen36",
            load_in_4bit=load_in_4bit,
        )
    if kind == "deepseek_r1_14b":
        return GenerativeSpec(
            kind=kind,
            base_model=base_model or DEFAULT_BASES["deepseek_r1_14b"],
            adapter_path=None,
            family="qwen",
            load_in_4bit=load_in_4bit,
        )
    raise ValueError(f"Unknown generative kind: {kind}")


def effective_max_context(model: Any) -> int:
    cfg = model.config
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if max_pos:
        return int(max_pos)
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        text_max = getattr(text_cfg, "max_position_embeddings", None)
        if text_max:
            return int(text_max)
    return 4096


def format_prompt(
    tokenizer: Any,
    instruction: str,
    segment: str,
    *,
    family: ModelFamily = "llama",
) -> str:
    user = f"{instruction}\n\n{segment}"
    if getattr(tokenizer, "chat_template", None):
        template_kwargs: dict[str, Any] = {}
        if family == "qwen36":
            # Benchmark needs direct JSON, not chain-of-thought traces.
            template_kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    if family == "qwen":
        return (
            f"<|im_start|>user\n{user}\n"
            f"<|im_start|>assistant\n"
        )
    return f"<s>[INST] {user} [/INST]"


def load_generative_model(spec: GenerativeSpec) -> tuple[Any, Any]:
    quant = None
    if spec.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tok_source = spec.adapter_path or spec.base_model
    if spec.kind == "gemma4":
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(spec.base_model)
            tokenizer = processor.tokenizer
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(spec.base_model, trust_remote_code=True)
    elif spec.kind == "alpaca":
        # Llama2/SentencePiece — avoid broken tiktoken conversion on tokenizer.model
        tokenizer = AutoTokenizer.from_pretrained(
            spec.base_model,
            use_fast=False,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not spec.load_in_4bit else None,
    )
    if spec.adapter_path:
        model = PeftModel.from_pretrained(model, spec.adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_input_text(
    model: Any,
    tokenizer: Any,
    instruction: str,
    input_text: str,
    *,
    family: ModelFamily = "llama",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    label: str = "title",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run generation on cropped input text as-is (no re-crop)."""
    prompt = format_prompt(tokenizer, instruction, input_text, family=family)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    max_pos = effective_max_context(model)
    truncated = prompt_tokens >= max_pos - max_new_tokens

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    output_tokens = int(new_tokens.shape[0])
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    parsed = parse_spans_json(raw)
    parse_ok = bool(raw.startswith("{") or JSON_RE.search(raw))
    spans = [
        {
            "label": label,
            "span_start": int(s["start"]),
            "span_end": int(s["end"]),
            "text": s.get("text", ""),
        }
        for s in parsed.get("spans", [])
        if "start" in s and "end" in s
    ]
    meta = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "truncated": truncated,
        "parse_ok": parse_ok,
        "raw_response": raw,
    }
    return spans, meta

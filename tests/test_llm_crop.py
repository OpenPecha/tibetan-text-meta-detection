"""Tests for LLM SFT cropping on sample_4doc."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from llm_sft.config import SFTConfig
from llm_sft.crop import (
    char_span_to_token_range,
    generate_crops_for_task,
    remap_spans,
    tokenize_segment,
    validate_spans,
)
from llm_sft.iterate import iter_segments

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_4doc"
TILAMB = "YoLo2000/TiLamb-7B"


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TILAMB)


@pytest.fixture
def sft_config(tmp_path) -> SFTConfig:
    return SFTConfig(
        extracted_dir=SAMPLE_DIR,
        output_dir=tmp_path / "out",
        tokenizer_name=TILAMB,
        max_context_tokens=512,
        crops_per_positive=3,
        crops_per_negative=1,
        seed=42,
    )


def test_iter_segments_sample_count():
    segments = list(iter_segments(SAMPLE_DIR))
    assert len(segments) >= 4


def test_title_author_roundtrip_on_52a125b4(tokenizer, sft_config):
    seg = next(
        s
        for s in iter_segments(SAMPLE_DIR)
        if s["doc_id"] == "52a125b4-1f4f-48eb-9aca-42a0da412902"
        and s["segment_id"] == "36daa465-ce65-4a9c-a667-6612f6a605cf"
    )
    tokenized = tokenize_segment(tokenizer, seg["text"])
    rng = random.Random(99)

    for task in ("title", "author"):
        crops = generate_crops_for_task(
            tokenized, seg["annotations"], task, sft_config, rng
        )
        assert crops
        for crop in crops:
            spans = remap_spans(
                crop.text,
                seg["annotations"], task, crop.char_start, crop.char_end
            )
            assert spans, f"{task} crop should contain gold span"
            validate_spans(crop.text, spans)


def test_title_negative_239cdff5(tokenizer, sft_config):
    seg = next(
        s
        for s in iter_segments(SAMPLE_DIR)
        if s["doc_id"] == "239cdff5-03e5-4c3c-91fd-f8c20f2b69e3"
    )
    tokenized = tokenize_segment(tokenizer, seg["text"])
    rng = random.Random(7)
    crops = generate_crops_for_task(
        tokenized, seg["annotations"], "title", sft_config, rng
    )
    for crop in crops:
        spans = remap_spans(
            crop.text,
            seg["annotations"],
            "title",
            crop.char_start,
            crop.char_end,
        )
        assert spans == []


def test_author_not_begin_anchored_on_long_segment(tokenizer):
    """Author near segment end should appear in crop, not only at crop start."""
    seg = next(
        s
        for s in iter_segments(SAMPLE_DIR)
        if s["doc_id"] == "52a125b4-1f4f-48eb-9aca-42a0da412902"
        and s["segment_id"] == "36daa465-ce65-4a9c-a667-6612f6a605cf"
    )
    cfg = SFTConfig(
        extracted_dir=SAMPLE_DIR,
        output_dir=Path("."),
        max_context_tokens=256,
        crops_per_positive=5,
        seed=123,
    )
    tokenized = tokenize_segment(tokenizer, seg["text"])
    rng = random.Random(123)
    ratios = []
    for crop in generate_crops_for_task(
        tokenized, seg["annotations"], "author", cfg, rng
    ):
        spans = remap_spans(
            crop.text,
            seg["annotations"],
            "author",
            crop.char_start,
            crop.char_end,
        )
        assert spans
        center = (spans[0].start + spans[0].end) / 2
        ratios.append(center / len(crop.text))
    assert max(ratios) > 0.3


def test_span_position_not_always_at_edge(tokenizer):
    """Anti-bias: span center should not always be > 0.9 in crop."""
    seg = next(
        s
        for s in iter_segments(SAMPLE_DIR)
        if s["doc_id"] == "52a125b4-1f4f-48eb-9aca-42a0da412902"
        and s["segment_id"] == "36daa465-ce65-4a9c-a667-6612f6a605cf"
    )
    cfg = SFTConfig(
        extracted_dir=SAMPLE_DIR,
        output_dir=Path("."),
        max_context_tokens=400,
        crops_per_positive=50,
        seed=2024,
    )
    tokenized = tokenize_segment(tokenizer, seg["text"])
    rng = random.Random(2024)
    ratios = []
    for crop in generate_crops_for_task(
        tokenized, seg["annotations"], "title", cfg, rng
    ):
        spans = remap_spans(
            crop.text,
            seg["annotations"],
            "title",
            crop.char_start,
            crop.char_end,
        )
        if spans:
            center = (spans[0].start + spans[0].end) / 2
            ratios.append(center / len(crop.text))
    assert len(ratios) >= 50
    assert sum(1 for r in ratios if r < 0.85) >= 10


@pytest.mark.slow
def test_build_dataset_smoke(tokenizer, tmp_path):
    from llm_sft.build_dataset import build

    cfg = SFTConfig(
        extracted_dir=SAMPLE_DIR,
        output_dir=tmp_path / "llm_sft",
        max_context_tokens=3584,
        crops_per_positive=2,
        seed=42,
    )
    stats = build(cfg)
    assert stats["rows"] > 0
    title_train = tmp_path / "llm_sft" / "title" / "train.jsonl"
    assert title_train.is_file()
    line = title_train.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert "instruction" in row and "input" in row and "output" in row

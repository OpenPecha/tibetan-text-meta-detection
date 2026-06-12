"""Tests for window-relative BIO labeling in roberta_windows."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pipeline.bio import LABEL_TO_ID
from pipeline.roberta_windows import (
    WindowSpec,
    build_training_example,
    label_window,
    slide_segment,
    validate_segment_roundtrip,
    validate_window_roundtrip,
)


def make_offsets(n_tokens: int) -> list[tuple[int, int]]:
    return [(i * 4, (i + 1) * 4) for i in range(n_tokens)]


def test_begin_00_labels_entity_at_segment_start():
    offsets = make_offsets(600)
    anns = [{"label": "author", "span_start": 16, "span_end": 32}]
    window = WindowSpec(0, 512, "begin", 0)

    tags, window_anns = label_window(offsets, window, anns)

    assert len(window_anns) == 1
    assert any(t.startswith("B-") or t.startswith("I-") for t in tags)
    assert tags[4] == "B-AUTHOR"
    assert tags[5] == "I-AUTHOR"


def test_begin_01_labels_entity_not_at_segment_start():
    offsets = make_offsets(600)
    anns = [{"label": "title", "span_start": 1100, "span_end": 1120}]
    window = WindowSpec(256, 600, "begin", 1)

    tags, window_anns = label_window(offsets, window, anns)

    assert len(window_anns) == 1
    assert window_anns[0]["span_start"] == 76
    assert window_anns[0]["span_end"] == 96
    assert any(t != "O" for t in tags), "begin_01 must not be all-O when span is inside window"
    assert tags[19] == "B-TITLE"
    assert tags[20] == "I-TITLE"
    assert tags[21] == "I-TITLE"
    assert tags[22] == "I-TITLE"
    assert tags[23] == "I-TITLE"


def test_end_window_labels_entity_near_segment_end():
    offsets = make_offsets(600)
    anns = [{"label": "author", "span_start": 2350, "span_end": 2380}]
    window = WindowSpec(88, 600, "end", 0)

    tags, window_anns = label_window(offsets, window, anns)

    assert len(window_anns) == 1
    assert any("AUTHOR" in t for t in tags)


def test_outside_span_is_all_o():
    offsets = make_offsets(600)
    anns = [{"label": "title", "span_start": 100, "span_end": 120}]
    window = WindowSpec(256, 600, "begin", 1)

    tags, window_anns = label_window(offsets, window, anns)

    assert window_anns == []
    assert all(t == "O" for t in tags)


def test_roundtrip_begin_01():
    offsets = make_offsets(600)
    anns = [{"label": "title", "span_start": 1100, "span_end": 1120}]
    window = WindowSpec(256, 600, "begin", 1)
    tags, _ = label_window(offsets, window, anns)

    ok, errors = validate_window_roundtrip(offsets, window, anns, tags)
    assert ok, errors


@dataclass
class MockTokenizer:
    cls_token_id: int = 0
    sep_token_id: int = 2
    pad_token_id: int = 1


def test_stored_offsets_remain_segment_relative():
    offsets = make_offsets(600)
    anns = [{"label": "title", "span_start": 1100, "span_end": 1120}]
    window = WindowSpec(256, 600, "begin", 1)
    tags, _ = label_window(offsets, window, anns)
    input_ids = list(range(len(offsets)))
    tokenizer = MockTokenizer()

    example = build_training_example(
        tokenizer,
        window,
        input_ids,
        offsets,
        tags,
        {"doc_id": "d1", "segment_id": "s1"},
        max_length=window.end_tok - window.start_tok + 2,
    )

    win_offsets = example["offset_mapping"][1:-1]
    seg_offsets = offsets[window.start_tok : window.end_tok][: len(win_offsets)]
    assert win_offsets == seg_offsets


def _example_has_entity_label(example: dict) -> bool:
    o_id = LABEL_TO_ID["O"]
    for label_id in example["labels"]:
        if label_id not in (-100, o_id):
            return True
    return False


@pytest.mark.slow
def test_slide_segment_window_annotations_have_labels():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "spsither/tibetan_RoBERTa_S_e3",
        add_prefix_space=True,
    )
    # Long segment: title mid-text, author near end (not at char 0).
    prefix = "བོད་ཡིག " * 400
    title = "རྒྱུད་ཀྱི་རྒྱལ་མཚན"
    middle = " " + "ཡིག་ཆ " * 200
    author = "ཀུན་མཁྱེན"
    text = prefix + title + middle + author + " " + "མཐའ " * 100
    title_start = len(prefix)
    title_end = title_start + len(title)
    author_start = len(prefix + title + middle)
    author_end = author_start + len(author)
    annotations = [
        {"label": "title", "span_start": title_start, "span_end": title_end},
        {"label": "author", "span_start": author_start, "span_end": author_end},
    ]

    examples = slide_segment(
        tokenizer,
        text,
        annotations,
        metadata={"doc_id": "test", "segment_id": "seg1"},
        max_begin=15,
        max_end=15,
    )
    assert examples

    for ex in examples:
        if ex.get("window_annotations"):
            assert _example_has_entity_label(ex), (
                f"{ex['window_name']}: window_annotations set but all-O labels"
            )

    failures, errors = validate_segment_roundtrip(
        tokenizer, text, annotations, max_begin=15, max_end=15
    )
    assert failures == 0, errors

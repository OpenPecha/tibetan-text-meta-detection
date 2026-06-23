"""Tests for overlap and offset span evaluation metrics."""

from eval_common import (
    char_iou,
    compare_title_spans,
    exact_span_match,
    offset_relaxed_span_match,
    overlap_span_match,
    span_eval_metrics,
    text_equal_span_match,
)

TEXT = "0123456789abcdef"
TEXT_DUP = "titleXtitleY"


def test_char_iou_disjoint():
    assert char_iou(0, 3, 5, 8) == 0.0


def test_char_iou_full_overlap():
    assert char_iou(2, 8, 2, 8) == 1.0


def test_char_iou_partial():
    # [2,8) vs [5,10) -> inter=3, union=8
    assert char_iou(2, 8, 5, 10) == 0.375


def test_exact_span_match():
    gold = [{"label": "title", "span_start": 1, "span_end": 5}]
    pred_ok = [{"label": "title", "span_start": 1, "span_end": 5}]
    pred_bad = [{"label": "title", "span_start": 1, "span_end": 6}]
    assert exact_span_match(gold, pred_ok) == (1, 0, 0)
    assert exact_span_match(gold, pred_bad) == (0, 1, 1)


def test_overlap_span_match():
    gold = [{"label": "title", "span_start": 0, "span_end": 10}]
    pred = [{"label": "title", "span_start": 2, "span_end": 9}]
    assert overlap_span_match(gold, pred, iou_threshold=0.5) == (1, 0, 0)
    assert overlap_span_match(gold, pred, iou_threshold=0.95) == (0, 1, 1)


def test_text_equal_span_match():
    gold = [{"label": "title", "span_start": 0, "span_end": 5}]
    pred = [{"label": "title", "span_start": 6, "span_end": 11}]
    assert text_equal_span_match(gold, pred, TEXT_DUP) == (1, 0, 0)


def test_offset_relaxed_span_match():
    gold = [{"label": "title", "span_start": 10, "span_end": 20}]
    pred = [{"label": "title", "span_start": 12, "span_end": 22}]
    assert offset_relaxed_span_match(gold, pred, start_tol=5, end_tol=5) == (1, 0, 0)
    assert offset_relaxed_span_match(gold, pred, start_tol=1, end_tol=1) == (0, 1, 1)


def test_span_eval_metrics_bundle():
    gold = [{"label": "title", "span_start": 0, "span_end": 4}]
    pred = [{"label": "title", "span_start": 0, "span_end": 4}]
    m = span_eval_metrics(gold, pred, TEXT)
    assert m["exact"]["f1"] == 1.0
    assert m["overlap_iou50"]["f1"] == 1.0
    assert m["text_equal"]["f1"] == 1.0


def test_offset_relaxed_50():
    gold = [{"label": "title", "span_start": 10, "span_end": 60}]
    pred = [{"label": "title", "span_start": 40, "span_end": 90}]
    assert offset_relaxed_span_match(gold, pred, start_tol=50, end_tol=50) == (1, 0, 0)
    assert offset_relaxed_span_match(gold, pred, start_tol=10, end_tol=10) == (0, 1, 1)


def test_compare_title_spans_with_text():
    gold = [{"label": "title", "span_start": 0, "span_end": 4, "text": "0123"}]
    pred = [{"start": 1, "end": 5, "text": "1234"}]
    cmp = compare_title_spans(gold, pred, text=TEXT)
    assert cmp["text_match"] is False
    assert cmp["char_iou"] > 0.5
    assert cmp["start_delta"] == 1


def test_offset_start_end_only():
    gold = [{"label": "title", "span_start": 10, "span_end": 20}]
    pred = [{"label": "title", "span_start": 12, "span_end": 30}]
    from eval_common import offset_end_relaxed_span_match, offset_start_relaxed_span_match

    assert offset_start_relaxed_span_match(gold, pred, start_tol=5) == (1, 0, 0)
    assert offset_end_relaxed_span_match(gold, pred, end_tol=5) == (0, 1, 1)
    assert offset_end_relaxed_span_match(gold, pred, end_tol=15) == (1, 0, 0)


def test_row_offset_diagnostics():
    gold = [{"label": "title", "span_start": 10, "span_end": 20}]
    pred = [{"label": "title", "span_start": 15, "span_end": 25}]
    from eval_common import row_offset_diagnostics

    d = row_offset_diagnostics(gold, pred, tolerances=(10, 50))
    assert d["start_abs_err"] == 5
    assert d["end_abs_err"] == 5
    assert d["start_within_10"] is True
    assert d["both_within_10"] is True
    assert d["both_within_10_overlap"] is True

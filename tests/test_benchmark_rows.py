"""Unit tests for benchmark row loading and metric aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_benchmark_rows import (
    aggregate_row_metrics,
    gold_spans_from_output,
    load_test_rows,
)


def test_gold_spans_from_output():
    gold = gold_spans_from_output('{"spans":[{"text":"abc","start":1,"end":4}]}')
    assert len(gold) == 1
    assert gold[0]["span_start"] == 1
    assert gold[0]["span_end"] == 4
    assert gold[0]["label"] == "title"


def test_load_test_rows(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    meta_path = tmp_path / "test_meta.jsonl"
    test_path.write_text(
        json.dumps(
            {
                "instruction": "Extract title",
                "input": "hello world",
                "output": '{"spans":[]}',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "id": "doc1:seg1:title:0",
                "doc_id": "doc1",
                "segment_id": "seg1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_test_rows(test_path, meta_path)
    assert len(rows) == 1
    assert rows[0]["row_id"] == "doc1:seg1:title:0"
    assert rows[0]["input"] == "hello world"


def test_aggregate_row_metrics(tmp_path: Path):
    pred_path = tmp_path / "pred.jsonl"
    pred_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "r1",
                        "input": "title text here",
                        "gold_spans": [
                            {"label": "title", "span_start": 0, "span_end": 5}
                        ],
                        "pred_spans": [
                            {"label": "title", "span_start": 0, "span_end": 5}
                        ],
                        "inference_ms": 10.0,
                        "parse_ok": True,
                    }
                ),
                json.dumps(
                    {
                        "row_id": "r2",
                        "input": "no match",
                        "gold_spans": [
                            {"label": "title", "span_start": 0, "span_end": 2}
                        ],
                        "pred_spans": [],
                        "inference_ms": 20.0,
                        "parse_ok": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    agg = aggregate_row_metrics(pred_path)
    assert agg["rows_evaluated"] == 2
    assert agg["exact_title"]["tp"] == 1
    assert agg["exact_title"]["fn"] == 1

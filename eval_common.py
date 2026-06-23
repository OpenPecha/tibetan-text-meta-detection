"""Shared helpers for segment-level evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def load_doc_entries(extracted_dir: Path) -> dict[str, dict]:
    index_path = extracted_dir / "index.jsonl"
    by_doc: dict[str, dict] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            by_doc[entry["doc_id"]] = entry
    return by_doc


def load_segment_payload(extracted_dir: Path, entry: dict, segment_id: str) -> dict | None:
    ann_path = extracted_dir / entry["annotations_path"]
    with ann_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    for segment in payload.get("segments", []):
        if segment["segment_id"] == segment_id:
            return segment
    return None


def gold_from_annotations(
    annotations: list[dict],
    label: str | None = None,
) -> list[dict]:
    out: list[dict] = []
    for ann in annotations:
        if label is not None and ann.get("label") != label:
            continue
        out.append(
            {
                "label": ann["label"],
                "span_start": ann["span_start"],
                "span_end": ann["span_end"],
            }
        )
    return out


def exact_span_match(gold: list[dict], pred: list[dict]) -> tuple[int, int, int]:
    gold_set = {
        (a["label"].lower(), a["span_start"], a["span_end"]) for a in gold
    }
    pred_set = {
        (s["label"].lower(), s["span_start"], s["span_end"]) for s in pred
    }
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def char_iou(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """Character IoU on half-open intervals [start, end)."""
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    if inter == 0:
        return 0.0
    union = max(end_a, end_b) - min(start_a, start_b)
    return inter / union if union > 0 else 0.0


def extract_span_text(text: str, span: dict) -> str:
    start = int(span["span_start"])
    end = int(span["span_end"])
    return text[max(0, start) : min(len(text), end)]


def greedy_span_match(
    gold: list[dict],
    pred: list[dict],
    *,
    matches: Callable[[dict, dict], bool],
) -> tuple[int, int, int]:
    """One-to-one greedy matching: each pred matches at most one gold."""
    used_pred: set[int] = set()
    tp = 0
    fn = 0
    for g in gold:
        matched = False
        for j, p in enumerate(pred):
            if j in used_pred:
                continue
            if matches(g, p):
                used_pred.add(j)
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1
    fp = len(pred) - len(used_pred)
    return tp, fp, fn


def overlap_span_match(
    gold: list[dict],
    pred: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    def _matches(g: dict, p: dict) -> bool:
        if g["label"].lower() != p["label"].lower():
            return False
        return (
            char_iou(
                int(g["span_start"]),
                int(g["span_end"]),
                int(p["span_start"]),
                int(p["span_end"]),
            )
            >= iou_threshold
        )

    return greedy_span_match(gold, pred, matches=_matches)


def text_equal_span_match(
    gold: list[dict],
    pred: list[dict],
    text: str,
) -> tuple[int, int, int]:
    def _matches(g: dict, p: dict) -> bool:
        if g["label"].lower() != p["label"].lower():
            return False
        return extract_span_text(text, g).strip() == extract_span_text(text, p).strip()

    return greedy_span_match(gold, pred, matches=_matches)


def offset_start_relaxed_span_match(
    gold: list[dict],
    pred: list[dict],
    *,
    start_tol: int = 10,
) -> tuple[int, int, int]:
    """Match when label agrees and |gold_start - pred_start| <= start_tol (end ignored)."""

    def _matches(g: dict, p: dict) -> bool:
        if g["label"].lower() != p["label"].lower():
            return False
        return abs(int(g["span_start"]) - int(p["span_start"])) <= start_tol

    return greedy_span_match(gold, pred, matches=_matches)


def offset_end_relaxed_span_match(
    gold: list[dict],
    pred: list[dict],
    *,
    end_tol: int = 10,
) -> tuple[int, int, int]:
    """Match when label agrees and |gold_end - pred_end| <= end_tol (start ignored)."""

    def _matches(g: dict, p: dict) -> bool:
        if g["label"].lower() != p["label"].lower():
            return False
        return abs(int(g["span_end"]) - int(p["span_end"])) <= end_tol

    return greedy_span_match(gold, pred, matches=_matches)


def offset_relaxed_span_match(
    gold: list[dict],
    pred: list[dict],
    *,
    start_tol: int = 10,
    end_tol: int = 10,
    require_overlap: bool = True,
) -> tuple[int, int, int]:
    """Match when label agrees, both boundaries within tolerance, optionally requiring overlap."""

    def _matches(g: dict, p: dict) -> bool:
        if g["label"].lower() != p["label"].lower():
            return False
        gs, ge = int(g["span_start"]), int(g["span_end"])
        ps, pe = int(p["span_start"]), int(p["span_end"])
        if abs(gs - ps) > start_tol or abs(ge - pe) > end_tol:
            return False
        if require_overlap and char_iou(gs, ge, ps, pe) <= 0.0:
            return False
        return True

    return greedy_span_match(gold, pred, matches=_matches)


def best_label_pair_by_iou(
    gold: list[dict],
    pred: list[dict],
) -> tuple[dict | None, dict | None, float]:
    """Highest-IoU same-label (gold, pred) pair; used for per-row offset diagnostics."""
    best_g: dict | None = None
    best_p: dict | None = None
    best_iou = -1.0
    for g in gold:
        for p in pred:
            if g["label"].lower() != p["label"].lower():
                continue
            iou = char_iou(
                int(g["span_start"]),
                int(g["span_end"]),
                int(p["span_start"]),
                int(p["span_end"]),
            )
            if iou > best_iou:
                best_iou = iou
                best_g, best_p = g, p
    if best_g is None:
        return None, None, 0.0
    return best_g, best_p, best_iou


def row_offset_diagnostics(
    gold: list[dict],
    pred: list[dict],
    *,
    tolerances: tuple[int, ...] = (10, 50),
) -> dict:
    """Per-row boundary errors for the best IoU same-label pair (offset-first view)."""
    g, p, iou = best_label_pair_by_iou(gold, pred)
    out: dict = {
        "has_gold": bool(gold),
        "has_pred": bool(pred),
        "paired": g is not None and p is not None,
        "char_iou": iou,
        "start_abs_err": None,
        "end_abs_err": None,
        "start_signed_err": None,
        "end_signed_err": None,
    }
    for tol in tolerances:
        out[f"start_within_{tol}"] = False
        out[f"end_within_{tol}"] = False
        out[f"both_within_{tol}"] = False
        out[f"both_within_{tol}_overlap"] = False

    if g is None or p is None:
        return out

    gs, ge = int(g["span_start"]), int(g["span_end"])
    ps, pe = int(p["span_start"]), int(p["span_end"])
    start_abs = abs(gs - ps)
    end_abs = abs(ge - pe)
    out["start_abs_err"] = start_abs
    out["end_abs_err"] = end_abs
    out["start_signed_err"] = ps - gs
    out["end_signed_err"] = pe - ge
    for tol in tolerances:
        sw = start_abs <= tol
        ew = end_abs <= tol
        out[f"start_within_{tol}"] = sw
        out[f"end_within_{tol}"] = ew
        out[f"both_within_{tol}"] = sw and ew
        out[f"both_within_{tol}_overlap"] = sw and ew and iou > 0.0
    return out


def span_eval_counts(
    gold: list[dict],
    pred: list[dict],
    text: str | None = None,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
) -> dict[str, tuple[int, int, int]]:
    """Return tp/fp/fn for exact, overlap, text-equal, and offset-relaxed matching."""
    counts: dict[str, tuple[int, int, int]] = {
        "exact": exact_span_match(gold, pred),
        "overlap_iou50": overlap_span_match(gold, pred, iou_threshold=0.5),
        "overlap_iou80": overlap_span_match(gold, pred, iou_threshold=0.8),
    }
    for tol in offset_tolerances:
        counts[f"offset_start_{tol}"] = offset_start_relaxed_span_match(
            gold, pred, start_tol=tol
        )
        counts[f"offset_end_{tol}"] = offset_end_relaxed_span_match(
            gold, pred, end_tol=tol
        )
        counts[f"offset_both_{tol}"] = offset_relaxed_span_match(
            gold,
            pred,
            start_tol=tol,
            end_tol=tol,
            require_overlap=False,
        )
        counts[f"offset_relaxed_{tol}"] = offset_relaxed_span_match(
            gold,
            pred,
            start_tol=tol,
            end_tol=tol,
            require_overlap=True,
        )
    if offset_tolerances:
        counts["offset_relaxed"] = counts[f"offset_relaxed_{offset_tolerances[0]}"]
    if text is not None:
        counts["text_equal"] = text_equal_span_match(gold, pred, text)
    return counts


def span_eval_metrics(
    gold: list[dict],
    pred: list[dict],
    text: str | None = None,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
) -> dict[str, dict]:
    """Precision/recall/F1 for exact, overlap, text-equal, and offset-relaxed matching."""
    out: dict[str, dict] = {}
    for name, (tp, fp, fn) in span_eval_counts(
        gold,
        pred,
        text,
        offset_tolerances=offset_tolerances,
    ).items():
        out[name] = {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
    return out


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compare_title_spans(
    gold: list[dict],
    pred: list[dict],
    *,
    text: str | None = None,
) -> dict:
    g_spans = [s for s in gold if s.get("label") == "title"]
    p_spans: list[dict] = []
    for s in pred:
        if "span_start" in s:
            p_spans.append(
                {
                    "label": s.get("label", "title"),
                    "span_start": s["span_start"],
                    "span_end": s["span_end"],
                    "text": s.get("text", ""),
                }
            )
        else:
            p_spans.append(
                {
                    "label": "title",
                    "span_start": s["start"],
                    "span_end": s["end"],
                    "text": s.get("text", ""),
                }
            )
    if not g_spans and not p_spans:
        return {
            "text_match": True,
            "start_match": True,
            "end_match": True,
            "char_iou": 1.0,
            "start_delta": 0,
            "end_delta": 0,
        }
    if not g_spans or not p_spans:
        return {
            "text_match": False,
            "start_match": False,
            "end_match": False,
            "char_iou": 0.0,
            "start_delta": None,
            "end_delta": None,
        }
    g, p = g_spans[0], p_spans[0]
    gs, ge = int(g["span_start"]), int(g["span_end"])
    ps, pe = int(p["span_start"]), int(p["span_end"])
    g_text = g.get("text") or (extract_span_text(text, g) if text else "")
    p_text = p.get("text") or (extract_span_text(text, p) if text else "")
    return {
        "text_match": g_text.strip() == p_text.strip() if g_text or p_text else g_text == p_text,
        "start_match": gs == ps,
        "end_match": ge == pe,
        "char_iou": char_iou(gs, ge, ps, pe),
        "start_delta": ps - gs,
        "end_delta": pe - ge,
        "gold_span": g,
        "pred_span": p,
    }


def collect_test_segments(
    splits_dir: Path,
    extracted_dir: Path,
) -> list[dict]:
    test_path = splits_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}")

    seen: set[tuple[str, str]] = set()
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            seen.add((row["doc_id"], row["segment_id"]))

    doc_entries = load_doc_entries(extracted_dir)
    segments: list[dict] = []

    for doc_id, segment_id in sorted(seen):
        entry = doc_entries.get(doc_id)
        if entry is None:
            continue
        segment = load_segment_payload(extracted_dir, entry, segment_id)
        if segment is None:
            continue
        annotations = segment.get("annotations", [])
        if not annotations:
            continue
        segments.append(
            {
                "doc_id": doc_id,
                "segment_id": segment_id,
                "text": segment["text"],
                "annotations": annotations,
            }
        )
    return segments


def load_segment_keys_from_predictions(predictions_path: Path) -> list[tuple[str, str]]:
    """Load (doc_id, segment_id) keys in JSONL row order."""
    keys: list[tuple[str, str]] = []
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.append((row["doc_id"], row["segment_id"]))
    return keys


def filter_segments_by_keys(
    segments: list[dict],
    keys: list[tuple[str, str]],
) -> list[dict]:
    """Return segments matching keys, preserving key order."""
    by_key = {(s["doc_id"], s["segment_id"]): s for s in segments}
    out: list[dict] = []
    missing = 0
    for key in keys:
        seg = by_key.get(key)
        if seg is None:
            missing += 1
            continue
        out.append(seg)
    if missing:
        raise ValueError(f"Missing {missing} segments for requested keys")
    return out


def load_completed_keys(predictions_path: Path) -> set[tuple[str, str]]:
    if not predictions_path.is_file():
        return set()
    done: set[tuple[str, str]] = set()
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["doc_id"], row["segment_id"]))
    return done

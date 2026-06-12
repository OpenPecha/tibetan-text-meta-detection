"""BIO tag sequence generation and validation."""

from __future__ import annotations

from pipeline.tokenize import SyllableToken

BIO_LABELS = ["O", "B-TITLE", "I-TITLE", "B-AUTHOR", "I-AUTHOR"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
IGNORE_LABEL_ID = -100


def _entity_prefix(label: str) -> str:
    return label.upper()


def annotations_to_bio(
    tokens: list[SyllableToken],
    annotations: list[dict],
) -> list[str]:
    """Assign BIO tags to syllable tokens from span annotations."""
    tags = ["O"] * len(tokens)
    if not tokens:
        return tags

    sorted_anns = sorted(
        annotations,
        key=lambda a: (a["span_start"], -(a["span_end"] - a["span_start"])),
    )

    for ann in sorted_anns:
        prefix = _entity_prefix(ann["label"])
        b_tag = f"B-{prefix}"
        i_tag = f"I-{prefix}"
        ann_start = ann["span_start"]
        ann_end = ann["span_end"]
        first = True

        for idx, tok in enumerate(tokens):
            if tok.end <= ann_start:
                continue
            if tok.start >= ann_end:
                break
            if not (tok.start < ann_end and tok.end > ann_start):
                continue

            new_tag = b_tag if first else i_tag
            current = tags[idx]
            if current == "O" or current.endswith(prefix):
                tags[idx] = new_tag
            first = False

    return tags


def subword_annotations_to_bio(
    offsets: list[tuple[int, int]],
    annotations: list[dict],
) -> list[str]:
    """Assign BIO tags to subword tokens from span annotations.

    ``offsets`` are character (start, end) pairs for each subword token in the
    window, relative to the window text (not the full segment).
    """
    tags = ["O"] * len(offsets)
    if not offsets:
        return tags

    sorted_anns = sorted(
        annotations,
        key=lambda a: (a["span_start"], -(a["span_end"] - a["span_start"])),
    )

    for ann in sorted_anns:
        prefix = _entity_prefix(ann["label"])
        b_tag = f"B-{prefix}"
        i_tag = f"I-{prefix}"
        ann_start = ann["span_start"]
        ann_end = ann["span_end"]
        first = True

        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= ann_start:
                continue
            if tok_start >= ann_end:
                break
            if not (tok_start < ann_end and tok_end > ann_start):
                continue

            new_tag = b_tag if first else i_tag
            current = tags[idx]
            if current == "O" or current.endswith(prefix):
                tags[idx] = new_tag
            first = False

    return tags


def subword_bio_to_spans(
    offsets: list[tuple[int, int]],
    tags: list[str],
) -> list[dict]:
    """Reconstruct span annotations from subword BIO tags."""
    spans: list[dict] = []
    idx = 0
    while idx < len(tags):
        tag = tags[idx]
        if tag == "O" or "-" not in tag:
            idx += 1
            continue

        prefix, entity = tag.split("-", 1)
        if prefix != "B":
            idx += 1
            continue

        start = offsets[idx][0]
        end = offsets[idx][1]
        idx += 1
        while idx < len(tags) and tags[idx] == f"I-{entity}":
            end = offsets[idx][1]
            idx += 1

        spans.append(
            {
                "label": entity.lower(),
                "span_start": start,
                "span_end": end,
            }
        )

    return spans


def bio_to_spans(tokens: list[SyllableToken], tags: list[str]) -> list[dict]:
    """Reconstruct span annotations from BIO tags."""
    spans: list[dict] = []
    idx = 0
    while idx < len(tags):
        tag = tags[idx]
        if tag == "O" or "-" not in tag:
            idx += 1
            continue

        prefix, entity = tag.split("-", 1)
        if prefix != "B":
            idx += 1
            continue

        start = tokens[idx].start
        end = tokens[idx].end
        idx += 1
        while idx < len(tags) and tags[idx] == f"I-{entity}":
            end = tokens[idx].end
            idx += 1

        spans.append(
            {
                "label": entity.lower(),
                "span_start": start,
                "span_end": end,
            }
        )

    return spans


def _span_to_token_bounds(
    tokens: list[SyllableToken],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Map a character span to the covering token-aligned span."""
    overlapping = [tok for tok in tokens if tok.start < end and tok.end > start]
    if not overlapping:
        return None
    return overlapping[0].start, overlapping[-1].end


def validate_bio_reconstruction(
    tokens: list[SyllableToken],
    annotations: list[dict],
    tags: list[str],
) -> tuple[bool, list[str]]:
    """Verify BIO tags reconstruct token-aligned spans for each annotation."""
    errors: list[str] = []
    reconstructed = bio_to_spans(tokens, tags)

    expected: list[dict] = []
    for ann in annotations:
        bounds = _span_to_token_bounds(tokens, ann["span_start"], ann["span_end"])
        if bounds is None:
            errors.append(f"missing tokens for {ann['label']} span")
            continue
        expected.append(
            {
                "label": ann["label"],
                "span_start": bounds[0],
                "span_end": bounds[1],
            }
        )

    expected = sorted(
        expected,
        key=lambda x: (x["label"], x["span_start"], x["span_end"]),
    )
    actual = sorted(
        reconstructed,
        key=lambda x: (x["label"], x["span_start"], x["span_end"]),
    )

    if expected != actual:
        errors.append(
            f"span mismatch expected={len(expected)} reconstructed={len(actual)}"
        )
    return len(errors) == 0, errors

"""Read-only exploration of outliner database for title/author extraction."""

from __future__ import annotations

import psycopg2

from config import DB_CONFIG


def _connect():
    for key in ("host", "user", "password"):
        if not DB_CONFIG[key]:
            raise ValueError(f"Missing BENCHMARK_DB_{key.upper()} in environment")
    return psycopg2.connect(**DB_CONFIG)


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _distinct(cur, table: str, column: str) -> None:
    cur.execute(
        f"SELECT {column}, COUNT(*) FROM {table} "
        f"GROUP BY {column} ORDER BY COUNT(*) DESC"
    )
    for val, cnt in cur.fetchall():
        display = repr(val) if val is not None else "NULL"
        print(f"  {display:40s} {cnt:>8,}")


def main() -> None:
    conn = _connect()
    try:
        cur = conn.cursor()

        _print_section("Table row counts")
        for table in ("outliner_documents", "outliner_segments"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  {table}: {cur.fetchone()[0]:,}")

        _print_section("outliner_documents.status")
        _distinct(cur, "outliner_documents", "status")

        _print_section("outliner_segments.label")
        _distinct(cur, "outliner_segments", "label")

        _print_section("outliner_segments.status")
        _distinct(cur, "outliner_segments", "status")

        _print_section("Annotated segments (is_annotated = true)")
        cur.execute(
            "SELECT COUNT(*) FROM outliner_segments WHERE is_annotated = true"
        )
        print(f"  Total: {cur.fetchone()[0]:,}")

        _print_section("Title/author span coverage (is_annotated = true)")
        cur.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (
                    WHERE title_span_start IS NOT NULL
                    AND title_span_end IS NOT NULL
                ) AS with_title_span,
                COUNT(*) FILTER (
                    WHERE author_span_start IS NOT NULL
                    AND author_span_end IS NOT NULL
                ) AS with_author_span,
                COUNT(*) FILTER (
                    WHERE title_span_start IS NOT NULL
                    AND title_span_end IS NOT NULL
                    AND author_span_start IS NOT NULL
                    AND author_span_end IS NOT NULL
                ) AS with_both,
                COUNT(*) FILTER (
                    WHERE (title_span_start IS NULL OR title_span_end IS NULL)
                    AND (author_span_start IS NULL OR author_span_end IS NULL)
                ) AS with_neither
            FROM outliner_segments
            WHERE is_annotated = true
            """
        )
        row = cur.fetchone()
        labels = [
            "total",
            "with_title_span",
            "with_author_span",
            "with_both",
            "with_neither",
        ]
        for label, val in zip(labels, row):
            print(f"  {label:20s} {val:>8,}")

        _print_section("Documents with annotated segments")
        cur.execute(
            """
            SELECT COUNT(DISTINCT document_id)
            FROM outliner_segments
            WHERE is_annotated = true
            """
        )
        print(f"  Distinct documents: {cur.fetchone()[0]:,}")

        cur.execute(
            """
            SELECT COUNT(DISTINCT d.id)
            FROM outliner_documents d
            JOIN outliner_segments s ON s.document_id = d.id
            WHERE s.is_annotated = true AND d.status = 'approved'
            """
        )
        print(f"  Approved documents with annotated segments: {cur.fetchone()[0]:,}")

        _print_section("Sample annotated segments (5 rows)")
        cur.execute(
            """
            SELECT
                s.id,
                s.document_id,
                s.label,
                s.status,
                s.is_annotated,
                s.title,
                s.title_span_start,
                s.title_span_end,
                s.author,
                s.author_span_start,
                s.author_span_end,
                LEFT(s.text, 80) AS text_preview
            FROM outliner_segments s
            WHERE s.is_annotated = true
              AND (
                  (s.title_span_start IS NOT NULL AND s.title_span_end IS NOT NULL)
                  OR (s.author_span_start IS NOT NULL AND s.author_span_end IS NOT NULL)
              )
            LIMIT 5
            """
        )
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            print("  ---")
            for col, val in zip(cols, row):
                if val is None:
                    display = "NULL"
                else:
                    display = str(val)[:120].encode("ascii", errors="replace").decode("ascii")
                print(f"    {col}: {display}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

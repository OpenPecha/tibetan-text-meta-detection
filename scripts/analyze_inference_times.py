#!/usr/bin/env python3
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
times = [float(r["inference_ms"]) for r in rows if r.get("inference_ms")]
print(f"rows={len(rows)} with_timing={len(times)}")
if not times:
    raise SystemExit(0)
for label, subset in [
    ("all", times),
    ("first50", times[:50]),
    ("rows600-650", times[600:650] if len(times) > 600 else []),
    ("last50", times[-50:]),
]:
    if not subset:
        continue
    print(
        f"{label}: mean={statistics.mean(subset):.0f}ms "
        f"median={statistics.median(subset):.0f}ms "
        f"p95={sorted(subset)[int(0.95*len(subset))-1]:.0f}ms"
    )
inputs = [len(r.get("input", "")) for r in rows]
print(f"input_chars: mean={statistics.mean(inputs):.0f} median={statistics.median(inputs):.0f}")

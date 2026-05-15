"""Per-op timing accumulator. Speed is a claim — exposing the numbers turns it
into evidence. Decorate every public function with ``@timed`` and call
``timing_summary()`` to get printable rows.
"""

from __future__ import annotations

import time
from collections import defaultdict
from functools import wraps
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_TIMINGS: dict[str, list[float]] = defaultdict(list)


def timed(label: str | None = None) -> Callable[[F], F]:
    """Record wall-clock ms per call. One ``perf_counter`` per invocation."""

    def decorator(fn: F) -> F:
        name = label or f"{fn.__module__}.{fn.__qualname__}"

        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _TIMINGS[name].append((time.perf_counter() - t0) * 1000.0)

        return wrapper  # type: ignore[return-value]

    return decorator


def reset_timings() -> None:
    _TIMINGS.clear()


def timing_summary() -> list[dict]:
    """Per-op rows: name, calls, total_ms, mean_ms, median_ms, p95_ms."""
    rows = []
    for name, samples in sorted(_TIMINGS.items()):
        if not samples:
            continue
        s = sorted(samples)
        n = len(s)
        rows.append(
            {
                "name": name,
                "calls": n,
                "total_ms": round(sum(s), 3),
                "mean_ms": round(sum(s) / n, 3),
                "median_ms": round(s[n // 2], 3),
                "p95_ms": round(s[min(n - 1, int(n * 0.95))], 3),
            }
        )
    return rows


def format_timing_table() -> str:
    rows = timing_summary()
    if not rows:
        return "(no timings recorded)"
    headers = ["name", "calls", "total_ms", "mean_ms", "median_ms", "p95_ms"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = "  ".join("-" * widths[h] for h in headers)
    lines = ["  ".join(h.ljust(widths[h]) for h in headers), sep]
    for r in rows:
        lines.append("  ".join(str(r[h]).ljust(widths[h]) for h in headers))
    return "\n".join(lines)

from __future__ import annotations

import re
from pathlib import Path

__all__ = (
    "extract_page_num",
    "page_num",
    "pair_str",
)

_PAGE_RE = re.compile(r"p(\d+)(?:_spread)?\.(jpg|jpeg|png|webp)", re.IGNORECASE)


def extract_page_num(path: Path) -> int | None:
    """Extract the page number from a ``pNNN.ext`` filename."""
    m = _PAGE_RE.match(path.name)
    if m:
        return int(m.group(1))
    return None


def page_num(path: Path) -> int:
    """Like :func:`extract_page_num` but raises on failure."""
    n = extract_page_num(path)
    if n is None:
        raise ValueError(f"cannot extract page number from {path.name}")
    return n


def pair_str(a: int, b: int) -> str:
    """Format page numbers as nmanga-compatible ``"A-B"``."""
    return f"{a}-{b}"

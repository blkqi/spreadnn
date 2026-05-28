from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .model import SpreadModel
from .naming import extract_page_num, pair_str

log = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

__all__ = (
    "DetectionResult",
    "detect",
    "detect_spreads",
)


@dataclass
class DetectionResult:
    """Result for a single candidate page pair."""

    even: str
    odd: str
    score: float
    merged: bool
    note: str | None = None


def detect_spreads(
    directory: str | Path,
    *,
    threshold: float = 0.5,
    model_path: str | Path | None = None,
    reverse: bool = False,
    offset: int | None = None,
) -> list[tuple[str, str]]:
    """Detect spreads and return ``[(left_fn, right_fn), ...]``.

    Pairs are in joined output order — the first filename goes on the
    left side of the joined image, the second on the right.  For RTL
    (default) the pair is ``(odd_fn, even_fn)`` in sort order, which
    becomes ``(even_fn, odd_fn)`` after the reading-order swap so that
    a left-to-right scan gives correct narrative order.
    """
    results = detect(directory, threshold=threshold, model_path=model_path, reverse=reverse, offset=offset)
    pairs = [(r.even, r.odd) for r in results if r.merged]
    if not reverse:
        pairs = [(b, a) for a, b in pairs]
    return pairs


def detect(
    directory: str | Path,
    *,
    threshold: float = 0.5,
    model_path: str | Path | None = None,
    reverse: bool = False,
    offset: int | None = None,
) -> list[DetectionResult]:
    """Analyse page images in *directory* and return results for every pair.

    *offset* can be *0* or *1* to force a specific parity, or *None* to
    auto-detect (tries offset 1 first, falls back to 0).
    """
    images = _collect_images(Path(directory))
    if not images:
        raise ValueError(f"no supported images found in {directory}")

    model = SpreadModel(model_path)

    if offset is not None:
        return _detect_at_offset(directory, images, model, offset=offset, threshold=threshold, reverse=reverse)

    # Try offset 1 first (skip first page — most volumes have a single cover)
    results = _detect_at_offset(directory, images, model, offset=1, threshold=threshold, reverse=reverse)
    if any(r.merged for r in results):
        log.info("alignment: offset 1 (first page skipped)")
        return results

    # Fall back to offset 0
    results0 = _detect_at_offset(directory, images, model, offset=0, threshold=threshold, reverse=reverse)
    if any(r0.merged for r0 in results0):
        log.info("alignment: offset 0 (no pages skipped)")
        return results0

    log.info("no spreads detected at either offset")
    return results


def _detect_at_offset(
    directory: str | Path,
    images: list[Path],
    model: SpreadModel,
    offset: int,
    threshold: float,
    reverse: bool = False,
) -> list[DetectionResult]:
    """Pair images starting at *offset* and score each pair."""
    interior = images[offset:]
    results: list[DetectionResult] = []

    for pair in _pairwise(interior):
        if len(pair) == 1:
            log.debug("lone tail: %s", pair[0])
            continue

        name_e, name_o = pair
        img_e, img_o = _read_pair(directory, name_e, name_o)

        if img_e is None or img_o is None:
            results.append(DetectionResult(
                even=name_e, odd=name_o, score=0.0, merged=False,
                note="decode failure",
            ))
            log.warning("decode failure: %s / %s", name_e, name_o)
            continue

        prob = model.score_pair(img_o, img_e) if reverse else model.score_pair(img_e, img_o)
        merged = prob >= threshold
        results.append(DetectionResult(
            even=name_e, odd=name_o, score=round(prob, 4), merged=merged,
        ))

        if merged:
            log.info("SPREAD  %-28s | %-28s | %.4f", name_e, name_o, prob)
        else:
            log.debug("split  %-28s | %-28s | %.4f", name_e, name_o, prob)

    return results


def merged_pairs(results: list[DetectionResult]) -> list[str]:
    """Extract only the merged pairs as ``"A-B"`` strings."""
    return [
        pair_str(
            extract_page_num(Path(r.even)) or 0,
            extract_page_num(Path(r.odd)) or 0,
        )
        for r in results
        if r.merged
    ]



def _collect_images(directory: Path) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMG_EXTS and not p.name.startswith(".")
    )
    log.debug("found %d images in %s", len(paths), directory)
    return paths


def _read_pair(directory: str | Path, name_e: str, name_o: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    def _load(name: str) -> np.ndarray | None:
        buf = (Path(directory) / name).read_bytes()
        arr = np.frombuffer(buf, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return _load(name_e), _load(name_o)


def _pairwise(seq: list[Path]) -> Iterator[tuple[str, ...]]:
    it = iter(seq)
    while True:
        a = next(it, None)
        if a is None:
            return
        b = next(it, None)
        yield (a.name,) if b is None else (a.name, b.name)

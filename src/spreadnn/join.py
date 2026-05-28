from __future__ import annotations

from pathlib import Path

import cv2

__all__ = (
    "join_pair",
    "join_pairs",
)


def join_pair(
    left: Path,
    right: Path,
    output: Path,
    *,
    quality: int = 100,
) -> None:
    """Join two images side-by-side using OpenCV."""
    left_img = cv2.imread(str(left))
    right_img = cv2.imread(str(right))

    if left_img is None:
        raise RuntimeError(f"failed to decode: {left}")
    if right_img is None:
        raise RuntimeError(f"failed to decode: {right}")

    h = max(left_img.shape[0], right_img.shape[0])
    if left_img.shape[0] != h:
        left_img = cv2.resize(left_img, (left_img.shape[1], h), interpolation=cv2.INTER_LINEAR)
    if right_img.shape[0] != h:
        right_img = cv2.resize(right_img, (right_img.shape[1], h), interpolation=cv2.INTER_LINEAR)

    joined = cv2.hconcat([left_img, right_img])

    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok = cv2.imwrite(str(output), joined, params)
    if not ok:
        raise RuntimeError(f"failed to write: {output}")


def join_pairs(
    directory: Path,
    pairs: list[tuple[Path, Path]],
    *,
    quality: int = 100,
    output_dir: Path | None = None,
    dry_run: bool = False,
    no_cleanup: bool = False,
) -> list[Path]:
    """Join multiple image pairs via OpenCV.

    Returns list of output paths that were written (empty if *dry_run*).
    """
    out_dir = output_dir or directory
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for left, right in pairs:
        stems = sorted([left.stem, right.stem])
        out_name = f"{stems[0]}-{stems[1]}.jpg"
        out_path = out_dir / out_name

        if dry_run:
            continue

        join_pair(left, right, out_path, quality=quality)
        written.append(out_path)

        if not no_cleanup:
            left.unlink(missing_ok=True)
            right.unlink(missing_ok=True)

    return written

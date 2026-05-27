from __future__ import annotations

import subprocess as sp
from pathlib import Path

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
    magick_path: str = "magick",
) -> None:
    """Join two images side-by-side using ImageMagick ``+append``.

    Equivalent to::

        magick left.jpg right.jpg -quality 100% +append output.jpg
    """
    cmd = [
        magick_path,
        str(left),
        str(right),
        "-quality", f"{quality}.00%",
        "+append",
        str(output),
    ]
    try:
        sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    except sp.CalledProcessError as exc:
        raise RuntimeError(f"ImageMagick join failed: {' '.join(cmd)}") from exc


def join_pairs(
    directory: Path,
    pairs: list[tuple[Path, Path]],
    *,
    quality: int = 100,
    magick_path: str = "magick",
    output_dir: Path | None = None,
    dry_run: bool = False,
    no_cleanup: bool = False,
) -> list[Path]:
    """Join multiple image pairs via ImageMagick.

    Returns list of output paths that were written (empty if *dry_run*).
    """
    out_dir = output_dir or directory
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for left, right in pairs:
        out_name = f"{left.stem}-{right.stem}.jpg"
        out_path = out_dir / out_name

        if dry_run:
            continue

        join_pair(left, right, out_path, quality=quality, magick_path=magick_path)
        written.append(out_path)

        if not no_cleanup:
            left.unlink(missing_ok=True)
            right.unlink(missing_ok=True)

    return written

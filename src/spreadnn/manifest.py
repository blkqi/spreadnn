from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = (
    "Manifest",
    "dump_manifest",
    "load_manifest",
    "resolve_manifest_pairs",
)

Manifest = list[tuple[str, str]]


def load_manifest(source: Path | str) -> Manifest:
    """Read and parse a ``spreads.json`` manifest.

    Accepts a ``Path`` (reads from disk) or a string (parsed directly).
    Each pair is ``[left_filename, right_filename]``.
    """
    raw = source.read_text() if isinstance(source, Path) else source
    data: list[list[str]] = json.loads(raw)
    return [(str(a), str(b)) for a, b in data]


def dump_manifest(manifest: Manifest, path: Path) -> None:
    """Serialize and write a manifest to *path*."""
    path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")


def resolve_manifest_pairs(
    directory: Path,
    manifest: Manifest,
) -> list[tuple[str, str]]:
    """Resolve filename pairs from a manifest, skipping missing files.

    Pair order is preserved as-is from the manifest.
    """
    resolved: list[tuple[str, str]] = []
    missing: list[str] = []
    for left_name, right_name in manifest:
        left_missing = not (directory / left_name).exists()
        right_missing = not (directory / right_name).exists()
        if left_missing:
            missing.append(left_name)
        if right_missing:
            missing.append(right_name)
        if left_missing or right_missing:
            continue
        resolved.append((left_name, right_name))

    if missing:
        log.warning("manifest references missing files: %s", ", ".join(missing))
    return resolved

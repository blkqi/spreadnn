"""Command-line interface for spreadnn."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import rich_click as click

from ._version import __version__

log = logging.getLogger("spreadnn")

# --------------------------------------------------------------------------- #
# Shared CLI options
# --------------------------------------------------------------------------- #

threshold = click.option(
    "--threshold", "threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.5, show_default=True, metavar="F",
    help="Spread probability threshold.",
)

model_path = click.option(
    "--model", "model_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None, metavar="PATH",
    help="Override bundled model .pth file.",
)

ltr_flag = click.option(
    "--ltr", "ltr",
    is_flag=True, default=False,
    help="Left-to-right reading order (western/flopped). Default is RTL (manga).",
)

offset_opt = click.option(
    "--offset", "offset",
    type=click.Choice(["0", "1", "auto"]),
    default="auto", show_default=True,
    help="Pair alignment offset. 'auto' tries offset 1 first, falls back to 0.",
)

verbose = click.option(
    "-v", "--verbose", "verbose",
    count=True, default=0,
    help="Increase verbosity.",
)


# --------------------------------------------------------------------------- #
# CLI group
# --------------------------------------------------------------------------- #


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version")
def cli() -> None:
    """spreadnn — CNN-based manga spread detection."""


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@threshold
@model_path
@ltr_flag
@offset_opt
@click.option("--manifest", "-m", "manifest", is_flag=True, default=False,
              help="Write spreads.json sidecar instead of stdout.")
@click.option("--output", "-o", "manifest_path", type=click.Path(path_type=Path), default=None, metavar="PATH",
              help="Manifest output path (default: <dir>/spreads.json).")
@verbose
def detect(
    directory: Path,
    threshold: float,
    model_path: str | None,
    ltr: bool,
    offset: str,
    manifest: bool,
    manifest_path: Path | None,
    verbose: int,
) -> None:
    r"""Detect spread pairs in an image directory.

    Outputs a JSON array of ``[left_fn, right_fn]`` pairs to stdout::

        [["p008.jpg", "p007.jpg"]]

    Pass ``--manifest`` to write a spreads.json sidecar instead.
    Both outputs use the same format.
    """
    from .detect import detect_spreads
    from .manifest import dump_manifest

    _configure_logging(verbose)

    try:
        _offset = None if offset == "auto" else int(offset)
        pairs = detect_spreads(directory, threshold=threshold, model_path=model_path, ltr=ltr, offset=_offset)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc

    if manifest:
        out = manifest_path or (directory / "spreads.json")
        dump_manifest(pairs, out)
        log.info("wrote manifest: %s", out)
    else:
        click.echo(json.dumps(pairs))


# --------------------------------------------------------------------------- #
# join
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--manifest", "-m", "manifest_path", type=str,
              default=None, metavar="PATH",
              help="Path to spreads.json manifest (default: <dir>/spreads.json).  Use '-' for stdin.")
@click.option("--quality", "-q", "quality", type=click.IntRange(1, 100),
              default=100, show_default=True, help="JPEG quality for joined output.")
@click.option("--output-dir", "output_dir", type=click.Path(path_type=Path), default=None, metavar="PATH",
              help="Output directory (default: same as input).")
@click.option("--dry-run", "-n", "dry_run", is_flag=True, default=False,
              help="Print what would be joined without writing.")
@click.option("--no-cleanup", is_flag=True, default=False,
              help="Keep original halves of joined pages.")
@verbose
def join(
    directory: Path,
    manifest_path: str | None,
    quality: int,
    output_dir: Path | None,
    dry_run: bool,
    no_cleanup: bool,
    verbose: int,
) -> None:
    """Join spreads from a manifest.

    Pass ``-`` to ``--manifest`` to read from stdin (pipe from ``spreadnn detect``).

    Otherwise expects a ``spreads.json`` in *directory* or an explicit path.
    """
    from .join import join_pairs
    from .manifest import load_manifest, resolve_manifest_pairs

    _configure_logging(verbose)

    if manifest_path == "-":
        manifest = load_manifest(sys.stdin.read())
        log.info("using manifest from stdin")
    else:
        p = Path(manifest_path) if manifest_path else directory / "spreads.json"
        if not p.exists():
            log.error("no manifest found at %s — run 'spreadnn detect --manifest' first", p)
            raise SystemExit(1)
        log.info("using manifest: %s", p)
        manifest = load_manifest(p)

    named = resolve_manifest_pairs(directory, manifest)
    if not named:
        log.info("no valid pairs in manifest — nothing to join")
        return

    pairs = [(directory / a, directory / b) for a, b in named]

    if dry_run:
        log.info("dry-run: would join %d pair(s)", len(pairs))
        for left, right in pairs:
            log.info("  %s + %s", left.name, right.name)
        return

    written = join_pairs(directory, pairs, quality=quality, output_dir=output_dir, no_cleanup=no_cleanup)
    log.info("joined %d spread(s)", len(written))
    for w in written:
        log.info("  created: %s", w.name)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    """Entry point for the ``spreadnn`` console script."""
    _configure_logging(0)
    try:
        cli(auto_envvar_prefix="SPREADNN")
    except SystemExit as exc:
        return exc.code
    except Exception as exc:
        log.error("unexpected error: %s", exc)
        return 1
    return 0


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

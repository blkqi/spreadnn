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
# Shared CLI options (nmanga-style module-level constants)
# --------------------------------------------------------------------------- #

skip_pages = click.option(
    "--skip-pages", "skip_pages",
    type=int, default=1, show_default=True, metavar="N",
    help="Number of leading pages to skip (covers, ToC).",
)

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
    """spreadnn — CNN-based spread detection for nmanga."""


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@skip_pages
@threshold
@model_path
@verbose
def detect(
    directory: Path,
    skip_pages: int,
    threshold: float,
    model_path: str | None,
    verbose: int,
) -> None:
    r"""Detect spread pairs in an image directory.

    Outputs a JSON array of ``[start, end]`` pairs to stdout::

        [[7, 8], [17, 18]]

    Pipe into nmanga::

        nmanga spreads join ./images/ \
          $(spreadnn detect ./images/ | jq -r '.[] | "-s \(.[0])-\(.[1])"')
    """
    from .detect import detect_spreads

    _configure_logging(verbose)

    try:
        pairs = detect_spreads(directory, skip_pages=skip_pages, threshold=threshold, model_path=model_path)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc

    click.echo(json.dumps(pairs))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@skip_pages
@threshold
@model_path
@click.option("--output", "-o", "output_path", type=click.Path(path_type=Path), default=None, metavar="PATH",
              help="Output path for manifest (default: <dir>/spreads.json).")
@click.option("--no-write", is_flag=True, default=False, help="Print manifest to stdout instead.")
@verbose
def manifest(  # noqa: PLR0913
    directory: Path,
    skip_pages: int,
    threshold: float,
    model_path: str | None,
    output_path: Path | None,
    no_write: bool,
    verbose: int,
) -> None:
    """Write a spreads.json sidecar into the image directory."""
    from .detect import detect_spreads

    _configure_logging(verbose)

    try:
        pairs = detect_spreads(directory, skip_pages=skip_pages, threshold=threshold, model_path=model_path)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc

    payload = json.dumps(pairs) + "\n"

    if no_write:
        click.echo(payload, nl=False)
        return

    out = output_path or (directory / "spreads.json")
    out.write_text(payload)
    log.info("wrote manifest: %s", out)


# --------------------------------------------------------------------------- #
# join
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@skip_pages
@threshold
@model_path
@click.option("--quality", "-q", "quality", type=click.IntRange(1, 100),
              default=100, show_default=True, help="JPEG quality for joined output.")
@click.option("--output-dir", "output_dir", type=click.Path(path_type=Path), default=None, metavar="PATH",
              help="Output directory (default: same as input).")
@click.option("--dry-run", "-n", "dry_run", is_flag=True, default=False,
              help="Print what would be joined without writing.")
@click.option("--no-cleanup", is_flag=True, default=False,
              help="Keep original halves of joined pages.")
@verbose
def join(  # noqa: PLR0913
    directory: Path,
    skip_pages: int,
    threshold: float,
    model_path: str | None,
    quality: int,
    output_dir: Path | None,
    dry_run: bool,
    no_cleanup: bool,
    verbose: int,
) -> None:
    """Detect spreads and join them via ImageMagick in one pass."""
    from .detect import detect as _detect
    from .join import join_pairs

    _configure_logging(verbose)

    try:
        results = _detect(directory, skip_pages=skip_pages, threshold=threshold, model_path=model_path)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(1) from exc

    merged = [r for r in results if r.merged]
    if not merged:
        log.info("no spreads detected — nothing to join")
        return

    pairs: list[tuple[Path, Path]] = []
    for r in merged:
        left = directory / r.even
        right = directory / r.odd
        if not left.exists() or not right.exists():
            log.warning("missing file for pair %s / %s — skipping", r.even, r.odd)
            continue
        pairs.append((left, right))

    if dry_run:
        log.info("dry-run: would join %d pair(s)", len(pairs))
        for l, r in pairs:
            log.info("  %s + %s", l.name, r.name)
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

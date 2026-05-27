# spreadnn — spread detection for nmanga pipelines

## Overview

`spreadnn` is a standalone CLI tool that detects two-page spreads in manga
dumps using a MobileNetV3-small binary classifier (the same CNN from
[stitch-spreads](https://github.com/niclas363/stitch-spreads)). It operates on
a flat directory of page images (pNNN files), outputs structured JSON for
piping into jq/nmanga, and optionally runs the ImageMagick join.

It is **not** a CBZ in/out tool — it fills the gap between "unpacked CBZ
directory" and "nmanga spreads join."

## Design Goals

1. **Decoupled detection** — `detect` only produces JSON, never touches images.
2. **nmanga-optimised output** — JSON array of `"A-B"` strings, directly
   pipeable into `nmanga spreads join -s`.
3. **Bundled model** — ships `manga-digital.pth` (9 MB) as a package resource.
4. **Minimal new code** — reuse stitch-spreads' CNN loader and scoring logic.
5. **No CBZ I/O** — works on loose leaf images; user handles archive packing.

## CLI

```
spreadnn detect [OPTIONS] <DIR>
spreadnn manifest [OPTIONS] <DIR>
spreadnn join [OPTIONS] <DIR>
```

### Global options

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-pages N` | 1 | Passthrough leading N pages (covers, ToCs). |
| `--threshold F` | 0.5 | Spread probability threshold [0–1]. |
| `--model PATH` | bundled | Override model .pth file. |

### Subcommands

#### `detect`

Analyse interior pages in pairs. Output newline-delimited JSON (NDJSON) with
one object per pair, ending with a final JSON array of merged-pair strings.

**Output (stdout):**

First, NDJSON lines for every interior pair:

```jsonl
{"even":"p005","odd":"p006","score":0.998,"merged":true}
{"even":"p007","odd":"p008","score":0.003,"merged":false}
```

Then a final summary array of merged pairs (can be fed to `jq -s 'last'` or
`jq -cs '.[-1]'`):

```json
["5-6"]
```

The final array is the **primary machine-consumable output**. The NDJSON lines
are informational/logging for human review.

**Usage with nmanga:**

```bash
nmanga spreads join ./images/ \
  -s $(spreadnn detect --skip-pages 0 ./images/ | jq -cs '.[-1][]')
```

Or for the simpler manifest-based workflow:

```bash
spreadnn manifest --skip-pages 0 ./images/
nmanga spreads join ./images/ -s "$(cat spreads.json | jq -r '.[] | "-s \(.)"')"
```

#### `manifest`

Like `detect`, but writes a `spreads.json` file into `<DIR>` (or a specified
path) containing only the merged pairs as an array of `"A-B"` strings.

```bash
spreadnn manifest ./images/
# → writes ./images/spreads.json with contents:
# ["5-6","17-18","33-34"]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | `<DIR>/spreads.json` | Output path for manifest. |
| `--no-write` | false | Print manifest to stdout instead. |

#### `join`

Combines `detect` + ImageMagick `+append` in a single pass. Saves joined
spreads as `<DIR>/pNNN-pMMM.jpg` and leaves non-merged files untouched.

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | false | Print what would be joined, don't write. |
| `--quality N` | 100 | JPEG quality for joined output. |
| `--output-dir PATH` | `<DIR>` | Where to write output images. |
| `--no-cleanup` | false | Keep originals of joined pages. |

## Detection Algorithm

Identical to stitch-spreads `_score_pair`:

1. Resize both pages to the same height (max of the two).
2. Extract a 256×64 strip centred on the inner edge (32 px from each page).
3. Normalise with ImageNet stats (mean=[0.485,0.456,0.406],
   std=[0.229,0.224,0.225]).
4. Run through MobileNetV3-small with binary classification head.
5. Apply sigmoid → spread probability.
6. Return (probability, gate-reason-or-None).

The "gate" rejects pairs with extreme aspect ratios or very small
dimensions before the CNN runs (same as stitch-spreads).

## Output Schema

### `detect` NDJSON lines

```json
{
  "even": "p005.jpg",
  "odd": "p006.jpg",
  "score": 0.998,
  "merged": true,
  "note": null
}
```

Fields:
- `even`, `odd` — filenames of the pair (left/right in manga reading order).
- `score` — sigmoid probability [0–1].
- `merged` — `true` if score >= threshold.
- `note` — optional string (gate rejection reason, decode failure).

### `detect` final array / `manifest` contents

```json
["A-B", "C-D"]
```

Strings match the `pNNN-pMMM` format nmanga expects (zero-padded page
numbers extracted from filenames).

## File Name Convention

Input filenames are expected as `p(\d+)(?:_spread)?\.(jpg|jpeg|png|webp)`,
matching nmanga's output convention. Page numbers are extracted for
human-readable pair strings in the JSON output.

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model inference (CPU-only). |
| `torchvision` | `mobilenet_v3_small` architecture definition. |
| `numpy` | Array ops for CNN preprocessing. |
| `opencv-python-headless` | Image decode, resize, strip extraction. |
| `rich-click` | CLI definition (click alternative). |
| `importlib-resources` \| stdlib | Bundled model loading. |

## Model

Ships `manga-digital.pth` (MobileNetV3-small, 2-class head) bundled alongside
the Python package. Referenced in source via
`importlib.resources.files("spreadnn") / "models" / "manga-digital.pth"`.

The model is a MobileNetV3-small with the original 1000-class classifier head
replaced by `nn.Sequential(nn.Linear(960, 1))` (single logit → sigmoid).

## Project Structure

```
spreadnn/
├── pyproject.toml
├── README.md
├── SPEC.md
├── tests/
│   ├── conftest.py
│   ├── test_detect.py
│   ├── test_cli.py
│   └── fixtures/
│       └── models/
│           └── dummy.pth          # minimal model for CI
├── src/
│   └── spreadnn/
│       ├── __init__.py
│       ├── __main__.py            # python -m spreadnn
│       ├── cli.py                 # rich-click CLI defs
│       ├── detect.py              # page_pair generator + _score_pair
│       ├── model.py               # model loading + inference
│       ├── formats.py             # JSON output formatting
│       ├── join.py                # ImageMagick join logic
│       ├── naming.py              # page number extraction
│       └── models/
│           └── manga-digital.pth  # bundled CNN weights
```

## Edge Cases

- **Odd page count**: Final single page is not scored, not joined, not in
  output.
- **Decode failure**: Both halves emitted individually; pair omitted from
  manifest.
- **Height mismatch**: Pages resized to `max(h_e, h_o)` before strip
  extraction (same as stitch-spreads).
- **Non-sequential filenames**: Files sorted lexicographically; pairs taken
  in sort order.
- **No images found**: Exit with code 1 and message to stderr.
- **No spreads detected**: Empty array `[]`.

## Future

- **Adaptive skip-pages**: Heuristic to detect frontmatter vs chapter start
  (e.g., colour pages, title pages with large text, pages with ISBN).
- **Confidence histograms**: `detect --hist` outputs score distribution.
- **Batch mode**: `detect` multiple directories, aggregate results.

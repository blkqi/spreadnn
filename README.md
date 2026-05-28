# spreadnn

## Overview

`spreadnn` detects two-page spreads in manga dumps using a
MobileNetV3-small binary classifier.  It operates on a flat directory of
page images (pNNN files), detects spreads, and writes a manifest that
downstream tools can use to mechanically join spreads without
re-encoding the source images.

Defaults to manga (right-to-left) reading order.  Pass ``--ltr`` for
western/flopped pages.  Pair alignment is auto-detected (offset 1
first, fallback to 0), or pin it with ``--offset {0,1}``.

**Not** a CBZ in/out tool — it works on loose leaf images.

## CLI

```
spreadnn detect [OPTIONS] <DIR>
spreadnn join [OPTIONS] <DIR>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold F` | 0.5 | Spread probability threshold [0–1]. |
| `--model PATH` | bundled | Override model .pth file. |
| `--ltr` | off | Left-to-right (western) reading order. |
| `--offset {0,1,auto}` | auto | Force pair alignment offset. |

### `detect`

Run ML detection on a directory of page images.  Outputs a JSON array of
``[left, right]`` pairs to stdout.  For RTL (default) the pairs are
``[even, odd]`` so joining left-to-right preserves narrative order::

```json
[[8, 7], [18, 17]]
```

Useful for piping into downstream tools (nmanga, cbtools):

```bash
spreadnn detect ./images/
```

Pass ``--manifest`` to write a ``spreads.json`` sidecar instead.
The sidecar uses the same format as stdout:

```bash
spreadnn detect ./images/ --manifest
# → ./images/spreads.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest` | off | Write spreads.json sidecar instead of stdout. |
| `--output PATH` | `<DIR>/spreads.json` | Manifest output path. |

### `join`

Mechanically join spread pairs from a manifest.  Requires a
``spreads.json`` in the directory (or pass ``--manifest``).  Does not
run ML detection — run ``spreadnn detect --manifest`` first:

```bash
spreadnn detect --manifest ./images/ && spreadnn join ./images/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--manifest PATH` | `<DIR>/spreads.json` | Path to manifest. |
| `--dry-run` | false | Print what would be joined, don't write. |
| `--quality N` | 100 | JPEG quality for joined output. |
| `--output-dir PATH` | `<DIR>` | Output directory. |
| `--no-cleanup` | false | Keep originals of joined pages. |

## Manifest Schema

```
[["p008.jpg", "p007.jpg"], ["p018.jpg", "p017.jpg"]]
```

Each inner array is ``[left_filename, right_filename]`` in joined output
order — the first file goes on the left side of the joined image, the
second on the right.  For RTL (default) the pair is ``(even_fn, odd_fn)``
so reading left-to-right gives correct narrative order; for LTR it is
``(odd_fn, even_fn)``.

## Naming Convention

Page images should follow the ``p\d+`` convention (e.g. ``p007.jpg``)
for compatibility with downstream tools like nmanga.  No filename regex
is enforced internally — the manifest stores filenames directly.

## API

```python
from spreadnn.detect import detect_spreads

pairs = detect_spreads("./images/")
# [("p008.jpg", "p007.jpg")]   (RTL default — even on left)

pairs = detect_spreads("./images/", ltr=True, offset=0)
# [("p007.jpg", "p008.jpg")]   (LTR — odd on left)
```

```python
from spreadnn.manifest import load_manifest, dump_manifest

manifest = load_manifest(Path("spreads.json"))
# [("p008.jpg", "p007.jpg")]
```
from spreadnn.model import SpreadModel

model = SpreadModel()
prob = model.score_pair(right_img, left_img)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model inference (CPU-only). |
| `torchvision` | `mobilenet_v3_small` architecture. |
| `numpy` | CNN preprocessing. |
| `opencv-python-headless` | Image decode, resize, strip extraction. |
| `rich-click` | CLI. |

## Model

Ships `manga-digital.pth` (MobileNetV3-small, binary classifier head)
bundled at `spreadnn/models/manga-digital.pth`.

## Edge Cases

- **Decode failure**: Both halves emitted individually; pair omitted.
- **Height mismatch**: Pages resized to `max(h_e, h_o)` before scoring.
- **Non-sequential filenames**: Sorted lexicographically; pairs taken
  in sort order.
- **No images found**: Exit code 1.
- **No spreads detected**: Empty array `[]`.

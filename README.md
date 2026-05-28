# spreadnn

## Overview

`spreadnn` detects two-page spreads in manga dumps using a
MobileNetV3-small binary classifier.  It operates on a flat directory of
page images (pNNN files), outputs structured JSON, and optionally runs
the ImageMagick join.

Defaults to manga (right-to-left) reading order.  Pass ``--ltr`` for
western/flopped pages.  Pair alignment is auto-detected (offset 1
first, fallback to 0), or pin it with ``--offset {0,1}``.

**Not** a CBZ in/out tool — it works on loose leaf images.

## CLI

```
spreadnn detect [OPTIONS] <DIR>
spreadnn manifest [OPTIONS] <DIR>
spreadnn join [OPTIONS] <DIR>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--threshold F` | 0.5 | Spread probability threshold [0–1]. |
| `--model PATH` | bundled | Override model .pth file. |
| `--ltr` | off | Left-to-right (western) reading order. |
| `--offset {0,1,auto}` | auto | Force pair alignment offset. |

### `detect`

Output a JSON array of `[start, end]` page-number pairs:

```json
[[7, 8], [17, 18]]
```

```bash
spreadnn detect ./images/
```

### `manifest`

Write a `spreads.json` sidecar into the image directory:

```bash
spreadnn manifest ./images/
# → ./images/spreads.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | `<DIR>/spreads.json` | Output path for manifest. |
| `--no-write` | false | Print manifest to stdout instead. |

### `join`

Detect + ImageMagick `+append` in one pass:

```bash
spreadnn join ./images/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | false | Print what would be joined, don't write. |
| `--quality N` | 100 | JPEG quality for joined output. |
| `--output-dir PATH` | `<DIR>` | Output directory. |
| `--no-cleanup` | false | Keep originals of joined pages. |

## Output Schema

```
[[7, 8], [17, 18]]
```

Each inner array is `[first_page, next_page]` in reading order.

## File Name Convention

Input filenames: `p(\d+)\.(jpg|jpeg|png|webp)`.  Page numbers are
extracted from matches for pair output.

## API

```python
from spreadnn.detect import detect_spreads

pairs = detect_spreads("./images/")
# [(7, 8), (17, 18)]

pairs = detect_spreads("./images/", ltr=True, offset=0)
# western pages, offset 0 (no page skipped)
```

```python
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

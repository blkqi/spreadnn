# spreadnn

## Overview

`spreadnn` detects two-page spreads in manga dumps using a
MobileNetV3-small binary classifier.  It operates on a flat directory of
page images (pNNN files), outputs structured JSON, and optionally runs
the ImageMagick join.

**Not** a CBZ in/out tool — it works on loose leaf images.

## CLI

```
spreadnn detect [OPTIONS] <DIR>
spreadnn manifest [OPTIONS] <DIR>
spreadnn join [OPTIONS] <DIR>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--skip-pages N` | 1 | Passthrough leading N pages (covers, ToCs). |
| `--threshold F` | 0.5 | Spread probability threshold [0–1]. |
| `--model PATH` | bundled | Override model .pth file. |

### `detect`

Output a JSON array of `[start, end]` page-number pairs:

```json
[[7, 8], [17, 18]]
```

```bash
spreadnn detect --skip-pages 0 ./images/
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
spreadnn join --skip-pages 1 ./images/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | false | Print what would be joined, don't write. |
| `--quality N` | 100 | JPEG quality for joined output. |
| `--output-dir PATH` | `<DIR>` | Output directory. |
| `--no-cleanup` | false | Keep originals of joined pages. |

## Detection Algorithm

1. Resize both pages to the same height (max of the two).
2. Extract a 256×64 strip centred on the inner edge (32 px from each page).
3. Normalise with ImageNet stats (mean=[0.485,0.456,0.406],
   std=[0.229,0.224,0.225]).
4. Run through MobileNetV3-small with binary classification head.
5. Apply sigmoid → spread probability.

## Output Schema

```
[[7, 8], [17, 18]]
```

Each inner array is `[left_page, right_page]` in reading order.

## File Name Convention

Input filenames: `p(\d+)\.(jpg|jpeg|png|webp)`.  Page numbers are
extracted from matches for pair output.

## API

```python
from spreadnn.detect import detect_spreads

pairs = detect_spreads("./images/", skip_pages=1)
# [(7, 8), (17, 18)]
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

- **Odd page count**: Final single page is not scored or joined.
- **Decode failure**: Both halves emitted individually; pair omitted.
- **Height mismatch**: Pages resized to `max(h_e, h_o)` before scoring.
- **Non-sequential filenames**: Sorted lexicographically; pairs taken
  in sort order.
- **No images found**: Exit code 1.
- **No spreads detected**: Empty array `[]`.

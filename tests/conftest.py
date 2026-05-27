from pathlib import Path

import pytest


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a directory with 6 test JPEG images (p001-p006)."""
    from PIL import Image
    d = tmp_path / "images"
    d.mkdir()
    for i in range(1, 7):
        img = Image.new("RGB", (200, 400), (255, 255, 255))
        img.save(d / f"p{i:03d}.jpg")
    return d

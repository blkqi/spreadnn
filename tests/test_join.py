from pathlib import Path
from unittest.mock import patch

import numpy as np

from spreadnn.join import join_pairs


def _dummy_img(mock_imread):
    """Make cv2.imread return a 100x100 dummy BGR image."""
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)


class TestJoinPairs:
    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_output_filename_reading_order(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Filename uses sorted stems (reading order), not visual +append order."""
        _dummy_img(mock_imread)
        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()

        pairs = [(d / "p008.jpg", d / "p007.jpg")]

        written = join_pairs(d, pairs, dry_run=False, no_cleanup=True)
        assert len(written) == 1
        assert written[0].name == "p007-p008.jpg"

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_output_filename_reverse(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Same result for reversed visual order."""
        _dummy_img(mock_imread)
        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()

        pairs = [(d / "p007.jpg", d / "p008.jpg")]

        written = join_pairs(d, pairs, dry_run=False, no_cleanup=True)
        assert written[0].name == "p007-p008.jpg"

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_originals_deleted(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Originals are removed after join when no-cleanup is off."""
        _dummy_img(mock_imread)
        d = tmp_path / "images"
        d.mkdir()
        for f in ("p001.jpg", "p002.jpg"):
            (d / f).touch()

        written = join_pairs(d, [(d / "p001.jpg", d / "p002.jpg")], dry_run=False, no_cleanup=False)
        assert len(written) == 1
        assert not (d / "p001.jpg").exists()
        assert not (d / "p002.jpg").exists()

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_height_matched(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Images with different heights are resized before joining."""
        d = tmp_path / "images"
        d.mkdir()
        for f in ("p001.jpg", "p002.jpg"):
            (d / f).touch()

        # Return images with different heights
        def side_effect(path):
            stem = Path(path).stem
            h = 200 if stem == "p001" else 300
            return np.zeros((h, 100, 3), dtype=np.uint8)
        mock_imread.side_effect = side_effect

        written = join_pairs(d, [(d / "p001.jpg", d / "p002.jpg")], dry_run=False, no_cleanup=True)
        assert len(written) == 1

        # Both images should have been resized to max height (300), then hconcat
        args, _ = mock_imwrite.call_args
        written_img = args[1]
        assert written_img.shape[0] == 300  # height = max(200, 300)
        assert written_img.shape[1] == 200  # width = 100 + 100

    def test_dry_run_returns_empty(self, tmp_path: Path):
        d = tmp_path / "images"
        d.mkdir()
        (d / "p001.jpg").touch()
        (d / "p002.jpg").touch()
        written = join_pairs(d, [(d / "p001.jpg", d / "p002.jpg")], dry_run=True)
        assert written == []
        assert (d / "p001.jpg").exists()  # not deleted
        assert (d / "p002.jpg").exists()


class TestJoinWithManifest:
    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_auto_detect_manifest(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Join uses spreads.json when it exists in the directory."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()
        (d / "spreads.json").write_text('[["p007.jpg", "p008.jpg"]]\n')

        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        runner = CliRunner()
        result = runner.invoke(join, [str(d), "-v"])

        assert result.exit_code == 0
        assert "using manifest" in result.output
        assert "joined 1 spread" in result.output

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_explicit_manifest_path(self, mock_imread, mock_imwrite, tmp_path: Path):
        """--manifest flag overrides auto-detect."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()

        mf = tmp_path / "custom.json"
        mf.write_text('[["p007.jpg", "p008.jpg"]]\n')

        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        runner = CliRunner()
        result = runner.invoke(join, [str(d), "--manifest", str(mf), "-v"])

        assert result.exit_code == 0
        assert "using manifest" in result.output
        assert "joined 1 spread" in result.output

    @patch("spreadnn.detect.detect")
    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_errors_without_manifest(self, mock_imread, mock_imwrite, mock_detect, tmp_path: Path):
        """Join exits with error when no manifest exists."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        (d / "p001.jpg").touch()

        runner = CliRunner()
        result = runner.invoke(join, [str(d)])

        assert result.exit_code == 1
        assert "run 'spreadnn detect --manifest' first" in result.output
        mock_detect.assert_not_called()

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_manifest_reversed_visual_order(self, mock_imread, mock_imwrite, tmp_path: Path):
        """Manifest with reversed visual order joins correctly."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()
        (d / "spreads.json").write_text('[["p007.jpg", "p008.jpg"]]\n')

        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        runner = CliRunner()
        result = runner.invoke(join, [str(d), "-v"])

        assert result.exit_code == 0
        assert "joined 1 spread" in result.output

    @patch("spreadnn.join.cv2.imwrite", return_value=True)
    @patch("spreadnn.join.cv2.imread")
    def test_manifest_dry_run(self, mock_imread, mock_imwrite, tmp_path: Path):
        """--dry-run works with manifest-based joins."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()
        (d / "spreads.json").write_text('[["p007.jpg", "p008.jpg"]]\n')

        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        runner = CliRunner()
        result = runner.invoke(join, [str(d), "--dry-run", "-v"])

        assert result.exit_code == 0
        assert "dry-run" in result.output
        assert "p007" in result.output
        assert "p008" in result.output

    @patch("spreadnn.join.join_pair")
    def test_manifest_rtl_even_on_left(self, mock_join_pair, tmp_path: Path):
        """RTL manifest puts even page on left of joined output."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()
        (d / "spreads.json").write_text('[["p008.jpg", "p007.jpg"]]\n')

        runner = CliRunner()
        runner.invoke(join, [str(d), "-v"])

        assert mock_join_pair.called
        left, right, _ = mock_join_pair.call_args[0]
        assert left.name == "p008.jpg", f"expected p008.jpg on left (RTL), got {left.name}"
        assert right.name == "p007.jpg", f"expected p007.jpg on right (RTL), got {right.name}"

    @patch("spreadnn.join.join_pair")
    def test_manifest_reversed_odd_on_left(self, mock_join_pair, tmp_path: Path):
        """Reversed manifest puts odd page on left of joined output."""
        from spreadnn.cli import join
        from click.testing import CliRunner

        d = tmp_path / "images"
        d.mkdir()
        for f in ("p007.jpg", "p008.jpg"):
            (d / f).touch()
        (d / "spreads.json").write_text('[["p007.jpg", "p008.jpg"]]\n')

        runner = CliRunner()
        runner.invoke(join, [str(d), "-v"])

        assert mock_join_pair.called
        left, right, _ = mock_join_pair.call_args[0]
        assert left.name == "p007.jpg", f"expected p007.jpg on left (LTR), got {left.name}"
        assert right.name == "p008.jpg", f"expected p008.jpg on right (LTR), got {right.name}"

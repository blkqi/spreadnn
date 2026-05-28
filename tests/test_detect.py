from pathlib import Path

import pytest
from spreadnn.detect import merged_pairs
from spreadnn.naming import extract_page_num


class TestMergedPairs:
    def test_empty(self):
        assert merged_pairs([]) == []

    def test_single_merged(self):
        from spreadnn.detect import DetectionResult
        results = [
            DetectionResult(even="p007.jpg", odd="p008.jpg", score=0.99, merged=True),
        ]
        assert merged_pairs(results) == ["7-8"]

    def test_mixed(self):
        from spreadnn.detect import DetectionResult
        results = [
            DetectionResult(even="p001.jpg", odd="p002.jpg", score=0.01, merged=False),
            DetectionResult(even="p007.jpg", odd="p008.jpg", score=0.99, merged=True),
            DetectionResult(even="p017.jpg", odd="p018.jpg", score=0.99, merged=True),
        ]
        assert merged_pairs(results) == ["7-8", "17-18"]


class TestDetectParameters:
    """detect() and detect_spreads() accept ltr and offset parameters."""

    def test_accepts_ltr(self, sample_dir: Path):
        from spreadnn.detect import detect
        results = detect(sample_dir, ltr=True)
        assert isinstance(results, list)
        assert all(hasattr(r, "even") for r in results)

    def test_accepts_offset_0(self, sample_dir: Path):
        from spreadnn.detect import detect
        results = detect(sample_dir, offset=0)
        assert len(results) == 3  # p001-p002, p003-p004, p005-p006

    def test_accepts_offset_1(self, sample_dir: Path):
        from spreadnn.detect import detect
        results = detect(sample_dir, offset=1)
        assert len(results) == 2  # p002-p003, p004-p005; p006 is lone tail

    def test_accepts_both(self, sample_dir: Path):
        from spreadnn.detect import detect
        results = detect(sample_dir, ltr=True, offset=0)
        assert len(results) == 3

    def test_spreads_accepts_ltr(self, sample_dir: Path):
        from spreadnn.detect import detect_spreads
        pairs = detect_spreads(sample_dir, ltr=True)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_spreads_accepts_offset(self, sample_dir: Path):
        from spreadnn.detect import detect_spreads
        pairs = detect_spreads(sample_dir, offset=0)
        assert isinstance(pairs, list)

    def test_offset_skips_auto(self, sample_dir: Path):
        """offset != None should not fall through to the auto-detect path."""
        from spreadnn.detect import detect, _detect_at_offset
        import spreadnn.detect as dmod
        original = dmod._detect_at_offset
        calls = []
        def tracking_fn(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)
        dmod._detect_at_offset = tracking_fn
        try:
            detect(sample_dir, offset=0)
            assert len(calls) == 1  # called exactly once, no fallback
        finally:
            dmod._detect_at_offset = original

    def test_invalid_dir(self):
        from spreadnn.detect import detect_spreads
        with pytest.raises((ValueError, FileNotFoundError)):
            detect_spreads("/nonexistent")

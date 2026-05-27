from pathlib import Path

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

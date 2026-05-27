from pathlib import Path

from spreadnn.naming import extract_page_num, page_num, pair_str


class TestExtractPageNum:
    def test_simple(self):
        assert extract_page_num(Path("p001.jpg")) == 1
        assert extract_page_num(Path("p123.jpg")) == 123

    def test_with_spread_suffix(self):
        assert extract_page_num(Path("p005_spread.jpg")) == 5

    def test_case_insensitive(self):
        assert extract_page_num(Path("P010.JPG")) == 10
        assert extract_page_num(Path("p001.PNG")) == 1

    def test_no_match(self):
        assert extract_page_num(Path("page001.jpg")) is None
        assert extract_page_num(Path("001.jpg")) is None
        assert extract_page_num(Path("cover.jpg")) is None


class TestPageNum:
    def test_valid(self):
        assert page_num(Path("p042.jpg")) == 42

    def test_invalid(self):
        import pytest
        with pytest.raises(ValueError):
            page_num(Path("foo.jpg"))


class TestPairStr:
    def test_format(self):
        assert pair_str(7, 8) == "7-8"
        assert pair_str(1, 2) == "1-2"

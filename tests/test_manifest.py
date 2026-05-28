from pathlib import Path

from spreadnn.manifest import dump_manifest, load_manifest, resolve_manifest_pairs


class TestLoadManifest:
    def test_loads_filename_pairs(self, tmp_path: Path):
        mf = tmp_path / "spreads.json"
        mf.write_text('[["p008.jpg", "p007.jpg"]]\n')
        assert load_manifest(mf) == [("p008.jpg", "p007.jpg")]

    def test_empty(self, tmp_path: Path):
        mf = tmp_path / "spreads.json"
        mf.write_text("[]\n")
        assert load_manifest(mf) == []


class TestDumpManifest:
    def test_writes_json(self, tmp_path: Path):
        mf = tmp_path / "spreads.json"
        dump_manifest([("p008.jpg", "p007.jpg")], mf)
        assert mf.read_text() == '[["p008.jpg", "p007.jpg"]]\n'

    def test_roundtrip(self, tmp_path: Path):
        mf = tmp_path / "spreads.json"
        original = [("p008.jpg", "p007.jpg")]
        dump_manifest(original, mf)
        assert load_manifest(mf) == original


class TestResolveManifestPairs:
    def test_resolves_filenames(self, tmp_path: Path):
        d = tmp_path / "images"
        d.mkdir()
        for f in ("p008.jpg", "p007.jpg", "p018.jpg", "p017.jpg"):
            (d / f).touch()

        pairs = resolve_manifest_pairs(d, [("p008.jpg", "p007.jpg"), ("p018.jpg", "p017.jpg")])
        assert len(pairs) == 2
        assert pairs[0] == ("p008.jpg", "p007.jpg")
        assert pairs[1] == ("p018.jpg", "p017.jpg")

    def test_missing_files_skipped(self, tmp_path: Path):
        d = tmp_path / "images"
        d.mkdir()
        (d / "p008.jpg").touch()

        pairs = resolve_manifest_pairs(d, [("p008.jpg", "p007.jpg"), ("p018.jpg", "p017.jpg")])
        assert len(pairs) == 0

    def test_only_first_missing(self, tmp_path: Path):
        d = tmp_path / "images"
        d.mkdir()
        (d / "p008.jpg").touch()

        pairs = resolve_manifest_pairs(d, [("p007.jpg", "p008.jpg")])
        assert len(pairs) == 0

    def test_nonexistent_files_ignored(self, tmp_path: Path):
        d = tmp_path / "images"
        d.mkdir()
        (d / "p008.jpg").touch()
        (d / "p007.jpg").touch()
        (d / "cover.jpg").touch()
        (d / "spreads.json").touch()

        pairs = resolve_manifest_pairs(d, [("p008.jpg", "p007.jpg")])
        assert len(pairs) == 1
        assert pairs[0] == ("p008.jpg", "p007.jpg")

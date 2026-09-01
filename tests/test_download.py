import io

import pytest

from urbanpulse import download as module


class Response(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(length if length is not None else len(data))}


def test_existing_raw_file_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "raw.parquet"
    path.write_bytes(b"original")
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: pytest.fail("Must not fetch existing raw file"))
    module.fetch("https://example.invalid/data", path)
    assert path.read_bytes() == b"original"


def test_partial_download_not_promoted(tmp_path, monkeypatch):
    path = tmp_path / "raw.parquet"
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response(b"short", length=100))
    with pytest.raises(ValueError, match="不完整"):
        module.fetch("https://example.invalid/data", path)
    assert not path.exists()


def test_complete_download_promoted(tmp_path, monkeypatch):
    path = tmp_path / "raw.parquet"
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: Response(b"complete"))
    module.fetch("https://example.invalid/data", path)
    assert path.read_bytes() == b"complete"
    assert not path.with_suffix(".parquet.part").exists()

"""Filename sanitisation and size enforcement.

The property that matters for the size helpers is *when* the limit is checked: a
cap enforced after the upload has already been read into memory protects
nothing, because the memory is already spent.
"""

from __future__ import annotations

import io

import pytest
from fastapi import UploadFile

from prelabel.errors import PayloadTooLarge
from prelabel.uploads import enforce_file_count, read_capped, safe_filename, save_streaming


def make_upload(name: str | None, data: bytes = b"") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data))


class CountingStream(io.BytesIO):
    """Records how many bytes were actually pulled out of the stream."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


# --- filenames --------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("model.pt", "model.pt"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.pt", "evil.pt"),
        ("C:\\Windows\\System32\\evil.pt", "evil.pt"),
        ("a/b/c.onnx", "c.onnx"),
        ("with spaces.pt", "with spaces.pt"),
        ("写真.jpg", "写真.jpg"),          # non-Latin names survive intact
        ("Фото-1.png", "Фото-1.png"),
        ("pipe|name.pt", "pipe_name.pt"),
        ("trailing.  ", "trailing"),
    ],
)
def test_safe_filename_reduces_to_one_component(given, expected):
    assert safe_filename(given) == expected


def test_safe_filename_handles_a_missing_name():
    """A client may omit the filename entirely; that must not be a 500."""
    result = safe_filename(None, ".mp4")
    assert result.startswith("upload-") and result.endswith(".mp4")


def test_safe_filename_handles_a_name_that_sanitises_to_nothing():
    assert safe_filename("...").startswith("upload-")


def test_safe_filename_never_escapes_its_directory(tmp_path):
    for hostile in ("../../escape.pt", "..\\..\\escape.pt", "/etc/passwd", "C:\\evil.pt"):
        destination = tmp_path / safe_filename(hostile)
        assert destination.parent == tmp_path


# --- size caps --------------------------------------------------------------


def test_read_capped_returns_data_under_the_limit():
    assert read_capped(make_upload("a.jpg", b"x" * 100), max_mb=1) == b"x" * 100


def test_read_capped_stops_before_buffering_the_whole_upload():
    """The cap must abort mid-stream, not after everything is in memory."""
    payload = b"x" * (5 * 1024 * 1024)
    stream = CountingStream(payload)
    upload = UploadFile(filename="big.jpg", file=stream)

    with pytest.raises(PayloadTooLarge):
        read_capped(upload, max_mb=1)

    assert stream.bytes_read < len(payload), "the whole upload was read despite the cap"
    assert stream.bytes_read <= 2 * 1024 * 1024


def test_read_capped_with_no_limit_reads_everything():
    payload = b"x" * (3 * 1024 * 1024)
    assert len(read_capped(make_upload("a.bin", payload), max_mb=0)) == len(payload)


def test_save_streaming_writes_the_file(tmp_path):
    destination = save_streaming(make_upload("model.pt", b"weights"), tmp_path, max_mb=10)
    assert destination.read_bytes() == b"weights"
    assert destination.parent == tmp_path


def test_save_streaming_removes_the_partial_file_when_the_cap_trips(tmp_path):
    """A rejected upload must not leave a truncated file behind."""
    destination = tmp_path / "uploads"
    with pytest.raises(PayloadTooLarge):
        save_streaming(make_upload("big.pt", b"\0" * (3 * 1024 * 1024)), destination, max_mb=1)
    assert list(destination.iterdir()) == []


def test_save_streaming_sanitises_the_destination(tmp_path):
    destination = save_streaming(make_upload("../../evil.pt", b"x"), tmp_path, max_mb=10)
    assert destination.parent == tmp_path
    assert destination.name == "evil.pt"


# --- counts -----------------------------------------------------------------


def test_enforce_file_count_accepts_within_the_limit():
    uploads = [make_upload(f"{i}.jpg") for i in range(3)]
    assert len(enforce_file_count(uploads, 5, "images")) == 3


def test_enforce_file_count_rejects_too_many():
    uploads = [make_upload(f"{i}.jpg") for i in range(6)]
    with pytest.raises(PayloadTooLarge):
        enforce_file_count(uploads, 5, "images")


def test_enforce_file_count_rejects_an_empty_request():
    with pytest.raises(PayloadTooLarge):
        enforce_file_count([], 5, "images")


def test_enforce_file_count_with_no_limit():
    uploads = [make_upload(f"{i}.jpg") for i in range(50)]
    assert len(enforce_file_count(uploads, 0, "images")) == 50

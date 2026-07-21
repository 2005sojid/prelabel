"""Cross-validation of the browser's ZIP writer against a real implementation.

The JS unit tests (``tests/js/zip.test.mjs``) check the bytes we *intend* to
write. This checks that a different, independent ZIP implementation — Python's
``zipfile`` — agrees: that the archive opens, the CRCs verify, the names decode
and the contents round-trip.

That is the property that actually matters. A hand-rolled archive writer can
produce a file that satisfies every assertion about its own layout and still be
rejected by the tool the user opens it with.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ZIP_MODULE = ROOT / "frontend" / "assets" / "js" / "export" / "zip.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def build_zip(entries: list[dict]) -> bytes:
    """Build an archive with the *frontend's* writer and return its bytes."""
    # A file:// URL, not a bare path: Node's ESM loader rejects absolute Windows
    # paths ("C:/...") because it reads the drive letter as a URL scheme.
    script = f"""
import {{ makeZip }} from {json.dumps(ZIP_MODULE.as_uri())};

const entries = {json.dumps(entries)}.map((item) => ({{
  name: item.name,
  data: new TextEncoder().encode(item.text),
}}));
const buffer = await makeZip(entries).arrayBuffer();
process.stdout.write(Buffer.from(buffer).toString("base64"));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        capture_output=True,
        check=True,
        cwd=ROOT,
    )
    return base64.b64decode(completed.stdout)


def open_zip(entries: list[dict]) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(build_zip(entries)))


def test_archive_opens_and_passes_its_own_integrity_check():
    with open_zip([{"name": "a.txt", "text": "hello"}, {"name": "b.txt", "text": "world"}]) as archive:
        assert archive.testzip() is None, "a CRC did not verify"
        assert archive.namelist() == ["a.txt", "b.txt"]


def test_contents_round_trip():
    with open_zip([{"name": "note.txt", "text": "line one\nline two"}]) as archive:
        assert archive.read("note.txt").decode("utf-8") == "line one\nline two"


def test_non_latin_filenames_decode_correctly():
    """Without the UTF-8 flag these come back as mojibake."""
    names = ["写真.jpg", "Фото.png", "café.webp", "日本語のファイル.txt"]
    with open_zip([{"name": name, "text": "x"} for name in names]) as archive:
        assert archive.namelist() == names
        for info in archive.infolist():
            assert info.flag_bits & 0x800, f"{info.filename} is missing the UTF-8 flag"


def test_nested_paths_are_preserved():
    entries = [
        {"name": "images/default/a.jpg", "text": "a"},
        {"name": "annotations/instances_default.json", "text": "{}"},
    ]
    with open_zip(entries) as archive:
        assert set(archive.namelist()) == {
            "images/default/a.jpg",
            "annotations/instances_default.json",
        }


def test_duplicate_names_survive_as_distinct_entries():
    """Two files with the same name must both be recoverable, not one silently lost."""
    entries = [
        {"name": "img.jpg", "text": "first"},
        {"name": "img.jpg", "text": "second"},
    ]
    with open_zip(entries) as archive:
        assert archive.namelist() == ["img.jpg", "img (2).jpg"]
        assert archive.read("img.jpg") == b"first"
        assert archive.read("img (2).jpg") == b"second"


def test_timestamps_are_valid():
    """A zeroed DOS timestamp decodes to an invalid date and upsets some tools."""
    with open_zip([{"name": "a.txt", "text": "x"}]) as archive:
        year = archive.infolist()[0].date_time[0]
        assert year >= 1980


def test_entries_are_stored_not_deflated():
    """Images are already compressed; deflating them would cost CPU for nothing."""
    with open_zip([{"name": "a.txt", "text": "x" * 500}]) as archive:
        info = archive.infolist()[0]
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.compress_size == info.file_size


def test_an_empty_archive_is_valid():
    with open_zip([]) as archive:
        assert archive.namelist() == []

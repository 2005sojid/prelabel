"""Reading image folders from the server's filesystem.

This is the one place the server opens paths a client asked for, so the tests
that matter most are the ones that try to get out of the sandbox.
"""

from __future__ import annotations

import pytest

from prelabel import config, datasets
from prelabel.errors import DatasetAccessError

# --- the guard --------------------------------------------------------------


def test_disabled_until_a_root_is_configured(tmp_path, monkeypatch):
    """Reading server-side paths is off by default, and says how to turn it on."""
    monkeypatch.setattr(config, "DATA_ROOTS", [])
    with pytest.raises(DatasetAccessError, match="PL_DATA_ROOTS"):
        datasets.resolve_dataset_dir(str(tmp_path))
    assert datasets.is_configured() is False


def test_an_allowed_directory_resolves(dataset_root):
    assert datasets.resolve_dataset_dir(str(dataset_root)) == dataset_root


def test_a_subdirectory_of_a_root_is_allowed(dataset_root):
    assert datasets.resolve_dataset_dir(str(dataset_root / "sub")) == dataset_root / "sub"


def test_a_path_outside_every_root_is_refused(dataset_root, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(DatasetAccessError, match="outside the allowed"):
        datasets.resolve_dataset_dir(str(outside))


def test_traversal_out_of_a_root_is_refused(dataset_root):
    """``..`` is resolved before the check, so it cannot be used to escape."""
    with pytest.raises(DatasetAccessError, match="outside the allowed"):
        datasets.resolve_dataset_dir(str(dataset_root / ".." / ".."))


@pytest.mark.parametrize("hostile", ["", "   ", "\x00"])
def test_empty_and_malformed_paths_are_refused(dataset_root, hostile):
    with pytest.raises(DatasetAccessError):
        datasets.resolve_dataset_dir(hostile)


def test_a_missing_directory_says_so(dataset_root):
    with pytest.raises(DatasetAccessError, match="does not exist"):
        datasets.resolve_dataset_dir(str(dataset_root / "nope"))


def test_a_file_is_not_a_directory(dataset_root):
    with pytest.raises(DatasetAccessError, match="not a directory"):
        datasets.resolve_dataset_dir(str(dataset_root / "a.jpg"))


# --- scanning ---------------------------------------------------------------


def test_scan_finds_images_recursively_and_skips_other_files(dataset_root):
    found = {ref.rel_path for ref in datasets.scan_images(dataset_root)}
    assert found == {"a.jpg", "b.jpg", "c.png", "sub/nested.jpg"}


def test_scan_order_is_stable(dataset_root):
    """Item numbering has to be reproducible, or export ids move between runs."""
    first = [ref.rel_path for ref in datasets.scan_images(dataset_root)]
    second = [ref.rel_path for ref in datasets.scan_images(dataset_root)]
    assert first == second == sorted(first)


def test_scan_skips_empty_files(dataset_root):
    (dataset_root / "empty.jpg").write_bytes(b"")
    assert "empty.jpg" not in {ref.rel_path for ref in datasets.scan_images(dataset_root)}


def test_scan_respects_a_limit(dataset_root):
    assert len(datasets.scan_images(dataset_root, limit=2)) == 2


def test_scan_skips_noise_directories(dataset_root):
    noisy = dataset_root / "__pycache__"
    noisy.mkdir()
    (noisy / "cached.jpg").write_bytes(b"x" * 100)
    assert all("__pycache__" not in ref.rel_path for ref in datasets.scan_images(dataset_root))


# --- item paths -------------------------------------------------------------


def test_item_path_resolves_inside_the_root(dataset_root):
    assert datasets.resolve_item_path(dataset_root, "sub/nested.jpg").exists()


def test_item_path_cannot_escape_the_root(dataset_root, tmp_path):
    """The relative path comes from the database, which the API can write to."""
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    with pytest.raises(DatasetAccessError, match="escapes"):
        datasets.resolve_item_path(dataset_root, "../secret.txt")


def test_item_path_reports_a_deleted_file(dataset_root):
    with pytest.raises(DatasetAccessError, match="no longer on disk"):
        datasets.resolve_item_path(dataset_root, "gone.jpg")

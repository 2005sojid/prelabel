"""Turning an upload directory into a loadable target.

The split matters: :func:`inspect` is read-only, so the API can decide whether an
upload is usable *before* touching anything. That is what lets an incomplete
multi-part upload be accepted without disturbing the running model.
"""

from __future__ import annotations

import time

import pytest

from prelabel import model_loader
from prelabel.errors import ModelLoadError


def touch(directory, name, data=b"x"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(data)
    return path


# --- primary detection ------------------------------------------------------


@pytest.mark.parametrize("name", ["a.pt", "a.onnx", "a.engine", "a.torchscript", "a.tflite", "a.xml", "A.PT"])
def test_primary_files_start_a_fresh_load(name):
    assert model_loader.is_primary(name) is True


@pytest.mark.parametrize("name", ["a.bin", "metadata.yaml", "labels.txt", "", None, "readme.md"])
def test_companion_and_unknown_files_do_not(name):
    assert model_loader.is_primary(name) is False


# --- inspect ----------------------------------------------------------------


def test_empty_directory_is_not_ready(tmp_path):
    plan = model_loader.inspect(tmp_path)
    assert plan.is_ready is False
    assert plan.status == "waiting"


def test_missing_directory_is_not_ready(tmp_path):
    assert model_loader.inspect(tmp_path / "nope").is_ready is False


def test_single_file_model_is_ready(tmp_path):
    touch(tmp_path, "model.pt")
    plan = model_loader.inspect(tmp_path)
    assert plan.is_ready and plan.primary.name == "model.pt"


def test_xml_without_bin_waits_and_says_why(tmp_path):
    touch(tmp_path, "net.xml")
    plan = model_loader.inspect(tmp_path)
    assert plan.is_ready is False
    assert ".bin" in plan.message and "net.xml" in plan.message


def test_xml_with_bin_is_ready(tmp_path):
    touch(tmp_path, "net.xml")
    touch(tmp_path, "net.bin")
    assert model_loader.inspect(tmp_path).primary.name == "net.xml"


def test_companions_alone_are_not_ready(tmp_path):
    touch(tmp_path, "weights.bin")
    touch(tmp_path, "metadata.yaml")
    plan = model_loader.inspect(tmp_path)
    assert plan.is_ready is False
    assert ".bin" in plan.message


def test_the_most_recent_primary_wins(tmp_path):
    """Re-uploading must switch models rather than keep the old one."""
    touch(tmp_path, "old.pt")
    time.sleep(0.01)
    newer = touch(tmp_path, "new.onnx")
    import os

    os.utime(newer, (time.time() + 10, time.time() + 10))
    assert model_loader.inspect(tmp_path).primary.name == "new.onnx"


def test_inspect_does_not_mutate_the_directory(tmp_path):
    touch(tmp_path, "net.xml")
    touch(tmp_path, "net.bin")
    before = sorted(p.name for p in tmp_path.iterdir())
    model_loader.inspect(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --- prepare ----------------------------------------------------------------


def test_prepare_returns_a_single_file_unchanged(tmp_path):
    path = touch(tmp_path, "model.pt")
    assert model_loader.prepare(tmp_path) == path


def test_prepare_raises_when_not_ready(tmp_path):
    touch(tmp_path, "net.xml")
    with pytest.raises(ModelLoadError, match=r"\.bin"):
        model_loader.prepare(tmp_path)


def test_assemble_openvino_requires_the_bin(tmp_path):
    xml = touch(tmp_path, "net.xml")
    with pytest.raises(ModelLoadError):
        model_loader.assemble_openvino(xml)


def test_assemble_openvino_uses_a_supplied_metadata_file(tmp_path):
    """A user-provided metadata.yaml must win over a synthesised one."""
    touch(tmp_path, "net.xml", b"<net/>")
    touch(tmp_path, "net.bin", b"\0")
    touch(tmp_path, "metadata.yaml", b"task: segment\n")

    ov_dir = model_loader.assemble_openvino(tmp_path / "net.xml")

    assert ov_dir.name == "net_openvino_model"
    assert (ov_dir / "net.xml").exists()
    assert (ov_dir / "net.bin").exists()
    assert "segment" in (ov_dir / "metadata.yaml").read_text(encoding="utf-8")
    # No marker: the task came from the user, not from a guess.
    assert not (ov_dir / model_loader.TASK_ASSUMED_MARKER).exists()

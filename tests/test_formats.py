"""Integration: every supported format must load *and* infer correctly.

These export a tiny model to each format at a **non-default image size** and
assert the engine resolves that size and runs without a shape mismatch. Exporting
at 640 would pass even if the size resolution were broken, because 640 is also
the fallback — which is exactly how that bug survived.

Needs ``ultralytics`` and downloads a ~6 MB ``yolov8n`` once. The module skips
itself when neither the package nor the weights are available.

Run just these:  ``pytest tests/test_formats.py -v -m integration``
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from prelabel import model_loader
from prelabel.engines.base import InferenceResult
from prelabel.engines.factory import build_engine

pytestmark = pytest.mark.integration

IMGSZ = 768  # deliberately NOT the 640 default, to catch size-resolution bugs

EXPORT_FORMATS = ["torchscript", "onnx", "openvino"]


@pytest.fixture(scope="session")
def base_model(tmp_path_factory) -> Path:
    """A small YOLO checkpoint copied into a temp dir (exports land beside it)."""
    ultralytics = pytest.importorskip("ultralytics")
    work = tmp_path_factory.mktemp("models")
    destination = work / "model.pt"
    try:
        model = ultralytics.YOLO("yolov8n.pt")  # downloads to cwd if missing
        shutil.copy(model.ckpt_path, destination)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not obtain base model: {exc}")
    return destination


@pytest.fixture(scope="session")
def exported(base_model, tmp_path_factory):
    """Export once per format, shared across the tests that need it."""
    from ultralytics import YOLO

    results = {}
    for fmt in EXPORT_FORMATS:
        try:
            results[fmt] = Path(YOLO(str(base_model)).export(format=fmt, imgsz=IMGSZ))
        except Exception as exc:  # noqa: BLE001 - optional exporter dependency
            results[fmt] = exc
    return results


def _target(exported, fmt: str) -> Path:
    result = exported[fmt]
    if isinstance(result, Exception):
        pytest.skip(f"{fmt} export unavailable: {result}")
    return result


def _assert_runs(engine, expected_imgsz: int = IMGSZ) -> None:
    assert engine.imgsz == expected_imgsz, (
        f"engine resolved imgsz={engine.imgsz}, expected {expected_imgsz}"
    )
    # An arbitrarily-sized BGR frame — the engine must letterbox to its own size.
    image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = engine.predict(image, conf=0.25)

    assert isinstance(result, InferenceResult)
    assert result.timings.inference_ms >= 0.0
    json.dumps(result.to_dict())  # must be serialisable — this is what the API returns


@pytest.mark.parametrize("fmt", EXPORT_FORMATS)
def test_exported_format_loads_and_infers(exported, fmt):
    engine = build_engine(str(_target(exported, fmt)))
    try:
        _assert_runs(engine)
    finally:
        engine.close()


@pytest.mark.parametrize("fmt", EXPORT_FORMATS)
def test_exported_format_predicts_a_batch(exported, fmt):
    """A batch of several images must work for *every* format.

    Regression for the batch-gallery failure on static-batch exports: OpenVINO
    and Ultralytics' default ONNX export bake in ``batch=1``, so feeding a
    multi-image tensor in one forward pass raises a shape mismatch. The engine
    must take another route and still return one result per image, in order.
    """
    engine = build_engine(str(_target(exported, fmt)))
    try:
        images = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(3)]
        results = engine.predict_batch(images, conf=0.25)
        assert len(results) == len(images)
        assert all(isinstance(result, InferenceResult) for result in results)
    finally:
        engine.close()


def test_pytorch_honours_an_explicit_imgsz(base_model):
    """PyTorch input is dynamic, so an explicit override must be respected."""
    engine = build_engine(str(base_model), imgsz=IMGSZ)
    try:
        _assert_runs(engine)
    finally:
        engine.close()


def test_openvino_from_bare_xml_and_bin(exported, tmp_path):
    """The exact failure case: only .xml + .bin, no metadata.yaml.

    The loader must assemble the model, recover the 768 input size from the IR,
    and infer without a tensor shape mismatch.
    """
    ov_dir = _target(exported, "openvino")

    bare = tmp_path / "bare"
    bare.mkdir()
    shutil.copy(glob.glob(str(ov_dir / "*.xml"))[0], bare / "PLATE.xml")
    shutil.copy(glob.glob(str(ov_dir / "*.bin"))[0], bare / "PLATE.bin")

    plan = model_loader.inspect(bare)
    assert plan.is_ready, plan.message

    engine = build_engine(str(model_loader.prepare(bare)))
    try:
        _assert_runs(engine)
        info = engine.info().to_dict()
        assert info["imgsz"] == IMGSZ
        # No metadata.yaml -> the task was defaulted, so it must be flagged.
        assert info["task_assumed"] is True

        # OpenVINO's parallel async path must genuinely beat single-image latency.
        bench = engine.benchmark(runs=20)
        assert bench["throughput_ms_per_image"] <= bench["latency_ms"]
    finally:
        engine.close()


def test_openvino_waits_for_its_bin(exported, tmp_path):
    """Half an OpenVINO model must be reported as incomplete, not attempted."""
    ov_dir = _target(exported, "openvino")
    bare = tmp_path / "half"
    bare.mkdir()
    shutil.copy(glob.glob(str(ov_dir / "*.xml"))[0], bare / "PLATE.xml")

    plan = model_loader.inspect(bare)
    assert plan.is_ready is False
    assert ".bin" in plan.message


def test_upload_endpoint_loads_a_real_model(client, base_model):
    """End-to-end: upload a real checkpoint and run inference through the API."""
    from tests.helpers import upload

    response = client.post(
        "/api/model",
        files={"files": ("model.pt", base_model.read_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["task"] == "detect"
    assert body["model"]["num_classes"] == 80

    predicted = client.post("/api/predict", files={"file": upload()}, data={"conf": "0.5"})
    assert predicted.status_code == 200
    assert predicted.json()["status"] == "ok"

    assert client.delete("/api/model").json()["model_loaded"] is False


def test_incomplete_upload_keeps_a_real_model_loaded(client, base_model):
    """The lifecycle guarantee, verified against a real engine rather than a stub."""
    from tests.helpers import upload

    client.post("/api/model", files={"files": ("model.pt", base_model.read_bytes(), "application/octet-stream")})
    assert client.get("/api/health").json()["model_loaded"] is True

    waiting = client.post("/api/model", files={"files": ("net.xml", b"<net/>", "text/xml")})
    assert waiting.json()["status"] == "waiting"

    # The real model is still loaded and still works.
    assert client.get("/api/health").json()["model_loaded"] is True
    assert client.post("/api/predict", files={"file": upload()}).status_code == 200

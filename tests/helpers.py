"""Helpers shared by the test modules.

Kept out of ``conftest.py`` so they can be imported explicitly — a fixture file
is for fixtures, and an import that reads ``from tests.helpers import ...`` says
where the function comes from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from prelabel.engines.base import Detection, InferenceResult, ModelInfo, Timings


def jpeg_bytes(width: int = 16, height: int = 16) -> bytes:
    """A small, valid JPEG."""
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def upload(name: str = "x.jpg", data: bytes | None = None, mime: str = "image/jpeg") -> tuple:
    """A tuple in the shape ``TestClient`` expects for a file field."""
    return (name, data if data is not None else jpeg_bytes(), mime)


def write_test_video(path: Path, frames: int, fps: float = 30.0, size=(64, 48)) -> Path:
    """Write a clip whose frames are individually identifiable.

    Frame *i* is filled with the constant value ``min(255, i * 2)``, so a test can
    recover roughly which source frames a reader returned and assert they span
    the whole clip rather than only its beginning.
    """
    from prelabel.media.video import preferred_codec

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*preferred_codec().fourcc), fps, size)
    assert writer.isOpened(), "could not open a video writer for the fixture"
    try:
        for index in range(frames):
            value = min(255, index * 2)
            writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path


class StubEngine:
    """Minimal engine double implementing the parts the API actually uses.

    Mirrors :class:`~prelabel.engines.base.BaseEngine`'s signatures exactly. A
    double that accepts arguments the real thing rejects (or vice versa) passes
    tests the production code would fail.
    """

    device_label = "CPU"
    imgsz = 640
    supports_tracking = True

    def __init__(self, fail: bool = False, task: str = "detect", name: str = "stub") -> None:
        self.fail = fail
        self.task = task
        self.name = name
        self.calls = 0
        self.batch_calls = 0
        self.track_calls = 0
        self.resets = 0
        self.closed = False
        self.last_conf: float | None = None
        self.last_classes: Sequence[int] | None = None

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> InferenceResult:
        self.calls += 1
        self.last_conf = conf
        self.last_classes = classes
        if self.fail:
            raise RuntimeError("input tensor size mismatch")

        detections = [Detection("car", 0.9, "box", box=[1, 2, 3, 4], class_id=0)]
        if classes is not None:
            detections = [d for d in detections if d.class_id in set(classes)]
        return InferenceResult(
            detections=detections,
            task=self.task,
            timings=Timings(preprocess_ms=1, inference_ms=5, postprocess_ms=1),
            device="CPU",
            image_shape=list(image.shape[:2]),
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> list[InferenceResult]:
        self.batch_calls += 1
        return [self.predict(image, conf=conf, classes=classes) for image in images]

    def track(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
        reset: bool = False,
    ) -> InferenceResult:
        self.track_calls += 1
        if reset:
            self.resets += 1
        result = self.predict(image, conf=conf, classes=classes)
        for position, detection in enumerate(result.detections, start=1):
            detection.track_id = position
        return result

    def info(self) -> ModelInfo:
        return ModelInfo(self.name, "ultralytics", "PyTorch", self.task, 640, 1, {0: "car"}, "CPU")

    def benchmark(self, runs: int = 50) -> dict:
        return {
            "device": "CPU", "imgsz": 640, "runs": runs,
            "latency_ms": 20.0, "latency_fps": 50.0,
            "throughput_ms_per_image": 9.0, "throughput_fps": 111.0,
        }

    def close(self) -> None:
        self.closed = True

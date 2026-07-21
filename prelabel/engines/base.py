"""Core abstractions shared by every inference backend.

The whole system is built around two ideas:

* :class:`InferenceResult` — a *backend-agnostic* description of what a model
  found in a single frame. Detection boxes, segmentation masks, pose keypoints
  and classification scores are all expressed through the same structure, so the
  API and the frontend never need to know which engine produced them.

* :class:`BaseEngine` — the contract every backend implements. Add a new model
  format by writing one subclass; nothing else in the codebase has to change.
"""

from __future__ import annotations

import logging
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("prelabel.engines.base")


@dataclass
class Detection:
    """A single thing a model found in one frame.

    Every field except ``class_name``/``confidence`` is optional, which is what
    lets one structure cover detection, segmentation, pose and classification:

    * ``box``        -> ``[x1, y1, x2, y2]`` in pixel coordinates (detect/seg/pose)
    * ``mask``       -> list of ``[x, y]`` polygon points (segmentation)
    * ``keypoints``  -> list of ``[x, y, score]`` points (pose)
    * neither of the above -> a classification result

    ``track_id`` is set only when tracking a video, where it identifies the same
    object across frames.
    """

    class_name: str
    confidence: float
    kind: str = "box"  # box | segmentation | pose | classification | obb
    box: list[float] | None = None
    mask: list[list[float]] | None = None
    keypoints: list[list[float]] | None = None
    class_id: int | None = None
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "kind": self.kind,
        }
        if self.class_id is not None:
            data["class_id"] = int(self.class_id)
        if self.box is not None:
            data["box"] = [round(float(v), 2) for v in self.box]
        if self.mask is not None:
            data["mask"] = self.mask
        if self.keypoints is not None:
            data["keypoints"] = self.keypoints
        if self.track_id is not None:
            data["track_id"] = int(self.track_id)
        return data

    @property
    def area(self) -> float:
        """Box area in pixels, or 0 for detections without a box."""
        if not self.box:
            return 0.0
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Timings:
    """A breakdown of how long one inference took, in milliseconds.

    Kept separate and explicit because "inference time" is ambiguous: people
    mean different things by it. We measure each stage so the UI can show the
    pure model time (``inference_ms``) *and* the end-to-end pipeline time,
    rather than conflating them into one fuzzy number.
    """

    preprocess_ms: float = 0.0   # decode-independent prep (resize, normalize, letterbox)
    inference_ms: float = 0.0    # the model forward pass only
    postprocess_ms: float = 0.0  # NMS / decoding model output into detections

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess_ms": round(self.preprocess_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "postprocess_ms": round(self.postprocess_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


#: Confidence at which a model is maximally undecided.
_DECISION_BOUNDARY = 0.5


@dataclass
class InferenceResult:
    """Everything produced for one frame, plus a little metadata."""

    detections: list[Detection] = field(default_factory=list)
    task: str = "detect"
    timings: Timings = field(default_factory=Timings)
    device: str = "cpu"
    image_shape: list[int] | None = None  # [height, width]

    @property
    def speed_ms(self) -> float:
        """Headline number: the pure model inference time."""
        return self.timings.inference_ms

    @property
    def review_priority(self) -> float:
        """How much this frame would benefit from a human look, from 0 to 1.

        This is *margin sampling*, the standard active-learning heuristic: a
        prediction at 0.95 or at 0.02 tells you little, because the model is sure
        either way. One sitting at 0.5 is where the model is genuinely undecided,
        and where a correction teaches it the most.

        A frame with no detections at all scores 1.0 — it is either legitimately
        empty or a complete miss, and only a person can tell which.
        """
        if not self.detections:
            return 1.0
        return max(
            1.0 - abs(float(d.confidence) - _DECISION_BOUNDARY) / _DECISION_BOUNDARY
            for d in self.detections
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "speed_ms": round(self.timings.inference_ms, 2),  # back-compat headline
            "timings": self.timings.to_dict(),
            "device": self.device,
            "image_shape": self.image_shape,
            "count": len(self.detections),
            "review_priority": round(self.review_priority, 4),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class ModelInfo:
    """Human-readable summary shown in the UI after a model loads."""

    name: str
    backend: str
    format: str
    task: str
    imgsz: int
    num_classes: int
    class_names: dict[int, str]
    device: str = "cpu"
    path: str = ""
    #: True when the task wasn't in the model and we had to assume it.
    task_assumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "format": self.format,
            "task": self.task,
            "imgsz": self.imgsz,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": self.device,
            "path": self.path,
            "task_assumed": self.task_assumed,
        }


class BaseEngine(ABC):
    """Contract implemented by every inference backend.

    A subclass loads a model from disk and turns a single decoded image
    (an ``np.ndarray`` in BGR order, as produced by OpenCV) into an
    :class:`InferenceResult`.
    """

    #: File extensions this engine can load, e.g. ``{".pt", ".onnx"}``.
    extensions: set[str] = set()

    #: Short, human-friendly backend name shown in the UI.
    backend_name: str = "base"

    #: Image size the model runs at; set by ``load()``.
    imgsz: int = 640

    #: Whether this backend can follow objects across video frames.
    supports_tracking: bool = False

    def __init__(
        self,
        model_path: str,
        imgsz: int | None = None,
        device: str | None = None,
        warmup: bool = True,
    ) -> None:
        self.model_path = model_path
        self._imgsz = imgsz
        #: Explicit device request ("cpu" / "cuda" / "cuda:0"); None = auto.
        self._device_override = device
        self.load()
        self.selftest()
        if warmup:
            self.warmup()

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory. Called once from ``__init__``."""

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> InferenceResult:
        """Run inference on one BGR image and return a unified result.

        ``classes`` restricts the output to those class ids. Filtering inside the
        backend is not just tidier — it lets the model's own NMS drop the other
        classes, which is both faster and avoids a suppressed-but-wanted box.
        """

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> list[InferenceResult]:
        """Run inference on several images.

        The default loops :meth:`predict` (correct everywhere). Backends that can
        truly batch — e.g. a GPU forward pass — override this for real throughput.
        """
        return [self.predict(img, conf=conf, classes=classes) for img in images]

    def track(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
        reset: bool = False,
    ) -> InferenceResult:
        """Detect *and* follow objects across successive frames.

        Backends that cannot track fall back to plain detection, so callers never
        have to branch — they just get results without ``track_id`` set.
        ``reset=True`` starts a new sequence, discarding any previous state.
        """
        return self.predict(image, conf=conf, classes=classes)

    @abstractmethod
    def info(self) -> ModelInfo:
        """Return a summary of the loaded model."""

    @classmethod
    def can_load(cls, extension: str) -> bool:
        return extension.lower() in cls.extensions

    def selftest(self) -> None:
        """Run one inference and let any failure propagate.

        Constructing an engine is not the same as it working. A backend can
        accept a file with the right extension, populate itself with nonsense,
        and only fall over when asked to infer — a plain ResNet handed to a YOLO
        loader does exactly that, reporting a made-up task and class count.

        Raising here is what lets :func:`~prelabel.engines.factory.build_engine`
        move on to the next candidate backend, instead of installing something
        that will fail on the user's first real image.
        """
        rng = np.random.default_rng(0)
        size = int(getattr(self, "imgsz", 640) or 640)
        self.predict(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), conf=0.25)

    def warmup(self, runs: int = 3) -> None:
        """Prime the model with a few dummy inferences.

        Lazy CUDA context creation, cuDNN autotuning, graph optimisation and
        memory allocation all happen on the *first* calls, which would otherwise
        be charged to the first real image. A realistic (noisy, not all-zero)
        frame at the model's own image size exercises the full path, so the
        timings the user sees reflect steady-state performance.

        Note: a GPU that has been idle can still clock down between requests, so
        the very first inference after a long pause may read marginally higher —
        that's hardware power management, not a cold model.
        """
        try:
            size = int(getattr(self, "imgsz", 640) or 640)
            rng = np.random.default_rng(0)
            dummy = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            for _ in range(max(1, runs)):
                self.predict(dummy, conf=0.25)
        except Exception as exc:  # noqa: BLE001 - warmup must never break loading
            # Not fatal — but a warmup failure usually means the first *real*
            # inference will fail the same way, so never swallow it silently.
            log.warning(
                "Warmup failed for %s (%s). The model loaded, but inference may fail: %s",
                type(self).__name__, getattr(self, "model_path", "?"), exc,
            )

    def benchmark(self, runs: int = 50) -> dict[str, Any]:
        """Measure single-image **latency** and best-case **throughput**.

        These are two different things and the gap surprises people:

        * *latency* — how long one image takes start-to-finish. This is what the
          UI reports per request.
        * *throughput* — how many images/second you get when they're processed
          back-to-back / in parallel. This is the number people quote from "I ran
          100 images and it was 5 ms each", and it is always better than latency
          because per-image overhead is amortised and (for some backends) several
          inferences run at once.

        The default measures latency through the real predict path and, lacking a
        parallel path, reports throughput equal to latency. Backends that *can*
        parallelise (see the OpenVINO override) report their true throughput.
        """
        size = int(getattr(self, "imgsz", 640) or 640)
        rng = np.random.default_rng(0)
        dummy = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)

        for _ in range(5):  # warm
            self.predict(dummy, conf=0.25)

        lat = []
        for _ in range(max(5, runs)):
            t = time.perf_counter()
            self.predict(dummy, conf=0.25)
            lat.append((time.perf_counter() - t) * 1000)
        latency_ms = statistics.median(lat)

        throughput_ms = self._throughput_ms(runs) or latency_ms

        return {
            "device": getattr(self, "device_label", "?"),
            "imgsz": size,
            "runs": int(max(5, runs)),
            "latency_ms": round(latency_ms, 2),
            "latency_fps": round(1000.0 / latency_ms, 1),
            "throughput_ms_per_image": round(throughput_ms, 2),
            "throughput_fps": round(1000.0 / throughput_ms, 1),
        }

    def _throughput_ms(self, runs: int) -> float | None:
        """Best-case ms/image when running in parallel; None if unsupported."""
        return None

    def close(self) -> None:
        """Release any held resources.

        Deliberately concrete and empty rather than abstract: a backend that
        holds nothing (a pure-numpy engine, a test double) should not be forced
        to write an empty override just to satisfy the interface.
        """
        return

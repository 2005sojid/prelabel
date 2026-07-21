"""Ultralytics backend — the workhorse of the project.

A single Ultralytics ``YOLO`` object can load weights exported to a remarkable
range of formats and auto-detects the task (detect / segment / classify / pose /
obb). That makes it the most practical path to "universal" inference, so it is
the default engine and covers the formats most people actually ship:

    .pt          PyTorch
    .onnx        ONNX
    OpenVINO     a ``*_openvino_model/`` directory (.xml + .bin + metadata.yaml)
    .engine      TensorRT
    .torchscript TorchScript
    .tflite      TensorFlow Lite
    .mlpackage   CoreML

This wrapper's only job is to translate Ultralytics' rich ``Results`` object
into the project's backend-agnostic :class:`InferenceResult`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import BaseEngine, Detection, InferenceResult, ModelInfo, Timings

log = logging.getLogger("prelabel.engines.ultralytics")

#: Batch size baked into the dedicated OpenVINO throughput model (see
#: ``_ov_throughput_model``). Drives how many images OpenVINO keeps in flight at
#: once for the batch gallery; the gallery sends chunks of this size.
OV_THROUGHPUT_BATCH = 16

#: Trackers Ultralytics ships. ByteTrack is faster; BoT-SORT re-identifies better
#: after an occlusion, at some cost.
TRACKERS = ("bytetrack.yaml", "botsort.yaml")
DEFAULT_TRACKER = TRACKERS[0]

#: Formats whose exported graph keeps the state tracking needs. Tracking works
#: through the same predict path, so anything Ultralytics can run, it can track.
TRACKABLE_FORMATS = frozenset({"PyTorch", "TorchScript", "TensorRT", "ONNX", "OpenVINO"})


@lru_cache(maxsize=1)
def tracking_available() -> bool:
    """Whether the tracker's own dependency is installed.

    Ultralytics' trackers need ``lap`` for the assignment step, and it is an
    optional dependency. Probing once at import time turns "works" into a fact we
    can report up front, instead of a crash halfway through rendering a video.
    """
    try:
        import lap  # noqa: F401
    except ImportError:
        log.warning(
            "Object tracking is unavailable: the 'lap' package is not installed. "
            "Run 'pip install lap' to enable it; detection still works."
        )
        return False
    return True


def _class_list(classes: Sequence[int] | None) -> list[int] | None:
    """Normalise a class filter for Ultralytics, which wants a list or None."""
    if classes is None:
        return None
    unique = sorted({int(c) for c in classes})
    return unique or None


class UltralyticsEngine(BaseEngine):
    # Note: raw ``.xml`` is intentionally absent — OpenVINO is loaded from the
    # assembled ``*_openvino_model/`` directory (see app.model_loader), which the
    # factory routes here explicitly.
    extensions = {
        ".pt",
        ".onnx",
        ".engine",
        ".torchscript",
        ".tflite",
        ".mlpackage",
        ".pb",
    }
    backend_name = "ultralytics"

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)
        self._assert_usable()
        self.device, self.device_label = self._resolve_device()
        self.imgsz = self._resolve_imgsz()
        self.static_batch = self._fixed_batch_size()
        #: Lazily-built, CPU-only OpenVINO model compiled for parallel async
        #: throughput, used by ``predict_batch`` (the primary model stays in
        #: LATENCY mode for fast single-image / video / webcam inference).
        self._throughput_model = None
        self._throughput_dir: str | None = None
        self.tracker = DEFAULT_TRACKER
        self.supports_tracking = (
            self._format() in TRACKABLE_FORMATS
            and self.model.task != "classify"
            and tracking_available()
        )

        log.info(
            "Loaded %s | task=%s | format=%s | imgsz=%d | device=%s",
            os.path.basename(self.model_path.rstrip("/\\")),
            self.model.task, self._format(), self.imgsz, self.device_label,
        )

    def _assert_usable(self) -> None:
        """Fail loudly if ``YOLO()`` accepted the file but produced nothing usable.

        Ultralytics does not always validate eagerly: handed a file with the right
        extension and the wrong contents, it can return an object whose internals
        are unpopulated, and the problem only surfaces much later. Checking the
        two attributes every code path needs turns that into a clear load error.
        """
        try:
            names = self.model.names
            task = self.model.task
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"not a readable model file ({exc})") from exc
        if not names or not task:
            raise ValueError("model file carries no class names or task")

    # -- resolution helpers --------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        """Pick an inference device and an *honest* label for the UI.

        Honours the ``UI_DEVICE`` override; otherwise auto-detects. The label has
        to reflect where the format **actually** runs, which is not always the
        torch device:

        * PyTorch / TorchScript / TensorRT — run on the torch device (GPU/CPU).
        * ONNX — runs under ONNX Runtime; only uses the GPU if the CUDA
          execution provider is actually installed (``onnxruntime-gpu``).
        * OpenVINO / TFLite / others — their own CPU runtime here.
        """
        from .. import config

        if self._is_openvino():
            return "cpu", "CPU (OpenVINO)"

        # Priority: explicit per-load request > UI_DEVICE env > auto-detect.
        override = (self._device_override or config.DEVICE or "").lower()
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda_ok = False

        if override == "cpu":
            want_gpu, idx = False, 0
        elif override.startswith("cuda") or override.isdigit():
            idx = int(override.split(":")[-1]) if ":" in override else int(override) if override.isdigit() else 0
            want_gpu = cuda_ok
            if not cuda_ok:
                log.warning("UI_DEVICE=%s requested but CUDA is unavailable — using CPU", override)
        else:  # auto
            want_gpu, idx = cuda_ok, 0

        fmt = self._format()

        if fmt == "ONNX":
            try:
                import onnxruntime as ort

                gpu_provider = "CUDAExecutionProvider" in ort.get_available_providers()
            except Exception:  # noqa: BLE001
                gpu_provider = False
            if want_gpu and gpu_provider:
                return str(idx), self._gpu_label(idx)
            return "cpu", "CPU (ONNX Runtime)"

        if fmt in ("PyTorch", "TorchScript", "TensorRT"):
            if want_gpu:
                return str(idx), self._gpu_label(idx)
            return "cpu", "CPU"

        # TFLite, TensorFlow, etc. — CPU runtime in this setup.
        return "cpu", "CPU"

    @staticmethod
    def _gpu_label(idx: int) -> str:
        try:
            import torch

            return f"GPU · {torch.cuda.get_device_name(idx)}"
        except Exception:  # noqa: BLE001
            return "GPU"

    def _resolve_imgsz(self) -> int:
        """Determine the inference image size.

        Resolution order, most authoritative first:

        1. **The model's own fixed input shape.** Formats like OpenVINO export
           with a *static* input tensor (e.g. ``768×768``); feeding any other
           size raises a hard shape-mismatch at inference. When the model fixes
           its size, that size wins — even over an explicit override, which we
           warn about rather than honour into a guaranteed crash.
        2. An explicit override passed by the caller.
        3. The size recorded in the checkpoint metadata.
        4. A 640 fallback (logged, never silent).
        """
        fixed = self._fixed_input_size()
        if fixed:
            if self._imgsz and int(self._imgsz) != fixed:
                log.warning("Ignoring imgsz=%s — model fixes its input at %d", self._imgsz, fixed)
            log.info("imgsz=%d (read from model artifact)", fixed)
            return fixed

        if self._imgsz:
            log.info("imgsz=%d (explicit override)", int(self._imgsz))
            return int(self._imgsz)

        candidates = [
            ("overrides", getattr(self.model, "overrides", {}) or {}),
            ("model.args", getattr(getattr(self.model, "model", None), "args", {}) or {}),
        ]
        for source, store in candidates:
            value = store.get("imgsz") if hasattr(store, "get") else None
            if value:
                if isinstance(value, (list, tuple)):
                    value = value[0]
                log.info("imgsz=%d (from %s)", int(value), source)
                return int(value)

        log.warning("imgsz not found in model metadata — defaulting to 640")
        return 640

    def _fixed_input_size(self) -> int | None:
        """Read the model's intended input size straight from the artifact.

        Exported formats bake in the size they were exported at; feeding any
        other size either raises a hard shape-mismatch (static graphs like
        OpenVINO / fixed ONNX) or silently runs at the wrong scale. We read it
        from the file itself — the only authoritative source. Returns ``None``
        for PyTorch, whose input is dynamic (any multiple of 32 is valid).
        """
        fmt = self._format()
        try:
            if fmt == "OpenVINO":
                import glob

                import openvino as ov

                xmls = glob.glob(os.path.join(self.model_path, "*.xml"))
                if xmls:
                    shape = ov.Core().read_model(xmls[0]).inputs[0].get_partial_shape()
                    if len(shape) == 4 and shape[2].is_static and shape[3].is_static:
                        return int(max(shape[2].get_length(), shape[3].get_length()))

            elif fmt == "ONNX":
                import onnxruntime as ort

                shape = ort.InferenceSession(
                    self.model_path, providers=["CPUExecutionProvider"]
                ).get_inputs()[0].shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    return int(max(shape[2], shape[3]))

            elif fmt == "TorchScript":
                import json

                import torch

                extra = {"config.txt": ""}
                torch.jit.load(self.model_path, _extra_files=extra, map_location="cpu")
                meta = json.loads(extra["config.txt"] or "{}")
                value = meta.get("imgsz")
                if value:
                    return int(value[0] if isinstance(value, (list, tuple)) else value)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not read fixed input size for %s: %s", fmt, exc)
        return None

    def _fixed_batch_size(self) -> int | None:
        """Return the model's static batch dimension, or ``None`` if dynamic.

        Several exported formats bake in a fixed batch size: Ultralytics exports
        OpenVINO — and, by default, ONNX — with ``batch=1``. Feeding such a model
        more than one image in a single forward pass raises a hard shape error
        (``got [N,3,H,W] expecting [1,3,H,W]``), so :meth:`predict_batch` must run
        them one image at a time. PyTorch / TorchScript keep a dynamic batch
        dimension and return ``None`` (true batching is fine).
        """
        fmt = self._format()
        try:
            if fmt == "OpenVINO":
                import glob

                import openvino as ov

                xmls = glob.glob(os.path.join(self.model_path, "*.xml"))
                if xmls:
                    shape = ov.Core().read_model(xmls[0]).inputs[0].get_partial_shape()
                    if len(shape) >= 1 and shape[0].is_static:
                        return int(shape[0].get_length())

            elif fmt == "ONNX":
                import onnxruntime as ort

                shape = ort.InferenceSession(
                    self.model_path, providers=["CPUExecutionProvider"]
                ).get_inputs()[0].shape
                if shape and isinstance(shape[0], int):
                    return int(shape[0])
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not read fixed batch size for %s: %s", fmt, exc)
        return None

    def _is_openvino(self) -> bool:
        return os.path.isdir(self.model_path) and "_openvino_model" in os.path.basename(
            self.model_path.rstrip("/\\")
        )

    def _format(self) -> str:
        if self._is_openvino():
            return "OpenVINO"
        ext = Path(self.model_path).suffix.lower()
        return {
            ".pt": "PyTorch",
            ".onnx": "ONNX",
            ".engine": "TensorRT",
            ".torchscript": "TorchScript",
            ".tflite": "TensorFlow Lite",
            ".mlpackage": "CoreML",
            ".pb": "TensorFlow",
        }.get(ext, ext)

    # -- inference -----------------------------------------------------------

    def info(self) -> ModelInfo:
        names = {int(k): str(v) for k, v in self.model.names.items()}
        return ModelInfo(
            name=os.path.basename(self.model_path.rstrip("/\\")),
            backend=self.backend_name,
            format=self._format(),
            task=self.model.task,
            imgsz=self.imgsz,
            num_classes=len(names),
            class_names=names,
            device=self.device_label,
            path=os.path.abspath(self.model_path),
            task_assumed=os.path.exists(os.path.join(self.model_path, ".task_assumed")),
        )

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> InferenceResult:
        import torch

        with torch.no_grad():
            results = self.model(
                image, conf=conf, imgsz=self.imgsz, device=self.device,
                classes=_class_list(classes), verbose=False,
            )[0]
        return self._build_result(results, list(image.shape[:2]))

    def track(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
        reset: bool = False,
    ) -> InferenceResult:
        """Detect and follow objects across frames, assigning stable ``track_id``s.

        Ultralytics keeps tracker state on the model object between calls when
        ``persist=True``, which is what makes an id mean the same object from one
        frame to the next. That state is per-model, so a new sequence has to
        clear it — otherwise the second video inherits the first one's ids.

        Falls back to plain detection when tracking is not available, so a
        missing optional dependency costs you the ids, not the whole render.
        """
        import torch

        if not self.supports_tracking:
            return self.predict(image, conf=conf, classes=classes)

        if reset:
            self._reset_tracker()

        with torch.no_grad():
            results = self.model.track(
                image, conf=conf, imgsz=self.imgsz, device=self.device,
                classes=_class_list(classes), tracker=self.tracker, persist=True, verbose=False,
            )[0]
        return self._build_result(results, list(image.shape[:2]))

    def _reset_tracker(self) -> None:
        """Drop tracker state so the next frame starts a fresh sequence."""
        try:
            for tracker in getattr(self.model.predictor, "trackers", []) or []:
                tracker.reset()
        except (AttributeError, TypeError):
            # No predictor yet (first call) or a tracker without reset(): both
            # mean there is no stale state to clear.
            pass

    def _ov_throughput_model(self):
        """Lazily build (and cache) a CPU OpenVINO model tuned for parallel batch.

        The primary OpenVINO model is compiled in LATENCY mode — best for the
        single-image paths (``/api/predict``, video, webcam). Ultralytics only
        switches OpenVINO to its parallel async inference (``AsyncInferQueue``,
        ``CUMULATIVE_THROUGHPUT``) when the model metadata declares a dynamic,
        multi-image batch — which would otherwise wreck single-image latency.

        So for the batch gallery we build a *second* model from the same weights
        with ``batch>1`` + ``dynamic`` in its metadata and pinned to the CPU
        (AUTO would otherwise spill onto a slow integrated GPU). It's cached for
        the life of the engine and torn down in :meth:`close`.
        """
        if self._throughput_model is not None:
            return self._throughput_model

        import shutil

        import yaml
        from ultralytics import YOLO

        src = self.model_path.rstrip("/\\")
        dst = src.replace("_openvino_model", "_throughput_openvino_model")
        if os.path.abspath(dst) == os.path.abspath(src):
            dst = src + "_throughput_openvino_model"
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)

        meta_path = os.path.join(dst, "metadata.yaml")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh) or {}
        meta["batch"] = OV_THROUGHPUT_BATCH
        args = meta.get("args") or {}
        args["dynamic"] = True
        meta["args"] = args
        with open(meta_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)

        self._throughput_dir = dst
        self._throughput_model = YOLO(dst)
        log.info("Built OpenVINO throughput model (batch=%d, CPU) for parallel batch inference", OV_THROUGHPUT_BATCH)
        return self._throughput_model

    def predict_batch(
        self,
        images: list[np.ndarray],
        conf: float = 0.25,
        classes: Sequence[int] | None = None,
    ) -> list[InferenceResult]:
        """Run inference on several images, in parallel wherever the backend allows.

        Three paths, all parallel where it's possible to be:

        * **Dynamic-batch (PyTorch / TorchScript / TensorRT)** — one batched
          forward pass; the GPU processes the whole stack at once.
        * **OpenVINO** — a dedicated CPU throughput model runs the images through
          OpenVINO's async inference queue, several in flight concurrently.
        * **Static-batch ONNX & friends** — the export bakes in ``batch=1`` and a
          single CPU inference already saturates all cores, so there's no parallel
          win to be had; we run them one at a time (correct, CPU-bound).

        Ultralytics' ``speed`` dict is reported **per image**, so each result uses
        it directly.
        """
        import torch

        if not images:
            return []

        # Static-batch formats can't take a multi-image tensor in one forward.
        if self.static_batch is not None and self.static_batch < len(images):
            if self._is_openvino():
                model = self._ov_throughput_model()
                # ``rect=False`` forces a square letterbox: the throughput model's
                # metadata advertises a dynamic batch (to unlock async inference),
                # which would otherwise let ultralytics use a rectangular input the
                # static OpenVINO IR rejects.
                with torch.no_grad():
                    results = model(
                        images, conf=conf, imgsz=self.imgsz, device="intel:CPU",
                        classes=_class_list(classes), rect=False, verbose=False,
                    )
                return [self._build_result(r, list(img.shape[:2])) for r, img in zip(results, images, strict=True)]
            # ONNX / other static formats: per-image (CPU already fully utilised).
            return [self.predict(img, conf=conf, classes=classes) for img in images]

        # Dynamic batch: one batched forward pass (real GPU parallelism).
        with torch.no_grad():
            results = self.model(
                images, conf=conf, imgsz=self.imgsz, device=self.device,
                classes=_class_list(classes), verbose=False,
            )
        return [self._build_result(r, list(img.shape[:2])) for r, img in zip(results, images, strict=True)]

    def _build_result(self, results, image_shape: list[int], timings: Timings | None = None) -> InferenceResult:
        """Translate one Ultralytics ``Results`` into a backend-agnostic result."""
        names = self.model.names
        result = InferenceResult(task=self.model.task, device=self.device_label, image_shape=image_shape)

        # Ultralytics already times each stage internally and exposes it as a
        # dict of milliseconds — use it directly so our numbers match what the
        # ``yolo`` CLI reports.
        if timings is not None:
            result.timings = timings
        else:
            speed = getattr(results, "speed", None) or {}
            result.timings = Timings(
                preprocess_ms=float(speed.get("preprocess", 0.0) or 0.0),
                inference_ms=float(speed.get("inference", 0.0) or 0.0),
                postprocess_ms=float(speed.get("postprocess", 0.0) or 0.0),
            )

        # --- Classification ----------------------------------------------------
        if getattr(results, "probs", None) is not None:
            probs = results.probs
            for idx in probs.top5:
                result.detections.append(
                    Detection(
                        class_name=names[int(idx)],
                        confidence=float(probs.data[int(idx)]),
                        kind="classification",
                        class_id=int(idx),
                    )
                )
            del results
            return result

        # --- Oriented bounding boxes ------------------------------------------
        obb = getattr(results, "obb", None)
        boxes = getattr(results, "boxes", None)
        if obb is not None and boxes is None:
            xyxy = obb.xyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            clss = obb.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                result.detections.append(
                    Detection(
                        class_name=names[int(clss[i])],
                        confidence=float(confs[i]),
                        kind="obb",
                        box=xyxy[i].tolist(),
                        class_id=int(clss[i]),
                    )
                )
            del results
            return result

        # --- Detection / Segmentation / Pose ----------------------------------
        if boxes is None or boxes.xyxy is None:
            del results
            return result

        masks = getattr(results, "masks", None)
        keypoints = getattr(results, "keypoints", None)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        mask_polys = masks.xy if masks is not None else None
        kpts = keypoints.data.cpu().numpy() if keypoints is not None else None
        # Present only when this frame came through track(); absent for predict().
        track_ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None

        for i in range(len(xyxy)):
            det = Detection(
                class_name=names[int(clss[i])],
                confidence=float(confs[i]),
                kind="box",
                box=xyxy[i].tolist(),
                class_id=int(clss[i]),
                track_id=int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None,
            )
            if mask_polys is not None and i < len(mask_polys):
                det.mask = mask_polys[i].astype(float).tolist()
                det.kind = "segmentation"
            if kpts is not None and i < len(kpts):
                det.keypoints = kpts[i].astype(float).tolist()
                det.kind = "pose"
            result.detections.append(det)

        del results
        return result

    def _throughput_ms(self, runs: int) -> float | None:
        """Best-case ms/image when images are processed together.

        Two regimes give the same kind of "100 images at ~5 ms each" number:

        * **OpenVINO** runs several inference requests in parallel
          (``AsyncInferQueue``), which packs the CPU far better than one request.
        * **Everything else** (PyTorch / TorchScript / ONNX) is measured by
          batching images into one forward pass — a big win on a GPU, ~neutral on
          a CPU (which is the honest answer there, not a bug).
        """
        if self._is_openvino():
            return self._openvino_throughput_ms(runs)
        return self._batched_throughput_ms(runs)

    def _batched_throughput_ms(self, runs: int) -> float | None:
        import statistics
        import time

        import numpy as np

        batch = 16
        imgs = [np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8) for _ in range(batch)]
        try:
            self.model(imgs, imgsz=self.imgsz, device=self.device, verbose=False)  # warm
            per_image = []
            for _ in range(max(3, runs // batch)):
                start = time.perf_counter()
                self.model(imgs, imgsz=self.imgsz, device=self.device, verbose=False)
                per_image.append((time.perf_counter() - start) * 1000.0 / batch)
            return statistics.median(per_image)
        except Exception as exc:  # noqa: BLE001
            log.debug("Batched throughput benchmark failed: %s", exc)
            return None

    def _openvino_throughput_ms(self, runs: int) -> float | None:
        try:
            import glob
            import time

            import numpy as np
            import openvino as ov

            core = ov.Core()
            xml = glob.glob(os.path.join(self.model_path, "*.xml"))[0]
            compiled = core.compile_model(xml, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
            nireq = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")) or 4
            queue = ov.AsyncInferQueue(compiled, nireq)
            tensor = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)

            for _ in range(nireq * 2):  # warm
                queue.start_async({0: tensor})
            queue.wait_all()

            n = max(runs, 50)
            start = time.perf_counter()
            for _ in range(n):
                queue.start_async({0: tensor})
            queue.wait_all()
            return (time.perf_counter() - start) * 1000.0 / n
        except Exception as exc:  # noqa: BLE001
            log.debug("OpenVINO throughput benchmark failed: %s", exc)
            return None

    def close(self) -> None:
        """Release weights and any locked files (matters on Windows, where a
        loaded OpenVINO ``.bin`` stays locked until its handles are freed)."""
        import gc

        try:
            del self.model
        except AttributeError:
            pass
        # Release the dedicated OpenVINO throughput model and its temp directory.
        if getattr(self, "_throughput_model", None) is not None:
            try:
                del self._throughput_model
            except AttributeError:
                pass
            self._throughput_model = None
        gc.collect()
        if getattr(self, "_throughput_dir", None):
            import shutil

            shutil.rmtree(self._throughput_dir, ignore_errors=True)
            self._throughput_dir = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

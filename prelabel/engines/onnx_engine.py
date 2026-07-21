"""Generic ONNX backend for plain image classifiers.

The Ultralytics backend covers ONNX models that follow the YOLO export layout.
This engine is the fallback for *arbitrary* ONNX classification networks
(ResNet, MobileNet, EfficientNet, ViT, …) exported by other frameworks.

It performs the standard ImageNet-style preprocessing pipeline (resize ->
center-ish letterbox -> normalize) and a softmax over the output logits. Class
labels are read from a sibling ``labels.txt``/``<model>.txt`` file when present,
falling back to the bundled ImageNet-1k names for 1000-class models, and to
``class_<i>`` otherwise.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import cv2
import numpy as np

from .base import BaseEngine, Detection, InferenceResult, ModelInfo, Timings

#: How many classes a classifier reports per image.
TOP_K = 5


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class GenericOnnxClassifier(BaseEngine):
    # Registered for .onnx too, but the factory only reaches it after the
    # Ultralytics backend declines the file, so the two never clash.
    extensions = {".onnx"}
    backend_name = "onnxruntime"

    # ImageNet normalization (the de-facto default for classifier exports).
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load(self) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        override = (self._device_override or "").lower()
        if override == "cpu":
            providers = ["CPUExecutionProvider"]
        elif override.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = available  # auto: best available
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        active = self.session.get_providers()[0]
        self.device_label = {
            "CUDAExecutionProvider": "GPU · CUDA",
            "TensorrtExecutionProvider": "GPU · TensorRT",
            "CPUExecutionProvider": "CPU (ONNX Runtime)",
        }.get(active, active)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Expected shape is typically [N, 3, H, W]; fall back to 224 when dynamic.
        shape = inp.shape
        self._nchw = len(shape) == 4 and shape[1] in (3, 1)
        h = shape[2] if isinstance(shape[2], int) else 224
        w = shape[3] if isinstance(shape[3], int) else 224
        self.imgsz = int(self._imgsz or max(h, w) or 224)

        out_shape = self.session.get_outputs()[0].shape
        self.num_classes = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 0
        self.class_names = self._load_labels(self.num_classes)

    def _load_labels(self, num_classes: int) -> dict[int, str]:
        base = os.path.splitext(self.model_path)[0]
        for candidate in (base + ".txt", os.path.join(os.path.dirname(self.model_path), "labels.txt")):
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as fh:
                    labels = [ln.strip() for ln in fh if ln.strip()]
                return dict(enumerate(labels))
        return {i: f"class_{i}" for i in range(num_classes or 0)}

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self.model_path),
            backend=self.backend_name,
            format="ONNX",
            task="classify",
            imgsz=self.imgsz,
            num_classes=self.num_classes,
            class_names=self.class_names,
            device=self.device_label,
            path=os.path.abspath(self.model_path),
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.MEAN) / self.STD
        if self._nchw:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img[np.newaxis, ...].astype(np.float32)

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.0,
        classes: Sequence[int] | None = None,
    ) -> InferenceResult:
        from time import perf_counter

        wanted = {int(c) for c in classes} if classes else None

        # Time each stage separately so the breakdown is consistent with the
        # Ultralytics backend (preprocess / inference / postprocess).
        t0 = perf_counter()
        tensor = self._preprocess(image)
        t1 = perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        t2 = perf_counter()
        logits = np.asarray(outputs[0]).reshape(-1)
        probs = _softmax(logits)
        # Rank first, then filter: the class filter must not promote a class the
        # model ranked 900th into the top 5.
        ranked = np.argsort(probs)[::-1]
        if wanted is not None:
            ranked = np.array([i for i in ranked if int(i) in wanted], dtype=int)
        top = ranked[:TOP_K]
        t3 = perf_counter()

        result = InferenceResult(
            task="classify",
            timings=Timings(
                preprocess_ms=(t1 - t0) * 1000.0,
                inference_ms=(t2 - t1) * 1000.0,
                postprocess_ms=(t3 - t2) * 1000.0,
            ),
            device=self.device_label,
            image_shape=list(image.shape[:2]),
        )
        for idx in top:
            if probs[idx] < conf:
                continue
            result.detections.append(
                Detection(
                    class_name=self.class_names.get(int(idx), f"class_{int(idx)}"),
                    confidence=float(probs[idx]),
                    kind="classification",
                    class_id=int(idx),
                )
            )
        return result

    def close(self) -> None:
        import gc

        try:
            del self.session
        except AttributeError:
            pass
        gc.collect()

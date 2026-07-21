"""Engine selection.

Given a model file, decide which backend should load it. The strategy is simple
and predictable:

1. Collect every registered engine whose extension set matches the file.
2. Try them in priority order, returning the first that loads successfully.

Ultralytics is tried first because it auto-detects task and handles the widest
range of formats; the generic ONNX classifier is the fallback for ``.onnx``
files that are not YOLO exports.
"""

from __future__ import annotations

import logging
import os

from .base import BaseEngine
from .onnx_engine import GenericOnnxClassifier
from .ultralytics_engine import UltralyticsEngine

log = logging.getLogger("prelabel.engines.factory")

_cuda_dlls_registered = False


def _ensure_cuda_dll_path() -> None:
    """Let ``onnxruntime-gpu`` find the CUDA/cuDNN DLLs that ``torch`` already ships.

    On Windows the ONNX Runtime CUDA provider needs ``cudnn*``/``cublas*`` DLLs on
    the process DLL search path; without them it silently falls back to the CPU
    provider (so ONNX inference runs on the CPU despite a GPU being present). The
    installed ``torch`` wheel bundles a matching CUDA build, so we register its
    ``lib`` directory once, before any model is loaded. No-op off Windows.
    """
    global _cuda_dlls_registered
    if _cuda_dlls_registered:
        return
    _cuda_dlls_registered = True
    if os.name != "nt":
        return
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            log.debug("Registered torch CUDA/cuDNN DLLs for onnxruntime: %s", lib)
    except Exception as exc:  # noqa: BLE001 - GPU is best-effort; CPU still works
        log.debug("Could not register CUDA DLL path: %s", exc)

# Order matters: higher priority engines come first.
ENGINE_PRIORITY: list[type[BaseEngine]] = [
    UltralyticsEngine,
    GenericOnnxClassifier,
]

# Advertised to the UI. ``.xml``/``.bin`` are user-facing inputs even though the
# engine actually loads the assembled ``*_openvino_model/`` directory.
SUPPORTED_EXTENSIONS = sorted(
    {ext for engine in ENGINE_PRIORITY for ext in engine.extensions} | {".xml", ".bin"}
)


def build_engine(
    model_path: str,
    imgsz: int | None = None,
    device: str | None = None,
) -> BaseEngine:
    """Instantiate the best engine for ``model_path``.

    ``model_path`` may be a single file or an assembled ``*_openvino_model/``
    directory (handled by :class:`UltralyticsEngine`). ``device`` forces a
    backend ("cpu" / "cuda" / "cuda:0"); ``None`` auto-detects.

    Raises ``ValueError`` for unknown formats and ``RuntimeError`` if every
    candidate backend fails to load the target (with all errors aggregated).
    """
    _ensure_cuda_dll_path()  # so an ONNX model can actually reach the GPU

    # OpenVINO arrives as a prepared directory, not a file with an extension.
    if os.path.isdir(model_path) and "_openvino_model" in os.path.basename(model_path.rstrip("/\\")):
        return UltralyticsEngine(model_path, imgsz=imgsz, device=device)

    ext = os.path.splitext(model_path)[1].lower()
    candidates = [e for e in ENGINE_PRIORITY if e.can_load(ext)]

    if not candidates:
        raise ValueError(
            f"Unsupported model format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    errors = []
    for engine_cls in candidates:
        try:
            return engine_cls(model_path, imgsz=imgsz, device=device)
        except Exception as exc:  # noqa: BLE001 - we aggregate and re-raise
            errors.append(f"{engine_cls.backend_name}: {exc}")

    raise RuntimeError(
        "Failed to load model with any backend:\n  " + "\n  ".join(errors)
    )

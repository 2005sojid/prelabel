"""Prelabel — the missing layer between inference and annotation."""

import os

# Ultralytics tries to ``pip install onnxruntime`` whenever an ONNX model loads —
# it doesn't recognise the ``onnxruntime-gpu`` distribution we ship, so its check
# "fails" and it auto-updates, which is slow and would replace the GPU runtime
# with the CPU one. We manage dependencies ourselves; disable the auto-install.
# Must be set before ultralytics is first imported (engines import it lazily).
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

__version__ = "0.3.0"

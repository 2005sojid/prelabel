"""Turning a folder of uploaded files into something an engine can load.

Most formats are a single file, but a few are not: OpenVINO ships as a ``.xml``
topology plus a ``.bin`` of weights, and — when exported by Ultralytics — a
``metadata.yaml`` carrying task, image size and class names.

This module isolates that:

* :func:`inspect` looks at what has been uploaded so far and reports whether it
  is loadable yet, *without touching anything*. That read-only property is what
  lets the API accept a partial upload without disturbing the running model.
* :func:`prepare` does the mutating half — laying an ``.xml`` + ``.bin`` out in
  the ``*_openvino_model/`` directory Ultralytics expects, synthesising
  ``metadata.yaml`` from the IR itself when the user didn't upload one.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import ModelLoadError

log = logging.getLogger("prelabel.loader")

#: Single-file formats an engine can load directly.
PRIMARY_EXTENSIONS = frozenset({".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlpackage", ".pb"})

#: Multi-part: an ``.xml`` is loadable only once its ``.bin`` is present.
OPENVINO_PRIMARY = ".xml"

#: Files that support a primary model rather than being one themselves.
COMPANION_EXTENSIONS = frozenset({".bin", ".yaml", ".yml", ".txt"})

#: Dropped into an assembled OpenVINO directory when we had to default the task.
TASK_ASSUMED_MARKER = ".task_assumed"

ACCEPTED_EXTENSIONS = PRIMARY_EXTENSIONS | {OPENVINO_PRIMARY} | COMPANION_EXTENSIONS


def is_primary(filename: str | None) -> bool:
    """True if uploading this file should start a *fresh* model load."""
    if not filename:
        return False
    suffix = Path(filename).suffix.lower()
    return suffix in PRIMARY_EXTENSIONS or suffix == OPENVINO_PRIMARY


@dataclass(frozen=True)
class LoadPlan:
    """What :func:`inspect` found in an upload directory."""

    #: The primary file an engine would be built from, if one is usable yet.
    primary: Path | None
    #: Human-readable explanation, shown in the UI when not ready.
    message: str

    @property
    def is_ready(self) -> bool:
        return self.primary is not None

    @property
    def status(self) -> str:
        return "ok" if self.is_ready else "waiting"


def _newest(paths: list[Path]) -> Path | None:
    """Most recently modified path, so re-uploading switches models."""
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def inspect(directory: Path) -> LoadPlan:
    """Report whether ``directory`` holds a loadable model. Never mutates."""
    if not directory.exists():
        return LoadPlan(None, "No model uploaded yet.")

    files = [f for f in directory.iterdir() if f.is_file()]
    if not files:
        return LoadPlan(None, "No model uploaded yet.")

    xml_files = [f for f in files if f.suffix.lower() == OPENVINO_PRIMARY]
    single_files = [f for f in files if f.suffix.lower() in PRIMARY_EXTENSIONS]

    newest_xml = _newest(xml_files)
    newest_single = _newest(single_files)

    # Whichever primary arrived most recently is the one the user meant.
    prefer_xml = newest_xml is not None and (
        newest_single is None or newest_xml.stat().st_mtime >= newest_single.stat().st_mtime
    )

    if prefer_xml and newest_xml is not None:
        if not newest_xml.with_suffix(".bin").exists():
            return LoadPlan(None, f"OpenVINO model '{newest_xml.name}' needs its .bin file — upload it too.")
        return LoadPlan(newest_xml, "")

    if newest_single is not None:
        return LoadPlan(newest_single, "")

    uploaded = ", ".join(sorted({f.suffix.lower() or f.name for f in files}))
    return LoadPlan(None, f"No loadable model file yet (got: {uploaded}).")


def prepare(directory: Path) -> Path:
    """Return the path an engine should be pointed at, assembling if needed.

    Raises :class:`~app.errors.ModelLoadError` when ``directory`` is not ready —
    callers are expected to have checked with :func:`inspect` first.
    """
    plan = inspect(directory)
    if plan.primary is None:
        raise ModelLoadError(plan.message or "No loadable model file.")

    if plan.primary.suffix.lower() == OPENVINO_PRIMARY:
        return assemble_openvino(plan.primary)
    return plan.primary


def assemble_openvino(xml_path: Path) -> Path:
    """Arrange an OpenVINO model into the ``*_openvino_model/`` layout.

    A user-provided ``metadata.yaml`` sitting alongside the model is respected;
    otherwise one is synthesised from the IR.
    """
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.exists():
        raise ModelLoadError(f"OpenVINO model '{xml_path.name}' is missing its .bin file")

    ov_dir = xml_path.parent / f"{xml_path.stem}_openvino_model"
    shutil.rmtree(ov_dir, ignore_errors=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(xml_path, ov_dir / xml_path.name)
    shutil.copy(bin_path, ov_dir / f"{xml_path.stem}.bin")

    user_metadata = xml_path.parent / "metadata.yaml"
    if user_metadata.exists():
        shutil.copy(user_metadata, ov_dir / "metadata.yaml")
        log.info("Using user-provided metadata.yaml for OpenVINO model")
    else:
        synthesize_metadata(xml_path, ov_dir)
        # Record that the task was defaulted — a bare IR does not carry it — so
        # the UI can flag the task as assumed rather than confirmed.
        (ov_dir / TASK_ASSUMED_MARKER).write_text("1", encoding="utf-8")

    return ov_dir


def _static_shape(port) -> list[int]:  # noqa: ANN001 - openvino type
    return [dim.get_length() if dim.is_static else -1 for dim in port.get_partial_shape()]


def synthesize_metadata(xml_path: Path, dest_dir: Path) -> dict:
    """Build a minimal ``metadata.yaml`` by reading the OpenVINO IR.

    Ultralytics refuses to load an OpenVINO model without metadata, so when the
    user only supplies ``.xml`` + ``.bin`` we reconstruct what the IR can tell us:

    * ``imgsz`` from the input tensor shape
    * ``names`` from ``rt_info['model_info']['labels']``, falling back to
      ``class_<i>`` derived from the output channel count
    * ``task`` assumed ``detect`` — the common case; segmentation and pose exports
      carry their own metadata, which the user can supply
    """
    import openvino as ov

    model = ov.Core().read_model(str(xml_path))

    input_shape = _static_shape(model.inputs[0])
    imgsz = input_shape[2] if len(input_shape) == 4 and input_shape[2] > 0 else 640

    names = _labels_from_rt_info(model)
    if not names:
        output_shape = _static_shape(model.outputs[0])
        # A YOLO detect head has 4 box channels plus one per class.
        num_classes = max(1, output_shape[1] - 4) if len(output_shape) >= 2 and output_shape[1] > 0 else 1
        names = {index: f"class_{index}" for index in range(num_classes)}

    metadata = {
        "task": "detect",
        "batch": 1,
        "imgsz": [imgsz, imgsz],
        "stride": 32,
        "names": names,
    }
    with open(dest_dir / "metadata.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)

    log.info("Synthesized OpenVINO metadata: imgsz=%d, %d classes", imgsz, len(names))
    return metadata


def _labels_from_rt_info(model) -> dict[int, str]:  # noqa: ANN001 - openvino type
    """Class names recorded in the IR, if the exporter wrote any."""
    try:
        labels = model.get_rt_info(["model_info", "labels"]).astype(str)
    except Exception:  # noqa: BLE001 - rt_info is best-effort and often absent
        return {}
    parts = [part for part in labels.split(" ") if part]
    return dict(enumerate(parts))

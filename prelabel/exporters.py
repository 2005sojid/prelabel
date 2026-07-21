"""Writing a project out as a dataset.

Exporting server-side rather than in the browser is what lifts the ceiling: the
archive is streamed to a temporary file with Python's ``zipfile``, which switches
to ZIP64 on its own, so a hundred-gigabyte project is no different from a small
one. The browser only ever downloads the result.

Formats are the standard interchange ones — COCO for the box-shaped tasks,
ImageNet folders for classification. Anything downstream (CVAT, Label Studio,
Roboflow, a training script) reads at least one of them.
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import datasets
from .errors import UnsupportedMedia
from .store import Item, Project, Store

log = logging.getLogger("prelabel.exporters")

IMAGE_ROOT = "images/default/"

#: Which exporter suits which task.
FORMATS_BY_TASK = {
    "detect": ("coco",),
    "segment": ("coco",),
    "obb": ("coco",),
    "pose": ("coco_keypoints",),
    "classify": ("imagenet",),
}

FORMAT_LABELS = {
    "coco": "COCO 1.0",
    "coco_keypoints": "COCO Keypoints 1.0",
    "imagenet": "ImageNet folders",
}


@dataclass
class ExportSummary:
    path: Path
    format: str
    images: int
    annotations: int


def formats_for(task: str) -> list[dict[str, str]]:
    """Exporters offered for a task, as the UI presents them."""
    names = FORMATS_BY_TASK.get(task, FORMATS_BY_TASK["detect"])
    return [{"id": name, "label": FORMAT_LABELS[name]} for name in names]


def _round(value: float) -> float:
    return round(float(value), 2)


def _visible(item: Item, threshold: float) -> list[dict[str, Any]]:
    return [d for d in item.detections if float(d.get("confidence", 1.0)) >= threshold]


def _unique_names(items: Iterator[Item]) -> Iterator[tuple[Item, str]]:
    """Yield each item with an archive-unique file name.

    Two folders can hold a ``frame_0001.jpg`` each. Flattening them into one
    archive without renaming means one silently replaces the other, so a
    collision gets the folder path folded into the name.
    """
    seen: set[str] = set()
    for item in items:
        name = Path(item.rel_path).name
        if name.lower() in seen:
            name = item.rel_path.replace("/", "__")
        seen.add(name.lower())
        yield item, name


def _class_index(store: Store, project: Project) -> dict[str, int]:
    """Class name to id, preferring the model's own ordering."""
    index: dict[str, int] = {}
    for position, name in enumerate((project.model or {}).get("class_names", {}).values()):
        index.setdefault(str(name), position)
    for entry in store.class_counts(project.id):
        index.setdefault(entry["class_name"], len(index))
    return index


def export_project(
    store: Store,
    project: Project,
    destination: Path,
    fmt: str = "coco",
    threshold: float = 0.0,
) -> ExportSummary:
    """Write ``project`` to ``destination`` as a dataset archive."""
    if fmt not in FORMAT_LABELS:
        raise UnsupportedMedia(f"Unknown export format '{fmt}'. Available: {', '.join(FORMAT_LABELS)}")

    root = datasets.resolve_dataset_dir(project.source_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        if fmt == "imagenet":
            summary = _write_imagenet(store, project, root, archive, threshold)
        else:
            summary = _write_coco(
                store, project, root, archive, threshold, keypoints=(fmt == "coco_keypoints")
            )

    log.info("Exported project %s as %s: %d images, %d annotations",
             project.id, fmt, summary["images"], summary["annotations"])
    return ExportSummary(destination, fmt, summary["images"], summary["annotations"])


def _add_image(archive: zipfile.ZipFile, root: Path, item: Item, arcname: str) -> bool:
    """Copy the source image into the archive. False when it has gone missing."""
    try:
        source = datasets.resolve_item_path(root, item.rel_path)
    except Exception as exc:  # noqa: BLE001 - a deleted file must not fail the export
        log.warning("Skipping %s: %s", item.rel_path, exc)
        return False
    archive.write(source, arcname)
    return True


def _write_coco(
    store: Store,
    project: Project,
    root: Path,
    archive: zipfile.ZipFile,
    threshold: float,
    keypoints: bool,
) -> dict[str, int]:
    index = _class_index(store, project)
    keypoint_count = 0
    if keypoints:
        keypoint_count = max(
            (len(d.get("keypoints") or []) for item in store.iter_done_items(project.id) for d in item.detections),
            default=0,
        )

    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for image_id, (item, arcname) in enumerate(_unique_names(store.iter_done_items(project.id)), start=1):
        if not _add_image(archive, root, item, IMAGE_ROOT + arcname):
            continue
        images.append({"id": image_id, "file_name": arcname, "width": item.width, "height": item.height})

        for detection in _visible(item, threshold):
            box = detection.get("box")
            if not box:
                continue
            x1, y1, x2, y2 = box
            record: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_id,
                # COCO category ids are 1-based.
                "category_id": index.get(detection.get("class_name", "?"), 0) + 1,
                "bbox": [_round(x1), _round(y1), _round(x2 - x1), _round(y2 - y1)],
                "area": _round((x2 - x1) * (y2 - y1)),
                "iscrowd": 0,
                "segmentation": [],
            }
            if detection.get("mask"):
                record["segmentation"] = [[_round(v) for point in detection["mask"] for v in point]]
            if keypoints:
                flat: list[float] = []
                visible = 0
                for point in detection.get("keypoints") or []:
                    score = point[2] if len(point) > 2 else 0
                    visibility = 2 if score > 0.5 else 1 if score > 0 else 0
                    visible += 1 if visibility else 0
                    flat.extend([_round(point[0]), _round(point[1]), visibility])
                record["keypoints"] = flat
                record["num_keypoints"] = visible
            annotations.append(record)
            annotation_id += 1

    categories = [
        {
            "id": position + 1,
            "name": name,
            "supercategory": "",
            **({"keypoints": [f"kp{i}" for i in range(keypoint_count)], "skeleton": []} if keypoints else {}),
        }
        for name, position in sorted(index.items(), key=lambda kv: kv[1])
    ]

    document = {
        "info": {
            "description": f"Prelabel auto-annotations — {project.name}",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    filename = "person_keypoints_default.json" if keypoints else "instances_default.json"
    archive.writestr(f"annotations/{filename}", json.dumps(document))
    return {"images": len(images), "annotations": len(annotations)}


def _write_imagenet(
    store: Store,
    project: Project,
    root: Path,
    archive: zipfile.ZipFile,
    threshold: float,
) -> dict[str, int]:
    written = 0
    for item, arcname in _unique_names(store.iter_done_items(project.id)):
        visible = _visible(item, threshold)
        label = visible[0]["class_name"] if visible else "unknown"
        folder = "".join(c if c.isalnum() or c in " ._-" else "_" for c in label) or "unknown"
        if _add_image(archive, root, item, f"{folder}/{arcname}"):
            written += 1
    return {"images": written, "annotations": written}

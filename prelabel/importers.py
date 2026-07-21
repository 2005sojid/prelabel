"""Reading annotations back in.

Export alone is a one-way door: once a dataset leaves for CVAT there is no way to
bring the corrected version back, compare it, or re-export it in another format.
Importing closes the loop — corrections made downstream become the project's
stored annotations.

COCO is the only format read here, because it is the one every tool in this
chain writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import UnsupportedMedia
from .store import Store

log = logging.getLogger("prelabel.importers")


@dataclass
class ImportSummary:
    matched: int
    unmatched: int
    annotations: int
    #: File names present in the archive that no project item corresponds to.
    unknown_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "unmatched": self.unmatched,
            "annotations": self.annotations,
            "unknown_files": self.unknown_files[:20],
            "unknown_file_count": len(self.unknown_files),
        }


def parse_coco(raw: bytes) -> dict[str, Any]:
    """Parse and sanity-check a COCO document."""
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UnsupportedMedia(f"Not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise UnsupportedMedia("A COCO file must be a JSON object")
    for key in ("images", "annotations"):
        if not isinstance(document.get(key), list):
            raise UnsupportedMedia(f"A COCO file needs an '{key}' array")
    return document


def _match_key(name: str) -> str:
    """Key a COCO ``file_name`` so it matches however the project stored it.

    Exporters flatten folders, importers may prefix them, and case differs
    between platforms. Matching on the lowercased base name is the one thing that
    survives all of those.
    """
    return Path(str(name).replace("\\", "/")).name.lower()


def import_coco(
    store: Store,
    project_id: str,
    raw: bytes,
    replace: bool = True,
    into: str = "current",
) -> ImportSummary:
    """Apply COCO annotations to a project's items.

    Items are matched by file name. ``replace=True`` overwrites an item's
    detections; ``False`` appends, which is how you merge a second pass over the
    same folder.

    ``into="baseline"`` writes to the comparison set instead of the working one,
    which is what turns corrected labels into something you can diff the model
    against rather than something that overwrites it.
    """
    document = parse_coco(raw)

    categories = {
        int(category["id"]): str(category.get("name", f"class_{category['id']}"))
        for category in document.get("categories", [])
        if "id" in category
    }
    images = {
        int(image["id"]): image
        for image in document["images"]
        if "id" in image and "file_name" in image
    }

    by_image: dict[int, list[dict]] = {}
    for annotation in document["annotations"]:
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        by_image.setdefault(int(image_id), []).append(annotation)

    # Index the project's items once; a project can hold a hundred thousand.
    items_by_key: dict[str, Any] = {}
    offset = 0
    while True:
        page = store.list_items(project_id, offset=offset, limit=1000, with_detections=False)
        if not page:
            break
        for item in page:
            items_by_key.setdefault(_match_key(item.rel_path), item)
        offset += len(page)

    matched = 0
    total_annotations = 0
    unknown: list[str] = []

    for image_id, image in images.items():
        key = _match_key(image["file_name"])
        item = items_by_key.get(key)
        if item is None:
            unknown.append(str(image["file_name"]))
            continue

        detections = [
            converted
            for annotation in by_image.get(image_id, [])
            if (converted := _to_detection(annotation, categories)) is not None
        ]
        existing = store.get_item(project_id, item.id)
        if not replace:
            previous = (existing.baseline if into == "baseline" else existing.detections) if existing else []
            detections = previous + detections

        if into == "baseline":
            store.set_baseline(project_id, item.id, detections)
        else:
            width = int(image.get("width") or item.width or 0)
            height = int(image.get("height") or item.height or 0)
            store.save_result(
                project_id, item.id,
                width=width, height=height,
                task=item.task or "detect",
                inference_ms=item.inference_ms,
                review_priority=0.0,  # imported labels are ground truth, not predictions
                detections=detections,
            )
        matched += 1
        total_annotations += len(detections)

    log.info("Imported COCO into %s: %d images matched, %d unmatched, %d annotations",
             project_id, matched, len(unknown), total_annotations)
    return ImportSummary(
        matched=matched,
        unmatched=len(unknown),
        annotations=total_annotations,
        unknown_files=unknown,
    )


def _to_detection(annotation: dict, categories: dict[int, str]) -> dict[str, Any] | None:
    """Convert one COCO annotation into this project's detection shape."""
    bbox = annotation.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    x, y, width, height = (float(v) for v in bbox[:4])

    category_id = annotation.get("category_id")
    detection: dict[str, Any] = {
        "class_name": categories.get(int(category_id), f"class_{category_id}") if category_id is not None else "object",
        # Imported annotations are corrections, not guesses: certainty is the point.
        "confidence": 1.0,
        "kind": "box",
        "box": [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)],
    }

    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        flat = segmentation[0]
        detection["mask"] = [[round(flat[i], 2), round(flat[i + 1], 2)] for i in range(0, len(flat) - 1, 2)]
        detection["kind"] = "segmentation"

    keypoints = annotation.get("keypoints")
    if isinstance(keypoints, list) and len(keypoints) >= 3:
        detection["keypoints"] = [
            [round(keypoints[i], 2), round(keypoints[i + 1], 2), 1.0 if keypoints[i + 2] else 0.0]
            for i in range(0, len(keypoints) - 2, 3)
        ]
        detection["kind"] = "pose"

    return detection

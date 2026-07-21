"""Sliced inference for small objects in large images.

A detector runs at a fixed input size — 640 px is typical. Feed it a 4000 px
aerial photo and everything is scaled down by six before the model ever sees it,
so a 30-pixel car becomes 5 pixels and disappears. This is the single most common
reason a good model "doesn't work" on high-resolution imagery.

The fix is to cut the image into overlapping tiles, run the model at native
resolution on each, and merge the results back. Overlap matters: an object
straddling a tile boundary is cut in half in both tiles, so tiles have to share a
margin wide enough for at least one of them to contain the whole object.

Works with *any* engine — it only uses :meth:`predict_batch`, so a new backend
gets sliced inference for free.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .base import BaseEngine, Detection, InferenceResult, Timings

log = logging.getLogger("prelabel.engines.tiling")

#: Fraction of a tile shared with its neighbour. 20% of 640 px is 128 px, which
#: comfortably contains the objects sliced inference is meant to find.
DEFAULT_OVERLAP = 0.2

#: IoU above which two detections from different tiles are treated as one object.
DEFAULT_MERGE_IOU = 0.5

#: Refuse to slice into more than this many tiles — a guard against someone
#: pointing a 100-megapixel scan at a 320 px tile size.
MAX_TILES = 256


@dataclass(frozen=True)
class Tile:
    """One crop, and where it sits in the source image."""

    x: int
    y: int
    width: int
    height: int

    @property
    def offset(self) -> tuple[int, int]:
        return self.x, self.y


def plan_tiles(width: int, height: int, tile: int, overlap: float = DEFAULT_OVERLAP) -> list[Tile]:
    """Lay out overlapping tiles covering the whole image.

    The final tile in each direction is pulled back to the edge rather than
    hanging off it, so every pixel is covered exactly once or twice and no tile is
    a partial, padded crop.
    """
    if tile <= 0:
        raise ValueError("tile size must be positive")
    step = max(1, int(round(tile * (1.0 - min(max(overlap, 0.0), 0.9)))))

    def origins(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        positions = list(range(0, extent - tile + 1, step))
        if positions[-1] != extent - tile:
            positions.append(extent - tile)
        return positions

    return [
        Tile(x, y, min(tile, width), min(tile, height))
        for y in origins(height)
        for x in origins(width)
    ]


def should_tile(image: np.ndarray, tile: int) -> bool:
    """True when the image is big enough for slicing to be worth it."""
    height, width = image.shape[:2]
    return max(height, width) > tile


def _crops(image: np.ndarray, tiles: Sequence[Tile]) -> Iterator[np.ndarray]:
    for t in tiles:
        yield image[t.y : t.y + t.height, t.x : t.x + t.width]


def _shift(detection: Detection, dx: int, dy: int) -> Detection:
    """Move a detection from tile coordinates into full-image coordinates."""
    box = None
    if detection.box is not None:
        x1, y1, x2, y2 = detection.box
        box = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    mask = None
    if detection.mask is not None:
        mask = [[px + dx, py + dy] for px, py in detection.mask]

    keypoints = None
    if detection.keypoints is not None:
        keypoints = [[kx + dx, ky + dy, *rest] for kx, ky, *rest in detection.keypoints]

    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        kind=detection.kind,
        box=box,
        mask=mask,
        keypoints=keypoints,
        class_id=detection.class_id,
        track_id=detection.track_id,
    )


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections: list[Detection], iou_threshold: float = DEFAULT_MERGE_IOU) -> list[Detection]:
    """Collapse duplicates produced by overlapping tiles.

    Greedy non-maximum suppression, applied **per class**: two different classes
    on the same object are a legitimate disagreement to surface, not a duplicate
    to hide. Detections without a box (classification) pass through untouched —
    there is no geometry to compare.
    """
    boxed = [d for d in detections if d.box]
    passthrough = [d for d in detections if not d.box]

    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = {}
    for detection in boxed:
        by_class.setdefault(detection.class_name, []).append(detection)

    for group in by_class.values():
        group.sort(key=lambda d: d.confidence, reverse=True)
        survivors: list[Detection] = []
        for candidate in group:
            if all(_iou(candidate.box, other.box) < iou_threshold for other in survivors):
                survivors.append(candidate)
        kept.extend(survivors)

    kept.sort(key=lambda d: d.confidence, reverse=True)
    return passthrough + kept


def predict_tiled(
    engine: BaseEngine,
    image: np.ndarray,
    conf: float = 0.25,
    classes: Sequence[int] | None = None,
    tile: int | None = None,
    overlap: float = DEFAULT_OVERLAP,
    merge_iou: float = DEFAULT_MERGE_IOU,
    include_full_image: bool = True,
) -> InferenceResult:
    """Run ``engine`` over overlapping tiles and merge the results.

    ``tile`` defaults to the engine's own input size, which is the point: each
    tile is fed at 1:1 scale instead of being shrunk to fit.

    ``include_full_image`` adds one pass over the whole (downscaled) image. Tiles
    find small objects; the full pass finds objects *larger* than a tile, which
    slicing alone would only ever see fragments of. The merge step reconciles the
    two.
    """
    tile_size = int(tile or getattr(engine, "imgsz", 640) or 640)
    height, width = image.shape[:2]

    if not should_tile(image, tile_size):
        return engine.predict(image, conf=conf, classes=classes)

    tiles = plan_tiles(width, height, tile_size, overlap)
    if len(tiles) > MAX_TILES:
        log.warning(
            "Image %dx%d would need %d tiles at %dpx; falling back to a single pass. "
            "Raise the tile size to slice it.",
            width, height, len(tiles), tile_size,
        )
        return engine.predict(image, conf=conf, classes=classes)

    results = engine.predict_batch(list(_crops(image, tiles)), conf=conf, classes=classes)

    merged: list[Detection] = []
    total = Timings()
    for tile_spec, result in zip(tiles, results, strict=True):
        dx, dy = tile_spec.offset
        merged.extend(_shift(d, dx, dy) for d in result.detections)
        total.preprocess_ms += result.timings.preprocess_ms
        total.inference_ms += result.timings.inference_ms
        total.postprocess_ms += result.timings.postprocess_ms

    if include_full_image:
        overview = engine.predict(image, conf=conf, classes=classes)
        merged.extend(overview.detections)
        total.preprocess_ms += overview.timings.preprocess_ms
        total.inference_ms += overview.timings.inference_ms
        total.postprocess_ms += overview.timings.postprocess_ms

    first = results[0] if results else None
    return InferenceResult(
        detections=merge_detections(merged, merge_iou),
        task=first.task if first else "detect",
        timings=total,
        device=first.device if first else "cpu",
        image_shape=[height, width],
    )

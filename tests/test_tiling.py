"""Sliced inference.

The behaviour that matters: an object too small to survive being downscaled to
the model's input size must be found, and the same object seen in two
overlapping tiles must come back once, in full-image coordinates.
"""

from __future__ import annotations

import numpy as np
import pytest

from prelabel.engines.base import Detection, InferenceResult, Timings
from prelabel.engines.tiling import (
    DEFAULT_OVERLAP,
    MAX_TILES,
    merge_detections,
    plan_tiles,
    predict_tiled,
    should_tile,
)


class GridEngine:
    """Engine double that reports one detection at a fixed spot in every tile.

    Lets a test reason about coordinates exactly: whatever the tile, the
    detection sits at (10,10)-(30,30) *within* it, so the merged result must show
    it at the tile's offset plus that.
    """

    imgsz = 64

    def __init__(self, per_tile=1, class_name="car", confidence=0.9):
        self.per_tile = per_tile
        self.class_name = class_name
        self.confidence = confidence
        self.calls = 0
        self.batch_sizes = []
        self.last_classes = None

    def _result(self, image):
        return InferenceResult(
            detections=[
                Detection(self.class_name, self.confidence, "box", box=[10, 10, 30, 30])
                for _ in range(self.per_tile)
            ],
            task="detect",
            timings=Timings(preprocess_ms=1, inference_ms=2, postprocess_ms=1),
            device="CPU",
            image_shape=list(image.shape[:2]),
        )

    def predict(self, image, conf=0.25, classes=None):
        self.calls += 1
        self.last_classes = classes
        return self._result(image)

    def predict_batch(self, images, conf=0.25, classes=None):
        self.batch_sizes.append(len(images))
        self.last_classes = classes
        return [self._result(image) for image in images]


# --- planning ---------------------------------------------------------------


def test_a_small_image_needs_one_tile():
    assert plan_tiles(50, 50, tile=64) == plan_tiles(50, 50, tile=64)
    assert len(plan_tiles(50, 50, tile=64)) == 1


def test_tiles_cover_every_pixel():
    width, height, tile = 300, 200, 64
    tiles = plan_tiles(width, height, tile, overlap=DEFAULT_OVERLAP)

    covered = np.zeros((height, width), dtype=bool)
    for t in tiles:
        covered[t.y : t.y + t.height, t.x : t.x + t.width] = True
    assert covered.all(), "some pixels are in no tile at all"


def test_tiles_stay_inside_the_image():
    width, height, tile = 300, 200, 64
    for t in plan_tiles(width, height, tile):
        assert t.x >= 0 and t.y >= 0
        assert t.x + t.width <= width
        assert t.y + t.height <= height


def test_tiles_overlap_their_neighbours():
    """Without overlap, an object on a seam is cut in half in every tile."""
    tiles = plan_tiles(200, 64, tile=64, overlap=0.25)
    xs = sorted({t.x for t in tiles})
    gaps = [b - a for a, b in zip(xs, xs[1:], strict=False)]
    assert all(gap < 64 for gap in gaps), f"tiles do not overlap: origins {xs}"


def test_zero_overlap_is_allowed():
    tiles = plan_tiles(128, 64, tile=64, overlap=0.0)
    assert sorted({t.x for t in tiles}) == [0, 64]


def test_invalid_tile_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        plan_tiles(100, 100, tile=0)


def test_should_tile_only_for_images_above_the_model_size():
    assert should_tile(np.zeros((32, 32, 3), np.uint8), 64) is False
    assert should_tile(np.zeros((64, 64, 3), np.uint8), 64) is False
    assert should_tile(np.zeros((65, 64, 3), np.uint8), 64) is True


# --- merging ----------------------------------------------------------------


def test_overlapping_duplicates_collapse_to_one():
    detections = [
        Detection("car", 0.9, "box", box=[10, 10, 50, 50]),
        Detection("car", 0.8, "box", box=[12, 12, 52, 52]),  # same object, other tile
    ]
    merged = merge_detections(detections, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].confidence == 0.9, "the more confident duplicate should win"


def test_distinct_objects_are_kept():
    detections = [
        Detection("car", 0.9, "box", box=[10, 10, 50, 50]),
        Detection("car", 0.8, "box", box=[200, 200, 240, 240]),
    ]
    assert len(merge_detections(detections)) == 2


def test_different_classes_on_the_same_spot_both_survive():
    """A genuine disagreement between classes is information, not a duplicate."""
    detections = [
        Detection("car", 0.9, "box", box=[10, 10, 50, 50]),
        Detection("truck", 0.8, "box", box=[10, 10, 50, 50]),
    ]
    assert len(merge_detections(detections)) == 2


def test_classification_results_pass_through():
    detections = [Detection("cat", 0.9, "classification"), Detection("dog", 0.4, "classification")]
    assert len(merge_detections(detections)) == 2


# --- end to end -------------------------------------------------------------


def test_tiling_is_skipped_for_a_small_image():
    engine = GridEngine()
    image = np.zeros((40, 40, 3), np.uint8)
    predict_tiled(engine, image, tile=64)
    assert engine.calls == 1
    assert engine.batch_sizes == [], "a small image must not be sliced"


def test_detections_come_back_in_full_image_coordinates():
    """The core of it: a tile-local box must be translated to the source image."""
    engine = GridEngine(per_tile=1)
    image = np.zeros((128, 128, 3), np.uint8)

    result = predict_tiled(engine, image, tile=64, overlap=0.0, include_full_image=False)

    # Tiles at (0,0),(64,0),(0,64),(64,64); each reports a box at (10,10)-(30,30).
    found = sorted(tuple(d.box) for d in result.detections)
    assert found == [
        (10.0, 10.0, 30.0, 30.0),
        (10.0, 74.0, 30.0, 94.0),
        (74.0, 10.0, 94.0, 30.0),
        (74.0, 74.0, 94.0, 94.0),
    ]


def test_a_full_image_pass_is_included_by_default():
    """Objects larger than a tile are only ever seen whole by the overview pass."""
    engine = GridEngine()
    image = np.zeros((128, 128, 3), np.uint8)
    predict_tiled(engine, image, tile=64, overlap=0.0)
    assert engine.calls == 1, "expected exactly one whole-image pass"


def test_timings_are_summed_across_tiles():
    engine = GridEngine()
    image = np.zeros((128, 128, 3), np.uint8)
    result = predict_tiled(engine, image, tile=64, overlap=0.0, include_full_image=False)
    assert result.timings.inference_ms == pytest.approx(4 * 2)  # 4 tiles x 2 ms


def test_image_shape_is_the_source_not_a_tile():
    engine = GridEngine()
    image = np.zeros((100, 150, 3), np.uint8)
    result = predict_tiled(engine, image, tile=64, include_full_image=False)
    assert result.image_shape == [100, 150]


def test_class_filter_reaches_the_engine():
    engine = GridEngine()
    image = np.zeros((128, 128, 3), np.uint8)
    predict_tiled(engine, image, tile=64, classes=[0, 2], include_full_image=False)
    assert engine.last_classes == [0, 2]


def test_an_absurd_tile_count_falls_back_to_one_pass():
    """A guard against slicing a huge scan into thousands of crops."""
    engine = GridEngine()
    image = np.zeros((4000, 4000, 3), np.uint8)
    result = predict_tiled(engine, image, tile=32)
    assert engine.batch_sizes == [], f"expected no slicing beyond {MAX_TILES} tiles"
    assert result.image_shape == [4000, 4000]


def test_tiles_are_sent_as_one_batch():
    """Slicing must not become one HTTP-style round trip per tile."""
    engine = GridEngine()
    image = np.zeros((256, 256, 3), np.uint8)
    predict_tiled(engine, image, tile=64, overlap=0.0, include_full_image=False)
    assert len(engine.batch_sizes) == 1
    assert engine.batch_sizes[0] == 16

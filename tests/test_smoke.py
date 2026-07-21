"""Unit tests for the pure pieces: result types, engine selection, drawing.

No model weights, no network — these must stay fast enough to run on every save.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from prelabel.engines import SUPPORTED_EXTENSIONS, build_engine
from prelabel.engines.base import Detection, InferenceResult, ModelInfo, Timings
from prelabel.media.drawing import color_for, draw

# --- result model -----------------------------------------------------------


def test_detection_serialisation_rounds_and_omits_absent_fields():
    detection = Detection(class_name="car", confidence=0.91234, kind="box", box=[1.111, 2.0, 3.0, 4.0])
    data = detection.to_dict()

    assert data["class_name"] == "car"
    assert data["confidence"] == 0.9123          # 4 dp
    assert data["box"] == [1.11, 2.0, 3.0, 4.0]  # 2 dp
    assert "mask" not in data
    assert "keypoints" not in data


def test_detection_carries_masks_and_keypoints_when_present():
    detection = Detection("person", 0.8, "pose", box=[0, 0, 1, 1], keypoints=[[1.0, 2.0, 0.9]])
    assert detection.to_dict()["keypoints"] == [[1.0, 2.0, 0.9]]


def test_timings_total_is_the_sum_of_its_stages():
    timings = Timings(preprocess_ms=1.0, inference_ms=10.0, postprocess_ms=2.0)
    assert timings.total_ms == 13.0
    assert timings.to_dict()["total_ms"] == 13.0


def test_result_serialisation():
    result = InferenceResult(
        detections=[Detection("a", 0.5), Detection("b", 0.4)],
        task="detect",
        timings=Timings(preprocess_ms=1.0, inference_ms=12.345, postprocess_ms=0.5),
        image_shape=[480, 640],
    )
    data = result.to_dict()

    assert data["count"] == 2
    assert data["speed_ms"] == 12.35            # headline is the pure model time
    assert data["timings"]["total_ms"] == 13.845
    assert result.speed_ms == 12.345
    json.dumps(data)                            # must survive the wire


def test_model_info_serialisation():
    info = ModelInfo("m.pt", "ultralytics", "PyTorch", "detect", 640, 2, {0: "a", 1: "b"})
    data = info.to_dict()
    assert data["num_classes"] == 2
    assert data["task_assumed"] is False


# --- engine selection -------------------------------------------------------


def test_supported_extensions_cover_the_common_formats():
    for extension in (".pt", ".onnx", ".xml", ".engine", ".tflite", ".torchscript"):
        assert extension in SUPPORTED_EXTENSIONS


def test_build_engine_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="Unsupported"):
        build_engine("model.xyz")


def test_build_engine_lists_what_it_does_support():
    with pytest.raises(ValueError) as failure:
        build_engine("model.xyz")
    assert ".pt" in str(failure.value)


def test_an_engine_that_cannot_infer_is_rejected_at_construction():
    """Constructing is not the same as working, and the difference matters.

    A backend can accept a file with the right extension, populate itself with
    nonsense and only fall over on the first real image — which is exactly what
    a plain ResNet handed to a YOLO loader does. The self-test turns that into a
    load failure, which is what lets the factory try the next backend.
    """
    from prelabel.engines.base import BaseEngine, ModelInfo

    class BrokenEngine(BaseEngine):
        extensions = {".broken"}
        backend_name = "broken"

        def load(self):
            self.imgsz = 32

        def predict(self, image, conf=0.25, classes=None):
            raise RuntimeError("input name mismatch")

        def info(self):
            return ModelInfo("broken", "broken", "?", "detect", 32, 0, {})

    with pytest.raises(RuntimeError, match="input name mismatch"):
        BrokenEngine("whatever.broken")


def test_selftest_passes_for_a_working_engine():
    from prelabel.engines.base import BaseEngine, ModelInfo

    class WorkingEngine(BaseEngine):
        extensions = {".ok"}

        def load(self):
            self.imgsz = 32
            self.seen = 0

        def predict(self, image, conf=0.25, classes=None):
            self.seen += 1
            return InferenceResult(task="detect")

        def info(self):
            return ModelInfo("ok", "ok", "?", "detect", 32, 0, {})

    engine = WorkingEngine("whatever.ok", warmup=False)
    assert engine.seen == 1, "the self-test should run exactly one inference"


# --- drawing ----------------------------------------------------------------


def test_class_colours_are_stable_and_distinct():
    assert color_for("car") == color_for("car")
    assert color_for("car") != color_for("truck")


def test_colour_matches_the_frontend_hash():
    """The frontend computes the same hue; a drift would show as mismatched colours.

    Mirrors `colorFor()` in assets/js/colors.js: a rolling hash into hsl(h,90%,60%).
    """
    import colorsys

    name = "person"
    hue = 0
    for char in name:
        hue = (hue * 31 + ord(char)) % 360
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.90)
    assert color_for(name) == (int(blue * 255), int(green * 255), int(red * 255))


def test_draw_returns_a_new_frame_and_leaves_the_original_alone():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    result = InferenceResult(
        detections=[Detection("car", 0.9, "box", box=[10, 10, 60, 50])],
        task="detect",
    )
    drawn = draw(frame, result)

    assert drawn is not frame
    assert frame.sum() == 0, "the source frame was mutated"
    assert drawn.sum() > 0, "nothing was drawn"


def test_draw_keeps_a_label_inside_the_frame():
    """A box flush against the top edge must not push its caption off-screen."""
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    result = InferenceResult(detections=[Detection("car", 0.9, "box", box=[5, 0, 60, 40])], task="detect")
    drawn = draw(frame, result)
    assert drawn[0:12, 5:60].sum() > 0, "the label was drawn outside the frame"


def test_draw_shows_the_track_id_when_there_is_one():
    """Without the id on the frame, a tracked video looks exactly like an untracked one.

    That was the whole gap: tracking set ``track_id`` in the data model and
    nothing the user could see ever mentioned it.
    """
    import cv2

    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    tracked = InferenceResult(
        detections=[Detection("car", 0.9, "box", box=[20, 30, 120, 100], track_id=7)],
        task="detect",
    )
    plain = InferenceResult(
        detections=[Detection("car", 0.9, "box", box=[20, 30, 120, 100])],
        task="detect",
    )

    with_id = draw(frame, tracked)
    without_id = draw(frame, plain)

    assert not np.array_equal(with_id, without_id), "the track id is not rendered"
    # The label grows by the "#7 " prefix, so it needs more room than the plain one.
    scale, thickness = max(0.4, 120 / 1200), max(1, int(120 / 500))
    plain_width = cv2.getTextSize("car 90%", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    id_width = cv2.getTextSize("#7 car 90%", cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    assert id_width > plain_width


def test_draw_handles_every_task_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cases = [
        InferenceResult(detections=[Detection("a", 0.9, "classification")], task="classify"),
        InferenceResult(
            detections=[Detection("b", 0.9, "segmentation", box=[1, 1, 50, 50], mask=[[1, 1], [40, 1], [40, 40]])],
            task="segment",
        ),
        InferenceResult(
            detections=[Detection("c", 0.9, "pose", box=[1, 1, 50, 50], keypoints=[[5.0, 5.0, 0.9]] * 17)],
            task="pose",
        ),
        InferenceResult(detections=[], task="detect"),
    ]
    for result in cases:
        assert draw(frame, result).shape == frame.shape

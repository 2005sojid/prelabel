"""Burning detections onto frames, server-side.

Used for the video path (the single-image and batch views draw in the browser
instead). Class colours are computed with the *same* hash the frontend uses, so a
class looks identical whether you see it on the canvas or in a rendered MP4.
"""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np

from ..engines.base import Detection, InferenceResult

BGR = tuple[int, int, int]

#: COCO-style skeleton for 17-keypoint pose models.
POSE_SKELETON: Sequence[tuple[int, int]] = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)

#: Minimum keypoint score before a joint is drawn.
KEYPOINT_THRESHOLD = 0.3

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ACCENT: BGR = (162, 255, 0)     # matches the UI accent, in BGR
_KEYPOINT_COLOR: BGR = (0, 140, 255)
_MASK_ALPHA = 0.35


def color_for(name: str) -> BGR:
    """Stable BGR colour for a class name.

    Mirrors the frontend's `colorFor()` exactly — same rolling hash, same
    ``hsl(h, 90%, 60%)`` — so colours agree between the browser canvas and a
    server-rendered video.
    """
    hue = 0
    for char in name:
        hue = (hue * 31 + ord(char)) % 360
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.60, 0.90)
    return int(blue * 255), int(green * 255), int(red * 255)


def _text_scale(height: int) -> tuple[float, int]:
    """Font scale and line thickness proportional to the frame height."""
    return max(0.4, height / 1200), max(1, int(height / 500))


def draw(frame: np.ndarray, result: InferenceResult) -> np.ndarray:
    """Return a copy of ``frame`` with every detection drawn on it."""
    canvas = frame.copy()
    scale, thickness = _text_scale(canvas.shape[0])

    if result.task == "classify":
        return _draw_classification(canvas, result.detections, scale, thickness)

    for detection in result.detections:
        color = color_for(detection.class_name)
        if detection.mask:
            canvas = _draw_mask(canvas, detection.mask, color, thickness)
        if detection.box:
            _draw_box(canvas, detection, color, scale, thickness)
        if detection.keypoints:
            _draw_pose(canvas, detection.keypoints, thickness)
    return canvas


def _draw_classification(
    canvas: np.ndarray,
    detections: list[Detection],
    scale: float,
    thickness: int,
) -> np.ndarray:
    """Stamp the top labels in the corner — there is no box to attach them to."""
    y = int(30 * scale) + 10
    for detection in detections[:5]:
        label = f"{detection.class_name}: {detection.confidence * 100:.1f}%"
        cv2.putText(canvas, label, (10, y), _FONT, scale, _ACCENT, thickness, cv2.LINE_AA)
        y += int(34 * scale)
    return canvas


def _draw_mask(canvas: np.ndarray, polygon: list[list[float]], color: BGR, thickness: int) -> np.ndarray:
    points = np.array(polygon, dtype=np.int32)
    if points.size == 0:
        return canvas
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [points], color)
    blended = cv2.addWeighted(overlay, _MASK_ALPHA, canvas, 1.0 - _MASK_ALPHA, 0)
    cv2.polylines(blended, [points], True, color, thickness, cv2.LINE_AA)
    return blended


def _draw_box(
    canvas: np.ndarray,
    detection: Detection,
    color: BGR,
    scale: float,
    thickness: int,
) -> None:
    x1, y1, x2, y2 = (int(value) for value in detection.box or ())
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    # The track id is the whole point of tracking: without it on the frame, a
    # tracked video is indistinguishable from an untracked one.
    identity = f"#{detection.track_id} " if detection.track_id is not None else ""
    label = f"{identity}{detection.class_name} {detection.confidence * 100:.0f}%"
    (text_width, text_height), _ = cv2.getTextSize(label, _FONT, scale, thickness)
    # Keep the label inside the frame when the box touches the top edge.
    label_top = max(0, y1 - text_height - 6)
    cv2.rectangle(canvas, (x1, label_top), (x1 + text_width + 4, label_top + text_height + 6), color, -1)
    cv2.putText(
        canvas, label, (x1 + 2, label_top + text_height + 1),
        _FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )


def _draw_pose(canvas: np.ndarray, keypoints: list[list[float]], thickness: int) -> None:
    for start, end in POSE_SKELETON:
        if start >= len(keypoints) or end >= len(keypoints):
            continue
        first, second = keypoints[start], keypoints[end]
        if first[2] > KEYPOINT_THRESHOLD and second[2] > KEYPOINT_THRESHOLD:
            cv2.line(
                canvas,
                (int(first[0]), int(first[1])),
                (int(second[0]), int(second[1])),
                _ACCENT, thickness, cv2.LINE_AA,
            )
    for point in keypoints:
        if point[2] > KEYPOINT_THRESHOLD:
            cv2.circle(canvas, (int(point[0]), int(point[1])), thickness + 1, _KEYPOINT_COLOR, -1)

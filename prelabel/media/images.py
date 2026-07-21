"""Image decoding."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

import cv2
import numpy as np

from ..errors import UnsupportedMedia


def decode_image(raw: bytes, label: str = "image") -> np.ndarray:
    """Decode encoded image bytes into a BGR array.

    Raises :class:`~app.errors.UnsupportedMedia` rather than returning ``None``,
    so a corrupt upload can never travel further as a silent ``None``.
    """
    if not raw:
        return _fail(label, "empty upload")
    array = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return _fail(label, "not a supported image format")
    return image


def try_decode_image(raw: bytes) -> np.ndarray | None:
    """Decode, returning ``None`` on failure.

    Used by the batch endpoint, where one unreadable file is reported in its own
    result slot instead of failing the whole request.
    """
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def decode_data_url(data_url: str, max_bytes: int = 0) -> np.ndarray:
    """Decode a ``data:image/jpeg;base64,...`` string into a BGR image."""
    if not isinstance(data_url, str) or not data_url:
        return _fail("frame", "missing image data")

    payload = data_url.partition(",")[2] if "," in data_url else data_url
    # 4 base64 characters encode 3 bytes; check before allocating.
    if max_bytes and (len(payload) * 3) // 4 > max_bytes:
        return _fail("frame", f"exceeds the {max_bytes // (1024 * 1024)} MB limit")

    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return _fail("frame", "malformed base64 data")
    return decode_image(raw, "frame")


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from disk as BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return _fail("image", f"could not read {Path(path).name}")
    return image


def _fail(label: str, reason: str) -> np.ndarray:
    raise UnsupportedMedia(f"Could not decode {label}: {reason}")

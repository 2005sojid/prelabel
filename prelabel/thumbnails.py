"""Cached gallery thumbnails.

Scrolling a project of ten thousand photos would otherwise decode full-size
JPEGs on every pass. Thumbnails are generated once, written next to the database,
and served from there; the cache is keyed by project and item, so deleting a
project drops its thumbnails with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from . import config

log = logging.getLogger("prelabel.thumbnails")

JPEG_QUALITY = 82


def cache_dir() -> Path:
    return config.STORAGE_DIR / "thumbnails"


def project_dir(project_id: str) -> Path:
    return cache_dir() / project_id


def thumbnail_path(project_id: str, item_id: int) -> Path:
    return project_dir(project_id) / f"{item_id}.jpg"


def build(source: Path, destination: Path, edge: int | None = None) -> Path | None:
    """Write a downscaled copy of ``source``. Returns ``None`` if unreadable.

    Images already smaller than the target are copied at their own size rather
    than upscaled — enlarging a thumbnail wastes bytes and gains no detail.
    """
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        log.debug("Cannot read %s for a thumbnail", source)
        return None

    target = int(edge or config.THUMBNAIL_EDGE)
    height, width = image.shape[:2]
    scale = min(1.0, target / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        log.warning("Could not write thumbnail %s", destination)
        return None
    return destination


def get_or_build(project_id: str, item_id: int, source: Path) -> Path | None:
    """Return a cached thumbnail, generating it on first request.

    A source that has been modified since the thumbnail was written invalidates
    it, so editing an image in place does not leave a stale preview behind.
    """
    destination = thumbnail_path(project_id, item_id)
    if destination.exists():
        try:
            if destination.stat().st_mtime >= source.stat().st_mtime:
                return destination
        except OSError:
            pass  # fall through and rebuild
    return build(source, destination)


def discard_project(project_id: str) -> None:
    """Remove a project's cached thumbnails."""
    import shutil

    shutil.rmtree(project_dir(project_id), ignore_errors=True)

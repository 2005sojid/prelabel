"""Reading image folders from the server's own filesystem.

Pointing the server at a directory is what makes a project durable: the images
stay where they are, nothing is uploaded, and a run can be closed and resumed
because the source is still there.

It is also arbitrary file access, so every path goes through
:func:`resolve_dataset_dir` first. Only directories inside a configured
:data:`~prelabel.config.DATA_ROOTS` entry are reachable, and the check is done on
the *resolved* path so a symlink or ``..`` cannot lead outside.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from . import config
from .errors import DatasetAccessError

log = logging.getLogger("prelabel.datasets")

#: Directory names never worth walking into.
SKIP_DIRECTORIES = frozenset({
    ".git", ".svn", "__pycache__", "node_modules", ".ipynb_checkpoints",
    ".thumbnails", "$RECYCLE.BIN", "System Volume Information",
})


@dataclass(frozen=True)
class ImageRef:
    """One image found in a dataset directory."""

    #: Path relative to the dataset root — the stable identity of the image.
    rel_path: str
    #: Size in bytes, used to spot a file that changed between runs.
    size: int

    @property
    def name(self) -> str:
        return Path(self.rel_path).name


def is_configured() -> bool:
    """True when at least one dataset root has been allowed."""
    return bool(config.DATA_ROOTS)


def _within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_dataset_dir(raw_path: str) -> Path:
    """Validate a client-supplied directory and return its resolved path.

    Raises :class:`~prelabel.errors.DatasetAccessError` when the feature is not
    configured, the path is outside every allowed root, or it is not a readable
    directory. The messages say which of those it was, because "access denied"
    with no reason is the least useful error there is.
    """
    if not is_configured():
        raise DatasetAccessError(
            "Folder projects are disabled. Set PL_DATA_ROOTS to the directories "
            "this server may read images from, e.g. PL_DATA_ROOTS=/data/images"
        )

    if not raw_path or not raw_path.strip():
        raise DatasetAccessError("No folder given")

    try:
        candidate = Path(raw_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        # ValueError covers an embedded NUL byte, which the OS layer rejects
        # before it ever becomes an OSError.
        raise DatasetAccessError(f"Not a usable path: {exc}") from exc

    if not any(_within(candidate, root) for root in config.DATA_ROOTS):
        allowed = ", ".join(str(root) for root in config.DATA_ROOTS)
        raise DatasetAccessError(
            f"'{candidate}' is outside the allowed dataset roots ({allowed}). "
            "Add it to PL_DATA_ROOTS to use it."
        )

    if not candidate.exists():
        raise DatasetAccessError(f"'{candidate}' does not exist")
    if not candidate.is_dir():
        raise DatasetAccessError(f"'{candidate}' is not a directory")

    return candidate


def scan_images(root: Path, limit: int | None = None) -> list[ImageRef]:
    """List the images under ``root``, recursively and in a stable order.

    Sorted so a project's item numbering is reproducible: rescanning the same
    folder yields the same order, which keeps exported dataset ids stable.
    """
    maximum = config.MAX_PROJECT_IMAGES if limit is None else limit
    found: list[ImageRef] = []

    for path in _walk(root):
        if path.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:  # vanished mid-scan, or no permission
            log.debug("Skipping %s: %s", path, exc)
            continue
        if size == 0:
            continue
        found.append(ImageRef(rel_path=path.relative_to(root).as_posix(), size=size))
        if len(found) >= maximum:
            log.warning("Stopped scanning %s at the %d image limit", root, maximum)
            break

    found.sort(key=lambda ref: ref.rel_path)
    return found


def _walk(root: Path) -> Iterator[Path]:
    """Depth-first walk that skips noise directories and unreadable branches."""
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: log.debug("walk: %s", e)):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRECTORIES and not d.startswith("."))
        directory = Path(dirpath)
        for filename in sorted(filenames):
            yield directory / filename


def resolve_item_path(root: Path, rel_path: str) -> Path:
    """Turn a stored relative path back into an absolute one, safely.

    The relative path comes from the database, but the database is reachable
    through the API, so it is treated as untrusted: the result still has to land
    inside the dataset root.
    """
    candidate = (root / rel_path).resolve()
    if not _within(candidate, root.resolve()):
        raise DatasetAccessError(f"'{rel_path}' escapes its dataset directory")
    if not candidate.is_file():
        raise DatasetAccessError(f"'{rel_path}' is no longer on disk")
    return candidate

"""Upload handling: safe names, and size caps enforced *while* reading.

The important property here is that a size limit is checked **as bytes arrive**,
not after the whole upload already sits in memory. Reading first and measuring
afterwards means the limit protects nothing — the memory has already been spent
by the time the check runs.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from pathlib import Path

from fastapi import UploadFile

from .errors import PayloadTooLarge

#: Read granularity for streaming uploads.
CHUNK_BYTES = 1024 * 1024

#: Characters that are unsafe or meaningless in a stored filename. Everything
#: else — including non-Latin letters — is preserved, so a file called
#: ``写真.jpg`` keeps its name instead of collapsing into ``_.jpg``.
_UNSAFE = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def safe_filename(name: str | None, fallback_suffix: str = "") -> str:
    """Reduce an upload's filename to a single safe path component.

    Handles the cases a browser or a scripted client can produce: a missing
    filename, a path (``../../etc/passwd``, ``C:\\Windows\\evil.pt``), reserved
    characters, and names that are entirely stripped away.
    """
    if not name:
        return f"upload-{uuid.uuid4().hex[:8]}{fallback_suffix}"

    # Take the last component under *both* separator conventions: a POSIX server
    # receiving a Windows-style path would otherwise keep the backslashes.
    candidate = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    candidate = _UNSAFE.sub("_", candidate).strip(" .")

    if not candidate:
        return f"upload-{uuid.uuid4().hex[:8]}{fallback_suffix}"
    return candidate[:255]


def _limit_bytes(max_mb: int) -> int:
    """Convert a megabyte cap to bytes; 0 (unlimited) becomes 0."""
    return max(0, int(max_mb)) * 1024 * 1024


def read_capped(upload: UploadFile, max_mb: int, label: str = "File") -> bytes:
    """Read an upload into memory, aborting as soon as it exceeds ``max_mb``.

    Returns the bytes. Raises :class:`~app.errors.PayloadTooLarge` the moment the
    cap is passed, so an oversized upload never gets fully buffered.
    """
    limit = _limit_bytes(max_mb)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if limit and total > limit:
            raise PayloadTooLarge(f"{label} '{safe_filename(upload.filename)}' exceeds the {max_mb} MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def save_streaming(upload: UploadFile, dest_dir: Path, max_mb: int) -> Path:
    """Stream an upload to ``dest_dir``, enforcing ``max_mb`` as it is written.

    Streaming keeps a large model or video from blowing up RAM, and lets us abort
    (and clean up the partial file) as soon as the cap is exceeded rather than
    after the whole upload has landed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_filename(upload.filename)
    limit = _limit_bytes(max_mb)
    written = 0
    try:
        with open(dest, "wb") as buffer:
            while True:
                chunk = upload.file.read(CHUNK_BYTES)
                if not chunk:
                    break
                written += len(chunk)
                if limit and written > limit:
                    raise PayloadTooLarge(f"'{dest.name}' exceeds the {max_mb} MB limit")
                buffer.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)  # never leave a truncated file behind
        raise
    return dest


def enforce_file_count(files: Iterable[UploadFile], maximum: int, label: str) -> list[UploadFile]:
    """Reject a request carrying more files than we are willing to hold at once."""
    items = list(files)
    if not items:
        raise PayloadTooLarge(f"No {label} supplied")
    if maximum and len(items) > maximum:
        raise PayloadTooLarge(f"Too many {label}: {len(items)} (limit {maximum})")
    return items

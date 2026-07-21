"""Projects: durable runs over a folder of images on the server.

A project is the answer to "I have 20 000 photos and this will take an hour".
The folder stays where it is, results go to the database as they are produced,
and the browser is just a view — closing the tab does not lose the run, and
reopening it shows where things got to.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .. import comparison, config, datasets, exporters, importers, thumbnails
from ..errors import Conflict, DatasetAccessError, NotFound, UnsupportedMedia
from ..jobs import ProjectRunner, RunSettings, TrainingRunner
from ..state import ModelRegistry
from ..store import Project, Store
from ..training import TrainingSettings
from ..uploads import read_capped
from .deps import get_registry, get_runner, get_store, get_trainer

log = logging.getLogger("prelabel.api.projects")

router = APIRouter(prefix="/api/projects", tags=["projects"])

#: Items returned by one page of the gallery.
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 1000


def _require(store: Store, project_id: str) -> Project:
    project = store.get_project(project_id)
    if project is None:
        raise NotFound(f"No project '{project_id}'")
    return project


def _describe(store: Store, runner: ProjectRunner, project: Project) -> dict[str, Any]:
    return {
        **project.to_dict(),
        "stats": store.stats(project.id),
        "running": runner.is_running(project.id),
    }


# --- browsing ---------------------------------------------------------------


@router.get("/-/roots")
def dataset_roots() -> dict:
    """Directories this server is allowed to read images from.

    Returned so the UI can offer a picker instead of asking the user to type a
    path and guess why it was refused.
    """
    return {
        "configured": datasets.is_configured(),
        "roots": [str(root) for root in config.DATA_ROOTS],
        "hint": "Set PL_DATA_ROOTS to the folders this server may read.",
    }


@router.get("/-/browse")
def browse(path: str = "") -> dict:
    """List the subdirectories of an allowed path, with an image count."""
    if not path:
        return {
            "path": "",
            "parent": None,
            "directories": [{"name": str(root), "path": str(root)} for root in config.DATA_ROOTS],
            "image_count": 0,
        }

    directory = datasets.resolve_dataset_dir(path)
    subdirectories = sorted(
        (entry for entry in directory.iterdir() if entry.is_dir() and entry.name not in datasets.SKIP_DIRECTORIES),
        key=lambda entry: entry.name.lower(),
    )
    parent = str(directory.parent) if any(
        directory != root for root in config.DATA_ROOTS
    ) and directory.parent != directory else None

    return {
        "path": str(directory),
        "parent": parent,
        "directories": [{"name": entry.name, "path": str(entry)} for entry in subdirectories],
        "image_count": len(datasets.scan_images(directory)),
    }


# --- lifecycle --------------------------------------------------------------


@router.get("")
def list_projects(
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    return {"projects": [_describe(store, runner, project) for project in store.list_projects()]}


@router.post("", status_code=201)
async def create_project(
    request: Request,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    """Create a project from a server-side folder and register its images."""
    body = await request.json()
    source = str(body.get("path", "")).strip()
    directory = datasets.resolve_dataset_dir(source)

    images = datasets.scan_images(directory)
    if not images:
        raise DatasetAccessError(f"No images found in '{directory}'")

    project = store.create_project(
        name=str(body.get("name") or directory.name),
        source_dir=str(directory),
        settings=RunSettings.from_dict(body.get("settings")).to_dict(),
    )
    store.add_items(project.id, [image.rel_path for image in images])
    log.info("Created project %s (%s) with %d images", project.id, project.name, len(images))
    return _describe(store, runner, _require(store, project.id))


@router.get("/{project_id}")
def get_project(
    project_id: str,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    project = _require(store, project_id)
    return {
        **_describe(store, runner, project),
        "classes": store.class_counts(project_id),
        "export_formats": exporters.formats_for(project.model.get("task", "detect")),
    }


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    request: Request,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    _require(store, project_id)
    body = await request.json()
    changes: dict[str, Any] = {}
    if "name" in body:
        changes["name"] = str(body["name"]).strip() or "Untitled"
    if "settings" in body:
        changes["settings"] = RunSettings.from_dict(body["settings"]).to_dict()
    store.update_project(project_id, **changes)
    return _describe(store, runner, _require(store, project_id))


@router.delete("/{project_id}")
def delete_project(
    project_id: str,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    _require(store, project_id)
    if runner.is_running(project_id):
        runner.cancel(project_id)
    store.delete_project(project_id)
    thumbnails.discard_project(project_id)
    return {"status": "ok", "deleted": project_id}


# --- running ----------------------------------------------------------------


@router.post("/{project_id}/run")
async def run_project(
    project_id: str,
    request: Request,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
    trainer: TrainingRunner = Depends(get_trainer),
) -> dict:
    """Start or resume a run. Only the images still pending are processed."""
    project = _require(store, project_id)
    if trainer.active_project is not None:
        raise Conflict("A training run is using the model — wait for it to finish, then run.")
    body = await _optional_json(request)

    settings = RunSettings.from_dict(body.get("settings") or project.settings)
    if body.get("restart"):
        store.reset_items(project_id)

    runner.start(project_id, settings)
    return _describe(store, runner, _require(store, project_id))


@router.post("/{project_id}/cancel")
def cancel_project(
    project_id: str,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    _require(store, project_id)
    if not runner.cancel(project_id):
        raise Conflict("That project is not running")
    return {"status": "ok"}


@router.post("/{project_id}/rescan")
def rescan_project(
    project_id: str,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    """Pick up images added to the folder since the project was created."""
    project = _require(store, project_id)
    directory = datasets.resolve_dataset_dir(project.source_dir)
    added = store.add_items(project_id, [image.rel_path for image in datasets.scan_images(directory)])
    return {**_describe(store, runner, _require(store, project_id)), "added": added}


# --- items ------------------------------------------------------------------


@router.get("/{project_id}/items")
def list_items(
    project_id: str,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    order: str = "path",
    only: str | None = None,
    search: str | None = None,
    store: Store = Depends(get_store),
) -> dict:
    """One page of results.

    ``order=priority`` is the active-learning view: the images where the model is
    least sure come first, so review effort lands where it changes the most.
    """
    _require(store, project_id)
    limit = max(1, min(int(limit), MAX_PAGE_SIZE))
    items = store.list_items(
        project_id, offset=max(0, int(offset)), limit=limit,
        order=order, only=only, search=search,
    )
    return {
        "items": [item.to_dict() for item in items],
        "offset": offset,
        "limit": limit,
        "stats": store.stats(project_id),
    }


@router.get("/{project_id}/items/{item_id}/image")
def item_image(
    project_id: str,
    item_id: int,
    thumb: bool = False,
    store: Store = Depends(get_store),
) -> FileResponse:
    """Serve one of a project's images, full size or as a cached thumbnail."""
    project = _require(store, project_id)
    item = store.get_item(project_id, item_id)
    if item is None:
        raise NotFound(f"No item {item_id} in project '{project_id}'")

    root = datasets.resolve_dataset_dir(project.source_dir)
    source = datasets.resolve_item_path(root, item.rel_path)

    if thumb:
        cached = thumbnails.get_or_build(project_id, item_id, source)
        if cached is not None:
            return FileResponse(cached, media_type="image/jpeg")
    return FileResponse(source)


# --- comparison -------------------------------------------------------------


@router.post("/{project_id}/baseline")
def capture_baseline(
    project_id: str,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
) -> dict:
    """Freeze the current annotations as the set to compare the next run against.

    The usual sequence: run model A, capture, load model B, re-run. What you get
    afterwards is a diff rather than a replacement.
    """
    _require(store, project_id)
    if runner.is_running(project_id):
        raise Conflict("Wait for the run to finish before capturing a baseline")

    captured = store.capture_baseline(project_id)
    if not captured:
        raise Conflict("Nothing to capture — run the project first")

    result = comparison.refresh(store, project_id)
    log.info("Captured %d items as the baseline for %s", captured, project_id)
    return {"status": "ok", "captured": captured, "comparison": result.to_dict()}


@router.delete("/{project_id}/baseline")
def clear_baseline(project_id: str, store: Store = Depends(get_store)) -> dict:
    """Drop the comparison set. The current annotations are untouched."""
    _require(store, project_id)
    store.clear_baseline(project_id)
    comparison.invalidate(store, project_id)
    return {"status": "ok"}


@router.get("/{project_id}/comparison")
def project_comparison(project_id: str, store: Store = Depends(get_store)) -> dict:
    """How the two annotation sets differ, across the whole project."""
    _require(store, project_id)
    return comparison.describe(store, project_id)


@router.post("/{project_id}/comparison")
def recompute_comparison(
    project_id: str,
    iou: float = 0.5,
    store: Store = Depends(get_store),
) -> dict:
    """Recompare both sets, optionally at a different overlap threshold."""
    _require(store, project_id)
    if store.baseline_size(project_id) == 0:
        raise Conflict("There is no baseline to compare against")
    return comparison.refresh(store, project_id, iou_threshold=max(0.05, min(float(iou), 0.95))).to_dict()


@router.get("/{project_id}/items/{item_id}/comparison")
def item_comparison(
    project_id: str,
    item_id: int,
    store: Store = Depends(get_store),
) -> dict:
    """The per-object diff for one image: what agreed, moved, vanished, appeared."""
    _require(store, project_id)
    item = store.get_item(project_id, item_id)
    if item is None:
        raise NotFound(f"No item {item_id} in project '{project_id}'")

    diff = comparison.compare_item(item)
    return {
        "item_id": item_id,
        "rel_path": item.rel_path,
        "has_baseline": item.has_baseline,
        "baseline": item.baseline,
        "current": item.detections,
        **diff.to_dict(),
    }


# --- retraining -------------------------------------------------------------


@router.post("/{project_id}/train")
async def train_project(
    project_id: str,
    request: Request,
    store: Store = Depends(get_store),
    runner: ProjectRunner = Depends(get_runner),
    trainer: TrainingRunner = Depends(get_trainer),
) -> dict:
    """Fine-tune the loaded model on this project's labels.

    Closes the loop: correct the model's worst guesses, import them back, and
    teach the model from them. Trains on the ``current`` set by default — which,
    after a corrected COCO is imported over the model's output, is the
    corrections — or on ``baseline`` if the corrections were imported there.

    Refused while a run is going: training and inference share the one GPU.
    """
    _require(store, project_id)
    if runner.active_project is not None:
        raise Conflict("A project run is using the model — wait for it to finish before training.")

    body = await _optional_json(request)
    settings = TrainingSettings.from_dict(body.get("settings") or body)
    state = trainer.start(project_id, settings)
    return {"status": "ok", "training": state}


@router.post("/{project_id}/train/cancel")
def cancel_training(
    project_id: str,
    store: Store = Depends(get_store),
    trainer: TrainingRunner = Depends(get_trainer),
) -> dict:
    """Ask a running fine-tune to stop at the next epoch boundary."""
    _require(store, project_id)
    if not trainer.cancel(project_id):
        raise Conflict("No training is running for this project")
    return {"status": "ok"}


@router.get("/{project_id}/training")
def training_status(
    project_id: str,
    store: Store = Depends(get_store),
    trainer: TrainingRunner = Depends(get_trainer),
) -> dict:
    """The state of the most recent (or running) fine-tune."""
    project = _require(store, project_id)
    return {**project.training, "active": trainer.is_training(project_id)}


@router.post("/{project_id}/train/adopt")
def adopt_retrained(
    project_id: str,
    store: Store = Depends(get_store),
    registry: ModelRegistry = Depends(get_registry),
    trainer: TrainingRunner = Depends(get_trainer),
) -> dict:
    """Make the retrained weights the active model, ready to re-run and diff.

    Copies ``best.pt`` into a fresh model slot and loads it the same way an upload
    would — self-tested before it replaces the incumbent, so a corrupt checkpoint
    costs nothing.
    """
    project = _require(store, project_id)
    if trainer.is_training(project_id):
        raise Conflict("Training is still running")

    training_state = project.training
    if training_state.get("status") != "done":
        raise Conflict("There is no finished training to adopt")
    weights = training_state.get("weights")
    if not weights or not Path(weights).exists():
        raise Conflict("The retrained weights are no longer on disk — train again")

    slot = registry.new_slot()
    try:
        destination = slot / "retrained.pt"
        shutil.copy2(weights, destination)
        info = registry.load(destination, slot)
    except BaseException:
        registry.discard_slot(slot)
        raise

    log.info("Adopted retrained model for %s: %s", project_id, info.get("name"))
    return {"status": "ok", "model": info}


# --- transfer ---------------------------------------------------------------


@router.get("/{project_id}/export")
def export_project(
    project_id: str,
    format: str = "coco",  # noqa: A002 - the query parameter is named `format`
    conf: float = 0.0,
    store: Store = Depends(get_store),
) -> FileResponse:
    """Download the project as a dataset archive.

    Built on the server, so the size is bounded by disk rather than by browser
    memory, and ZIP64 handles archives past 4 GB without special handling.
    """
    project = _require(store, project_id)
    destination = config.OUTPUTS_DIR / f"export_{project_id}_{uuid.uuid4().hex[:8]}.zip"

    try:
        summary = exporters.export_project(store, project, destination, fmt=format, threshold=conf)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project.name) or "dataset"
    return FileResponse(
        destination,
        media_type="application/zip",
        filename=f"{safe_name}_{format}.zip",
        background=BackgroundTask(lambda: destination.unlink(missing_ok=True)),
        headers={
            "X-Prelabel-Images": str(summary.images),
            "X-Prelabel-Annotations": str(summary.annotations),
        },
    )


@router.post("/{project_id}/import")
def import_annotations(
    project_id: str,
    file: UploadFile = File(...),
    replace: bool = True,
    into: str = "current",
    store: Store = Depends(get_store),
) -> dict:
    """Load COCO annotations into the project.

    ``into=baseline`` is the interesting one: it puts corrected labels beside the
    model's own output instead of over it, so the two can be compared. That is
    how you find where the model is wrong — and, just as often, where the
    labelling is.
    """
    _require(store, project_id)
    if into not in ("current", "baseline"):
        raise UnsupportedMedia("'into' must be 'current' or 'baseline'")

    raw = read_capped(file, config.MAX_MEDIA_MB, "Annotation file")
    summary = importers.import_coco(store, project_id, raw, replace=replace, into=into)

    payload = {"status": "ok", "into": into, **summary.to_dict(), "stats": store.stats(project_id)}
    if into == "baseline":
        payload["comparison"] = comparison.refresh(store, project_id).to_dict()
    else:
        comparison.invalidate(store, project_id)
    return payload


async def _optional_json(request: Request) -> dict:
    """Body as a dict, tolerating an empty POST."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - no body at all is a valid way to say "defaults"
        return {}
    return body if isinstance(body, dict) else {}

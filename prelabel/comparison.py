"""Running a comparison over a whole project.

Sits between :mod:`prelabel.diffing`, which knows how to compare two sets of
boxes and nothing else, and the API. Its job is to walk a project, compare each
item, cache the per-item verdict so the review queue can sort on it in SQL, and
add everything up.

The cached verdict is the reason this is not computed on demand: sorting a
hundred thousand images by "how much do these two disagree" has to happen in the
database, and the database cannot run an IoU match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .diffing import DEFAULT_IOU, DiffSummary, ItemDiff, compare
from .store import Item, Store

log = logging.getLogger("prelabel.comparison")


@dataclass(frozen=True)
class ComparisonResult:
    """What a project-wide comparison produced."""

    summary: DiffSummary
    compared: int

    def to_dict(self) -> dict[str, Any]:
        return {"compared": self.compared, **self.summary.to_dict()}


def compare_item(item: Item, iou_threshold: float = DEFAULT_IOU) -> ItemDiff:
    """Compare one item's stored baseline against its current annotations."""
    return compare(item.baseline, item.detections, iou_threshold=iou_threshold)


def refresh(store: Store, project_id: str, iou_threshold: float = DEFAULT_IOU) -> ComparisonResult:
    """Recompare every item in a project, caching the result at both levels.

    Per item, so the review queue can sort by disagreement in SQL. Per project,
    so reading the summary later does not mean matching every box again.

    Called whenever either side changes — a run finishing, a baseline being
    captured or imported — which are exactly the moments the cache goes stale.
    """
    summary = DiffSummary()
    compared = 0

    for item in store.iter_comparable_items(project_id):
        diff = compare_item(item, iou_threshold)
        summary.add(diff)
        store.save_comparison(project_id, item.id, diff.disputed, diff.agreement)
        compared += 1

    result = ComparisonResult(summary, compared)
    store.update_project(project_id, comparison=result.to_dict())

    log.info(
        "Compared %s: %d items, %d disagreements, agreement %.1f%%",
        project_id, compared, summary.disputed, summary.agreement * 100,
    )
    return result


def describe(store: Store, project_id: str) -> dict[str, Any]:
    """The stored comparison, or an empty one when nothing has been compared.

    A plain read: :func:`refresh` is what makes it accurate.
    """
    project = store.get_project(project_id)
    if project is None or not project.comparison:
        return {
            "compared": 0,
            "available": False,
            "baseline_items": store.baseline_size(project_id) if project else 0,
        }
    return {**project.comparison, "available": True,
            "baseline_items": store.baseline_size(project_id)}


def invalidate(store: Store, project_id: str) -> None:
    """Forget the stored comparison — the sets no longer line up with it."""
    store.update_project(project_id, comparison={})

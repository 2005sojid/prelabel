"""Comparing two sets of annotations over the same images.

Written as one operation rather than a "model comparison" feature, because the
interesting question is always the same shape — *these labels versus those* — and
the answer is useful in at least three situations:

* **model A versus model B** — which one to ship;
* **model versus corrected ground truth** — where the model is wrong, and, just
  as often, where the *human* missed something;
* **before versus after retraining** — what improved and what regressed.

The vocabulary is deliberately neutral: a **baseline** and a **current** set.
Neither is assumed to be correct. Which one is "truth" depends on why you are
looking, and the tool should not decide that for you.

Everything here is pure: dictionaries in, dictionaries out, no database and no
model. That keeps the part which is easy to get subtly wrong — matching boxes —
testable on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Overlap at which two boxes are considered the same object. 0.5 is the COCO
#: convention and what people expect when they read "matched".
DEFAULT_IOU = 0.5

#: How a detection in one set relates to the other.
AGREED = "agreed"            # same object, same class
RECLASSIFIED = "reclassified"  # same object, different class
MISSING = "missing"          # in the baseline, absent from the current set
ADDED = "added"              # in the current set, absent from the baseline

KINDS = (AGREED, RECLASSIFIED, MISSING, ADDED)

#: Kinds that represent a disagreement worth a human's attention.
DISPUTED = (RECLASSIFIED, MISSING, ADDED)


def _box(detection: dict) -> list[float] | None:
    box = detection.get("box")
    return box if box and len(box) == 4 else None


def iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    left, top = max(ax1, bx1), max(ay1, by1)
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class Pairing:
    """One row of the comparison: what happened to a single object."""

    kind: str
    baseline: dict | None = None
    current: dict | None = None
    overlap: float = 0.0

    @property
    def class_name(self) -> str:
        source = self.current or self.baseline or {}
        return str(source.get("class_name", "?"))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "overlap": round(self.overlap, 3)}
        if self.baseline is not None:
            data["baseline"] = self.baseline
        if self.current is not None:
            data["current"] = self.current
        if self.kind == RECLASSIFIED:
            data["from_class"] = str((self.baseline or {}).get("class_name", "?"))
            data["to_class"] = str((self.current or {}).get("class_name", "?"))
        return data


@dataclass
class ItemDiff:
    """The comparison for one image."""

    pairings: list[Pairing] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        tally = dict.fromkeys(KINDS, 0)
        for pairing in self.pairings:
            tally[pairing.kind] += 1
        return tally

    @property
    def disputed(self) -> int:
        return sum(1 for p in self.pairings if p.kind in DISPUTED)

    @property
    def agreement(self) -> float:
        """1.0 when the two sets say exactly the same thing, 0.0 when nothing lines up.

        This is the Jaccard index over matched objects: agreed / everything seen.
        Two empty sets agree perfectly — an image both sides call empty is not a
        disagreement, and scoring it 0 would push blank frames to the top of a
        review queue sorted by conflict.
        """
        if not self.pairings:
            return 1.0
        return self.counts[AGREED] / len(self.pairings)

    def to_dict(self, with_pairings: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "counts": self.counts,
            "disputed": self.disputed,
            "agreement": round(self.agreement, 4),
        }
        if with_pairings:
            data["pairings"] = [p.to_dict() for p in self.pairings]
        return data


def compare(
    baseline: Iterable[dict],
    current: Iterable[dict],
    iou_threshold: float = DEFAULT_IOU,
) -> ItemDiff:
    """Match two sets of detections for one image and classify every difference.

    Matching runs in two passes, and the order matters. The first pass pairs
    detections **of the same class**; only then does a second pass look for
    overlapping boxes whose classes differ. Done in one pass, a high-overlap
    cross-class pair could steal the match from a slightly-lower-overlap correct
    one, turning an agreement into two errors.
    """
    baseline_list = list(baseline)
    current_list = list(current)

    # Classification results have no geometry; compare them as a set of labels.
    if _is_classification(baseline_list) or _is_classification(current_list):
        return _compare_labels(baseline_list, current_list)

    used_baseline: set[int] = set()
    used_current: set[int] = set()
    pairings: list[Pairing] = []

    for same_class_only in (True, False):
        candidates = []
        for i, left in enumerate(baseline_list):
            if i in used_baseline or _box(left) is None:
                continue
            for j, right in enumerate(current_list):
                if j in used_current or _box(right) is None:
                    continue
                same = left.get("class_name") == right.get("class_name")
                if same_class_only and not same:
                    continue
                if not same_class_only and same:
                    continue  # already handled in the first pass
                overlap = iou(_box(left), _box(right))
                if overlap >= iou_threshold:
                    candidates.append((overlap, i, j))

        for overlap, i, j in sorted(candidates, key=lambda c: -c[0]):
            if i in used_baseline or j in used_current:
                continue
            used_baseline.add(i)
            used_current.add(j)
            pairings.append(
                Pairing(
                    kind=AGREED if same_class_only else RECLASSIFIED,
                    baseline=baseline_list[i],
                    current=current_list[j],
                    overlap=overlap,
                )
            )

    pairings.extend(
        Pairing(kind=MISSING, baseline=detection)
        for i, detection in enumerate(baseline_list)
        if i not in used_baseline
    )
    pairings.extend(
        Pairing(kind=ADDED, current=detection)
        for j, detection in enumerate(current_list)
        if j not in used_current
    )
    return ItemDiff(pairings)


def _is_classification(detections: Sequence[dict]) -> bool:
    return bool(detections) and all(_box(d) is None for d in detections)


def _compare_labels(baseline: Sequence[dict], current: Sequence[dict]) -> ItemDiff:
    """Compare label sets when there is no geometry to match on.

    Only the top prediction is compared: a classifier's runners-up are context,
    not claims, and diffing all five would report disagreement on every image.
    """
    left = baseline[0] if baseline else None
    right = current[0] if current else None

    if left is None and right is None:
        return ItemDiff([])
    if left is None:
        return ItemDiff([Pairing(kind=ADDED, current=right)])
    if right is None:
        return ItemDiff([Pairing(kind=MISSING, baseline=left)])

    same = left.get("class_name") == right.get("class_name")
    return ItemDiff([
        Pairing(kind=AGREED if same else RECLASSIFIED, baseline=left, current=right, overlap=1.0)
    ])


@dataclass
class DiffSummary:
    """Totals across a whole project."""

    items: int = 0
    items_identical: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    #: Per-class disagreement, so a systematic problem stands out from noise.
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The most common label swaps, e.g. ("truck" -> "bus"): 12.
    reclassifications: dict[str, int] = field(default_factory=dict)

    @property
    def disputed(self) -> int:
        return sum(self.counts[kind] for kind in DISPUTED)

    @property
    def agreement(self) -> float:
        total = sum(self.counts.values())
        return self.counts[AGREED] / total if total else 1.0

    def add(self, diff: ItemDiff) -> None:
        self.items += 1
        if diff.disputed == 0:
            self.items_identical += 1
        for pairing in diff.pairings:
            self.counts[pairing.kind] += 1
            per_class = self.by_class.setdefault(pairing.class_name, dict.fromkeys(KINDS, 0))
            per_class[pairing.kind] += 1
            if pairing.kind == RECLASSIFIED:
                swap = (
                    f"{(pairing.baseline or {}).get('class_name', '?')}"
                    f" -> {(pairing.current or {}).get('class_name', '?')}"
                )
                self.reclassifications[swap] = self.reclassifications.get(swap, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        ranked = sorted(
            (
                {"class_name": name, **kinds,
                 "disputed": sum(kinds[k] for k in DISPUTED)}
                for name, kinds in self.by_class.items()
            ),
            key=lambda entry: -entry["disputed"],
        )
        return {
            "items": self.items,
            "items_identical": self.items_identical,
            "items_disputed": self.items - self.items_identical,
            "counts": self.counts,
            "disputed": self.disputed,
            "agreement": round(self.agreement, 4),
            "by_class": ranked,
            "reclassifications": [
                {"swap": swap, "count": count}
                for swap, count in sorted(self.reclassifications.items(), key=lambda kv: -kv[1])
            ],
        }

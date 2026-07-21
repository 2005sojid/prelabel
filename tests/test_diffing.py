"""Comparing two annotation sets.

Box matching is the part that goes subtly wrong, so most of these are about
*which* detection got paired with which — not just how many came out.
"""

from __future__ import annotations

import pytest

from prelabel.diffing import (
    ADDED,
    AGREED,
    MISSING,
    RECLASSIFIED,
    DiffSummary,
    compare,
    iou,
)


def box(x1, y1, x2, y2, name="car", confidence=0.9):
    return {"class_name": name, "confidence": confidence, "kind": "box", "box": [x1, y1, x2, y2]}


def label(name, confidence=0.9):
    return {"class_name": name, "confidence": confidence, "kind": "classification"}


def kinds(diff):
    return [p.kind for p in diff.pairings]


# --- geometry ---------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_of_touching_boxes_is_zero():
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_of_half_overlap():
    # 10x10 and 10x10 sharing a 5x10 strip: 50 / (100 + 100 - 50)
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


# --- the four outcomes ------------------------------------------------------


def test_identical_sets_agree_completely():
    detections = [box(0, 0, 10, 10), box(50, 50, 70, 70, "bus")]
    diff = compare(detections, detections)

    assert kinds(diff) == [AGREED, AGREED]
    assert diff.agreement == 1.0
    assert diff.disputed == 0


def test_a_detection_only_in_the_baseline_is_missing():
    diff = compare([box(0, 0, 10, 10)], [])
    assert kinds(diff) == [MISSING]
    assert diff.agreement == 0.0


def test_a_detection_only_in_the_current_set_is_added():
    diff = compare([], [box(0, 0, 10, 10)])
    assert kinds(diff) == [ADDED]


def test_the_same_object_under_a_different_name_is_reclassified():
    diff = compare([box(0, 0, 10, 10, "truck")], [box(0, 0, 10, 10, "bus")])

    assert kinds(diff) == [RECLASSIFIED]
    entry = diff.pairings[0].to_dict()
    assert entry["from_class"] == "truck"
    assert entry["to_class"] == "bus"


def test_boxes_that_barely_overlap_are_two_separate_objects():
    diff = compare([box(0, 0, 10, 10)], [box(8, 8, 18, 18)])
    assert sorted(kinds(diff)) == [ADDED, MISSING]


def test_the_overlap_threshold_is_honoured():
    baseline = [box(0, 0, 10, 10)]
    current = [box(5, 0, 15, 10)]  # IoU = 1/3

    assert kinds(compare(baseline, current, iou_threshold=0.5)) != [AGREED]
    assert kinds(compare(baseline, current, iou_threshold=0.3)) == [AGREED]


# --- matching order ---------------------------------------------------------


def test_a_correct_match_is_not_stolen_by_a_better_overlapping_wrong_class():
    """The reason matching runs in two passes.

    A cross-class box overlapping *more* must not consume the same-class match;
    done in one pass that turns one agreement into two errors.
    """
    baseline = [box(0, 0, 10, 10, "car")]
    current = [
        box(0, 0, 10, 10, "truck"),   # perfect overlap, wrong class
        box(1, 1, 11, 11, "car"),     # slightly worse overlap, right class
    ]

    diff = compare(baseline, current)
    assert sorted(kinds(diff)) == [ADDED, AGREED]

    agreed = next(p for p in diff.pairings if p.kind == AGREED)
    assert agreed.current["class_name"] == "car"


def test_each_detection_is_used_at_most_once():
    baseline = [box(0, 0, 10, 10)]
    current = [box(0, 0, 10, 10), box(1, 1, 11, 11)]

    diff = compare(baseline, current)
    assert kinds(diff).count(AGREED) == 1
    assert kinds(diff).count(ADDED) == 1


def test_the_best_overlap_wins_among_equals():
    baseline = [box(0, 0, 10, 10)]
    current = [box(4, 4, 14, 14), box(0, 0, 10, 10)]

    diff = compare(baseline, current)
    agreed = next(p for p in diff.pairings if p.kind == AGREED)
    assert agreed.overlap == 1.0


def test_many_objects_pair_up_independently():
    baseline = [box(i * 100, 0, i * 100 + 50, 50) for i in range(5)]
    current = [box(i * 100, 0, i * 100 + 50, 50) for i in range(5)]
    assert kinds(compare(baseline, current)) == [AGREED] * 5


# --- classification ---------------------------------------------------------


def test_classification_compares_only_the_top_label():
    """A classifier's runners-up are context, not claims."""
    baseline = [label("cat"), label("lynx", 0.2), label("dog", 0.1)]
    current = [label("cat"), label("tiger", 0.3)]
    assert kinds(compare(baseline, current)) == [AGREED]


def test_a_different_top_label_is_a_reclassification():
    assert kinds(compare([label("cat")], [label("dog")])) == [RECLASSIFIED]


def test_classification_against_nothing():
    assert kinds(compare([label("cat")], [])) == [MISSING]
    assert kinds(compare([], [label("cat")])) == [ADDED]


# --- agreement --------------------------------------------------------------


def test_two_empty_sets_agree():
    """An image both sides call empty is not a conflict to review."""
    diff = compare([], [])
    assert diff.agreement == 1.0
    assert diff.disputed == 0


def test_agreement_is_the_matched_share():
    baseline = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    current = [box(0, 0, 10, 10)]
    # one agreed, one missing
    assert compare(baseline, current).agreement == pytest.approx(0.5)


# --- project summary --------------------------------------------------------


def test_summary_totals_across_items():
    summary = DiffSummary()
    summary.add(compare([box(0, 0, 10, 10)], [box(0, 0, 10, 10)]))            # agreed
    summary.add(compare([box(0, 0, 10, 10, "truck")], [box(0, 0, 10, 10, "bus")]))
    summary.add(compare([box(0, 0, 10, 10)], []))                             # missing

    data = summary.to_dict()
    assert data["items"] == 3
    assert data["items_identical"] == 1
    assert data["items_disputed"] == 2
    assert data["counts"] == {AGREED: 1, RECLASSIFIED: 1, MISSING: 1, ADDED: 0}
    # to_dict() rounds for the wire; compare against the rounded value.
    assert data["agreement"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary.agreement == pytest.approx(1 / 3)


def test_summary_ranks_classes_by_how_much_they_are_disputed():
    summary = DiffSummary()
    for _ in range(3):
        summary.add(compare([box(0, 0, 10, 10, "person")], []))
    summary.add(compare([box(0, 0, 10, 10, "car")], []))

    ranked = summary.to_dict()["by_class"]
    assert ranked[0]["class_name"] == "person"
    assert ranked[0]["disputed"] == 3


def test_summary_reports_the_common_label_swaps():
    """A systematic confusion is the useful signal, not one-off noise."""
    summary = DiffSummary()
    for _ in range(4):
        summary.add(compare([box(0, 0, 10, 10, "truck")], [box(0, 0, 10, 10, "bus")]))
    summary.add(compare([box(0, 0, 10, 10, "cat")], [box(0, 0, 10, 10, "dog")]))

    swaps = summary.to_dict()["reclassifications"]
    assert swaps[0] == {"swap": "truck -> bus", "count": 4}
    assert swaps[1]["count"] == 1


def test_an_empty_summary_agrees():
    assert DiffSummary().to_dict()["agreement"] == 1.0

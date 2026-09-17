"""Offline calibration over labeled runs.

This is the loop that makes confidence thresholds real: log decisions +
outcomes, then measure whether "70% confident" really means 70% correct.
"""
from __future__ import annotations

from collections import defaultdict

from .telemetry import TurnRecord


def expected_calibration_error(
    records: list[TurnRecord],
    outcome_positive: str = "success",
    n_bins: int = 10,
) -> float:
    """Coarse ECE using per-turn outcome as the label and the minimum decision
    confidence as the model's reported confidence.

    For full calibration, label per-question correctness instead and bucket each
    decision's confidence separately.
    """
    confidences: list[float] = []
    labels: list[float] = []
    for record in records:
        if record.outcome is None or not record.decisions:
            continue
        conf = min(d["confidence"] for d in record.decisions.values())
        confidences.append(conf)
        labels.append(1.0 if record.outcome == outcome_positive else 0.0)

    if not confidences:
        return 0.0

    bins: dict[int, tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))  # idx -> (sum conf, sum label)
    for conf, label in zip(confidences, labels):
        idx = min(int(conf * n_bins), n_bins - 1)
        s_conf, s_label = bins[idx]
        bins[idx] = (s_conf + conf, s_label + label)

    ece = 0.0
    for idx in range(n_bins):
        s_conf, s_label = bins.get(idx, (0.0, 0.0))
        count = len([c for c in confidences if min(int(c * n_bins), n_bins - 1) == idx])
        if count == 0:
            continue
        avg_conf = s_conf / count
        avg_label = s_label / count
        ece += (count / len(confidences)) * abs(avg_conf - avg_label)
    return ece


def reliability_table(records: list[TurnRecord], n_bins: int = 10) -> list[dict]:
    """Per-bin (confidence, accuracy, count) for eyeballing calibration."""
    rows: list[dict] = []
    confidences: list[float] = []
    labels: list[float] = []
    for record in records:
        if record.outcome is None or not record.decisions:
            continue
        confidences.append(min(d["confidence"] for d in record.decisions.values()))
        labels.append(1.0 if record.outcome == "success" else 0.0)

    for idx in range(n_bins):
        lo, hi = idx / n_bins, (idx + 1) / n_bins
        in_bin = [(c, l) for c, l in zip(confidences, labels) if lo <= c < hi or (idx == n_bins - 1 and c == hi)]
        if not in_bin:
            continue
        avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(l for _, l in in_bin) / len(in_bin)
        rows.append({"conf_range": f"{lo:.1f}-{hi:.1f}", "avg_conf": round(avg_conf, 3),
                     "accuracy": round(acc, 3), "count": len(in_bin)})
    return rows

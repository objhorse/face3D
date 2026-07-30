from __future__ import annotations

import numpy as np

from src.geometry.eye_state import (
    EyeViewEvidence,
    aggregate_eye_state,
    classify_eye_ratio,
    eye_view_evidence,
)


def test_classifies_open_closed_and_uncertain_ratio() -> None:
    assert classify_eye_ratio(0.07).state == "closed"
    assert classify_eye_ratio(0.23).state == "open"
    assert classify_eye_ratio(0.15).state == "uncertain"


def test_three_closed_views_produce_closed_consensus() -> None:
    views = {
        name: EyeViewEvidence.from_ratios(0.08, 0.09)
        for name in ("left", "front", "right")
    }
    result = aggregate_eye_state(views)

    assert result.usable
    assert result.states == {"subject_right": "closed", "subject_left": "closed"}


def test_front_and_one_side_override_single_conflicting_side() -> None:
    views = {
        "front": EyeViewEvidence.from_ratios(0.07, 0.08),
        "left": EyeViewEvidence.from_ratios(0.09, 0.09),
        "right": EyeViewEvidence.from_ratios(0.24, 0.23),
    }
    result = aggregate_eye_state(views)

    assert result.usable
    assert result.states["subject_right"] == "closed"
    assert result.states["subject_left"] == "closed"
    assert "right" in result.conflicting_views


def test_unresolved_open_closed_split_is_uncertain() -> None:
    views = {
        "front": EyeViewEvidence.from_ratios(0.15, 0.15),
        "left": EyeViewEvidence.from_ratios(0.07, 0.08),
        "right": EyeViewEvidence.from_ratios(0.23, 0.24),
    }
    result = aggregate_eye_state(views)

    assert not result.usable
    assert result.states == {"subject_right": "uncertain", "subject_left": "uncertain"}


def test_dense_landmarks_report_each_eye_separately() -> None:
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = [0, 0]
    points[133] = [100, 0]
    points[160] = [35, -4]
    points[144] = [35, 4]
    points[158] = [65, -4]
    points[153] = [65, 4]
    points[362] = [200, 0]
    points[263] = [300, 0]
    points[385] = [235, -24]
    points[380] = [235, 24]
    points[387] = [265, -24]
    points[373] = [265, 24]

    evidence = eye_view_evidence(points)

    assert evidence.eyes["subject_right"].state == "closed"
    assert evidence.eyes["subject_left"].state == "open"

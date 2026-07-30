"""Per-eye, multi-view open/closed state evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EYE_NAMES = ("subject_right", "subject_left")


@dataclass(frozen=True)
class EyeRatioState:
    ratio: float
    state: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "ratio": float(self.ratio),
            "state": self.state,
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class EyeViewEvidence:
    eyes: Mapping[str, EyeRatioState]

    @classmethod
    def from_ratios(cls, subject_right: float, subject_left: float) -> "EyeViewEvidence":
        return cls(
            eyes={
                "subject_right": classify_eye_ratio(subject_right),
                "subject_left": classify_eye_ratio(subject_left),
            }
        )

    def to_dict(self) -> dict:
        return {name: state.to_dict() for name, state in self.eyes.items()}


@dataclass(frozen=True)
class EyeStateConsensus:
    states: Mapping[str, str]
    confidences: Mapping[str, float]
    usable: bool
    conflicting_views: tuple[str, ...]
    views: Mapping[str, EyeViewEvidence]

    def to_dict(self) -> dict:
        return {
            "states": dict(self.states),
            "confidences": {key: float(value) for key, value in self.confidences.items()},
            "usable": bool(self.usable),
            "conflicting_views": list(self.conflicting_views),
            "views": {name: evidence.to_dict() for name, evidence in self.views.items()},
        }


def classify_eye_ratio(
    ratio: float,
    *,
    closed_threshold: float = 0.13,
    open_threshold: float = 0.18,
) -> EyeRatioState:
    value = float(ratio)
    if not np.isfinite(value) or value < 0.0:
        return EyeRatioState(value, "uncertain", 0.0)
    if value <= float(closed_threshold):
        span = max(float(closed_threshold), 1e-6)
        confidence = np.clip((closed_threshold - value) / (0.55 * span), 0.0, 1.0)
        return EyeRatioState(value, "closed", float(confidence))
    if value >= float(open_threshold):
        span = max(0.55 * float(open_threshold), 1e-6)
        confidence = np.clip((value - open_threshold) / span + 0.55, 0.0, 1.0)
        return EyeRatioState(value, "open", float(confidence))
    return EyeRatioState(value, "uncertain", 0.0)


def eye_view_evidence(landmarks: np.ndarray) -> EyeViewEvidence:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        raise ValueError("MediaPipe landmarks must have shape (468+, 2+)")
    definitions = {
        "subject_right": (33, 133, 160, 144, 158, 153),
        "subject_left": (362, 263, 385, 380, 387, 373),
    }
    ratios = {}
    for name, (outer, inner, upper_a, lower_a, upper_b, lower_b) in definitions.items():
        width = float(np.linalg.norm(points[outer, :2] - points[inner, :2]))
        height = float(
            np.linalg.norm(points[upper_a, :2] - points[lower_a, :2])
            + np.linalg.norm(points[upper_b, :2] - points[lower_b, :2])
        )
        ratios[name] = height / max(2.0 * width, 1e-6)
    return EyeViewEvidence(
        eyes={name: classify_eye_ratio(ratio) for name, ratio in ratios.items()}
    )


def aggregate_eye_state(
    views: Mapping[str, EyeViewEvidence],
) -> EyeStateConsensus:
    if not views:
        raise ValueError("at least one view is required")
    states: dict[str, str] = {}
    confidences: dict[str, float] = {}
    for eye_name in EYE_NAMES:
        votes = {"open": [], "closed": []}
        for view_name, evidence in views.items():
            state = evidence.eyes[eye_name]
            if state.state in votes:
                votes[state.state].append((view_name, float(state.confidence)))
        open_count = len(votes["open"])
        closed_count = len(votes["closed"])
        if open_count >= 2 and open_count > closed_count:
            selected = "open"
        elif closed_count >= 2 and closed_count > open_count:
            selected = "closed"
        else:
            selected = "uncertain"
        states[eye_name] = selected
        confidences[eye_name] = (
            float(np.mean([value for _name, value in votes[selected]]))
            if selected in votes and votes[selected]
            else 0.0
        )

    conflicts = set()
    for view_name, evidence in views.items():
        for eye_name in EYE_NAMES:
            selected = states[eye_name]
            observed = evidence.eyes[eye_name].state
            if selected in {"open", "closed"} and observed in {"open", "closed"}:
                if observed != selected:
                    conflicts.add(view_name)
    usable = all(states[name] in {"open", "closed"} for name in EYE_NAMES)
    return EyeStateConsensus(
        states=states,
        confidences=confidences,
        usable=bool(usable),
        conflicting_views=tuple(sorted(conflicts)),
        views=dict(views),
    )

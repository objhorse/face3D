"""Expression-state evidence shared by fitting and reconstruction reports."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


LandmarkPair = Tuple[int, int]

EYE_GAP_PAIRS: Tuple[LandmarkPair, ...] = (
    (37, 41),
    (38, 40),
    (43, 47),
    (44, 46),
)
INNER_MOUTH_GAP_PAIRS: Tuple[LandmarkPair, ...] = (
    (61, 67),
    (62, 66),
    (63, 65),
)


def paired_vertical_gap_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    pairs: Sequence[LandmarkPair],
    *,
    beta: float = 0.002,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """Match observed vertical feature gaps without assuming they are closed."""
    losses = []
    for upper_idx, lower_idx in pairs:
        projected_gap = torch.abs(projected[upper_idx, 1] - projected[lower_idx, 1])
        target_gap = (
            torch.abs(target[upper_idx, 1] - target[lower_idx, 1])
            * float(target_scale)
        )
        losses.append(
            F.smooth_l1_loss(
                projected_gap,
                target_gap,
                reduction="mean",
                beta=float(beta),
            )
        )
    if not losses:
        return projected.new_zeros(())
    return torch.stack(losses).mean()


def _mean_gap(points: np.ndarray, pairs: Iterable[LandmarkPair]) -> float:
    values = [abs(float(points[a, 1]) - float(points[b, 1])) for a, b in pairs]
    return float(np.mean(values)) if values else 0.0


def feature_gap_diagnostics(
    projected: np.ndarray,
    target: np.ndarray,
    *,
    force_closed_eyes: bool = False,
    force_closed_mouth: bool = False,
    closed_eye_target_scale: float = 0.0,
    closed_mouth_target_scale: float = 0.0,
) -> Dict[str, float]:
    projected = np.asarray(projected, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if projected.shape != target.shape or projected.ndim != 2 or projected.shape[0] < 68:
        raise ValueError("projected and target landmarks must have matching (68, 2) shapes")

    projected_eye = _mean_gap(projected, EYE_GAP_PAIRS)
    raw_target_eye = _mean_gap(target, EYE_GAP_PAIRS)
    target_eye = (
        raw_target_eye * float(closed_eye_target_scale)
        if force_closed_eyes else raw_target_eye
    )
    projected_mouth = _mean_gap(projected, INNER_MOUTH_GAP_PAIRS)
    raw_target_mouth = _mean_gap(target, INNER_MOUTH_GAP_PAIRS)
    target_mouth = (
        raw_target_mouth * float(closed_mouth_target_scale)
        if force_closed_mouth else raw_target_mouth
    )
    return {
        "projected_eye_gap_px": projected_eye,
        "target_eye_gap_px": target_eye,
        "eye_gap_error_px": abs(projected_eye - target_eye),
        "projected_mouth_gap_px": projected_mouth,
        "target_mouth_gap_px": target_mouth,
        "mouth_gap_error_px": abs(projected_mouth - target_mouth),
    }


def mediapipe_expression_state(
    landmarks: np.ndarray,
    *,
    closed_eye_threshold: float = 0.13,
    closed_mouth_threshold: float = 0.06,
) -> Dict[str, float | bool]:
    """Estimate eye and mouth closure from MediaPipe's dense landmarks."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        raise ValueError("MediaPipe landmarks must have shape (468+, 2+)")

    eye_definitions = (
        (33, 133, 160, 144, 158, 153),
        (362, 263, 385, 380, 387, 373),
    )
    eye_ratios = []
    for outer, inner, upper_a, lower_a, upper_b, lower_b in eye_definitions:
        width = float(np.linalg.norm(points[outer, :2] - points[inner, :2]))
        height = float(
            np.linalg.norm(points[upper_a, :2] - points[lower_a, :2])
            + np.linalg.norm(points[upper_b, :2] - points[lower_b, :2])
        )
        eye_ratios.append(height / max(2.0 * width, 1e-6))
    eye_ratio = float(np.mean(eye_ratios))

    mouth_width = float(np.linalg.norm(points[78, :2] - points[308, :2]))
    mouth_ratio = float(
        np.linalg.norm(points[13, :2] - points[14, :2])
        / max(mouth_width, 1e-6)
    )
    return {
        "eye_aspect_ratio": eye_ratio,
        "mouth_aspect_ratio": mouth_ratio,
        "closed_eyes": eye_ratio <= float(closed_eye_threshold),
        "closed_mouth": mouth_ratio <= float(closed_mouth_threshold),
    }

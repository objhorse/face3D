"""Low-frequency photometric normalization for calibrated face images."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def normalize_skin_chroma(
    image: np.ndarray,
    skin_mask: np.ndarray,
    exclude_mask: np.ndarray,
    field_size: int = 256,
    sigma: float = 18.0,
    strength: float = 0.72,
    max_lab_shift: float = 8.0,
) -> Tuple[np.ndarray, dict, np.ndarray]:
    """Remove spatial color casts while preserving luminance and fine detail."""
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    small = cv2.resize(rgb, (field_size, field_size), interpolation=cv2.INTER_AREA)
    skin = cv2.resize(
        (np.asarray(skin_mask) > 0).astype(np.uint8),
        (field_size, field_size),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    excluded = cv2.resize(
        (np.asarray(exclude_mask) > 0).astype(np.uint8),
        (field_size, field_size),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    valid = skin & ~excluded
    if int(valid.sum()) < 500:
        return (
            rgb.copy(),
            {"applied": False, "skin_samples": int(valid.sum())},
            np.zeros((field_size, field_size, 2), dtype=np.float32),
        )

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = np.median(lab[valid, 1:3], axis=0)
    weight = valid.astype(np.float32)
    smooth_weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    local = []
    for channel in (1, 2):
        numerator = cv2.GaussianBlur(
            lab[:, :, channel] * weight,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
        )
        local.append(numerator / np.maximum(smooth_weight, 1e-4))
    local_chroma = np.stack(local, axis=2)
    correction = np.clip(
        target[None, None, :] - local_chroma,
        -float(max_lab_shift),
        float(max_lab_shift),
    )
    support = np.clip(smooth_weight / 0.25, 0.0, 1.0)
    correction *= support[:, :, None] * float(strength)

    full_correction = cv2.resize(correction, (width, height), interpolation=cv2.INTER_CUBIC)
    full_skin = cv2.resize(
        skin.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    full_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    full_lab[:, :, 1:3] += full_correction * full_skin[:, :, None]
    corrected = cv2.cvtColor(np.clip(full_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return corrected, {
        "applied": True,
        "skin_samples": int(valid.sum()),
        "target_lab_ab": target.astype(float).tolist(),
        "max_applied_lab_shift": float(np.max(np.abs(correction))),
        "mean_applied_lab_shift": float(np.mean(np.linalg.norm(correction[valid], axis=1))),
    }, correction.astype(np.float32)

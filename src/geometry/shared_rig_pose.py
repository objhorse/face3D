from __future__ import annotations

from typing import Mapping

import numpy as np

from src.cross_view_geometry import Camera, relative_camera_transform


def model_to_reference_camera(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform FLAME/model-space points into the reference camera frame."""
    points_np = np.asarray(points, dtype=np.float64)
    rotation_np = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation_np = np.asarray(translation, dtype=np.float64).reshape(3)
    return points_np @ rotation_np.T + translation_np


def reference_camera_to_model(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Transform reference-camera points back into FLAME/model space."""
    points_np = np.asarray(points, dtype=np.float64)
    rotation_np = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation_np = np.asarray(translation, dtype=np.float64).reshape(3)
    return (points_np - translation_np) @ rotation_np


def derive_fixed_rig_view_poses(
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
    cameras_by_view: Mapping[str, Camera],
    reference_view: str = "front",
) -> dict[str, dict[str, np.ndarray]]:
    """Anchor a fitted mesh once, then derive every view from the fixed rig."""
    if reference_view not in cameras_by_view:
        raise KeyError(f"missing reference camera view: {reference_view}")
    reference_camera = cameras_by_view[reference_view]
    model_rotation = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
    model_translation = np.asarray(reference_translation, dtype=np.float64).reshape(3)
    poses: dict[str, dict[str, np.ndarray]] = {}
    for view, camera in cameras_by_view.items():
        relative_rotation, relative_translation = relative_camera_transform(
            reference_camera, camera
        )
        poses[view] = {
            "R": (relative_rotation @ model_rotation).astype(np.float64),
            "t": (
                relative_rotation @ model_translation + relative_translation
            ).astype(np.float64),
        }
    return poses


def effective_camera_extrinsics(
    rotation: np.ndarray,
    translation: np.ndarray,
    object_rotation: np.ndarray,
    object_translation: np.ndarray,
    object_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Express one shared object pose as an equivalent camera extrinsic.

    The object transform is ``V_world = scale * R_object * V + t_object``.
    Rewriting the projective equation leaves the local exported mesh unchanged
    and preserves all rig-relative camera geometry.
    """
    scale = float(object_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("object_scale must be finite and positive")
    camera_rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    camera_translation = np.asarray(translation, dtype=np.float64).reshape(3)
    object_rotation = np.asarray(object_rotation, dtype=np.float64).reshape(3, 3)
    object_translation = np.asarray(object_translation, dtype=np.float64).reshape(3)
    return (
        camera_rotation @ object_rotation,
        (camera_rotation @ object_translation + camera_translation) / scale,
    )


def apply_shared_pose_to_views(
    per_view_results: Mapping[str, Mapping],
    object_rotation: np.ndarray,
    object_translation: np.ndarray,
    object_scale: float,
) -> dict[str, dict]:
    """Return view results with a shared object correction folded into R/t."""
    corrected: dict[str, dict] = {}
    for name, result in per_view_results.items():
        if "R" not in result or "t" not in result:
            corrected[name] = dict(result)
            continue
        rotation, translation = effective_camera_extrinsics(
            result["R"],
            result["t"],
            object_rotation,
            object_translation,
            object_scale,
        )
        copied = dict(result)
        copied["R"] = rotation.astype(np.float32)
        copied["t"] = translation.astype(np.float32)
        corrected[name] = copied
    return corrected

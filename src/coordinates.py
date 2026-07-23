"""Coordinate-system adapters used by the reconstruction pipeline.

Internal mesh geometry stays in FLAME/world coordinates.  Conversions to
OpenCV-style projection space and OBJ UV space should live here instead of
being handwritten at call sites.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .appearance.projective_sampling import project_points_strict


def flame_points_to_opencv_world(points: np.ndarray) -> np.ndarray:
    """Convert FLAME/world points (Y up) to the OpenCV-world convention (Y down)."""
    converted = np.asarray(points).copy()
    converted[..., 1] *= -1
    return converted


def texture_points_for_projection(points: np.ndarray) -> np.ndarray:
    """Return mesh points in the projection convention used by saved cameras."""
    return np.asarray(points).copy()


def camera_space_from_texture_extrinsics(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Transform FLAME mesh points to camera space for the current texture cameras."""
    projected_points = texture_points_for_projection(points)
    return (R @ projected_points.T + t[:, None]).T


def project_texture_points_to_image(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project FLAME mesh points into image pixels.

    Returns:
        v_cam: (N, 3) camera-space points
        z: (N,) camera depth
        pixels: (N, 2) image coordinates
        front: (N,) points in front of the camera
    """
    projected = project_points_strict(points, K, R, t)
    # Preserve the legacy tuple and writable-array contract for pipeline callers.
    return (
        projected.camera_points.copy(),
        projected.depth.copy(),
        projected.pixel_xy.copy(),
        projected.front_facing.copy(),
    )


def camera_center_for_texture_visibility(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the camera center used by FLAME-space normal visibility tests."""
    return (-R.T @ t).astype(np.float32)


def obj_uv_to_image_uv(uv: np.ndarray) -> np.ndarray:
    """Convert OBJ UVs (V up) to internal texture-image UVs (V down)."""
    converted = np.asarray(uv, dtype=np.float32).copy()
    converted[..., 1] = 1.0 - converted[..., 1]
    return converted


def image_uv_to_obj_uv(uv: np.ndarray) -> np.ndarray:
    """Convert internal texture-image UVs (V down) to OBJ UVs (V up)."""
    return obj_uv_to_image_uv(uv)

"""Subject-specific registration inputs for stable three-view texture baking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.appearance.photometric_normalization import normalize_skin_chroma
from src.appearance.texture_registration import build_sampling_warp, draw_registration_overlay


def _project_vertices(vertices: np.ndarray, camera: dict) -> np.ndarray:
    camera_points = (
        np.asarray(camera["R"], dtype=np.float64) @ np.asarray(vertices, dtype=np.float64).T
        + np.asarray(camera["t"], dtype=np.float64).reshape(3, 1)
    ).T
    homogeneous = (np.asarray(camera["K"], dtype=np.float64) @ camera_points.T).T
    return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-6)


def _landmark_projection(
    vertices: np.ndarray,
    camera: dict,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    projected = _project_vertices(vertices, camera)
    return (projected[np.asarray(landmark_triangles)] * barycentric[:, :, None]).sum(axis=1)


def _nose_boundary_controls(
    projected_landmarks: np.ndarray,
    nose_mask: np.ndarray,
    view: str,
    max_distance_px: float = 28.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Match projected nose samples to the observed semantic nose boundary."""
    binary = (np.asarray(nose_mask) > 0).astype(np.uint8)
    ys = np.where(binary > 0)[0]
    if len(ys) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    lower_fraction = 0.42 if view == "front" else 0.12
    cutoff = float(ys.min() + lower_fraction * (ys.max() - ys.min()))
    boundary[: max(0, int(cutoff)), :] = False
    by, bx = np.where(boundary)
    observed = np.stack((bx, by), axis=1).astype(np.float32)
    if len(observed) < 6:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    nose = np.asarray(projected_landmarks[27:36], dtype=np.float32)
    dense = [nose]
    for start, end in ((27, 30), (31, 32), (32, 33), (33, 34), (34, 35)):
        alpha = np.linspace(0.0, 1.0, 5, dtype=np.float32)[:, None]
        dense.append(
            projected_landmarks[start][None, :] * (1.0 - alpha)
            + projected_landmarks[end][None, :] * alpha
        )
    model = np.vstack(dense).astype(np.float32)
    model = model[model[:, 1] >= cutoff - 12.0]
    if len(model) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    distance, nearest = cKDTree(observed).query(model, k=1)
    keep = np.asarray(distance) <= float(max_distance_px)
    model = model[keep]
    matched = observed[np.asarray(nearest)[keep]]
    if len(model) > 24:
        take = np.linspace(0, len(model) - 1, 24).astype(np.int64)
        model, matched = model[take], matched[take]
    return model.astype(np.float32), matched.astype(np.float32)


def prepare_stable_texture_registration(
    *,
    mesh_dir: Path,
    preprocessed_views: Dict[str, dict],
    hires_images: Dict[str, np.ndarray],
    cfg: Any,
    debug_dir: Path,
) -> Dict[str, Any]:
    """Build per-subject warps, feature masks, and normalized source images."""
    from src.module2_geometry import FLAMEModel, _mediapipe_to_68, load_flame_landmark_mapping
    from src.module3_texture import load_cameras, load_mesh_obj

    debug_dir.mkdir(parents=True, exist_ok=True)
    registration_dir = debug_dir / "registration"
    photometric_dir = debug_dir / "photometric"
    registration_dir.mkdir(parents=True, exist_ok=True)
    photometric_dir.mkdir(parents=True, exist_ok=True)

    vertices, _faces, _uv, _uv_faces = load_mesh_obj(mesh_dir / "face_mesh.obj")
    cameras = load_cameras(mesh_dir / "cameras.json")
    flame = FLAMEModel(cfg.FLAME_MODEL_PATH, n_shape=1, n_exp=1)
    mapping = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
    if mapping is None:
        raise RuntimeError("FLAME landmark mapping is required for texture registration")
    flame_faces = flame.faces.detach().cpu().numpy()
    landmark_triangles = flame_faces[np.asarray(mapping["face_idx"], dtype=np.int64)]
    barycentric = np.asarray(mapping["bary_coords"], dtype=np.float32)

    feature_labels = {2, 3, 4, 5, 6, 10, 11, 12, 13}
    feature_masks = {
        view: np.isin(data["parser_labels"], list(feature_labels)).astype(np.uint8) * 255
        for view, data in preprocessed_views.items()
    }
    normalized_hires: Dict[str, np.ndarray] = {}
    photometric_report: Dict[str, dict] = {}
    for view, image in hires_images.items():
        face_mask = cv2.resize(
            preprocessed_views[view]["face_mask"],
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        excluded = cv2.resize(
            feature_masks[view],
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        corrected, stats, field = normalize_skin_chroma(image, face_mask, excluded)
        normalized_hires[view] = corrected
        photometric_report[view] = stats
        cv2.imwrite(
            str(photometric_dir / f"{view}_normalized.jpg"),
            cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        heat = cv2.applyColorMap(
            np.clip((np.linalg.norm(field, axis=2) / 8.0) * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        cv2.imwrite(str(photometric_dir / f"{view}_chroma_field.png"), heat)

    warps: Dict[str, object] = {}
    view_report: Dict[str, dict] = {}
    stable_indices = np.r_[36:48, 48:68]
    for view in ("left", "front", "right"):
        if view not in cameras or view not in preprocessed_views:
            continue
        data = preprocessed_views[view]
        observed_68 = _mediapipe_to_68(data["landmarks"])
        projected_68 = _landmark_projection(
            vertices, cameras[view], landmark_triangles, barycentric
        )
        model_controls = projected_68[stable_indices]
        observed_controls = observed_68[stable_indices]
        nasal_model, nasal_observed = _nose_boundary_controls(
            projected_68, data.get("nose_mask"), view
        )
        if len(nasal_model):
            model_controls = np.vstack((model_controls, nasal_model))
            observed_controls = np.vstack((observed_controls, nasal_observed))
        warp = build_sampling_warp(
            model_controls,
            observed_controls,
            data["image"].shape[:2],
            smoothing=24.0,
            max_control_displacement_px=28.0,
            max_field_displacement_px=24.0,
        )
        warps[view] = warp
        overlay = draw_registration_overlay(
            data["image"], model_controls, observed_controls, warp
        )
        cv2.imwrite(
            str(registration_dir / f"{view}_registration_overlay.jpg"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        magnitude = np.linalg.norm(warp.displacement_grid, axis=2)
        heat = cv2.applyColorMap(
            np.clip(magnitude / 24.0 * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        cv2.imwrite(str(registration_dir / f"{view}_displacement.png"), heat)
        view_report[view] = {
            "controls": int(len(model_controls)),
            "nasal_semantic_controls": int(len(nasal_model)),
            "residual_before_px": warp.control_residual_before_px,
            "residual_after_px": warp.control_residual_after_px,
            "max_displacement_px": warp.max_displacement_px,
        }

    report = {
        "views": view_report,
        "photometric_normalization": photometric_report,
        "nose_control_source": "semantic_nose_mask_boundary",
        "geometry_changed": False,
    }
    (debug_dir / "registration_inputs.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "sampling_warps": warps,
        "feature_masks": feature_masks,
        "hires_images": normalized_hires,
        "report": report,
    }

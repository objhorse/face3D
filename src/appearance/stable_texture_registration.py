"""Subject-specific registration inputs for stable three-view texture baking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from src.appearance.photometric_normalization import normalize_skin_chroma
from src.appearance.semantic_feature_registration import build_nose_controls
from src.appearance.texture_registration import (
    build_layered_feature_warp,
    draw_registration_overlay,
)


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
        nasal = build_nose_controls(
            projected_68,
            observed_68,
            data.get("nose_mask"),
            view=view,
            image=data["image"] if view == "front" else None,
        )
        warp = build_layered_feature_warp(
            model_controls,
            observed_controls,
            nasal.model_points,
            nasal.observed_points,
            data["image"].shape[:2],
            smoothing=8.0,
            max_translation_px=8.0,
            max_rotation_degrees=1.0,
            max_scale_delta=0.015,
            local_max_displacement_px=16.0,
            min_jacobian=0.35,
        )
        warps[view] = warp
        overlay_model = np.vstack((model_controls, nasal.model_points))
        overlay_observed = np.vstack((observed_controls, nasal.observed_points))
        overlay = draw_registration_overlay(
            data["image"], overlay_model, overlay_observed, warp
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
            "global_controls": int(len(model_controls)),
            "nasal_semantic_controls": int(len(nasal.model_points)),
            "nasal_confidence": float(nasal.confidence),
            "nasal_diagnostics": nasal.diagnostics,
            "nasal_correspondences": [
                {
                    "group": group,
                    "model": [float(value) for value in model],
                    "observed": [float(value) for value in observed],
                }
                for group, model, observed in zip(
                    nasal.groups, nasal.model_points, nasal.observed_points
                )
            ],
            "residual_before_px": warp.control_residual_before_px,
            "residual_after_px": warp.control_residual_after_px,
            "max_displacement_px": warp.max_displacement_px,
            "displacement_p95_px": warp.displacement_p95_px,
            "min_jacobian": warp.min_jacobian,
            "warp_diagnostics": warp.diagnostics,
        }

    report = {
        "views": view_report,
        "photometric_normalization": photometric_report,
        "nose_control_source": "ordered_semantic_lower_boundary",
        "registration_mode": "bounded_similarity_plus_local_nose",
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

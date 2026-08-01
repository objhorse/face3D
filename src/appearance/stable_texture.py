"""No-delete texture baking for the stable three-view pipeline."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.geometry.mesh_quality import (
    MeshQualityThresholds,
    assert_quality_gate,
    load_mesh_quality,
    make_quality_gate,
)
from src.geometry.template_fit import temporary_config_overrides

ProgressFn = Callable[[str, int, str], None]
STRICT_FEATURE_LABELS = frozenset({2, 3, 4, 5, 6, 10})
MP468_TO_68 = np.asarray(
    [
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288,
        323, 454, 389, 71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
        168, 197, 5, 4, 75, 97, 2, 326, 305,
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
        78, 82, 13, 312, 308, 317, 14, 87,
    ],
    dtype=np.int32,
)


def _feature_mask_from_landmarks(
    landmarks: np.ndarray,
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= int(MP468_TO_68.max()) or points.shape[1] < 2:
        return None

    import cv2

    points_68 = points[MP468_TO_68, :2]
    if not np.isfinite(points_68).all():
        return None
    height, width = image_shape
    feature_mask = np.zeros((height, width), dtype=np.uint8)
    for start, stop in ((17, 22), (22, 27), (27, 36), (36, 42), (42, 48)):
        region = np.rint(points_68[start:stop]).astype(np.int32)
        hull = cv2.convexHull(region)
        if len(hull) >= 3:
            cv2.fillConvexPoly(feature_mask, hull, 255)

    feature_width = float(np.ptp(points_68[17:60, 0]))
    margin = max(3, int(round(feature_width * 0.018)))
    kernel_size = margin * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(feature_mask, kernel)


def _build_strict_feature_masks(
    preprocessed_views: Optional[Dict[str, dict]],
) -> Dict[str, np.ndarray]:
    """Build unwarped semantic masks for central facial features."""
    if not preprocessed_views:
        return {}

    masks: Dict[str, np.ndarray] = {}
    labels_to_keep = tuple(sorted(STRICT_FEATURE_LABELS))
    for view, data in preprocessed_views.items():
        labels = data.get("parser_labels")
        if labels is not None:
            masks[view] = (
                np.isin(np.asarray(labels), labels_to_keep).astype(np.uint8) * 255
            )
            continue
        landmarks = data.get("landmarks")
        image = data.get("image")
        if landmarks is None or image is None:
            continue
        fallback = _feature_mask_from_landmarks(
            landmarks,
            np.asarray(image).shape[:2],
        )
        if fallback is not None:
            masks[view] = fallback
    return masks


def _image_summary(path: Path, valid_mask_path: Optional[Path] = None) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"exists": True, "readable": False}
        if img.ndim == 2:
            channels = 1
        else:
            channels = int(img.shape[2])
        dark = img[..., :3].mean(axis=2) < 3 if img.ndim == 3 else img < 3
        summary = {
            "exists": True,
            "readable": True,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "channels": channels,
            "near_black_ratio": float(np.mean(dark)),
        }
        metric_mask = None
        if valid_mask_path is not None and valid_mask_path.exists():
            valid_mask = cv2.imread(str(valid_mask_path), cv2.IMREAD_GRAYSCALE)
            if valid_mask is not None and valid_mask.shape == img.shape[:2]:
                metric_mask = valid_mask > 0
                summary["valid_texture_ratio"] = float(np.mean(metric_mask))
        if metric_mask is None and img.ndim == 3 and channels == 4:
            metric_mask = img[:, :, 3] > 0
        if metric_mask is None:
            metric_mask = np.ones(img.shape[:2], dtype=bool)
        if channels == 3:
            summary["opaque_ratio"] = 1.0
        elif channels == 4:
            summary["opaque_ratio"] = float(np.mean(img[:, :, 3] > 0))
        else:
            summary["opaque_ratio"] = 0.0
        summary["near_black_opaque_ratio"] = (
            float(np.mean(dark[metric_mask])) if np.any(metric_mask) else 1.0
        )
        return summary
    except Exception as exc:
        return {"exists": True, "readable": False, "error": str(exc)}


def _write_texture_confidence_proxy(
    texture_path: Path,
    output_texture_dir: Path,
    observation_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a conservative visual proxy for texture confidence.

    The current baker does not export per-pixel source weights, so v1 marks
    near-black atlas/background pixels as low confidence and all filled texture
    pixels as available texture. This is intentionally conservative and is used
    for report visibility only.
    """
    out_path = output_texture_dir / "texture_confidence_white.png"
    try:
        import cv2

        observation = None
        if observation_path is not None and observation_path.exists():
            observation = cv2.imread(str(observation_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"exists": False, "path": str(out_path), "reason": "texture_not_readable"}
        if observation is not None and observation.shape == img.shape[:2]:
            high = observation > 0
            method = "bake_observation_mask"
        elif img.ndim == 3 and img.shape[2] == 4:
            high = img[:, :, 3] > 0
            method = "visible_alpha_fallback"
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            high = rgb.mean(axis=2) > 5.0
            method = "near_black_proxy"
        low = ~high
        confidence = np.zeros((*high.shape, 3), dtype=np.uint8)
        confidence[high] = np.array([48, 184, 112], dtype=np.uint8)
        confidence[low] = np.array([245, 190, 72], dtype=np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(confidence, cv2.COLOR_RGB2BGR))
        return {
            "exists": True,
            "path": str(out_path),
            "high_confidence_ratio": float(np.mean(high)),
            "low_confidence_ratio": float(np.mean(low)),
            "method": method,
        }
    except Exception as exc:
        return {"exists": False, "path": str(out_path), "reason": str(exc)}


def run_stable_texture_pipeline(
    *,
    mesh_dir: Path,
    images: Dict[str, np.ndarray],
    output_texture_dir: Path,
    output_mesh_dir: Path,
    cfg: Any,
    preprocessed_views: Optional[Dict[str, dict]] = None,
    face_masks: Optional[Dict[str, np.ndarray]] = None,
    hires_images: Optional[Dict[str, np.ndarray]] = None,
    working_image_size: int = 1024,
    baseline_quality: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressFn] = None,
    sampling_mode: Optional[str] = None,
    enable_local_eye_registration: bool = False,
    enable_ordered_nasal_registration: bool = False,
    model_reference_images: Optional[Dict[str, np.ndarray]] = None,
    external_nasal_controls_by_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_texture_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("texture", 72, "稳定无删面贴图烘焙中...")

    from src.module3_texture import run_texture_pipeline

    registration = None
    texture_hires = hires_images
    sampling_warps = None
    feature_masks = None
    bake_diagnostics: Dict[str, Any] = {}
    if sampling_mode is None:
        sampling_mode = str(
            getattr(cfg, "STABLE_TEXTURE_MODE", "legacy_registered")
        )
    if sampling_mode not in {"legacy_registered", "strict_projective"}:
        raise ValueError(
            "sampling_mode must be 'legacy_registered' or 'strict_projective'"
        )
    if sampling_mode == "strict_projective":
        feature_masks = _build_strict_feature_masks(preprocessed_views)
    if (
        sampling_mode == "legacy_registered"
        and preprocessed_views is not None
        and hires_images is not None
    ):
        from src.appearance.stable_texture_registration import prepare_stable_texture_registration

        registration = prepare_stable_texture_registration(
            mesh_dir=mesh_dir,
            preprocessed_views=preprocessed_views,
            hires_images=hires_images,
            cfg=cfg,
            debug_dir=output_texture_dir.parent / "debug" / "stable_texture_registration",
            enable_local_eye_registration=bool(enable_local_eye_registration),
            enable_ordered_nasal_registration=bool(
                enable_ordered_nasal_registration
            ),
            model_reference_images=model_reference_images,
            external_nasal_controls_by_view=external_nasal_controls_by_view,
        )
        texture_hires = registration["hires_images"]
        sampling_warps = registration["sampling_warps"]
        feature_masks = registration["feature_masks"]

    with temporary_config_overrides(
        cfg,
        ENABLE_VISIBLE_FACE_CROP=bool(getattr(cfg, "STABLE_DELETE_INVISIBLE_FACES", False)),
        ENABLE_SIDE_EAR_TEXTURE_REPAIR=False,
    ):
        glb_path = run_texture_pipeline(
            mesh_dir=mesh_dir,
            images=images,
            output_texture_dir=output_texture_dir,
            output_mesh_dir=output_mesh_dir,
            tex_size=2048,
            lighting_type="white",
            lighting_display_name="白光",
            face_masks=face_masks,
            hires_images=texture_hires,
            working_image_size=working_image_size,
            sampling_warps=sampling_warps,
            feature_masks=feature_masks,
            diagnostics=bake_diagnostics,
            transparent_unobserved=False,
            transparent_bottom_quantile=0.05,
            smooth_geometry_on_export=False,
            sampling_mode=sampling_mode,
        )

    stable_glb = output_mesh_dir / "face_stable.glb"
    if glb_path.exists():
        shutil.copy2(glb_path, stable_glb)

    final_quality = load_mesh_quality(glb_path, label="stable_textured_final")
    if baseline_quality is None:
        baseline_quality = load_mesh_quality(mesh_dir / "face_stable_neutral.glb", label="stable_fit_neutral")
    gate = make_quality_gate(
        baseline=baseline_quality,
        candidate=final_quality,
        thresholds=MeshQualityThresholds(
            min_face_ratio=0.98,
            max_new_degenerate_faces=0,
            max_new_nonmanifold_edges=0,
            max_new_boundary_edges=0,
        ),
        region_name="textured_final",
    )
    assert_quality_gate(gate, context="stable texture no-delete mesh")

    texture_path = output_texture_dir / "albedo_white.png"
    texture_summary = _image_summary(
        texture_path,
        valid_mask_path=output_texture_dir / "texture_valid_white.png",
    )
    confidence_map = _write_texture_confidence_proxy(
        texture_path,
        output_texture_dir,
        observation_path=output_texture_dir / "texture_observation_white.png",
    )
    appearance_issues = []
    if not texture_summary.get("readable", False):
        appearance_issues.append("texture_not_readable")
    if int(texture_summary.get("channels", 0)) not in (3, 4):
        appearance_issues.append("unexpected_texture_channels")
    if not bool(bake_diagnostics.get("inpainted_valid_opaque", False)):
        appearance_issues.append("inpainted_valid_uv_not_opaque")
    if float(texture_summary.get("near_black_opaque_ratio", 1.0)) > 0.08:
        appearance_issues.append("opaque_texture_contains_excess_near_black")
    if bake_diagnostics.get("face_material_alpha_mode") != "OPAQUE":
        appearance_issues.append("face_material_not_opaque")
    appearance_gate = {
        "passed": not appearance_issues,
        "issues": appearance_issues,
        "thresholds": {"max_near_black_opaque_ratio": 0.08},
    }
    summary = {
        "mode": "no_delete",
        "sampling_mode": sampling_mode,
        "local_eye_registration": bool(enable_local_eye_registration),
        "delete_invisible_faces": False,
        "glb_path": str(glb_path),
        "stable_glb_path": str(stable_glb),
        "texture_path": str(texture_path),
        "texture": texture_summary,
        "quality": {
            "final": final_quality,
            "gate": gate,
            "appearance_gate": appearance_gate,
        },
        "confidence": {
            "geometry_low_confidence_policy": "mark_or_inpaint_texture_only",
            "unobserved_regions": "reported as low confidence; mesh faces are preserved",
            "medical_measurement_claim": False,
            "confidence_map": confidence_map,
        },
        "registration": registration["report"] if registration is not None else None,
        "bake_diagnostics": bake_diagnostics,
    }
    with open(output_mesh_dir / "stable_texture_meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if not appearance_gate["passed"]:
        raise RuntimeError(
            "Stable texture appearance gate failed: "
            + ", ".join(appearance_gate["issues"])
        )
    return summary

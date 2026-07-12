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


def _image_summary(path: Path) -> Dict[str, Any]:
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
        dark_ratio = float(np.mean(img[..., :3].mean(axis=2) < 3)) if img.ndim == 3 else float(np.mean(img < 3))
        return {
            "exists": True,
            "readable": True,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "channels": channels,
            "near_black_ratio": dark_ratio,
        }
    except Exception as exc:
        return {"exists": True, "readable": False, "error": str(exc)}


def _write_texture_confidence_proxy(texture_path: Path, output_texture_dir: Path) -> Dict[str, Any]:
    """Write a conservative visual proxy for texture confidence.

    The current baker does not export per-pixel source weights, so v1 marks
    near-black atlas/background pixels as low confidence and all filled texture
    pixels as available texture. This is intentionally conservative and is used
    for report visibility only.
    """
    out_path = output_texture_dir / "texture_confidence_white.png"
    try:
        import cv2

        img = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"exists": False, "path": str(out_path), "reason": "texture_not_readable"}
        if img.ndim == 3 and img.shape[2] == 4:
            high = img[:, :, 3] > 0
            method = "observed_alpha"
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
    if preprocessed_views is not None and hires_images is not None:
        from src.appearance.stable_texture_registration import prepare_stable_texture_registration

        registration = prepare_stable_texture_registration(
            mesh_dir=mesh_dir,
            preprocessed_views=preprocessed_views,
            hires_images=hires_images,
            cfg=cfg,
            debug_dir=output_texture_dir.parent / "debug" / "stable_texture_registration",
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
            transparent_unobserved=True,
            transparent_bottom_quantile=0.05,
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
    confidence_map = _write_texture_confidence_proxy(texture_path, output_texture_dir)
    summary = {
        "mode": "no_delete",
        "delete_invisible_faces": False,
        "glb_path": str(glb_path),
        "stable_glb_path": str(stable_glb),
        "texture_path": str(texture_path),
        "texture": _image_summary(texture_path),
        "quality": {
            "final": final_quality,
            "gate": gate,
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
    return summary

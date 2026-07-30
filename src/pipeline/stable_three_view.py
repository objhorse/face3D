"""Stable three-view reconstruction pipeline.

This pipeline keeps the current API/front-end contract while replacing the
unsafe geometry path with a fixed-topology template baseline.
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from src.appearance.stable_texture import run_stable_texture_pipeline
from src.geometry.template_fit import run_stable_template_fit
from src.reports.stable_reconstruction_report import write_stable_reconstruction_report

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, str], None]


def _progress(progress: Optional[ProgressFn], stage: str, pct: int, message: str) -> None:
    if progress:
        progress(stage, pct, message)


def _load_rgb_images(image_paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    for view, path in image_paths.items():
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {path}")
        images[view] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info("stable pipeline loaded [%s] %sx%s", view, img_bgr.shape[1], img_bgr.shape[0])
    return images


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def run_stable_three_view_pipeline(
    *,
    session_id: Optional[int],
    patient_id: str,
    image_paths: Dict[str, Path],
    session_output_dir: Path,
    manual_intrinsics: Optional[dict] = None,
    progress: Optional[ProgressFn] = None,
) -> Path:
    """Run the stable three-view template reconstruction and return face.glb."""
    from src import config as cfg
    from src.module0_intrinsics import get_intrinsics
    from src.module1_preprocess import preprocess_all_views

    seed = int(getattr(cfg, "STABLE_RANDOM_SEED", 20260718))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        logger.warning("Unable to apply deterministic Torch settings", exc_info=True)

    session_output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = session_output_dir / "meshes"
    texture_dir = session_output_dir / "textures"
    debug_dir = session_output_dir / "debug"
    report_dir = debug_dir / "stable_pipeline"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress, "loading", 5, "加载三视角图像...")
    images = _load_rgb_images(image_paths)

    calibration_intrinsics = None
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        _progress(progress, "intrinsics", 8, "应用相机标定去畸变...")
        from src.module0_intrinsics import undistort_images_with_calibration

        images, calibration_intrinsics = undistort_images_with_calibration(
            images,
            calibration_path=cfg.CAMERA_CALIBRATION_PATH,
            alpha=cfg.UNDISTORT_ALPHA,
        )

    original_hires = {k: v.copy() for k, v in images.items()}

    _progress(progress, "intrinsics", 12, "计算相机内参...")
    intrinsics = get_intrinsics(
        images=images,
        manual_intrinsics=manual_intrinsics or cfg.MANUAL_INTRINSICS,
        calibration_path=cfg.CAMERA_CALIBRATION_PATH,
        calibration_intrinsics=calibration_intrinsics,
        work_image_size=cfg.WORK_IMAGE_SIZE,
        dust3r_dir=cfg.DUST3R_DIR,
    )

    _progress(progress, "preprocess", 20, "检测人脸关键点和掩码...")
    view_data = preprocess_all_views(
        images,
        debug_dir=debug_dir,
        target_size=cfg.WORK_IMAGE_SIZE,
    )
    images_resized = {k: v["image"] for k, v in view_data.items()}
    face_masks = {k: v["face_mask"] for k, v in view_data.items()}

    fit_meta = run_stable_template_fit(
        preprocessed_views=view_data,
        intrinsics=intrinsics,
        output_dir=mesh_dir,
        cfg=cfg,
        progress=progress,
    )

    texture_meta = run_stable_texture_pipeline(
        mesh_dir=mesh_dir,
        images=images_resized,
        output_texture_dir=texture_dir,
        output_mesh_dir=mesh_dir,
        cfg=cfg,
        preprocessed_views=view_data,
        face_masks=face_masks,
        hires_images=original_hires,
        working_image_size=cfg.WORK_IMAGE_SIZE,
        baseline_quality=fit_meta["quality"]["geometry"],
        progress=progress,
    )

    _progress(progress, "quality_check", 88, "检查稳定模型质量...")
    quality = {
        "pipeline": "stable_three_view",
        "fit": {
            "neutral": fit_meta["quality"]["neutral"],
            "geometry": fit_meta["quality"]["geometry"],
            "gate": fit_meta["quality"]["gate"],
            "identity": fit_meta["quality"].get("identity", {}),
            "controlled_identity": (
                fit_meta.get("parameters", {})
                .get("optimized_parameters", {})
                .get("controlled_identity_deformation", {})
            ),
            "shape_refinement": (
                fit_meta.get("parameters", {})
                .get("optimized_shape", {})
                .get("shape_only_fine_tune", {})
            ),
        },
        "texture": {
            "final": texture_meta["quality"]["final"],
            "gate": texture_meta["quality"]["gate"],
            "confidence": texture_meta["confidence"],
            "texture": texture_meta["texture"],
        },
    }

    glb_path = Path(texture_meta["glb_path"])
    meta = {
        "pipeline": "stable_three_view",
        "legacy_pipeline": getattr(cfg, "LEGACY_PIPELINE", "legacy_residual"),
        "session_id": session_id,
        "patient_id": patient_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "three_view_constraint": True,
        "medical_measurement_claim": False,
        "disabled_geometry_stages": fit_meta["disabled_stages"],
        "artifact_paths": {
            "face_glb": _relative_or_absolute(glb_path, cfg.ROOT),
            "face_stable_glb": _relative_or_absolute(mesh_dir / "face_stable.glb", cfg.ROOT),
            "face_stable_neutral_glb": _relative_or_absolute(mesh_dir / "face_stable_neutral.glb", cfg.ROOT),
            "albedo_white": _relative_or_absolute(texture_dir / "albedo_white.png", cfg.ROOT),
            "texture_confidence_map": _relative_or_absolute(texture_dir / "texture_confidence_white.png", cfg.ROOT),
            "quality_json": _relative_or_absolute(report_dir / "quality.json", cfg.ROOT),
            "report_html": _relative_or_absolute(report_dir / "index.html", cfg.ROOT),
            "semantic_regions": _relative_or_absolute(mesh_dir / "stable_semantic_regions.json", cfg.ROOT),
            "controlled_identity_report": _relative_or_absolute(
                debug_dir / "controlled_identity_deformation" / "summary.json",
                cfg.ROOT,
            ),
        },
        "mesh_quality_summary": quality,
        "texture_confidence_summary": texture_meta["confidence"],
        "compare_sessions_note": "GLB point-cloud ICP is a coarse alignment aid, not medical-grade measurement.",
        "parameters": fit_meta.get("parameters", {}),
    }

    report_paths = write_stable_reconstruction_report(
        report_dir=report_dir,
        meta=meta,
        quality=quality,
        source_view_paths={k: Path(v) for k, v in image_paths.items()},
        debug_root=debug_dir,
    )
    meta["artifact_paths"]["report_html"] = _relative_or_absolute(Path(report_paths["index"]), cfg.ROOT)
    meta["artifact_paths"]["quality_json"] = _relative_or_absolute(Path(report_paths["quality"]), cfg.ROOT)
    with open(report_dir / "reconstruction_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _progress(progress, "done", 100, "稳定重建完成")
    return glb_path

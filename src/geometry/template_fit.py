"""Stable template fitting wrapper around the existing FLAME optimizer."""
from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np

from src.geometry.mesh_quality import (
    MeshQualityThresholds,
    assert_quality_gate,
    load_mesh_quality,
    make_quality_gate,
)

ProgressFn = Callable[[str, int, str], None]


@contextmanager
def temporary_config_overrides(cfg: Any, **overrides: Any) -> Iterator[None]:
    old_values = {}
    missing = object()
    for name, value in overrides.items():
        old_values[name] = getattr(cfg, name, missing)
        setattr(cfg, name, value)
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is missing:
                try:
                    delattr(cfg, name)
                except AttributeError:
                    pass
            else:
                setattr(cfg, name, value)


def _copy_if_exists(src: Path, dst: Path) -> Optional[str]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def stable_neutral_source(output_dir: Path) -> Path:
    """Return the geometry export produced with zero expression parameters."""
    source = Path(output_dir) / "face_mesh_neutral.glb"
    if not source.exists():
        raise FileNotFoundError(f"True neutral FLAME export is missing: {source}")
    return source


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _semantic_region_anchors(cfg: Any) -> Dict[str, Any]:
    """Return stable landmark-anchor indices for future controlled edits.

    These are anchor vertices on the original FLAME template. The current mesh
    keeps fixed topology after subdivision, so v1 stores them as semantic
    handles rather than allowing free residual deformation.
    """
    region_landmarks = {
        "nose_bridge": [27, 28, 29],
        "nose_tip": [30, 31, 32, 33, 34],
        "nose_wing": [31, 35],
        "chin": [7, 8, 9],
        "jaw": list(range(0, 17)),
        "cheek": [1, 2, 3, 13, 14, 15],
        "mouth": list(range(48, 68)),
    }
    try:
        from src.module2_geometry import FLAMEModel, load_flame_landmark_mapping

        flame = FLAMEModel(cfg.FLAME_MODEL_PATH, n_shape=cfg.N_SHAPE_PARAMS, n_exp=cfg.N_EXP_PARAMS)
        flame_faces_np = flame.faces.numpy()
        lmk_data = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
        if lmk_data is not None and "face_idx" in lmk_data:
            face_idx = lmk_data["face_idx"]
            bary = lmk_data["bary_coords"]
            landmark_vertex_indices = flame_faces_np[face_idx, np.argmax(bary, axis=1)]
        else:
            landmark_vertex_indices = np.arange(68, dtype=np.int64)

        regions = {}
        for name, landmark_ids in region_landmarks.items():
            vertex_ids = [int(landmark_vertex_indices[i]) for i in landmark_ids if i < len(landmark_vertex_indices)]
            regions[name] = sorted(set(vertex_ids))
        return {
            "source": "flame_68_landmark_anchors",
            "topology": "stable_subdivided_flame",
            "regions": regions,
        }
    except Exception as exc:
        return {
            "source": "unavailable",
            "reason": str(exc),
            "regions": {name: [] for name in region_landmarks},
        }


def run_stable_template_fit(
    *,
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    output_dir: Path,
    cfg: Any,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Run stable FLAME template fitting and block unsafe geometry stages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("stable_fit", 35, "稳定参数化人脸拟合中...")

    from src.module2_geometry import run_geometry_reconstruction

    disabled_stages = {
        "personal_residual_deform": True,
        "nose_mouth_local_residual": True,
        "nose_region_dense_residual": True,
        "free_identity_deform": True,
        "free_face_deform": True,
        "depth_anything_displacement": True,
    }
    overrides = {
        "ENABLE_PERSONAL_RESIDUAL_DEFORM": bool(getattr(cfg, "STABLE_ENABLE_FREE_RESIDUAL", False)),
        "ENABLE_NOSE_MOUTH_LOCAL_RESIDUAL": False,
        "ENABLE_NOSE_REGION_DENSE_RESIDUAL": False,
        "ENABLE_FREE_IDENTITY_DEFORM": False,
        "ENABLE_FREE_FACE_DEFORM": False,
        "DISPLACEMENT_SCALE": 0.0,
    }
    if bool(getattr(cfg, "STABLE_USE_CALIBRATED_RIG_EXTRINSICS", False)) and bool(
        getattr(cfg, "STABLE_DISABLE_POSE_REFINEMENT_WITH_RIG", True)
    ):
        overrides["ENABLE_POSE_REFINEMENT"] = False

    with temporary_config_overrides(cfg, **overrides):
        run_geometry_reconstruction(
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            flame_model_path=cfg.FLAME_MODEL_PATH,
            flame_landmark_path=cfg.FLAME_LANDMARK_PATH,
            deca_dir=cfg.DECA_REPO_DIR,
            deca_checkpoint=cfg.DECA_MODEL_PATH,
            depth_model_dir=cfg.DEPTH_MODEL_DIR,
            output_dir=output_dir,
            device=cfg.DEVICE,
            n_shape=cfg.N_SHAPE_PARAMS,
            n_exp=cfg.N_EXP_PARAMS,
            lambda_shape=cfg.LAMBDA_SHAPE,
            lambda_exp=cfg.LAMBDA_EXP,
            lbfgs_max_iter=cfg.LBFGS_MAX_ITER,
            lbfgs_lr=cfg.LBFGS_LR,
            depth_model_size=cfg.DEPTH_MODEL_SIZE,
            max_displacement=0.0,
            enable_depth_displacement=bool(getattr(cfg, "STABLE_ENABLE_DEPTH_DISPLACEMENT", False)),
            init_backend=cfg.INIT_BACKEND,
            mica_dir=cfg.MICA_DIR,
            mica_checkpoint=cfg.MICA_CHECKPOINT,
            emoca_dir=cfg.EMOCA_DIR,
            emoca_checkpoint=cfg.EMOCA_CHECKPOINT,
        )

    neutral_glb = stable_neutral_source(output_dir)
    stable_neutral_glb = output_dir / "face_stable_neutral.glb"
    stable_geometry_glb = output_dir / "face_stable_geometry.glb"
    _copy_if_exists(neutral_glb, stable_neutral_glb)
    _copy_if_exists(output_dir / "face_mesh_with_depth.glb", stable_geometry_glb)

    neutral_quality = load_mesh_quality(neutral_glb, label="stable_fit_neutral")
    geometry_quality = load_mesh_quality(stable_geometry_glb, label="stable_fit_geometry")
    gate = make_quality_gate(
        baseline=neutral_quality,
        candidate=geometry_quality,
        thresholds=MeshQualityThresholds(
            min_face_ratio=1.0,
            max_new_degenerate_faces=0,
            max_new_nonmanifold_edges=0,
            max_new_boundary_edges=0,
        ),
        region_name="stable_geometry",
    )
    assert_quality_gate(gate, context="stable geometry")

    semantic_regions = _semantic_region_anchors(cfg)
    semantic_path = output_dir / "stable_semantic_regions.json"
    with open(semantic_path, "w", encoding="utf-8") as f:
        json.dump(semantic_regions, f, ensure_ascii=False, indent=2)

    debug_dir = output_dir.parent / "debug"
    optimized_params = _load_json(debug_dir / "optimized_parameters.json")
    optimized_shape = _load_json(debug_dir / "optimized_shape.json")
    identity_quality = {
        "joint": optimized_shape.get("joint_identity_anchor", {}),
        "final": optimized_params.get("identity_preservation", {}),
        "controlled_low_frequency": optimized_params.get(
            "controlled_identity_deformation", {}
        ),
    }
    fit_meta = {
        "pipeline": "stable_three_view",
        "disabled_stages": disabled_stages,
        "paths": {
            "neutral_glb": str(stable_neutral_glb),
            "stable_geometry_glb": str(stable_geometry_glb),
            "semantic_regions": str(semantic_path),
        },
        "quality": {
            "neutral": neutral_quality,
            "geometry": geometry_quality,
            "gate": gate,
            "identity": identity_quality,
        },
        "parameters": {
            "optimized_parameters": optimized_params,
            "optimized_shape": optimized_shape,
            "semantic_regions": semantic_regions,
        },
    }
    with open(output_dir / "stable_fit_meta.json", "w", encoding="utf-8") as f:
        json.dump(fit_meta, f, ensure_ascii=False, indent=2)
    return fit_meta

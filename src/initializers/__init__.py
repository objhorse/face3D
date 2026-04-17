"""
Initializer factory.

Each initializer returns a standardized dict:
{
    "shape":    np.ndarray (n_shape,)   shared identity shape params
    "per_view": {
        view_name: {
            "exp":    np.ndarray (n_exp,)
            "R_init": np.ndarray (3, 3)
            "t_init": np.ndarray (3,)
        }
    }
    "backend":  str
    "view_status": {view_name: "ok" | "failed"}
}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_initializer(
    backend: str,
    images: Dict[str, np.ndarray],
    flame_verts: np.ndarray,
    lmk_vertex_indices: np.ndarray,
    intrinsics: Dict[str, np.ndarray],
    device: str,
    n_shape: int,
    n_exp: int,
    deca_dir: Optional[Path] = None,
    deca_checkpoint: Optional[Path] = None,
    mica_dir: Optional[Path] = None,
    mica_checkpoint: Optional[Path] = None,
    emoca_dir: Optional[Path] = None,
    emoca_checkpoint=None,
) -> dict:
    """
    Dispatch to the appropriate initialization backend.

    Falls back gracefully:
      mica_deca  -> tries MICA then DECA; if both fail -> face_alignment
      mica_emoca -> tries MICA then EMOCA; if both fail -> face_alignment
      deca       -> tries DECA; if fail -> face_alignment
      face_alignment -> baseline
    """
    backend = backend.lower()
    logger.info(f"Initializer backend: {backend}")

    if backend == "mica_deca":
        return _run_mica_deca(
            images,
            flame_verts,
            lmk_vertex_indices,
            intrinsics,
            device,
            n_shape,
            n_exp,
            mica_dir,
            mica_checkpoint,
            deca_dir,
            deca_checkpoint,
        )
    if backend == "mica_emoca":
        return _run_mica_emoca(
            images,
            flame_verts,
            lmk_vertex_indices,
            intrinsics,
            device,
            n_shape,
            n_exp,
            mica_dir,
            mica_checkpoint,
            emoca_dir,
            emoca_checkpoint,
        )
    if backend == "deca":
        return _run_deca_only(
            images,
            flame_verts,
            lmk_vertex_indices,
            intrinsics,
            device,
            n_shape,
            n_exp,
            deca_dir,
            deca_checkpoint,
        )

    if backend not in ("face_alignment",):
        logger.warning(f"Unknown backend '{backend}', fallback to face_alignment")
    return _run_face_alignment(
        images,
        flame_verts,
        lmk_vertex_indices,
        intrinsics,
        device,
        n_shape,
        n_exp,
    )


def _run_mica_deca(
    images,
    flame_verts,
    lmk_vertex_indices,
    intrinsics,
    device,
    n_shape,
    n_exp,
    mica_dir,
    mica_checkpoint,
    deca_dir,
    deca_checkpoint: Optional[Path] = None,
):
    from .deca_initializer import get_deca_per_view
    from .mica_initializer import get_mica_shape

    result = _empty_result("mica_deca", images)

    shape, mica_ok = get_mica_shape(images, mica_dir, mica_checkpoint, device, n_shape)
    result["shape"] = shape
    result["shape_source"] = "mica" if mica_ok else "zero"

    deca_views = get_deca_per_view(images, deca_dir, device, n_exp, deca_checkpoint)
    if deca_views is not None:
        for vname, vdata in deca_views.items():
            if vdata is not None:
                result["per_view"][vname]["exp"] = vdata["exp"]
        logger.info("DECA expression initialized; pose uses face_alignment PnP")
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)
        _fill_missing_pose_from_deca_cam(result, deca_views, intrinsics, "DECA")
    else:
        logger.info("DECA unavailable, fallback to face_alignment pose")
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)

    return result


def _run_mica_emoca(
    images,
    flame_verts,
    lmk_vertex_indices,
    intrinsics,
    device,
    n_shape,
    n_exp,
    mica_dir,
    mica_checkpoint,
    emoca_dir,
    emoca_checkpoint,
):
    from .emoca_initializer import get_emoca_per_view
    from .mica_initializer import get_mica_shape

    result = _empty_result("mica_emoca", images)

    shape, mica_ok = get_mica_shape(images, mica_dir, mica_checkpoint, device, n_shape)
    result["shape"] = shape
    result["shape_source"] = "mica" if mica_ok else "zero"

    emoca_views = get_emoca_per_view(images, emoca_dir, emoca_checkpoint, device, n_exp)
    if emoca_views is not None:
        for vname, vdata in emoca_views.items():
            if vdata is not None:
                result["per_view"][vname]["exp"] = vdata["exp"]
        logger.info("EMOCA expression initialized; pose uses face_alignment PnP")
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)
        _fill_missing_pose_from_deca_cam(result, emoca_views, intrinsics, "EMOCA")
    else:
        logger.info("EMOCA unavailable, fallback to face_alignment pose")
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)

    return result


def _run_deca_only(
    images,
    flame_verts,
    lmk_vertex_indices,
    intrinsics,
    device,
    n_shape,
    n_exp,
    deca_dir,
    deca_checkpoint: Optional[Path] = None,
):
    from .deca_initializer import get_deca_per_view

    result = _empty_result("deca", images)
    deca_views = get_deca_per_view(images, deca_dir, device, n_exp, deca_checkpoint)
    shape_candidates = []

    if deca_views is not None:
        for vname, vdata in deca_views.items():
            if vdata is not None:
                result["per_view"][vname]["exp"] = vdata["exp"]
                if "shape" in vdata:
                    shape_candidates.append(vdata["shape"][:n_shape])
        logger.info("DECA initialized; pose uses face_alignment PnP")
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)
        _fill_missing_pose_from_deca_cam(result, deca_views, intrinsics, "DECA")

    if shape_candidates:
        result["shape"] = np.mean(shape_candidates, axis=0)
        result["shape_source"] = "deca_mean"
    else:
        _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)

    return result


def _run_face_alignment(
    images,
    flame_verts,
    lmk_vertex_indices,
    intrinsics,
    device,
    n_shape,
    n_exp,
):
    result = _empty_result("face_alignment", images)
    _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device)
    result["shape_source"] = "zero"
    return result


def _empty_result(backend: str, images: dict) -> dict:
    return {
        "backend": backend,
        "shape": None,
        "shape_source": "zero",
        "per_view": {k: {"exp": None, "R_init": None, "t_init": None} for k in images},
        "view_status": {k: "failed" for k in images},
        "fa_lmk_2d": {},
    }


def _fill_fa_pose(result, images, flame_verts, lmk_vertex_indices, intrinsics, device):
    from .face_alignment_initializer import fa_pose_from_lmk, get_fa_per_view

    fa_views = get_fa_per_view(images, device)
    for vname, lmk in fa_views.items():
        if lmk is not None:
            lmk_2d = lmk[:, :2].astype(np.float32)
            result["fa_lmk_2d"][vname] = lmk_2d
            r_mat, t_vec = fa_pose_from_lmk(
                lmk_2d,
                flame_verts[lmk_vertex_indices],
                intrinsics[vname],
            )
            result["per_view"][vname]["R_init"] = r_mat
            result["per_view"][vname]["t_init"] = t_vec
            result["view_status"][vname] = "ok"
        else:
            result["fa_lmk_2d"][vname] = None


def _fill_missing_pose_from_deca_cam(result, view_data, intrinsics, source_name: str):
    for vname, vdata in view_data.items():
        pv = result["per_view"][vname]
        if (
            vdata is not None
            and (pv.get("R_init") is None or pv.get("t_init") is None)
        ):
            pv["R_init"], pv["t_init"] = _pose_from_deca_cam(vdata["cam"], intrinsics[vname])
            result["view_status"][vname] = "ok"
            logger.info(
                f"  [{vname}] face_alignment missing, fallback to {source_name} weak-perspective pose"
            )


def _pose_from_deca_cam(cam: np.ndarray, k_mat: np.ndarray):
    """
    Convert DECA weak-perspective camera [s, tx, ty] to an approximate perspective t.

    This is only a fallback when a landmark-based PnP pose is unavailable.
    Rotation remains identity because DECA camera alone does not encode a full
    perspective extrinsic rotation for our optimizer.
    """
    s = float(cam[0]) if cam[0] > 1e-4 else 1.0
    tx = float(cam[1])
    ty = float(cam[2])
    fx = float(k_mat[0, 0])
    t_z = fx / s
    t_x = tx * t_z
    t_y = ty * t_z
    r_mat = np.eye(3, dtype=np.float32)
    t_vec = np.array([t_x, t_y, t_z], dtype=np.float32)
    return r_mat, t_vec

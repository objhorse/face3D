"""
face_alignment fallback initializer.
Extracted from module2_geometry.py for clean reuse.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_PNP_STABLE_IDX = np.arange(27, 68, dtype=np.int32)


def get_fa_per_view(
    images: Dict[str, np.ndarray],
    device: str,
    max_size: int = 800,
) -> Dict[str, Optional[np.ndarray]]:
    """Return {view_name: lmk (68,3) or None}."""
    try:
        import face_alignment as fa_lib
    except ImportError:
        logger.error("face_alignment 未安装: pip install face-alignment")
        return {k: None for k in images}

    fa = fa_lib.FaceAlignment(fa_lib.LandmarksType.THREE_D, device=device, flip_input=False)
    results = {}
    for view_name, img in images.items():
        try:
            h, w = img.shape[:2]
            scale = min(1.0, max_size / max(h, w))
            img_small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
            preds = fa.get_landmarks(img_small)
            if preds and len(preds) > 0:
                lmk = preds[0].copy()
                lmk[:, :2] /= scale
                lmk[:, 2] /= scale
                results[view_name] = lmk
                logger.info(f"  [{view_name}] face_alignment 成功 (scale={scale:.2f})")
            else:
                logger.warning(f"  [{view_name}] face_alignment 未检测到人脸")
                results[view_name] = None
        except Exception as e:
            logger.warning(f"  [{view_name}] face_alignment 失败: {e}")
            results[view_name] = None
    return results


def fa_pose_from_lmk(
    lmk_2d: np.ndarray,        # (68, 2) image pixel coords
    flame_68_3d: np.ndarray,   # (68, 3) FLAME template vertices
    K: np.ndarray,             # (3, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """PnP pose estimation in the same FLAME coordinates used by the optimizer."""
    pts3d = np.asarray(flame_68_3d, dtype=np.float64)

    try:
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d[_PNP_STABLE_IDX],
            lmk_2d.astype(np.float64)[_PNP_STABLE_IDX],
            K.astype(np.float64),
            None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and tvec is not None and float(tvec[2]) > 0.05:
            R, _ = cv2.Rodrigues(rvec)
            return R.astype(np.float32), tvec.flatten().astype(np.float32)
    except Exception:
        pass

    return _estimate_pose_from_size(lmk_2d, pts3d, K)


def _estimate_pose_from_size(lmk_2d, pts3d, K):
    w_px  = float(lmk_2d[:, 0].max() - lmk_2d[:, 0].min())
    h_px  = float(lmk_2d[:, 1].max() - lmk_2d[:, 1].min())
    w_3d  = float(pts3d[:, 0].max() - pts3d[:, 0].min())
    h_3d  = float(pts3d[:, 1].max() - pts3d[:, 1].min())
    fx    = float(K[0, 0])
    fy    = float(K[1, 1])
    t_z   = fx * ((w_3d + h_3d) / 2.0) / max((w_px + h_px) / 2.0, 1.0)
    cx_px = float(lmk_2d[:, 0].mean())
    cy_px = float(lmk_2d[:, 1].mean())
    t_x   = (cx_px - K[0, 2]) * t_z / fx - float(pts3d[:, 0].mean())
    t_y   = (cy_px - K[1, 2]) * t_z / fy - float(pts3d[:, 1].mean())
    R = np.eye(3, dtype=np.float32)
    t = np.array([t_x, t_y, t_z], dtype=np.float32)
    logger.info(f"    尺寸估计初始姿态: t=({t_x:.3f}, {t_y:.3f}, {t_z:.3f})")
    return R, t

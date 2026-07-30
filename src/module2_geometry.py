"""
模块2：3DMM 几何重构

流程：
  Step 1 — DECA 单图初值（若可用）或 face_alignment 68点初值
  Step 2 — L-BFGS 联合优化（共享形状α，独立表情β/位姿R/t）
  Step 3 — Depth-Anything-V2 深度估计 + 尺度对齐 → 顶点置换
  Step 4 — 导出带 UV 的精细 Mesh (.obj)

依赖:
  - models/FLAME/generic_model.pkl  （FLAME 2020）
  - models/FLAME/landmark_embedding.npy  （FLAME→68点映射）
  - external/DECA/  （可选，提供更好初值）
  - Depth-Anything-V2 权重 （HuggingFace 自动下载）
"""
import sys
import logging
import pickle
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from src.coordinates import image_uv_to_obj_uv
from src.geometry.differentiable_silhouette import (
    IDENTITY_SILHOUETTE_REGIONS,
    build_silhouette_target,
    create_cuda_raster_context,
    evaluate_geometry_candidate,
    make_interior_landmark_weights,
    regional_signed_distance_boundary_loss,
    render_soft_silhouette,
    save_silhouette_debug,
    signed_distance_boundary_loss,
    silhouette_metrics,
    weighted_silhouette_loss,
)
from src.geometry.mesh_quality import (
    MeshQualityThresholds,
    compute_mesh_quality,
    make_quality_gate,
)
from src.geometry.identity_quality import (
    IdentityDriftThresholds,
    make_identity_drift_gate,
    mica_centered_shape_regularization,
    select_identity_safe_candidate,
    select_stable_refinement_checkpoint,
)
from src.geometry.expression_fidelity import (
    EYE_GAP_PAIRS,
    INNER_MOUTH_GAP_PAIRS,
    feature_gap_diagnostics,
    mediapipe_expression_state,
    paired_vertical_gap_loss,
)
from src.geometry.controlled_identity_deformation import (
    LowFrequencyAcceptanceThresholds,
    LowFrequencyBasisConfig,
    LowFrequencyOptimizationConfig,
    LowFrequencySafetyThresholds,
    build_low_frequency_identity_basis,
    evaluate_low_frequency_observations,
    evaluate_low_frequency_safety,
    optimize_low_frequency_identity,
    restrict_low_frequency_identity_basis,
    select_low_frequency_checkpoint,
)

logger = logging.getLogger(__name__)

_PNP_STABLE_IDX = np.arange(27, 68, dtype=np.int32)
INIT_POSE_MEAN_BAD_PX = 25.0
INIT_POSE_MAX_BAD_PX = 100.0
INIT_POSE_MIN_IMPROVE_PX = 1.0
LMK_CONTOUR_IDX = np.arange(0, 17, dtype=np.int64)
LMK_BROW_IDX = np.arange(17, 27, dtype=np.int64)
LMK_NOSE_IDX = np.arange(27, 36, dtype=np.int64)
LMK_EYE_IDX = np.arange(36, 48, dtype=np.int64)
LMK_MOUTH_IDX = np.arange(48, 68, dtype=np.int64)
LMK_INNER_MOUTH_IDX = np.arange(60, 68, dtype=np.int64)
LMK_NOSE_BRIDGE_IDX = np.arange(27, 31, dtype=np.int64)
LMK_NOSE_BASE_IDX = np.arange(31, 36, dtype=np.int64)
LMK_OUTER_MOUTH_IDX = np.arange(48, 60, dtype=np.int64)
LMK_NOSE_MOUTH_TARGET_IDX = np.concatenate([LMK_NOSE_BASE_IDX, LMK_OUTER_MOUTH_IDX, LMK_INNER_MOUTH_IDX])
LMK_NOSE_MOUTH_PROTECT_IDX = np.concatenate([LMK_NOSE_BRIDGE_IDX, LMK_EYE_IDX])
LMK_NOSE_REGION_PROTECT_IDX = np.concatenate([LMK_EYE_IDX, LMK_MOUTH_IDX])
LMK_GEOMETRY_IDX = np.concatenate([LMK_CONTOUR_IDX, LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
LMK_ERROR_GROUPS = (
    ("轮廓", LMK_CONTOUR_IDX),
    ("眉毛", LMK_BROW_IDX),
    ("鼻子", LMK_NOSE_IDX),
    ("眼睛", LMK_EYE_IDX),
    ("嘴巴", LMK_MOUTH_IDX),
)
LMK_GEOMETRY_GROUPS = (
    ("轮廓", LMK_CONTOUR_IDX),
    ("鼻子", LMK_NOSE_IDX),
    ("眼睛", LMK_EYE_IDX),
    ("嘴巴", LMK_MOUTH_IDX),
)
LMK_APPEARANCE_GROUPS = (
    ("眉毛", LMK_BROW_IDX),
)
LMK_LEFT_EYE_IDX = np.arange(36, 42, dtype=np.int64)
LMK_RIGHT_EYE_IDX = np.arange(42, 48, dtype=np.int64)
MP_FACE_OVAL_IDX = np.array([
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
], dtype=np.int64)
LMK_EYE_CLOSE_PAIRS = (
    (37, 41),
    (38, 40),
    (43, 47),
    (44, 46),
)


def _patch_numpy_compat():
    """
    FLAME 2020 的 .pkl 文件依赖 chumpy，chumpy 使用了
    numpy 1.20 之前的类型别名（np.bool/np.int/np.float 等）。
    numpy 2.x 已彻底删除这些别名，在 pickle.load 前临时注入。
    """
    if not hasattr(np, "bool"):
        np.bool    = np.bool_
    if not hasattr(np, "int"):
        np.int     = np.int_
    if not hasattr(np, "float"):
        np.float   = np.float_
    if not hasattr(np, "complex"):
        np.complex = np.complex_
    if not hasattr(np, "object"):
        np.object  = np.object_
    if not hasattr(np, "str"):
        np.str     = np.str_
    if not hasattr(np, "unicode"):
        np.unicode = np.str_


# ══════════════════════════════════════════════════════════════════════════════
# FLAME 模型
# ══════════════════════════════════════════════════════════════════════════════

class FLAMEModel(nn.Module):
    """
    FLAME 2020 可微前向模型（简化版，用于优化阶段）。
    仅实现形状+表情混合形态，不含完整 LBS（姿态通过外部 R/t 处理）。
    """

    def __init__(self, model_path: Path, n_shape: int = 100, n_exp: int = 50):
        super().__init__()
        if not model_path.exists():
            raise FileNotFoundError(
                f"FLAME 模型文件不存在: {model_path}\n"
                "请从 https://flame.is.tue.mpg.de/ 下载 FLAME 2020，\n"
                "解压后将 generic_model.pkl 放至 models/FLAME/ 目录。"
            )
        # chumpy 使用了已被 numpy 2.x 删除的别名，加载前打补丁
        _patch_numpy_compat()
        with open(model_path, "rb") as f:
            fm = pickle.load(f, encoding="latin1")

        # v_template: (5023, 3)
        v_template = torch.tensor(np.array(fm["v_template"]), dtype=torch.float32)
        self.register_buffer("v_template", v_template)

        # shapedirs: (5023*3, 400)  前300=形状，后100=表情
        shapedirs = np.array(fm["shapedirs"])          # may be (5023, 3, 400)
        if shapedirs.ndim == 3:
            shapedirs = shapedirs.reshape(-1, shapedirs.shape[-1])  # (15069, 400)
        shape_basis = torch.tensor(shapedirs[:, :n_shape], dtype=torch.float32)   # (15069, n_shape)
        exp_basis   = torch.tensor(shapedirs[:, 300:300 + n_exp], dtype=torch.float32)
        self.register_buffer("shape_basis", shape_basis)
        self.register_buffer("exp_basis",   exp_basis)

        # 三角面片: (9976, 3)
        faces = torch.tensor(np.array(fm["f"]).astype(np.int64), dtype=torch.long)
        self.register_buffer("faces", faces)

        self.n_verts  = int(v_template.shape[0])
        self.n_shape  = n_shape
        self.n_exp    = n_exp

    def forward(
        self,
        shape_params: torch.Tensor,   # (n_shape,) — 共享
        exp_params: torch.Tensor,     # (n_exp,)   — 每视角独立
    ) -> torch.Tensor:
        """Returns: vertices (N, 3)"""
        v = self.v_template.clone()                              # (N, 3)
        # 形状变形
        delta_shape = (self.shape_basis @ shape_params).reshape(self.n_verts, 3)
        v = v + delta_shape
        # 表情变形
        delta_exp = (self.exp_basis @ exp_params).reshape(self.n_verts, 3)
        v = v + delta_exp
        return v


# ══════════════════════════════════════════════════════════════════════════════
# 关键点映射（FLAME → 68 标准人脸点）
# ══════════════════════════════════════════════════════════════════════════════

def load_flame_landmark_mapping(landmark_path: Path) -> Optional[dict]:
    """
    加载 FLAME → 68点 的重心坐标映射（来自 DECA data/landmark_embedding.npy）。
    文件结构:
      full_lmk_faces_idx:  (1, 68) int64  — 面片索引
      full_lmk_bary_coords:(1, 68, 3) float — 重心坐标
    返回 dict 或 None（文件不存在时）。
    """
    if not landmark_path.exists():
        logger.warning(
            f"landmark_embedding.npy 不存在: {landmark_path}\n"
            "将使用近似顶点索引，精度略低。"
        )
        return None
    data = np.load(str(landmark_path), allow_pickle=True).item()
    # 统一映射到标准 key 名
    result = {}
    if "full_lmk_faces_idx" in data:
        result["face_idx"]   = np.array(data["full_lmk_faces_idx"]).reshape(68)       # (68,)
        result["bary_coords"]= np.array(data["full_lmk_bary_coords"]).reshape(68, 3)  # (68, 3)
    elif "lmk_face_idx" in data:
        result["face_idx"]   = np.array(data["lmk_face_idx"]).reshape(68)
        result["bary_coords"]= np.array(data["lmk_b_coords"]).reshape(68, 3)
    else:
        logger.warning(f"landmark_embedding.npy 格式未知，keys={list(data.keys())}")
        return None
    logger.info(f"FLAME landmark 映射加载成功（68点）")
    return result


def project_vertices(
    vertices: torch.Tensor,   # (N, 3)
    K: torch.Tensor,          # (3, 3)
    R: torch.Tensor,          # (3, 3)
    t: torch.Tensor,          # (3,)
) -> torch.Tensor:
    """透视投影: Returns (N, 2) 像素坐标"""
    v_cam = (R @ vertices.T + t.unsqueeze(1)).T      # (N, 3)
    v_hom = (K @ v_cam.T).T                           # (N, 3)
    v_2d  = v_hom[:, :2] / v_hom[:, 2:3].clamp(min=1e-6)
    return v_2d


def rodrigues_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    """轴角 (3,) → 旋转矩阵 (3,3)，批量可微"""
    angle = torch.norm(rvec + 1e-8)
    axis  = rvec / angle
    K_mat = torch.zeros(3, 3, device=rvec.device, dtype=rvec.dtype)
    K_mat[0, 1] = -axis[2]; K_mat[0, 2] =  axis[1]
    K_mat[1, 0] =  axis[2]; K_mat[1, 2] = -axis[0]
    K_mat[2, 0] = -axis[1]; K_mat[2, 1] =  axis[0]
    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype)
    R = torch.cos(angle) * I + (1 - torch.cos(angle)) * (axis.unsqueeze(1) @ axis.unsqueeze(0)) + torch.sin(angle) * K_mat
    return R


# ══════════════════════════════════════════════════════════════════════════════
# DECA 初值估计
# ══════════════════════════════════════════════════════════════════════════════

def get_deca_initial_params(
    images: Dict[str, np.ndarray],
    deca_dir: Path,
    device: str,
) -> Optional[Dict[str, dict]]:
    """
    使用 DECA 对每张图独立推理，得到 FLAME 参数初值。
    返回: {view_name: {"shape": (100,), "exp": (50,), "pose": (6,), "cam": (3,)}}
    若 DECA 不可用返回 None。
    """
    if not deca_dir.exists():
        return None
    if str(deca_dir) not in sys.path:
        sys.path.insert(0, str(deca_dir))
    try:
        from decalib.deca import DECA
        from decalib.utils.config import cfg as deca_cfg
        from decalib.datasets.detectors import FAN
        from decalib.utils import util as deca_util
    except ImportError:
        logger.info("DECA 未安装，将使用 face_alignment 初值")
        return None

    logger.info("使用 DECA 估计初始 FLAME 参数...")
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=device)

    results = {}
    for view_name, img in images.items():
        try:
            # DECA 要求 (224,224) BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_bgr, (224, 224))
            img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                codedict = deca.encode(img_tensor)

            results[view_name] = {
                "shape": codedict["shape"].squeeze(0).cpu().numpy(),
                "exp":   codedict["exp"].squeeze(0).cpu().numpy(),
                "pose":  codedict["pose"].squeeze(0).cpu().numpy(),
                "cam":   codedict["cam"].squeeze(0).cpu().numpy(),
            }
            logger.info(f"  [{view_name}] DECA 初值获取成功")
        except Exception as e:
            logger.warning(f"  [{view_name}] DECA 推理失败: {e}")
            results[view_name] = None

    return results if any(v is not None for v in results.values()) else None


def get_fa_initial_params(
    images: Dict[str, np.ndarray],
    device: str,
    max_size: int = 800,
) -> Dict[str, np.ndarray]:
    """
    face_alignment fallback: 返回 68 个 3D 关键点（图像像素坐标）。
    内部缩放处理大图（face_alignment 在超大分辨率下检测失败）。
    """
    try:
        import face_alignment as fa_lib
    except ImportError:
        raise ImportError("请安装 face_alignment: pip install face-alignment")

    fa = fa_lib.FaceAlignment(
        fa_lib.LandmarksType.THREE_D,
        device=device,
        flip_input=False,
    )

    results = {}
    for view_name, img in images.items():
        try:
            h, w = img.shape[:2]
            scale = min(1.0, max_size / max(h, w))
            if scale < 1.0:
                img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
            else:
                img_small = img
                scale = 1.0

            preds = fa.get_landmarks(img_small)
            if preds is not None and len(preds) > 0:
                lmk = preds[0].copy()           # (68, 3)
                lmk[:, :2] /= scale             # 反变换回原图坐标
                lmk[:, 2]  /= scale             # z 也按比例还原
                results[view_name] = lmk
                logger.info(f"  [{view_name}] face_alignment 3D 关键点获取成功 (scale={scale:.2f})")
            else:
                logger.warning(f"  [{view_name}] face_alignment 未检测到人脸")
                results[view_name] = None
        except Exception as e:
            logger.warning(f"  [{view_name}] face_alignment 失败: {e}")
            results[view_name] = None

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 联合 L-BFGS 优化
# ══════════════════════════════════════════════════════════════════════════════

def estimate_pose_from_landmarks_pnp(
    lmks_2d: np.ndarray,       # (68, 2) 图像像素坐标（Y 朝下）
    lmks_3d_model: np.ndarray, # (68, 3) FLAME 模板顶点
    K: np.ndarray,             # (3, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PnP 估计初始 R, t。
    PnP 和后续 optimizer 使用同一套 FLAME 坐标，不在这里额外翻转 Y。
    同时检测退化解（t_z ≤ 0）并回退到基于尺寸的估计。
    """
    pts3d = np.asarray(lmks_3d_model, dtype=np.float64)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d[_PNP_STABLE_IDX],
            lmks_2d.astype(np.float64)[_PNP_STABLE_IDX],
            K.astype(np.float64),
            None,
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success and tvec is not None:
            t_z = float(tvec[2])
            if t_z > 0.05:  # 合理的深度值
                R, _ = cv2.Rodrigues(rvec)
                return R.astype(np.float32), tvec.flatten().astype(np.float32)
    except Exception:
        pass

    # fallback：基于人脸关键点尺寸估计初始深度
    return _estimate_pose_from_size(lmks_2d, pts3d, K)


def _estimate_pose_from_size(
    lmks_2d: np.ndarray,   # (68, 2) 像素
    lmks_3d: np.ndarray,   # (68, 3) FLAME（Y已翻转）
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于人脸包围盒尺寸估计初始深度：
    t_z ≈ fx * face_size_3d / face_size_px
    t_x, t_y 由人脸中心反投影确定。
    假设正脸（R=I），适合作为优化起点。
    """
    # 2D 人脸尺寸（像素）
    w_px = float(lmks_2d[:, 0].max() - lmks_2d[:, 0].min())
    h_px = float(lmks_2d[:, 1].max() - lmks_2d[:, 1].min())
    size_px = (w_px + h_px) / 2.0

    # 3D FLAME 人脸尺寸
    w_3d = float(lmks_3d[:, 0].max() - lmks_3d[:, 0].min())
    h_3d = float(lmks_3d[:, 1].max() - lmks_3d[:, 1].min())
    size_3d = (w_3d + h_3d) / 2.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    t_z = fx * size_3d / max(size_px, 1.0)

    # 人脸中心在图像中的位置
    cx_px = float(lmks_2d[:, 0].mean())
    cy_px = float(lmks_2d[:, 1].mean())

    # 3D 人脸中心
    cx_3d = float(lmks_3d[:, 0].mean())
    cy_3d = float(lmks_3d[:, 1].mean())

    # 反投影中心差
    t_x = (cx_px - cx) * t_z / fx - cx_3d
    t_y = (cy_px - cy) * t_z / fy - cy_3d

    R = np.eye(3, dtype=np.float32)
    t = np.array([t_x, t_y, t_z], dtype=np.float32)
    logger.info(f"    尺寸估计初始姿态: t=({t_x:.3f}, {t_y:.3f}, {t_z:.3f})")
    return R, t


def _closed_eye_landmark_loss(lmk_proj_n: torch.Tensor, lmk_target_n: torch.Tensor) -> torch.Tensor:
    losses = []
    for upper_idx, lower_idx in LMK_EYE_CLOSE_PAIRS:
        target_mid = 0.5 * (lmk_target_n[upper_idx] + lmk_target_n[lower_idx])
        proj_pair = torch.stack([lmk_proj_n[upper_idx], lmk_proj_n[lower_idx]], dim=0)
        losses.append(
            F.smooth_l1_loss(
                proj_pair,
                target_mid.unsqueeze(0).expand_as(proj_pair),
                reduction="mean",
                beta=0.004,
            )
        )
        proj_gap = torch.abs(lmk_proj_n[upper_idx, 1] - lmk_proj_n[lower_idx, 1])
        losses.append(
            F.smooth_l1_loss(
                proj_gap,
                torch.zeros_like(proj_gap),
                reduction="mean",
                beta=0.002,
            )
        )
    return torch.stack(losses).mean()


def _force_close_eye_geometry(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    target_landmarks: np.ndarray,
    faces: Optional[np.ndarray] = None,
    vertex_normals: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    strength: float = 0.95,
    debug_image: Optional[np.ndarray] = None,
    debug_path: Optional[Path] = None,
) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64).copy()
    target_landmarks = np.asarray(target_landmarks, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    v_cam = (R @ verts.T + t[:, None]).T
    z = v_cam[:, 2]
    front = z > 1e-4
    proj = np.zeros((len(verts), 2), dtype=np.float64)
    proj[front, 0] = K[0, 0] * v_cam[front, 0] / z[front] + K[0, 2]
    proj[front, 1] = K[1, 1] * v_cam[front, 1] / z[front] + K[1, 2]

    if image_shape is None and debug_image is not None:
        image_shape = debug_image.shape[:2]
    in_frame = front.copy()
    if image_shape is not None:
        H, W = image_shape
        in_frame &= (
            (proj[:, 0] >= 0.0)
            & (proj[:, 0] < float(W))
            & (proj[:, 1] >= 0.0)
            & (proj[:, 1] < float(H))
        )

    normal_ok = np.ones(len(verts), dtype=bool)
    if vertex_normals is not None:
        normals = np.asarray(vertex_normals, dtype=np.float64)
        if normals.shape == verts.shape:
            view_dir_world = -R[2, :]
            normal_ok = (normals @ view_dir_world) > 0.05

    depth_ok = np.ones(len(verts), dtype=bool)
    if image_shape is not None:
        H, W = image_shape
        depth_points = np.full((H, W), np.inf, dtype=np.float32)
        px = np.rint(proj[:, 0]).astype(np.int64)
        py = np.rint(proj[:, 1]).astype(np.int64)
        valid_px = in_frame & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        np.minimum.at(depth_points, (py[valid_px], px[valid_px]), z[valid_px].astype(np.float32))
        depth_ok[:] = False
        visible_indices = np.flatnonzero(valid_px)
        for idx in visible_indices:
            x = px[idx]
            y = py[idx]
            local = depth_points[max(0, y - 3):min(H, y + 4), max(0, x - 3):min(W, x + 4)]
            nearest_z = float(np.min(local))
            depth_ok[idx] = np.isfinite(nearest_z) and (z[idx] <= nearest_z + 0.004)

    selected_all = np.zeros(len(verts), dtype=bool)
    for eye_idx, corner_a, corner_b in (
        (LMK_LEFT_EYE_IDX, 36, 39),
        (LMK_RIGHT_EYE_IDX, 42, 45),
    ):
        eye_lmks = target_landmarks[eye_idx]
        x_min = float(eye_lmks[:, 0].min())
        x_max = float(eye_lmks[:, 0].max())
        width = max(x_max - x_min, 1.0)
        x_margin = max(5.0, width * 0.10)
        y_center = float(np.median(eye_lmks[:, 1]))
        y_margin = max(7.0, width * 0.18)
        x0, x1 = target_landmarks[corner_a, 0], target_landmarks[corner_b, 0]
        y0, y1 = target_landmarks[corner_a, 1], target_landmarks[corner_b, 1]

        in_eye = (
            in_frame
            & normal_ok
            & depth_ok
            & (proj[:, 0] >= x_min - x_margin)
            & (proj[:, 0] <= x_max + x_margin)
            & (proj[:, 1] >= y_center - y_margin)
            & (proj[:, 1] <= y_center + y_margin)
        )
        if not np.any(in_eye):
            continue

        if abs(float(x1 - x0)) < 1e-6:
            target_y = np.full(in_eye.sum(), y_center, dtype=np.float64)
        else:
            alpha = np.clip((proj[in_eye, 0] - x0) / (x1 - x0), 0.0, 1.0)
            target_y = (1.0 - alpha) * y0 + alpha * y1
            target_y = 0.65 * target_y + 0.35 * y_center

        dist = np.abs(proj[in_eye, 1] - target_y)
        local_weight = np.clip(1.0 - dist / y_margin, 0.0, 1.0) ** 0.5
        new_y = proj[in_eye, 1] + np.clip(strength, 0.0, 1.0) * local_weight * (target_y - proj[in_eye, 1])
        v_cam[in_eye, 1] = (new_y - K[1, 2]) * z[in_eye] / K[1, 1]
        selected_all |= in_eye

    closed_verts = (R.T @ (v_cam - t[None, :]).T).T

    if debug_image is not None and debug_path is not None:
        after_cam = (R @ closed_verts.T + t[:, None]).T
        after_z = after_cam[:, 2]
        after_front = after_z > 1e-4
        after = np.zeros_like(proj)
        after[after_front, 0] = K[0, 0] * after_cam[after_front, 0] / after_z[after_front] + K[0, 2]
        after[after_front, 1] = K[1, 1] * after_cam[after_front, 1] / after_z[after_front] + K[1, 2]
        img = cv2.cvtColor(debug_image.copy(), cv2.COLOR_RGB2BGR)
        pts_before = proj[selected_all]
        pts_after = after[selected_all]
        if len(pts_before) > 6000:
            take = np.linspace(0, len(pts_before) - 1, 6000).astype(np.int64)
            pts_before = pts_before[take]
            pts_after = pts_after[take]
        for x, y in pts_before:
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (0, 200, 255), -1)
        for x, y in pts_after:
            cv2.circle(img, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), img)
        logger.info(f"Closed-eye geometry debug saved: {debug_path} ({int(selected_all.sum())} vertices)")

    return closed_verts.astype(np.asarray(vertices).dtype, copy=False)


class JointFLAMEOptimizer:
    """
    联合 L-BFGS 优化器。
    优化目标：shape α（共享） + 每视角的 exp β、rvec（轴角）、t
    损失：Σ_v ||proj(FLAME(α,βv), Kv, Rv, tv)[lmk_bary] - lmk_2d_v||² + λ||α||² + λ||β||²

    landmark 位置用重心坐标插值（精度高于取单个顶点）。
    """

    def __init__(
        self,
        flame: FLAMEModel,
        lmk_vertex_indices: np.ndarray,  # (68,) int — 近似顶点索引（fallback用）
        lambda_shape: float = 1e-3,
        identity_anchor_weight: Optional[float] = None,
        mean_shape_prior_weight: float = 0.0,
        lambda_exp: float = 1e-3,
        lambda_contour: float = 0.0,
        front_contour_weight: float = 2.4,
        front_jaw_weight: float = 3.2,
        side_contour_weight: float = 1.2,
        side_jaw_weight: float = 1.8,
        side_brow_weight: float = 0.25,
        side_extra_soft_weight: float = 0.6,
        eye_gap_loss_weight: float = 4.0,
        mouth_gap_loss_weight: float = 6.0,
        closed_eye_target_scale: float = 0.10,
        closed_mouth_target_scale: float = 0.08,
        max_iter: int = 100,
        lr: float = 0.5,
        device: str = "cuda",
        shared_expression: bool = False,
        lmk_face_idx: Optional[np.ndarray] = None,    # (68,) — 精确重心坐标用
        lmk_bary_coords: Optional[np.ndarray] = None, # (68, 3)
    ):
        self.flame   = flame.to(device)
        self.lambda_shape = lambda_shape
        self.identity_anchor_weight = float(
            lambda_shape if identity_anchor_weight is None else identity_anchor_weight
        )
        self.mean_shape_prior_weight = float(mean_shape_prior_weight)
        self.lambda_exp   = lambda_exp
        self.lambda_contour = lambda_contour
        self.front_contour_weight = front_contour_weight
        self.front_jaw_weight = front_jaw_weight
        self.side_contour_weight = side_contour_weight
        self.side_jaw_weight = side_jaw_weight
        self.side_brow_weight = side_brow_weight
        self.side_extra_soft_weight = side_extra_soft_weight
        self.eye_gap_loss_weight = float(eye_gap_loss_weight)
        self.mouth_gap_loss_weight = float(mouth_gap_loss_weight)
        self.closed_eye_target_scale = float(closed_eye_target_scale)
        self.closed_mouth_target_scale = float(closed_mouth_target_scale)
        self.max_iter = max_iter
        self.lr       = lr
        self.device   = device
        self.shared_expression = bool(shared_expression)

        # 优先使用重心坐标插值
        if lmk_face_idx is not None and lmk_bary_coords is not None:
            self.use_bary = True
            faces_np = flame.faces.cpu().numpy()  # (F, 3)
            # 预计算每个 landmark 对应的3个顶点索引
            lmk_tri_verts = faces_np[lmk_face_idx]   # (68, 3)
            self.lmk_v0 = torch.tensor(lmk_tri_verts[:, 0], dtype=torch.long, device=device)
            self.lmk_v1 = torch.tensor(lmk_tri_verts[:, 1], dtype=torch.long, device=device)
            self.lmk_v2 = torch.tensor(lmk_tri_verts[:, 2], dtype=torch.long, device=device)
            bary = torch.tensor(lmk_bary_coords, dtype=torch.float32, device=device)  # (68, 3)
            self.bary_w0 = bary[:, 0:1]  # (68, 1)
            self.bary_w1 = bary[:, 1:2]
            self.bary_w2 = bary[:, 2:3]
        else:
            self.use_bary = False
            self.lmk_idx = torch.tensor(lmk_vertex_indices, dtype=torch.long, device=device)

    def optimize(
        self,
        views: Dict[str, dict],
        # views[name] = {"lmk_2d": (68,2), "K": (3,3), "R_init": (3,3), "t_init": (3,)}
        init_shape: Optional[np.ndarray] = None,
        init_exps:  Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, dict]]:
        """
        Returns:
            shape_params: (n_shape,) — 最优共享形状
            per_view: {name: {"R": (3,3), "t": (3,), "exp": (n_exp,)}}
        """
        n_shape = self.flame.n_shape
        n_exp   = self.flame.n_exp
        dev     = self.device

        # 初始化优化参数
        shape_param = torch.zeros(n_shape, device=dev, dtype=torch.float32)
        if init_shape is not None:
            shape_param.data[:min(n_shape, len(init_shape))] = torch.tensor(
                init_shape[:n_shape], dtype=torch.float32, device=dev
            )
        identity_anchor = shape_param.detach().clone()
        shape_param.requires_grad_(True)

        view_names = list(views.keys())
        exp_params, rvec_params, t_params = {}, {}, {}

        shared_exp_param = None
        if self.shared_expression:
            initial_expressions = [
                np.asarray(init_exps[name][:n_exp], dtype=np.float32)
                for name in view_names
                if init_exps and name in init_exps and init_exps[name] is not None
            ]
            shared_init = np.zeros(n_exp, dtype=np.float32)
            if initial_expressions:
                mean_expression = np.mean(np.stack(initial_expressions, axis=0), axis=0)
                shared_init[:len(mean_expression)] = mean_expression
            shared_exp_param = torch.tensor(
                shared_init, device=dev, dtype=torch.float32, requires_grad=True
            )

        for name in view_names:
            v = views[name]

            if shared_exp_param is not None:
                ep = shared_exp_param
            else:
                ep = torch.zeros(n_exp, device=dev, dtype=torch.float32)
                if init_exps and name in init_exps and init_exps[name] is not None:
                    e = init_exps[name][:n_exp]
                    ep.data[:len(e)] = torch.tensor(e, dtype=torch.float32, device=dev)
                ep.requires_grad_(True)
            exp_params[name] = ep

            # 从初始 R 转换为轴角
            R_np = v["R_init"]
            rvec_np = Rotation.from_matrix(R_np).as_rotvec().astype(np.float32)
            rv = torch.tensor(rvec_np, device=dev, dtype=torch.float32, requires_grad=True)
            rvec_params[name] = rv

            tv = torch.tensor(v["t_init"], device=dev, dtype=torch.float32, requires_grad=True)
            t_params[name] = tv

        # 准备固定量（K、target landmarks）
        Ks = {
            name: torch.tensor(views[name]["K"], device=dev, dtype=torch.float32)
            for name in view_names
        }
        lmk_targets = {
            name: torch.tensor(views[name]["lmk_2d"], device=dev, dtype=torch.float32)
            for name in view_names
        }
        contour_rows = {}
        for name in view_names:
            mask = views[name].get("shape_mask")
            if mask is None:
                mask = views[name].get("face_mask")
            if mask is None:
                contour_rows[name] = None
                continue
            xmin, xmax, valid = _build_mask_row_bounds(mask)
            contour_rows[name] = {
                "xmin": torch.tensor(xmin, device=dev, dtype=torch.float32),
                "xmax": torch.tensor(xmax, device=dev, dtype=torch.float32),
                "valid": torch.tensor(valid, device=dev, dtype=torch.bool),
            }
        lmk_weights = {}
        contour_idx_t = torch.tensor(LMK_CONTOUR_IDX, device=dev)
        jaw_idx_t = torch.tensor(np.arange(4, 13, dtype=np.int64), device=dev)
        brow_idx_t = torch.tensor(LMK_BROW_IDX, device=dev)
        side_extra_soft_idx_t = torch.tensor(
            np.array([36, 37, 38, 39, 42, 43, 44, 45], dtype=np.int64),
            device=dev,
        )
        try:
            from src import config as cfg
            force_closed_eyes = bool(getattr(cfg, "FORCE_CLOSED_EYES", False))
            closed_eye_loss_weight = float(getattr(cfg, "CLOSED_EYE_LOSS_WEIGHT", 0.0))
        except Exception:
            force_closed_eyes = False
            closed_eye_loss_weight = 0.0
        eye_idx_t = torch.tensor(LMK_EYE_IDX, device=dev)
        for name in view_names:
            w = torch.ones(68, device=dev, dtype=torch.float32)
            if name == "front":
                w[contour_idx_t] = float(self.front_contour_weight)
                w[jaw_idx_t] = float(self.front_jaw_weight)
            else:
                w[contour_idx_t] = float(self.side_contour_weight)
                w[jaw_idx_t] = float(self.side_jaw_weight)
                w[brow_idx_t] = float(self.side_brow_weight)
                w[side_extra_soft_idx_t] = float(self.side_extra_soft_weight)
            if force_closed_eyes:
                w[eye_idx_t] = torch.maximum(w[eye_idx_t], torch.full_like(w[eye_idx_t], 3.0))
            # Image-space face-oval points slide over the surface and are not
            # fixed anatomical correspondences, including in the front view.
            w[contour_idx_t] = 0.0
            lmk_weights[name] = w

        use_rig_extrinsics = all(
            views[name].get("rig_R_ref_to_camera") is not None
            and views[name].get("rig_t_ref_to_camera") is not None
            for name in view_names
        )
        rig_reference_name = None
        rig_R_ref_to_camera = {}
        rig_t_ref_to_camera = {}
        if use_rig_extrinsics:
            candidates = [
                name for name in view_names
                if views[name].get("rig_reference_view") == name
            ]
            if not candidates and "front" in view_names:
                candidates = ["front"]
            rig_reference_name = candidates[0] if candidates else view_names[0]
            for name in view_names:
                rig_R_ref_to_camera[name] = torch.tensor(
                    views[name]["rig_R_ref_to_camera"],
                    device=dev,
                    dtype=torch.float32,
                )
                rig_t_ref_to_camera[name] = torch.tensor(
                    views[name]["rig_t_ref_to_camera"],
                    device=dev,
                    dtype=torch.float32,
                )
            logger.info(
                "启用固定 rig 外参优化：reference=%s，优化一个全局人脸位姿并推导左右相机。",
                rig_reference_name,
            )

        def _pose_for_view(name: str):
            if use_rig_extrinsics:
                R_ref = rodrigues_to_matrix(rvec_params[rig_reference_name])
                t_ref = t_params[rig_reference_name]
                R_rel = rig_R_ref_to_camera[name]
                t_rel = rig_t_ref_to_camera[name]
                return R_rel @ R_ref, R_rel @ t_ref + t_rel
            return rodrigues_to_matrix(rvec_params[name]), t_params[name]

        pose_params = (
            [rvec_params[rig_reference_name], t_params[rig_reference_name]]
            if use_rig_extrinsics
            else list(rvec_params.values()) + list(t_params.values())
        )
        all_params = (
            [shape_param]
            + ([shared_exp_param] if shared_exp_param is not None else list(exp_params.values()))
            + pose_params
        )
        optimizer = torch.optim.LBFGS(
            all_params, lr=self.lr, max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=dev)

            for name in view_names:
                verts = self.flame(shape_param, exp_params[name])   # (N, 3)
                R_cur, t_cur = _pose_for_view(name)
                proj  = project_vertices(verts, Ks[name], R_cur, t_cur)  # (N, 2)

                if self.use_bary:
                    lmk_proj = (
                        proj[self.lmk_v0] * self.bary_w0 +
                        proj[self.lmk_v1] * self.bary_w1 +
                        proj[self.lmk_v2] * self.bary_w2
                    )  # (68, 2)
                else:
                    lmk_proj = proj[self.lmk_idx]

                # 归一化坐标（降低绝对像素值带来的数值问题）
                lmk_target_n = lmk_targets[name] / 1000.0
                lmk_proj_n   = lmk_proj / 1000.0

                per_point = F.smooth_l1_loss(
                    lmk_proj_n, lmk_target_n, reduction="none", beta=0.01
                ).mean(dim=1)
                lmk_loss = (per_point * lmk_weights[name]).sum() / lmk_weights[name].sum().clamp_min(1e-6)
                total_loss = total_loss + lmk_loss
                total_loss = total_loss + self.eye_gap_loss_weight * paired_vertical_gap_loss(
                    lmk_proj_n,
                    lmk_target_n,
                    EYE_GAP_PAIRS,
                    target_scale=(
                        self.closed_eye_target_scale
                        if views[name].get("closed_eyes", False) else 1.0
                    ),
                )
                total_loss = total_loss + self.mouth_gap_loss_weight * paired_vertical_gap_loss(
                    lmk_proj_n,
                    lmk_target_n,
                    INNER_MOUTH_GAP_PAIRS,
                    target_scale=(
                        self.closed_mouth_target_scale
                        if views[name].get("closed_mouth", False) else 1.0
                    ),
                )
                if force_closed_eyes and closed_eye_loss_weight > 0:
                    eye_close_loss = _closed_eye_landmark_loss(lmk_proj_n, lmk_target_n)
                    total_loss = total_loss + closed_eye_loss_weight * eye_close_loss

                contour_data = contour_rows.get(name)
                if contour_data is not None and self.lambda_contour > 0:
                    contour_proj = lmk_proj[contour_idx_t]
                    row_idx = torch.round(contour_proj[:, 1]).long()
                    row_idx = row_idx.clamp(0, contour_data["valid"].shape[0] - 1)
                    valid_rows = contour_data["valid"][row_idx]
                    if torch.any(valid_rows):
                        jaw_x = contour_proj[:, 0][valid_rows]
                        xmin = contour_data["xmin"][row_idx][valid_rows]
                        xmax = contour_data["xmax"][row_idx][valid_rows]
                        side_dist = torch.minimum(torch.abs(jaw_x - xmin), torch.abs(jaw_x - xmax))
                        excess = torch.clamp(side_dist - 6.0, min=0.0)
                        contour_loss = F.smooth_l1_loss(
                            excess / 1000.0,
                            torch.zeros_like(excess),
                            reduction="mean",
                            beta=0.004,
                        )
                        if torch.isfinite(contour_loss):
                            total_loss = total_loss + self.lambda_contour * contour_loss

            total_loss = total_loss + mica_centered_shape_regularization(
                shape_param,
                identity_anchor,
                anchor_weight=self.identity_anchor_weight,
                mean_shape_weight=self.mean_shape_prior_weight,
            )
            if shared_exp_param is not None:
                total_loss = total_loss + self.lambda_exp * (shared_exp_param ** 2).mean()
            else:
                for name in view_names:
                    total_loss = total_loss + self.lambda_exp * (exp_params[name] ** 2).mean()

            total_loss.backward()
            # 梯度裁剪防止 NaN
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            return total_loss

        logger.info("开始 L-BFGS 联合优化...")
        best_loss = float("inf")
        for step in range(10):
            loss_val = optimizer.step(closure)
            loss_f = loss_val.item() if loss_val is not None else float("nan")
            if not np.isfinite(loss_f):
                logger.warning(f"  step {step+1}: loss=NaN，提前停止")
                break
            logger.info(f"  step {step+1}/10, loss={loss_f:.6f} ({loss_f*1e6:.1f}px²/pt)")
            if abs(loss_f - best_loss) < 1e-8 * best_loss:
                logger.info("  loss 收敛，提前停止")
                break
            best_loss = min(best_loss, loss_f)

        # 提取结果
        shape_np = shape_param.detach().cpu().numpy()
        per_view = {}
        for name in view_names:
            R_final_t, t_final_t = _pose_for_view(name)
            R_final = R_final_t.detach().cpu().numpy()
            t_final = t_final_t.detach().cpu().numpy()
            e_final = exp_params[name].detach().cpu().numpy()
            per_view[name] = {"R": R_final, "t": t_final, "exp": e_final}

        return shape_np, per_view


def _logsumexp_np(a: np.ndarray, axis: int = 1) -> np.ndarray:
    max_a = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(max_a, axis=axis) + np.log(np.sum(np.exp(a - max_a), axis=axis))


def _dense_side_envelope_np(
    proj_np: np.ndarray,
    candidate_idx: np.ndarray,
    rows: np.ndarray,
    row_sigma: float,
    tau: float,
    side: str,
) -> np.ndarray:
    pts = proj_np[candidate_idx]
    x = pts[:, 0][None, :]
    y = pts[:, 1][None, :]
    row_y = rows.astype(np.float32)[:, None]
    logits = -0.5 * ((y - row_y) / max(float(row_sigma), 1e-3)) ** 2
    logits = np.clip(logits, -60.0, 0.0)
    if side == "left":
        return -float(tau) * _logsumexp_np(logits - x / float(tau), axis=1)
    return float(tau) * _logsumexp_np(logits + x / float(tau), axis=1)


def _dense_contour_metric_np(proj_np: np.ndarray, dense: dict) -> Tuple[float, np.ndarray, np.ndarray]:
    rows = dense["rows_np"]
    row_weight = dense["row_weight_np"]
    target_left = dense["target_left_np"]
    target_right = dense["target_right_np"]
    row_sigma = float(dense["row_sigma"])
    tau = float(dense["tau"])
    left_pred = _dense_side_envelope_np(
        proj_np, dense["left_idx_np"], rows, row_sigma, tau, side="left"
    )
    right_pred = _dense_side_envelope_np(
        proj_np, dense["right_idx_np"], rows, row_sigma, tau, side="right"
    )
    err = (np.abs(left_pred - target_left) + np.abs(right_pred - target_right)) * 0.5
    metric = float((err * row_weight).sum() / np.clip(row_weight.sum(), 1e-6, None))
    return metric, left_pred, right_pred


def _dense_contour_metric_np_for_sides(
    proj_np: np.ndarray,
    dense: dict,
    sides: Tuple[str, ...] = ("left", "right"),
) -> Tuple[float, np.ndarray, np.ndarray]:
    rows = dense["rows_np"]
    row_weight = dense["row_weight_np"]
    row_sigma = float(dense["row_sigma"])
    tau = float(dense["tau"])
    left_pred = _dense_side_envelope_np(
        proj_np, dense["left_idx_np"], rows, row_sigma, tau, side="left"
    )
    right_pred = _dense_side_envelope_np(
        proj_np, dense["right_idx_np"], rows, row_sigma, tau, side="right"
    )
    errs = []
    if "left" in sides:
        errs.append(np.abs(left_pred - dense["target_left_np"]))
    if "right" in sides:
        errs.append(np.abs(right_pred - dense["target_right_np"]))
    if not errs:
        errs.append((np.abs(left_pred - dense["target_left_np"]) + np.abs(right_pred - dense["target_right_np"])) * 0.5)
    err = np.mean(np.stack(errs, axis=0), axis=0)
    metric = float((err * row_weight).sum() / np.clip(row_weight.sum(), 1e-6, None))
    return metric, left_pred, right_pred


def _save_dense_contour_debug_image(
    image: np.ndarray,
    dense: dict,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    out_path: Path,
):
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rows = dense["rows_np"].astype(np.int32)
    left_target = dense["target_left_np"]
    right_target = dense["target_right_np"]
    for row, xl_t, xr_t, xl_p, xr_p in zip(rows, left_target, right_target, left_pred, right_pred):
        y = int(row)
        cv2.circle(img, (int(round(xl_t)), y), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(round(xr_t)), y), 2, (0, 255, 0), -1)
        cv2.circle(img, (int(round(xl_p)), y), 2, (0, 0, 255), -1)
        cv2.circle(img, (int(round(xr_p)), y), 2, (0, 0, 255), -1)
        cv2.line(img, (int(round(xl_t)), y), (int(round(xl_p)), y), (0, 255, 255), 1)
        cv2.line(img, (int(round(xr_t)), y), (int(round(xr_p)), y), (0, 255, 255), 1)
    cv2.imwrite(str(out_path), img)


def _build_shape_only_dense_contours(
    flame: FLAMEModel,
    shape_anchor: torch.Tensor,
    frozen: Dict[str, dict],
    view_names: list,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    device: str,
    row_step: int,
    row_sigma: float,
    tau: float,
    boundary_band_px: float,
    search_margin_px: float,
) -> Dict[str, dict]:
    dense = {}
    row_step = max(2, int(row_step))
    with torch.no_grad():
        for name in view_names:
            mask = None
            if name in preprocessed_views:
                mask = preprocessed_views[name].get("shape_mask")
                if mask is None:
                    mask = preprocessed_views[name].get("face_mask")
            if mask is None:
                mask = view_data[name].get("shape_mask")
            if mask is None:
                continue

            h, _w = mask.shape[:2]
            xmin, xmax, valid = _build_mask_row_bounds(mask)
            target_lmk = np.asarray(view_data[name]["lmk_2d"], dtype=np.float32)
            contour_y = target_lmk[LMK_CONTOUR_IDX, 1]
            y_min = int(max(0, np.percentile(contour_y, 12) - 8))
            y_max = int(min(h - 1, np.max(contour_y) + 14))
            rows = np.arange(y_min, y_max + 1, row_step, dtype=np.int32)
            rows = rows[valid[rows]]
            if rows.size < 8:
                continue

            verts0 = flame(shape_anchor, frozen[name]["exp"])
            K_t = torch.tensor(view_data[name]["K"], device=device, dtype=torch.float32)
            proj0 = project_vertices(verts0, K_t, frozen[name]["R"], frozen[name]["t"])
            proj0_np = proj0.detach().cpu().numpy()
            row_idx = np.round(proj0_np[:, 1]).astype(np.int32)
            row_idx = np.clip(row_idx, 0, h - 1)
            v_valid = valid[row_idx]
            left = xmin[row_idx]
            right = xmax[row_idx]
            center = (left + right) * 0.5
            in_y = (proj0_np[:, 1] >= y_min - boundary_band_px) & (proj0_np[:, 1] <= y_max + boundary_band_px)
            in_x = (proj0_np[:, 0] >= left - search_margin_px) & (proj0_np[:, 0] <= right + search_margin_px)
            base = v_valid & in_y & in_x
            left_candidates = base & (
                (np.abs(proj0_np[:, 0] - left) <= boundary_band_px) | (proj0_np[:, 0] <= center)
            )
            right_candidates = base & (
                (np.abs(proj0_np[:, 0] - right) <= boundary_band_px) | (proj0_np[:, 0] >= center)
            )
            left_idx = np.flatnonzero(left_candidates).astype(np.int64)
            right_idx = np.flatnonzero(right_candidates).astype(np.int64)
            if left_idx.size < 24 or right_idx.size < 24:
                continue

            y_norm = (rows.astype(np.float32) - float(rows.min())) / max(float(rows.max() - rows.min()), 1.0)
            row_weight = 0.75 + 0.65 * y_norm
            item = {
                "rows_np": rows.astype(np.float32),
                "target_left_np": xmin[rows].astype(np.float32),
                "target_right_np": xmax[rows].astype(np.float32),
                "row_weight_np": row_weight.astype(np.float32),
                "left_idx_np": left_idx,
                "right_idx_np": right_idx,
                "row_sigma": float(row_sigma),
                "tau": float(tau),
                "rows_t": torch.tensor(rows.astype(np.float32), device=device),
                "target_left_t": torch.tensor(xmin[rows].astype(np.float32), device=device),
                "target_right_t": torch.tensor(xmax[rows].astype(np.float32), device=device),
                "row_weight_t": torch.tensor(row_weight.astype(np.float32), device=device),
                "left_idx_t": torch.tensor(left_idx, dtype=torch.long, device=device),
                "right_idx_t": torch.tensor(right_idx, dtype=torch.long, device=device),
            }
            dense[name] = item
            logger.info(
                f"  [{name}] dense contour rows={rows.size}, "
                f"left_vertices={left_idx.size}, right_vertices={right_idx.size}"
            )
    return dense


def _dense_contour_loss_torch(proj: torch.Tensor, dense: dict) -> torch.Tensor:
    def side_loss(candidate_idx: torch.Tensor, target: torch.Tensor, side: str) -> torch.Tensor:
        pts = proj[candidate_idx]
        x = pts[:, 0].unsqueeze(0)
        y = pts[:, 1].unsqueeze(0)
        rows = dense["rows_t"].unsqueeze(1)
        sigma = max(float(dense["row_sigma"]), 1e-3)
        tau = max(float(dense["tau"]), 1e-3)
        logits = -0.5 * ((y - rows) / sigma) ** 2
        logits = torch.clamp(logits, min=-60.0, max=0.0)
        if side == "left":
            pred = -tau * torch.logsumexp(logits - x / tau, dim=1)
        else:
            pred = tau * torch.logsumexp(logits + x / tau, dim=1)
        err = (pred - target) / 1000.0
        per_row = F.smooth_l1_loss(
            err,
            torch.zeros_like(err),
            reduction="none",
            beta=0.006,
        )
        weight = dense["row_weight_t"]
        return (per_row * weight).sum() / weight.sum().clamp_min(1e-6)

    left = side_loss(dense["left_idx_t"], dense["target_left_t"], "left")
    right = side_loss(dense["right_idx_t"], dense["target_right_t"], "right")
    return 0.5 * (left + right)


def _project_vertices_np(vertices: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v_cam = (R @ vertices.T + t[:, None]).T
    z = np.clip(v_cam[:, 2], 1e-6, None)
    v_hom = (K @ v_cam.T).T
    proj = np.stack([v_hom[:, 0] / z, v_hom[:, 1] / z], axis=1)
    return proj, v_cam


def _build_free_face_contour_data(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    landmarks_2d: np.ndarray,
    row_step: int,
    row_sigma: float,
    tau: float,
    boundary_band_px: float,
    search_margin_px: float,
    visible_mask: Optional[np.ndarray] = None,
    y_min_override: Optional[int] = None,
    y_max_override: Optional[int] = None,
    row_weight_mode: str = "legacy",
) -> Optional[dict]:
    h, _w = mask.shape[:2]
    xmin, xmax, valid = _build_mask_row_bounds(mask)
    contour_y = np.asarray(landmarks_2d, dtype=np.float32)[LMK_CONTOUR_IDX, 1]
    if y_min_override is None:
        y_min = int(max(0, np.percentile(contour_y, 12) - 8))
    else:
        y_min = int(np.clip(y_min_override, 0, h - 1))
    if y_max_override is None:
        y_max = int(min(h - 1, np.max(contour_y) + 16))
    else:
        y_max = int(np.clip(y_max_override, 0, h - 1))
    if y_max <= y_min:
        return None
    rows = np.arange(y_min, y_max + 1, max(2, int(row_step)), dtype=np.int32)
    rows = rows[valid[rows]]
    if rows.size < 8:
        return None

    proj, _v_cam = _project_vertices_np(vertices, K, R, t)
    row_idx = np.round(proj[:, 1]).astype(np.int32)
    row_idx = np.clip(row_idx, 0, h - 1)
    v_valid = valid[row_idx]
    left = xmin[row_idx]
    right = xmax[row_idx]
    center = (left + right) * 0.5
    in_y = (proj[:, 1] >= y_min - boundary_band_px) & (proj[:, 1] <= y_max + boundary_band_px)
    in_x = (proj[:, 0] >= left - search_margin_px) & (proj[:, 0] <= right + search_margin_px)
    base = v_valid & in_y & in_x
    if visible_mask is not None:
        base &= np.asarray(visible_mask, dtype=bool)
    left_candidates = base & (
        (np.abs(proj[:, 0] - left) <= boundary_band_px) | (proj[:, 0] <= center)
    )
    right_candidates = base & (
        (np.abs(proj[:, 0] - right) <= boundary_band_px) | (proj[:, 0] >= center)
    )
    left_idx = np.flatnonzero(left_candidates).astype(np.int64)
    right_idx = np.flatnonzero(right_candidates).astype(np.int64)
    if left_idx.size < 32 or right_idx.size < 32:
        return None

    y_norm = (rows.astype(np.float32) - float(rows.min())) / max(float(rows.max() - rows.min()), 1.0)
    if row_weight_mode == "semantic_face_shape":
        row_weight = 0.42 + 0.95 * y_norm
        row_weight += np.where(y_norm > 0.68, 0.35 * (y_norm - 0.68) / 0.32, 0.0)
        row_weight = np.clip(row_weight, 0.38, 1.75)
    else:
        row_weight = 0.75 + 0.65 * y_norm
    return {
        "rows_np": rows.astype(np.float32),
        "target_left_np": xmin[rows].astype(np.float32),
        "target_right_np": xmax[rows].astype(np.float32),
        "row_weight_np": row_weight.astype(np.float32),
        "left_idx_np": left_idx,
        "right_idx_np": right_idx,
        "row_sigma": float(row_sigma),
        "tau": float(tau),
        "y_min": y_min,
        "y_max": y_max,
        "xmin": xmin,
        "xmax": xmax,
        "valid": valid,
        "row_weight_mode": row_weight_mode,
    }


def _build_semantic_face_shape_mask(
    base_mask: np.ndarray,
    landmarks_2d: np.ndarray,
    mediapipe_landmarks: Optional[np.ndarray] = None,
    dilate_px: float = 6.0,
) -> Tuple[np.ndarray, dict]:
    h, w = base_mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    source = "fallback_68_landmarks"
    y_min = None
    y_max = None

    mp = None if mediapipe_landmarks is None else np.asarray(mediapipe_landmarks, dtype=np.float32)
    if mp is not None and mp.ndim == 2 and mp.shape[0] > int(MP_FACE_OVAL_IDX.max()):
        pts = mp[MP_FACE_OVAL_IDX, :2]
        finite = np.isfinite(pts).all(axis=1)
        if finite.sum() >= 16:
            pts = pts[finite]
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
            source = "mediapipe_face_oval"
            face_h = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
            y_min = int(max(0, pts[:, 1].min() + 0.055 * face_h))
            y_max = int(min(h - 1, pts[:, 1].max() + 0.025 * face_h))

    if mask.max() == 0:
        lmk = np.asarray(landmarks_2d, dtype=np.float32)
        pts = lmk[np.concatenate([LMK_CONTOUR_IDX, LMK_BROW_IDX, LMK_NOSE_IDX])]
        finite = np.isfinite(pts).all(axis=1)
        if finite.sum() >= 8:
            pts = pts[finite]
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            face_h = max(float(pts[:, 1].max() - pts[:, 1].min()), 1.0)
            y_min = int(max(0, pts[:, 1].min() - 0.05 * face_h))
            y_max = int(min(h - 1, pts[:, 1].max() + 0.05 * face_h))

    k = max(1, int(round(float(dilate_px))))
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    base = (base_mask > 0).astype(np.uint8) * 255
    intersect = cv2.bitwise_and(mask, base) if base.max() > 0 and mask.max() > 0 else mask
    if int(np.count_nonzero(intersect)) >= max(128, int(0.35 * np.count_nonzero(mask))):
        mask = intersect

    if mask.max() == 0:
        mask = base
        source = "base_mask_fallback"
        ys = np.flatnonzero(mask.max(axis=1) > 0)
        if ys.size:
            y_min = int(ys.min())
            y_max = int(ys.max())

    return mask.astype(np.uint8, copy=False), {
        "semantic_mode": "face_shape",
        "semantic_source": source,
        "semantic_area_px": int(np.count_nonzero(mask)),
        "semantic_y_min": None if y_min is None else int(y_min),
        "semantic_y_max": None if y_max is None else int(y_max),
        "excluded_regions": ["ear", "hair_outer_boundary", "neck", "background"],
        "soft_regions": ["forehead"],
        "strong_regions": ["cheek", "jawline", "chin"],
    }


def _build_semantic_face_contour_data(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    landmarks_2d: np.ndarray,
    mediapipe_landmarks: Optional[np.ndarray],
    row_step: int,
    row_sigma: float,
    tau: float,
    boundary_band_px: float,
    search_margin_px: float,
    visible_mask: Optional[np.ndarray] = None,
    semantic_mask_dilate_px: float = 6.0,
) -> Optional[dict]:
    semantic_mask, meta = _build_semantic_face_shape_mask(
        mask,
        landmarks_2d,
        mediapipe_landmarks=mediapipe_landmarks,
        dilate_px=semantic_mask_dilate_px,
    )
    data = _build_free_face_contour_data(
        vertices=vertices,
        K=K,
        R=R,
        t=t,
        mask=semantic_mask,
        landmarks_2d=landmarks_2d,
        row_step=row_step,
        row_sigma=row_sigma,
        tau=tau,
        boundary_band_px=boundary_band_px,
        search_margin_px=search_margin_px,
        visible_mask=visible_mask,
        y_min_override=meta.get("semantic_y_min"),
        y_max_override=meta.get("semantic_y_max"),
        row_weight_mode="semantic_face_shape",
    )
    if data is None:
        return None
    data.update(meta)
    data["target_mask"] = semantic_mask
    data["metric_name"] = "semantic_face_contour_px"
    return data


def _stable_protect_vertices(
    proj: np.ndarray,
    landmarks_2d: np.ndarray,
    radius_px: float,
) -> np.ndarray:
    stable = np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
    pts = np.asarray(landmarks_2d, dtype=np.float32)[stable]
    if pts.size == 0:
        return np.zeros(len(proj), dtype=bool)
    protect = np.zeros(len(proj), dtype=bool)
    radius2 = float(radius_px) ** 2
    for start in range(0, len(pts), 8):
        chunk = pts[start:start + 8]
        d2 = ((proj[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        protect |= np.min(d2, axis=1) <= radius2
    return protect


def _vertices_near_points_2d(proj: np.ndarray, points: np.ndarray, radius_px: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(proj) == 0 or points.size == 0 or radius_px <= 0:
        return np.zeros(len(proj), dtype=bool)
    keep = np.zeros(len(proj), dtype=bool)
    radius2 = float(radius_px) ** 2
    for start in range(0, len(points), 12):
        chunk = points[start:start + 12]
        d2 = ((proj[:, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        keep |= np.min(d2, axis=1) <= radius2
    return keep


def _stable_anchor_vertices_from_views(
    vertices: np.ndarray,
    view_data: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    lmk_vertex_indices: Optional[np.ndarray] = None,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
    enabled: bool = True,
    nose_radius_px: float = 38.0,
    eye_radius_px: float = 30.0,
    inner_mouth_radius_px: float = 26.0,
) -> Tuple[np.ndarray, dict]:
    anchor_mask = np.zeros(len(vertices), dtype=bool)
    group_masks = {
        "nose_bridge": np.zeros(len(vertices), dtype=bool),
        "eye_centers": np.zeros(len(vertices), dtype=bool),
        "inner_mouth": np.zeros(len(vertices), dtype=bool),
    }
    report = {
        "enabled": bool(enabled),
        "anchor_vertices": 0,
        "anchor_ratio": 0.0,
        "source_views": [],
        "groups": [],
    }
    if not enabled or len(vertices) == 0:
        return anchor_mask, report

    landmark_proj_by_view = {}
    if lmk_vertex_indices is not None:
        try:
            lmk_3d = _landmark_points_3d(
                vertices,
                lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )
            for view_name, view_result in per_view_results.items():
                if view_name not in intrinsics:
                    continue
                proj_lmk, _ = _project_vertices_np(
                    lmk_3d,
                    intrinsics[view_name],
                    view_result["R"],
                    view_result["t"],
                )
                landmark_proj_by_view[view_name] = proj_lmk
        except Exception as exc:
            logger.warning("Stable anchor landmark projection failed: %s", exc)

    specs = (
        ("nose_bridge", LMK_NOSE_IDX, float(nose_radius_px)),
        ("eye_centers", LMK_EYE_IDX, float(eye_radius_px)),
        ("inner_mouth", LMK_INNER_MOUTH_IDX, float(inner_mouth_radius_px)),
    )
    if "front" in per_view_results and "front" in view_data and "front" in intrinsics:
        source_view_names = ["front"]
    else:
        source_view_names = [
            name for name in per_view_results
            if name in view_data and name in intrinsics
        ]

    for view_name in source_view_names:
        view_result = per_view_results[view_name]
        if view_name not in view_data or view_name not in intrinsics:
            continue
        target_lmk = np.asarray(view_data[view_name].get("lmk_2d"), dtype=np.float32)
        if target_lmk.ndim != 2 or target_lmk.shape[0] < 68:
            continue
        proj, _ = _project_vertices_np(
            vertices,
            intrinsics[view_name],
            view_result["R"],
            view_result["t"],
        )
        current_lmk = landmark_proj_by_view.get(view_name)
        for group_name, idx, radius in specs:
            pts = [target_lmk[idx]]
            if current_lmk is not None and current_lmk.shape[0] >= 68:
                pts.append(current_lmk[idx])
            group_mask = _vertices_near_points_2d(proj, np.concatenate(pts, axis=0), radius)
            group_masks[group_name] |= group_mask
            anchor_mask |= group_mask

    report.update({
        "anchor_vertices": int(anchor_mask.sum()),
        "anchor_ratio": round(float(anchor_mask.sum()) / float(len(anchor_mask)), 4) if len(anchor_mask) else 0.0,
        "source_views": source_view_names,
        "groups": [
            {
                "name": name,
                "vertices": int(mask.sum()),
            }
            for name, mask in group_masks.items()
        ],
    })
    return anchor_mask, report


def _mesh_offset_safety_report(
    offsets: np.ndarray,
    faces: np.ndarray,
    stable_anchor_mask: Optional[np.ndarray] = None,
) -> dict:
    offsets = np.asarray(offsets, dtype=np.float64)
    offset_norm = np.linalg.norm(offsets, axis=1) if len(offsets) else np.zeros(0, dtype=np.float64)
    moved = offset_norm > 1e-6
    if len(faces):
        f = np.asarray(faces, dtype=np.int64)
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        edge_jump = np.linalg.norm(offsets[edges[:, 0]] - offsets[edges[:, 1]], axis=1)
    else:
        edge_jump = np.zeros(0, dtype=np.float64)

    if stable_anchor_mask is not None and len(stable_anchor_mask) == len(offset_norm):
        anchor_offsets = offset_norm[np.asarray(stable_anchor_mask, dtype=bool)]
    else:
        anchor_offsets = np.zeros(0, dtype=np.float64)

    return {
        "moved_vertices": int(moved.sum()),
        "moved_ratio": round(float(moved.sum()) / float(len(offset_norm)), 4) if len(offset_norm) else 0.0,
        "mean_offset_m": round(float(offset_norm[moved].mean()) if np.any(moved) else 0.0, 6),
        "p95_offset_m": round(float(np.percentile(offset_norm[moved], 95)) if np.any(moved) else 0.0, 6),
        "max_offset_m": round(float(offset_norm.max()) if len(offset_norm) else 0.0, 6),
        "edge_jump_mean_m": round(float(edge_jump.mean()) if len(edge_jump) else 0.0, 6),
        "edge_jump_p95_m": round(float(np.percentile(edge_jump, 95)) if len(edge_jump) else 0.0, 6),
        "edge_jump_max_m": round(float(edge_jump.max()) if len(edge_jump) else 0.0, 6),
        "anchor_vertices": int(anchor_offsets.size),
        "anchor_moved_vertices": int(np.count_nonzero(anchor_offsets > 1e-7)),
        "anchor_mean_offset_m": round(float(anchor_offsets.mean()) if anchor_offsets.size else 0.0, 6),
        "anchor_max_offset_m": round(float(anchor_offsets.max()) if anchor_offsets.size else 0.0, 6),
    }


def _deform_safety_reject_reasons(
    safety: dict,
    max_moved_ratio: float,
    max_anchor_move_m: float,
    max_offset_jump_p95_m: float,
    max_offset_jump_m: float,
) -> list:
    reasons = []
    if float(safety.get("moved_ratio", 0.0)) > float(max_moved_ratio):
        reasons.append("moved-ratio")
    if float(safety.get("anchor_max_offset_m", 0.0)) > float(max_anchor_move_m):
        reasons.append("anchor-motion")
    if float(safety.get("edge_jump_p95_m", 0.0)) > float(max_offset_jump_p95_m):
        reasons.append("offset-jump-p95")
    if float(safety.get("edge_jump_max_m", 0.0)) > float(max_offset_jump_m):
        reasons.append("offset-jump-max")
    return reasons


def _multi_view_validation_report(
    view_records: list,
    before_key: str,
    after_key: str,
    improve_key: str,
    min_front_improve_px: float = 0.25,
    min_overall_improve_px: float = 0.25,
    max_side_worsen_px: float = 0.35,
    max_side_mean_worsen_px: float = 0.05,
    require_side_views: bool = True,
) -> dict:
    records = []
    before_vals = []
    after_vals = []
    front_improves = []
    side_improves = []

    for rec in view_records:
        if before_key not in rec or after_key not in rec:
            continue
        before = float(rec[before_key])
        after = float(rec[after_key])
        improve = float(rec.get(improve_key, before - after))
        item = {
            "view": rec.get("view", ""),
            "before_px": round(before, 3),
            "after_px": round(after, 3),
            "improve_px": round(improve, 3),
            "role": "front" if rec.get("view") == "front" else "side",
        }
        records.append(item)
        before_vals.append(before)
        after_vals.append(after)
        if item["role"] == "front":
            front_improves.append(improve)
        else:
            side_improves.append(improve)

    before_mean = float(np.mean(before_vals)) if before_vals else 0.0
    after_mean = float(np.mean(after_vals)) if after_vals else 0.0
    overall_improve = before_mean - after_mean
    front_improve = max(front_improves) if front_improves else overall_improve
    max_side_worsen = max((max(0.0, -v) for v in side_improves), default=0.0)
    side_mean_improve = float(np.mean(side_improves)) if side_improves else 0.0

    reject_reasons = []
    if not front_improves:
        reject_reasons.append("missing-front-view")
    if require_side_views and not side_improves:
        reject_reasons.append("missing-side-views")
    if front_improve < float(min_front_improve_px):
        reject_reasons.append("front-improve-too-small")
    if overall_improve < float(min_overall_improve_px):
        reject_reasons.append("overall-improve-too-small")
    if max_side_worsen > float(max_side_worsen_px):
        reject_reasons.append("side-view-worsened")
    if side_improves and side_mean_improve < -float(max_side_mean_worsen_px):
        reject_reasons.append("side-mean-worsened")

    return {
        "enabled": True,
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "records": records,
        "front_improve_px": round(float(front_improve), 3),
        "overall_before_px": round(float(before_mean), 3),
        "overall_after_px": round(float(after_mean), 3),
        "overall_improve_px": round(float(overall_improve), 3),
        "side_mean_improve_px": round(float(side_mean_improve), 3),
        "max_side_worsen_px": round(float(max_side_worsen), 3),
        "thresholds": {
            "min_front_improve_px": round(float(min_front_improve_px), 3),
            "min_overall_improve_px": round(float(min_overall_improve_px), 3),
            "max_side_worsen_px": round(float(max_side_worsen_px), 3),
            "max_side_mean_worsen_px": round(float(max_side_mean_worsen_px), 3),
            "require_side_views": bool(require_side_views),
        },
    }


def _smooth_vertex_offsets(
    offsets: np.ndarray,
    faces: np.ndarray,
    editable: np.ndarray,
    constraints: np.ndarray,
    constraint_offsets: np.ndarray,
    max_offset_m: float,
    iterations: int,
    alpha: float,
    constraint_keep: float,
) -> np.ndarray:
    f = faces.astype(np.int64, copy=False)
    a = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    counts = np.zeros(len(offsets), dtype=np.float64)
    np.add.at(counts, src, 1.0)
    counts = np.clip(counts, 1.0, None)

    editable = editable.astype(bool)
    constraints = constraints.astype(bool)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    constraint_keep = float(np.clip(constraint_keep, 0.0, 1.0))

    smoothed = offsets.copy()
    for _ in range(max(0, int(iterations))):
        accum = np.zeros_like(smoothed)
        np.add.at(accum, src, smoothed[dst])
        avg = accum / counts[:, None]
        nxt = smoothed.copy()
        nxt[editable] = (1.0 - alpha) * smoothed[editable] + alpha * avg[editable]
        nxt[~editable] = 0.0
        nxt[constraints] = (
            (1.0 - constraint_keep) * nxt[constraints] +
            constraint_keep * constraint_offsets[constraints]
        )
        norm = np.linalg.norm(nxt, axis=1)
        too_far = norm > max_offset_m
        if np.any(too_far):
            nxt[too_far] *= (max_offset_m / np.clip(norm[too_far], 1e-8, None))[:, None]
        smoothed = nxt
    return smoothed


def _free_identity_landmark_profile(idx: int, base_radius_px: float, contour_radius_px: float, mouth_radius_px: float) -> Tuple[float, float]:
    if idx in set(LMK_CONTOUR_IDX.tolist()):
        return 1.0, float(contour_radius_px)
    if idx in set(LMK_MOUTH_IDX.tolist()):
        return 0.72, float(mouth_radius_px)
    if idx in set(LMK_BROW_IDX.tolist()):
        return 0.28, float(base_radius_px * 0.85)
    if idx in set(LMK_NOSE_IDX.tolist()):
        return 0.16, float(base_radius_px * 0.75)
    if idx in set(LMK_EYE_IDX.tolist()):
        return 0.18, float(base_radius_px * 0.72)
    return 0.35, float(base_radius_px)


def _free_identity_metric_report(
    vertices: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    stable_vals = []
    mean_vals = []
    contour_vals = []
    mouth_vals = []
    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        mean_err, max_err, errors = _save_landmark_reprojection_debug(
            vertices=vertices,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            image=preprocessed_views[view_name]["image"],
            target_landmarks=view_data[view_name]["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=out_dir / f"{view_name}_{prefix}_landmarks.png",
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
            return_errors=True,
        )
        stable = _landmark_subset_stats(errors, np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX]))
        contour = _landmark_subset_stats(errors, LMK_CONTOUR_IDX)
        mouth = _landmark_subset_stats(errors, LMK_MOUTH_IDX)
        record = {
            "view": view_name,
            "mean_px": round(float(mean_err), 3),
            "max_px": round(float(max_err), 3),
            "stable_mean_px": stable["mean_px"],
            "contour_mean_px": contour["mean_px"],
            "mouth_mean_px": mouth["mean_px"],
        }
        records.append(record)
        mean_vals.append(float(mean_err))
        stable_vals.append(float(stable["mean_px"]))
        contour_vals.append(float(contour["mean_px"]))
        mouth_vals.append(float(mouth["mean_px"]))

    return {
        "records": records,
        "mean_px": round(float(np.mean(mean_vals)) if mean_vals else 0.0, 3),
        "stable_mean_px": round(float(np.mean(stable_vals)) if stable_vals else 0.0, 3),
        "contour_mean_px": round(float(np.mean(contour_vals)) if contour_vals else 0.0, 3),
        "mouth_mean_px": round(float(np.mean(mouth_vals)) if mouth_vals else 0.0, 3),
    }


def _save_nose_mouth_overlay(
    image: np.ndarray,
    target_landmarks: np.ndarray,
    projected_landmarks: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_groups = (
        (LMK_NOSE_BRIDGE_IDX, (255, 180, 0)),
        (LMK_NOSE_BASE_IDX, (0, 160, 255)),
        (LMK_OUTER_MOUTH_IDX, (255, 0, 180)),
        (LMK_INNER_MOUTH_IDX, (180, 0, 255)),
    )
    for idxs, color in draw_groups:
        for idx in idxs:
            gt = tuple(np.round(target_landmarks[idx]).astype(int))
            pred = tuple(np.round(projected_landmarks[idx]).astype(int))
            cv2.circle(img, gt, 3, (0, 255, 0), -1)
            cv2.circle(img, pred, 3, (0, 0, 255), -1)
            cv2.line(img, gt, pred, color, 2, cv2.LINE_AA)
            cv2.putText(
                img,
                str(int(idx)),
                (pred[0] + 3, pred[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

    pts = np.vstack([
        target_landmarks[LMK_NOSE_MOUTH_TARGET_IDX],
        projected_landmarks[LMK_NOSE_MOUTH_TARGET_IDX],
    ])
    x0, y0 = np.floor(pts.min(axis=0) - 70).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0) + 70).astype(int)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), 2)
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_crop.jpg")), img[y0:y1, x0:x1])
    cv2.imwrite(str(out_path), img)


def _nose_mouth_metric_report(
    vertices: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
    target_idx: Optional[np.ndarray] = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    active_target_idx = LMK_NOSE_MOUTH_TARGET_IDX if target_idx is None else np.asarray(target_idx, dtype=np.int64)
    records = []
    target_vals = []
    front_target_vals = []
    protected_vals = []
    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        target_lmk = np.asarray(view_data[view_name]["lmk_2d"], dtype=np.float64)
        lmk_proj, errors = _landmark_reprojection_details(
            vertices=vertices,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
        )
        _save_nose_mouth_overlay(
            image=preprocessed_views[view_name]["image"],
            target_landmarks=target_lmk,
            projected_landmarks=lmk_proj,
            out_path=out_dir / f"{view_name}_{prefix}_nose_mouth.png",
        )
        nose_base = _landmark_subset_stats(errors, LMK_NOSE_BASE_IDX)
        outer_mouth = _landmark_subset_stats(errors, LMK_OUTER_MOUTH_IDX)
        inner_mouth = _landmark_subset_stats(errors, LMK_INNER_MOUTH_IDX)
        target = _landmark_subset_stats(errors, active_target_idx)
        protected = _landmark_subset_stats(errors, LMK_NOSE_MOUTH_PROTECT_IDX)
        record = {
            "view": view_name,
            "nose_base_mean_px": nose_base["mean_px"],
            "outer_mouth_mean_px": outer_mouth["mean_px"],
            "inner_mouth_mean_px": inner_mouth["mean_px"],
            "target_mean_px": target["mean_px"],
            "protected_mean_px": protected["mean_px"],
        }
        if view_name == "front":
            nose_width_target = float(np.linalg.norm(target_lmk[31] - target_lmk[35]))
            nose_width_model = float(np.linalg.norm(lmk_proj[31] - lmk_proj[35]))
            mouth_width_target = float(np.linalg.norm(target_lmk[48] - target_lmk[54]))
            mouth_width_model = float(np.linalg.norm(lmk_proj[48] - lmk_proj[54]))
            record.update({
                "nose_width_target_px": round(nose_width_target, 3),
                "nose_width_model_px": round(nose_width_model, 3),
                "nose_width_delta_px": round(nose_width_model - nose_width_target, 3),
                "mouth_width_target_px": round(mouth_width_target, 3),
                "mouth_width_model_px": round(mouth_width_model, 3),
                "mouth_width_delta_px": round(mouth_width_model - mouth_width_target, 3),
            })
        records.append(record)
        target_vals.append(float(target["mean_px"]))
        protected_vals.append(float(protected["mean_px"]))
        if view_name == "front":
            front_target_vals.append(float(target["mean_px"]))

    return {
        "records": records,
        "target_landmarks": [int(x) for x in active_target_idx],
        "target_mean_px": round(float(np.mean(target_vals)) if target_vals else 0.0, 3),
        "front_target_mean_px": round(float(np.mean(front_target_vals)) if front_target_vals else 0.0, 3),
        "protected_mean_px": round(float(np.mean(protected_vals)) if protected_vals else 0.0, 3),
    }


def _local_profile_contour_mean(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    row_step: int = 6,
    row_sigma: float = 8.0,
    tau: float = 10.0,
    boundary_band_px: float = 95.0,
    search_margin_px: float = 80.0,
) -> dict:
    values = []
    records = []
    normals = compute_vertex_normals(vertices, faces)
    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        mask = preprocessed_views[view_name].get("shape_mask")
        if mask is None:
            mask = preprocessed_views[view_name].get("face_mask")
        if mask is None:
            continue
        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        lmk = view_data[view_name]["lmk_2d"]
        mp_lmk = preprocessed_views[view_name].get("landmarks")
        visible_mask = None
        if len(normals) == len(vertices):
            view_dir_world = -R[2, :]
            visible_mask = (normals @ view_dir_world) > 0.03
        contour_data = _build_semantic_face_contour_data(
            vertices=vertices,
            K=K,
            R=R,
            t=t,
            mask=mask,
            landmarks_2d=lmk,
            mediapipe_landmarks=mp_lmk,
            row_step=row_step,
            row_sigma=row_sigma,
            tau=tau,
            boundary_band_px=boundary_band_px,
            search_margin_px=search_margin_px,
            visible_mask=visible_mask,
            semantic_mask_dilate_px=6.0,
        )
        if contour_data is None:
            continue
        target_sides, _profile_side = _personal_residual_target_sides(
            view_name,
            use_profile_side_contour=True,
        )
        proj, _ = _project_vertices_np(vertices, K, R, t)
        metric, _left, _right = _dense_contour_metric_np_for_sides(proj, contour_data, target_sides)
        records.append({"view": view_name, "mean_px": round(float(metric), 3)})
        values.append(float(metric))
    return {
        "records": records,
        "mean_px": round(float(np.mean(values)) if values else 0.0, 3),
    }


def _write_nose_mouth_local_index(out_dir: Path, report: dict) -> None:
    rows = []
    for phase_key, label in (("before", "before"), ("after", "after")):
        metrics = report.get(f"{phase_key}_metrics", {})
        for rec in metrics.get("records", []):
            rows.append(
                f"<tr><td>{label}</td><td>{rec.get('view')}</td>"
                f"<td>{rec.get('target_mean_px')}</td><td>{rec.get('nose_base_mean_px')}</td>"
                f"<td>{rec.get('outer_mouth_mean_px')}</td><td>{rec.get('protected_mean_px')}</td></tr>"
            )
    cards = []
    for view_name in ("left", "front", "right"):
        cards.append(
            f"""
            <section class="card">
              <h2>{view_name}</h2>
              <img src="{view_name}_before_nose_mouth.png" />
              <img src="{view_name}_after_nose_mouth.png" />
              <img src="{view_name}_residual_heatmap.png" />
            </section>
            """
        )
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>Nose mouth local residual audit</title>
<style>
body {{ margin:0; padding:28px; background:#101826; color:#e8f1ff; font-family:Arial,"Microsoft YaHei",sans-serif; }}
.status {{ padding:16px 18px; border-radius:14px; background:{'#193d2a' if report.get('accepted') else '#432323'}; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
.card {{ background:#172235; border:1px solid #26364d; border-radius:14px; padding:14px; }}
img {{ max-width:100%; background:#000; border-radius:10px; margin:8px 0; }}
table {{ border-collapse:collapse; width:100%; margin:18px 0; background:#111b2a; }}
td,th {{ border:1px solid #2b3d55; padding:8px 10px; }}
th {{ background:#20304a; }}
.muted {{ color:#9fb0c8; }}
</style></head><body>
<h1>Nose mouth local residual audit</h1>
<div class="status">
  <h2>{'Accepted' if report.get('accepted') else 'Rejected'}</h2>
  <p>{report.get('reason', '')}</p>
</div>
<p class="muted">Green points are targets, red points are current model projections. The local pass only targets nose base and lips while protecting eyes and nose bridge.</p>
<table>
<tr><th>phase</th><th>view</th><th>target mean</th><th>nose base</th><th>outer mouth</th><th>protected</th></tr>
{''.join(rows)}
</table>
<pre>{json.dumps({k: report.get(k) for k in ['target_improve_px','front_target_improve_px','protected_worsen_px','profile_worsen_px','nose_width_abs_worsen_px','safety','reject_reasons']}, ensure_ascii=False, indent=2)}</pre>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def _nose_mouth_local_residual_deform_mesh(
    verts_base: np.ndarray,
    verts_displaced: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    debug_dir: Path,
    lmk_vertex_indices: Optional[np.ndarray],
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    enabled: bool = True,
    max_offset_m: float = 0.010,
    vertex_radius_px: float = 34.0,
    protect_radius_px: float = 30.0,
    max_step_px: float = 18.0,
    front_weight: float = 1.0,
    side_weight: float = 0.35,
    nose_base_weight: float = 1.0,
    outer_mouth_weight: float = 1.0,
    inner_mouth_weight: float = 0.45,
    guard_nose_width: bool = True,
    smooth_iter: int = 18,
    smooth_alpha: float = 0.24,
    constraint_keep: float = 0.70,
    min_improve_px: float = 1.0,
    min_front_improve_px: float = 1.5,
    max_protected_worsen_px: float = 0.35,
    max_profile_worsen_px: float = 1.0,
    max_nose_width_abs_worsen_px: float = 2.0,
    max_moved_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    out_dir = debug_dir / "nose_mouth_local_residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "accepted": False,
        "protected_landmarks": [int(x) for x in LMK_NOSE_MOUTH_PROTECT_IDX],
    }
    if not enabled or lmk_vertex_indices is None or len(verts_displaced) == 0:
        report["reason"] = "disabled or missing landmark mapping"
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_nose_mouth_local_index(out_dir, report)
        return verts_base, verts_displaced, report

    active_parts = []
    inactive_parts = []
    if float(nose_base_weight) > 1e-8:
        active_parts.append(LMK_NOSE_BASE_IDX)
    else:
        inactive_parts.append(LMK_NOSE_BASE_IDX)
    if float(outer_mouth_weight) > 1e-8:
        active_parts.append(LMK_OUTER_MOUTH_IDX)
    else:
        inactive_parts.append(LMK_OUTER_MOUTH_IDX)
    if float(inner_mouth_weight) > 1e-8:
        active_parts.append(LMK_INNER_MOUTH_IDX)
    else:
        inactive_parts.append(LMK_INNER_MOUTH_IDX)
    if not active_parts:
        report["reason"] = "no active nose/mouth target groups"
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_nose_mouth_local_index(out_dir, report)
        return verts_base, verts_displaced, report
    active_target_idx = np.concatenate(active_parts).astype(np.int64)
    inactive_target_idx = np.concatenate(inactive_parts).astype(np.int64) if inactive_parts else np.zeros(0, dtype=np.int64)
    protected_landmarks = np.concatenate([LMK_NOSE_MOUTH_PROTECT_IDX, inactive_target_idx]).astype(np.int64)
    report["target_landmarks"] = [int(x) for x in active_target_idx]
    report["protected_landmarks"] = [int(x) for x in protected_landmarks]

    before_metrics = _nose_mouth_metric_report(
        verts_displaced,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "before",
        target_idx=active_target_idx,
    )
    before_profile = _local_profile_contour_mean(
        verts_displaced,
        faces,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
    )

    n = len(verts_displaced)
    offsets_accum = np.zeros((n, 3), dtype=np.float64)
    weights_accum = np.zeros(n, dtype=np.float64)
    editable = np.zeros(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    normals = compute_vertex_normals(verts_displaced, faces)

    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        target_lmk = np.asarray(view_data[view_name]["lmk_2d"], dtype=np.float64)
        proj, v_cam = _project_vertices_np(verts_displaced, K, R, t)
        lmk_proj, _errors = _landmark_reprojection_details(
            vertices=verts_displaced,
            K=K,
            R=R,
            t=t,
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
        )
        view_dir_world = -R[2, :]
        visible = (normals @ view_dir_world) > 0.03
        visible &= v_cam[:, 2] > 1e-5

        protect_pts = np.concatenate([
            target_lmk[protected_landmarks],
            lmk_proj[protected_landmarks],
        ], axis=0)
        view_protected = _vertices_near_points_2d(proj, protect_pts, float(protect_radius_px))
        protected |= view_protected

        view_weight = float(front_weight if view_name == "front" else side_weight)
        if view_weight <= 1e-8:
            continue
        camera_x_world = R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        camera_y_world = R.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fx = max(abs(float(K[0, 0])), 1e-6)
        fy = max(abs(float(K[1, 1])), 1e-6)
        front_nose_width_state = None
        if bool(guard_nose_width) and view_name == "front":
            target_width = float(np.linalg.norm(target_lmk[31] - target_lmk[35]))
            model_width = float(np.linalg.norm(lmk_proj[31] - lmk_proj[35]))
            center_x = float(lmk_proj[LMK_NOSE_BASE_IDX, 0].mean())
            if model_width > target_width + 1e-6:
                front_nose_width_state = ("too_wide", center_x)
            elif model_width < target_width - 1e-6:
                front_nose_width_state = ("too_narrow", center_x)
        for idx in active_target_idx:
            delta = target_lmk[idx] - lmk_proj[idx]
            if idx in set(LMK_NOSE_BASE_IDX.tolist()) and front_nose_width_state is not None:
                state, center_x = front_nose_width_state
                side_sign = np.sign(float(lmk_proj[idx, 0]) - center_x)
                horizontal_direction = side_sign * float(delta[0])
                if state == "too_wide" and horizontal_direction > 0:
                    delta[0] = 0.0
                elif state == "too_narrow" and horizontal_direction < 0:
                    delta[0] = 0.0
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm < 1e-4:
                continue
            if delta_norm > float(max_step_px):
                delta = delta * (float(max_step_px) / delta_norm)
                delta_norm = float(max_step_px)
            pts = np.stack([target_lmk[idx], lmk_proj[idx]], axis=0)
            d2 = ((proj - pts[0]) ** 2).sum(axis=1)
            d2 = np.minimum(d2, ((proj - pts[1]) ** 2).sum(axis=1))
            near = (d2 <= float(vertex_radius_px) ** 2) & visible & ~view_protected
            if not np.any(near):
                continue
            if idx in set(LMK_NOSE_BASE_IDX.tolist()):
                local_weight = float(nose_base_weight)
            elif idx in set(LMK_OUTER_MOUTH_IDX.tolist()):
                local_weight = float(outer_mouth_weight)
            else:
                local_weight = float(inner_mouth_weight)
            conf = np.exp(-0.5 * d2[near] / max(float(vertex_radius_px) ** 2, 1e-6))
            conf *= view_weight * local_weight
            vidx = np.flatnonzero(near)
            dx_cam = float(delta[0]) * np.clip(v_cam[vidx, 2], 1e-6, None) / fx
            dy_cam = float(delta[1]) * np.clip(v_cam[vidx, 2], 1e-6, None) / fy
            local_offsets = dx_cam[:, None] * camera_x_world[None, :] + dy_cam[:, None] * camera_y_world[None, :]
            np.add.at(offsets_accum, vidx, local_offsets * conf[:, None])
            np.add.at(weights_accum, vidx, conf)
            editable[vidx] = True

    constraints = (weights_accum > 1e-6) & editable & ~protected
    if not np.any(constraints):
        report.update({
            "reason": "no usable nose/mouth local constraints",
            "before_metrics": before_metrics,
            "before_profile": before_profile,
        })
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_nose_mouth_local_index(out_dir, report)
        return verts_base, verts_displaced, report

    constraint_offsets = np.zeros_like(offsets_accum)
    constraint_offsets[constraints] = offsets_accum[constraints] / weights_accum[constraints, None]
    norm = np.linalg.norm(constraint_offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        constraint_offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]

    offsets = _smooth_vertex_offsets(
        offsets=constraint_offsets.copy(),
        faces=faces,
        editable=editable & ~protected,
        constraints=constraints,
        constraint_offsets=constraint_offsets,
        max_offset_m=float(max_offset_m),
        iterations=int(smooth_iter),
        alpha=float(smooth_alpha),
        constraint_keep=float(constraint_keep),
    )
    width_guard_removed_vertices = 0
    width_guard_removed_mean_m = 0.0
    if bool(guard_nose_width) and "front" in per_view_results and "front" in view_data and "front" in intrinsics:
        try:
            front_result = per_view_results["front"]
            front_K = intrinsics["front"]
            front_R = front_result["R"]
            front_t = front_result["t"]
            front_target_lmk = np.asarray(view_data["front"]["lmk_2d"], dtype=np.float64)
            front_lmk_proj, _front_errors = _landmark_reprojection_details(
                vertices=verts_displaced,
                K=front_K,
                R=front_R,
                t=front_t,
                target_landmarks=front_target_lmk,
                lmk_vertex_indices=lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )
            target_width = float(np.linalg.norm(front_target_lmk[31] - front_target_lmk[35]))
            model_width = float(np.linalg.norm(front_lmk_proj[31] - front_lmk_proj[35]))
            width_state = None
            if model_width > target_width + 1e-6:
                width_state = "too_wide"
            elif model_width < target_width - 1e-6:
                width_state = "too_narrow"
            if width_state is not None:
                proj_before, _ = _project_vertices_np(verts_displaced, front_K, front_R, front_t)
                center_x = float(front_lmk_proj[LMK_NOSE_BASE_IDX, 0].mean())
                side_sign = np.sign(proj_before[:, 0] - center_x)
                camera_x_world = front_R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_component = offsets @ camera_x_world
                outward = side_sign * x_component
                if width_state == "too_wide":
                    bad = (outward > 0.0) & editable & ~protected
                else:
                    bad = (outward < 0.0) & editable & ~protected
                if np.any(bad):
                    removed = x_component[bad, None] * camera_x_world[None, :]
                    offsets[bad] -= removed
                    width_guard_removed_vertices = int(np.count_nonzero(bad))
                    width_guard_removed_mean_m = float(np.linalg.norm(removed, axis=1).mean())
        except Exception as exc:
            logger.warning("Nose width guard failed: %s", exc)

    norm = np.linalg.norm(offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]
    verts_base_out = verts_base + offsets.astype(verts_base.dtype, copy=False)
    verts_disp_out = verts_displaced + offsets.astype(verts_displaced.dtype, copy=False)

    after_metrics = _nose_mouth_metric_report(
        verts_disp_out,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "after",
        target_idx=active_target_idx,
    )
    after_profile = _local_profile_contour_mean(
        verts_disp_out,
        faces,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
    )
    for view_name, view_result in per_view_results.items():
        if view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        _save_residual_heatmap_projection(
            image=preprocessed_views[view_name]["image"],
            vertices=verts_disp_out,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            offsets=offsets,
            out_path=out_dir / f"{view_name}_residual_heatmap.png",
        )

    target_improve = float(before_metrics.get("target_mean_px", 0.0) - after_metrics.get("target_mean_px", 0.0))
    front_improve = float(before_metrics.get("front_target_mean_px", 0.0) - after_metrics.get("front_target_mean_px", 0.0))
    protected_worsen = float(after_metrics.get("protected_mean_px", 0.0) - before_metrics.get("protected_mean_px", 0.0))
    profile_worsen = float(after_profile.get("mean_px", 0.0) - before_profile.get("mean_px", 0.0))
    before_front = next((r for r in before_metrics.get("records", []) if r.get("view") == "front"), {})
    after_front = next((r for r in after_metrics.get("records", []) if r.get("view") == "front"), {})
    before_width_abs = abs(float(before_front.get("nose_width_delta_px", 0.0)))
    after_width_abs = abs(float(after_front.get("nose_width_delta_px", 0.0)))
    nose_width_abs_worsen = float(after_width_abs - before_width_abs)
    safety = _mesh_offset_safety_report(offsets, faces, protected)
    reject_reasons = []
    if target_improve < float(min_improve_px):
        reject_reasons.append("target-improve-too-small")
    if front_improve < float(min_front_improve_px):
        reject_reasons.append("front-improve-too-small")
    if protected_worsen > float(max_protected_worsen_px):
        reject_reasons.append("protected-worsened")
    if profile_worsen > float(max_profile_worsen_px):
        reject_reasons.append("profile-contour-worsened")
    if nose_width_abs_worsen > float(max_nose_width_abs_worsen_px):
        reject_reasons.append("nose-width-worsened")
    if float(safety.get("moved_ratio", 0.0)) > float(max_moved_ratio):
        reject_reasons.append("moved-ratio")
    if float(safety.get("max_offset_m", 0.0)) > float(max_offset_m) + 1e-8:
        reject_reasons.append("max-offset")
    accepted = not reject_reasons
    report.update({
        "applied": bool(accepted),
        "accepted": bool(accepted),
        "reason": "accepted nose/mouth local residual" if accepted else "rejected: " + ", ".join(reject_reasons),
        "reject_reasons": reject_reasons,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_profile": before_profile,
        "after_profile": after_profile,
        "target_improve_px": round(float(target_improve), 3),
        "front_target_improve_px": round(float(front_improve), 3),
        "protected_worsen_px": round(float(protected_worsen), 3),
        "profile_worsen_px": round(float(profile_worsen), 3),
        "nose_width_abs_worsen_px": round(float(nose_width_abs_worsen), 3),
        "constraint_vertices": int(constraints.sum()),
        "editable_vertices": int(editable.sum()),
        "width_guard_removed_vertices": int(width_guard_removed_vertices),
        "width_guard_removed_mean_m": round(float(width_guard_removed_mean_m), 6),
        "safety": safety,
        "thresholds": {
            "min_improve_px": round(float(min_improve_px), 3),
            "min_front_improve_px": round(float(min_front_improve_px), 3),
            "max_protected_worsen_px": round(float(max_protected_worsen_px), 3),
            "max_profile_worsen_px": round(float(max_profile_worsen_px), 3),
            "max_nose_width_abs_worsen_px": round(float(max_nose_width_abs_worsen_px), 3),
            "max_offset_m": round(float(max_offset_m), 6),
            "max_moved_ratio": round(float(max_moved_ratio), 4),
        },
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_nose_mouth_local_index(out_dir, report)
    logger.info(
        "Nose/mouth local residual: accepted=%s target %.2fpx front %.2fpx protected_worsen %.2fpx moved=%s",
        accepted,
        target_improve,
        front_improve,
        protected_worsen,
        safety.get("moved_vertices"),
    )
    if not accepted:
        return verts_base, verts_displaced, report
    return verts_base_out, verts_disp_out, report


def _nose_region_roi_mask(
    proj: np.ndarray,
    target_landmarks: np.ndarray,
    projected_landmarks: np.ndarray,
    margin_px: float,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    nose_pts = np.vstack([
        np.asarray(target_landmarks, dtype=np.float64)[LMK_NOSE_IDX],
        np.asarray(projected_landmarks, dtype=np.float64)[LMK_NOSE_IDX],
    ])
    finite = np.isfinite(nose_pts).all(axis=1)
    if finite.sum() < 4 or len(proj) == 0:
        return np.zeros(len(proj), dtype=bool), np.zeros((0, 2), dtype=np.int32)
    nose_pts = nose_pts[finite]
    x_min, y_min = nose_pts.min(axis=0)
    x_max, y_max = nose_pts.max(axis=0)
    width = max(float(x_max - x_min), 1.0)
    height = max(float(y_max - y_min), 1.0)
    margin = float(margin_px)
    x0 = x_min - max(0.50 * margin, 0.20 * width)
    x1 = x_max + max(0.50 * margin, 0.20 * width)
    y0 = y_min - max(0.35 * margin, 0.12 * height)
    y1 = y_max + max(0.55 * margin, 0.22 * height)
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    rx = max((x1 - x0) * 0.5, 1.0)
    ry = max((y1 - y0) * 0.5, 1.0)
    norm = ((proj[:, 0] - center[0]) / rx) ** 2 + ((proj[:, 1] - center[1]) / ry) ** 2
    in_roi = norm <= 1.0
    if image_shape is not None:
        h, w = image_shape
        in_roi &= (
            (proj[:, 0] >= 0.0)
            & (proj[:, 0] < float(w))
            & (proj[:, 1] >= 0.0)
            & (proj[:, 1] < float(h))
        )
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    poly = np.stack([center[0] + rx * np.cos(angles), center[1] + ry * np.sin(angles)], axis=1)
    return in_roi, np.round(poly).astype(np.int32)


def _save_nose_region_overlay(
    image: np.ndarray,
    target_landmarks: np.ndarray,
    projected_landmarks: np.ndarray,
    out_path: Path,
    roi_poly: Optional[np.ndarray] = None,
    selected_proj: Optional[np.ndarray] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if roi_poly is not None and len(roi_poly) >= 3:
        cv2.polylines(img, [roi_poly.astype(np.int32)], True, (255, 220, 80), 2, cv2.LINE_AA)
    if selected_proj is not None and len(selected_proj):
        pts = np.asarray(selected_proj, dtype=np.float64)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) > 5000:
            pts = pts[np.linspace(0, len(pts) - 1, 5000).astype(np.int64)]
        h, w = img.shape[:2]
        for p in pts:
            x, y = int(round(float(p[0]))), int(round(float(p[1])))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 1, (255, 120, 20), -1)
    for idx in LMK_NOSE_IDX:
        gt = tuple(np.round(target_landmarks[idx]).astype(int))
        pred = tuple(np.round(projected_landmarks[idx]).astype(int))
        color = (0, 180, 255) if idx in set(LMK_NOSE_BASE_IDX.tolist()) else (255, 180, 0)
        cv2.circle(img, gt, 4, (0, 255, 0), -1)
        cv2.circle(img, pred, 4, (0, 0, 255), -1)
        cv2.line(img, gt, pred, color, 2, cv2.LINE_AA)
        cv2.putText(img, str(int(idx)), (pred[0] + 4, pred[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    pts = np.vstack([target_landmarks[LMK_NOSE_IDX], projected_landmarks[LMK_NOSE_IDX]])
    if roi_poly is not None and len(roi_poly) >= 3:
        pts = np.vstack([pts, roi_poly.astype(np.float64)])
    x0, y0 = np.floor(pts.min(axis=0) - 60).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0) + 60).astype(int)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), 2)
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_crop.jpg")), img[y0:y1, x0:x1])
    cv2.imwrite(str(out_path), img)


def _nose_region_metric_report(
    vertices: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    out_dir: Path,
    prefix: str,
    selected_by_view: Optional[dict] = None,
    roi_margin_px: float = 28.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    nose_vals = []
    front_vals = []
    protected_vals = []
    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        target_lmk = np.asarray(view_data[view_name]["lmk_2d"], dtype=np.float64)
        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        lmk_proj, errors = _landmark_reprojection_details(
            vertices=vertices,
            K=K,
            R=R,
            t=t,
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
        )
        proj, _ = _project_vertices_np(vertices, K, R, t)
        _roi_mask, roi_poly = _nose_region_roi_mask(
            proj=proj,
            target_landmarks=target_lmk,
            projected_landmarks=lmk_proj,
            margin_px=roi_margin_px,
            image_shape=preprocessed_views[view_name]["image"].shape[:2],
        )
        selected_proj = None
        if selected_by_view is not None and view_name in selected_by_view:
            selected_idx = np.asarray(selected_by_view.get(view_name), dtype=np.int64)
            if selected_idx.size:
                selected_proj = proj[selected_idx]
        _save_nose_region_overlay(
            image=preprocessed_views[view_name]["image"],
            target_landmarks=target_lmk,
            projected_landmarks=lmk_proj,
            roi_poly=roi_poly,
            selected_proj=selected_proj,
            out_path=out_dir / f"{view_name}_{prefix}_nose_region.png",
        )
        nose = _landmark_subset_stats(errors, LMK_NOSE_IDX)
        bridge = _landmark_subset_stats(errors, LMK_NOSE_BRIDGE_IDX)
        base = _landmark_subset_stats(errors, LMK_NOSE_BASE_IDX)
        protected = _landmark_subset_stats(errors, LMK_NOSE_REGION_PROTECT_IDX)
        record = {
            "view": view_name,
            "nose_mean_px": nose["mean_px"],
            "nose_bridge_mean_px": bridge["mean_px"],
            "nose_base_mean_px": base["mean_px"],
            "protected_mean_px": protected["mean_px"],
        }
        if view_name == "front":
            target_width = float(np.linalg.norm(target_lmk[31] - target_lmk[35]))
            model_width = float(np.linalg.norm(lmk_proj[31] - lmk_proj[35]))
            record.update({
                "nose_width_target_px": round(target_width, 3),
                "nose_width_model_px": round(model_width, 3),
                "nose_width_delta_px": round(model_width - target_width, 3),
                "nose_center_shift_px": [
                    round(float(lmk_proj[LMK_NOSE_IDX, 0].mean() - target_lmk[LMK_NOSE_IDX, 0].mean()), 3),
                    round(float(lmk_proj[LMK_NOSE_IDX, 1].mean() - target_lmk[LMK_NOSE_IDX, 1].mean()), 3),
                ],
            })
        records.append(record)
        nose_vals.append(float(nose["mean_px"]))
        protected_vals.append(float(protected["mean_px"]))
        if view_name == "front":
            front_vals.append(float(nose["mean_px"]))
    return {
        "records": records,
        "target_landmarks": [int(x) for x in LMK_NOSE_IDX],
        "nose_mean_px": round(float(np.mean(nose_vals)) if nose_vals else 0.0, 3),
        "front_nose_mean_px": round(float(np.mean(front_vals)) if front_vals else 0.0, 3),
        "protected_mean_px": round(float(np.mean(protected_vals)) if protected_vals else 0.0, 3),
    }


def _write_nose_region_dense_index(out_dir: Path, report: dict) -> None:
    rows = []
    for phase_key, label in (("before", "before"), ("after", "after")):
        metrics = report.get(f"{phase_key}_metrics", {})
        for rec in metrics.get("records", []):
            rows.append(
                f"<tr><td>{label}</td><td>{rec.get('view')}</td>"
                f"<td>{rec.get('nose_mean_px')}</td><td>{rec.get('nose_bridge_mean_px')}</td>"
                f"<td>{rec.get('nose_base_mean_px')}</td><td>{rec.get('protected_mean_px')}</td>"
                f"<td>{rec.get('nose_width_delta_px', '')}</td></tr>"
            )
    cards = []
    for view_name in ("left", "front", "right"):
        cards.append(
            f"""
            <section class="card">
              <h2>{view_name}</h2>
              <img src="{view_name}_before_nose_region.png" />
              <img src="{view_name}_after_nose_region.png" />
              <img src="{view_name}_residual_heatmap.png" />
            </section>
            """
        )
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<title>Nose region dense residual audit</title>
<style>
body {{ margin:0; padding:28px; background:#101826; color:#e8f1ff; font-family:Arial,"Microsoft YaHei",sans-serif; }}
.status {{ padding:16px 18px; border-radius:14px; background:{'#193d2a' if report.get('accepted') else '#432323'}; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
.card {{ background:#172235; border:1px solid #26364d; border-radius:14px; padding:14px; }}
img {{ max-width:100%; background:#000; border-radius:10px; margin:8px 0; }}
table {{ border-collapse:collapse; width:100%; margin:18px 0; background:#111b2a; }}
td,th {{ border:1px solid #2b3d55; padding:8px 10px; }}
th {{ background:#20304a; }}
.muted {{ color:#9fb0c8; }}
</style></head><body>
<h1>Nose region dense residual audit</h1>
<div class="status">
  <h2>{'Accepted' if report.get('accepted') else 'Rejected'}</h2>
  <p>{report.get('reason', '')}</p>
</div>
<p class="muted">Green points are target landmarks, red points are model projections, yellow ellipse is the nose ROI, orange dots are editable dense nose-region vertices.</p>
<table>
<tr><th>phase</th><th>view</th><th>nose mean</th><th>bridge</th><th>base</th><th>protected</th><th>front width delta</th></tr>
{''.join(rows)}
</table>
<pre>{json.dumps({k: report.get(k) for k in ['nose_improve_px','front_nose_improve_px','protected_worsen_px','profile_worsen_px','nose_width_abs_worsen_px','constraint_vertices','editable_vertices','safety','reject_reasons']}, ensure_ascii=False, indent=2)}</pre>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def _nose_region_dense_residual_deform_mesh(
    verts_base: np.ndarray,
    verts_displaced: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    debug_dir: Path,
    lmk_vertex_indices: Optional[np.ndarray],
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    enabled: bool = True,
    max_offset_m: float = 0.008,
    vertex_radius_px: float = 34.0,
    roi_margin_px: float = 28.0,
    protect_radius_px: float = 34.0,
    max_step_px: float = 14.0,
    front_weight: float = 1.0,
    side_weight: float = 0.20,
    bridge_weight: float = 0.45,
    tip_weight: float = 0.80,
    wing_weight: float = 1.0,
    guard_nose_width: bool = True,
    smooth_iter: int = 14,
    smooth_alpha: float = 0.20,
    constraint_keep: float = 0.64,
    min_improve_px: float = 0.25,
    min_front_improve_px: float = 0.40,
    max_protected_worsen_px: float = 0.35,
    max_profile_worsen_px: float = 1.0,
    max_nose_width_abs_worsen_px: float = 1.0,
    max_moved_ratio: float = 0.045,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    out_dir = debug_dir / "nose_region_dense_residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "accepted": False,
        "target_landmarks": [int(x) for x in LMK_NOSE_IDX],
        "protected_landmarks": [int(x) for x in LMK_NOSE_REGION_PROTECT_IDX],
    }
    if not enabled or lmk_vertex_indices is None or len(verts_displaced) == 0:
        report["reason"] = "disabled or missing landmark mapping"
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_nose_region_dense_index(out_dir, report)
        return verts_base, verts_displaced, report

    before_metrics = _nose_region_metric_report(
        verts_displaced,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "before",
        roi_margin_px=roi_margin_px,
    )
    before_profile = _local_profile_contour_mean(
        verts_displaced,
        faces,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
    )

    n = len(verts_displaced)
    offsets_accum = np.zeros((n, 3), dtype=np.float64)
    weights_accum = np.zeros(n, dtype=np.float64)
    editable = np.zeros(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    selected_by_view = {}
    normals = compute_vertex_normals(verts_displaced, faces)
    front_roi_prior = None
    if "front" in per_view_results and "front" in view_data and "front" in preprocessed_views and "front" in intrinsics:
        try:
            front_result = per_view_results["front"]
            front_target_lmk = np.asarray(view_data["front"]["lmk_2d"], dtype=np.float64)
            front_lmk_proj, _ = _landmark_reprojection_details(
                vertices=verts_displaced,
                K=intrinsics["front"],
                R=front_result["R"],
                t=front_result["t"],
                target_landmarks=front_target_lmk,
                lmk_vertex_indices=lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )
            front_proj, front_v_cam = _project_vertices_np(
                verts_displaced,
                intrinsics["front"],
                front_result["R"],
                front_result["t"],
            )
            front_roi_prior, _ = _nose_region_roi_mask(
                proj=front_proj,
                target_landmarks=front_target_lmk,
                projected_landmarks=front_lmk_proj,
                margin_px=roi_margin_px,
                image_shape=preprocessed_views["front"]["image"].shape[:2],
            )
            front_roi_prior &= front_v_cam[:, 2] > 1e-5
        except Exception as exc:
            logger.warning("Nose region front ROI prior failed: %s", exc)
            front_roi_prior = None

    bridge_set = set(LMK_NOSE_BRIDGE_IDX.tolist())
    wing_set = {31, 35}
    tip_set = {30, 32, 33, 34}
    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        image = preprocessed_views[view_name]["image"]
        target_lmk = np.asarray(view_data[view_name]["lmk_2d"], dtype=np.float64)
        proj, v_cam = _project_vertices_np(verts_displaced, K, R, t)
        lmk_proj, _errors = _landmark_reprojection_details(
            vertices=verts_displaced,
            K=K,
            R=R,
            t=t,
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
        )
        roi_mask, _roi_poly = _nose_region_roi_mask(
            proj=proj,
            target_landmarks=target_lmk,
            projected_landmarks=lmk_proj,
            margin_px=roi_margin_px,
            image_shape=image.shape[:2],
        )
        view_dir_world = -R[2, :]
        visible = (normals @ view_dir_world) > 0.02
        visible &= v_cam[:, 2] > 1e-5
        protect_pts = np.concatenate([
            target_lmk[LMK_NOSE_REGION_PROTECT_IDX],
            lmk_proj[LMK_NOSE_REGION_PROTECT_IDX],
        ], axis=0)
        view_protected = _vertices_near_points_2d(proj, protect_pts, float(protect_radius_px))
        protected |= view_protected
        view_editable = roi_mask & visible & ~view_protected
        if front_roi_prior is not None and len(front_roi_prior) == len(view_editable):
            view_editable &= front_roi_prior
        selected_by_view[view_name] = np.flatnonzero(view_editable).astype(np.int64)

        view_weight = float(front_weight if view_name == "front" else side_weight)
        if view_weight <= 1e-8 or not np.any(view_editable):
            continue
        camera_x_world = R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        camera_y_world = R.T @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        fx = max(abs(float(K[0, 0])), 1e-6)
        fy = max(abs(float(K[1, 1])), 1e-6)
        width_state = None
        if bool(guard_nose_width) and view_name == "front":
            target_width = float(np.linalg.norm(target_lmk[31] - target_lmk[35]))
            model_width = float(np.linalg.norm(lmk_proj[31] - lmk_proj[35]))
            center_x = float(lmk_proj[LMK_NOSE_BASE_IDX, 0].mean())
            if model_width > target_width + 1e-6:
                width_state = ("too_wide", center_x)
            elif model_width < target_width - 1e-6:
                width_state = ("too_narrow", center_x)
        edit_idx = np.flatnonzero(view_editable)
        for idx in LMK_NOSE_IDX:
            delta = target_lmk[idx] - lmk_proj[idx]
            if idx in wing_set and width_state is not None:
                state, center_x = width_state
                side_sign = np.sign(float(lmk_proj[idx, 0]) - center_x)
                horizontal_direction = side_sign * float(delta[0])
                if state == "too_wide" and horizontal_direction > 0:
                    delta[0] = 0.0
                elif state == "too_narrow" and horizontal_direction < 0:
                    delta[0] = 0.0
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm < 1e-4:
                continue
            if delta_norm > float(max_step_px):
                delta = delta * (float(max_step_px) / delta_norm)
            if idx in wing_set:
                handle_weight = float(wing_weight)
            elif idx in tip_set:
                handle_weight = float(tip_weight)
            elif idx in bridge_set:
                handle_weight = float(bridge_weight)
            else:
                handle_weight = 0.65
            if handle_weight <= 1e-8:
                continue
            d2_target = ((proj[edit_idx] - target_lmk[idx]) ** 2).sum(axis=1)
            d2_model = ((proj[edit_idx] - lmk_proj[idx]) ** 2).sum(axis=1)
            d2 = np.minimum(d2_target, d2_model)
            near_local = d2 <= float(vertex_radius_px) ** 2
            if not np.any(near_local):
                continue
            vidx = edit_idx[near_local]
            conf = np.exp(-0.5 * d2[near_local] / max(float(vertex_radius_px) ** 2, 1e-6))
            conf *= view_weight * handle_weight
            dx_cam = float(delta[0]) * np.clip(v_cam[vidx, 2], 1e-6, None) / fx
            dy_cam = float(delta[1]) * np.clip(v_cam[vidx, 2], 1e-6, None) / fy
            local_offsets = dx_cam[:, None] * camera_x_world[None, :] + dy_cam[:, None] * camera_y_world[None, :]
            np.add.at(offsets_accum, vidx, local_offsets * conf[:, None])
            np.add.at(weights_accum, vidx, conf)
            editable[vidx] = True

    constraints = (weights_accum > 1e-6) & editable & ~protected
    if not np.any(constraints):
        report.update({
            "reason": "no usable nose-region constraints",
            "before_metrics": before_metrics,
            "before_profile": before_profile,
            "editable_vertices": int(editable.sum()),
        })
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_nose_region_dense_index(out_dir, report)
        return verts_base, verts_displaced, report

    constraint_offsets = np.zeros_like(offsets_accum)
    constraint_offsets[constraints] = offsets_accum[constraints] / weights_accum[constraints, None]
    norm = np.linalg.norm(constraint_offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        constraint_offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]

    offsets = _smooth_vertex_offsets(
        offsets=constraint_offsets.copy(),
        faces=faces,
        editable=editable & ~protected,
        constraints=constraints,
        constraint_offsets=constraint_offsets,
        max_offset_m=float(max_offset_m),
        iterations=int(smooth_iter),
        alpha=float(smooth_alpha),
        constraint_keep=float(constraint_keep),
    )
    width_guard_removed_vertices = 0
    width_guard_removed_mean_m = 0.0
    if bool(guard_nose_width) and "front" in per_view_results and "front" in view_data and "front" in intrinsics:
        try:
            front_result = per_view_results["front"]
            front_K = intrinsics["front"]
            front_R = front_result["R"]
            front_t = front_result["t"]
            front_target_lmk = np.asarray(view_data["front"]["lmk_2d"], dtype=np.float64)
            front_lmk_proj, _front_errors = _landmark_reprojection_details(
                vertices=verts_displaced,
                K=front_K,
                R=front_R,
                t=front_t,
                target_landmarks=front_target_lmk,
                lmk_vertex_indices=lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )
            target_width = float(np.linalg.norm(front_target_lmk[31] - front_target_lmk[35]))
            model_width = float(np.linalg.norm(front_lmk_proj[31] - front_lmk_proj[35]))
            width_state = None
            if model_width > target_width + 1e-6:
                width_state = "too_wide"
            elif model_width < target_width - 1e-6:
                width_state = "too_narrow"
            if width_state is not None:
                proj_before, _ = _project_vertices_np(verts_displaced, front_K, front_R, front_t)
                center_x = float(front_lmk_proj[LMK_NOSE_BASE_IDX, 0].mean())
                side_sign = np.sign(proj_before[:, 0] - center_x)
                camera_x_world = front_R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_component = offsets @ camera_x_world
                outward = side_sign * x_component
                if width_state == "too_wide":
                    bad = (outward > 0.0) & editable & ~protected
                else:
                    bad = (outward < 0.0) & editable & ~protected
                if np.any(bad):
                    removed = x_component[bad, None] * camera_x_world[None, :]
                    offsets[bad] -= removed
                    width_guard_removed_vertices = int(np.count_nonzero(bad))
                    width_guard_removed_mean_m = float(np.linalg.norm(removed, axis=1).mean())
        except Exception as exc:
            logger.warning("Nose region width guard failed: %s", exc)

    norm = np.linalg.norm(offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]

    verts_base_out = verts_base + offsets.astype(verts_base.dtype, copy=False)
    verts_disp_out = verts_displaced + offsets.astype(verts_displaced.dtype, copy=False)
    after_metrics = _nose_region_metric_report(
        verts_disp_out,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "after",
        selected_by_view=selected_by_view,
        roi_margin_px=roi_margin_px,
    )
    after_profile = _local_profile_contour_mean(
        verts_disp_out,
        faces,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
    )
    for view_name, view_result in per_view_results.items():
        if view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        _save_residual_heatmap_projection(
            image=preprocessed_views[view_name]["image"],
            vertices=verts_disp_out,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            offsets=offsets,
            out_path=out_dir / f"{view_name}_residual_heatmap.png",
        )

    nose_improve = float(before_metrics.get("nose_mean_px", 0.0) - after_metrics.get("nose_mean_px", 0.0))
    front_improve = float(before_metrics.get("front_nose_mean_px", 0.0) - after_metrics.get("front_nose_mean_px", 0.0))
    protected_worsen = float(after_metrics.get("protected_mean_px", 0.0) - before_metrics.get("protected_mean_px", 0.0))
    profile_worsen = float(after_profile.get("mean_px", 0.0) - before_profile.get("mean_px", 0.0))
    before_front = next((r for r in before_metrics.get("records", []) if r.get("view") == "front"), {})
    after_front = next((r for r in after_metrics.get("records", []) if r.get("view") == "front"), {})
    nose_width_abs_worsen = abs(float(after_front.get("nose_width_delta_px", 0.0))) - abs(float(before_front.get("nose_width_delta_px", 0.0)))
    safety = _mesh_offset_safety_report(offsets, faces, protected)
    reject_reasons = []
    if nose_improve < float(min_improve_px):
        reject_reasons.append("nose-improve-too-small")
    if front_improve < float(min_front_improve_px):
        reject_reasons.append("front-nose-improve-too-small")
    if protected_worsen > float(max_protected_worsen_px):
        reject_reasons.append("protected-worsened")
    if profile_worsen > float(max_profile_worsen_px):
        reject_reasons.append("profile-contour-worsened")
    if nose_width_abs_worsen > float(max_nose_width_abs_worsen_px):
        reject_reasons.append("nose-width-worsened")
    if float(safety.get("moved_ratio", 0.0)) > float(max_moved_ratio):
        reject_reasons.append("moved-ratio")
    if float(safety.get("max_offset_m", 0.0)) > float(max_offset_m) + 1e-8:
        reject_reasons.append("max-offset")
    accepted = not reject_reasons
    report.update({
        "applied": bool(accepted),
        "accepted": bool(accepted),
        "reason": "accepted nose region dense residual" if accepted else "rejected: " + ", ".join(reject_reasons),
        "reject_reasons": reject_reasons,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_profile": before_profile,
        "after_profile": after_profile,
        "nose_improve_px": round(float(nose_improve), 3),
        "front_nose_improve_px": round(float(front_improve), 3),
        "protected_worsen_px": round(float(protected_worsen), 3),
        "profile_worsen_px": round(float(profile_worsen), 3),
        "nose_width_abs_worsen_px": round(float(nose_width_abs_worsen), 3),
        "constraint_vertices": int(constraints.sum()),
        "editable_vertices": int(editable.sum()),
        "width_guard_removed_vertices": int(width_guard_removed_vertices),
        "width_guard_removed_mean_m": round(float(width_guard_removed_mean_m), 6),
        "selected_vertices_by_view": {k: int(len(v)) for k, v in selected_by_view.items()},
        "safety": safety,
        "thresholds": {
            "min_improve_px": round(float(min_improve_px), 3),
            "min_front_improve_px": round(float(min_front_improve_px), 3),
            "max_protected_worsen_px": round(float(max_protected_worsen_px), 3),
            "max_profile_worsen_px": round(float(max_profile_worsen_px), 3),
            "max_nose_width_abs_worsen_px": round(float(max_nose_width_abs_worsen_px), 3),
            "max_offset_m": round(float(max_offset_m), 6),
            "max_moved_ratio": round(float(max_moved_ratio), 4),
        },
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_nose_region_dense_index(out_dir, report)
    logger.info(
        "Nose region dense residual: accepted=%s nose %.2fpx front %.2fpx protected_worsen %.2fpx moved=%s",
        accepted,
        nose_improve,
        front_improve,
        protected_worsen,
        safety.get("moved_vertices"),
    )
    if not accepted:
        return verts_base, verts_displaced, report
    return verts_base_out, verts_disp_out, report


def _free_identity_deform_mesh(
    verts_base: np.ndarray,
    verts_displaced: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    debug_dir: Path,
    enabled: bool = True,
    max_offset_m: float = 0.028,
    max_step_px: float = 45.0,
    radius_px: float = 42.0,
    contour_radius_px: float = 56.0,
    mouth_radius_px: float = 36.0,
    view_weight_front: float = 1.0,
    view_weight_side: float = 0.22,
    min_view_cos: float = 0.02,
    smooth_iter: int = 18,
    smooth_alpha: float = 0.22,
    constraint_keep: float = 0.78,
    min_improve_px: float = 0.15,
    max_stable_worsen_px: float = 1.4,
    max_side_worsen_px: float = 2.5,
    max_moved_ratio: float = 0.18,
    stable_anchor_enabled: bool = True,
    anchor_nose_radius_px: float = 38.0,
    anchor_eye_radius_px: float = 30.0,
    anchor_inner_mouth_radius_px: float = 26.0,
    max_anchor_move_m: float = 0.0005,
    max_offset_jump_p95_m: float = 0.012,
    max_offset_jump_m: float = 0.035,
    multiview_enabled: bool = True,
    multiview_min_front_improve_px: float = 0.25,
    multiview_min_overall_improve_px: float = 0.25,
    multiview_max_side_worsen_px: float = 0.35,
    multiview_max_side_mean_worsen_px: float = 0.05,
    multiview_require_side_views: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "accepted": False,
        "view_records": [],
        "contour_target": "semantic_face_shape" if use_semantic_contour else "full_mask",
        "metric_name": "semantic_face_contour_px" if use_semantic_contour else "full_mask_dense_contour_px",
    }
    if not enabled or len(verts_displaced) == 0 or lmk_vertex_indices is None:
        report["reason"] = "disabled or missing landmarks"
        return verts_base, verts_displaced, report

    out_dir = debug_dir / "free_identity_deform"
    before = _free_identity_metric_report(
        verts_displaced,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "before",
    )
    stable_anchors, stable_anchor_report = _stable_anchor_vertices_from_views(
        vertices=verts_displaced,
        view_data=view_data,
        intrinsics=intrinsics,
        per_view_results=per_view_results,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_bary_coords,
        enabled=stable_anchor_enabled,
        nose_radius_px=anchor_nose_radius_px,
        eye_radius_px=anchor_eye_radius_px,
        inner_mouth_radius_px=anchor_inner_mouth_radius_px,
    )
    report["stable_anchors"] = stable_anchor_report

    n = len(verts_displaced)
    offset_accum = np.zeros((n, 3), dtype=np.float64)
    weight_accum = np.zeros(n, dtype=np.float64)
    editable = np.zeros(n, dtype=bool)
    vertex_normals = compute_vertex_normals(verts_displaced, faces)
    constraint_handles = 0

    for view_name, view_result in per_view_results.items():
        if view_name not in view_data or view_name not in preprocessed_views or view_name not in intrinsics:
            continue
        mask = preprocessed_views[view_name].get("shape_mask")
        if mask is None:
            mask = preprocessed_views[view_name].get("face_mask")
        if mask is None:
            continue

        view_weight = float(view_weight_front if view_name == "front" else view_weight_side)
        if view_weight <= 1e-8:
            continue

        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        target_lmk = np.asarray(view_data[view_name]["lmk_2d"], dtype=np.float64)
        proj, v_cam = _project_vertices_np(verts_displaced, K, R, t)
        lmk_world = _landmark_points_3d(
            verts_displaced,
            lmk_vertex_indices,
            lmk_tri_vidx,
            lmk_bary_coords,
        )
        lmk_proj, _lmk_cam = _project_vertices_np(lmk_world, K, R, t)

        h, w = mask.shape[:2]
        sample_x = np.clip(np.round(proj[:, 0]).astype(np.int32), 0, w - 1)
        sample_y = np.clip(np.round(proj[:, 1]).astype(np.int32), 0, h - 1)
        view_dir_world = -R[2, :]
        visible = (vertex_normals @ view_dir_world) > float(min_view_cos)
        valid = (
            (v_cam[:, 2] > 1e-5) &
            (proj[:, 0] >= 0) & (proj[:, 0] < w) &
            (proj[:, 1] >= 0) & (proj[:, 1] < h) &
            (mask[sample_y, sample_x] > 0) &
            visible
        )
        if not np.any(valid):
            continue

        fx = float(K[0, 0]) if abs(float(K[0, 0])) > 1e-6 else 1.0
        fy = float(K[1, 1]) if abs(float(K[1, 1])) > 1e-6 else fx
        for lmk_idx in range(min(68, len(target_lmk), len(lmk_proj))):
            lmk_weight, local_radius = _free_identity_landmark_profile(
                lmk_idx,
                base_radius_px=radius_px,
                contour_radius_px=contour_radius_px,
                mouth_radius_px=mouth_radius_px,
            )
            if lmk_weight <= 0.0:
                continue
            delta = target_lmk[lmk_idx] - lmk_proj[lmk_idx]
            step = float(np.linalg.norm(delta))
            if not np.isfinite(step) or step < 0.35:
                continue
            if step > float(max_step_px):
                delta = delta * (float(max_step_px) / step)
                step = float(max_step_px)

            d2 = ((proj[:, 0] - lmk_proj[lmk_idx, 0]) ** 2) + ((proj[:, 1] - lmk_proj[lmk_idx, 1]) ** 2)
            near = valid & (d2 <= float(local_radius) ** 2) & ~stable_anchors
            if not np.any(near):
                continue

            ids = np.flatnonzero(near)
            sigma2 = max((float(local_radius) * 0.55) ** 2, 1e-6)
            conf = np.exp(-0.5 * d2[ids] / sigma2)
            conf *= float(lmk_weight) * view_weight
            dx_cam = delta[0] * np.clip(v_cam[ids, 2], 1e-6, None) / fx
            dy_cam = delta[1] * np.clip(v_cam[ids, 2], 1e-6, None) / fy
            cam_offsets = np.stack([dx_cam, dy_cam, np.zeros_like(dx_cam)], axis=1)
            world_offsets = cam_offsets @ R
            np.add.at(offset_accum, ids, world_offsets * conf[:, None])
            np.add.at(weight_accum, ids, conf)
            editable[ids] = True
            constraint_handles += 1

    constraints = (weight_accum > 1e-6) & ~stable_anchors
    if not np.any(constraints):
        report["reason"] = "no usable free-identity constraints"
        report["before"] = before
        return verts_base, verts_displaced, report

    constraint_offsets = np.zeros_like(offset_accum)
    constraint_offsets[constraints] = offset_accum[constraints] / weight_accum[constraints, None]
    norm = np.linalg.norm(constraint_offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        constraint_offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]

    editable &= ~stable_anchors
    offsets = _smooth_vertex_offsets(
        offsets=constraint_offsets.copy(),
        faces=faces,
        editable=editable,
        constraints=constraints,
        constraint_offsets=constraint_offsets,
        max_offset_m=float(max_offset_m),
        iterations=int(smooth_iter),
        alpha=float(smooth_alpha),
        constraint_keep=float(constraint_keep),
    )
    verts_base_out = verts_base + offsets.astype(verts_base.dtype, copy=False)
    verts_disp_out = verts_displaced + offsets.astype(verts_displaced.dtype, copy=False)

    after = _free_identity_metric_report(
        verts_disp_out,
        view_data,
        preprocessed_views,
        intrinsics,
        per_view_results,
        lmk_vertex_indices,
        lmk_tri_vidx,
        lmk_bary_coords,
        out_dir,
        "after",
    )

    before_by_view = {r["view"]: r for r in before.get("records", [])}
    after_by_view = {r["view"]: r for r in after.get("records", [])}
    side_worsen = 0.0
    front_improve = None
    view_records = []
    for name, b in before_by_view.items():
        a = after_by_view.get(name)
        if a is None:
            continue
        improve = float(b["mean_px"] - a["mean_px"])
        stable_worsen_view = float(a["stable_mean_px"] - b["stable_mean_px"])
        view_records.append({
            "view": name,
            "before_mean_px": b["mean_px"],
            "after_mean_px": a["mean_px"],
            "mean_improve_px": round(improve, 3),
            "stable_worsen_px": round(stable_worsen_view, 3),
        })
        if name == "front":
            front_improve = improve
        else:
            side_worsen = max(side_worsen, -improve)

    mean_improve = float(before["mean_px"] - after["mean_px"])
    stable_worsen = float(after["stable_mean_px"] - before["stable_mean_px"])
    accept_improve = front_improve if front_improve is not None else mean_improve
    accepted = (
        (mean_improve >= float(min_improve_px) or accept_improve >= float(min_improve_px)) and
        stable_worsen <= float(max_stable_worsen_px) and
        side_worsen <= float(max_side_worsen_px)
    )
    landmark_gate_accepted = bool(accepted)

    if multiview_enabled:
        multiview_report = _multi_view_validation_report(
            view_records=view_records,
            before_key="before_mean_px",
            after_key="after_mean_px",
            improve_key="mean_improve_px",
            min_front_improve_px=float(min_improve_px),
            min_overall_improve_px=float(min_improve_px),
            max_side_worsen_px=float(max_side_worsen_px),
            max_side_mean_worsen_px=float(max_side_worsen_px),
            require_side_views=bool(multiview_require_side_views),
        )
    else:
        multiview_report = _multi_view_validation_report(
            view_records=view_records,
            before_key="before_mean_px",
            after_key="after_mean_px",
            improve_key="mean_improve_px",
            min_front_improve_px=float(min_improve_px),
            min_overall_improve_px=-1e9,
            max_side_worsen_px=float(max_side_worsen_px),
            max_side_mean_worsen_px=1e9,
            require_side_views=False,
        )
        multiview_report["enabled"] = False
    accepted = bool(accepted and multiview_report["accepted"])

    safety = _mesh_offset_safety_report(offsets, faces, stable_anchors)
    safety_reasons = _deform_safety_reject_reasons(
        safety=safety,
        max_moved_ratio=max_moved_ratio,
        max_anchor_move_m=max_anchor_move_m,
        max_offset_jump_p95_m=max_offset_jump_p95_m,
        max_offset_jump_m=max_offset_jump_m,
    )
    validation_dir = debug_dir / "multiview_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    with open(validation_dir / "free_identity_landmarks.json", "w", encoding="utf-8") as f:
        json.dump(multiview_report, f, ensure_ascii=False, indent=2)

    reject_reason = "accepted free identity deformation"
    if not accepted:
        if not landmark_gate_accepted:
            reject_reason = "rejected free identity deformation by landmark gate"
        else:
            reject_reason = "rejected free identity deformation by multiview gate: " + ", ".join(multiview_report["reject_reasons"])
    report.update({
        "applied": bool(accepted),
        "accepted": bool(accepted),
        "reason": reject_reason,
        "before": before,
        "after": after,
        "view_records": view_records,
        "multiview_validation": multiview_report,
        "constraint_handles": int(constraint_handles),
        "constraint_vertices": int(constraints.sum()),
        "moved_vertices": int(safety["moved_vertices"]),
        "moved_ratio": safety["moved_ratio"],
        "mean_offset_m": safety["mean_offset_m"],
        "max_offset_m": safety["max_offset_m"],
        "safety": safety,
        "mean_improve_px": round(mean_improve, 3),
        "front_improve_px": round(float(accept_improve), 3),
        "stable_worsen_px": round(stable_worsen, 3),
        "max_side_worsen_px": round(float(side_worsen), 3),
    })
    if safety_reasons:
        accepted = False
        report["applied"] = False
        report["accepted"] = False
        report["reason"] = "rejected free identity deformation by safety gate: " + ", ".join(safety_reasons)
        report["max_moved_ratio"] = round(float(max_moved_ratio), 4)
        report["max_anchor_move_m"] = round(float(max_anchor_move_m), 6)
        report["max_offset_jump_p95_m"] = round(float(max_offset_jump_p95_m), 6)
        report["max_offset_jump_m"] = round(float(max_offset_jump_m), 6)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(
        "Free-identity deform: "
        f"accepted={accepted}, moved={report['moved_vertices']}, "
        f"landmark {before['mean_px']:.2f}->{after['mean_px']:.2f}px, "
        f"stable_worsen={stable_worsen:.2f}px"
    )
    if not accepted:
        return verts_base, verts_displaced, report
    return verts_base_out, verts_disp_out, report


def _select_free_face_envelope_vertices(
    proj: np.ndarray,
    contour_data: dict,
    side: str,
    topk: int,
    row_sigma: float,
    protected: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = contour_data["rows_np"]
    target = contour_data["target_left_np"] if side == "left" else contour_data["target_right_np"]
    candidate_idx = contour_data["left_idx_np"] if side == "left" else contour_data["right_idx_np"]
    if candidate_idx.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f

    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if protected is not None:
        candidate_idx = candidate_idx[~protected[candidate_idx]]
    if candidate_idx.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f

    k = max(1, int(topk))
    sigma = max(float(row_sigma), 1e-3)
    picked_idx = []
    picked_target = []
    picked_row_weight = []
    row_weight = contour_data["row_weight_np"]
    proj_y = proj[candidate_idx, 1]
    proj_x = proj[candidate_idx, 0]
    for row_i, row in enumerate(rows):
        near = np.flatnonzero(np.abs(proj_y - float(row)) <= sigma)
        if near.size == 0:
            continue
        if side == "left":
            order = np.argsort(proj_x[near])[:k]
        else:
            order = np.argsort(-proj_x[near])[:k]
        chosen = candidate_idx[near[order]]
        picked_idx.append(chosen)
        picked_target.append(np.full(chosen.shape, float(target[row_i]), dtype=np.float64))
        picked_row_weight.append(np.full(chosen.shape, float(row_weight[row_i]), dtype=np.float64))

    if not picked_idx:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_f, empty_f
    return (
        np.concatenate(picked_idx).astype(np.int64, copy=False),
        np.concatenate(picked_target).astype(np.float64, copy=False),
        np.concatenate(picked_row_weight).astype(np.float64, copy=False),
    )


def _personal_residual_target_sides(view_name: str, use_profile_side_contour: bool) -> Tuple[Tuple[str, ...], Optional[str]]:
    if not use_profile_side_contour or view_name == "front":
        return ("left", "right"), None
    lower = str(view_name).lower()
    if lower.startswith("left"):
        return ("right",), "right"
    if lower.startswith("right"):
        return ("left",), "left"
    return ("left", "right"), None


def _save_residual_heatmap_projection(
    image: np.ndarray,
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    offsets: np.ndarray,
    out_path: Path,
) -> None:
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    proj, v_cam = _project_vertices_np(vertices, K, R, t)
    offset_mm = np.linalg.norm(np.asarray(offsets, dtype=np.float64), axis=1) * 1000.0
    max_mm = max(float(np.percentile(offset_mm, 99)) if offset_mm.size else 0.0, 1e-6)
    norm = np.clip(offset_mm / max_mm, 0.0, 1.0)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    valid = (
        (v_cam[:, 2] > 1e-5)
        & (proj[:, 0] >= 0)
        & (proj[:, 0] < w)
        & (proj[:, 1] >= 0)
        & (proj[:, 1] < h)
        & (offset_mm > 0.05)
    )
    ids = np.flatnonzero(valid)
    if ids.size > 18000:
        # Keep the debug image responsive while preserving the strongest residuals.
        order = np.argsort(-offset_mm[ids])[:18000]
        ids = ids[order]
    for idx in ids:
        px = int(round(float(proj[idx, 0])))
        py = int(round(float(proj[idx, 1])))
        cv2.circle(img, (px, py), 1, tuple(int(c) for c in colors[idx, 0]), -1)
    cv2.putText(
        img,
        f"residual heatmap, p99={max_mm:.1f}mm",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), img)


def _write_personal_residual_index(out_dir: Path, report: dict) -> None:
    import html

    def esc(value) -> str:
        return html.escape(str(value))

    rows = []
    figures = []
    for rec in report.get("view_records", []):
        view = rec.get("view", "")
        rows.append(
            "<tr>"
            f"<td>{esc(view)}</td>"
            f"<td>{esc(rec.get('role'))}</td>"
            f"<td>{esc(','.join(rec.get('target_sides', []) or []))}</td>"
            f"<td>{esc(rec.get('profile_target_side'))}</td>"
            f"<td>{esc(rec.get('before_dense_contour_px'))}</td>"
            f"<td>{esc(rec.get('after_dense_contour_px'))}</td>"
            f"<td>{esc(rec.get('dense_improve_px'))}</td>"
            f"<td>{esc(rec.get('constraint_weight'))}</td>"
            f"</tr>"
        )
        for suffix, caption in (
            ("before_contour.png", "before dense contour"),
            ("after_contour.png", "after dense contour"),
            ("after_mesh.png", "after mesh projection"),
            ("residual_heatmap.png", "residual heatmap"),
        ):
            path = out_dir / f"{view}_{suffix}"
            if path.exists():
                figures.append(
                    "<figure>"
                    f"<img src='{esc(path.name)}'>"
                    f"<figcaption>{esc(view)} - {esc(caption)}</figcaption>"
                    "</figure>"
                )
    accepted_cls = "ok" if report.get("accepted") else "bad"
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Personal residual deformation audit</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;background:#07111f;color:#edf5ff;margin:24px}}
.panel{{background:#0e1b2d;border:1px solid #28415f;border-radius:14px;padding:18px;margin:0 0 18px}}
table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #28415f;padding:9px;text-align:left}}
.ok{{color:#4ade80}}.bad{{color:#fb7185}}.muted{{color:#9fb2ca}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:16px}}
figure{{margin:0;background:#091525;border:1px solid #28415f;border-radius:12px;overflow:hidden}}
img{{width:100%;display:block}}figcaption{{padding:8px 10px;color:#9fb2ca}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>Personal residual deformation audit</h1>
<section class="panel">
<p>Accepted: <b class="{accepted_cls}">{esc(report.get('accepted'))}</b></p>
<p>Reason: {esc(report.get('reason'))}</p>
<p class="muted">This stage adds bounded per-vertex residual offsets on top of FLAME. It runs before depth displacement and rolls back when gates fail.</p>
</section>
<section class="panel">
<h2>Metrics</h2>
<table>
<thead><tr><th>view</th><th>role</th><th>target sides</th><th>profile side</th><th>before dense</th><th>after dense</th><th>improve</th><th>weight</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</section>
<section class="panel">
<h2>Safety</h2>
<pre>{esc(json.dumps(report.get('safety', {}), ensure_ascii=False, indent=2))}</pre>
</section>
<section class="grid">
{''.join(figures)}
</section>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def _personal_residual_deform_mesh(
    verts_base: np.ndarray,
    verts_displaced: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    debug_dir: Path,
    lmk_vertex_indices: Optional[np.ndarray] = None,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
    enabled: bool = True,
    max_offset_m: float = 0.018,
    boundary_band_px: float = 95.0,
    search_margin_px: float = 80.0,
    stable_protect_radius_px: float = 44.0,
    row_step: int = 6,
    row_sigma: float = 8.0,
    tau: float = 10.0,
    envelope_topk: int = 26,
    view_weight_front: float = 0.35,
    view_weight_side: float = 1.0,
    min_view_cos: float = 0.03,
    max_step_px: float = 56.0,
    smooth_iter: int = 30,
    smooth_alpha: float = 0.30,
    constraint_keep: float = 0.55,
    target_side_improve_px: float = 8.0,
    min_side_mean_improve_px: float = 2.0,
    max_stable_worsen_px: float = 0.75,
    max_global_worsen_px: float = 1.0,
    max_side_worsen_px: float = 1.0,
    max_moved_ratio: float = 0.12,
    stable_anchor_enabled: bool = True,
    anchor_nose_radius_px: float = 38.0,
    anchor_eye_radius_px: float = 30.0,
    anchor_inner_mouth_radius_px: float = 26.0,
    max_anchor_move_m: float = 0.0005,
    max_offset_jump_p95_m: float = 0.012,
    max_offset_jump_m: float = 0.028,
    use_semantic_contour: bool = True,
    use_profile_side_contour: bool = True,
    semantic_mask_dilate_px: float = 6.0,
    semantic_edit_margin_px: float = 28.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "accepted": False,
        "view_records": [],
        "contour_target": "semantic_profile_side" if use_profile_side_contour else (
            "semantic_face_shape" if use_semantic_contour else "full_mask"
        ),
        "metric_name": "profile_semantic_contour_px" if use_profile_side_contour else (
            "semantic_face_contour_px" if use_semantic_contour else "full_mask_dense_contour_px"
        ),
    }
    out_dir = debug_dir / "personal_residual_deform"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not enabled or len(verts_displaced) == 0:
        report["reason"] = "disabled"
        _write_personal_residual_index(out_dir, report)
        return verts_base, verts_displaced, report

    stable_anchors, stable_anchor_report = _stable_anchor_vertices_from_views(
        vertices=verts_displaced,
        view_data=view_data,
        intrinsics=intrinsics,
        per_view_results=per_view_results,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_bary_coords,
        enabled=stable_anchor_enabled and lmk_vertex_indices is not None,
        nose_radius_px=anchor_nose_radius_px,
        eye_radius_px=anchor_eye_radius_px,
        inner_mouth_radius_px=anchor_inner_mouth_radius_px,
    )
    report["stable_anchors"] = stable_anchor_report

    before_landmarks = {}
    if lmk_vertex_indices is not None:
        before_landmarks = _free_identity_metric_report(
            verts_displaced,
            view_data,
            preprocessed_views,
            intrinsics,
            per_view_results,
            lmk_vertex_indices,
            lmk_tri_vidx,
            lmk_bary_coords,
            out_dir,
            "before",
        )

    n = len(verts_displaced)
    offset_accum = np.zeros((n, 3), dtype=np.float64)
    weight_accum = np.zeros(n, dtype=np.float64)
    editable = np.zeros(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    contour_data_by_view = {}
    full_contour_data_by_view = {}
    vertex_normals = compute_vertex_normals(verts_displaced, faces)

    for view_name, view_result in per_view_results.items():
        if view_name not in preprocessed_views or view_name not in intrinsics or view_name not in view_data:
            continue
        mask = preprocessed_views[view_name].get("shape_mask")
        if mask is None:
            mask = preprocessed_views[view_name].get("face_mask")
        if mask is None:
            continue

        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        lmk = view_data[view_name]["lmk_2d"]
        mp_lmk = preprocessed_views[view_name].get("landmarks")
        view_dir_world = -R[2, :]
        visible_mask = (vertex_normals @ view_dir_world) > float(min_view_cos)
        if use_semantic_contour:
            contour_data = _build_semantic_face_contour_data(
                vertices=verts_displaced,
                K=K,
                R=R,
                t=t,
                mask=mask,
                landmarks_2d=lmk,
                mediapipe_landmarks=mp_lmk,
                row_step=row_step,
                row_sigma=row_sigma,
                tau=tau,
                boundary_band_px=boundary_band_px,
                search_margin_px=search_margin_px,
                visible_mask=visible_mask,
                semantic_mask_dilate_px=semantic_mask_dilate_px,
            )
        else:
            contour_data = _build_free_face_contour_data(
                vertices=verts_displaced,
                K=K,
                R=R,
                t=t,
                mask=mask,
                landmarks_2d=lmk,
                row_step=row_step,
                row_sigma=row_sigma,
                tau=tau,
                boundary_band_px=boundary_band_px,
                search_margin_px=search_margin_px,
                visible_mask=visible_mask,
            )
        if contour_data is None:
            continue
        contour_data_by_view[view_name] = contour_data
        if use_semantic_contour:
            full_contour_data = _build_free_face_contour_data(
                vertices=verts_displaced,
                K=K,
                R=R,
                t=t,
                mask=mask,
                landmarks_2d=lmk,
                row_step=row_step,
                row_sigma=row_sigma,
                tau=tau,
                boundary_band_px=boundary_band_px,
                search_margin_px=search_margin_px,
                visible_mask=visible_mask,
            )
            if full_contour_data is not None:
                full_contour_data_by_view[view_name] = full_contour_data

        proj, v_cam = _project_vertices_np(verts_displaced, K, R, t)
        target_sides, profile_target_side = _personal_residual_target_sides(
            view_name,
            use_profile_side_contour=use_profile_side_contour,
        )
        before_metric, before_left, before_right = _dense_contour_metric_np_for_sides(
            proj,
            contour_data,
            target_sides,
        )
        _save_dense_contour_debug_image(
            preprocessed_views[view_name]["image"],
            contour_data,
            before_left,
            before_right,
            out_dir / f"{view_name}_before_contour.png",
        )
        target_mask = contour_data.get("target_mask")
        if target_mask is not None:
            cv2.imwrite(str(out_dir / f"{view_name}_semantic_target_mask.png"), target_mask)
        before_full_metric = None
        full_data = full_contour_data_by_view.get(view_name)
        if full_data is not None:
            before_full_metric, _full_before_left, _full_before_right = _dense_contour_metric_np_for_sides(
                proj,
                full_data,
                target_sides,
            )

        row_idx = np.round(proj[:, 1]).astype(np.int32)
        row_idx = np.clip(row_idx, 0, mask.shape[0] - 1)
        valid = contour_data["valid"][row_idx]
        xmin = contour_data["xmin"][row_idx]
        xmax = contour_data["xmax"][row_idx]
        y_min = contour_data["y_min"]
        y_max = contour_data["y_max"]
        edit_margin_y = float(semantic_edit_margin_px if use_semantic_contour else boundary_band_px)
        edit_margin_y = max(edit_margin_y, float(row_sigma) * 2.0)
        in_y = (proj[:, 1] >= y_min - edit_margin_y) & (proj[:, 1] <= y_max + edit_margin_y)
        in_x = (proj[:, 0] >= xmin - search_margin_px) & (proj[:, 0] <= xmax + search_margin_px)
        view_editable = valid & in_y & in_x & visible_mask & (v_cam[:, 2] > 1e-5)
        landmark_protected = _stable_protect_vertices(proj, lmk, stable_protect_radius_px)
        view_protected = (stable_anchors | landmark_protected) if stable_anchor_enabled else landmark_protected
        protected |= view_protected

        view_weight = float(view_weight_front if view_name == "front" else view_weight_side)
        if view_weight > 1e-8:
            editable |= view_editable
        camera_x_world = R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fx = float(K[0, 0]) if abs(float(K[0, 0])) > 1e-6 else 1.0
        side_constraint_counts = {}
        for side in target_sides:
            idx, target_x, row_weight = _select_free_face_envelope_vertices(
                proj=proj,
                contour_data=contour_data,
                side=side,
                topk=envelope_topk,
                row_sigma=row_sigma,
                protected=view_protected,
            )
            side_constraint_counts[side] = int(len(idx))
            if idx.size == 0 or view_weight <= 1e-8:
                continue
            du = np.clip(target_x - proj[idx, 0], -float(max_step_px), float(max_step_px))
            dist = np.abs(target_x - proj[idx, 0])
            conf = np.exp(-0.5 * (dist / max(float(boundary_band_px), 1e-3)) ** 2)
            conf *= row_weight * view_weight
            dx_cam = du * np.clip(v_cam[idx, 2], 1e-6, None) / fx
            offset_world = dx_cam[:, None] * camera_x_world[None, :]
            np.add.at(offset_accum, idx, offset_world * conf[:, None])
            np.add.at(weight_accum, idx, conf)

        report["view_records"].append({
            "view": view_name,
            "role": "front" if view_name == "front" else "side",
            "target_sides": list(target_sides),
            "profile_target_side": profile_target_side,
            "contour_target": "semantic_profile_side" if profile_target_side else (
                "semantic_face_shape" if use_semantic_contour else "full_mask"
            ),
            "metric_name": "profile_semantic_contour_px" if profile_target_side else contour_data.get("metric_name", "full_mask_dense_contour_px"),
            "semantic_source": contour_data.get("semantic_source"),
            "semantic_area_px": contour_data.get("semantic_area_px"),
            "before_dense_contour_px": round(float(before_metric), 3),
            "before_profile_contour_px": round(float(before_metric), 3) if profile_target_side else None,
            "before_semantic_face_contour_px": round(float(before_metric), 3) if use_semantic_contour else None,
            "before_full_mask_dense_contour_px": (
                None if before_full_metric is None else round(float(before_full_metric), 3)
            ),
            "rows": int(len(contour_data["rows_np"])),
            "left_candidates": int(len(contour_data["left_idx_np"])),
            "right_candidates": int(len(contour_data["right_idx_np"])),
            "left_envelope_vertices": int(side_constraint_counts.get("left", 0)),
            "right_envelope_vertices": int(side_constraint_counts.get("right", 0)),
            "constraint_weight": round(float(view_weight), 3),
        })

    constraints = (weight_accum > 1e-6) & editable & ~protected
    if not np.any(constraints):
        report["reason"] = "no usable personal residual constraints"
        report["before_landmarks"] = before_landmarks
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        _write_personal_residual_index(out_dir, report)
        return verts_base, verts_displaced, report

    constraint_offsets = np.zeros_like(offset_accum)
    constraint_offsets[constraints] = offset_accum[constraints] / weight_accum[constraints, None]
    norm = np.linalg.norm(constraint_offsets, axis=1)
    too_far = norm > float(max_offset_m)
    if np.any(too_far):
        constraint_offsets[too_far] *= (float(max_offset_m) / np.clip(norm[too_far], 1e-8, None))[:, None]

    editable &= ~protected
    offsets = _smooth_vertex_offsets(
        offsets=constraint_offsets.copy(),
        faces=faces,
        editable=editable,
        constraints=constraints,
        constraint_offsets=constraint_offsets,
        max_offset_m=float(max_offset_m),
        iterations=int(smooth_iter),
        alpha=float(smooth_alpha),
        constraint_keep=float(constraint_keep),
    )
    verts_base_out = verts_base + offsets.astype(verts_base.dtype, copy=False)
    verts_disp_out = verts_displaced + offsets.astype(verts_displaced.dtype, copy=False)

    for rec in report["view_records"]:
        view_name = rec["view"]
        contour_data = contour_data_by_view[view_name]
        view_result = per_view_results[view_name]
        proj_after, _ = _project_vertices_np(
            verts_disp_out,
            intrinsics[view_name],
            view_result["R"],
            view_result["t"],
        )
        after_metric, after_left, after_right = _dense_contour_metric_np_for_sides(
            proj_after,
            contour_data,
            tuple(rec.get("target_sides", ["left", "right"])),
        )
        _save_dense_contour_debug_image(
            preprocessed_views[view_name]["image"],
            contour_data,
            after_left,
            after_right,
            out_dir / f"{view_name}_after_contour.png",
        )
        rec["after_dense_contour_px"] = round(float(after_metric), 3)
        rec["dense_improve_px"] = round(float(rec["before_dense_contour_px"] - after_metric), 3)
        if rec.get("profile_target_side"):
            rec["after_profile_contour_px"] = round(float(after_metric), 3)
            rec["profile_improve_px"] = round(float(rec["before_dense_contour_px"] - after_metric), 3)
        if rec.get("contour_target") == "semantic_face_shape":
            rec["after_semantic_face_contour_px"] = round(float(after_metric), 3)
            rec["semantic_face_improve_px"] = round(float(rec["before_dense_contour_px"] - after_metric), 3)
        full_data = full_contour_data_by_view.get(view_name)
        if full_data is not None:
            full_after_metric, _full_after_left, _full_after_right = _dense_contour_metric_np_for_sides(
                proj_after,
                full_data,
                tuple(rec.get("target_sides", ["left", "right"])),
            )
            rec["after_full_mask_dense_contour_px"] = round(float(full_after_metric), 3)
            before_full = rec.get("before_full_mask_dense_contour_px")
            if before_full is not None:
                rec["full_mask_dense_improve_px"] = round(float(before_full - full_after_metric), 3)
        face_mask = preprocessed_views[view_name].get("face_mask")
        if face_mask is None:
            face_mask = preprocessed_views[view_name].get("shape_mask")
        _save_projection_debug(
            verts_disp_out,
            faces,
            intrinsics[view_name],
            view_result["R"],
            view_result["t"],
            preprocessed_views[view_name]["image"],
            face_mask,
            out_dir / f"{view_name}_after_mesh.png",
        )
        _save_residual_heatmap_projection(
            image=preprocessed_views[view_name]["image"],
            vertices=verts_disp_out,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            offsets=offsets,
            out_path=out_dir / f"{view_name}_residual_heatmap.png",
        )

    after_landmarks = {}
    stable_worsen = 0.0
    global_worsen = 0.0
    if lmk_vertex_indices is not None:
        after_landmarks = _free_identity_metric_report(
            verts_disp_out,
            view_data,
            preprocessed_views,
            intrinsics,
            per_view_results,
            lmk_vertex_indices,
            lmk_tri_vidx,
            lmk_bary_coords,
            out_dir,
            "after",
        )
        stable_worsen = float(after_landmarks.get("stable_mean_px", 0.0) - before_landmarks.get("stable_mean_px", 0.0))
        global_worsen = float(after_landmarks.get("mean_px", 0.0) - before_landmarks.get("mean_px", 0.0))

    side_improves = [
        float(rec.get("dense_improve_px", 0.0))
        for rec in report["view_records"]
        if rec.get("role") == "side"
    ]
    front_improves = [
        float(rec.get("dense_improve_px", 0.0))
        for rec in report["view_records"]
        if rec.get("role") == "front"
    ]
    max_side_improve = max(side_improves, default=0.0)
    side_mean_improve = float(np.mean(side_improves)) if side_improves else 0.0
    max_side_worsen = max((max(0.0, -v) for v in side_improves), default=0.0)
    overall_before = float(np.mean([float(r["before_dense_contour_px"]) for r in report["view_records"]])) if report["view_records"] else 0.0
    overall_after = float(np.mean([float(r.get("after_dense_contour_px", r["before_dense_contour_px"])) for r in report["view_records"]])) if report["view_records"] else 0.0

    safety = _mesh_offset_safety_report(offsets, faces, stable_anchors if stable_anchor_enabled else protected)
    safety_reasons = _deform_safety_reject_reasons(
        safety=safety,
        max_moved_ratio=max_moved_ratio,
        max_anchor_move_m=max_anchor_move_m,
        max_offset_jump_p95_m=max_offset_jump_p95_m,
        max_offset_jump_m=max_offset_jump_m,
    )
    reject_reasons = []
    if not side_improves:
        reject_reasons.append("missing-side-views")
    if max_side_improve < float(target_side_improve_px):
        reject_reasons.append("side-target-improve-too-small")
    if side_mean_improve < float(min_side_mean_improve_px):
        reject_reasons.append("side-mean-improve-too-small")
    if max_side_worsen > float(max_side_worsen_px):
        reject_reasons.append("side-view-worsened")
    if stable_worsen > float(max_stable_worsen_px):
        reject_reasons.append("stable-landmarks-worsened")
    if global_worsen > float(max_global_worsen_px):
        reject_reasons.append("global-landmarks-worsened")
    reject_reasons.extend([f"safety-{reason}" for reason in safety_reasons])
    accepted = not reject_reasons

    report.update({
        "applied": bool(accepted),
        "accepted": bool(accepted),
        "reason": "accepted personal residual deformation" if accepted else "rejected: " + ", ".join(reject_reasons),
        "reject_reasons": reject_reasons,
        "before_landmarks": before_landmarks,
        "after_landmarks": after_landmarks,
        "stable_worsen_px": round(float(stable_worsen), 3),
        "global_worsen_px": round(float(global_worsen), 3),
        "max_side_improve_px": round(float(max_side_improve), 3),
        "side_mean_improve_px": round(float(side_mean_improve), 3),
        "max_side_worsen_px": round(float(max_side_worsen), 3),
        "overall_before_dense_contour_px": round(float(overall_before), 3),
        "overall_after_dense_contour_px": round(float(overall_after), 3),
        "overall_dense_improve_px": round(float(overall_before - overall_after), 3),
        "constraint_vertices": int(constraints.sum()),
        "editable_vertices": int(editable.sum()),
        "safety": safety,
        "thresholds": {
            "target_side_improve_px": round(float(target_side_improve_px), 3),
            "min_side_mean_improve_px": round(float(min_side_mean_improve_px), 3),
            "max_stable_worsen_px": round(float(max_stable_worsen_px), 3),
            "max_global_worsen_px": round(float(max_global_worsen_px), 3),
            "max_side_worsen_px": round(float(max_side_worsen_px), 3),
            "max_offset_m": round(float(max_offset_m), 6),
            "max_moved_ratio": round(float(max_moved_ratio), 4),
        },
    })
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_personal_residual_index(out_dir, report)
    logger.info(
        "Personal residual deform: accepted=%s side %.2fpx mean %.2fpx stable_worsen %.2fpx moved=%s",
        accepted,
        max_side_improve,
        side_mean_improve,
        stable_worsen,
        safety.get("moved_vertices"),
    )
    if not accepted:
        return verts_base, verts_displaced, report
    return verts_base_out, verts_disp_out, report


def _free_face_deform_mesh(
    verts_base: np.ndarray,
    verts_displaced: np.ndarray,
    faces: np.ndarray,
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    per_view_results: Dict[str, dict],
    debug_dir: Path,
    lmk_vertex_indices: Optional[np.ndarray] = None,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
    enabled: bool = True,
    max_offset_m: float = 0.018,
    boundary_band_px: float = 95.0,
    search_margin_px: float = 80.0,
    stable_protect_radius_px: float = 58.0,
    row_step: int = 6,
    row_sigma: float = 8.0,
    tau: float = 10.0,
    envelope_topk: int = 28,
    view_weight_front: float = 1.0,
    view_weight_side: float = 0.75,
    min_view_cos: float = 0.03,
    max_step_px: float = 36.0,
    smooth_iter: int = 35,
    smooth_alpha: float = 0.35,
    constraint_keep: float = 0.45,
    accept_min_improve_px: float = 0.25,
    max_side_worsen_px: float = 2.0,
    max_moved_ratio: float = 0.08,
    stable_anchor_enabled: bool = True,
    anchor_nose_radius_px: float = 38.0,
    anchor_eye_radius_px: float = 30.0,
    anchor_inner_mouth_radius_px: float = 26.0,
    max_anchor_move_m: float = 0.0005,
    max_offset_jump_p95_m: float = 0.012,
    max_offset_jump_m: float = 0.035,
    multiview_enabled: bool = True,
    multiview_min_front_improve_px: float = 0.25,
    multiview_min_overall_improve_px: float = 0.25,
    multiview_max_side_worsen_px: float = 0.35,
    multiview_max_side_mean_worsen_px: float = 0.05,
    multiview_require_side_views: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    report = {
        "enabled": bool(enabled),
        "applied": False,
        "view_records": [],
    }
    if not enabled or len(verts_displaced) == 0:
        report["reason"] = "disabled"
        return verts_base, verts_displaced, report

    out_dir = debug_dir / "free_face_deform"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(verts_displaced)
    offset_accum = np.zeros((n, 3), dtype=np.float64)
    weight_accum = np.zeros(n, dtype=np.float64)
    editable = np.zeros(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    contour_data_by_view = {}
    vertex_normals = compute_vertex_normals(verts_displaced, faces)
    stable_anchors, stable_anchor_report = _stable_anchor_vertices_from_views(
        vertices=verts_displaced,
        view_data=view_data,
        intrinsics=intrinsics,
        per_view_results=per_view_results,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_bary_coords,
        enabled=stable_anchor_enabled,
        nose_radius_px=anchor_nose_radius_px,
        eye_radius_px=anchor_eye_radius_px,
        inner_mouth_radius_px=anchor_inner_mouth_radius_px,
    )
    report["stable_anchors"] = stable_anchor_report

    for view_name, view_result in per_view_results.items():
        if view_name not in preprocessed_views or view_name not in intrinsics or view_name not in view_data:
            continue
        mask = preprocessed_views[view_name].get("shape_mask")
        if mask is None:
            mask = preprocessed_views[view_name].get("face_mask")
        if mask is None:
            continue

        K = intrinsics[view_name]
        R = view_result["R"]
        t = view_result["t"]
        lmk = view_data[view_name]["lmk_2d"]
        view_dir_world = -R[2, :]
        visible_mask = (vertex_normals @ view_dir_world) > float(min_view_cos)
        contour_data = _build_free_face_contour_data(
            vertices=verts_displaced,
            K=K,
            R=R,
            t=t,
            mask=mask,
            landmarks_2d=lmk,
            row_step=row_step,
            row_sigma=row_sigma,
            tau=tau,
            boundary_band_px=boundary_band_px,
            search_margin_px=search_margin_px,
            visible_mask=visible_mask,
        )
        if contour_data is None:
            continue
        contour_data_by_view[view_name] = contour_data

        proj, v_cam = _project_vertices_np(verts_displaced, K, R, t)
        before_metric, before_left, before_right = _dense_contour_metric_np(proj, contour_data)
        _save_dense_contour_debug_image(
            preprocessed_views[view_name]["image"],
            contour_data,
            before_left,
            before_right,
            out_dir / f"{view_name}_before_contour.png",
        )

        row_idx = np.round(proj[:, 1]).astype(np.int32)
        row_idx = np.clip(row_idx, 0, mask.shape[0] - 1)
        valid = contour_data["valid"][row_idx]
        xmin = contour_data["xmin"][row_idx]
        xmax = contour_data["xmax"][row_idx]
        y_min = contour_data["y_min"]
        y_max = contour_data["y_max"]
        in_lower = (proj[:, 1] >= y_min - boundary_band_px) & (proj[:, 1] <= y_max + boundary_band_px)
        in_x = (proj[:, 0] >= xmin - search_margin_px) & (proj[:, 0] <= xmax + search_margin_px)
        view_editable = valid & in_lower & in_x & visible_mask & (v_cam[:, 2] > 1e-5)
        legacy_protected = _stable_protect_vertices(proj, lmk, stable_protect_radius_px)
        if stable_anchor_enabled:
            view_protected = stable_anchors | legacy_protected
        else:
            view_protected = legacy_protected
        protected |= view_protected

        view_weight = float(view_weight_front if view_name == "front" else view_weight_side)
        if view_weight > 1e-8:
            editable |= view_editable
        camera_x_world = R.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fx = float(K[0, 0]) if abs(float(K[0, 0])) > 1e-6 else 1.0
        side_constraint_counts = {}
        for side in ("left", "right"):
            idx, target_x, row_weight = _select_free_face_envelope_vertices(
                proj=proj,
                contour_data=contour_data,
                side=side,
                topk=envelope_topk,
                row_sigma=row_sigma,
                protected=view_protected,
            )
            side_constraint_counts[side] = int(len(idx))
            if idx.size == 0:
                continue
            if view_weight <= 1e-8:
                continue
            du = np.clip(target_x - proj[idx, 0], -float(max_step_px), float(max_step_px))
            dist = np.abs(target_x - proj[idx, 0])
            conf = np.exp(-0.5 * (dist / max(float(boundary_band_px), 1e-3)) ** 2)
            conf *= row_weight
            conf *= view_weight
            dx_cam = du * np.clip(v_cam[idx, 2], 1e-6, None) / fx
            offset_world = dx_cam[:, None] * camera_x_world[None, :]
            np.add.at(offset_accum, idx, offset_world * conf[:, None])
            np.add.at(weight_accum, idx, conf)

        report["view_records"].append({
            "view": view_name,
            "before_dense_contour_px": round(float(before_metric), 3),
            "rows": int(len(contour_data["rows_np"])),
            "left_candidates": int(len(contour_data["left_idx_np"])),
            "right_candidates": int(len(contour_data["right_idx_np"])),
            "left_envelope_vertices": int(side_constraint_counts.get("left", 0)),
            "right_envelope_vertices": int(side_constraint_counts.get("right", 0)),
            "constraint_weight": round(float(view_weight), 3),
        })

    constraints = (weight_accum > 1e-6) & editable & ~protected
    if not np.any(constraints):
        report["reason"] = "no usable free-face constraints"
        return verts_base, verts_displaced, report

    constraint_offsets = np.zeros_like(offset_accum)
    constraint_offsets[constraints] = offset_accum[constraints] / weight_accum[constraints, None]
    norm = np.linalg.norm(constraint_offsets, axis=1)
    too_far = norm > max_offset_m
    if np.any(too_far):
        constraint_offsets[too_far] *= (max_offset_m / np.clip(norm[too_far], 1e-8, None))[:, None]

    editable &= ~protected
    offsets = _smooth_vertex_offsets(
        offsets=constraint_offsets.copy(),
        faces=faces,
        editable=editable,
        constraints=constraints,
        constraint_offsets=constraint_offsets,
        max_offset_m=float(max_offset_m),
        iterations=int(smooth_iter),
        alpha=float(smooth_alpha),
        constraint_keep=float(constraint_keep),
    )
    verts_base_out = verts_base + offsets.astype(verts_base.dtype, copy=False)
    verts_disp_out = verts_displaced + offsets.astype(verts_displaced.dtype, copy=False)

    after_metrics = []
    for rec in report["view_records"]:
        view_name = rec["view"]
        contour_data = contour_data_by_view[view_name]
        view_result = per_view_results[view_name]
        proj_after, _ = _project_vertices_np(
            verts_disp_out,
            intrinsics[view_name],
            view_result["R"],
            view_result["t"],
        )
        after_metric, after_left, after_right = _dense_contour_metric_np(proj_after, contour_data)
        _save_dense_contour_debug_image(
            preprocessed_views[view_name]["image"],
            contour_data,
            after_left,
            after_right,
            out_dir / f"{view_name}_after_contour.png",
        )
        rec["after_dense_contour_px"] = round(float(after_metric), 3)
        rec["dense_improve_px"] = round(float(rec["before_dense_contour_px"] - after_metric), 3)
        after_metrics.append(float(after_metric))
        face_mask = preprocessed_views[view_name].get("face_mask")
        if face_mask is None:
            face_mask = preprocessed_views[view_name].get("shape_mask")
        _save_projection_debug(
            verts_disp_out,
            faces,
            intrinsics[view_name],
            view_result["R"],
            view_result["t"],
            preprocessed_views[view_name]["image"],
            face_mask,
            out_dir / f"{view_name}_after_mesh.png",
        )

    safety = _mesh_offset_safety_report(offsets, faces, stable_anchors if stable_anchor_enabled else protected)
    safety_reasons = _deform_safety_reject_reasons(
        safety=safety,
        max_moved_ratio=max_moved_ratio,
        max_anchor_move_m=max_anchor_move_m,
        max_offset_jump_p95_m=max_offset_jump_p95_m,
        max_offset_jump_m=max_offset_jump_m,
    )
    if multiview_enabled:
        multiview_report = _multi_view_validation_report(
            view_records=report["view_records"],
            before_key="before_dense_contour_px",
            after_key="after_dense_contour_px",
            improve_key="dense_improve_px",
            min_front_improve_px=max(float(accept_min_improve_px), float(multiview_min_front_improve_px)),
            min_overall_improve_px=float(multiview_min_overall_improve_px),
            max_side_worsen_px=min(float(max_side_worsen_px), float(multiview_max_side_worsen_px)),
            max_side_mean_worsen_px=float(multiview_max_side_mean_worsen_px),
            require_side_views=bool(multiview_require_side_views),
        )
    else:
        multiview_report = _multi_view_validation_report(
            view_records=report["view_records"],
            before_key="before_dense_contour_px",
            after_key="after_dense_contour_px",
            improve_key="dense_improve_px",
            min_front_improve_px=float(accept_min_improve_px),
            min_overall_improve_px=-1e9,
            max_side_worsen_px=float(max_side_worsen_px),
            max_side_mean_worsen_px=1e9,
            require_side_views=False,
        )
        multiview_report["enabled"] = False

    before_mean = float(multiview_report["overall_before_px"])
    after_mean = float(multiview_report["overall_after_px"])
    accept_improve = float(multiview_report["front_improve_px"])
    side_worsen = float(multiview_report["max_side_worsen_px"])
    accepted = bool(multiview_report["accepted"])
    if safety_reasons:
        accepted = False

    validation_dir = debug_dir / "multiview_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    with open(validation_dir / "free_face_dense_contour.json", "w", encoding="utf-8") as f:
        json.dump(multiview_report, f, ensure_ascii=False, indent=2)

    report.update({
        "applied": bool(accepted),
        "accepted": bool(accepted),
        "reason": (
            "accepted free-face envelope deformation"
            if accepted else
            (
                "rejected free-face deformation by safety gate: " + ", ".join(safety_reasons)
                if safety_reasons else
                "rejected free-face deformation by multiview gate: " + ", ".join(multiview_report["reject_reasons"])
            )
        ),
        "multiview_validation": multiview_report,
        "editable_vertices": int(editable.sum()),
        "constraint_vertices": int(constraints.sum()),
        "moved_vertices": int(safety["moved_vertices"]),
        "moved_ratio": safety["moved_ratio"],
        "mean_offset_m": safety["mean_offset_m"],
        "max_offset_m": safety["max_offset_m"],
        "safety": safety,
        "max_moved_ratio": round(float(max_moved_ratio), 4),
        "max_anchor_move_m": round(float(max_anchor_move_m), 6),
        "max_offset_jump_p95_m": round(float(max_offset_jump_p95_m), 6),
        "max_offset_jump_m": round(float(max_offset_jump_m), 6),
        "before_dense_contour_px": round(before_mean, 3),
        "after_dense_contour_px": round(after_mean, 3),
        "front_dense_improve_px": round(float(accept_improve), 3),
        "max_side_worsen_px": round(float(side_worsen), 3),
    })
    report["dense_improve_px"] = round(
        float(report["before_dense_contour_px"] - report["after_dense_contour_px"]),
        3,
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(
        "Free-face deform: "
        f"accepted={accepted}, moved={report['moved_vertices']}, "
        f"dense {report['before_dense_contour_px']:.2f}->{report['after_dense_contour_px']:.2f}px, "
        f"mean_offset={report['mean_offset_m']:.4f}m, max_offset={report['max_offset_m']:.4f}m"
    )
    if not accepted:
        return verts_base, verts_displaced, report
    return verts_base_out, verts_disp_out, report


def _shape_only_fine_tune(
    flame: FLAMEModel,
    shape_init: np.ndarray,
    identity_anchor_shape: Optional[np.ndarray],
    per_view_results: Dict[str, dict],
    view_data: Dict[str, dict],
    lmk_vertex_indices: np.ndarray,
    lmk_face_idx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    debug_dir: Path,
    device: str,
    max_iter: int = 80,
    lr: float = 0.03,
    contour_scale: float = 1.5,
    stable_anchor_weight: float = 1.8,
    delta_weight: float = 0.05,
    max_param_delta: float = 0.35,
    min_contour_improve_px: float = 0.05,
    max_stable_worsen_px: float = 2.0,
    max_total_worsen_px: float = 1.5,
    front_contour_weight: float = 2.4,
    front_jaw_weight: float = 3.2,
    side_contour_weight: float = 1.2,
    side_jaw_weight: float = 1.8,
    enable_dense_contour: bool = True,
    dense_contour_weight: float = 4.0,
    dense_row_step: int = 6,
    dense_row_sigma: float = 7.0,
    dense_softmin_tau: float = 10.0,
    dense_boundary_band_px: float = 90.0,
    dense_search_margin_px: float = 70.0,
    min_dense_contour_improve_px: float = 0.25,
    max_contour_worsen_px: float = 1.5,
    enable_silhouette: bool = False,
    silhouette_resolution: int = 256,
    silhouette_weight: float = 0.25,
    silhouette_sdf_weight: float = 2.0,
    silhouette_sdf_side_scale: float = 0.5,
    silhouette_dice_weight: float = 1.0,
    silhouette_l1_weight: float = 0.5,
    min_silhouette_improve_pct: float = 0.10,
    max_silhouette_view_worsen_pct: float = 0.15,
    min_silhouette_improved_views: int = 2,
    max_silhouette_overlap_drop: float = 0.005,
    max_interior_mean_worsen_pct: float = 0.15,
    max_interior_view_worsen_pct: float = 0.25,
    min_relative_boundary_improve: float = 0.0,
    min_front_relative_boundary_improve: float = 0.0,
    identity_thresholds: Optional[IdentityDriftThresholds] = None,
) -> Tuple[np.ndarray, dict]:
    """Fine-tune only shared shape while keeping each view's pose/expression fixed."""
    report = {
        "enabled": bool(max_iter > 0),
        "accepted": False,
        "reason": "",
        "before": {},
        "after": {},
    }
    if max_iter <= 0 or not per_view_results:
        report["reason"] = "disabled"
        return shape_init, report

    out_dir = debug_dir / "shape_only_fine_tune"
    out_dir.mkdir(parents=True, exist_ok=True)

    flame = flame.to(device)
    model_device = flame.v_template.device
    model_dtype = flame.v_template.dtype

    shape_anchor = torch.tensor(shape_init, device=device, dtype=torch.float32)
    shape_param = shape_anchor.clone().detach().requires_grad_(True)
    identity_anchor_np = np.asarray(
        identity_anchor_shape if identity_anchor_shape is not None else shape_init,
        dtype=np.float32,
    )
    identity_thresholds = identity_thresholds or IdentityDriftThresholds()
    view_names = [name for name in per_view_results.keys() if name in view_data]

    faces_np = flame.faces.detach().cpu().numpy()
    lmk_tri_vidx_np = None
    if lmk_face_idx is not None and lmk_bary_coords is not None:
        lmk_tri_vidx_np = faces_np[lmk_face_idx]
        use_bary = True
        lmk_v0 = torch.tensor(lmk_tri_vidx_np[:, 0], dtype=torch.long, device=device)
        lmk_v1 = torch.tensor(lmk_tri_vidx_np[:, 1], dtype=torch.long, device=device)
        lmk_v2 = torch.tensor(lmk_tri_vidx_np[:, 2], dtype=torch.long, device=device)
        bary = torch.tensor(lmk_bary_coords, dtype=torch.float32, device=device)
        bary_w0 = bary[:, 0:1]
        bary_w1 = bary[:, 1:2]
        bary_w2 = bary[:, 2:3]
    else:
        use_bary = False
        lmk_idx = torch.tensor(lmk_vertex_indices, dtype=torch.long, device=device)

    def torch_lmk_proj(proj: torch.Tensor) -> torch.Tensor:
        if use_bary:
            return proj[lmk_v0] * bary_w0 + proj[lmk_v1] * bary_w1 + proj[lmk_v2] * bary_w2
        return proj[lmk_idx]

    Ks = {
        name: torch.tensor(view_data[name]["K"], device=device, dtype=torch.float32)
        for name in view_names
    }
    targets = {
        name: torch.tensor(view_data[name]["lmk_2d"], device=device, dtype=torch.float32)
        for name in view_names
    }
    frozen = {}
    for name in view_names:
        res = per_view_results[name]
        frozen[name] = {
            "exp": torch.tensor(res["exp"], device=device, dtype=torch.float32),
            "R": torch.tensor(res["R"], device=device, dtype=torch.float32),
            "t": torch.tensor(res["t"], device=device, dtype=torch.float32),
        }

    neutral_exp = torch.zeros(flame.n_exp, device=model_device, dtype=model_dtype)
    with torch.no_grad():
        identity_anchor_vertices = flame(
            torch.tensor(identity_anchor_np, device=model_device, dtype=model_dtype),
            neutral_exp,
        ).detach().cpu().numpy()
        refinement_baseline_vertices = flame(
            shape_anchor.to(device=model_device, dtype=model_dtype), neutral_exp
        ).detach().cpu().numpy()
    refinement_baseline_quality = compute_mesh_quality(
        refinement_baseline_vertices, faces_np, label="shape_refinement_baseline"
    )

    def evaluate_candidate_quality(shape_np: np.ndarray, label: str):
        with torch.no_grad():
            candidate_vertices = flame(
                torch.tensor(shape_np, device=model_device, dtype=model_dtype),
                neutral_exp,
            ).detach().cpu().numpy()
        identity_gate = make_identity_drift_gate(
            anchor_shape=identity_anchor_np,
            candidate_shape=shape_np,
            anchor_vertices=identity_anchor_vertices,
            candidate_vertices=candidate_vertices,
            thresholds=identity_thresholds,
        )
        candidate_mesh_quality = compute_mesh_quality(
            candidate_vertices, faces_np, label=label
        )
        mesh_quality_gate = make_quality_gate(
            baseline=refinement_baseline_quality,
            candidate=candidate_mesh_quality,
            thresholds=MeshQualityThresholds(
                min_face_ratio=1.0,
                max_new_degenerate_faces=0,
                max_new_nonmanifold_edges=0,
                max_new_boundary_edges=0,
            ),
            region_name="shape_refinement",
        )
        return identity_gate, mesh_quality_gate

    silhouette_targets = {}
    silhouette_tensors = {}
    silhouette_sdf_tensors = {}
    silhouette_context = None
    silhouette_faces = flame.faces.to(device=device, dtype=torch.int32).contiguous()
    if enable_silhouette:
        try:
            silhouette_context = create_cuda_raster_context(device)
            for name in view_names:
                mask = preprocessed_views.get(name, {}).get("shape_mask")
                if mask is None:
                    mask = preprocessed_views.get(name, {}).get("face_mask")
                if mask is None:
                    continue
                target = build_silhouette_target(
                    mask,
                    view_name=name,
                    resolution=int(silhouette_resolution),
                )
                silhouette_targets[name] = target
                silhouette_tensors[name] = target.tensors(device)
                silhouette_sdf_tensors[name] = target.sdf_tensors(device)
        except Exception as exc:
            report["reason"] = f"silhouette initialization failed: {exc}"
            report["silhouette_error"] = repr(exc)
            logger.exception("Differentiable silhouette initialization failed")
            return np.asarray(shape_init, dtype=np.float32), report
        if len(silhouette_targets) < 2:
            report["reason"] = "fewer than two usable silhouette targets"
            report["silhouette_views"] = list(silhouette_targets)
            return np.asarray(shape_init, dtype=np.float32), report
        report["silhouette_views"] = list(silhouette_targets)
        report["silhouette_targets"] = {
            name: target.metadata for name, target in silhouette_targets.items()
        }

    stable_idx_np = np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
    stable_idx_t = torch.tensor(stable_idx_np, device=device)

    target_weights = {}
    anchor_weights = {}
    for name in view_names:
        tw = make_interior_landmark_weights(
            point_count=68,
            base_weight=0.25,
            stable_indices=[],
            stable_weight=0.0,
            device=device,
        )
        aw = torch.zeros(68, device=device, dtype=torch.float32)
        aw[stable_idx_t] = float(stable_anchor_weight)
        target_weights[name] = tw
        anchor_weights[name] = aw

    baseline_proj = {}
    with torch.no_grad():
        for name in view_names:
            verts0 = flame(shape_anchor, frozen[name]["exp"])
            proj0 = project_vertices(verts0, Ks[name], frozen[name]["R"], frozen[name]["t"])
            baseline_proj[name] = torch_lmk_proj(proj0).detach()

    dense_contours = {}
    if enable_dense_contour and dense_contour_weight > 0 and not silhouette_targets:
        dense_contours = _build_shape_only_dense_contours(
            flame=flame,
            shape_anchor=shape_anchor,
            frozen=frozen,
            view_names=view_names,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            device=device,
            row_step=dense_row_step,
            row_sigma=dense_row_sigma,
            tau=dense_softmin_tau,
            boundary_band_px=dense_boundary_band_px,
            search_margin_px=dense_search_margin_px,
        )
        if not dense_contours:
            logger.warning("Shape-only dense contour enabled but no usable dense contour rows were built.")

    optimizer = torch.optim.Adam([shape_param], lr=lr)
    shape_checkpoints = []
    for step in range(max_iter):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for name in view_names:
            verts = flame(shape_param, frozen[name]["exp"])
            proj = project_vertices(verts, Ks[name], frozen[name]["R"], frozen[name]["t"])
            lmk_proj = torch_lmk_proj(proj)
            lmk_proj_n = lmk_proj / 1000.0
            target_n = targets[name] / 1000.0
            anchor_n = baseline_proj[name] / 1000.0

            target_loss = F.smooth_l1_loss(
                lmk_proj_n,
                target_n,
                reduction="none",
                beta=0.01,
            ).mean(dim=1)
            target_loss = (target_loss * target_weights[name]).sum() / target_weights[name].sum().clamp_min(1e-6)

            anchor_loss = F.smooth_l1_loss(
                lmk_proj_n,
                anchor_n,
                reduction="none",
                beta=0.006,
            ).mean(dim=1)
            anchor_loss = (anchor_loss * anchor_weights[name]).sum() / anchor_weights[name].sum().clamp_min(1e-6)
            total_loss = total_loss + target_loss + anchor_loss

            dense = dense_contours.get(name)
            if dense is not None:
                dense_loss = _dense_contour_loss_torch(proj, dense)
                if torch.isfinite(dense_loss):
                    total_loss = total_loss + float(dense_contour_weight) * dense_loss

            silhouette_target = silhouette_targets.get(name)
            if silhouette_target is not None:
                target_mask, reliability = silhouette_tensors[name]
                prediction = render_soft_silhouette(
                    vertices=verts,
                    faces=silhouette_faces,
                    K=Ks[name],
                    R=frozen[name]["R"],
                    t=frozen[name]["t"],
                    image_shape=silhouette_target.source_shape,
                    render_shape=silhouette_target.render_shape,
                    context=silhouette_context,
                )
                mask_loss = weighted_silhouette_loss(
                    prediction,
                    target_mask,
                    reliability,
                    dice_weight=float(silhouette_dice_weight),
                    l1_weight=float(silhouette_l1_weight),
                )
                if torch.isfinite(mask_loss):
                    total_loss = total_loss + float(silhouette_weight) * mask_loss
                signed_distance, sdf_support = silhouette_sdf_tensors[name]
                sdf_loss = signed_distance_boundary_loss(
                    prediction,
                    signed_distance,
                    sdf_support,
                )
                if torch.isfinite(sdf_loss):
                    sdf_view_scale = (
                        1.0 if name == "front" else float(silhouette_sdf_side_scale)
                    )
                    total_loss = (
                        total_loss
                        + float(silhouette_sdf_weight) * sdf_view_scale * sdf_loss
                    )

        delta = shape_param - shape_anchor
        total_loss = total_loss + float(delta_weight) * (delta ** 2).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([shape_param], max_norm=0.5)
        optimizer.step()
        if max_param_delta > 0:
            with torch.no_grad():
                delta = (shape_param - shape_anchor).clamp(-float(max_param_delta), float(max_param_delta))
                shape_param.copy_(shape_anchor + delta)
        if step == 0 or (step + 1) % 20 == 0:
            logger.info(f"  shape-only step {step + 1}/{max_iter}, loss={float(total_loss.detach().cpu()):.6f}")
        if (step + 1) % 5 == 0 or step + 1 == max_iter:
            shape_checkpoints.append({
                "step": int(step + 1),
                "shape": shape_param.detach().cpu().numpy().astype(np.float32).copy(),
            })

    shape_final = shape_param.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "shape_final_candidate.npy", shape_final)

    def evaluate_shape(shape_np: np.ndarray, image_prefix: str, save_debug: bool = True) -> dict:
        records = []
        for name in view_names:
            res = per_view_results[name]
            with torch.no_grad():
                verts_np = flame(
                    torch.tensor(shape_np, device=model_device, dtype=model_dtype),
                    torch.tensor(res["exp"], device=model_device, dtype=model_dtype),
                ).detach().cpu().numpy()
            if save_debug:
                mean_err, max_err, errors = _save_landmark_reprojection_debug(
                    vertices=verts_np,
                    K=intrinsics[name],
                    R=res["R"],
                    t=res["t"],
                    image=preprocessed_views[name]["image"],
                    target_landmarks=view_data[name]["lmk_2d"],
                    lmk_vertex_indices=lmk_vertex_indices,
                    out_path=out_dir / f"{name}_{image_prefix}_reprojection.png",
                    lmk_tri_vidx=lmk_tri_vidx_np,
                    lmk_bary_coords=lmk_bary_coords,
                    return_errors=True,
                )
            else:
                _lmk_proj, errors = _landmark_reprojection_details(
                    vertices=verts_np,
                    K=intrinsics[name],
                    R=res["R"],
                    t=res["t"],
                    target_landmarks=view_data[name]["lmk_2d"],
                    lmk_vertex_indices=lmk_vertex_indices,
                    lmk_tri_vidx=lmk_tri_vidx_np,
                    lmk_bary_coords=lmk_bary_coords,
                )
                mean_err = float(errors.mean())
                max_err = float(errors.max())
            dense_metric = None
            dense = dense_contours.get(name)
            if dense is not None:
                v_cam = (res["R"] @ verts_np.T + res["t"][:, None]).T
                z = np.clip(v_cam[:, 2], 1e-6, None)
                v_hom = (intrinsics[name] @ v_cam.T).T
                proj_np = np.stack([v_hom[:, 0] / z, v_hom[:, 1] / z], axis=1)
                dense_metric, left_pred, right_pred = _dense_contour_metric_np(proj_np, dense)
                if save_debug:
                    _save_dense_contour_debug_image(
                        image=preprocessed_views[name]["image"],
                        dense=dense,
                        left_pred=left_pred,
                        right_pred=right_pred,
                        out_path=out_dir / f"{name}_{image_prefix}_dense_contour.png",
                    )
            silhouette_record = None
            silhouette_target = silhouette_targets.get(name)
            if silhouette_target is not None:
                with torch.no_grad():
                    verts_t = flame(
                        torch.tensor(shape_np, device=model_device, dtype=model_dtype),
                        torch.tensor(res["exp"], device=model_device, dtype=model_dtype),
                    )
                    prediction_t = render_soft_silhouette(
                        vertices=verts_t,
                        faces=silhouette_faces,
                        K=Ks[name],
                        R=frozen[name]["R"],
                        t=frozen[name]["t"],
                        image_shape=silhouette_target.source_shape,
                        render_shape=silhouette_target.render_shape,
                        context=silhouette_context,
                    )
                prediction_np = prediction_t.detach().cpu().numpy()
                silhouette_record = silhouette_metrics(prediction_np, silhouette_target)
                if save_debug:
                    save_silhouette_debug(
                        str(out_dir / f"{name}_{image_prefix}_silhouette.png"),
                        preprocessed_views[name]["image"],
                        prediction_np,
                        silhouette_target,
                    )
            records.append({
                "view": name,
                "mean_px": round(float(mean_err), 3),
                "interior_mean_px": _landmark_subset_stats(
                    errors, np.arange(17, 68, dtype=np.int64)
                )["mean_px"],
                "max_px": round(float(max_err), 3),
                "legacy_fixed_contour_diagnostic_px": _landmark_subset_stats(
                    errors, LMK_CONTOUR_IDX
                )["mean_px"],
                "jaw_mean_px": _landmark_subset_stats(errors, np.arange(4, 13, dtype=np.int64))["mean_px"],
                "stable_mean_px": _landmark_subset_stats(errors, stable_idx_np)["mean_px"],
                "dense_contour_mean_px": round(float(dense_metric), 3) if dense_metric is not None else None,
                "silhouette": silhouette_record,
            })
        if not records:
            return {
                "records": [],
                "mean_px": float("inf"),
                "interior_mean_px": float("inf"),
                "legacy_fixed_contour_diagnostic_px": float("inf"),
                "jaw_mean_px": float("inf"),
                "stable_mean_px": float("inf"),
                "dense_contour_mean_px": None,
                "silhouette_boundary_mean_px": None,
                "silhouette_boundary_mean_pct": None,
                "silhouette_overlap_dice": None,
            }
        dense_vals = [
            float(r["dense_contour_mean_px"])
            for r in records
            if r.get("dense_contour_mean_px") is not None
        ]
        silhouette_values = [
            float(r["silhouette"]["boundary_mean_px"])
            for r in records
            if r.get("silhouette") is not None
            and np.isfinite(float(r["silhouette"]["boundary_mean_px"]))
        ]
        silhouette_pct_values = [
            float(r["silhouette"]["trusted_boundary_face_width_pct"])
            for r in records
            if r.get("silhouette") is not None
            and np.isfinite(float(r["silhouette"]["trusted_boundary_face_width_pct"]))
        ]
        overlap_values = [
            float(r["silhouette"]["trusted_region_dice"])
            for r in records
            if r.get("silhouette") is not None
            and np.isfinite(float(r["silhouette"]["trusted_region_dice"]))
        ]
        return {
            "records": records,
            "mean_px": round(float(np.mean([r["mean_px"] for r in records])), 3),
            "interior_mean_px": round(
                float(np.mean([r["interior_mean_px"] for r in records])), 3
            ),
            "legacy_fixed_contour_diagnostic_px": round(
                float(np.mean([r["legacy_fixed_contour_diagnostic_px"] for r in records])), 3
            ),
            "jaw_mean_px": round(float(np.mean([r["jaw_mean_px"] for r in records])), 3),
            "stable_mean_px": round(float(np.mean([r["stable_mean_px"] for r in records])), 3),
            "dense_contour_mean_px": round(float(np.mean(dense_vals)), 3) if dense_vals else None,
            "silhouette_boundary_mean_px": (
                round(float(np.mean(silhouette_values)), 3) if silhouette_values else None
            ),
            "silhouette_boundary_mean_pct": (
                round(float(np.mean(silhouette_pct_values)), 6)
                if silhouette_pct_values else None
            ),
            "silhouette_overlap_dice": (
                round(float(np.mean(overlap_values)), 6) if overlap_values else None
            ),
        }

    before = evaluate_shape(np.asarray(shape_init, dtype=np.float32), "before")
    final_candidate = evaluate_shape(shape_final, "candidate_final")
    report["final_candidate"] = final_candidate
    selected_step = 0
    checkpoint_trials = []
    accepted_trials = []
    render_tolerances = [
        float(record["silhouette"]["render_pixel_face_width_pct"])
        for record in before.get("records", [])
        if record.get("silhouette") is not None
        and np.isfinite(float(record["silhouette"]["render_pixel_face_width_pct"]))
    ]
    silhouette_observation_tolerance = (
        float(np.mean(render_tolerances)) if render_tolerances else 0.0
    )
    if silhouette_targets:
        for checkpoint in shape_checkpoints:
            checkpoint_metrics = evaluate_shape(
                checkpoint["shape"], f"step_{checkpoint['step']}", save_debug=False
            )
            identity_gate, mesh_quality_gate = evaluate_candidate_quality(
                checkpoint["shape"], f"shape_refinement_step_{checkpoint['step']}"
            )
            checkpoint_decision = evaluate_geometry_candidate(
                before["records"],
                checkpoint_metrics["records"],
                mesh_quality_gate=mesh_quality_gate,
                min_boundary_improve_pct=float(min_silhouette_improve_pct),
                min_improved_views=int(min_silhouette_improved_views),
                max_view_worsen_pct=float(max_silhouette_view_worsen_pct),
                max_overlap_drop=float(max_silhouette_overlap_drop),
                max_interior_mean_worsen_pct=float(max_interior_mean_worsen_pct),
                max_interior_view_worsen_pct=float(max_interior_view_worsen_pct),
                min_relative_boundary_improve=float(min_relative_boundary_improve),
                min_front_relative_boundary_improve=float(
                    min_front_relative_boundary_improve
                ),
            )
            accepted = bool(
                checkpoint_decision["accepted"]
                and identity_gate["passed"]
            )
            failed_gates = list(checkpoint_decision["failed_gates"])
            if not identity_gate["passed"]:
                failed_gates.append("identity_preservation")
            trial = {
                "attempt": checkpoint["step"],
                "step": checkpoint["step"],
                "accepted": accepted,
                "failed_gates": failed_gates,
                "metrics": checkpoint_decision["metrics"],
                "observation_score_px": checkpoint_metrics["silhouette_boundary_mean_pct"],
                "identity_gate": identity_gate,
                "mesh_quality_gate": mesh_quality_gate,
                "shape": checkpoint["shape"],
            }
            checkpoint_trials.append(trial)
            if accepted:
                accepted_trials.append(trial)
    if accepted_trials:
        selected_trial = select_stable_refinement_checkpoint(
            accepted_trials,
            observation_tolerance=silhouette_observation_tolerance,
        )
        shape_after = selected_trial["shape"]
        selected_step = int(selected_trial["step"])
    elif not silhouette_targets:
        shape_after = shape_final
        selected_step = int(max_iter)
    else:
        shape_after = np.asarray(shape_init, dtype=np.float32)
    after = evaluate_shape(shape_after, "after")
    np.save(out_dir / "shape_selected.npy", shape_after)
    report["before"] = before
    report["after"] = after
    report["checkpoint_selection"] = {
        "selected_step": selected_step,
        "accepted_checkpoint_count": len(accepted_trials),
        "observation_tolerance_face_width_pct": silhouette_observation_tolerance,
        "trials": [
            {key: value for key, value in trial.items() if key != "shape"}
            for trial in checkpoint_trials
        ],
    }

    stable_worsen = float(after["stable_mean_px"] - before["stable_mean_px"])
    interior_worsen = float(after["interior_mean_px"] - before["interior_mean_px"])
    before_dense = before.get("dense_contour_mean_px")
    after_dense = after.get("dense_contour_mean_px")
    dense_improve = 0.0
    if before_dense is not None and after_dense is not None:
        dense_improve = float(before_dense - after_dense)
    before_silhouette = before.get("silhouette_boundary_mean_px")
    after_silhouette = after.get("silhouette_boundary_mean_px")

    identity_gate, mesh_quality_gate = evaluate_candidate_quality(
        shape_after, "shape_refinement_candidate"
    )
    if silhouette_targets:
        decision = evaluate_geometry_candidate(
            before["records"],
            after["records"],
            mesh_quality_gate=mesh_quality_gate,
            min_boundary_improve_pct=float(min_silhouette_improve_pct),
            min_improved_views=int(min_silhouette_improved_views),
            max_view_worsen_pct=float(max_silhouette_view_worsen_pct),
            max_overlap_drop=float(max_silhouette_overlap_drop),
            max_interior_mean_worsen_pct=float(max_interior_mean_worsen_pct),
            max_interior_view_worsen_pct=float(max_interior_view_worsen_pct),
            min_relative_boundary_improve=float(min_relative_boundary_improve),
            min_front_relative_boundary_improve=float(
                min_front_relative_boundary_improve
            ),
        )
        decision.setdefault("gates", {})["identity_preservation"] = bool(
            identity_gate["passed"]
        )
        if not identity_gate["passed"]:
            decision.setdefault("failed_gates", []).append("identity_preservation")
        accepted = bool(decision["accepted"] and identity_gate["passed"])
        decision["accepted"] = accepted
        reason = str(decision["reason"])
        if not identity_gate["passed"]:
            reason = "rejected by MICA identity preservation gate"
            decision["reason"] = reason
        report["geometry_decision"] = decision
    else:
        accepted = (
            dense_improve >= float(min_dense_contour_improve_px)
            and stable_worsen <= float(max_stable_worsen_px)
            and interior_worsen <= float(max_total_worsen_px)
            and identity_gate["passed"]
            and mesh_quality_gate["passed"]
        )
        reason = "dense silhouette improved with frozen pose" if accepted else "rejected by acceptance gate"
    report.update({
        "accepted": bool(accepted),
        "identity_drift": identity_gate,
        "dense_contour_improve_px": round(dense_improve, 3),
        "silhouette_improve_px": (
            round(float(before_silhouette - after_silhouette), 3)
            if before_silhouette is not None and after_silhouette is not None else None
        ),
        "silhouette_improve_pct_points": (
            report.get("geometry_decision", {})
            .get("metrics", {})
            .get("trusted_boundary_improve_pct_points")
        ),
        "stable_worsen_px": round(stable_worsen, 3),
        "interior_worsen_px": round(interior_worsen, 3),
        "shape_delta_norm": round(float(np.linalg.norm(shape_after - shape_init)), 6),
        "reason": reason,
    })

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Shape-only fine tune: "
        f"accepted={report['accepted']}, "
        f"silhouette {before_silhouette}->{after_silhouette}px, "
        f"dense {before_dense}->{after_dense}px, "
        f"stable {before['stable_mean_px']:.2f}->{after['stable_mean_px']:.2f}px, "
        f"interior {before['interior_mean_px']:.2f}->{after['interior_mean_px']:.2f}px"
    )
    if accepted:
        return shape_after, report
    return np.asarray(shape_init, dtype=np.float32), report


# ══════════════════════════════════════════════════════════════════════════════
# Depth-Anything-V2 深度置换
# ══════════════════════════════════════════════════════════════════════════════

def _controlled_low_frequency_identity_tune(
    *,
    baseline_final_vertices: np.ndarray,
    baseline_neutral_vertices: np.ndarray,
    per_view_vertices: Dict[str, np.ndarray],
    faces: np.ndarray,
    per_view_results: Dict[str, dict],
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    debug_dir: Path,
    device: str,
    settings: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fit a shared 16-parameter identity field with fixed cameras and expression."""
    cfg = dict(settings or {})
    out_dir = debug_dir / "controlled_identity_deformation"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_final = np.asarray(baseline_final_vertices, dtype=np.float32)
    baseline_neutral = np.asarray(baseline_neutral_vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    report = {
        "enabled": True,
        "accepted": False,
        "reason": "",
        "parameterization": "semantic_low_frequency_16",
    }

    def landmarks_from_vertices(vertices_np: np.ndarray) -> np.ndarray:
        if lmk_tri_vidx is not None and lmk_bary_coords is not None:
            tri = np.asarray(vertices_np)[np.asarray(lmk_tri_vidx, dtype=np.int64)]
            return np.sum(tri * np.asarray(lmk_bary_coords, dtype=np.float32)[:, :, None], axis=1)
        return np.asarray(vertices_np)[np.asarray(lmk_vertex_indices, dtype=np.int64)]

    neutral_landmarks = landmarks_from_vertices(baseline_neutral)
    basis = build_low_frequency_identity_basis(
        baseline_neutral,
        faces_np,
        neutral_landmarks,
        config=LowFrequencyBasisConfig(
            width_sigma=float(cfg.get("width_sigma", 0.24)),
            protection_core_ratio=float(cfg.get("protection_core_ratio", 0.035)),
            protection_outer_ratio=float(cfg.get("protection_outer_ratio", 0.12)),
            width_displacement_ratio=float(cfg.get("width_displacement_ratio", 0.040)),
            depth_displacement_ratio=float(cfg.get("depth_displacement_ratio", 0.025)),
            chin_displacement_ratio=float(cfg.get("chin_displacement_ratio", 0.030)),
            max_vertex_displacement_ratio=float(cfg.get("max_vertex_displacement_ratio", 0.060)),
        ),
    )
    basis, active_parameter_names = restrict_low_frequency_identity_basis(
        basis,
        allowed_prefixes=("temple_", "cheekbone_", "cheek_"),
    )
    report["basis"] = {
        "parameter_count": len(basis.names),
        "parameter_names": list(basis.names),
        "symmetry_pairs": [list(pair) for pair in basis.symmetry_pairs],
        "protected_vertex_count": int(np.count_nonzero(basis.protected_mask)),
        "editable_vertex_count": int(np.count_nonzero(basis.editable_mask)),
        "observation_source": "full_flame_outer_silhouette",
        "identity_region_names": list(IDENTITY_SILHOUETTE_REGIONS),
        "active_parameter_names": list(active_parameter_names),
        "face_width": float(basis.face_width),
        "face_height": float(basis.face_height),
        "max_vertex_displacement": float(basis.max_vertex_displacement),
    }

    view_names = [
        name
        for name in ("front", "left", "right")
        if name in per_view_vertices
        and name in per_view_results
        and name in view_data
        and name in intrinsics
    ]
    if len(view_names) < 3:
        report["reason"] = f"requires front, left, and right views; got {view_names}"
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return baseline_final, baseline_neutral, report

    silhouette_targets = {}
    silhouette_tensors = {}
    silhouette_regional_sdf_tensors = {}
    observation_faces_np = faces_np
    report["basis"]["observation_face_count"] = int(len(observation_faces_np))
    report["basis"]["full_face_count"] = int(len(faces_np))
    try:
        silhouette_context = create_cuda_raster_context(device)
        for name in view_names:
            mask = preprocessed_views[name].get("shape_mask")
            if mask is None:
                mask = preprocessed_views[name].get("face_mask")
            target = build_silhouette_target(
                mask,
                view_name=name,
                resolution=int(cfg.get("render_resolution", 256)),
            )
            silhouette_targets[name] = target
            silhouette_tensors[name] = target.tensors(device)
            signed_distance, region_supports = target.regional_sdf_tensors(device)
            silhouette_regional_sdf_tensors[name] = (
                signed_distance,
                {
                    region: support
                    for region, support in region_supports.items()
                    if region in IDENTITY_SILHOUETTE_REGIONS
                },
            )
    except Exception as exc:
        report["reason"] = f"silhouette initialization failed: {exc}"
        report["error"] = repr(exc)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return baseline_final, baseline_neutral, report

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    baseline_neutral_t = torch.tensor(baseline_neutral, dtype=torch.float32, device=dev)
    per_view_t = {
        name: torch.tensor(per_view_vertices[name], dtype=torch.float32, device=dev)
        for name in view_names
    }
    Ks = {
        name: torch.tensor(intrinsics[name], dtype=torch.float32, device=dev)
        for name in view_names
    }
    Rs = {
        name: torch.tensor(per_view_results[name]["R"], dtype=torch.float32, device=dev)
        for name in view_names
    }
    ts = {
        name: torch.tensor(per_view_results[name]["t"], dtype=torch.float32, device=dev)
        for name in view_names
    }
    targets_2d = {
        name: torch.tensor(view_data[name]["lmk_2d"], dtype=torch.float32, device=dev)
        for name in view_names
    }
    observation_faces_t = torch.tensor(
        observation_faces_np, dtype=torch.int32, device=dev
    ).contiguous()
    if lmk_tri_vidx is not None and lmk_bary_coords is not None:
        lmk_tri_t = torch.tensor(lmk_tri_vidx, dtype=torch.long, device=dev)
        lmk_bary_t = torch.tensor(lmk_bary_coords, dtype=torch.float32, device=dev)

        def torch_landmarks(projected: torch.Tensor) -> torch.Tensor:
            return (projected[lmk_tri_t] * lmk_bary_t[:, :, None]).sum(dim=1)
    else:
        lmk_idx_t = torch.tensor(lmk_vertex_indices, dtype=torch.long, device=dev)

        def torch_landmarks(projected: torch.Tensor) -> torch.Tensor:
            return projected[lmk_idx_t]

    stable_indices = torch.tensor(
        np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX]),
        dtype=torch.long,
        device=dev,
    )
    mask_weight = float(cfg.get("mask_weight", 0.0))
    sdf_weight = float(cfg.get("sdf_weight", 3.0))
    sdf_side_scale = float(cfg.get("sdf_side_scale", 0.65))
    landmark_weight = float(cfg.get("landmark_weight", 0.25))

    def observation_loss(candidate_neutral: torch.Tensor) -> torch.Tensor:
        displacement = candidate_neutral - baseline_neutral_t
        total = torch.zeros((), dtype=torch.float32, device=dev)
        for name in view_names:
            candidate_view = per_view_t[name] + displacement
            prediction = render_soft_silhouette(
                vertices=candidate_view,
                faces=observation_faces_t,
                K=Ks[name],
                R=Rs[name],
                t=ts[name],
                image_shape=silhouette_targets[name].source_shape,
                render_shape=silhouette_targets[name].render_shape,
                context=silhouette_context,
            )
            target_mask, reliability = silhouette_tensors[name]
            mask_loss = weighted_silhouette_loss(
                prediction,
                target_mask,
                reliability,
                dice_weight=1.0,
                l1_weight=0.5,
            )
            signed_distance, region_supports = silhouette_regional_sdf_tensors[name]
            sdf_loss = regional_signed_distance_boundary_loss(
                prediction,
                signed_distance,
                region_supports,
            )
            projected = project_vertices(candidate_view, Ks[name], Rs[name], ts[name])
            projected_lmk = torch_landmarks(projected)
            face_width_px = float(silhouette_targets[name].metadata["face_width_source_px"])
            landmark_delta = (
                projected_lmk[stable_indices] - targets_2d[name][stable_indices]
            ) / max(face_width_px, 1.0)
            landmark_loss = F.smooth_l1_loss(
                landmark_delta,
                torch.zeros_like(landmark_delta),
                beta=0.01,
            )
            side_scale = 1.0 if name == "front" else sdf_side_scale
            total = total + mask_weight * mask_loss + sdf_weight * side_scale * sdf_loss
            total = total + landmark_weight * landmark_loss
        return total / float(len(view_names))

    optimization = optimize_low_frequency_identity(
        baseline_neutral,
        faces_np,
        basis,
        observation_loss,
        config=LowFrequencyOptimizationConfig(
            max_iterations=int(cfg.get("max_iterations", 120)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            checkpoint_interval=int(cfg.get("checkpoint_interval", 5)),
            observation_weight=float(cfg.get("observation_weight", 1.0)),
            coefficient_weight=float(cfg.get("coefficient_weight", 0.005)),
            symmetry_weight=float(cfg.get("symmetry_weight", 0.02)),
            edge_weight=float(cfg.get("edge_weight", 0.20)),
            laplacian_weight=float(cfg.get("laplacian_weight", 0.50)),
            gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
        ),
        device=str(dev),
    )

    def evaluate_displacement(
        displacement_np: np.ndarray,
        prefix: str,
        *,
        save_debug: bool,
    ) -> dict:
        records = []
        for name in view_names:
            candidate_view = np.asarray(per_view_vertices[name], dtype=np.float32) + displacement_np
            projected_lmk, errors = _landmark_reprojection_details(
                vertices=candidate_view,
                K=intrinsics[name],
                R=per_view_results[name]["R"],
                t=per_view_results[name]["t"],
                target_landmarks=view_data[name]["lmk_2d"],
                lmk_vertex_indices=lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )
            with torch.no_grad():
                candidate_t = per_view_t[name] + torch.tensor(
                    displacement_np, dtype=torch.float32, device=dev
                )
                prediction_t = render_soft_silhouette(
                    vertices=candidate_t,
                    faces=observation_faces_t,
                    K=Ks[name],
                    R=Rs[name],
                    t=ts[name],
                    image_shape=silhouette_targets[name].source_shape,
                    render_shape=silhouette_targets[name].render_shape,
                    context=silhouette_context,
                )
            prediction_np = prediction_t.detach().cpu().numpy()
            silhouette_record = silhouette_metrics(
                prediction_np, silhouette_targets[name]
            )
            if save_debug:
                _save_landmark_reprojection_debug(
                    vertices=candidate_view,
                    K=intrinsics[name],
                    R=per_view_results[name]["R"],
                    t=per_view_results[name]["t"],
                    image=preprocessed_views[name]["image"],
                    target_landmarks=view_data[name]["lmk_2d"],
                    lmk_vertex_indices=lmk_vertex_indices,
                    out_path=out_dir / f"{name}_{prefix}_reprojection.png",
                    lmk_tri_vidx=lmk_tri_vidx,
                    lmk_bary_coords=lmk_bary_coords,
                )
                save_silhouette_debug(
                    str(out_dir / f"{name}_{prefix}_silhouette.png"),
                    preprocessed_views[name]["image"],
                    prediction_np,
                    silhouette_targets[name],
                )
            records.append(
                {
                    "view": name,
                    "interior_mean_px": _landmark_subset_stats(
                        errors, np.arange(17, 68, dtype=np.int64)
                    )["mean_px"],
                    "stable_mean_px": _landmark_subset_stats(
                        errors, np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
                    )["mean_px"],
                    "silhouette": silhouette_record,
                }
            )
        return {"records": records}

    baseline_metrics = evaluate_displacement(
        np.zeros_like(baseline_neutral), "baseline", save_debug=True
    )
    baseline_quality = compute_mesh_quality(
        baseline_neutral, faces_np, label="controlled_identity_baseline"
    )
    safety_thresholds = LowFrequencySafetyThresholds(
        max_vertex_displacement_ratio=float(cfg.get("max_vertex_displacement_ratio", 0.060)),
        max_mean_displacement_ratio=float(cfg.get("max_mean_displacement_ratio", 0.025)),
        max_protected_displacement_ratio=float(cfg.get("max_protected_displacement_ratio", 0.001)),
        max_new_normal_flips=int(cfg.get("max_new_normal_flips", 0)),
    )
    acceptance_thresholds = LowFrequencyAcceptanceThresholds(
        min_front_boundary_improvement_ratio=float(cfg.get("min_front_boundary_improvement_ratio", 0.30)),
        max_front_region_worsen_ratio=float(cfg.get("max_front_region_worsen_ratio", 0.10)),
        max_profile_boundary_worsen_ratio=float(cfg.get("max_profile_boundary_worsen_ratio", 0.10)),
        max_interior_landmark_worsen_ratio=float(cfg.get("max_interior_landmark_worsen_ratio", 0.10)),
        max_overlap_drop=float(cfg.get("max_overlap_drop", 0.01)),
    )

    def checkpoint_evaluator(checkpoint: dict) -> dict:
        displacement_np = np.asarray(checkpoint["displacement"], dtype=np.float32)
        candidate_neutral = baseline_neutral + displacement_np
        candidate_metrics = evaluate_displacement(
            displacement_np, f"step_{checkpoint['step']}", save_debug=False
        )
        safety = evaluate_low_frequency_safety(
            baseline_neutral,
            candidate_neutral,
            faces_np,
            basis,
            thresholds=safety_thresholds,
        )
        candidate_quality = compute_mesh_quality(
            candidate_neutral,
            faces_np,
            label=f"controlled_identity_step_{checkpoint['step']}",
        )
        quality_gate = make_quality_gate(
            baseline=baseline_quality,
            candidate=candidate_quality,
            thresholds=MeshQualityThresholds(
                min_face_ratio=1.0,
                max_new_degenerate_faces=0,
                max_new_nonmanifold_edges=0,
                max_new_boundary_edges=0,
            ),
            region_name="controlled_identity",
        )
        decision = evaluate_low_frequency_observations(
            baseline_metrics["records"],
            candidate_metrics["records"],
            safety_gate=safety,
            mesh_quality_gate=quality_gate,
            thresholds=acceptance_thresholds,
        )
        decision["candidate_records"] = candidate_metrics["records"]
        return decision

    _selected_neutral, selected_trial, trials = select_low_frequency_checkpoint(
        baseline_neutral,
        optimization["checkpoints"],
        checkpoint_evaluator,
    )
    if selected_trial is None:
        selected_displacement = np.zeros_like(baseline_neutral)
        selected_coefficients = np.zeros(len(basis.names), dtype=np.float32)
        selected_step = 0
    else:
        selected_displacement = np.asarray(
            selected_trial["checkpoint"]["displacement"], dtype=np.float32
        )
        selected_coefficients = np.asarray(
            selected_trial["checkpoint"]["coefficients"], dtype=np.float32
        )
        selected_step = int(selected_trial["step"])

    final_checkpoint = optimization["checkpoints"][-1]
    final_candidate_displacement = np.asarray(
        final_checkpoint["displacement"], dtype=np.float32
    )
    final_candidate_coefficients = np.asarray(
        final_checkpoint["coefficients"], dtype=np.float32
    )
    final_candidate_metrics = evaluate_displacement(
        final_candidate_displacement, "candidate_final", save_debug=True
    )
    selected_metrics = evaluate_displacement(
        selected_displacement, "selected", save_debug=True
    )
    selected_final = baseline_final + selected_displacement
    selected_neutral = baseline_neutral + selected_displacement

    np.save(out_dir / "candidate_final_displacement.npy", final_candidate_displacement)
    np.save(out_dir / "candidate_final_coefficients.npy", final_candidate_coefficients)
    np.save(out_dir / "selected_displacement.npy", selected_displacement)
    np.save(out_dir / "selected_coefficients.npy", selected_coefficients)
    with open(out_dir / "coefficients.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_step": selected_step,
                "names": list(basis.names),
                "selected_values": selected_coefficients.tolist(),
                "candidate_final_values": final_candidate_coefficients.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    try:
        import trimesh as _controlled_trimesh

        _controlled_trimesh.Trimesh(
            vertices=baseline_final, faces=faces_np, process=False
        ).export(str(out_dir / "baseline_geometry.glb"))
        _controlled_trimesh.Trimesh(
            vertices=baseline_final + final_candidate_displacement,
            faces=faces_np,
            process=False,
        ).export(str(out_dir / "candidate_final_geometry.glb"))
        _controlled_trimesh.Trimesh(
            vertices=selected_final, faces=faces_np, process=False
        ).export(str(out_dir / "selected_geometry.glb"))
    except Exception as exc:
        report["geometry_export_error"] = repr(exc)

    front_name = "front"
    front_vertices = np.asarray(per_view_vertices[front_name], dtype=np.float32)
    v_cam = (
        per_view_results[front_name]["R"] @ front_vertices.T
        + per_view_results[front_name]["t"][:, None]
    ).T
    z = np.clip(v_cam[:, 2], 1e-6, None)
    hom = (intrinsics[front_name] @ v_cam.T).T
    projected = np.stack([hom[:, 0] / z, hom[:, 1] / z], axis=1)
    support_strength = np.linalg.norm(basis.vectors, axis=2).max(axis=0)
    support_strength /= max(float(support_strength.max()), 1e-12)
    support_image = preprocessed_views[front_name]["image"].copy()
    colors = cv2.applyColorMap(
        np.clip(np.round(support_strength * 255.0), 0, 255).astype(np.uint8)[:, None],
        cv2.COLORMAP_TURBO,
    )[:, 0]
    order = np.argsort(v_cam[:, 2])[::-1]
    h, w = support_image.shape[:2]
    for index in order:
        x, y = np.round(projected[index]).astype(int)
        if 0 <= x < w and 0 <= y < h and support_strength[index] > 0.02:
            cv2.circle(
                support_image,
                (x, y),
                1,
                tuple(int(value) for value in colors[index]),
                -1,
                lineType=cv2.LINE_AA,
            )
    cv2.imwrite(str(out_dir / "basis_support.png"), support_image)

    selected_summary = None
    if selected_trial is not None:
        selected_summary = {
            key: value for key, value in selected_trial.items() if key != "checkpoint"
        }
    report.update(
        {
            "accepted": selected_trial is not None,
            "reason": (
                "accepted controlled low-frequency identity candidate"
                if selected_trial is not None
                else "no checkpoint passed controlled identity gates; exact baseline retained"
            ),
            "selected_step": selected_step,
            "selected": selected_summary,
            "trials": trials,
            "history": optimization["history"],
            "baseline": baseline_metrics,
            "candidate_final": final_candidate_metrics,
            "after": selected_metrics,
        }
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(
        "Controlled low-frequency identity: accepted=%s, selected_step=%d",
        report["accepted"],
        selected_step,
    )
    return selected_final, selected_neutral, report


def _pose_refine_cameras(
    flame: FLAMEModel,
    shape_opt: np.ndarray,
    per_view_results: Dict[str, dict],
    view_data: Dict[str, dict],
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    lmk_vertex_indices: np.ndarray,
    lmk_face_idx: Optional[np.ndarray],
    lmk_bary_coords: Optional[np.ndarray],
    debug_dir: Path,
    device: str,
    enabled: bool = True,
    max_iter: int = 80,
    lr: float = 0.01,
    lmk_weight: float = 1.0,
    stable_weight: float = 1.4,
    dense_contour_weight: float = 0.35,
    rot_reg: float = 0.02,
    trans_reg: float = 0.02,
    min_mean_improve_px: float = 1.0,
    min_dense_improve_px: float = 3.0,
    max_mean_worsen_px: float = 1.0,
    max_stable_worsen_px: float = 1.0,
    max_maxerr_worsen_px: float = 5.0,
    max_rot_deg: float = 6.0,
    max_trans_rel: float = 0.08,
    global_max_overall_worsen_px: float = 0.75,
    global_max_front_worsen_px: float = 0.5,
    global_max_side_dense_worsen_px: float = 2.0,
) -> Tuple[Dict[str, dict], dict]:
    """Refine per-view camera pose only; keep shape, expression, mesh and UV fixed."""
    report = {
        "enabled": bool(enabled),
        "accepted": False,
        "applied_views": [],
        "view_records": [],
        "global_reject_reasons": [],
    }
    if not enabled or max_iter <= 0 or not per_view_results:
        report["reason"] = "disabled"
        return per_view_results, report

    out_dir = debug_dir / "pose_refinement"
    out_dir.mkdir(parents=True, exist_ok=True)

    flame = flame.to(device)
    model_device = flame.v_template.device
    model_dtype = flame.v_template.dtype
    shape_t = torch.tensor(shape_opt, device=model_device, dtype=model_dtype)
    faces_np = flame.faces.detach().cpu().numpy()
    lmk_tri_vidx_np = faces_np[lmk_face_idx] if lmk_face_idx is not None and lmk_bary_coords is not None else None
    view_names = [name for name in per_view_results.keys() if name in view_data and name in intrinsics]
    stable_idx_np = np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
    interior_idx_np = np.arange(17, 68, dtype=np.int64)

    def view_vertices(name: str) -> np.ndarray:
        exp = np.asarray(per_view_results[name]["exp"], dtype=np.float32)
        with torch.no_grad():
            verts = flame(shape_t, torch.tensor(exp, device=model_device, dtype=model_dtype))
        return verts.detach().cpu().numpy()

    def measure_view(name: str, verts_np: np.ndarray, R: np.ndarray, t: np.ndarray) -> dict:
        _lmk_proj, errors = _landmark_reprojection_details(
            vertices=verts_np,
            K=intrinsics[name],
            R=R,
            t=t,
            target_landmarks=view_data[name]["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx_np,
            lmk_bary_coords=lmk_bary_coords,
        )
        dense_metric = None
        dense = dense_contours.get(name)
        if dense is not None:
            proj_np, _ = _project_vertices_np(verts_np, intrinsics[name], R, t)
            dense_metric, _left_pred, _right_pred = _dense_contour_metric_np(proj_np, dense)
        return {
            "mean_px": float(np.asarray(errors)[interior_idx_np].mean()),
            "max_px": float(np.asarray(errors)[interior_idx_np].max()),
            "legacy_full_mean_px": float(errors.mean()),
            "stable_mean_px": float(np.asarray(errors)[stable_idx_np].mean()),
            "legacy_fixed_contour_diagnostic_px": float(
                np.asarray(errors)[LMK_CONTOUR_IDX].mean()
            ),
            "dense_contour_mean_px": None if dense_metric is None else float(dense_metric),
        }

    def rounded_metrics(metrics: dict) -> dict:
        return {
            "mean_px": round(float(metrics["mean_px"]), 3),
            "max_px": round(float(metrics["max_px"]), 3),
            "legacy_full_mean_px": round(float(metrics["legacy_full_mean_px"]), 3),
            "stable_mean_px": round(float(metrics["stable_mean_px"]), 3),
            "legacy_fixed_contour_diagnostic_px": round(
                float(metrics["legacy_fixed_contour_diagnostic_px"]), 3
            ),
            "dense_contour_mean_px": (
                None if metrics.get("dense_contour_mean_px") is None
                else round(float(metrics["dense_contour_mean_px"]), 3)
            ),
        }

    def evaluate_view(name: str, R: np.ndarray, t: np.ndarray, prefix: str) -> dict:
        verts_np = view_vertices(name)
        mean_err, max_err, errors = _save_landmark_reprojection_debug(
            vertices=verts_np,
            K=intrinsics[name],
            R=R,
            t=t,
            image=preprocessed_views[name]["image"],
            target_landmarks=view_data[name]["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=out_dir / f"{name}_{prefix}_reprojection.png",
            lmk_tri_vidx=lmk_tri_vidx_np,
            lmk_bary_coords=lmk_bary_coords,
            return_errors=True,
        )
        face_mask = preprocessed_views[name].get("face_mask")
        if face_mask is None:
            face_mask = preprocessed_views[name].get("shape_mask")
        if face_mask is not None:
            _save_projection_debug(
                verts_np,
                faces_np,
                intrinsics[name],
                R,
                t,
                preprocessed_views[name]["image"],
                face_mask,
                out_dir / f"{name}_{prefix}_mesh.png",
            )
        dense_metric = None
        dense = dense_contours.get(name)
        if dense is not None:
            proj_np, _ = _project_vertices_np(verts_np, intrinsics[name], R, t)
            dense_metric, left_pred, right_pred = _dense_contour_metric_np(proj_np, dense)
            _save_dense_contour_debug_image(
                image=preprocessed_views[name]["image"],
                dense=dense,
                left_pred=left_pred,
                right_pred=right_pred,
                out_path=out_dir / f"{name}_{prefix}_dense_contour.png",
            )
        return {
            "mean_px": _landmark_subset_stats(errors, interior_idx_np)["mean_px"],
            "max_px": round(float(np.asarray(errors)[interior_idx_np].max()), 3),
            "legacy_full_mean_px": round(float(mean_err), 3),
            "stable_mean_px": _landmark_subset_stats(errors, stable_idx_np)["mean_px"],
            "legacy_fixed_contour_diagnostic_px": _landmark_subset_stats(
                errors, LMK_CONTOUR_IDX
            )["mean_px"],
            "dense_contour_mean_px": round(float(dense_metric), 3) if dense_metric is not None else None,
        }

    frozen_for_dense = {}
    for name in view_names:
        res = per_view_results[name]
        frozen_for_dense[name] = {
            "exp": torch.tensor(res["exp"], device=device, dtype=torch.float32),
            "R": torch.tensor(res["R"], device=device, dtype=torch.float32),
            "t": torch.tensor(res["t"], device=device, dtype=torch.float32),
        }
    dense_contours = _build_shape_only_dense_contours(
        flame=flame,
        shape_anchor=torch.tensor(shape_opt, device=device, dtype=torch.float32),
        frozen=frozen_for_dense,
        view_names=view_names,
        view_data=view_data,
        preprocessed_views=preprocessed_views,
        device=device,
        row_step=6,
        row_sigma=7.0,
        tau=10.0,
        boundary_band_px=90.0,
        search_margin_px=70.0,
    )

    refined = {
        name: {
            "R": np.asarray(res["R"], dtype=np.float32).copy(),
            "t": np.asarray(res["t"], dtype=np.float32).copy(),
            "exp": np.asarray(res["exp"], dtype=np.float32).copy(),
        }
        for name, res in per_view_results.items()
    }

    for name in view_names:
        res = per_view_results[name]
        R0 = np.asarray(res["R"], dtype=np.float32)
        t0 = np.asarray(res["t"], dtype=np.float32)
        exp_t = torch.tensor(res["exp"], device=device, dtype=torch.float32)
        verts = flame(torch.tensor(shape_opt, device=device, dtype=torch.float32), exp_t).detach()
        verts_np = verts.detach().cpu().numpy()
        K_t = torch.tensor(intrinsics[name], device=device, dtype=torch.float32)
        target = torch.tensor(view_data[name]["lmk_2d"], device=device, dtype=torch.float32)
        r0 = Rotation.from_matrix(R0).as_rotvec().astype(np.float32)
        r_param = torch.tensor(r0, device=device, dtype=torch.float32, requires_grad=True)
        t_param = torch.tensor(t0, device=device, dtype=torch.float32, requires_grad=True)
        r_anchor = torch.tensor(r0, device=device, dtype=torch.float32)
        t_anchor = torch.tensor(t0, device=device, dtype=torch.float32)

        if lmk_tri_vidx_np is not None and lmk_bary_coords is not None:
            lmk_v0 = torch.tensor(lmk_tri_vidx_np[:, 0], dtype=torch.long, device=device)
            lmk_v1 = torch.tensor(lmk_tri_vidx_np[:, 1], dtype=torch.long, device=device)
            lmk_v2 = torch.tensor(lmk_tri_vidx_np[:, 2], dtype=torch.long, device=device)
            bary = torch.tensor(lmk_bary_coords, dtype=torch.float32, device=device)

            def lmk_from_proj(proj: torch.Tensor) -> torch.Tensor:
                return (
                    proj[lmk_v0] * bary[:, 0:1]
                    + proj[lmk_v1] * bary[:, 1:2]
                    + proj[lmk_v2] * bary[:, 2:3]
                )
        else:
            lmk_idx = torch.tensor(lmk_vertex_indices, dtype=torch.long, device=device)

            def lmk_from_proj(proj: torch.Tensor) -> torch.Tensor:
                return proj[lmk_idx]

        point_weights = make_interior_landmark_weights(
            point_count=68,
            base_weight=float(lmk_weight),
            stable_indices=stable_idx_np,
            stable_weight=float(stable_weight),
            device=device,
        )
        view_lr = float(lr) * (0.5 if name == "front" else 1.0)
        view_dense_weight = float(dense_contour_weight) * (0.25 if name == "front" else 1.0)
        optimizer = torch.optim.Adam([r_param, t_param], lr=view_lr)
        dense = dense_contours.get(name)
        before_quick = measure_view(name, verts_np, R0, t0)
        best_candidate = None
        best_score = -float("inf")

        def candidate_deltas(metrics: dict, R_candidate: np.ndarray, t_candidate: np.ndarray) -> dict:
            dense_improve_value = 0.0
            if (
                before_quick.get("dense_contour_mean_px") is not None
                and metrics.get("dense_contour_mean_px") is not None
            ):
                dense_improve_value = float(before_quick["dense_contour_mean_px"] - metrics["dense_contour_mean_px"])
            rot_delta = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(R_candidate @ R0.T).as_rotvec())))
            trans_delta = float(np.linalg.norm(t_candidate - t0) / max(abs(float(t0[2])), float(np.linalg.norm(t0)), 1e-6))
            return {
                "mean_improve": float(before_quick["mean_px"] - metrics["mean_px"]),
                "mean_worsen": float(metrics["mean_px"] - before_quick["mean_px"]),
                "stable_worsen": float(metrics["stable_mean_px"] - before_quick["stable_mean_px"]),
                "maxerr_worsen": float(metrics["max_px"] - before_quick["max_px"]),
                "dense_improve": dense_improve_value,
                "rot_delta_deg": rot_delta,
                "trans_rel": trans_delta,
            }

        for _step in range(int(max_iter)):
            optimizer.zero_grad()
            R_cur = rodrigues_to_matrix(r_param)
            proj = project_vertices(verts, K_t, R_cur, t_param)
            lmk_proj = lmk_from_proj(proj)
            per_point = F.smooth_l1_loss(
                lmk_proj / 1000.0,
                target / 1000.0,
                reduction="none",
                beta=0.01,
            ).mean(dim=1)
            loss = (per_point * point_weights).sum() / point_weights.sum().clamp_min(1e-6)
            if dense is not None and view_dense_weight > 0:
                dense_loss = _dense_contour_loss_torch(proj, dense)
                if torch.isfinite(dense_loss):
                    loss = loss + view_dense_weight * dense_loss
            loss = loss + float(rot_reg) * ((r_param - r_anchor) ** 2).mean()
            t_scale = torch.clamp(torch.abs(t_anchor[2]), min=0.05)
            loss = loss + float(trans_reg) * (((t_param - t_anchor) / t_scale) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([r_param, t_param], max_norm=0.25)
            optimizer.step()
            if (_step + 1) % 5 == 0 or (_step + 1) == int(max_iter):
                R_candidate = rodrigues_to_matrix(r_param.detach()).detach().cpu().numpy().astype(np.float32)
                t_candidate = t_param.detach().cpu().numpy().astype(np.float32)
                metrics = measure_view(name, verts_np, R_candidate, t_candidate)
                deltas = candidate_deltas(metrics, R_candidate, t_candidate)
                view_max_mean_worsen = float(max_mean_worsen_px) * (0.25 if name == "front" else 1.0)
                enough_improvement = (
                    deltas["mean_improve"] >= float(min_mean_improve_px)
                    or deltas["dense_improve"] >= float(min_dense_improve_px)
                )
                candidate_ok = (
                    enough_improvement
                    and deltas["mean_worsen"] <= view_max_mean_worsen
                    and deltas["stable_worsen"] <= float(max_stable_worsen_px)
                    and deltas["maxerr_worsen"] <= float(max_maxerr_worsen_px)
                    and deltas["rot_delta_deg"] <= float(max_rot_deg)
                    and deltas["trans_rel"] <= float(max_trans_rel)
                )
                score = (
                    deltas["dense_improve"]
                    + max(deltas["mean_improve"], 0.0) * 2.0
                    - max(deltas["mean_worsen"], 0.0) * 3.0
                    - max(deltas["stable_worsen"], 0.0) * 8.0
                )
                if candidate_ok and score > best_score:
                    best_score = score
                    best_candidate = {
                        "R": R_candidate.copy(),
                        "t": t_candidate.copy(),
                        "metrics": metrics,
                        "deltas": deltas,
                        "step": _step + 1,
                        "score": float(score),
                    }

        before = evaluate_view(name, R0, t0, "before")
        if best_candidate is not None:
            R1 = best_candidate["R"]
            t1 = best_candidate["t"]
        else:
            R1 = rodrigues_to_matrix(r_param.detach()).detach().cpu().numpy().astype(np.float32)
            t1 = t_param.detach().cpu().numpy().astype(np.float32)
        after = evaluate_view(name, R1, t1, "after")
        mean_improve = float(before["mean_px"] - after["mean_px"])
        mean_worsen = float(after["mean_px"] - before["mean_px"])
        stable_worsen = float(after["stable_mean_px"] - before["stable_mean_px"])
        maxerr_worsen = float(after["max_px"] - before["max_px"])
        dense_improve = 0.0
        if before.get("dense_contour_mean_px") is not None and after.get("dense_contour_mean_px") is not None:
            dense_improve = float(before["dense_contour_mean_px"] - after["dense_contour_mean_px"])
        rot_delta_deg = float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(R1 @ R0.T).as_rotvec())))
        trans_rel = float(np.linalg.norm(t1 - t0) / max(abs(float(t0[2])), float(np.linalg.norm(t0)), 1e-6))
        reasons = []
        if mean_improve < float(min_mean_improve_px) and dense_improve < float(min_dense_improve_px):
            reasons.append("insufficient_improvement")
        view_max_mean_worsen = float(max_mean_worsen_px) * (0.25 if name == "front" else 1.0)
        if mean_worsen > view_max_mean_worsen:
            reasons.append("mean_worsened")
        if stable_worsen > float(max_stable_worsen_px):
            reasons.append("stable_worsened")
        if maxerr_worsen > float(max_maxerr_worsen_px):
            reasons.append("max_error_worsened")
        if rot_delta_deg > float(max_rot_deg):
            reasons.append("rotation_too_large")
        if trans_rel > float(max_trans_rel):
            reasons.append("translation_too_large")
        accepted = not reasons
        record = {
            "view": name,
            "accepted": bool(accepted),
            "reject_reasons": reasons,
            "before": before,
            "after": after,
            "mean_improve_px": round(mean_improve, 3),
            "dense_improve_px": round(dense_improve, 3),
            "stable_worsen_px": round(stable_worsen, 3),
            "maxerr_worsen_px": round(maxerr_worsen, 3),
            "rot_delta_deg": round(rot_delta_deg, 3),
            "trans_rel": round(trans_rel, 5),
            "selected_step": None if best_candidate is None else int(best_candidate["step"]),
            "selected_score": None if best_candidate is None else round(float(best_candidate["score"]), 4),
        }
        report["view_records"].append(record)
        if accepted:
            refined[name]["R"] = R1
            refined[name]["t"] = t1
            report["applied_views"].append(name)
        logger.info(
            "Pose refine [%s]: accepted=%s mean %.2f->%.2fpx dense %s->%s rot=%.2fdeg trans_rel=%.4f",
            name,
            accepted,
            before["mean_px"],
            after["mean_px"],
            before.get("dense_contour_mean_px"),
            after.get("dense_contour_mean_px"),
            rot_delta_deg,
            trans_rel,
        )

    def aggregate(records: list, use_after: bool) -> dict:
        vals = []
        front = None
        side_dense_worsen = 0.0
        for rec in records:
            key = "after" if use_after and rec["accepted"] else "before"
            vals.append(float(rec[key]["mean_px"]))
            if rec["view"] == "front":
                front = float(rec[key]["mean_px"])
            if rec["view"] != "front":
                b = rec["before"].get("dense_contour_mean_px")
                a = rec[key].get("dense_contour_mean_px")
                if b is not None and a is not None:
                    side_dense_worsen = max(side_dense_worsen, float(a - b))
        return {
            "mean_px": round(float(np.mean(vals)) if vals else 0.0, 3),
            "front_mean_px": round(float(front), 3) if front is not None else None,
            "max_side_dense_worsen_px": round(float(side_dense_worsen), 3),
        }

    before_global = aggregate(report["view_records"], use_after=False)
    after_global = aggregate(report["view_records"], use_after=True)
    report["before_global"] = before_global
    report["after_global"] = after_global
    if after_global["mean_px"] - before_global["mean_px"] > float(global_max_overall_worsen_px):
        report["global_reject_reasons"].append("overall_landmarks_worsened")
    if (
        before_global["front_mean_px"] is not None
        and after_global["front_mean_px"] is not None
        and after_global["front_mean_px"] - before_global["front_mean_px"] > float(global_max_front_worsen_px)
    ):
        report["global_reject_reasons"].append("front_landmarks_worsened")
    if after_global["max_side_dense_worsen_px"] > float(global_max_side_dense_worsen_px):
        report["global_reject_reasons"].append("side_dense_contour_worsened")

    if report["global_reject_reasons"]:
        refined = per_view_results
        report["applied_views"] = []
        report["accepted"] = False
        report["reason"] = "global validation failed: " + ", ".join(report["global_reject_reasons"])
    else:
        report["accepted"] = bool(report["applied_views"])
        report["reason"] = "accepted pose refinement" if report["accepted"] else "no view accepted"

    _write_pose_refinement_index(out_dir, report)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return refined, report


def _write_pose_refinement_index(out_dir: Path, report: dict) -> None:
    import html

    def esc(value) -> str:
        return html.escape(str(value))

    rows = []
    figures = []
    for rec in report.get("view_records", []):
        view = rec["view"]
        rows.append(
            "<tr>"
            f"<td>{esc(view)}</td>"
            f"<td>{'yes' if rec.get('accepted') else 'no'}</td>"
            f"<td>{esc(rec['before']['mean_px'])}</td>"
            f"<td>{esc(rec['after']['mean_px'])}</td>"
            f"<td>{esc(rec.get('mean_improve_px'))}</td>"
            f"<td>{esc(rec['before'].get('dense_contour_mean_px'))}</td>"
            f"<td>{esc(rec['after'].get('dense_contour_mean_px'))}</td>"
            f"<td>{esc(', '.join(rec.get('reject_reasons', [])))}</td>"
            "</tr>"
        )
        for kind in ("reprojection", "mesh", "dense_contour"):
            before = out_dir / f"{view}_before_{kind}.png"
            after = out_dir / f"{view}_after_{kind}.png"
            if before.exists() and after.exists():
                figures.append(
                    "<section>"
                    f"<h2>{esc(view)} {esc(kind)}</h2>"
                    f"<figure><img src='{esc(before.name)}'><figcaption>before</figcaption></figure>"
                    f"<figure><img src='{esc(after.name)}'><figcaption>after</figcaption></figure>"
                    "</section>"
                )
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Pose refinement audit</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;background:#101827;color:#eef3ff;margin:24px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #314866;padding:8px}}
section{{margin:24px 0;padding:16px;background:#17243a;border-radius:10px}}
figure{{display:inline-block;width:48%;vertical-align:top;margin:0 1% 16px 0}}img{{width:100%;background:#000}}
.ok{{color:#4ade80}}.bad{{color:#fb7185}}
</style>
<h1>Pose refinement audit</h1>
<p>Accepted: <b class="{'ok' if report.get('accepted') else 'bad'}">{esc(report.get('accepted'))}</b></p>
<p>Reason: {esc(report.get('reason'))}</p>
<table><thead><tr><th>view</th><th>accepted</th><th>mean before</th><th>mean after</th><th>improve</th><th>dense before</th><th>dense after</th><th>reject reasons</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
{''.join(figures)}
"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def load_depth_model(model_dir: Path, model_size: str = "large", device: str = "cuda"):
    """Load Depth-Anything-V2, preferring a local snapshot directory."""
    try:
        from transformers import pipeline
        model_id = {
            "small":  "depth-anything/Depth-Anything-V2-Small-hf",
            "base":   "depth-anything/Depth-Anything-V2-Base-hf",
            "large":  "depth-anything/Depth-Anything-V2-Large-hf",
        }[model_size]

        local_candidates = [
            model_dir / f"Depth-Anything-V2-{model_size.title()}-hf",
            model_dir / model_size,
            model_dir,
        ]

        local_model_dir = None
        for cand in local_candidates:
            if (cand / "config.json").exists():
                local_model_dir = cand
                break

        pipe_kwargs = {
            "task": "depth-estimation",
            "device": 0 if device == "cuda" else -1,
        }
        if local_model_dir is not None:
            pipe_kwargs["model"] = str(local_model_dir)
            pipe_kwargs["local_files_only"] = True
            logger.info(f"Depth-Anything-V2-{model_size} 使用本地模型: {local_model_dir}")
        else:
            pipe_kwargs["model"] = model_id
            logger.info(f"Depth-Anything-V2-{model_size} 本地模型缺失，回退到 HuggingFace: {model_id}")

        pipe = pipeline(**pipe_kwargs)
        logger.info(f"Depth-Anything-V2-{model_size} 加载成功")
        return pipe
    except ImportError:
        raise ImportError("请安装 transformers: pip install transformers")
    except Exception as e:
        raise RuntimeError(f"Depth 模型加载失败: {e}")


def estimate_depth(depth_pipe, image: np.ndarray) -> np.ndarray:
    """返回归一化深度图 [0,1]，shape (H,W) float32"""
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image)
    result = depth_pipe(pil_img)
    depth = np.array(result["depth"], dtype=np.float32)
    # 归一化到 [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def render_mesh_depth(
    vertices: np.ndarray,     # (N, 3)
    faces: np.ndarray,        # (F, 3)
    K: np.ndarray,            # (3, 3)
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int],  # (H, W)
) -> np.ndarray:
    """
    软光栅化：将 Mesh 渲染为深度图（Z-buffer）。
    返回 (H, W) float32 深度图，无效区域为 np.inf。
    """
    H, W = image_shape
    depth_map = np.full((H, W), np.inf, dtype=np.float32)

    # 投影所有顶点
    v_cam = (R @ vertices.T + t[:, None]).T      # (N, 3)
    v_hom = (K @ v_cam.T).T                       # (N, 3)
    z     = v_hom[:, 2]                            # (N,)
    u     = v_hom[:, 0] / np.clip(z, 1e-6, None)  # (N,)
    v_px  = v_hom[:, 1] / np.clip(z, 1e-6, None)  # (N,)

    # 逐面片光栅化（CPU 简化版，Phase 2 可换 pyrender/nvdiffrast）
    for face in faces:
        i0, i1, i2 = face
        pts = np.array([[u[i0], v_px[i0]], [u[i1], v_px[i1]], [u[i2], v_px[i2]]])
        zs  = np.array([z[i0], z[i1], z[i2]])

        # 包围盒裁剪
        x_min = max(0, int(pts[:, 0].min()))
        x_max = min(W - 1, int(pts[:, 0].max()))
        y_min = max(0, int(pts[:, 1].min()))
        y_max = min(H - 1, int(pts[:, 1].max()))
        if x_min > x_max or y_min > y_max:
            continue

        # 重心坐标插值（简化版，适合稀疏面片）
        for px in range(x_min, x_max + 1):
            for py in range(y_min, y_max + 1):
                p = np.array([px + 0.5, py + 0.5])
                bary = _barycentric(p, pts[0], pts[1], pts[2])
                if np.all(bary >= 0):
                    z_interp = float(bary @ zs)
                    if z_interp < depth_map[py, px]:
                        depth_map[py, px] = z_interp

    return depth_map


def _barycentric(p, a, b, c):
    v0 = c - a; v1 = b - a; v2 = p - a
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
    d20 = v2 @ v0; d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([-1., -1., -1.])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def align_depth_scale(
    d_pred: np.ndarray,    # Depth-Anything 相对深度 [0,1]
    d_3dmm: np.ndarray,    # 渲染的3DMM伪深度图 (H,W) float32
    valid_mask: np.ndarray,# (H,W) bool
) -> Tuple[float, float]:
    """
    最小二乘估计 s, t 使得 d_aligned = s * d_pred + t ≈ d_3dmm
    只在有效（非inf）的3DMM深度区域内计算。
    """
    mask = valid_mask & np.isfinite(d_3dmm) & (d_pred > 0.01)
    if mask.sum() < 100:
        logger.warning("有效深度像素不足100，使用默认缩放 s=1, t=0")
        return 1.0, 0.0
    x = d_pred[mask].flatten()
    y = d_3dmm[mask].flatten()
    # 最小二乘: [x, 1] @ [s, t]^T = y
    A = np.stack([x, np.ones_like(x)], axis=1)
    result = np.linalg.lstsq(A, y, rcond=None)
    s, t = float(result[0][0]), float(result[0][1])
    logger.info(f"  深度尺度对齐: s={s:.4f}, t={t:.4f}")
    return s, t


def project_vertices_sparse(
    vertices: np.ndarray,      # (N, 3)
    K: np.ndarray,             # (3, 3)
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 FLAME 顶点投影到图像空间，返回每个顶点的深度值和像素坐标。
    比全 Mesh 光栅化快100倍以上。

    Returns:
        z_vals: (M,) 相机空间深度
        px_u:   (M,) 像素 x
        px_v:   (M,) 像素 y
        （仅保留在图像范围内且 z>0 的顶点）
    """
    H, W = image_shape
    v_cam = (R @ vertices.T + t[:, None]).T           # (N, 3)
    z     = v_cam[:, 2]
    valid = z > 1e-4
    v_cam = v_cam[valid]
    z     = z[valid]

    v_hom = (K @ v_cam.T).T                           # (M, 3)
    u     = (v_hom[:, 0] / z).astype(int)
    v     = (v_hom[:, 1] / z).astype(int)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return z[in_bounds], u[in_bounds], v[in_bounds]


def align_depth_scale_sparse(
    d_pred_pts: np.ndarray,   # Depth-Anything 在顶点位置的预测值 (M,)
    d_3dmm_pts: np.ndarray,   # FLAME 顶点在相机坐标中的真实深度 (M,)
    valid: np.ndarray,        # (M,) bool mask
) -> Tuple[float, float]:
    """稀疏版深度尺度对齐，使用顶点深度样本点"""
    if valid.sum() < 20:
        logger.warning(f"有效顶点深度样本不足20个({valid.sum()})，使用默认 s=1, t=0")
        return 1.0, 0.0
    x = d_pred_pts[valid]
    y = d_3dmm_pts[valid]
    A = np.stack([x, np.ones_like(x)], axis=1)
    res = np.linalg.lstsq(A, y, rcond=None)
    s, t = float(res[0][0]), float(res[0][1])
    if not np.isfinite(s) or not np.isfinite(t):
        logger.warning("深度对齐出现非有限尺度，回退到 s=1, t=0")
        return 1.0, 0.0
    if s <= 0.0:
        logger.warning(f"深度对齐出现负尺度 s={s:.4f}，回退到安全值 s=1, t=0")
        return 1.0, 0.0
    residual = float(np.mean((A @ res[0] - y) ** 2) ** 0.5)
    logger.info(f"  深度对齐（稀疏）: s={s:.4f}, t={t:.4f}, RMSE={residual:.4f} (n={valid.sum()})")
    return s, t


def _sparse_to_dense_depth(
    z_vals: np.ndarray,
    px_u: np.ndarray,
    px_v: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """将稀疏顶点深度填充为密集深度图（用于调试/置换）"""
    H, W = image_shape
    d = np.full((H, W), np.inf, dtype=np.float32)
    for z, u, v in zip(z_vals, px_u, px_v):
        if d[v, u] > z:
            d[v, u] = z
    return d


def apply_depth_displacement(
    vertices: np.ndarray,        # (N, 3) 世界空间顶点
    faces: np.ndarray,           # (F, 3)
    vertex_normals: np.ndarray,  # (N, 3) 单位法向（世界空间）
    d_aligned: np.ndarray,       # (H, W) 对齐后的绝对深度图（米，Depth-Anything 输出）
    face_mask: np.ndarray,       # (H, W) 人脸掩码
    feather_map: np.ndarray,     # (H, W) mask 边缘衰减权重 [0,1]
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int],
    max_displacement: float = 0.005,
) -> np.ndarray:
    """
    逐顶点直接采样深度残差，沿法向方向位移。

    对每个顶点：
      1. 投影到图像坐标 (u, v)，同时得到相机空间实际深度 z_cam
      2. 从 d_aligned 采样该位置的预测深度 d_pred_at_v
      3. 位移量 = d_pred_at_v - z_cam（单位：米）
      4. 仅在 face_mask > 0 的范围内有效，边缘用 feather_map 衰减
      5. 沿顶点法线方向偏移

    【修复】彻底避免旧方案中 d_3dmm=inf 导致 disp_map 充满极值的 Bug。
    """
    H, W = image_shape

    # 投影到相机空间
    v_cam = (R @ vertices.T + t[:, None]).T       # (N, 3) 相机空间坐标
    z_cam = v_cam[:, 2]                            # 每顶点的实际相机空间深度（米）
    v_hom = (K @ v_cam.T).T                        # (N, 3)
    u_px  = (v_hom[:, 0] / np.clip(z_cam, 1e-6, None)).astype(int)
    v_px  = (v_hom[:, 1] / np.clip(z_cam, 1e-6, None)).astype(int)

    in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H) & (z_cam > 1e-4)
    safe_u = np.clip(u_px, 0, W - 1)
    safe_v = np.clip(v_px, 0, H - 1)

    # 采样各顶点投影位置的预测深度（d_aligned 已是米单位，且全部有限）
    d_pred_at_v = d_aligned[safe_v, safe_u].astype(np.float64)

    # 深度残差 = 预测深度 - 顶点实际深度（正值→向外突，负值→向内缩）
    disp_vals = d_pred_at_v - z_cam

    # 有效性掩码：越界 / face_mask 外 / 法线朝后（耳内/眼内凹陷）→ 置 0
    in_face  = face_mask[safe_v, safe_u] > 0
    good_nrm = vertex_normals[:, 2] >= -0.1
    valid    = in_bounds & in_face & good_nrm & np.isfinite(disp_vals)

    # feathering 衰减（mask 边缘减弱位移，避免阶跃缝隙）
    feather_w = feather_map[safe_v, safe_u].astype(np.float64)
    disp_vals = np.where(
        valid,
        np.clip(disp_vals * feather_w, -max_displacement, max_displacement),
        0.0,
    )

    return vertices + vertex_normals * disp_vals[:, None]


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """计算顶点法向（面法向加权平均，向量化实现）"""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)
    # np.add.at 向量化累加（比 Python 循环快 ~100x）
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)
    return normals


# ══════════════════════════════════════════════════════════════════════════════
# Mesh 导出
# ══════════════════════════════════════════════════════════════════════════════

def export_mesh_obj(
    vertices: np.ndarray,       # (N, 3)  几何顶点
    faces: np.ndarray,          # (F, 3)  几何面片索引（0-based）
    uv_verts: np.ndarray,       # (T, 2)  UV 坐标（可能与几何顶点数不同）
    uv_faces: np.ndarray,       # (F, 3)  UV 面片索引（0-based），与 faces 一一对应
    output_path: Path,
):
    """
    导出 .obj 文件（FLAME 分离 UV 拓扑格式，附顶点法线）。
    每个面片的几何顶点索引和 UV 顶点索引是独立的。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normals = compute_vertex_normals(vertices, faces)   # (N, 3) 用于 vn 行
    with open(str(output_path), "w") as f:
        f.write("# 3D Face Reconstruction - FLAME Mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in image_uv_to_obj_uv(uv_verts):
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        # 面片：几何/UV/法线（法线索引与几何顶点索引相同）
        for gf, uf in zip(faces, uv_faces):
            g0, g1, g2 = gf[0]+1, gf[1]+1, gf[2]+1
            u0, u1, u2 = uf[0]+1, uf[1]+1, uf[2]+1
            f.write(f"f {g0}/{u0}/{g0} {g1}/{u1}/{g1} {g2}/{u2}/{g2}\n")
    logger.info(f"Mesh 已导出: {output_path} ({len(vertices)} 顶点, {len(faces)} 面片, {len(uv_verts)} UV点)")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def export_mesh_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_verts: np.ndarray,
    uv_faces: np.ndarray,
    output_path: Path,
):
    """Export mesh as GLB for direct preview."""
    import trimesh
    from trimesh.visual.texture import TextureVisuals

    output_path.parent.mkdir(parents=True, exist_ok=True)

    uv_per_vertex = None
    if len(uv_verts) == len(vertices):
        uv_per_vertex = uv_verts.astype(np.float32)
    elif len(uv_faces) == len(faces):
        uv_per_vertex = np.zeros((len(vertices), 2), dtype=np.float32)
        uv_written = np.zeros(len(vertices), dtype=bool)
        for gf, uf in zip(faces, uv_faces):
            for g_idx, u_idx in zip(gf, uf):
                if not uv_written[g_idx]:
                    uv_per_vertex[g_idx] = uv_verts[u_idx]
                    uv_written[g_idx] = True

    visual = TextureVisuals(uv=uv_per_vertex) if uv_per_vertex is not None else None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, visual=visual)
    mesh.export(str(output_path))
    logger.info(f"GLB 已导出: {output_path}")


def _load_calibrated_rig_views(view_names) -> Optional[dict]:
    try:
        from src import config as cfg
        enabled = bool(getattr(cfg, "STABLE_USE_CALIBRATED_RIG_EXTRINSICS", False))
        calibration_path = Path(getattr(cfg, "CAMERA_CALIBRATION_PATH", ""))
    except Exception:
        return None
    if not enabled or not calibration_path.exists():
        return None
    try:
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("无法读取 rig 标定文件 %s: %s", calibration_path, exc)
        return None

    cameras = data.get("cameras", {})
    camera_by_view = {
        camera_data.get("view", camera_name): (camera_name, camera_data)
        for camera_name, camera_data in cameras.items()
    }
    reference_camera = data.get("reference_camera")
    reference_view = None
    if reference_camera in cameras:
        reference_view = cameras[reference_camera].get("view", reference_camera)
    if reference_view not in view_names and "front" in view_names:
        reference_view = "front"
    if reference_view not in view_names or reference_view not in camera_by_view:
        logger.warning("rig 标定 reference view 不可用: %s", reference_view)
        return None

    raw = {}
    for view_name in view_names:
        item = camera_by_view.get(view_name)
        if item is None:
            logger.warning("rig 标定缺少视角 %s 的相机数据", view_name)
            return None
        camera_name, camera_data = item
        rig = camera_data.get("rig_to_camera", {})
        if "R" not in rig or "t" not in rig:
            logger.warning("rig 标定视角 %s 缺少 rig_to_camera", view_name)
            return None
        raw[view_name] = {
            "camera": camera_name,
            "R": np.asarray(rig["R"], dtype=np.float64),
            "t": np.asarray(rig["t"], dtype=np.float64).reshape(3),
        }

    R_ref = raw[reference_view]["R"]
    t_ref = raw[reference_view]["t"]
    views = {}
    for view_name, item in raw.items():
        R_rel = item["R"] @ R_ref.T
        t_rel = item["t"] - R_rel @ t_ref
        views[view_name] = {
            "camera": item["camera"],
            "R_ref_to_camera": R_rel.astype(np.float32),
            "t_ref_to_camera": t_rel.astype(np.float32),
        }
    return {
        "enabled": True,
        "calibration_path": str(calibration_path),
        "reference_view": reference_view,
        "reference_camera": raw[reference_view]["camera"],
        "views": views,
    }


def _apply_calibrated_rig_initial_poses(view_data: dict, rig_data: dict, debug_dir: Path) -> bool:
    if not rig_data or rig_data.get("reference_view") not in view_data:
        return False
    reference_view = rig_data["reference_view"]
    ref_R = np.asarray(view_data[reference_view]["R_init"], dtype=np.float32)
    ref_t = np.asarray(view_data[reference_view]["t_init"], dtype=np.float32).reshape(3)
    meta = {
        "enabled": True,
        "calibration_path": rig_data.get("calibration_path"),
        "reference_view": reference_view,
        "reference_camera": rig_data.get("reference_camera"),
        "views": {},
    }
    for view_name, vd in view_data.items():
        rel = rig_data["views"].get(view_name)
        if rel is None:
            return False
        R_rel = np.asarray(rel["R_ref_to_camera"], dtype=np.float32)
        t_rel = np.asarray(rel["t_ref_to_camera"], dtype=np.float32).reshape(3)
        R_init = R_rel @ ref_R
        t_init = R_rel @ ref_t + t_rel
        vd["R_init"] = R_init.astype(np.float32)
        vd["t_init"] = t_init.astype(np.float32)
        vd["rig_R_ref_to_camera"] = R_rel.astype(np.float32)
        vd["rig_t_ref_to_camera"] = t_rel.astype(np.float32)
        vd["rig_reference_view"] = reference_view
        meta["views"][view_name] = {
            "camera": rel.get("camera"),
            "R_ref_to_camera": R_rel.astype(float).tolist(),
            "t_ref_to_camera": t_rel.astype(float).tolist(),
            "derived_R_init": R_init.astype(float).tolist(),
            "derived_t_init": t_init.astype(float).tolist(),
        }

    try:
        with open(debug_dir / "calibrated_rig_pose_init.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("保存 calibrated rig debug 失败: %s", exc)
    logger.info(
        "已应用固定 rig 外参初值：reference=%s, calibration=%s",
        reference_view,
        rig_data.get("calibration_path"),
    )
    return True


def run_geometry_reconstruction(
    preprocessed_views: Dict[str, dict],
    intrinsics: Dict[str, np.ndarray],
    flame_model_path: Path,
    flame_landmark_path: Path,
    deca_dir: Path,
    depth_model_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    n_shape: int = 100,
    n_exp: int = 50,
    lambda_shape: float = 1e-3,
    lambda_exp: float = 1e-3,
    lbfgs_max_iter: int = 100,
    lbfgs_lr: float = 0.5,
    depth_model_size: str = "large",
    max_displacement: float = 0.005,
    enable_depth_displacement: bool = True,
    init_backend: str = "face_alignment",
    mica_dir: Optional[Path] = None,
    mica_checkpoint: Optional[Path] = None,
    emoca_dir: Optional[Path] = None,
    emoca_checkpoint=None,
    deca_checkpoint: Optional[Path] = None,
) -> Path:
    """
    完整几何重建流程。
    Returns: 输出 .obj 文件路径

    init_backend 可选值:
      "face_alignment" — 原有 baseline
      "deca"           — DECA 身份 + 表情/姿态
      "mica_deca"      — MICA 身份 + DECA 表情/姿态（推荐）
      "mica_emoca"     — MICA 身份 + EMOCA 表情/姿态
    """
    import json
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 FLAME ────────────────────────────────────────────────────────
    logger.info("加载 FLAME 模型...")
    flame = FLAMEModel(flame_model_path, n_shape=n_shape, n_exp=n_exp)
    flame_faces_np = flame.faces.numpy()       # (F, 3)
    flame_verts_np = flame.v_template.numpy()  # (N, 3)

    # ── 加载关键点映射 ─────────────────────────────────────────────────────
    lmk_data = load_flame_landmark_mapping(flame_landmark_path)
    if lmk_data is not None and "face_idx" in lmk_data:
        face_idx  = lmk_data["face_idx"]
        bary      = lmk_data["bary_coords"]
        lmk_vertex_indices = flame_faces_np[face_idx, np.argmax(bary, axis=1)]  # (68,)
    else:
        logger.warning("使用近似关键点索引（建议下载 landmark_embedding.npy）")
        lmk_vertex_indices = np.arange(0, 68)

    # ── 新初始化后端 ──────────────────────────────────────────────────────
    logger.info(f"获取 3DMM 初始参数 [backend={init_backend}]...")
    from src.initializers import get_initializer
    init_result = get_initializer(
        backend=init_backend,
        images={k: v["image"] for k, v in preprocessed_views.items()},
        flame_verts=flame_verts_np,
        lmk_vertex_indices=lmk_vertex_indices,
        intrinsics=intrinsics,
        device=device,
        n_shape=n_shape,
        n_exp=n_exp,
        deca_dir=deca_dir,
        deca_checkpoint=deca_checkpoint,
        mica_dir=mica_dir,
        mica_checkpoint=mica_checkpoint,
        emoca_dir=emoca_dir,
        emoca_checkpoint=emoca_checkpoint,
    )

    # ── 保存初始形状 debug ────────────────────────────────────────────────
    init_shape = init_result["shape"]
    if not np.isfinite(init_shape).all():
        if str(init_backend).lower().startswith("mica"):
            raise RuntimeError("MICA identity shape contains NaN/Inf; stable identity fit aborted")
        logger.warning("init_shape 含 NaN/Inf，非 MICA 后端重置为零向量")
        init_shape = np.zeros_like(init_shape)
        init_result["shape"] = init_shape
    init_shape = np.asarray(init_shape, dtype=np.float32).copy()
    _save_init_shape_debug(init_result, debug_dir)

    # ── 为每个视角准备优化输入，并合并初始化结果 ───────────────────────────
    view_data = {}
    init_exps = {}

    # 仍需 face_alignment 2D 关键点作为优化约束目标（无论哪种 backend 都需要）
    from src.initializers.face_alignment_initializer import get_fa_per_view
    fa_lmks = get_fa_per_view(
        {k: v["image"] for k, v in preprocessed_views.items()},
        device,
    )
    rig_data = _load_calibrated_rig_views(preprocessed_views.keys())

    expression_states = {}
    for state_view_name, state_data in preprocessed_views.items():
        dense_landmarks = state_data.get("landmarks")
        if dense_landmarks is None:
            continue
        try:
            expression_states[state_view_name] = mediapipe_expression_state(dense_landmarks)
        except ValueError:
            continue
    try:
        from src import config as _expression_cfg
        shared_expression_enabled = bool(
            getattr(_expression_cfg, "STABLE_SHARED_EXPRESSION", True)
        )
        closed_eye_target_scale = float(
            getattr(_expression_cfg, "STABLE_CLOSED_EYE_TARGET_SCALE", 0.10)
        )
        closed_mouth_target_scale = float(
            getattr(_expression_cfg, "STABLE_CLOSED_MOUTH_TARGET_SCALE", 0.08)
        )
    except Exception:
        shared_expression_enabled = True
        closed_eye_target_scale = 0.10
        closed_mouth_target_scale = 0.08
    reference_expression_state = expression_states.get("front")
    if reference_expression_state is None and expression_states:
        reference_expression_state = next(iter(expression_states.values()))
    logger.info("MediaPipe expression states: %s", expression_states)

    for view_name, pdata in preprocessed_views.items():
        K_np = intrinsics[view_name]

        # 2D landmark target for optimizer
        fa_lmk = fa_lmks.get(view_name)
        if fa_lmk is not None:
            lmk_68 = fa_lmk[:, :2].astype(np.float32)
        elif init_result["fa_lmk_2d"].get(view_name) is not None:
            lmk_68 = init_result["fa_lmk_2d"][view_name]
        else:
            mp_lmk = pdata.get("landmarks")
            if mp_lmk is None:
                logger.warning(f"  [{view_name}] 无关键点，跳过该视角")
                continue
            lmk_68 = _mediapipe_to_68(mp_lmk)

        # Pose init from initializer or PnP fallback
        pv = init_result["per_view"].get(view_name, {})
        R_init = pv.get("R_init")
        t_init = pv.get("t_init")

        if R_init is None or t_init is None:
            flame_68_3d = flame_verts_np[lmk_vertex_indices]
            try:
                R_init, t_init = estimate_pose_from_landmarks_pnp(lmk_68, flame_68_3d, K_np)
                logger.info(f"  [{view_name}] PnP 后备姿态, t_z={t_init[2]:.4f}")
            except Exception as e:
                logger.warning(f"  [{view_name}] PnP 失败: {e}，使用默认姿态")
                R_init = np.eye(3, dtype=np.float32)
                t_init = np.array([0.0, 0.0, 0.3], dtype=np.float32)

        view_data[view_name] = {
            "lmk_2d":   lmk_68,
            "K":        K_np.astype(np.float32),
            "R_init":   R_init,
            "t_init":   t_init,
            "face_mask": pdata.get("face_mask"),
            "shape_mask": pdata.get("shape_mask"),
            "expression_state": expression_states.get(view_name, {}),
            "closed_eyes": bool(
                (reference_expression_state or {}).get("closed_eyes", False)
                if shared_expression_enabled
                else expression_states.get(view_name, {}).get("closed_eyes", False)
            ),
            "closed_mouth": bool(
                (reference_expression_state or {}).get("closed_mouth", False)
                if shared_expression_enabled
                else expression_states.get(view_name, {}).get("closed_mouth", False)
            ),
            "closed_eye_target_scale": closed_eye_target_scale,
            "closed_mouth_target_scale": closed_mouth_target_scale,
        }
        init_exps[view_name] = pv.get("exp")

    if not view_data:
        raise RuntimeError("所有视角关键点检测失败，无法进行 3DMM 重建")

    # ── 保存每视角初始化参数 + 预优化重投影图 ────────────────────────────────
    rig_applied = _apply_calibrated_rig_initial_poses(view_data, rig_data, debug_dir) if rig_data else False
    lmk_tri_vidx = flame_faces_np[lmk_data["face_idx"]] if lmk_data is not None else None
    if not rig_applied:
        _audit_and_repair_initial_poses(
            view_data=view_data,
            init_result=init_result,
            init_exps=init_exps,
            init_shape=init_shape,
            flame=flame,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            debug_dir=debug_dir,
        )
    else:
        logger.info("固定 rig 外参已启用：跳过 per-view PnP 初始位姿修正。")
    _save_init_per_view_debug(
        view_data=view_data,
        init_result=init_result,
        init_shape=init_shape,
        flame=flame,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
        preprocessed_views=preprocessed_views,
        intrinsics=intrinsics,
        debug_dir=debug_dir,
        device=device,
    )

    # ── 联合 L-BFGS 优化 ──────────────────────────────────────────────────
    try:
        from src import config as _cfg
    except Exception:
        _cfg = None

    def _cfg_float(name: str, default: float) -> float:
        if _cfg is None:
            return default
        try:
            return float(getattr(_cfg, name, default))
        except Exception:
            return default

    def _cfg_bool(name: str, default: bool) -> bool:
        if _cfg is None:
            return default
        try:
            return bool(getattr(_cfg, name, default))
        except Exception:
            return default

    def _cfg_float_list(name: str, default) -> list:
        value = getattr(_cfg, name, default) if _cfg is not None else default
        if isinstance(value, str):
            value = [item.strip() for item in value.split(",") if item.strip()]
        try:
            parsed = [float(item) for item in value]
        except (TypeError, ValueError):
            parsed = [float(item) for item in default]
        return parsed or [float(item) for item in default]

    mean_shape_prior_weight = _cfg_float("JOINT_MEAN_SHAPE_PRIOR_WEIGHT", 5e-6)
    identity_thresholds = IdentityDriftThresholds(
        max_coefficient_l2=_cfg_float("IDENTITY_MAX_COEFFICIENT_L2", 7.0),
        max_mean_displacement_pct=_cfg_float(
            "IDENTITY_MAX_MEAN_DISPLACEMENT_PCT", 1.5
        ),
        max_p95_displacement_pct=_cfg_float(
            "IDENTITY_MAX_P95_DISPLACEMENT_PCT", 2.5
        ),
        max_displacement_pct=_cfg_float("IDENTITY_MAX_DISPLACEMENT_PCT", 4.0),
    )
    identity_attempt_weights = _cfg_float_list(
        "JOINT_IDENTITY_ANCHOR_WEIGHTS", (1.6e-2, 3.2e-2, 6.4e-2)
    )
    max_identity_attempts = max(
        1, int(_cfg_float("JOINT_MAX_IDENTITY_ATTEMPTS", 3))
    )
    identity_attempt_weights = identity_attempt_weights[:max_identity_attempts]

    flame = flame.to(device)
    neutral_exp_t = torch.zeros(n_exp, device=device, dtype=torch.float32)
    with torch.no_grad():
        mica_neutral_vertices = flame(
            torch.tensor(init_shape, device=device, dtype=torch.float32),
            neutral_exp_t,
        ).detach().cpu().numpy()
    mica_mesh_quality = compute_mesh_quality(
        mica_neutral_vertices, flame_faces_np, label="mica_identity_anchor"
    )
    joint_identity_dir = debug_dir / "joint_identity_anchor"
    joint_identity_dir.mkdir(parents=True, exist_ok=True)

    def _make_optimizer(identity_anchor_weight: float):
        return JointFLAMEOptimizer(
            flame=flame,
            lmk_vertex_indices=lmk_vertex_indices,
            lambda_shape=lambda_shape,
            identity_anchor_weight=identity_anchor_weight,
            mean_shape_prior_weight=mean_shape_prior_weight,
            lambda_exp=lambda_exp,
            lambda_contour=_cfg_float("LAMBDA_CONTOUR", 0.0),
            front_contour_weight=_cfg_float("FRONT_CONTOUR_WEIGHT", 2.4),
            front_jaw_weight=_cfg_float("FRONT_JAW_WEIGHT", 3.2),
            side_contour_weight=_cfg_float("SIDE_CONTOUR_WEIGHT", 1.2),
            side_jaw_weight=_cfg_float("SIDE_JAW_WEIGHT", 1.8),
            side_brow_weight=_cfg_float("SIDE_BROW_WEIGHT", 0.25),
            side_extra_soft_weight=_cfg_float("SIDE_EXTRA_SOFT_WEIGHT", 0.6),
            eye_gap_loss_weight=_cfg_float("STABLE_EYE_GAP_LOSS_WEIGHT", 4.0),
            mouth_gap_loss_weight=_cfg_float("STABLE_MOUTH_GAP_LOSS_WEIGHT", 6.0),
            closed_eye_target_scale=_cfg_float("STABLE_CLOSED_EYE_TARGET_SCALE", 0.10),
            closed_mouth_target_scale=_cfg_float("STABLE_CLOSED_MOUTH_TARGET_SCALE", 0.08),
            shared_expression=_cfg_bool("STABLE_SHARED_EXPRESSION", True),
            max_iter=lbfgs_max_iter,
            lr=lbfgs_lr,
            device=device,
            lmk_face_idx=lmk_data["face_idx"] if lmk_data is not None else None,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
        )

    def _run_optimizer_once(attempt_idx: int, identity_anchor_weight: float):
        optimizer = _make_optimizer(identity_anchor_weight)
        shape_run, per_view_run = optimizer.optimize(view_data, init_shape, init_exps)
        attempt_dir = joint_identity_dir / f"attempt_{attempt_idx}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        reproj_stats = {}
        for view_name, view_result in per_view_run.items():
            with torch.no_grad():
                verts_view = flame(
                    torch.tensor(shape_run, device=device),
                    torch.tensor(view_result["exp"], device=device),
                ).cpu().numpy()
            mean_err, max_err, errors = _save_landmark_reprojection_debug(
                vertices=verts_view,
                K=intrinsics[view_name],
                R=view_result["R"],
                t=view_result["t"],
                image=preprocessed_views[view_name]["image"],
                target_landmarks=view_data[view_name]["lmk_2d"],
                lmk_vertex_indices=lmk_vertex_indices,
                out_path=attempt_dir / f"landmark_reproj_{view_name}.png",
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
                return_errors=True,
            )
            interior_errors = np.asarray(errors)[17:68]
            interior_mean = float(interior_errors.mean())
            interior_max = float(interior_errors.max())
            projected_landmarks, _ = _landmark_reprojection_details(
                vertices=verts_view,
                K=intrinsics[view_name],
                R=view_result["R"],
                t=view_result["t"],
                target_landmarks=view_data[view_name]["lmk_2d"],
                lmk_vertex_indices=lmk_vertex_indices,
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            )
            reproj_stats[view_name] = {
                "interior_mean_px": interior_mean,
                "interior_max_px": interior_max,
                "legacy_full_mean_px": float(mean_err),
                "legacy_full_max_px": float(max_err),
                "expression_fidelity": feature_gap_diagnostics(
                    projected_landmarks,
                    view_data[view_name]["lmk_2d"],
                    force_closed_eyes=bool(view_data[view_name].get("closed_eyes", False)),
                    force_closed_mouth=bool(view_data[view_name].get("closed_mouth", False)),
                    closed_eye_target_scale=float(
                        view_data[view_name].get("closed_eye_target_scale", 0.10)
                    ),
                    closed_mouth_target_scale=float(
                        view_data[view_name].get("closed_mouth_target_scale", 0.08)
                    ),
                ),
            }
            logger.info(
                f"  [identity attempt {attempt_idx}] [{view_name}] landmark reprojection error: "
                f"interior_mean={interior_mean:.2f}px, interior_max={interior_max:.2f}px, "
                f"legacy_full_mean={mean_err:.2f}px"
            )
        with torch.no_grad():
            candidate_neutral_vertices = flame(
                torch.tensor(shape_run, device=device, dtype=torch.float32),
                neutral_exp_t,
            ).detach().cpu().numpy()
        identity_gate = make_identity_drift_gate(
            anchor_shape=init_shape,
            candidate_shape=shape_run,
            anchor_vertices=mica_neutral_vertices,
            candidate_vertices=candidate_neutral_vertices,
            thresholds=identity_thresholds,
        )
        candidate_mesh_quality = compute_mesh_quality(
            candidate_neutral_vertices,
            flame_faces_np,
            label=f"joint_identity_attempt_{attempt_idx}",
        )
        mesh_quality_gate = make_quality_gate(
            baseline=mica_mesh_quality,
            candidate=candidate_mesh_quality,
            thresholds=MeshQualityThresholds(
                min_face_ratio=1.0,
                max_new_degenerate_faces=0,
                max_new_nonmanifold_edges=0,
                max_new_boundary_edges=0,
            ),
            region_name="joint_identity_neutral_mesh",
        )
        observation_score = (
            float(np.mean([row["interior_mean_px"] for row in reproj_stats.values()]))
            if reproj_stats else float("inf")
        )
        return {
            "attempt": int(attempt_idx),
            "identity_anchor_weight": float(identity_anchor_weight),
            "mean_shape_prior_weight": float(mean_shape_prior_weight),
            "shape": np.asarray(shape_run, dtype=np.float32),
            "per_view": per_view_run,
            "reprojection": reproj_stats,
            "observation_score_px": observation_score,
            "identity_gate": identity_gate,
            "mesh_quality_gate": mesh_quality_gate,
        }

    joint_candidates = []
    attempt_summaries = []
    for attempt_idx, anchor_weight in enumerate(identity_attempt_weights, start=1):
        try:
            candidate = _run_optimizer_once(attempt_idx, anchor_weight)
            joint_candidates.append(candidate)
            serializable = {key: value for key, value in candidate.items() if key not in {"shape", "per_view"}}
            serializable["shape_params"] = candidate["shape"].tolist()
            serializable["status"] = (
                "eligible" if candidate["identity_gate"]["passed"]
                and candidate["mesh_quality_gate"]["passed"] else "rejected"
            )
        except Exception as exc:
            logger.exception("Joint identity attempt %s failed", attempt_idx)
            serializable = {
                "attempt": int(attempt_idx),
                "identity_anchor_weight": float(anchor_weight),
                "status": "error",
                "error": repr(exc),
            }
        attempt_summaries.append(serializable)
        with open(joint_identity_dir / f"attempt_{attempt_idx}.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    selected_joint = select_identity_safe_candidate(
        joint_candidates,
        observation_tolerance=_cfg_float("IDENTITY_OBSERVATION_TIE_PX", 1.0),
    )
    joint_identity_summary = {
        "mica_anchor_available": True,
        "attempt_weights": identity_attempt_weights,
        "attempts": attempt_summaries,
        "selected_attempt": int(selected_joint["attempt"]) if selected_joint else None,
        "passed": selected_joint is not None,
    }
    with open(joint_identity_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(joint_identity_summary, f, ensure_ascii=False, indent=2)

    if selected_joint is None:
        import trimesh as _identity_trimesh
        baseline_path = joint_identity_dir / "mica_identity_baseline.glb"
        _identity_trimesh.Trimesh(
            vertices=mica_neutral_vertices,
            faces=flame_faces_np,
            process=False,
        ).export(str(baseline_path))
        raise RuntimeError(
            "All joint FLAME candidates failed MICA identity or mesh quality gates; "
            f"diagnostics: {joint_identity_dir}"
        )

    shape_opt = selected_joint["shape"]
    per_view_results = selected_joint["per_view"]
    reproj_stats = selected_joint["reprojection"]
    selected_attempt_dir = joint_identity_dir / f"attempt_{selected_joint['attempt']}"
    for view_name in per_view_results:
        selected_reprojection = selected_attempt_dir / f"landmark_reproj_{view_name}.png"
        if selected_reprojection.exists():
            shutil.copy2(selected_reprojection, debug_dir / f"landmark_reproj_{view_name}.png")
    logger.info(
        "Selected MICA-anchored joint attempt %s: interior=%.3fpx, coefficient drift=%.3f",
        selected_joint["attempt"],
        selected_joint["observation_score_px"],
        selected_joint["identity_gate"]["metrics"]["coefficient_delta_l2"],
    )

    shape_only_report = {"enabled": False, "accepted": False}
    if _cfg_bool("ENABLE_SHAPE_ONLY_FINE_TUNE", True):
        shape_opt, shape_only_report = _shape_only_fine_tune(
            flame=flame,
            shape_init=shape_opt,
            identity_anchor_shape=init_shape,
            per_view_results=per_view_results,
            view_data=view_data,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_face_idx=lmk_data["face_idx"] if lmk_data is not None else None,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            debug_dir=debug_dir,
            device=device,
            max_iter=int(_cfg_float("SHAPE_ONLY_MAX_ITER", 80)),
            lr=_cfg_float("SHAPE_ONLY_LR", 0.03),
            contour_scale=_cfg_float("SHAPE_ONLY_CONTOUR_SCALE", 1.5),
            stable_anchor_weight=_cfg_float("SHAPE_ONLY_STABLE_ANCHOR_WEIGHT", 1.8),
            delta_weight=_cfg_float("SHAPE_ONLY_DELTA_WEIGHT", 0.05),
            max_param_delta=_cfg_float("SHAPE_ONLY_MAX_PARAM_DELTA", 0.35),
            min_contour_improve_px=_cfg_float("SHAPE_ONLY_MIN_CONTOUR_IMPROVE_PX", 0.05),
            max_stable_worsen_px=_cfg_float("SHAPE_ONLY_MAX_STABLE_WORSEN_PX", 2.0),
            max_total_worsen_px=_cfg_float("SHAPE_ONLY_MAX_TOTAL_WORSEN_PX", 1.5),
            front_contour_weight=_cfg_float("FRONT_CONTOUR_WEIGHT", 2.4),
            front_jaw_weight=_cfg_float("FRONT_JAW_WEIGHT", 3.2),
            side_contour_weight=_cfg_float("SIDE_CONTOUR_WEIGHT", 1.2),
            side_jaw_weight=_cfg_float("SIDE_JAW_WEIGHT", 1.8),
            enable_dense_contour=_cfg_bool("ENABLE_SHAPE_ONLY_DENSE_CONTOUR", True),
            dense_contour_weight=_cfg_float("SHAPE_ONLY_DENSE_CONTOUR_WEIGHT", 4.0),
            dense_row_step=int(_cfg_float("SHAPE_ONLY_DENSE_ROW_STEP", 6)),
            dense_row_sigma=_cfg_float("SHAPE_ONLY_DENSE_ROW_SIGMA", 7.0),
            dense_softmin_tau=_cfg_float("SHAPE_ONLY_DENSE_SOFTMIN_TAU", 10.0),
            dense_boundary_band_px=_cfg_float("SHAPE_ONLY_DENSE_BOUNDARY_BAND_PX", 90.0),
            dense_search_margin_px=_cfg_float("SHAPE_ONLY_DENSE_SEARCH_MARGIN_PX", 70.0),
            min_dense_contour_improve_px=_cfg_float("SHAPE_ONLY_MIN_DENSE_CONTOUR_IMPROVE_PX", 0.25),
            max_contour_worsen_px=_cfg_float("SHAPE_ONLY_MAX_CONTOUR_WORSEN_PX", 1.5),
            enable_silhouette=_cfg_bool("ENABLE_DIFFERENTIABLE_SILHOUETTE", False),
            silhouette_resolution=int(_cfg_float("SILHOUETTE_RENDER_RESOLUTION", 256)),
            silhouette_weight=_cfg_float("SILHOUETTE_LOSS_WEIGHT", 0.25),
            silhouette_sdf_weight=_cfg_float("SILHOUETTE_SDF_WEIGHT", 2.0),
            silhouette_sdf_side_scale=_cfg_float(
                "SILHOUETTE_SDF_SIDE_SCALE", 0.5
            ),
            silhouette_dice_weight=_cfg_float("SILHOUETTE_DICE_WEIGHT", 1.0),
            silhouette_l1_weight=_cfg_float("SILHOUETTE_L1_WEIGHT", 0.5),
            min_silhouette_improve_pct=_cfg_float(
                "SILHOUETTE_MIN_TRUSTED_IMPROVE_PCT", 0.10
            ),
            max_silhouette_view_worsen_pct=_cfg_float(
                "SILHOUETTE_MAX_VIEW_WORSEN_PCT", 0.15
            ),
            min_silhouette_improved_views=int(
                _cfg_float("SILHOUETTE_MIN_IMPROVED_VIEWS", 2)
            ),
            max_silhouette_overlap_drop=_cfg_float(
                "SILHOUETTE_MAX_OVERLAP_DROP", 0.005
            ),
            max_interior_mean_worsen_pct=_cfg_float(
                "SILHOUETTE_MAX_INTERIOR_MEAN_WORSEN_PCT", 0.15
            ),
            max_interior_view_worsen_pct=_cfg_float(
                "SILHOUETTE_MAX_INTERIOR_VIEW_WORSEN_PCT", 0.25
            ),
            min_relative_boundary_improve=_cfg_float(
                "SILHOUETTE_MIN_RELATIVE_BOUNDARY_IMPROVE", 0.15
            ),
            min_front_relative_boundary_improve=_cfg_float(
                "SILHOUETTE_MIN_FRONT_RELATIVE_BOUNDARY_IMPROVE", 0.30
            ),
            identity_thresholds=identity_thresholds,
        )

    pose_refine_report = {"enabled": False, "accepted": False}
    if _cfg_bool("ENABLE_POSE_REFINEMENT", True):
        per_view_results, pose_refine_report = _pose_refine_cameras(
            flame=flame,
            shape_opt=shape_opt,
            per_view_results=per_view_results,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_face_idx=lmk_data["face_idx"] if lmk_data is not None else None,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            debug_dir=debug_dir,
            device=device,
            enabled=True,
            max_iter=int(_cfg_float("POSE_REFINE_MAX_ITER", 80)),
            lr=_cfg_float("POSE_REFINE_LR", 0.01),
            lmk_weight=_cfg_float("POSE_REFINE_LMK_WEIGHT", 1.0),
            stable_weight=_cfg_float("POSE_REFINE_STABLE_WEIGHT", 1.4),
            dense_contour_weight=_cfg_float("POSE_REFINE_DENSE_CONTOUR_WEIGHT", 0.35),
            rot_reg=_cfg_float("POSE_REFINE_ROT_REG", 0.02),
            trans_reg=_cfg_float("POSE_REFINE_TRANS_REG", 0.02),
            min_mean_improve_px=_cfg_float("POSE_REFINE_MIN_MEAN_IMPROVE_PX", 1.0),
            min_dense_improve_px=_cfg_float("POSE_REFINE_MIN_DENSE_IMPROVE_PX", 3.0),
            max_mean_worsen_px=_cfg_float("POSE_REFINE_MAX_MEAN_WORSEN_PX", 1.0),
            max_stable_worsen_px=_cfg_float("POSE_REFINE_MAX_STABLE_WORSEN_PX", 1.0),
            max_maxerr_worsen_px=_cfg_float("POSE_REFINE_MAX_MAXERR_WORSEN_PX", 5.0),
            max_rot_deg=_cfg_float("POSE_REFINE_MAX_ROT_DEG", 6.0),
            max_trans_rel=_cfg_float("POSE_REFINE_MAX_TRANS_REL", 0.08),
            global_max_overall_worsen_px=_cfg_float("POSE_REFINE_GLOBAL_MAX_OVERALL_WORSEN_PX", 0.75),
            global_max_front_worsen_px=_cfg_float("POSE_REFINE_GLOBAL_MAX_FRONT_WORSEN_PX", 0.5),
            global_max_side_dense_worsen_px=_cfg_float("POSE_REFINE_GLOBAL_MAX_SIDE_DENSE_WORSEN_PX", 2.0),
        )

    with open(debug_dir / "optimized_shape.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "shape_norm": float(np.linalg.norm(shape_opt)) if shape_opt is not None else 0.0,
                "shape_params": shape_opt.tolist() if shape_opt is not None else [],
                "mica_identity_anchor": init_shape.tolist(),
                "joint_identity_anchor": joint_identity_summary,
                "shape_only_fine_tune": shape_only_report,
                "pose_refinement": pose_refine_report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    _save_optimized_pose_quality_debug(
        shape_opt=shape_opt,
        per_view_results=per_view_results,
        flame=flame,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
        view_data=view_data,
        preprocessed_views=preprocessed_views,
        intrinsics=intrinsics,
        debug_dir=debug_dir,
        device=device,
    )

    # 用多视角平均表情生成最终基础网格，避免被单一正脸视角绑定
    front_name = "front" if "front" in per_view_results else list(per_view_results.keys())[0]
    exp_stack = [np.asarray(v["exp"], dtype=np.float32) for v in per_view_results.values() if v.get("exp") is not None]
    if exp_stack:
        exp_final = np.mean(np.stack(exp_stack, axis=0), axis=0).astype(np.float32)
    else:
        exp_final = np.zeros(n_exp, dtype=np.float32)
    expression_fidelity_views = {}
    for name, result in per_view_results.items():
        with torch.no_grad():
            view_vertices = flame(
                torch.tensor(shape_opt, device=device, dtype=torch.float32),
                torch.tensor(result["exp"], device=device, dtype=torch.float32),
            ).cpu().numpy()
        projected_landmarks, _ = _landmark_reprojection_details(
            vertices=view_vertices,
            K=intrinsics[name],
            R=result["R"],
            t=result["t"],
            target_landmarks=view_data[name]["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
        )
        expression_fidelity_views[name] = feature_gap_diagnostics(
            projected_landmarks,
            view_data[name]["lmk_2d"],
            force_closed_eyes=bool(view_data[name].get("closed_eyes", False)),
            force_closed_mouth=bool(view_data[name].get("closed_mouth", False)),
            closed_eye_target_scale=float(
                view_data[name].get("closed_eye_target_scale", 0.10)
            ),
            closed_mouth_target_scale=float(
                view_data[name].get("closed_mouth_target_scale", 0.08)
            ),
        )
    mean_eye_gap_error = float(np.mean([
        row["eye_gap_error_px"] for row in expression_fidelity_views.values()
    ])) if expression_fidelity_views else float("inf")
    mean_mouth_gap_error = float(np.mean([
        row["mouth_gap_error_px"] for row in expression_fidelity_views.values()
    ])) if expression_fidelity_views else float("inf")
    max_eye_gap_error = _cfg_float("STABLE_MAX_EYE_GAP_ERROR_PX", 3.0)
    max_mouth_gap_error = _cfg_float("STABLE_MAX_MOUTH_GAP_ERROR_PX", 3.0)
    expression_fidelity = {
        "passed": bool(
            np.isfinite(mean_eye_gap_error)
            and np.isfinite(mean_mouth_gap_error)
            and mean_eye_gap_error <= max_eye_gap_error
            and mean_mouth_gap_error <= max_mouth_gap_error
        ),
        "mean_eye_gap_error_px": mean_eye_gap_error,
        "mean_mouth_gap_error_px": mean_mouth_gap_error,
        "thresholds": {
            "max_eye_gap_error_px": max_eye_gap_error,
            "max_mouth_gap_error_px": max_mouth_gap_error,
        },
        "observed_state": {
            name: {
                **dict(view_data[name].get("expression_state", {})),
                "enforced_closed_eyes": bool(view_data[name].get("closed_eyes", False)),
                "enforced_closed_mouth": bool(view_data[name].get("closed_mouth", False)),
            }
            for name in expression_fidelity_views
        },
        "views": expression_fidelity_views,
    }
    with open(debug_dir / "optimized_parameters.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "shape_params": shape_opt.tolist() if shape_opt is not None else [],
                "mica_identity_anchor": init_shape.tolist(),
                "identity_preservation": shape_only_report.get(
                    "identity_drift", selected_joint["identity_gate"]
                ),
                "expression_params": exp_final.tolist(),
                "expression_fidelity": expression_fidelity,
                "per_view": {
                    name: {
                        "expression_params": np.asarray(res.get("exp", []), dtype=np.float32).tolist(),
                        "R": np.asarray(res.get("R", []), dtype=np.float32).tolist(),
                        "t": np.asarray(res.get("t", []), dtype=np.float32).tolist(),
                    }
                    for name, res in per_view_results.items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if not expression_fidelity["passed"]:
        raise RuntimeError(
            "Expression fidelity gate failed: "
            f"eye_gap={mean_eye_gap_error:.3f}px, "
            f"mouth_gap={mean_mouth_gap_error:.3f}px"
        )

    with torch.no_grad():
        verts_final = flame(
            torch.tensor(shape_opt, device=device),
            torch.tensor(exp_final, device=device),
        ).cpu().numpy()  # (N, 3)
        verts_neutral = flame(
            torch.tensor(shape_opt, device=device),
            torch.zeros(n_exp, device=device, dtype=torch.float32),
        ).cpu().numpy()

    controlled_identity_report = {"enabled": False, "accepted": False}
    if _cfg_bool("STABLE_ENABLE_CONTROLLED_IDENTITY", False):
        logger.info("Controlled low-frequency identity optimization...")
        try:
            with torch.no_grad():
                controlled_view_vertices = {
                    name: flame(
                        torch.tensor(shape_opt, device=device, dtype=torch.float32),
                        torch.tensor(result["exp"], device=device, dtype=torch.float32),
                    ).cpu().numpy()
                    for name, result in per_view_results.items()
                }
            verts_final, verts_neutral, controlled_identity_report = (
                _controlled_low_frequency_identity_tune(
                    baseline_final_vertices=verts_final,
                    baseline_neutral_vertices=verts_neutral,
                    per_view_vertices=controlled_view_vertices,
                    faces=flame_faces_np,
                    per_view_results=per_view_results,
                    view_data=view_data,
                    preprocessed_views=preprocessed_views,
                    intrinsics=intrinsics,
                    lmk_vertex_indices=lmk_vertex_indices,
                    lmk_tri_vidx=lmk_tri_vidx,
                    lmk_bary_coords=(
                        lmk_data["bary_coords"] if lmk_data is not None else None
                    ),
                    debug_dir=debug_dir,
                    device=device,
                    settings={
                        "max_iterations": int(_cfg_float("CONTROLLED_IDENTITY_MAX_ITER", 120)),
                        "learning_rate": _cfg_float("CONTROLLED_IDENTITY_LR", 0.05),
                        "checkpoint_interval": int(_cfg_float("CONTROLLED_IDENTITY_CHECKPOINT_INTERVAL", 5)),
                        "render_resolution": int(_cfg_float("CONTROLLED_IDENTITY_RENDER_RESOLUTION", 256)),
                        "mask_weight": _cfg_float("CONTROLLED_IDENTITY_MASK_WEIGHT", 0.0),
                        "sdf_weight": _cfg_float("CONTROLLED_IDENTITY_SDF_WEIGHT", 3.0),
                        "sdf_side_scale": _cfg_float("CONTROLLED_IDENTITY_SDF_SIDE_SCALE", 0.65),
                        "landmark_weight": _cfg_float("CONTROLLED_IDENTITY_LANDMARK_WEIGHT", 0.25),
                        "observation_weight": _cfg_float("CONTROLLED_IDENTITY_OBSERVATION_WEIGHT", 1.0),
                        "coefficient_weight": _cfg_float("CONTROLLED_IDENTITY_COEFFICIENT_WEIGHT", 0.005),
                        "symmetry_weight": _cfg_float("CONTROLLED_IDENTITY_SYMMETRY_WEIGHT", 0.02),
                        "edge_weight": _cfg_float("CONTROLLED_IDENTITY_EDGE_WEIGHT", 0.20),
                        "laplacian_weight": _cfg_float("CONTROLLED_IDENTITY_LAPLACIAN_WEIGHT", 0.50),
                        "width_sigma": _cfg_float("CONTROLLED_IDENTITY_WIDTH_SIGMA", 0.24),
                        "protection_core_ratio": _cfg_float("CONTROLLED_IDENTITY_PROTECTION_CORE_RATIO", 0.035),
                        "protection_outer_ratio": _cfg_float("CONTROLLED_IDENTITY_PROTECTION_OUTER_RATIO", 0.12),
                        "width_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_WIDTH_DISPLACEMENT_RATIO", 0.040),
                        "depth_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_DEPTH_DISPLACEMENT_RATIO", 0.025),
                        "chin_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_CHIN_DISPLACEMENT_RATIO", 0.030),
                        "max_vertex_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_VERTEX_DISPLACEMENT_RATIO", 0.060),
                        "max_mean_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_MEAN_DISPLACEMENT_RATIO", 0.025),
                        "max_protected_displacement_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_PROTECTED_DISPLACEMENT_RATIO", 0.001),
                        "max_new_normal_flips": int(_cfg_float("CONTROLLED_IDENTITY_MAX_NEW_NORMAL_FLIPS", 0)),
                        "min_front_boundary_improvement_ratio": _cfg_float("CONTROLLED_IDENTITY_MIN_FRONT_BOUNDARY_IMPROVEMENT_RATIO", 0.30),
                        "max_front_region_worsen_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_FRONT_REGION_WORSEN_RATIO", 0.10),
                        "max_profile_boundary_worsen_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_PROFILE_BOUNDARY_WORSEN_RATIO", 0.10),
                        "max_interior_landmark_worsen_ratio": _cfg_float("CONTROLLED_IDENTITY_MAX_INTERIOR_LANDMARK_WORSEN_RATIO", 0.10),
                        "max_overlap_drop": _cfg_float("CONTROLLED_IDENTITY_MAX_OVERLAP_DROP", 0.01),
                    },
                )
            )
        except Exception as exc:
            logger.exception("Controlled low-frequency identity stage failed; baseline retained")
            controlled_identity_report = {
                "enabled": True,
                "accepted": False,
                "reason": "stage error; exact baseline retained",
                "error": repr(exc),
            }
            controlled_dir = debug_dir / "controlled_identity_deformation"
            controlled_dir.mkdir(parents=True, exist_ok=True)
            with open(controlled_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(controlled_identity_report, f, ensure_ascii=False, indent=2)

    optimized_parameters_path = debug_dir / "optimized_parameters.json"
    try:
        with open(optimized_parameters_path, "r", encoding="utf-8") as f:
            optimized_parameters_payload = json.load(f)
    except Exception:
        optimized_parameters_payload = {}
    optimized_parameters_payload["controlled_identity_deformation"] = controlled_identity_report
    with open(optimized_parameters_path, "w", encoding="utf-8") as f:
        json.dump(optimized_parameters_payload, f, ensure_ascii=False, indent=2)

    # ── 步骤2：Loop Subdivision（增加几何密度，在置换前细分提高精度）────────
    import trimesh as _trimesh
    logger.info("Loop Subdivision：细分 FLAME 基础网格...")
    # 提前加载原始 UV，与几何拓扑同步细分（保证面片一一对应）
    uv_verts_orig, uv_faces_orig = _get_flame_uv(flame_model_path, flame_faces_np)
    SUBDIV_ITERS = 2   # 9976 → ~40K → ~160K 面片
    verts_sub, faces_sub = _trimesh.remesh.subdivide_loop(
        verts_final, flame_faces_np, iterations=SUBDIV_ITERS
    )
    verts_neutral_sub, faces_neutral_sub = _trimesh.remesh.subdivide_loop(
        verts_neutral, flame_faces_np, iterations=SUBDIV_ITERS
    )
    if not np.array_equal(faces_sub, faces_neutral_sub):
        raise RuntimeError("Neutral and expression subdivision topology diverged")
    # UV 用线性细分（与 Loop 细分面片拓扑一致，保证几何-UV 面片一一对应）
    uv_verts_sub, uv_faces_sub = uv_verts_orig, uv_faces_orig
    for _ in range(SUBDIV_ITERS):
        uv_verts_sub, uv_faces_sub = _trimesh.remesh.subdivide(uv_verts_sub, uv_faces_sub)
    logger.info(f"细分完成: {len(faces_sub)} 面片, {len(verts_sub)} 顶点")

    vertex_normals = compute_vertex_normals(verts_sub, faces_sub)

    try:
        from src import config as _free_cfg
    except Exception:
        _free_cfg = None

    def _free_cfg_float(name: str, default: float) -> float:
        if _free_cfg is None:
            return default
        try:
            return float(getattr(_free_cfg, name, default))
        except Exception:
            return default

    def _free_cfg_bool(name: str, default: bool) -> bool:
        if _free_cfg is None:
            return default
        try:
            return bool(getattr(_free_cfg, name, default))
        except Exception:
            return default

    personal_residual_report = {"enabled": False, "applied": False, "accepted": False}
    if _free_cfg_bool("ENABLE_PERSONAL_RESIDUAL_DEFORM", False):
        verts_sub, _verts_residual, personal_residual_report = _personal_residual_deform_mesh(
            verts_base=verts_sub,
            verts_displaced=verts_sub,
            faces=faces_sub,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            per_view_results=per_view_results,
            debug_dir=output_dir.parent / "debug",
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            enabled=True,
            max_offset_m=_free_cfg_float("PERSONAL_RESIDUAL_MAX_OFFSET_M", 0.018),
            boundary_band_px=_free_cfg_float("PERSONAL_RESIDUAL_BOUNDARY_BAND_PX", 95.0),
            search_margin_px=_free_cfg_float("PERSONAL_RESIDUAL_SEARCH_MARGIN_PX", 80.0),
            stable_protect_radius_px=_free_cfg_float("PERSONAL_RESIDUAL_STABLE_PROTECT_RADIUS_PX", 44.0),
            row_step=int(_free_cfg_float("PERSONAL_RESIDUAL_ROW_STEP", 6)),
            row_sigma=_free_cfg_float("PERSONAL_RESIDUAL_ROW_SIGMA", 8.0),
            tau=_free_cfg_float("PERSONAL_RESIDUAL_SOFTMIN_TAU", 10.0),
            envelope_topk=int(_free_cfg_float("PERSONAL_RESIDUAL_ENVELOPE_TOPK", 26)),
            view_weight_front=_free_cfg_float("PERSONAL_RESIDUAL_VIEW_WEIGHT_FRONT", 0.35),
            view_weight_side=_free_cfg_float("PERSONAL_RESIDUAL_VIEW_WEIGHT_SIDE", 1.0),
            min_view_cos=_free_cfg_float("PERSONAL_RESIDUAL_MIN_VIEW_COS", 0.03),
            max_step_px=_free_cfg_float("PERSONAL_RESIDUAL_MAX_STEP_PX", 56.0),
            smooth_iter=int(_free_cfg_float("PERSONAL_RESIDUAL_SMOOTH_ITER", 30)),
            smooth_alpha=_free_cfg_float("PERSONAL_RESIDUAL_SMOOTH_ALPHA", 0.30),
            constraint_keep=_free_cfg_float("PERSONAL_RESIDUAL_CONSTRAINT_KEEP", 0.55),
            target_side_improve_px=_free_cfg_float("PERSONAL_RESIDUAL_TARGET_SIDE_IMPROVE_PX", 8.0),
            min_side_mean_improve_px=_free_cfg_float("PERSONAL_RESIDUAL_MIN_SIDE_MEAN_IMPROVE_PX", 2.0),
            max_stable_worsen_px=_free_cfg_float("PERSONAL_RESIDUAL_MAX_STABLE_WORSEN_PX", 0.75),
            max_global_worsen_px=_free_cfg_float("PERSONAL_RESIDUAL_MAX_GLOBAL_WORSEN_PX", 1.0),
            max_side_worsen_px=_free_cfg_float("PERSONAL_RESIDUAL_MAX_SIDE_WORSEN_PX", 1.0),
            max_moved_ratio=_free_cfg_float("PERSONAL_RESIDUAL_MAX_MOVED_RATIO", 0.12),
            stable_anchor_enabled=_free_cfg_bool("ENABLE_STABLE_FACE_ANCHORS", True),
            anchor_nose_radius_px=_free_cfg_float("STABLE_ANCHOR_NOSE_RADIUS_PX", 38.0),
            anchor_eye_radius_px=_free_cfg_float("STABLE_ANCHOR_EYE_RADIUS_PX", 30.0),
            anchor_inner_mouth_radius_px=_free_cfg_float("STABLE_ANCHOR_INNER_MOUTH_RADIUS_PX", 26.0),
            max_anchor_move_m=_free_cfg_float("STABLE_ANCHOR_MAX_MOVE_M", 0.0005),
            max_offset_jump_p95_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_P95_M", 0.012),
            max_offset_jump_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_M", 0.028),
            use_semantic_contour=_free_cfg_bool("PERSONAL_RESIDUAL_USE_SEMANTIC_CONTOUR", True),
            use_profile_side_contour=_free_cfg_bool("PERSONAL_RESIDUAL_USE_PROFILE_SIDE_CONTOUR", True),
            semantic_mask_dilate_px=_free_cfg_float("PERSONAL_RESIDUAL_SEMANTIC_MASK_DILATE_PX", 6.0),
            semantic_edit_margin_px=_free_cfg_float("PERSONAL_RESIDUAL_SEMANTIC_EDIT_MARGIN_PX", 28.0),
        )
        vertex_normals = compute_vertex_normals(verts_sub, faces_sub)
    else:
        residual_debug_dir = output_dir.parent / "debug" / "personal_residual_deform"
        residual_debug_dir.mkdir(parents=True, exist_ok=True)
        personal_residual_report = {
            "enabled": False,
            "applied": False,
            "accepted": False,
            "reason": "disabled",
        }
        with open(residual_debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(personal_residual_report, f, ensure_ascii=False, indent=2)
        _write_personal_residual_index(residual_debug_dir, personal_residual_report)
    with open(output_dir.parent / "debug" / "personal_residual_deform_summary.json", "w", encoding="utf-8") as f:
        json.dump(personal_residual_report, f, ensure_ascii=False, indent=2)

    # ── 深度置换：多视角融合（Fix 1 逐顶点采样 + Fix 2 多视角加权融合）──────
    logger.info("估计多视角深度并融合置换...")
    nose_mouth_local_report = {"enabled": False, "applied": False, "accepted": False}
    if _free_cfg_bool("ENABLE_NOSE_MOUTH_LOCAL_RESIDUAL", False):
        verts_sub, _verts_nose_mouth, nose_mouth_local_report = _nose_mouth_local_residual_deform_mesh(
            verts_base=verts_sub,
            verts_displaced=verts_sub,
            faces=faces_sub,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            per_view_results=per_view_results,
            debug_dir=output_dir.parent / "debug",
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            enabled=True,
            max_offset_m=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_OFFSET_M", 0.010),
            vertex_radius_px=_free_cfg_float("NOSE_MOUTH_LOCAL_VERTEX_RADIUS_PX", 34.0),
            protect_radius_px=_free_cfg_float("NOSE_MOUTH_LOCAL_PROTECT_RADIUS_PX", 30.0),
            max_step_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_STEP_PX", 18.0),
            front_weight=_free_cfg_float("NOSE_MOUTH_LOCAL_FRONT_WEIGHT", 1.0),
            side_weight=_free_cfg_float("NOSE_MOUTH_LOCAL_SIDE_WEIGHT", 0.35),
            nose_base_weight=_free_cfg_float("NOSE_MOUTH_LOCAL_NOSE_BASE_WEIGHT", 1.0),
            outer_mouth_weight=_free_cfg_float("NOSE_MOUTH_LOCAL_OUTER_MOUTH_WEIGHT", 1.0),
            inner_mouth_weight=_free_cfg_float("NOSE_MOUTH_LOCAL_INNER_MOUTH_WEIGHT", 0.45),
            guard_nose_width=_free_cfg_bool("NOSE_MOUTH_LOCAL_GUARD_NOSE_WIDTH", True),
            smooth_iter=int(_free_cfg_float("NOSE_MOUTH_LOCAL_SMOOTH_ITER", 18)),
            smooth_alpha=_free_cfg_float("NOSE_MOUTH_LOCAL_SMOOTH_ALPHA", 0.24),
            constraint_keep=_free_cfg_float("NOSE_MOUTH_LOCAL_CONSTRAINT_KEEP", 0.70),
            min_improve_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MIN_IMPROVE_PX", 1.0),
            min_front_improve_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MIN_FRONT_IMPROVE_PX", 1.5),
            max_protected_worsen_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_PROTECTED_WORSEN_PX", 0.35),
            max_profile_worsen_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_PROFILE_WORSEN_PX", 1.0),
            max_nose_width_abs_worsen_px=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_NOSE_WIDTH_ABS_WORSEN_PX", 2.0),
            max_moved_ratio=_free_cfg_float("NOSE_MOUTH_LOCAL_MAX_MOVED_RATIO", 0.05),
        )
        vertex_normals = compute_vertex_normals(verts_sub, faces_sub)
    else:
        nose_mouth_debug_dir = output_dir.parent / "debug" / "nose_mouth_local_residual"
        nose_mouth_debug_dir.mkdir(parents=True, exist_ok=True)
        nose_mouth_local_report = {
            "enabled": False,
            "applied": False,
            "accepted": False,
            "reason": "disabled",
        }
        with open(nose_mouth_debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(nose_mouth_local_report, f, ensure_ascii=False, indent=2)
        _write_nose_mouth_local_index(nose_mouth_debug_dir, nose_mouth_local_report)
    with open(output_dir.parent / "debug" / "nose_mouth_local_residual_summary.json", "w", encoding="utf-8") as f:
        json.dump(nose_mouth_local_report, f, ensure_ascii=False, indent=2)

    nose_region_dense_report = {"enabled": False, "applied": False, "accepted": False}
    if _free_cfg_bool("ENABLE_NOSE_REGION_DENSE_RESIDUAL", False):
        verts_sub, _verts_nose_region, nose_region_dense_report = _nose_region_dense_residual_deform_mesh(
            verts_base=verts_sub,
            verts_displaced=verts_sub,
            faces=faces_sub,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            per_view_results=per_view_results,
            debug_dir=output_dir.parent / "debug",
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            enabled=True,
            max_offset_m=_free_cfg_float("NOSE_REGION_DENSE_MAX_OFFSET_M", 0.008),
            vertex_radius_px=_free_cfg_float("NOSE_REGION_DENSE_VERTEX_RADIUS_PX", 34.0),
            roi_margin_px=_free_cfg_float("NOSE_REGION_DENSE_ROI_MARGIN_PX", 28.0),
            protect_radius_px=_free_cfg_float("NOSE_REGION_DENSE_PROTECT_RADIUS_PX", 34.0),
            max_step_px=_free_cfg_float("NOSE_REGION_DENSE_MAX_STEP_PX", 14.0),
            front_weight=_free_cfg_float("NOSE_REGION_DENSE_FRONT_WEIGHT", 1.0),
            side_weight=_free_cfg_float("NOSE_REGION_DENSE_SIDE_WEIGHT", 0.20),
            bridge_weight=_free_cfg_float("NOSE_REGION_DENSE_BRIDGE_WEIGHT", 0.45),
            tip_weight=_free_cfg_float("NOSE_REGION_DENSE_TIP_WEIGHT", 0.80),
            wing_weight=_free_cfg_float("NOSE_REGION_DENSE_WING_WEIGHT", 1.0),
            guard_nose_width=_free_cfg_bool("NOSE_REGION_DENSE_GUARD_NOSE_WIDTH", True),
            smooth_iter=int(_free_cfg_float("NOSE_REGION_DENSE_SMOOTH_ITER", 14)),
            smooth_alpha=_free_cfg_float("NOSE_REGION_DENSE_SMOOTH_ALPHA", 0.20),
            constraint_keep=_free_cfg_float("NOSE_REGION_DENSE_CONSTRAINT_KEEP", 0.64),
            min_improve_px=_free_cfg_float("NOSE_REGION_DENSE_MIN_IMPROVE_PX", 0.25),
            min_front_improve_px=_free_cfg_float("NOSE_REGION_DENSE_MIN_FRONT_IMPROVE_PX", 0.40),
            max_protected_worsen_px=_free_cfg_float("NOSE_REGION_DENSE_MAX_PROTECTED_WORSEN_PX", 0.35),
            max_profile_worsen_px=_free_cfg_float("NOSE_REGION_DENSE_MAX_PROFILE_WORSEN_PX", 1.0),
            max_nose_width_abs_worsen_px=_free_cfg_float("NOSE_REGION_DENSE_MAX_NOSE_WIDTH_ABS_WORSEN_PX", 1.0),
            max_moved_ratio=_free_cfg_float("NOSE_REGION_DENSE_MAX_MOVED_RATIO", 0.045),
        )
        vertex_normals = compute_vertex_normals(verts_sub, faces_sub)
    else:
        nose_region_debug_dir = output_dir.parent / "debug" / "nose_region_dense_residual"
        nose_region_debug_dir.mkdir(parents=True, exist_ok=True)
        nose_region_dense_report = {
            "enabled": False,
            "applied": False,
            "accepted": False,
            "reason": "disabled",
        }
        with open(nose_region_debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(nose_region_dense_report, f, ensure_ascii=False, indent=2)
        _write_nose_region_dense_index(nose_region_debug_dir, nose_region_dense_report)
    with open(output_dir.parent / "debug" / "nose_region_dense_residual_summary.json", "w", encoding="utf-8") as f:
        json.dump(nose_region_dense_report, f, ensure_ascii=False, indent=2)

    import cv2 as _cv2
    from scipy.ndimage import distance_transform_edt
    class _DepthDisplacementDisabled(Exception):
        pass

    try:
        if not enable_depth_displacement:
            logger.info("Depth-Anything geometry displacement disabled for stable pipeline")
            raise _DepthDisplacementDisabled()
        depth_pipe = load_depth_model(depth_model_dir, depth_model_size, device)

        N = len(verts_sub)
        disp_accum   = np.zeros(N, dtype=np.float64)
        weight_accum = np.zeros(N, dtype=np.float64)

        debug_dir = output_dir.parent / "debug"
        debug_dir.mkdir(exist_ok=True)

        for view_name, view_result in per_view_results.items():
            if view_name not in preprocessed_views or view_name not in intrinsics:
                continue
            view_image = preprocessed_views[view_name]["image"]
            # 膨胀 face_mask，覆盖鼻尖/鼻翼/眼周（MediaPipe 往往不覆盖这些区域）
            _kern = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (25, 25))
            view_mask  = _cv2.dilate(preprocessed_views[view_name]["face_mask"], _kern)
            K_v = intrinsics[view_name]
            R_v = view_result["R"]
            t_v = view_result["t"]
            H_v, W_v = view_image.shape[:2]

            logger.info(f"  视角 [{view_name}] 深度估计...")
            d_pred_v = estimate_depth(depth_pipe, view_image)   # [0,1] float32

            # 双边滤波（步骤1：消除高频噪点，保留边缘）
            d_pred_f = _cv2.bilateralFilter(
                d_pred_v.astype(np.float32), d=9, sigmaColor=0.08, sigmaSpace=15,
            )

            # 用原始 5023 顶点做尺度对齐（稀疏快速）
            d_sparse_v, sp_u, sp_v = project_vertices_sparse(
                verts_final, K_v, R_v, t_v, (H_v, W_v)
            )
            valid_pts = (
                (view_mask[sp_v, sp_u] > 0) &
                np.isfinite(d_sparse_v) &
                (d_pred_f[sp_v, sp_u] > 0.01)
            )
            s_v, t_off_v = align_depth_scale_sparse(
                d_pred_f[sp_v, sp_u], d_sparse_v, valid_pts
            )
            d_aligned_v = (s_v * d_pred_f + t_off_v).astype(np.float32)

            # feathering 掩码
            dist_v   = distance_transform_edt(view_mask > 0).astype(np.float32)
            feather_v = np.clip(dist_v / 40.0, 0.0, 1.0)

            # 计算该视角相机朝向（世界空间，从相机指向场景）
            # R_v 是 world-to-camera，所以相机 Z 轴在世界空间 = R_v[2, :] 的反方向
            view_dir_world = -R_v[2, :]   # (3,) 从相机指向场景的方向（世界空间）

            # 法线与视线夹角余弦（衡量该视角对此顶点的可见度）
            cos_w = np.clip(vertex_normals @ view_dir_world, 0.0, None)  # (N,)

            # 逐顶点投影 + 采样
            v_cam_v = (R_v @ verts_sub.T + t_v[:, None]).T     # (N,3) 相机空间
            z_cam_v = v_cam_v[:, 2]
            v_hom_v = (K_v @ v_cam_v.T).T
            u_v = (v_hom_v[:, 0] / np.clip(z_cam_v, 1e-6, None)).astype(int)
            p_v = (v_hom_v[:, 1] / np.clip(z_cam_v, 1e-6, None)).astype(int)

            in_bounds_v = (u_v >= 0) & (u_v < W_v) & (p_v >= 0) & (p_v < H_v) & (z_cam_v > 1e-4)
            s_u = np.clip(u_v, 0, W_v - 1)
            s_p = np.clip(p_v, 0, H_v - 1)

            d_at_v   = d_aligned_v[s_p, s_u].astype(np.float64)
            disp_v   = d_at_v - z_cam_v

            in_face_v = view_mask[s_p, s_u] > 0
            good_nrm  = vertex_normals[:, 2] >= -0.1
            valid_v   = in_bounds_v & in_face_v & good_nrm & np.isfinite(disp_v)

            feat_v = feather_v[s_p, s_u].astype(np.float64)

            # 权重 = feather × 法线可见度
            w_v = np.where(valid_v, feat_v * cos_w, 0.0)
            disp_accum   += w_v * disp_v
            weight_accum += w_v

            # 调试输出（仅正脸视角）
            if view_name == front_name:
                d_3dmm_dbg = _sparse_to_dense_depth(d_sparse_v, sp_u, sp_v, (H_v, W_v))
                _save_depth_debug(d_pred_v, d_3dmm_dbg, d_aligned_v,
                                  d_aligned_v - d_3dmm_dbg, debug_dir)

        # 融合：加权均值位移，先用 mask 规避无效除法告警
        disp_final = np.zeros(N, dtype=np.float64)
        valid_weight = weight_accum > 1e-6
        disp_final[valid_weight] = disp_accum[valid_weight] / weight_accum[valid_weight]
        disp_final = np.clip(disp_final, -max_displacement, max_displacement)

        abs_disp = np.abs(disp_final)
        moved_mask = abs_disp > 1e-8
        moved_count = int(moved_mask.sum())
        moved_ratio = float(moved_count) / float(N) if N > 0 else 0.0
        logger.info(
            "深度置换统计: moved=%d/%d (%.2f%%), mean_abs=%.6f m, max_abs=%.6f m"
            % (
                moved_count,
                N,
                moved_ratio * 100.0,
                float(abs_disp[moved_mask].mean()) if moved_count > 0 else 0.0,
                float(abs_disp.max()) if N > 0 else 0.0,
            )
        )

        logger.info(f"应用多视角融合深度置换到 {N} 个顶点...")
        verts_displaced = verts_sub + vertex_normals * disp_final[:, None]

    except _DepthDisplacementDisabled:
        verts_displaced = np.array(verts_sub, copy=True)
    except Exception as e:
        import traceback
        logger.warning(f"深度置换失败（{e}），跳过置换步骤，使用细分 3DMM Mesh")
        logger.debug(traceback.format_exc())
        verts_displaced = verts_sub

    # ── 步骤4：Laplacian 平滑（消除深度置换尖刺）──────────────────────────
    if enable_depth_displacement:
        logger.info("Laplacian 平滑置换后 Mesh...")
        try:
            _sm = _trimesh.Trimesh(vertices=verts_displaced, faces=faces_sub, process=False)
            _trimesh.smoothing.filter_laplacian(_sm, iterations=3, lamb=0.25)  # 减少迭代防止鼻尖过度平滑
            verts_displaced = np.array(_sm.vertices)
        except Exception as _e:
            logger.warning(f"Laplacian 平滑失败（{_e}），跳过")
    else:
        logger.info("Stable pipeline: skipping displacement smoothing")

    try:
        from src import config as _free_cfg
    except Exception:
        _free_cfg = None

    def _free_cfg_float(name: str, default: float) -> float:
        if _free_cfg is None:
            return default
        try:
            return float(getattr(_free_cfg, name, default))
        except Exception:
            return default

    def _free_cfg_bool(name: str, default: bool) -> bool:
        if _free_cfg is None:
            return default
        try:
            return bool(getattr(_free_cfg, name, default))
        except Exception:
            return default

    free_identity_report = {"enabled": False, "applied": False}
    if _free_cfg_bool("ENABLE_FREE_IDENTITY_DEFORM", False):
        verts_sub, verts_displaced, free_identity_report = _free_identity_deform_mesh(
            verts_base=verts_sub,
            verts_displaced=verts_displaced,
            faces=faces_sub,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            per_view_results=per_view_results,
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            debug_dir=output_dir.parent / "debug",
            enabled=True,
            max_offset_m=_free_cfg_float("FREE_IDENTITY_MAX_OFFSET_M", 0.028),
            max_step_px=_free_cfg_float("FREE_IDENTITY_MAX_STEP_PX", 45.0),
            radius_px=_free_cfg_float("FREE_IDENTITY_RADIUS_PX", 42.0),
            contour_radius_px=_free_cfg_float("FREE_IDENTITY_CONTOUR_RADIUS_PX", 56.0),
            mouth_radius_px=_free_cfg_float("FREE_IDENTITY_MOUTH_RADIUS_PX", 36.0),
            view_weight_front=_free_cfg_float("FREE_IDENTITY_VIEW_WEIGHT_FRONT", 1.0),
            view_weight_side=_free_cfg_float("FREE_IDENTITY_VIEW_WEIGHT_SIDE", 0.22),
            min_view_cos=_free_cfg_float("FREE_IDENTITY_MIN_VIEW_COS", 0.02),
            smooth_iter=int(_free_cfg_float("FREE_IDENTITY_SMOOTH_ITER", 18)),
            smooth_alpha=_free_cfg_float("FREE_IDENTITY_SMOOTH_ALPHA", 0.22),
            constraint_keep=_free_cfg_float("FREE_IDENTITY_CONSTRAINT_KEEP", 0.78),
            min_improve_px=_free_cfg_float("FREE_IDENTITY_MIN_IMPROVE_PX", 0.15),
            max_stable_worsen_px=_free_cfg_float("FREE_IDENTITY_MAX_STABLE_WORSEN_PX", 1.4),
            max_side_worsen_px=_free_cfg_float("FREE_IDENTITY_MAX_SIDE_WORSEN_PX", 2.5),
            max_moved_ratio=_free_cfg_float("FREE_IDENTITY_MAX_MOVED_RATIO", 0.18),
            stable_anchor_enabled=_free_cfg_bool("ENABLE_STABLE_FACE_ANCHORS", True),
            anchor_nose_radius_px=_free_cfg_float("STABLE_ANCHOR_NOSE_RADIUS_PX", 38.0),
            anchor_eye_radius_px=_free_cfg_float("STABLE_ANCHOR_EYE_RADIUS_PX", 30.0),
            anchor_inner_mouth_radius_px=_free_cfg_float("STABLE_ANCHOR_INNER_MOUTH_RADIUS_PX", 26.0),
            max_anchor_move_m=_free_cfg_float("STABLE_ANCHOR_MAX_MOVE_M", 0.0005),
            max_offset_jump_p95_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_P95_M", 0.012),
            max_offset_jump_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_M", 0.035),
            multiview_enabled=_free_cfg_bool("ENABLE_MULTIVIEW_DEFORM_VALIDATION", True),
            multiview_min_front_improve_px=_free_cfg_float("MULTIVIEW_MIN_FRONT_IMPROVE_PX", 0.25),
            multiview_min_overall_improve_px=_free_cfg_float("MULTIVIEW_MIN_OVERALL_IMPROVE_PX", 0.25),
            multiview_max_side_worsen_px=_free_cfg_float("MULTIVIEW_MAX_SIDE_WORSEN_PX", 0.35),
            multiview_max_side_mean_worsen_px=_free_cfg_float("MULTIVIEW_MAX_SIDE_MEAN_WORSEN_PX", 0.05),
            multiview_require_side_views=_free_cfg_bool("MULTIVIEW_REQUIRE_SIDE_VIEWS", True),
        )
    else:
        free_identity_report = {
            "enabled": False,
            "applied": False,
            "accepted": False,
            "reason": "disabled after over-deformation failure",
        }
        identity_debug_dir = output_dir.parent / "debug" / "free_identity_deform"
        identity_debug_dir.mkdir(parents=True, exist_ok=True)
        with open(identity_debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(free_identity_report, f, ensure_ascii=False, indent=2)
    with open(output_dir.parent / "debug" / "free_identity_deform_summary.json", "w", encoding="utf-8") as f:
        json.dump(free_identity_report, f, ensure_ascii=False, indent=2)

    free_deform_report = {"enabled": False, "applied": False}
    if _free_cfg_bool("ENABLE_FREE_FACE_DEFORM", False):
        verts_sub, verts_displaced, free_deform_report = _free_face_deform_mesh(
            verts_base=verts_sub,
            verts_displaced=verts_displaced,
            faces=faces_sub,
            view_data=view_data,
            preprocessed_views=preprocessed_views,
            intrinsics=intrinsics,
            per_view_results=per_view_results,
            debug_dir=output_dir.parent / "debug",
            lmk_vertex_indices=lmk_vertex_indices,
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            enabled=True,
            max_offset_m=_free_cfg_float("FREE_FACE_MAX_OFFSET_M", 0.018),
            boundary_band_px=_free_cfg_float("FREE_FACE_BOUNDARY_BAND_PX", 95.0),
            search_margin_px=_free_cfg_float("FREE_FACE_SEARCH_MARGIN_PX", 80.0),
            stable_protect_radius_px=_free_cfg_float("FREE_FACE_STABLE_PROTECT_RADIUS_PX", 58.0),
            row_step=int(_free_cfg_float("FREE_FACE_ROW_STEP", 6)),
            row_sigma=_free_cfg_float("FREE_FACE_ROW_SIGMA", 8.0),
            tau=_free_cfg_float("FREE_FACE_SOFTMIN_TAU", 10.0),
            envelope_topk=int(_free_cfg_float("FREE_FACE_ENVELOPE_TOPK", 28)),
            view_weight_front=_free_cfg_float("FREE_FACE_VIEW_WEIGHT_FRONT", 1.0),
            view_weight_side=_free_cfg_float("FREE_FACE_VIEW_WEIGHT_SIDE", 0.75),
            min_view_cos=_free_cfg_float("FREE_FACE_MIN_VIEW_COS", 0.03),
            max_step_px=_free_cfg_float("FREE_FACE_MAX_STEP_PX", 36.0),
            smooth_iter=int(_free_cfg_float("FREE_FACE_SMOOTH_ITER", 35)),
            smooth_alpha=_free_cfg_float("FREE_FACE_SMOOTH_ALPHA", 0.35),
            constraint_keep=_free_cfg_float("FREE_FACE_CONSTRAINT_KEEP", 0.45),
            accept_min_improve_px=_free_cfg_float("FREE_FACE_ACCEPT_MIN_IMPROVE_PX", 0.25),
            max_side_worsen_px=_free_cfg_float("FREE_FACE_MAX_SIDE_WORSEN_PX", 2.0),
            max_moved_ratio=_free_cfg_float("FREE_FACE_MAX_MOVED_RATIO", 0.08),
            stable_anchor_enabled=_free_cfg_bool("ENABLE_STABLE_FACE_ANCHORS", True),
            anchor_nose_radius_px=_free_cfg_float("STABLE_ANCHOR_NOSE_RADIUS_PX", 38.0),
            anchor_eye_radius_px=_free_cfg_float("STABLE_ANCHOR_EYE_RADIUS_PX", 30.0),
            anchor_inner_mouth_radius_px=_free_cfg_float("STABLE_ANCHOR_INNER_MOUTH_RADIUS_PX", 26.0),
            max_anchor_move_m=_free_cfg_float("STABLE_ANCHOR_MAX_MOVE_M", 0.0005),
            max_offset_jump_p95_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_P95_M", 0.012),
            max_offset_jump_m=_free_cfg_float("DEFORM_GUARD_MAX_OFFSET_JUMP_M", 0.035),
            multiview_enabled=_free_cfg_bool("ENABLE_MULTIVIEW_DEFORM_VALIDATION", True),
            multiview_min_front_improve_px=_free_cfg_float("MULTIVIEW_MIN_FRONT_IMPROVE_PX", 0.25),
            multiview_min_overall_improve_px=_free_cfg_float("MULTIVIEW_MIN_OVERALL_IMPROVE_PX", 0.25),
            multiview_max_side_worsen_px=_free_cfg_float("MULTIVIEW_MAX_SIDE_WORSEN_PX", 0.35),
            multiview_max_side_mean_worsen_px=_free_cfg_float("MULTIVIEW_MAX_SIDE_MEAN_WORSEN_PX", 0.05),
            multiview_require_side_views=_free_cfg_bool("MULTIVIEW_REQUIRE_SIDE_VIEWS", True),
        )
        with open(output_dir.parent / "debug" / "free_face_deform_summary.json", "w", encoding="utf-8") as f:
            json.dump(free_deform_report, f, ensure_ascii=False, indent=2)
    else:
        free_deform_report = {
            "enabled": False,
            "applied": False,
            "accepted": False,
            "reason": "disabled; personal residual deformation is the active face-shape experiment",
        }
        free_deform_dir = output_dir.parent / "debug" / "free_face_deform"
        free_deform_dir.mkdir(parents=True, exist_ok=True)
        with open(free_deform_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(free_deform_report, f, ensure_ascii=False, indent=2)
        with open(output_dir.parent / "debug" / "free_face_deform_summary.json", "w", encoding="utf-8") as f:
            json.dump(free_deform_report, f, ensure_ascii=False, indent=2)

    try:
        from src import config as cfg
        force_closed_eyes = bool(getattr(cfg, "FORCE_CLOSED_EYES", False))
        closed_eye_strength = float(getattr(cfg, "CLOSED_EYE_GEOMETRY_STRENGTH", 0.95))
    except Exception:
        force_closed_eyes = False
        closed_eye_strength = 0.95

    if force_closed_eyes and front_name in view_data and front_name in per_view_results:
        logger.info("默认闭眼：按正脸闭眼线直接闭合眼部 mesh 顶点...")
        front_result = per_view_results[front_name]
        front_lmks = view_data[front_name]["lmk_2d"]
        front_img = preprocessed_views[front_name]["image"]
        front_base_normals = compute_vertex_normals(verts_sub, faces_sub)
        front_depth_normals = compute_vertex_normals(verts_displaced, faces_sub)
        verts_sub = _force_close_eye_geometry(
            verts_sub,
            intrinsics[front_name],
            front_result["R"],
            front_result["t"],
            front_lmks,
            faces=faces_sub,
            vertex_normals=front_base_normals,
            image_shape=front_img.shape[:2],
            strength=closed_eye_strength,
            debug_image=front_img,
            debug_path=output_dir.parent / "debug" / "closed_eye_geometry" / "front_closed_eye_vertices.png",
        )
        verts_displaced = _force_close_eye_geometry(
            verts_displaced,
            intrinsics[front_name],
            front_result["R"],
            front_result["t"],
            front_lmks,
            faces=faces_sub,
            vertex_normals=front_depth_normals,
            image_shape=front_img.shape[:2],
            strength=closed_eye_strength,
        )

    # ── 导出 .obj ─────────────────────────────────────────────────────────
    base_output_path = output_dir / "face_mesh.obj"
    neutral_output_path = output_dir / "face_mesh_neutral.obj"
    depth_output_path = output_dir / "face_mesh_with_depth.obj"
    base_glb_path = output_dir / "face_mesh.glb"
    neutral_glb_path = output_dir / "face_mesh_neutral.glb"
    depth_glb_path = output_dir / "face_mesh_with_depth.glb"
    export_mesh_obj(verts_sub, faces_sub, uv_verts_sub, uv_faces_sub, base_output_path)
    export_mesh_obj(
        verts_neutral_sub,
        faces_neutral_sub,
        uv_verts_sub,
        uv_faces_sub,
        neutral_output_path,
    )
    export_mesh_obj(verts_displaced, faces_sub, uv_verts_sub, uv_faces_sub, depth_output_path)
    export_mesh_glb(verts_sub, faces_sub, uv_verts_sub, uv_faces_sub, base_glb_path)
    export_mesh_glb(
        verts_neutral_sub,
        faces_neutral_sub,
        uv_verts_sub,
        uv_faces_sub,
        neutral_glb_path,
    )
    export_mesh_glb(verts_displaced, faces_sub, uv_verts_sub, uv_faces_sub, depth_glb_path)

    # ── 保存相机参数（Phase 3 纹理映射需要）────────────────────────────────
    cam_data = {
        "views": {},
        "flame_space": "Y_up_normalized",
    }
    for view_name, res in per_view_results.items():
        K_v = intrinsics[view_name]
        cam_data["views"][view_name] = {
            "K":  K_v.tolist(),
            "R":  res["R"].tolist(),
            "t":  res["t"].tolist(),
        }
    cam_path = output_dir / "cameras.json"
    with open(str(cam_path), "w") as f:
        json.dump(cam_data, f, indent=2)
    logger.info(f"相机参数已保存: {cam_path}")

    # ── 调试：投影 mesh 到正脸图验证对齐 ───────────────────────────────────
    _save_projection_debug(
        verts_displaced, faces_sub,
        intrinsics[front_name],
        per_view_results[front_name]["R"],
        per_view_results[front_name]["t"],
        preprocessed_views[front_name]["image"],
        preprocessed_views[front_name]["face_mask"],
        output_dir.parent / "debug" / "mesh_projection_front.jpg",
    )

    return depth_glb_path


def _save_init_shape_debug(init_result: dict, debug_dir: Path):
    """Save MICA/initializer shape summary JSON."""
    import json
    shape = init_result.get("shape")
    data = {
        "backend":      init_result["backend"],
        "shape_source": init_result["shape_source"],
        "view_status":  init_result["view_status"],
        "shape_norm":   float(np.linalg.norm(shape)) if shape is not None else 0.0,
        "shape_params": shape.tolist() if shape is not None else [],
    }
    out = debug_dir / "init_mica_shape.json"
    with open(str(out), "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"初始化 shape 摘要已保存: {out}")


def _flame_vertices_for_init(flame, shape_np, exp_np: Optional[np.ndarray]) -> np.ndarray:
    shape_use = shape_np if shape_np is not None else np.zeros(flame.n_shape, dtype=np.float32)
    exp_use = np.zeros(flame.n_exp, dtype=np.float32)
    if exp_np is not None:
        exp_arr = np.asarray(exp_np, dtype=np.float32).reshape(-1)
        exp_use[:min(len(exp_arr), flame.n_exp)] = exp_arr[:flame.n_exp]

    model_device = flame.v_template.device
    model_dtype = flame.v_template.dtype
    with torch.no_grad():
        return flame(
            torch.as_tensor(shape_use, device=model_device, dtype=model_dtype),
            torch.as_tensor(exp_use, device=model_device, dtype=model_dtype),
        ).cpu().numpy()


def _landmark_points_3d(
    vertices: np.ndarray,
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    if lmk_tri_vidx is not None and lmk_bary_coords is not None:
        return (
            vertices[lmk_tri_vidx[:, 0]] * lmk_bary_coords[:, 0:1] +
            vertices[lmk_tri_vidx[:, 1]] * lmk_bary_coords[:, 1:2] +
            vertices[lmk_tri_vidx[:, 2]] * lmk_bary_coords[:, 2:3]
        )
    return vertices[lmk_vertex_indices]


def _audit_and_repair_initial_poses(
    view_data: dict,
    init_result: dict,
    init_exps: dict,
    init_shape,
    flame,
    lmk_vertex_indices,
    lmk_tri_vidx,
    lmk_bary_coords,
    preprocessed_views: dict,
    intrinsics: dict,
    debug_dir: Path,
):
    import json

    out_dir = debug_dir / "init_pose_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for view_name, vd in view_data.items():
        image = preprocessed_views[view_name]["image"]
        target_lmk = vd["lmk_2d"]
        K = intrinsics[view_name]
        verts_init = _flame_vertices_for_init(flame, init_shape, init_exps.get(view_name))

        orig_R = np.asarray(vd["R_init"], dtype=np.float64)
        orig_t = np.asarray(vd["t_init"], dtype=np.float64).reshape(3)
        orig_mean, orig_max, orig_errors = _save_landmark_reprojection_debug(
            vertices=verts_init,
            K=K,
            R=orig_R,
            t=orig_t,
            image=image,
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=out_dir / f"{view_name}_01_original.png",
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
            return_errors=True,
        )

        is_bad = (orig_mean > INIT_POSE_MEAN_BAD_PX) or (orig_max > INIT_POSE_MAX_BAD_PX)
        pnp_mean = None
        pnp_max = None
        pnp_errors = None
        selected = "原始初值"
        action = "通过"
        reason = "原始初值误差在阈值内，未重算 PnP。"

        if is_bad:
            try:
                flame_lmk_3d = _landmark_points_3d(
                    verts_init,
                    lmk_vertex_indices,
                    lmk_tri_vidx,
                    lmk_bary_coords,
                )
                pnp_R, pnp_t = estimate_pose_from_landmarks_pnp(target_lmk, flame_lmk_3d, K)
                pnp_R = np.asarray(pnp_R, dtype=np.float64)
                pnp_t = np.asarray(pnp_t, dtype=np.float64).reshape(3)
                pnp_mean, pnp_max, pnp_errors = _save_landmark_reprojection_debug(
                    vertices=verts_init,
                    K=K,
                    R=pnp_R,
                    t=pnp_t,
                    image=image,
                    target_landmarks=target_lmk,
                    lmk_vertex_indices=lmk_vertex_indices,
                    out_path=out_dir / f"{view_name}_02_recomputed_pnp.png",
                    lmk_tri_vidx=lmk_tri_vidx,
                    lmk_bary_coords=lmk_bary_coords,
                    return_errors=True,
                )
                orig_stable_score = _stable_landmark_score(orig_errors)
                pnp_stable_score = _stable_landmark_score(pnp_errors)
                orig_groups = _landmark_group_stats(orig_errors)
                pnp_groups = _landmark_group_stats(pnp_errors)
                nose_ok = pnp_groups["鼻子"]["mean_px"] <= orig_groups["鼻子"]["mean_px"] + 5.0
                stable_ok = pnp_stable_score < (orig_stable_score - INIT_POSE_MIN_IMPROVE_PX)
                if np.isfinite(pnp_mean) and stable_ok and nose_ok:
                    vd["R_init"] = pnp_R.astype(np.float32)
                    vd["t_init"] = pnp_t.astype(np.float32)
                    pv = init_result["per_view"].setdefault(view_name, {})
                    pv["R_init"] = vd["R_init"]
                    pv["t_init"] = vd["t_init"]
                    selected = "重算 PnP"
                    action = "已替换"
                    reason = f"原始初值超阈值，重算 PnP 后稳定区域误差降低 {orig_stable_score - pnp_stable_score:.2f}px。"
                else:
                    action = "保留原始"
                    reason = "原始初值超阈值，但重算 PnP 没有同时满足稳定区域变好、鼻子不明显变坏。"
            except Exception as exc:
                action = "重算失败"
                reason = f"原始初值超阈值，但重算 PnP 失败：{exc}"
        else:
            _save_landmark_reprojection_debug(
                vertices=verts_init,
                K=K,
                R=orig_R,
                t=orig_t,
                image=image,
                target_landmarks=target_lmk,
                lmk_vertex_indices=lmk_vertex_indices,
                out_path=out_dir / f"{view_name}_02_recomputed_pnp.png",
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_bary_coords,
            )

        final_R = np.asarray(vd["R_init"], dtype=np.float64)
        final_t = np.asarray(vd["t_init"], dtype=np.float64).reshape(3)
        final_mean, final_max, final_errors = _save_landmark_reprojection_debug(
            vertices=verts_init,
            K=K,
            R=final_R,
            t=final_t,
            image=image,
            target_landmarks=target_lmk,
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=out_dir / f"{view_name}_03_selected.png",
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
            return_errors=True,
        )
        face_mask = preprocessed_views[view_name].get("face_mask")
        mesh_image = f"{view_name}_04_selected_mesh.png"
        if face_mask is not None:
            _save_projection_debug(
                verts_init,
                flame.faces.numpy(),
                K,
                final_R,
                final_t,
                image,
                face_mask,
                out_dir / mesh_image,
            )

        records.append({
            "view": view_name,
            "original_mean_px": round(orig_mean, 3),
            "original_max_px": round(orig_max, 3),
            "bad_initial_pose": bool(is_bad),
            "pnp_mean_px": None if pnp_mean is None else round(float(pnp_mean), 3),
            "pnp_max_px": None if pnp_max is None else round(float(pnp_max), 3),
            "selected": selected,
            "final_mean_px": round(final_mean, 3),
            "final_max_px": round(final_max, 3),
            "action": action,
            "reason": reason,
            "group_stats": {
                "original": _landmark_group_stats(orig_errors),
                "pnp": None if pnp_errors is None else _landmark_group_stats(pnp_errors),
                "final": _landmark_group_stats(final_errors),
            },
            "images": {
                "original": f"{view_name}_01_original.png",
                "pnp": f"{view_name}_02_recomputed_pnp.png",
                "selected": f"{view_name}_03_selected.png",
                "mesh": mesh_image,
            },
        })
        logger.info(
            f"  [{view_name}] init pose audit: original mean={orig_mean:.2f}px max={orig_max:.2f}px, "
            f"selected={selected}, final mean={final_mean:.2f}px max={final_max:.2f}px"
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    _write_init_pose_quality_html(records, out_dir)


def _write_init_pose_quality_html(records: list, out_dir: Path):
    import html

    def fmt(v):
        return "未计算" if v is None else f"{float(v):.2f}"

    def fmt_group(stage_stats, group_name):
        if not stage_stats:
            return "未计算"
        return fmt(stage_stats[group_name]["mean_px"])

    def make_group_table(rec):
        group_stats = rec.get("group_stats", {})
        body = []
        final_stats = group_stats.get("final") or {}
        for group_name, _idx in LMK_ERROR_GROUPS:
            final_mean = final_stats.get(group_name, {}).get("mean_px", 0.0)
            cls = "bad" if final_mean > 25.0 else ("warn" if final_mean > 12.0 else "ok")
            body.append(
                "<tr>"
                f"<td>{html.escape(group_name)}</td>"
                f"<td>{fmt_group(group_stats.get('original'), group_name)}</td>"
                f"<td>{fmt_group(group_stats.get('pnp'), group_name)}</td>"
                f"<td class='{cls}'>{fmt_group(group_stats.get('final'), group_name)}</td>"
                "</tr>"
            )
        return (
            "<table class='group-table'>"
            "<thead><tr><th>关键点类别</th><th>原始平均误差</th><th>PnP平均误差</th><th>最终平均误差</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
        )

    rows = []
    cards = []
    for rec in records:
        status_cls = "bad" if rec["bad_initial_pose"] else "ok"
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec['view'])}</td>"
            f"<td class='{status_cls}'>{'坏初值' if rec['bad_initial_pose'] else '通过'}</td>"
            f"<td>{fmt(rec['original_mean_px'])} / {fmt(rec['original_max_px'])}</td>"
            f"<td>{fmt(rec['pnp_mean_px'])} / {fmt(rec['pnp_max_px'])}</td>"
            f"<td>{html.escape(rec['selected'])}</td>"
            f"<td>{fmt(rec['final_mean_px'])} / {fmt(rec['final_max_px'])}</td>"
            f"<td>{html.escape(rec['reason'])}</td>"
            "</tr>"
        )
        imgs = rec["images"]
        groups_html = make_group_table(rec)
        cards.append(
            f"""
            <section class="view-block">
              <h2>{html.escape(rec['view'])} 视角</h2>
              <p><b>结论：</b>{html.escape(rec['action'])}。{html.escape(rec['reason'])}</p>
              {groups_html}
              <div class="image-grid">
                <figure><img src="{html.escape(imgs['original'])}"><figcaption>1. 原始初值重投影</figcaption></figure>
                <figure><img src="{html.escape(imgs['pnp'])}"><figcaption>2. 重算 PnP 候选</figcaption></figure>
                <figure><img src="{html.escape(imgs['selected'])}"><figcaption>3. 最终采用结果</figcaption></figure>
                <figure><img src="{html.escape(imgs['mesh'])}"><figcaption>4. 最终初始 mesh 投影</figcaption></figure>
              </div>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>初始位姿质量检查</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ padding: 28px 36px 18px; background: #18212f; color: white; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    header p {{ margin: 6px 0; color: #d8dee8; line-height: 1.6; }}
    main {{ padding: 24px 36px 48px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 28px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef2f7; font-weight: 700; }}
    .ok {{ color: #047857; font-weight: 700; }}
    .warn {{ color: #b7791f; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .view-block {{ margin: 0 0 34px; padding: 22px 0 0; border-top: 2px solid #d8dee8; }}
    h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .view-block p {{ margin: 0 0 14px; line-height: 1.6; }}
    .group-table {{ margin: 0 0 16px; }}
    .group-table th, .group-table td {{ font-size: 13px; padding: 8px 10px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d7dde6; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px 12px; font-size: 14px; color: #4b5563; background: #fbfcfe; }}
    .legend {{ margin-top: 8px; font-size: 14px; }}
    @media (max-width: 900px) {{ .image-grid {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} header {{ padding: 22px 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>初始位姿质量检查 + 坏初值自动重算 PnP</h1>
    <p>判断规则：平均重投影误差 &gt; {INIT_POSE_MEAN_BAD_PX:.0f}px，或最大误差 &gt; {INIT_POSE_MAX_BAD_PX:.0f}px，就标记为坏初值。</p>
    <p>替换规则：只有重算 PnP 后平均误差至少降低 {INIT_POSE_MIN_IMPROVE_PX:.0f}px，才会替换当前初始位姿。</p>
    <p class="legend">图中绿色点是真实 2D 关键点，红色点是模型投影点，黄色线表示误差距离。</p>
  </header>
  <main>
    <table>
      <thead>
        <tr><th>视角</th><th>状态</th><th>原始 mean/max(px)</th><th>PnP mean/max(px)</th><th>最终采用</th><th>最终 mean/max(px)</th><th>说明</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info(f"Initial pose quality HTML saved: {out_dir / 'index.html'}")


def _save_optimized_pose_quality_debug(
    shape_opt: np.ndarray,
    per_view_results: dict,
    flame,
    lmk_vertex_indices,
    lmk_tri_vidx,
    lmk_bary_coords,
    view_data: dict,
    preprocessed_views: dict,
    intrinsics: dict,
    debug_dir: Path,
    device: str,
):
    import json

    out_dir = debug_dir / "optimized_pose_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    model_device = flame.v_template.device
    model_dtype = flame.v_template.dtype

    for view_name, view_result in per_view_results.items():
        with torch.no_grad():
            verts_view = flame(
                torch.as_tensor(shape_opt, device=model_device, dtype=model_dtype),
                torch.as_tensor(view_result["exp"], device=model_device, dtype=model_dtype),
            ).cpu().numpy()

        mean_err, max_err, errors = _save_landmark_reprojection_debug(
            vertices=verts_view,
            K=intrinsics[view_name],
            R=view_result["R"],
            t=view_result["t"],
            image=preprocessed_views[view_name]["image"],
            target_landmarks=view_data[view_name]["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=out_dir / f"{view_name}_optimized_reprojection.png",
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
            return_errors=True,
        )
        geometry_stats = _landmark_subset_stats(errors, LMK_GEOMETRY_IDX)
        group_stats = _landmark_group_stats(errors)

        face_mask = preprocessed_views[view_name].get("face_mask")
        mesh_image = f"{view_name}_optimized_mesh.png"
        if face_mask is not None:
            _save_projection_debug(
                verts_view,
                flame.faces.cpu().numpy(),
                intrinsics[view_name],
                view_result["R"],
                view_result["t"],
                preprocessed_views[view_name]["image"],
                face_mask,
                out_dir / mesh_image,
            )

        records.append({
            "view": view_name,
            "mean_px": round(float(mean_err), 3),
            "max_px": round(float(max_err), 3),
            "geometry_mean_px": geometry_stats["mean_px"],
            "geometry_max_px": geometry_stats["max_px"],
            "geometry_group_stats": _landmark_named_stats(errors, LMK_GEOMETRY_GROUPS),
            "appearance_group_stats": _landmark_named_stats(errors, LMK_APPEARANCE_GROUPS),
            "group_stats": group_stats,
            "images": {
                "reprojection": f"{view_name}_optimized_reprojection.png",
                "mesh": mesh_image,
            },
        })

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    _write_optimized_pose_quality_html(records, out_dir)


def _write_optimized_pose_quality_html(records: list, out_dir: Path):
    import html

    def fmt(v):
        return f"{float(v):.2f}"

    def err_class(mean_px: float):
        return "bad" if mean_px > 25.0 else ("warn" if mean_px > 12.0 else "ok")

    rows = []
    cards = []
    for rec in records:
        brow_stats = rec["appearance_group_stats"]["眉毛"]
        geom_cls = err_class(rec["geometry_mean_px"])
        brow_cls = err_class(brow_stats["mean_px"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec['view'])}</td>"
            f"<td class='{geom_cls}'>{fmt(rec['geometry_mean_px'])}</td>"
            f"<td>{fmt(rec['geometry_max_px'])}</td>"
            f"<td class='{brow_cls}'>{fmt(brow_stats['mean_px'])}</td>"
            f"<td>{fmt(rec['mean_px'])}</td>"
            "</tr>"
        )
        geometry_rows = []
        for group_name, _idx in LMK_GEOMETRY_GROUPS:
            stat = rec["geometry_group_stats"][group_name]
            cls = err_class(stat["mean_px"])
            geometry_rows.append(
                "<tr>"
                f"<td>{html.escape(group_name)}</td>"
                f"<td class='{cls}'>{fmt(stat['mean_px'])}</td>"
                f"<td>{fmt(stat['max_px'])}</td>"
                "</tr>"
            )
        appearance_rows = []
        for group_name, _idx in LMK_APPEARANCE_GROUPS:
            stat = rec["appearance_group_stats"][group_name]
            cls = err_class(stat["mean_px"])
            appearance_rows.append(
                "<tr>"
                f"<td>{html.escape(group_name)}</td>"
                f"<td class='{cls}'>{fmt(stat['mean_px'])}</td>"
                f"<td>{fmt(stat['max_px'])}</td>"
                "</tr>"
            )
        imgs = rec["images"]
        cards.append(
            f"""
            <section class="view-block">
              <h2>{html.escape(rec['view'])} 视角</h2>
              <h3>几何拟合质量（不含眉毛）</h3>
              <table class="group-table">
                <thead><tr><th>关键点类别</th><th>平均误差(px)</th><th>最大误差(px)</th></tr></thead>
                <tbody>{''.join(geometry_rows)}</tbody>
              </table>
              <h3>外观/纹理关注项</h3>
              <table class="group-table appearance-table">
                <thead><tr><th>区域</th><th>平均误差(px)</th><th>最大误差(px)</th></tr></thead>
                <tbody>{''.join(appearance_rows)}</tbody>
              </table>
              <div class="image-grid">
                <figure><img src="{html.escape(imgs['reprojection'])}"><figcaption>优化后关键点重投影</figcaption></figure>
                <figure><img src="{html.escape(imgs['mesh'])}"><figcaption>优化后 mesh 投影</figcaption></figure>
              </div>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>优化后关键点质量检查</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f7f9; color: #1f2933; }}
    header {{ padding: 28px 36px 18px; background: #17324d; color: white; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    header p {{ margin: 6px 0; color: #e7eef8; line-height: 1.6; }}
    main {{ padding: 24px 36px 48px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; margin-bottom: 22px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef2f7; font-weight: 700; }}
    .ok {{ color: #047857; font-weight: 700; }}
    .warn {{ color: #b7791f; font-weight: 700; }}
    .bad {{ color: #b42318; font-weight: 700; }}
    .view-block {{ margin: 0 0 34px; padding: 22px 0 0; border-top: 2px solid #d8dee8; }}
    h2 {{ margin: 0 0 8px; font-size: 22px; }}
    h3 {{ margin: 16px 0 8px; font-size: 16px; color: #263648; }}
    .group-table th, .group-table td {{ font-size: 13px; padding: 8px 10px; }}
    .appearance-table th {{ background: #f5efe6; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d7dde6; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px 12px; font-size: 14px; color: #4b5563; background: #fbfcfe; }}
    @media (max-width: 900px) {{ .image-grid {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} header {{ padding: 22px 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>优化后几何/外观质量检查</h1>
    <p>这个页面展示 L-BFGS 优化完成后的结果。几何拟合质量现在不再把眉毛算进主分数，避免把纹理/外观问题误判成脸型几何问题。</p>
    <p>眉毛单独作为“外观/纹理关注项”保留，用来后续检查纹理融合、眉毛颜色和局部贴图对齐。</p>
    <p>图中绿色点是真实 2D 关键点，红色点是模型投影点，黄色线表示误差距离。</p>
  </header>
  <main>
    <table>
      <thead><tr><th>视角</th><th>几何平均误差，不含眉毛(px)</th><th>几何最大误差(px)</th><th>眉毛外观平均误差(px)</th><th>全部点平均误差，仅供参考(px)</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(doc)
    logger.info(f"Optimized pose quality HTML saved: {out_dir / 'index.html'}")


def _save_init_per_view_debug(
    view_data: dict,
    init_result: dict,
    init_shape,
    flame,
    lmk_vertex_indices,
    lmk_tri_vidx,
    lmk_bary_coords,
    preprocessed_views: dict,
    intrinsics: dict,
    debug_dir: Path,
    device: str,
):
    """
    For each view, save:
      - init_view_{name}.json : initial params
      - init_reproj_{name}.png : landmark reprojection before optimization
      - init_mesh_{name}.png  : projected mesh before optimization
    """
    import json

    shape_np = init_shape if init_shape is not None else np.zeros(flame.n_shape, dtype=np.float32)
    model_device = flame.v_template.device
    model_dtype = flame.v_template.dtype

    for view_name, vd in view_data.items():
        pv      = init_result["per_view"].get(view_name, {})
        exp_np  = pv.get("exp")
        R_init  = vd["R_init"]
        t_init  = vd["t_init"]

        exp_use = np.zeros(flame.n_exp, dtype=np.float32)
        if exp_np is not None:
            exp_arr = np.asarray(exp_np, dtype=np.float32).reshape(-1)
            exp_use[:min(len(exp_arr), flame.n_exp)] = exp_arr[:flame.n_exp]

        with torch.no_grad():
            verts_init = flame(
                torch.as_tensor(shape_np, device=model_device, dtype=model_dtype),
                torch.as_tensor(exp_use, device=model_device, dtype=model_dtype),
            ).cpu().numpy()

        # Reprojection debug image
        mean_err, max_err = _save_landmark_reprojection_debug(
            vertices=verts_init,
            K=intrinsics[view_name],
            R=R_init,
            t=t_init,
            image=preprocessed_views[view_name]["image"],
            target_landmarks=vd["lmk_2d"],
            lmk_vertex_indices=lmk_vertex_indices,
            out_path=debug_dir / f"init_reproj_{view_name}.png",
            lmk_tri_vidx=lmk_tri_vidx,
            lmk_bary_coords=lmk_bary_coords,
        )

        # Mesh projection image
        face_mask = preprocessed_views[view_name].get("face_mask")
        if face_mask is not None:
            _save_projection_debug(
                verts_init, flame.faces.numpy(),
                intrinsics[view_name], R_init, t_init,
                preprocessed_views[view_name]["image"], face_mask,
                debug_dir / f"init_mesh_{view_name}.png",
            )

        # JSON summary
        view_json = {
            "view":          view_name,
            "backend":       init_result["backend"],
            "shape_source":  init_result["shape_source"],
            "exp_source":    "initializer" if exp_np is not None else "zero",
            "R_init":        R_init.tolist(),
            "t_init":        t_init.tolist(),
            "init_reproj_mean_px": round(mean_err, 3),
            "init_reproj_max_px":  round(max_err, 3),
        }
        out_json = debug_dir / f"init_view_{view_name}.json"
        with open(str(out_json), "w") as f:
            json.dump(view_json, f, indent=2)

        logger.info(
            f"  [{view_name}] 初始重投影误差: mean={mean_err:.2f}px, max={max_err:.2f}px  "
            f"(saved: {out_json.name}, init_reproj_{view_name}.png)"
        )


def _save_projection_debug(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image: np.ndarray,
    face_mask: np.ndarray,
    out_path: Path,
):
    """Save a full-resolution projection debug image constrained to the face mask."""
    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v_cam = (R @ vertices.T + t[:, None]).T
    z = v_cam[:, 2]
    valid = z > 1e-4
    v_hom = (K @ v_cam[valid].T).T
    u = (v_hom[:, 0] / v_cam[valid, 2]).astype(int)
    v = (v_hom[:, 1] / v_cam[valid, 2]).astype(int)

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    for px, py in zip(u, v):
        if 0 <= px < W and 0 <= py < H and face_mask[py, px] > 0:
            cv2.circle(img, (px, py), 1, (0, 0, 255), -1)

    cv2.imwrite(str(out_path), img)
    logger.info(f"Projection debug image saved: {out_path}")


def _build_mask_row_bounds(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    xmin = np.zeros(h, dtype=np.float32)
    xmax = np.zeros(h, dtype=np.float32)
    valid = np.zeros(h, dtype=bool)
    for y in range(h):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size == 0:
            continue
        xmin[y] = float(xs[0])
        xmax[y] = float(xs[-1])
        valid[y] = True
    return xmin, xmax, valid


def _landmark_group_stats(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    stats = {}
    for name, idx in LMK_ERROR_GROUPS:
        stats[name] = _landmark_subset_stats(errors, idx)
    return stats


def _landmark_subset_stats(errors: np.ndarray, idx: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    vals = errors[idx]
    return {
        "count": int(len(vals)),
        "mean_px": round(float(vals.mean()), 3),
        "max_px": round(float(vals.max()), 3),
    }


def _landmark_named_stats(errors: np.ndarray, groups: tuple) -> dict:
    return {name: _landmark_subset_stats(errors, idx) for name, idx in groups}


def _stable_landmark_score(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=np.float64)
    weights = np.zeros(68, dtype=np.float64)
    weights[LMK_NOSE_IDX] = 1.4
    weights[LMK_EYE_IDX] = 1.0
    weights[LMK_MOUTH_IDX] = 1.0
    return float((errors * weights).sum() / np.clip(weights.sum(), 1e-6, None))


def _landmark_reprojection_details(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    target_landmarks: np.ndarray,
    lmk_vertex_indices: np.ndarray,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    v_cam = (R @ vertices.T + t[:, None]).T
    z = np.clip(v_cam[:, 2], 1e-6, None)
    v_hom = (K @ v_cam.T).T
    proj = np.stack([v_hom[:, 0] / z, v_hom[:, 1] / z], axis=1)

    if lmk_tri_vidx is not None and lmk_bary_coords is not None:
        lmk_proj = (
            proj[lmk_tri_vidx[:, 0]] * lmk_bary_coords[:, 0:1] +
            proj[lmk_tri_vidx[:, 1]] * lmk_bary_coords[:, 1:2] +
            proj[lmk_tri_vidx[:, 2]] * lmk_bary_coords[:, 2:3]
        )
    else:
        lmk_proj = proj[lmk_vertex_indices]

    errors = np.linalg.norm(lmk_proj - target_landmarks, axis=1)
    return lmk_proj, errors


def _save_landmark_reprojection_debug(
    vertices: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image: np.ndarray,
    target_landmarks: np.ndarray,
    lmk_vertex_indices: np.ndarray,
    out_path: Path,
    lmk_tri_vidx: Optional[np.ndarray] = None,
    lmk_bary_coords: Optional[np.ndarray] = None,
    return_errors: bool = False,
):
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lmk_proj, errors = _landmark_reprojection_details(
        vertices=vertices,
        K=K,
        R=R,
        t=t,
        target_landmarks=target_landmarks,
        lmk_vertex_indices=lmk_vertex_indices,
        lmk_tri_vidx=lmk_tri_vidx,
        lmk_bary_coords=lmk_bary_coords,
    )
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for idx, (gt, pred, err) in enumerate(zip(target_landmarks, lmk_proj, errors)):
        gt_pt = tuple(np.round(gt).astype(int))
        pred_pt = tuple(np.round(pred).astype(int))
        cv2.circle(img, gt_pt, 2, (0, 255, 0), -1)
        cv2.circle(img, pred_pt, 2, (0, 0, 255), -1)
        cv2.line(img, gt_pt, pred_pt, (0, 255, 255), 1)
        if err > 8.0:
            cv2.putText(
                img,
                str(idx),
                (pred_pt[0] + 2, pred_pt[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(out_path), img)
    top_idx = np.argsort(errors)[-10:][::-1]
    report_path = out_path.with_suffix(".txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"mean_error_px={float(errors.mean()):.4f}\n")
        f.write(f"max_error_px={float(errors.max()):.4f}\n")
        f.write("top10_indices:\n")
        for idx in top_idx:
            gt = target_landmarks[idx]
            pred = lmk_proj[idx]
            f.write(
                f"{int(idx)} err={float(errors[idx]):.4f} "
                f"gt=({float(gt[0]):.2f},{float(gt[1]):.2f}) "
                f"pred=({float(pred[0]):.2f},{float(pred[1]):.2f})\n"
            )
    mean_err = float(errors.mean())
    max_err = float(errors.max())
    if return_errors:
        return mean_err, max_err, errors
    return mean_err, max_err


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe 468 → 68 标准人脸点的近似映射（基于位置对应）
_MP468_TO_68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288,
    323, 454, 389, 71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87,
]

def _mediapipe_to_68(lmks_468: np.ndarray) -> np.ndarray:
    """MediaPipe 468点 → 68标准点的近似映射"""
    idx = _MP468_TO_68[:68]
    return lmks_468[idx]   # (68, 2)


def _get_flame_uv(
    flame_model_path: Path,
    faces: np.ndarray,       # (F, 3) 几何面片
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 FLAME UV 拓扑（分离顶点格式）。
    Returns:
        uv_verts: (T, 2) UV 坐标
        uv_faces: (F, 3) UV 面片索引（与几何面片一一对应）
    """
    tex_path = flame_model_path.parent / "texture_data_256.npy"
    if tex_path.exists():
        try:
            d = np.load(str(tex_path), allow_pickle=True, encoding="bytes").item()
            vt = np.array(d[b"vt"], dtype=np.float32)    # (T, 2)
            ft = np.array(d[b"ft"], dtype=np.int64)       # (F, 3)
            if len(ft) == len(faces):
                logger.info(f"FLAME UV 加载成功: {len(vt)} UV点, {len(ft)} UV面片")
                return vt, ft
        except Exception as e:
            logger.warning(f"FLAME UV 加载失败: {e}")

    # Fallback：使用几何顶点索引作为 UV 索引（精度低但结构正确）
    logger.warning("使用 fallback UV（几何顶点直接映射）")
    n_verts = int(faces.max()) + 1
    theta  = np.linspace(0, np.pi, n_verts)
    phi    = np.linspace(0, 2 * np.pi, n_verts)
    uv_verts = np.stack([phi / (2 * np.pi), theta / np.pi], axis=1).astype(np.float32)
    return uv_verts, faces.copy()


def _save_depth_debug(d_pred, d_3dmm, d_aligned, disp_map, debug_dir: Path):
    """保存深度调试图"""
    def to_vis(arr):
        a = arr.copy()
        a[~np.isfinite(a)] = 0
        mn, mx = a.min(), a.max()
        if mx > mn:
            a = ((a - mn) / (mx - mn) * 255).astype(np.uint8)
        return a

    cv2.imwrite(str(debug_dir / "depth_pred.png"),    to_vis(d_pred))
    cv2.imwrite(str(debug_dir / "depth_3dmm.png"),    to_vis(d_3dmm))
    cv2.imwrite(str(debug_dir / "depth_aligned.png"), to_vis(d_aligned))
    cv2.imwrite(str(debug_dir / "depth_disp.png"),    to_vis(disp_map))

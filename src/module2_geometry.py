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
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from src.coordinates import image_uv_to_obj_uv

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
        lambda_exp: float = 1e-3,
        lambda_contour: float = 0.0,
        front_contour_weight: float = 2.4,
        front_jaw_weight: float = 3.2,
        side_contour_weight: float = 1.2,
        side_jaw_weight: float = 1.8,
        side_brow_weight: float = 0.25,
        side_extra_soft_weight: float = 0.6,
        max_iter: int = 100,
        lr: float = 0.5,
        device: str = "cuda",
        lmk_face_idx: Optional[np.ndarray] = None,    # (68,) — 精确重心坐标用
        lmk_bary_coords: Optional[np.ndarray] = None, # (68, 3)
    ):
        self.flame   = flame.to(device)
        self.lambda_shape = lambda_shape
        self.lambda_exp   = lambda_exp
        self.lambda_contour = lambda_contour
        self.front_contour_weight = front_contour_weight
        self.front_jaw_weight = front_jaw_weight
        self.side_contour_weight = side_contour_weight
        self.side_jaw_weight = side_jaw_weight
        self.side_brow_weight = side_brow_weight
        self.side_extra_soft_weight = side_extra_soft_weight
        self.max_iter = max_iter
        self.lr       = lr
        self.device   = device

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
        shape_param.requires_grad_(True)

        view_names = list(views.keys())
        exp_params, rvec_params, t_params = {}, {}, {}

        for name in view_names:
            v = views[name]

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
            lmk_weights[name] = w

        all_params = (
            [shape_param]
            + list(exp_params.values())
            + list(rvec_params.values())
            + list(t_params.values())
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
                R_cur = rodrigues_to_matrix(rvec_params[name])      # (3, 3)
                proj  = project_vertices(verts, Ks[name], R_cur, t_params[name])  # (N, 2)

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

            total_loss = total_loss + self.lambda_shape * (shape_param ** 2).mean()
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
            R_final = rodrigues_to_matrix(rvec_params[name]).detach().cpu().numpy()
            t_final = t_params[name].detach().cpu().numpy()
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


def _shape_only_fine_tune(
    flame: FLAMEModel,
    shape_init: np.ndarray,
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

    contour_idx_t = torch.tensor(LMK_CONTOUR_IDX, device=device)
    jaw_idx_t = torch.tensor(np.arange(4, 13, dtype=np.int64), device=device)
    stable_idx_np = np.concatenate([LMK_NOSE_IDX, LMK_EYE_IDX, LMK_MOUTH_IDX])
    stable_idx_t = torch.tensor(stable_idx_np, device=device)

    target_weights = {}
    anchor_weights = {}
    for name in view_names:
        tw = torch.full((68,), 0.25, device=device, dtype=torch.float32)
        if name == "front":
            tw[contour_idx_t] = float(front_contour_weight * contour_scale)
            tw[jaw_idx_t] = float(front_jaw_weight * contour_scale)
        else:
            tw[contour_idx_t] = float(side_contour_weight * contour_scale)
            tw[jaw_idx_t] = float(side_jaw_weight * contour_scale)
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
    if enable_dense_contour and dense_contour_weight > 0:
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

    shape_after = shape_param.detach().cpu().numpy().astype(np.float32)

    def evaluate_shape(shape_np: np.ndarray, image_prefix: str) -> dict:
        records = []
        for name in view_names:
            res = per_view_results[name]
            with torch.no_grad():
                verts_np = flame(
                    torch.tensor(shape_np, device=model_device, dtype=model_dtype),
                    torch.tensor(res["exp"], device=model_device, dtype=model_dtype),
                ).detach().cpu().numpy()
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
            dense_metric = None
            dense = dense_contours.get(name)
            if dense is not None:
                v_cam = (res["R"] @ verts_np.T + res["t"][:, None]).T
                z = np.clip(v_cam[:, 2], 1e-6, None)
                v_hom = (intrinsics[name] @ v_cam.T).T
                proj_np = np.stack([v_hom[:, 0] / z, v_hom[:, 1] / z], axis=1)
                dense_metric, left_pred, right_pred = _dense_contour_metric_np(proj_np, dense)
                _save_dense_contour_debug_image(
                    image=preprocessed_views[name]["image"],
                    dense=dense,
                    left_pred=left_pred,
                    right_pred=right_pred,
                    out_path=out_dir / f"{name}_{image_prefix}_dense_contour.png",
                )
            records.append({
                "view": name,
                "mean_px": round(float(mean_err), 3),
                "max_px": round(float(max_err), 3),
                "contour_mean_px": _landmark_subset_stats(errors, LMK_CONTOUR_IDX)["mean_px"],
                "jaw_mean_px": _landmark_subset_stats(errors, np.arange(4, 13, dtype=np.int64))["mean_px"],
                "stable_mean_px": _landmark_subset_stats(errors, stable_idx_np)["mean_px"],
                "dense_contour_mean_px": round(float(dense_metric), 3) if dense_metric is not None else None,
            })
        if not records:
            return {
                "records": [],
                "mean_px": float("inf"),
                "contour_mean_px": float("inf"),
                "jaw_mean_px": float("inf"),
                "stable_mean_px": float("inf"),
                "dense_contour_mean_px": None,
            }
        dense_vals = [
            float(r["dense_contour_mean_px"])
            for r in records
            if r.get("dense_contour_mean_px") is not None
        ]
        return {
            "records": records,
            "mean_px": round(float(np.mean([r["mean_px"] for r in records])), 3),
            "contour_mean_px": round(float(np.mean([r["contour_mean_px"] for r in records])), 3),
            "jaw_mean_px": round(float(np.mean([r["jaw_mean_px"] for r in records])), 3),
            "stable_mean_px": round(float(np.mean([r["stable_mean_px"] for r in records])), 3),
            "dense_contour_mean_px": round(float(np.mean(dense_vals)), 3) if dense_vals else None,
        }

    before = evaluate_shape(np.asarray(shape_init, dtype=np.float32), "before")
    after = evaluate_shape(shape_after, "after")
    report["before"] = before
    report["after"] = after

    contour_improve = float(before["contour_mean_px"] - after["contour_mean_px"])
    stable_worsen = float(after["stable_mean_px"] - before["stable_mean_px"])
    total_worsen = float(after["mean_px"] - before["mean_px"])
    before_dense = before.get("dense_contour_mean_px")
    after_dense = after.get("dense_contour_mean_px")
    dense_improve = 0.0
    if before_dense is not None and after_dense is not None:
        dense_improve = float(before_dense - after_dense)
    contour_worsen = float(after["contour_mean_px"] - before["contour_mean_px"])
    contour_or_dense_improved = (
        contour_improve >= float(min_contour_improve_px)
        or dense_improve >= float(min_dense_contour_improve_px)
    )
    accepted = (
        contour_or_dense_improved
        and contour_worsen <= float(max_contour_worsen_px)
        and stable_worsen <= float(max_stable_worsen_px)
        and total_worsen <= float(max_total_worsen_px)
    )
    report.update({
        "accepted": bool(accepted),
        "contour_improve_px": round(contour_improve, 3),
        "dense_contour_improve_px": round(dense_improve, 3),
        "contour_worsen_px": round(contour_worsen, 3),
        "stable_worsen_px": round(stable_worsen, 3),
        "total_worsen_px": round(total_worsen, 3),
        "shape_delta_norm": round(float(np.linalg.norm(shape_after - shape_init)), 6),
        "reason": "dense/landmark contour improved with frozen pose" if accepted else "rejected by acceptance gate",
    })

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Shape-only fine tune: "
        f"accepted={report['accepted']}, "
        f"contour {before['contour_mean_px']:.2f}->{after['contour_mean_px']:.2f}px, "
        f"dense {before_dense}->{after_dense}px, "
        f"stable {before['stable_mean_px']:.2f}->{after['stable_mean_px']:.2f}px, "
        f"total {before['mean_px']:.2f}->{after['mean_px']:.2f}px"
    )
    if accepted:
        return shape_after, report
    return np.asarray(shape_init, dtype=np.float32), report


# ══════════════════════════════════════════════════════════════════════════════
# Depth-Anything-V2 深度置换
# ══════════════════════════════════════════════════════════════════════════════

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
        logger.warning("init_shape 含 NaN/Inf，重置为零向量（中性脸初始化）")
        init_shape = np.zeros_like(init_shape)
        init_result["shape"] = init_shape
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
        }
        init_exps[view_name] = pv.get("exp")

    if not view_data:
        raise RuntimeError("所有视角关键点检测失败，无法进行 3DMM 重建")

    # ── 保存每视角初始化参数 + 预优化重投影图 ────────────────────────────────
    lmk_tri_vidx = flame_faces_np[lmk_data["face_idx"]] if lmk_data is not None else None
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

    def _make_optimizer():
        return JointFLAMEOptimizer(
            flame=flame,
            lmk_vertex_indices=lmk_vertex_indices,
            lambda_shape=lambda_shape,
            lambda_exp=lambda_exp,
            lambda_contour=_cfg_float("LAMBDA_CONTOUR", 0.0),
            front_contour_weight=_cfg_float("FRONT_CONTOUR_WEIGHT", 2.4),
            front_jaw_weight=_cfg_float("FRONT_JAW_WEIGHT", 3.2),
            side_contour_weight=_cfg_float("SIDE_CONTOUR_WEIGHT", 1.2),
            side_jaw_weight=_cfg_float("SIDE_JAW_WEIGHT", 1.8),
            side_brow_weight=_cfg_float("SIDE_BROW_WEIGHT", 0.25),
            side_extra_soft_weight=_cfg_float("SIDE_EXTRA_SOFT_WEIGHT", 0.6),
            max_iter=lbfgs_max_iter,
            lr=lbfgs_lr,
            device=device,
            lmk_face_idx=lmk_data["face_idx"] if lmk_data is not None else None,
            lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
        )

    def _run_optimizer_once(pass_idx: int):
        optimizer = _make_optimizer()
        shape_run, per_view_run = optimizer.optimize(view_data, init_shape, init_exps)
        reproj_stats = {}
        for view_name, view_result in per_view_run.items():
            with torch.no_grad():
                verts_view = flame(
                    torch.tensor(shape_run, device=device),
                    torch.tensor(view_result["exp"], device=device),
                ).cpu().numpy()
            mean_err, max_err = _save_landmark_reprojection_debug(
                vertices=verts_view,
                K=intrinsics[view_name],
                R=view_result["R"],
                t=view_result["t"],
                image=preprocessed_views[view_name]["image"],
                target_landmarks=view_data[view_name]["lmk_2d"],
                lmk_vertex_indices=lmk_vertex_indices,
                out_path=debug_dir / f"landmark_reproj_{view_name}.png",
                lmk_tri_vidx=lmk_tri_vidx,
                lmk_bary_coords=lmk_data["bary_coords"] if lmk_data is not None else None,
            )
            reproj_stats[view_name] = (mean_err, max_err)
            logger.info(
                f"  [pass {pass_idx}] [{view_name}] landmark reprojection error: "
                f"mean={mean_err:.2f}px, max={max_err:.2f}px"
            )
        return shape_run, per_view_run, reproj_stats

    shape_opt, per_view_results, reproj_stats = _run_optimizer_once(pass_idx=1)
    mean_err_avg = float(np.mean([v[0] for v in reproj_stats.values()])) if reproj_stats else float("inf")
    if mean_err_avg > 20.0:
        logger.warning(
            f"优化结果异常（avg reproj={mean_err_avg:.2f}px），自动重试一次以规避偶发坏解"
        )
        shape_retry, per_view_retry, reproj_retry = _run_optimizer_once(pass_idx=2)
        mean_err_retry = float(np.mean([v[0] for v in reproj_retry.values()])) if reproj_retry else float("inf")
        if mean_err_retry < mean_err_avg:
            logger.info(
                f"重试结果更优：avg reproj {mean_err_avg:.2f}px -> {mean_err_retry:.2f}px，采用重试结果"
            )
            shape_opt, per_view_results, reproj_stats = shape_retry, per_view_retry, reproj_retry
        else:
            logger.warning(
                f"重试未改善：avg reproj {mean_err_avg:.2f}px -> {mean_err_retry:.2f}px，保留首次结果"
            )

    shape_only_report = {"enabled": False, "accepted": False}
    if _cfg_bool("ENABLE_SHAPE_ONLY_FINE_TUNE", True):
        shape_opt, shape_only_report = _shape_only_fine_tune(
            flame=flame,
            shape_init=shape_opt,
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
        )

    with open(debug_dir / "optimized_shape.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "shape_norm": float(np.linalg.norm(shape_opt)) if shape_opt is not None else 0.0,
                "shape_params": shape_opt.tolist() if shape_opt is not None else [],
                "shape_only_fine_tune": shape_only_report,
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

    with torch.no_grad():
        verts_final = flame(
            torch.tensor(shape_opt, device=device),
            torch.tensor(exp_final, device=device),
        ).cpu().numpy()  # (N, 3)

    # ── 步骤2：Loop Subdivision（增加几何密度，在置换前细分提高精度）────────
    import trimesh as _trimesh
    logger.info("Loop Subdivision：细分 FLAME 基础网格...")
    # 提前加载原始 UV，与几何拓扑同步细分（保证面片一一对应）
    uv_verts_orig, uv_faces_orig = _get_flame_uv(flame_model_path, flame_faces_np)
    SUBDIV_ITERS = 2   # 9976 → ~40K → ~160K 面片
    verts_sub, faces_sub = _trimesh.remesh.subdivide_loop(
        verts_final, flame_faces_np, iterations=SUBDIV_ITERS
    )
    # UV 用线性细分（与 Loop 细分面片拓扑一致，保证几何-UV 面片一一对应）
    uv_verts_sub, uv_faces_sub = uv_verts_orig, uv_faces_orig
    for _ in range(SUBDIV_ITERS):
        uv_verts_sub, uv_faces_sub = _trimesh.remesh.subdivide(uv_verts_sub, uv_faces_sub)
    logger.info(f"细分完成: {len(faces_sub)} 面片, {len(verts_sub)} 顶点")

    vertex_normals = compute_vertex_normals(verts_sub, faces_sub)

    # ── 深度置换：多视角融合（Fix 1 逐顶点采样 + Fix 2 多视角加权融合）──────
    logger.info("估计多视角深度并融合置换...")
    import cv2 as _cv2
    from scipy.ndimage import distance_transform_edt
    try:
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

    except Exception as e:
        import traceback
        logger.warning(f"深度置换失败（{e}），跳过置换步骤，使用细分 3DMM Mesh")
        logger.debug(traceback.format_exc())
        verts_displaced = verts_sub

    # ── 步骤4：Laplacian 平滑（消除深度置换尖刺）──────────────────────────
    logger.info("Laplacian 平滑置换后 Mesh...")
    try:
        _sm = _trimesh.Trimesh(vertices=verts_displaced, faces=faces_sub, process=False)
        _trimesh.smoothing.filter_laplacian(_sm, iterations=3, lamb=0.25)  # 减少迭代防止鼻尖过度平滑
        verts_displaced = np.array(_sm.vertices)
    except Exception as _e:
        logger.warning(f"Laplacian 平滑失败（{_e}），跳过")

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
    depth_output_path = output_dir / "face_mesh_with_depth.obj"
    base_glb_path = output_dir / "face_mesh.glb"
    depth_glb_path = output_dir / "face_mesh_with_depth.glb"
    export_mesh_obj(verts_sub, faces_sub, uv_verts_sub, uv_faces_sub, base_output_path)
    export_mesh_obj(verts_displaced, faces_sub, uv_verts_sub, uv_faces_sub, depth_output_path)
    export_mesh_glb(verts_sub, faces_sub, uv_verts_sub, uv_faces_sub, base_glb_path)
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

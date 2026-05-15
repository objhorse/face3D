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
LMK_ERROR_GROUPS = (
    ("轮廓", LMK_CONTOUR_IDX),
    ("眉毛", LMK_BROW_IDX),
    ("鼻子", LMK_NOSE_IDX),
    ("眼睛", LMK_EYE_IDX),
    ("嘴巴", LMK_MOUTH_IDX),
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
            mask = views[name].get("face_mask")
            if name == "front" or mask is None:
                contour_rows[name] = None
                continue
            xmin, xmax, valid = _build_mask_row_bounds(mask)
            contour_rows[name] = {
                "xmin": torch.tensor(xmin, device=dev, dtype=torch.float32),
                "xmax": torch.tensor(xmax, device=dev, dtype=torch.float32),
                "valid": torch.tensor(valid, device=dev, dtype=torch.bool),
            }
        lmk_weights = {}
        side_risk_idx = np.array(list(range(27)), dtype=np.int64)
        side_extra_soft_idx = np.array([36, 37, 38, 39, 42, 43, 44, 45], dtype=np.int64)
        for name in view_names:
            w = torch.ones(68, device=dev, dtype=torch.float32)
            if name != "front":
                w[torch.tensor(side_risk_idx, device=dev)] = 0.35
                w[torch.tensor(side_extra_soft_idx, device=dev)] = 0.6
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

                contour_data = contour_rows.get(name)
                if contour_data is not None and self.lambda_contour > 0:
                    jaw_proj = lmk_proj[4:13]
                    row_idx = torch.round(jaw_proj[:, 1]).long()
                    row_idx = row_idx.clamp(0, contour_data["valid"].shape[0] - 1)
                    valid_rows = contour_data["valid"][row_idx]
                    if torch.any(valid_rows):
                        jaw_x = jaw_proj[:, 0][valid_rows]
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
    def _make_optimizer():
        return JointFLAMEOptimizer(
            flame=flame,
            lmk_vertex_indices=lmk_vertex_indices,
            lambda_shape=lambda_shape,
            lambda_exp=lambda_exp,
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
            "group_stats": _landmark_group_stats(errors),
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

    rows = []
    cards = []
    for rec in records:
        rows.append(
            "<tr>"
            f"<td>{html.escape(rec['view'])}</td>"
            f"<td>{fmt(rec['mean_px'])}</td>"
            f"<td>{fmt(rec['max_px'])}</td>"
            "</tr>"
        )
        group_rows = []
        for group_name, _idx in LMK_ERROR_GROUPS:
            stat = rec["group_stats"][group_name]
            cls = "bad" if stat["mean_px"] > 25.0 else ("warn" if stat["mean_px"] > 12.0 else "ok")
            group_rows.append(
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
              <table class="group-table">
                <thead><tr><th>关键点类别</th><th>平均误差(px)</th><th>最大误差(px)</th></tr></thead>
                <tbody>{''.join(group_rows)}</tbody>
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
    .group-table th, .group-table td {{ font-size: 13px; padding: 8px 10px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
    figure {{ margin: 0; background: white; border: 1px solid #d7dde6; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 9px 12px; font-size: 14px; color: #4b5563; background: #fbfcfe; }}
    @media (max-width: 900px) {{ .image-grid {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} header {{ padding: 22px 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>优化后关键点质量检查</h1>
    <p>这个页面展示 L-BFGS 优化完成后的结果，用来判断初始阶段的大误差是否已经被修回来。</p>
    <p>图中绿色点是真实 2D 关键点，红色点是模型投影点，黄色线表示误差距离。</p>
  </header>
  <main>
    <table>
      <thead><tr><th>视角</th><th>平均误差(px)</th><th>最大误差(px)</th></tr></thead>
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
        vals = errors[idx]
        stats[name] = {
            "count": int(len(vals)),
            "mean_px": round(float(vals.mean()), 3),
            "max_px": round(float(vals.max()), 3),
        }
    return stats


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

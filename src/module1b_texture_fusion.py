"""
模块1b：三视角纹理预融合

在三维建模之前，将三张视角图像融合成一张颜色统一的正面纹理。
消除多视角光照差异，避免后续纹理烘焙中的颜色拼接问题。

流程:
  1. Reinhard 颜色迁移：将 left/right 视角的颜色对齐到 front 基准色调
  2. TPS 薄板样条变形：将 left/right 视图关键点对齐到 front 坐标系
  3. 软边界加权混合：基于各视图 mask 的距离变换权重
  4. face mask 裁剪：仅保留面部区域，背景填黑
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh 鼻尖关键点索引
NOSE_TIP_IDX = 1


# ══════════════════════════════════════════════════════════════════════════════
# 颜色迁移（Reinhard）
# ══════════════════════════════════════════════════════════════════════════════

def _color_transfer(
    src: np.ndarray,       # (H, W, 3) uint8 RGB
    tgt: np.ndarray,       # (H, W, 3) uint8 RGB — 颜色基准
    src_mask: np.ndarray,  # (H, W) uint8
    tgt_mask: np.ndarray,  # (H, W) uint8
) -> np.ndarray:
    """
    Reinhard 颜色迁移：将 src 在 mask 区域内的颜色统计（均值+标准差）对齐到 tgt。
    不改变 src 的纹理细节，只调整整体色调/亮度/饱和度。
    """
    result = src.astype(np.float64)
    for c in range(3):
        src_px = src[:, :, c][src_mask > 0].astype(np.float64)
        tgt_px = tgt[:, :, c][tgt_mask > 0].astype(np.float64)
        if len(src_px) < 10 or len(tgt_px) < 10:
            continue
        mean_s, std_s = src_px.mean(), src_px.std() + 1e-6
        mean_t, std_t = tgt_px.mean(), tgt_px.std() + 1e-6
        result[:, :, c] = (result[:, :, c] - mean_s) / std_s * std_t + mean_t
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# TPS 薄板样条变形
# ══════════════════════════════════════════════════════════════════════════════

def _compute_tps_maps(
    src_lmks: np.ndarray,  # (N, 2) float — 源图关键点坐标 (x, y)
    dst_lmks: np.ndarray,  # (N, 2) float — 目标图关键点坐标 (x, y)
    H: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 TPS 逆映射：对目标图每个像素 (x, y)，找到对应的源图坐标。
    基于 RBFInterpolator(thin_plate_spline)，一次建模两次应用（图像 + Mask）。

    Returns:
        map_x, map_y: (H, W) float32 — cv2.remap 所需的逆映射坐标
    """
    from scipy.interpolate import RBFInterpolator  # 确认 scipy 可用

    # 逆映射：已知 dst_lmks → src_lmks，对目标像素网格预测源坐标
    interp = RBFInterpolator(
        dst_lmks.astype(np.float64),
        src_lmks.astype(np.float64),
        kernel="thin_plate_spline",
        smoothing=0.0,   # 精确插值（控制点处误差=0）
    )

    # 生成目标像素全局网格 (H*W, 2)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    query = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)

    src_coords = interp(query)  # (H*W, 2) → [x_src, y_src]
    map_x = src_coords[:, 0].reshape(H, W).astype(np.float32)
    map_y = src_coords[:, 1].reshape(H, W).astype(np.float32)
    return map_x, map_y


def _apply_tps(
    src_img: np.ndarray,   # (H, W, 3) uint8
    src_mask: np.ndarray,  # (H, W) uint8
    map_x: np.ndarray,     # (H, W) float32
    map_y: np.ndarray,     # (H, W) float32
) -> Tuple[np.ndarray, np.ndarray]:
    """用预计算的 TPS 映射图同时变形图像和 Mask。"""
    warped_img  = cv2.remap(src_img,  map_x, map_y,
                            cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.remap(src_mask, map_x, map_y,
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_img, warped_mask


# ══════════════════════════════════════════════════════════════════════════════
# 软边界权重（距离变换）
# ══════════════════════════════════════════════════════════════════════════════

def _feather_mask(mask: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """
    距离变换软边界：mask 内部离边界越远权重越高。
    返回 [0,1] float32。
    """
    binary = (mask > 0).astype(np.uint8)
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return np.clip(dist / (sigma + 1e-6), 0.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 主融合函数
# ══════════════════════════════════════════════════════════════════════════════

def fuse_face_views(
    preprocessed_views: Dict[str, dict],
    debug_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    将三视角图像融合成一张颜色统一的正面纹理。

    Args:
        preprocessed_views: module1 preprocess_all_views() 的输出字典
            每个 key 对应一个视角名，value 包含:
              "image":      (H, W, 3) uint8 RGB
              "landmarks":  (468, 2) float32 | None
              "face_mask":  (H, W) uint8
        debug_save_path: 如果给出，保存调试图到此目录

    Returns:
        (H, W, 3) uint8 RGB — 颜色统一的正面纹理（背景为黑）
    """
    # ── 基准视角：front ────────────────────────────────────────────────────────
    if "front" not in preprocessed_views:
        fallback = list(preprocessed_views.values())[0]
        logger.warning("无 front 视角，直接返回第一个视角图像")
        img = fallback["image"].copy()
        img[fallback["face_mask"] == 0] = 0
        return img

    front_data  = preprocessed_views["front"]
    front_img   = front_data["image"]    # (H, W, 3)
    front_lmks  = front_data["landmarks"]
    front_mask  = front_data["face_mask"]
    H, W        = front_img.shape[:2]

    if front_lmks is None:
        logger.warning("front 视角无关键点，直接返回正面图")
        result = front_img.copy()
        result[front_mask == 0] = 0
        return result

    # ── 各视角：颜色归一化 + TPS 对齐 ────────────────────────────────────────
    warped_imgs:  Dict[str, np.ndarray] = {"front": front_img.copy()}
    warped_masks: Dict[str, np.ndarray] = {"front": front_mask.copy()}

    for view_name in ["left", "right"]:
        if view_name not in preprocessed_views:
            continue

        vdata  = preprocessed_views[view_name]
        v_img   = vdata["image"]
        v_lmks  = vdata["landmarks"]
        v_mask  = vdata["face_mask"]

        # Step 1: 颜色归一化（在变形前做，避免 warp 插值扰动颜色统计）
        logger.info(f"[{view_name}] Reinhard 颜色迁移...")
        norm_img = _color_transfer(v_img, front_img, v_mask, front_mask)

        if v_lmks is None:
            logger.warning(f"[{view_name}] 无关键点，跳过 TPS 变形")
            warped_imgs[view_name]  = norm_img
            warped_masks[view_name] = v_mask.copy()
            continue

        # Step 2: TPS 变形
        logger.info(f"[{view_name}] 计算 TPS 逆映射（{len(front_lmks)} 个控制点）...")
        map_x, map_y = _compute_tps_maps(v_lmks, front_lmks, H, W)

        warped_img, warped_mask = _apply_tps(norm_img, v_mask, map_x, map_y)

        # 将变形后的 mask 约束在 front 面部区域内（去除 TPS 外推噪声）
        warped_mask = cv2.bitwise_and(warped_mask, front_mask)

        warped_imgs[view_name]  = warped_img
        warped_masks[view_name] = warped_mask
        n_valid = int((warped_mask > 0).sum())
        logger.info(f"[{view_name}] TPS 完成，有效像素={n_valid}")

    # ── 计算各视角软边界权重 ──────────────────────────────────────────────────
    # 使用距离变换（越远离 mask 边界权重越高），front 额外加 0.5 底座以保持主导
    w_front = _feather_mask(warped_masks["front"], sigma=40.0) + 0.5
    w_left  = _feather_mask(warped_masks.get("left",  np.zeros((H, W), np.uint8)), sigma=25.0)
    w_right = _feather_mask(warped_masks.get("right", np.zeros((H, W), np.uint8)), sigma=25.0)

    # 只在 front mask 内混合（背景保持 0）
    front_bin = (front_mask > 0).astype(np.float32)
    w_front  *= front_bin
    w_left   *= front_bin
    w_right  *= front_bin

    w_sum = w_front + w_left + w_right + 1e-8
    w_front /= w_sum
    w_left  /= w_sum
    w_right /= w_sum

    # ── 加权混合 ──────────────────────────────────────────────────────────────
    result = np.zeros((H, W, 3), dtype=np.float64)
    result += warped_imgs["front"].astype(np.float64) * w_front[:, :, None]
    if "left" in warped_imgs:
        result += warped_imgs["left"].astype(np.float64)  * w_left[:, :, None]
    if "right" in warped_imgs:
        result += warped_imgs["right"].astype(np.float64) * w_right[:, :, None]

    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)

    # 背景填黑（膨胀 front_mask 少许以包含边缘皮肤）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated_mask = cv2.dilate(front_mask, kernel)
    result_uint8[dilated_mask == 0] = 0

    # ── 调试输出 ──────────────────────────────────────────────────────────────
    if debug_save_path is not None:
        import os
        os.makedirs(debug_save_path, exist_ok=True)

        cv2.imwrite(
            os.path.join(debug_save_path, "unified_texture.png"),
            cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR),
        )
        for vname, w in [("front", w_front), ("left", w_left), ("right", w_right)]:
            cv2.imwrite(
                os.path.join(debug_save_path, f"fusion_weight_{vname}.png"),
                (w * 255).clip(0, 255).astype(np.uint8),
            )
        for vname, img in warped_imgs.items():
            cv2.imwrite(
                os.path.join(debug_save_path, f"warped_{vname}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        logger.info(f"融合调试图已保存到: {debug_save_path}")

    nz = int((result_uint8[:, :, 0] > 0).sum())
    logger.info(f"纹理融合完成: 输出 {result_uint8.shape}, 非零像素={nz}")
    return result_uint8

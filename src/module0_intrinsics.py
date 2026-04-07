"""
模块0：相机内参预测

优先级：
  1. 手动提供（config.MANUAL_INTRINSICS）
  2. Dust3R 预测
  3. 经验估计 fallback（基于图像尺寸 + 等效焦距假设）

输出: Dict[view_name, np.ndarray(3,3)]  — 每视角的 K 矩阵
"""
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def load_intrinsics_from_config(cfg_intrinsics: Optional[dict]) -> Optional[np.ndarray]:
    """从 config.MANUAL_INTRINSICS 构建 K 矩阵，若为 None 则返回 None"""
    if cfg_intrinsics is None:
        return None
    if isinstance(cfg_intrinsics, np.ndarray):
        assert cfg_intrinsics.shape == (3, 3)
        return cfg_intrinsics.astype(np.float64)
    fx = cfg_intrinsics["fx"]
    fy = cfg_intrinsics["fy"]
    cx = cfg_intrinsics["cx"]
    cy = cfg_intrinsics["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def estimate_intrinsics_from_image(image: np.ndarray, fov_deg: float = 50.0) -> np.ndarray:
    """
    经验估计 fallback：
    假设水平视角 fov_deg（常见医美设备约 40-60°），由此推算焦距。
    cx/cy 取图像中心。
    """
    h, w = image.shape[:2]
    fov_rad = np.deg2rad(fov_deg)
    fx = (w / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx  # 假设等比例像素
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    logger.warning(f"使用经验内参 fallback (FOV={fov_deg}°): fx={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    return K


def predict_intrinsics_dust3r(
    images: Dict[str, np.ndarray],
    dust3r_dir: Optional[Path] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    使用 Dust3R 预测每张图的内参。
    若 Dust3R 未安装则返回 None，由上层 fallback 处理。
    """
    # 尝试导入 Dust3R
    if dust3r_dir and str(dust3r_dir) not in sys.path:
        sys.path.insert(0, str(dust3r_dir))
    try:
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images as dust3r_load
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    except ImportError:
        logger.info("Dust3R 未安装，跳过内参预测")
        return None

    logger.info("使用 Dust3R 预测相机内参...")

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        logger.warning(f"Dust3R 模型加载失败: {e}")
        return None

    # 将内存中的 numpy 图像写临时文件（Dust3R 需要文件路径）
    import tempfile, os
    tmp_paths = {}
    tmp_dir = tempfile.mkdtemp()
    for name, img in images.items():
        p = os.path.join(tmp_dir, f"{name}.jpg")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tmp_paths[name] = p

    try:
        view_names = list(tmp_paths.keys())
        img_list = dust3r_load(list(tmp_paths.values()), size=512)
        pairs = make_pairs(img_list, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1)
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

        focals = scene.get_focals().cpu().numpy()   # (N,)
        pp = scene.get_principal_points().cpu().numpy()  # (N, 2)

        result = {}
        for i, name in enumerate(view_names):
            f  = float(focals[i])
            cx = float(pp[i, 0])
            cy = float(pp[i, 1])
            result[name] = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            logger.info(f"  [{name}] Dust3R K: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        return result
    except Exception as e:
        logger.warning(f"Dust3R 推理失败: {e}")
        return None
    finally:
        for p in tmp_paths.values():
            try:
                os.remove(p)
            except Exception:
                pass


def get_intrinsics(
    images: Dict[str, np.ndarray],
    manual_intrinsics=None,
    dust3r_dir: Optional[Path] = None,
    fov_fallback_deg: float = 50.0,
) -> Dict[str, np.ndarray]:
    """
    主入口：按优先级返回每视角的 K 矩阵。

    Returns:
        {view_name: K(3,3 ndarray)}  — float64
    """
    view_names = list(images.keys())

    # ── 优先级1：手动内参 ──────────────────────────────────────────
    manual_K = load_intrinsics_from_config(manual_intrinsics)
    if manual_K is not None:
        logger.info("使用手动提供的相机内参")
        return {name: manual_K.copy() for name in view_names}

    # ── 优先级2：Dust3R ────────────────────────────────────────────
    dust3r_result = predict_intrinsics_dust3r(images, dust3r_dir)
    if dust3r_result is not None:
        return dust3r_result

    # ── 优先级3：经验估计 fallback ──────────────────────────────────
    result = {}
    for name, img in images.items():
        result[name] = estimate_intrinsics_from_image(img, fov_fallback_deg)
    return result

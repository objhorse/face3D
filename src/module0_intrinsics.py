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
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

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


def _get_work_image_size(default: int = 512) -> int:
    try:
        from src import config as cfg
        return int(getattr(cfg, "WORK_IMAGE_SIZE", default))
    except Exception:
        return default


def _resize_params(image_shape, target=None):
    if target is None:
        target = _get_work_image_size()
    h, w = image_shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    return scale, x_off, y_off


def _scale_k_to_preprocess_canvas(k_mat, calib_size, image_shape, target=None):
    """Map a full-resolution K to the square letterboxed working canvas."""
    if target is None:
        target = _get_work_image_size()
    img_h, img_w = image_shape[:2]
    current_k = k_mat.copy().astype(np.float64)
    if calib_size is not None:
        calib_w, calib_h = calib_size
        current_k[0, 0] *= img_w / calib_w
        current_k[0, 2] *= img_w / calib_w
        current_k[1, 1] *= img_h / calib_h
        current_k[1, 2] *= img_h / calib_h

    scale, x_off, y_off = _resize_params(image_shape, target)
    canvas_k = current_k.copy()
    canvas_k[0, 0] *= scale
    canvas_k[1, 1] *= scale
    canvas_k[0, 2] = current_k[0, 2] * scale + x_off
    canvas_k[1, 2] = current_k[1, 2] * scale + y_off
    return canvas_k


def _load_calibration_data(calibration_path: Optional[Union[str, Path]] = None) -> Optional[dict]:
    if calibration_path is None:
        try:
            from src import config as cfg
            if not getattr(cfg, "USE_CAMERA_CALIBRATION", True):
                return None
            calibration_path = getattr(cfg, "CAMERA_CALIBRATION_PATH", None)
        except Exception:
            calibration_path = None
    if calibration_path is None:
        return None
    path = Path(calibration_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _camera_for_view(calib_data: dict, view_name: str) -> Optional[dict]:
    for camera_name, camera_data in calib_data.get("cameras", {}).items():
        if camera_data.get("view", camera_name) == view_name:
            return camera_data
    return None


def _camera_k_for_image(camera_data: dict, image_shape) -> np.ndarray:
    k_mat = np.asarray(camera_data["K"], dtype=np.float64)
    calib_size = tuple(camera_data.get("image_size", []))
    if len(calib_size) == 2:
        calib_w, calib_h = calib_size
        img_h, img_w = image_shape[:2]
        k_mat = k_mat.copy()
        k_mat[0, 0] *= img_w / calib_w
        k_mat[0, 2] *= img_w / calib_w
        k_mat[1, 1] *= img_h / calib_h
        k_mat[1, 2] *= img_h / calib_h
    return k_mat


def undistort_images_with_calibration(
    images: Dict[str, np.ndarray],
    calibration_path: Optional[Union[str, Path]] = None,
    alpha: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Undistort calibrated full-resolution images and return the new full-res K.

    Returns:
        (images_out, full_res_intrinsics)
        full_res_intrinsics is None when no usable calibration is found.
    """
    calib_data = _load_calibration_data(calibration_path)
    if calib_data is None:
        return images, None

    if alpha is None:
        try:
            from src import config as cfg
            alpha = float(getattr(cfg, "UNDISTORT_ALPHA", 0.0))
        except Exception:
            alpha = 0.0

    out_images: Dict[str, np.ndarray] = {}
    out_k: Dict[str, np.ndarray] = {}
    used_any = False
    for view_name, img in images.items():
        camera_data = _camera_for_view(calib_data, view_name)
        if camera_data is None:
            out_images[view_name] = img
            continue
        k_mat = _camera_k_for_image(camera_data, img.shape)
        dist = np.asarray(camera_data.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
        if dist.size == 0:
            out_images[view_name] = img
            out_k[view_name] = k_mat
            continue
        h, w = img.shape[:2]
        new_k, _ = cv2.getOptimalNewCameraMatrix(k_mat, dist, (w, h), alpha, (w, h))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        undist_bgr = cv2.undistort(img_bgr, k_mat, dist, None, new_k)
        out_images[view_name] = cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB)
        out_k[view_name] = new_k.astype(np.float64)
        used_any = True
        logger.info(f"  [{view_name}] undistorted with calibrated K, alpha={alpha:.2f}")

    if not used_any and not out_k:
        return images, None
    return out_images, out_k


def load_intrinsics_from_calibration_file(
    calibration_path: Optional[Union[str, Path]],
    images: Dict[str, np.ndarray],
    work_image_size: Optional[int] = None,
) -> Optional[Dict[str, np.ndarray]]:
    data = _load_calibration_data(calibration_path)
    if data is None:
        return None
    target = work_image_size or _get_work_image_size()
    result = {}
    for camera_name, camera_data in data.get("cameras", {}).items():
        view_name = camera_data.get("view", camera_name)
        if view_name not in images:
            continue
        k_mat = np.asarray(camera_data["K"], dtype=np.float64)
        calib_size = tuple(camera_data.get("image_size", []))
        if len(calib_size) == 2:
            k_mat = _scale_k_to_preprocess_canvas(k_mat, calib_size, images[view_name].shape, target)
        result[view_name] = k_mat

    missing = [name for name in images if name not in result]
    if missing:
        logger.warning(f"Calibration file is missing views: {missing}")
    if result:
        logger.info(f"Loaded camera calibration intrinsics for {target}x{target} canvas")
        return result
    return None


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
    calibration_path: Optional[Union[str, Path]] = None,
    calibration_intrinsics: Optional[Dict[str, np.ndarray]] = None,
    work_image_size: Optional[int] = None,
    dust3r_dir: Optional[Path] = None,
    fov_fallback_deg: float = 50.0,
) -> Dict[str, np.ndarray]:
    """
    主入口：按优先级返回每视角的 K 矩阵。

    Returns:
        {view_name: K(3,3 ndarray)}  — float64
    """
    view_names = list(images.keys())
    target = work_image_size or _get_work_image_size()

    # ── 优先级1：手动内参 ──────────────────────────────────────────
    manual_K = load_intrinsics_from_config(manual_intrinsics)
    if manual_K is not None:
        logger.info("使用手动提供的相机内参")
        return {name: manual_K.copy() for name in view_names}

    if calibration_intrinsics is not None:
        logger.info(f"Using undistorted calibration intrinsics for {target}x{target} canvas")
        result = {}
        for name, img in images.items():
            if name in calibration_intrinsics:
                result[name] = _scale_k_to_preprocess_canvas(
                    calibration_intrinsics[name], None, images[name].shape, target
                )
            else:
                raw_k = estimate_intrinsics_from_image(img, fov_fallback_deg)
                result[name] = _scale_k_to_preprocess_canvas(raw_k, None, img.shape, target)
        return result

    calibration_result = load_intrinsics_from_calibration_file(calibration_path, images, target)
    if calibration_result is not None:
        result = {}
        for name, img in images.items():
            if name in calibration_result:
                result[name] = calibration_result[name].copy()
            else:
                raw_k = estimate_intrinsics_from_image(img, fov_fallback_deg)
                result[name] = _scale_k_to_preprocess_canvas(raw_k, None, img.shape, target)
        return result

    # ── 优先级2：Dust3R ────────────────────────────────────────────
    dust3r_result = predict_intrinsics_dust3r(images, dust3r_dir)
    if dust3r_result is not None:
        if target != 512:
            scale = target / 512.0
            for k_mat in dust3r_result.values():
                k_mat[0, :] *= scale
                k_mat[1, :] *= scale
        return dust3r_result

    # ── 优先级3：经验估计 fallback ──────────────────────────────────
    result = {}
    for name, img in images.items():
        raw_k = estimate_intrinsics_from_image(img, fov_fallback_deg)
        result[name] = _scale_k_to_preprocess_canvas(raw_k, None, img.shape, target)
    return result

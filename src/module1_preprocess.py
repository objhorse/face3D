"""
模块1：数据预处理

功能：
  1. 加载3张图像（左/正/右）
  2. MediaPipe Face Mesh → 468个2D关键点（含可见度）
  3. 纯黑背景分割 → 皮肤 Mask
  4. 输出调试图（关键点可视化、Mask可视化）

注意：
  - 拍摄对象眼睛闭合，MediaPipe 仍可检测但眼区精度略低
  - 使用 min_detection_confidence=0.3 提高闭眼/侧脸鲁棒性
"""
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_images(
    image_dir: Path,
    view_names: Dict[str, str],  # {"left": "IMG_0004.jpg", ...}
) -> Dict[str, np.ndarray]:
    """加载图像，返回 RGB float32 [0,255]"""
    images = {}
    for view, filename in view_names.items():
        path = image_dir / filename
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        images[view] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"  [{view}] 加载: {path.name}, 尺寸 {img_bgr.shape[1]}×{img_bgr.shape[0]}")
    return images


def segment_face_black_bg(
    image: np.ndarray,
    threshold: int = 30,
    morph_kernel: int = 15,
) -> np.ndarray:
    """
    纯黑背景分割：亮度阈值 + 形态学操作。
    返回 uint8 Mask (0/255)，255=前景（人脸+支架区域）。
    后续用 MediaPipe Mesh Polygon 可进一步限定到面部。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # 形态学：闭操作（填充皮肤小空洞） + 开操作（去小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 只保留最大连通域（去除细小噪点）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = ((labels == largest) * 255).astype(np.uint8)

    return mask


def detect_landmarks_mediapipe(
    image: np.ndarray,
    view_name: str = "",
    min_detection_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
    max_size: int = 960,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    MediaPipe Face Mesh 关键点检测。
    内部先缩放到 max_size 短边以加速，检测后反变换回原始坐标。

    Returns:
        landmarks_2d: (468, 2) float32 — 原始图像像素坐标
        visibility:   (468,)  float32 — 可见度（MediaPipe Face Mesh 通常为0，可忽略）
        若未检测到人脸则返回 (None, None)
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("请安装 mediapipe: pip install mediapipe")

    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        proc_img = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        proc_img = image
        scale = 1.0

    proc_h, proc_w = proc_img.shape[:2]
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        results = face_mesh.process(proc_img)

    if not results.multi_face_landmarks:
        logger.warning(f"  [{view_name}] MediaPipe 未检测到人脸")
        return None, None

    face_lmks = results.multi_face_landmarks[0]

    # 归一化坐标 → 缩放图坐标 → 反变换回原图坐标
    landmarks_2d = np.array(
        [[lm.x * proc_w / scale, lm.y * proc_h / scale]
         for lm in face_lmks.landmark],
        dtype=np.float32,
    )  # (468, 2) 原图像素坐标

    visibility = np.array(
        [lm.visibility for lm in face_lmks.landmark],
        dtype=np.float32,
    )  # (468,)

    n_lmks = len(face_lmks.landmark)
    logger.info(f"  [{view_name}] 检测到 {n_lmks} 个关键点 (缩放比={scale:.2f})")
    return landmarks_2d, visibility


def create_face_mask_from_landmarks(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    用 MediaPipe 人脸轮廓索引生成精确面部 Mask（排除支架夹具）。
    使用 MediaPipe 标准的脸部轮廓点（silhouette 索引）。
    """
    # MediaPipe Face Mesh 轮廓索引（468点中对应面部边缘的点）
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ]
    h, w = image_shape[:2]
    hull_pts = landmarks_2d[FACE_OVAL].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [hull_pts], 255)

    # 轻微膨胀以包含边缘皮肤
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.dilate(mask, kernel)
    return mask


TARGET_SIZE = 512   # 统一缩放目标尺寸（与 MANUAL_INTRINSICS cx=256,cy=256 对应）


def _resize_to_target(image: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    """
    等比缩放 + 居中黑边填充，输出 target×target 的 RGB 图像。
    保持人脸居中，使内参 cx=cy=target/2 始终成立。
    """
    h, w = image.shape[:2]
    if h == target and w == target:
        return image
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def preprocess_all_views(
    images: Dict[str, np.ndarray],
    debug_dir: Optional[Path] = None,
) -> Dict[str, dict]:
    """
    对所有视角执行预处理。
    所有图像统一缩放到 TARGET_SIZE×TARGET_SIZE（等比 + 黑边填充），
    确保与 MANUAL_INTRINSICS 的 cx=cy=256 匹配。

    Returns:
        {
          view_name: {
            "image":       np.ndarray (H,W,3) uint8 RGB,
            "landmarks":   np.ndarray (468,2) float32 | None,
            "visibility":  np.ndarray (468,) float32 | None,
            "face_mask":   np.ndarray (H,W) uint8,
            "bg_mask":     np.ndarray (H,W) uint8,
          }
        }
    """
    results = {}
    for view_name, image in images.items():
        logger.info(f"预处理视角: {view_name}, 原始尺寸 {image.shape[1]}×{image.shape[0]}")

        # 统一缩放到 TARGET_SIZE（保持与内参 cx=cy=256 一致）
        image = _resize_to_target(image, TARGET_SIZE)
        logger.info(f"  → 缩放后: {image.shape[1]}×{image.shape[0]}")

        # 背景分割
        bg_mask = segment_face_black_bg(image)

        # MediaPipe 关键点
        lmks, vis = detect_landmarks_mediapipe(image, view_name)

        # 面部精确 Mask
        if lmks is not None:
            face_mask = create_face_mask_from_landmarks(lmks, image.shape)
            # 与背景 Mask 取交，去除支架区域
            face_mask = cv2.bitwise_and(face_mask, bg_mask)
        else:
            face_mask = bg_mask.copy()

        results[view_name] = {
            "image":      image,
            "landmarks":  lmks,
            "visibility": vis,
            "face_mask":  face_mask,
            "bg_mask":    bg_mask,
        }

        # 调试输出
        if debug_dir is not None:
            _save_debug_images(view_name, image, lmks, face_mask, bg_mask, debug_dir)

    return results


def _save_debug_images(
    view_name: str,
    image: np.ndarray,
    landmarks: Optional[np.ndarray],
    face_mask: np.ndarray,
    bg_mask: np.ndarray,
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 保存 Mask
    cv2.imwrite(str(debug_dir / f"{view_name}_face_mask.png"), face_mask)
    cv2.imwrite(str(debug_dir / f"{view_name}_bg_mask.png"), bg_mask)

    # 保存关键点可视化
    if landmarks is not None:
        vis_img = img_bgr.copy()
        for pt in landmarks.astype(int):
            cv2.circle(vis_img, tuple(pt), 1, (0, 255, 0), -1)
        cv2.imwrite(str(debug_dir / f"{view_name}_landmarks.png"), vis_img)

    # 保存蒙版叠加图
    overlay = img_bgr.copy()
    overlay[face_mask == 0] = (overlay[face_mask == 0] * 0.3).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{view_name}_masked.png"), overlay)

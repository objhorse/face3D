"""
阶段三运行脚本：纹理融合 + GLB 打包

前提：已运行 run_phase1.py 生成了 output/meshes/face_mesh.obj 和 cameras.json

用法:
  D:/Anaconda/envs/gaussian/python.exe run_phase3.py
"""
import sys
import logging
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    from src import config as cfg
    cfg.ensure_dirs()

    # 检查前置输出
    if not (cfg.OUTPUT_MESH_DIR / "face_mesh.obj").exists():
        logger.error("face_mesh.obj 不存在，请先运行 run_phase1.py")
        sys.exit(1)
    if not (cfg.OUTPUT_MESH_DIR / "cameras.json").exists():
        logger.error("cameras.json 不存在，请先运行 run_phase1.py")
        sys.exit(1)

    t0 = time.time()

    # 加载原始图像
    logger.info("加载原始图像...")
    from src.module1_preprocess import load_images
    images = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        from src.module0_intrinsics import undistort_images_with_calibration
        images, _ = undistort_images_with_calibration(
            images,
            calibration_path=cfg.CAMERA_CALIBRATION_PATH,
            alpha=cfg.UNDISTORT_ALPHA,
        )

    # 计算高清 face mask（GrabCut 精确分割，与 hires_images 画布对齐）
    logger.info("计算人脸分割 Mask（GrabCut 模式）...")
    import cv2
    import numpy as np
    from src.module1_preprocess import detect_landmarks_mediapipe, create_face_mask_from_landmarks
    face_masks = {}
    for view_name, img in images.items():
        h, w = img.shape[:2]
        max_sz = max(h, w)
        lmks, _ = detect_landmarks_mediapipe(img, view_name)
        if lmks is not None:
            # 1. 在原始分辨率生成人脸椭圆初始 Mask
            face_oval = create_face_mask_from_landmarks(lmks, img.shape)

            # 2. GrabCut：在 1/4 分辨率运行（速度快）
            SCALE = 4
            sh, sw = h // SCALE, w // SCALE
            img_small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (sw, sh))
            oval_s    = cv2.resize(face_oval, (sw, sh), interpolation=cv2.INTER_NEAREST)

            # GrabCut 初始化掩码
            # - 人脸椭圆内 → 可能前景 (PR_FGD)
            # - 椭圆外 30px 以内 → 可能背景 (PR_BGD)
            # - 更远处 → 确定背景 (BGD)
            k_exp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            expanded = cv2.dilate(oval_s, k_exp)
            gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
            gc_mask[expanded > 0] = cv2.GC_PR_BGD
            gc_mask[oval_s > 0]   = cv2.GC_PR_FGD

            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            try:
                cv2.grabCut(img_small, gc_mask, None,
                            bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                result_s = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except Exception as e:
                logger.warning(f"  [{view_name}] GrabCut 失败 ({e})，回退到椭圆 Mask")
                result_s = oval_s

            # 3. 上采样回原分辨率 + 形态学平滑
            mask_orig = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_orig = (mask_orig > 127).astype(np.uint8) * 255
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask_orig = cv2.morphologyEx(mask_orig, cv2.MORPH_CLOSE, k_close)
        else:
            # fallback：亮度阈值
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask_orig = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            logger.warning(f"  [{view_name}] 未检测到关键点，使用亮度阈值 Mask")

        # 4. 与 hires_images 相同的居中 padding
        canvas = np.zeros((max_sz, max_sz), dtype=np.uint8)
        y_off = (max_sz - h) // 2
        x_off = (max_sz - w) // 2
        canvas[y_off:y_off + h, x_off:x_off + w] = mask_orig
        face_masks[view_name] = canvas
        logger.info(f"  [{view_name}] GrabCut Mask: {w}×{h} → {max_sz}×{max_sz} (offset x={x_off}, y={y_off})")

    # 纹理融合
    logger.info("=" * 50)
    logger.info("Stage 3: 多视角纹理融合 + GLB 打包")
    from src.module3_texture import run_texture_pipeline
    glb_path = run_texture_pipeline(
        mesh_dir=cfg.OUTPUT_MESH_DIR,
        images=images,
        hires_images=images,   # 用高清路径：自动缩放 K 矩阵匹配原始分辨率
        output_texture_dir=cfg.OUTPUT_TEXTURE_DIR,
        output_mesh_dir=cfg.OUTPUT_MESH_DIR,
        tex_size=2048,
        lighting_type="white",
        lighting_display_name="白光",
        face_masks=face_masks,
        working_image_size=cfg.WORK_IMAGE_SIZE,
    )

    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"阶段三完成！耗时 {elapsed:.1f}s")
    logger.info(f"  GLB: {glb_path}")
    logger.info(f"  纹理: {cfg.OUTPUT_TEXTURE_DIR / 'albedo_white.png'}")
    logger.info("下一步: 运行阶段四（FastAPI 服务层）")


if __name__ == "__main__":
    main()

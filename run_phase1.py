"""
阶段一+二完整运行脚本

用法:
  D:/Anaconda/envs/gaussian/python.exe run_phase1.py

输出:
  output/meshes/face_mesh.obj      — 带 UV 的精细 3D Mesh
  output/debug/                    — 调试图（关键点、Mask、深度图）
"""
import sys
import logging
from pathlib import Path

# 确保 src/ 在路径中
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

    # ── 配置 ──────────────────────────────────────────────────────────────
    # 若已拿到相机内参，在此处覆盖（或修改 config.py 中的 MANUAL_INTRINSICS）:
    # cfg.MANUAL_INTRINSICS = {"fx": 2800.0, "fy": 2800.0, "cx": 2304.0, "cy": 1728.0}

    # ── 模块0：加载图像 ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Stage 0: 加载图像")
    from src.module1_preprocess import load_images
    images = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    calibration_intrinsics = None
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        from src.module0_intrinsics import undistort_images_with_calibration
        images, calibration_intrinsics = undistort_images_with_calibration(
            images,
            calibration_path=cfg.CAMERA_CALIBRATION_PATH,
            alpha=cfg.UNDISTORT_ALPHA,
        )

    # ── 模块0：相机内参 ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Stage 0: 估计相机内参")
    from src.module0_intrinsics import get_intrinsics
    intrinsics = get_intrinsics(
        images,
        manual_intrinsics=cfg.MANUAL_INTRINSICS,
        calibration_path=cfg.CAMERA_CALIBRATION_PATH,
        calibration_intrinsics=calibration_intrinsics,
        work_image_size=cfg.WORK_IMAGE_SIZE,
        dust3r_dir=cfg.DUST3R_DIR if cfg.DUST3R_DIR.exists() else None,
        fov_fallback_deg=50.0,
    )
    for name, K in intrinsics.items():
        logger.info(f"  [{name}] K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # ── 模块1：预处理 ─────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Stage 1: 关键点检测 + 分割")
    from src.module1_preprocess import preprocess_all_views
    preprocessed = preprocess_all_views(images, debug_dir=cfg.OUTPUT_DEBUG_DIR, target_size=cfg.WORK_IMAGE_SIZE)

    detected = {k: v for k, v in preprocessed.items() if v["landmarks"] is not None}
    logger.info(f"  成功检测到关键点的视角: {list(detected.keys())}")

    if not detected:
        logger.error("所有视角均未检测到人脸！请检查图像质量或调低 min_detection_confidence")
        sys.exit(1)

    # ── 模块2：几何重建 ───────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Stage 2: 3DMM 几何重建 + 深度置换")

    if not cfg.FLAME_MODEL_PATH.exists():
        logger.error(
            f"FLAME 模型文件不存在: {cfg.FLAME_MODEL_PATH}\n"
            "请前往 https://flame.is.tue.mpg.de/ 注册并下载 FLAME 2020，\n"
            "将 generic_model.pkl 放至 models/FLAME/ 目录后重新运行。"
        )
        sys.exit(1)

    from src.module2_geometry import run_geometry_reconstruction
    output_mesh = run_geometry_reconstruction(
        preprocessed_views=preprocessed,
        intrinsics=intrinsics,
        flame_model_path=cfg.FLAME_MODEL_PATH,
        flame_landmark_path=cfg.FLAME_LANDMARK_PATH,
        deca_dir=cfg.DECA_REPO_DIR,
        deca_checkpoint=cfg.DECA_MODEL_PATH,
        depth_model_dir=cfg.DEPTH_MODEL_DIR,
        output_dir=cfg.OUTPUT_MESH_DIR,
        device=cfg.DEVICE,
        n_shape=cfg.N_SHAPE_PARAMS,
        n_exp=cfg.N_EXP_PARAMS,
        lambda_shape=cfg.LAMBDA_SHAPE,
        lambda_exp=cfg.LAMBDA_EXP,
        lbfgs_max_iter=cfg.LBFGS_MAX_ITER,
        lbfgs_lr=cfg.LBFGS_LR,
        depth_model_size=cfg.DEPTH_MODEL_SIZE,
        max_displacement=cfg.DISPLACEMENT_SCALE,
        init_backend=cfg.INIT_BACKEND,
        mica_dir=cfg.MICA_DIR,
        mica_checkpoint=cfg.MICA_CHECKPOINT,
        emoca_dir=cfg.EMOCA_DIR,
        emoca_checkpoint=cfg.EMOCA_CHECKPOINT,
    )

    logger.info("=" * 50)
    logger.info(f"阶段一+二完成！输出文件: {output_mesh}")
    logger.info(f"调试图像: {cfg.OUTPUT_DEBUG_DIR}")
    logger.info("下一步: 运行阶段三（纹理融合）")


if __name__ == "__main__":
    main()

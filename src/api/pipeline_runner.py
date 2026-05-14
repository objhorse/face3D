"""
后台流水线执行器

在线程池中运行 CPU/GPU 密集型重建管道，
通过 asyncio.Queue 向 WebSocket 推送进度。
"""
import asyncio
import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── 进度消息类型 ────────────────────────────────────────────────────────────
ProgressFn = Callable[[str, int, str], None]  # (stage, pct, message)


def _run_pipeline(
    session_id: int,
    patient_id: str,
    image_paths: dict,              # {"left": Path, "front": Path, "right": Path}
    session_output_dir: Path,       # output/sessions/{session_id}/
    progress: ProgressFn,
    manual_intrinsics: Optional[dict] = None,
) -> Path:
    """
    在工作线程中同步执行完整重建管道。
    返回 GLB 文件路径。
    """
    import sys
    ROOT = Path(__file__).parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src import config as cfg
    import numpy as np
    import cv2

    session_output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir    = session_output_dir / "meshes"
    texture_dir = session_output_dir / "textures"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载图像 ──────────────────────────────────────────────────────────
    progress("loading", 5, "加载图像...")
    images = {}
    for view, path in image_paths.items():
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {path}")
        images[view] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"  [{view}] 已加载: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    # 保存所有视角原始高清图（预处理缩放之前），用于多视角高清直采
    calibration_intrinsics = None
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        progress("calibration", 8, "calibrated undistortion...")
        from src.module0_intrinsics import undistort_images_with_calibration
        images, calibration_intrinsics = undistort_images_with_calibration(
            images,
            calibration_path=cfg.CAMERA_CALIBRATION_PATH,
            alpha=cfg.UNDISTORT_ALPHA,
        )

    original_hires = {k: v.copy() for k, v in images.items()}
    original_front = original_hires.get("front", next(iter(original_hires.values())))

    # ── 2. 相机内参 ──────────────────────────────────────────────────────────
    progress("intrinsics", 10, "计算相机内参...")
    from src.module0_intrinsics import get_intrinsics
    K = get_intrinsics(
        images=images,
        manual_intrinsics=manual_intrinsics or cfg.MANUAL_INTRINSICS,
        calibration_path=cfg.CAMERA_CALIBRATION_PATH,
        calibration_intrinsics=calibration_intrinsics,
        work_image_size=cfg.WORK_IMAGE_SIZE,
        dust3r_dir=cfg.DUST3R_DIR,
    )

    # ── 3. 关键点检测 ────────────────────────────────────────────────────────
    progress("landmarks", 20, "检测面部关键点...")
    from src.module1_preprocess import preprocess_all_views
    view_data = preprocess_all_views(images, target_size=cfg.WORK_IMAGE_SIZE)
    # 后续所有模块统一使用缩放后的图像（512×512），确保与内参 cx=cy=256 匹配
    images_resized = {k: v["image"] for k, v in view_data.items()}

    # ── 3b. 三视角纹理预融合（多视角高清直采模式下跳过）──────────────────
    # 多视角高清直采：直接将原始高清三视角图投影到 UV，无需预融合 module1b

    # ── 4. 3DMM 拟合 + 深度置换 ──────────────────────────────────────────────
    progress("fitting", 30, "3DMM 几何拟合中...")
    from src.module2_geometry import run_geometry_reconstruction
    run_geometry_reconstruction(
        preprocessed_views=view_data,
        intrinsics=K,
        flame_model_path=cfg.FLAME_MODEL_PATH,
        flame_landmark_path=cfg.FLAME_LANDMARK_PATH,
        deca_dir=cfg.DECA_REPO_DIR,
        deca_checkpoint=cfg.DECA_MODEL_PATH,
        depth_model_dir=cfg.DEPTH_MODEL_DIR,
        output_dir=mesh_dir,
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
    progress("depth", 65, "深度估计完成")

    # ── 5. 高清直采纹理烘焙 ──────────────────────────────────────────────────
    progress("texture", 75, "高清纹理烘焙中...")
    face_masks = {k: v["face_mask"] for k, v in view_data.items()}

    logger.info(
        f"多视角高清直采: 视角={list(original_hires.keys())}, "
        f"正面图 {original_front.shape[1]}×{original_front.shape[0]}"
    )

    from src.module3_texture import run_texture_pipeline
    glb_path = run_texture_pipeline(
        mesh_dir=mesh_dir,
        images=images_resized,
        output_texture_dir=texture_dir,
        output_mesh_dir=mesh_dir,
        tex_size=2048,
        lighting_type="white",
        lighting_display_name="白光",
        face_masks=face_masks,
        hires_images=original_hires,
        working_image_size=cfg.WORK_IMAGE_SIZE,
    )

    progress("done", 100, "重建完成")
    return glb_path


# ── 异步包装：在线程池中运行并推送进度 ──────────────────────────────────────

async def run_pipeline_async(
    session_id: int,
    patient_id: str,
    image_paths: dict,
    session_output_dir: Path,
    progress_queue: asyncio.Queue,
    manual_intrinsics: Optional[dict] = None,
):
    """
    从 FastAPI 后台任务调用。
    把进度推入 progress_queue（{stage, pct, message}）。
    最终推入 {"stage": "done", ...} 或 {"stage": "error", ...}。
    """
    loop = asyncio.get_event_loop()

    def _progress(stage: str, pct: int, message: str):
        asyncio.run_coroutine_threadsafe(
            progress_queue.put({"stage": stage, "pct": pct, "message": message}),
            loop,
        )

    try:
        glb_path = await loop.run_in_executor(
            None,   # 使用默认线程池
            lambda: _run_pipeline(
                session_id, patient_id, image_paths,
                session_output_dir, _progress, manual_intrinsics,
            ),
        )
        await progress_queue.put({
            "stage": "done",
            "pct": 100,
            "message": "重建完成",
            "glb_path": str(glb_path),
        })
        return glb_path

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"Pipeline 错误 (session={session_id}): {exc}\n{tb}")
        await progress_queue.put({
            "stage": "error",
            "pct": -1,
            "message": str(exc),
        })
        raise

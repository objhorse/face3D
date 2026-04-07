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

    # 计算 face mask（用于纹理采样时过滤背景和支架）
    logger.info("计算人脸分割 Mask...")
    from src.module1_preprocess import preprocess_all_views
    preproc = preprocess_all_views(images)
    face_masks = {k: v["face_mask"] for k, v in preproc.items()}

    # 纹理融合
    logger.info("=" * 50)
    logger.info("Stage 3: 多视角纹理融合 + GLB 打包")
    from src.module3_texture import run_texture_pipeline
    glb_path = run_texture_pipeline(
        mesh_dir=cfg.OUTPUT_MESH_DIR,
        images=images,
        output_texture_dir=cfg.OUTPUT_TEXTURE_DIR,
        output_mesh_dir=cfg.OUTPUT_MESH_DIR,
        tex_size=2048,
        lighting_type="white",
        lighting_display_name="白光",
        face_masks=face_masks,
    )

    elapsed = time.time() - t0
    logger.info("=" * 50)
    logger.info(f"阶段三完成！耗时 {elapsed:.1f}s")
    logger.info(f"  GLB: {glb_path}")
    logger.info(f"  纹理: {cfg.OUTPUT_TEXTURE_DIR / 'albedo_white.png'}")
    logger.info("下一步: 运行阶段四（FastAPI 服务层）")


if __name__ == "__main__":
    main()

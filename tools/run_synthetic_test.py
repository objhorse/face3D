"""
tools/run_synthetic_test.py — 在合成数据上验证 pipeline

用已知内参完全绕过 module0（Dust3R/fallback），直接喂给几何重建模块。
输出落在 subject 目录下的 pipeline_output/，不污染主输出。

用法:
    python tools/run_synthetic_test.py --subject output/test_data_v2/subject_0
    python tools/run_synthetic_test.py --subject output/test_data_v2/subject_0 --all-subjects
    python tools/run_synthetic_test.py --all-subjects --data-dir output/test_data_v2
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_known_intrinsics(cameras_json: Path) -> dict:
    """从 cameras.json 读取每视角的 K 矩阵（完全已知，无需估计）。"""
    with open(cameras_json) as f:
        data = json.load(f)
    return {
        view: np.array(info["K"], dtype=np.float64)
        for view, info in data.items()
    }


def run_subject(subject_dir: Path, cfg):
    subject_dir = subject_dir.resolve()
    image_dir   = subject_dir / "images"
    cameras_json = subject_dir / "cameras.json"
    output_dir  = subject_dir / "pipeline_output"

    logger.info("=" * 60)
    logger.info(f"Subject: {subject_dir.name}")

    # ── 加载图像 ──────────────────────────────────────────────────────
    logger.info("加载合成图像...")
    from src.module1_preprocess import load_images
    view_names = {"left": "left.jpg", "front": "front.jpg", "right": "right.jpg"}
    images = load_images(image_dir, view_names)

    # ── 直接读取已知内参，跳过 module0 ────────────────────────────────
    logger.info("读取已知相机内参（跳过 Dust3R / fallback）...")
    intrinsics = load_known_intrinsics(cameras_json)
    for name, K in intrinsics.items():
        logger.info(f"  [{name}] fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # ── 预处理：关键点检测 + 分割 ─────────────────────────────────────
    logger.info("关键点检测 + 人脸分割...")
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    from src.module1_preprocess import preprocess_all_views
    preprocessed = preprocess_all_views(images, debug_dir=debug_dir)

    detected = {k: v for k, v in preprocessed.items() if v["landmarks"] is not None}
    logger.info(f"  成功检测视角: {list(detected.keys())}")

    if not detected:
        logger.error("所有视角均未检测到人脸！合成数据可能无法被 MediaPipe 识别。")
        return False

    if len(detected) < len(images):
        missing = set(images) - set(detected)
        logger.warning(f"  以下视角未检测到关键点，将跳过: {missing}")

    # ── 几何重建 ──────────────────────────────────────────────────────
    logger.info("3DMM 几何重建 + 深度置换...")
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    from src.module2_geometry import run_geometry_reconstruction
    output_mesh = run_geometry_reconstruction(
        preprocessed_views=preprocessed,
        intrinsics=intrinsics,
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

    logger.info(f"  输出 mesh: {output_mesh}")

    # ── 与 ground truth 做简单对比（用 obj，GLB 无法直接读顶点）────────
    gt_mesh = subject_dir / "ground_truth" / "mesh.obj"
    pred_obj = output_mesh.with_suffix(".obj")          # face_mesh_with_depth.obj
    if not pred_obj.exists():
        pred_obj = output_mesh.parent / "face_mesh.obj"
    if gt_mesh.exists() and pred_obj.exists():
        _compare_meshes(pred_obj, gt_mesh)

    return True


def _compare_meshes(pred_path: Path, gt_path: Path):
    """读取两个 obj，计算顶点均方根距离（粗略评估）。"""
    def read_verts(p):
        verts = []
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
        return np.array(verts, dtype=np.float32)

    try:
        pred_v = read_verts(pred_path)
        gt_v   = read_verts(gt_path)
        if pred_v.shape == gt_v.shape:
            rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
            logger.info(f"  Vertex RMSE vs ground truth: {rmse*1000:.2f} mm")
        else:
            logger.info(f"  顶点数不同 pred={pred_v.shape[0]} gt={gt_v.shape[0]}，跳过 RMSE")
    except Exception as e:
        logger.warning(f"  mesh 对比失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="在合成数据上运行 pipeline 验证")
    parser.add_argument("--subject",  type=Path, default=None,
                        help="单个 subject 目录，如 output/test_data_v2/subject_0")
    parser.add_argument("--all-subjects", action="store_true",
                        help="遍历 --data-dir 下所有 subject_* 目录")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "output" / "test_data_v2",
                        help="包含多个 subject_* 的父目录")
    args = parser.parse_args()

    from src import config as cfg
    cfg.ensure_dirs()

    subjects = []
    if args.all_subjects:
        subjects = sorted((ROOT / args.data_dir).glob("subject_*"))
        if not subjects:
            logger.error(f"在 {args.data_dir} 下未找到任何 subject_* 目录")
            sys.exit(1)
    elif args.subject:
        subjects = [ROOT / args.subject if not args.subject.is_absolute() else args.subject]
    else:
        # 默认跑 subject_0
        subjects = [ROOT / "output" / "test_data_v2" / "subject_0"]

    results = {}
    for s in subjects:
        try:
            ok = run_subject(s, cfg)
            results[s.name] = "OK" if ok else "LANDMARK_FAIL"
        except Exception as e:
            logger.exception(f"{s.name} 运行异常: {e}")
            results[s.name] = f"ERROR: {e}"

    logger.info("\n" + "=" * 60)
    logger.info("测试汇总:")
    for name, status in results.items():
        logger.info(f"  {name}: {status}")


if __name__ == "__main__":
    main()

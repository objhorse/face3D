"""
tools/generate_test_data.py — 用 FLAME 模型生成合成测试数据

用法:
    python tools/generate_test_data.py --subjects 3 --seed 42

输出目录（每个 subject）:
    output/test_data/subject_{i}/
    ├── images/
    │   ├── left.jpg    # 左侧 45° 视角
    │   ├── front.jpg   # 正面视角
    │   └── right.jpg   # 右侧 45° 视角
    ├── ground_truth/
    │   ├── mesh.obj    # FLAME 网格（用于评估）
    │   └── flame_params.npz  # shape/exp 参数（用于参数对比）
    └── cameras.json    # 每视角的 K, R, t（用于 evaluate.py）
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

# chumpy (FLAME pickle 依赖) 在 import 时就会尝试从 numpy 导入已废弃别名
# 必须在 import chumpy 之前（即 pickle.load 之前）打补丁
for _name, _val in [("bool", bool), ("int", int), ("float", float),
                    ("complex", complex), ("object", object),
                    ("str", str), ("unicode", str)]:
    if not hasattr(np, _name):
        setattr(np, _name, _val)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 相机参数 ────────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 512, 512
FX = FY = 1200.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
CAM_DISTANCE = 0.50   # 相机到原点距离（米）

# 三个视角的水平偏转角（度），正=相机在面部右侧（拍到面部左侧）
VIEW_AZIMUTHS = {
    "left":  +45.0,
    "front":   0.0,
    "right": -45.0,
}


def _patch_numpy_compat():
    pass  # 已在模块顶部完成


def load_flame(model_path: Path, n_shape=300, n_exp=100):
    """加载 FLAME pkl，返回 v_template, shapedirs, expdirs, faces（numpy）"""
    import pickle
    _patch_numpy_compat()
    with open(model_path, "rb") as f:
        fm = pickle.load(f, encoding="latin1")
    v_template = np.array(fm["v_template"], dtype=np.float32)   # (5023, 3)
    shapedirs   = np.array(fm["shapedirs"])                      # (5023, 3, 400) or (15069, 400)
    if shapedirs.ndim == 3:
        shapedirs = shapedirs.reshape(-1, shapedirs.shape[-1])   # (15069, 400)
    faces = np.array(fm["f"], dtype=np.int32)                    # (9976, 3)
    return v_template, shapedirs[:, :n_shape], shapedirs[:, 300:300+n_exp], faces


def generate_vertices(v_template, shape_basis, exp_basis, shape_params, exp_params):
    """正向 FLAME：v_template + shape + exp → (5023, 3)"""
    n_verts = v_template.shape[0]
    delta_s = (shape_basis @ shape_params).reshape(n_verts, 3)
    delta_e = (exp_basis   @ exp_params  ).reshape(n_verts, 3)
    return v_template + delta_s + delta_e


def make_camera_pose(azimuth_deg: float, distance: float):
    """
    返回 4×4 camera-to-world 矩阵（pyrender 约定：相机看向 -Z）。
    azimuth_deg: 水平偏转角，0=正面，+45=相机在右侧（拍到左脸）
    """
    az = np.radians(azimuth_deg)
    # 相机在 XZ 平面上绕 Y 轴旋转
    cam_pos = np.array([
        distance * np.sin(az),
        0.0,
        distance * np.cos(az),
    ], dtype=np.float64)

    # lookAt：相机指向原点
    forward = -cam_pos / np.linalg.norm(cam_pos)   # 世界 forward（+Z 到原点）
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # 相机 -Z 轴朝向场景（pyrender 约定）
    pose[:3, 3] = cam_pos
    return pose


def render_views(vertices: np.ndarray, faces: np.ndarray):
    """
    用 pyrender 渲染三视角，返回 {view_name: (H,W,3) uint8 RGB}
    """
    import pyrender
    import trimesh

    # 构建 pyrender mesh（自动计算法线）
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh_pr = pyrender.Mesh.from_trimesh(tri, smooth=True)

    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    camera = pyrender.IntrinsicsCamera(fx=FX, fy=FY, cx=CX, cy=CY,
                                        znear=0.01, zfar=10.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
    results = {}

    for view_name, azimuth in VIEW_AZIMUTHS.items():
        scene = pyrender.Scene(bg_color=[0, 0, 0, 255], ambient_light=[0.3, 0.3, 0.3])
        scene.add(mesh_pr)

        cam_pose = make_camera_pose(azimuth, CAM_DISTANCE)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)   # 光源跟相机走

        color, _ = renderer.render(scene)
        results[view_name] = color   # (H, W, 3) uint8

    renderer.delete()
    return results, K


def save_cameras_json(out_path: Path, K: np.ndarray):
    """保存三视角的相机参数到 cameras.json"""
    data = {}
    for view_name, azimuth in VIEW_AZIMUTHS.items():
        pose = make_camera_pose(azimuth, CAM_DISTANCE)
        R_world2cam = pose[:3, :3].T   # world-to-camera rotation
        t_cam = -R_world2cam @ pose[:3, 3]   # world-to-camera translation
        data[view_name] = {
            "K":  K.tolist(),
            "R":  R_world2cam.tolist(),
            "t":  t_cam.tolist(),
            "azimuth_deg": azimuth,
            "distance_m":  CAM_DISTANCE,
        }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_subject(subject_dir: Path, shape_params: np.ndarray, exp_params: np.ndarray,
                     v_template, shape_basis, exp_basis, faces):
    import cv2
    import trimesh

    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "images").mkdir(exist_ok=True)
    (subject_dir / "ground_truth").mkdir(exist_ok=True)

    # 生成顶点
    verts = generate_vertices(v_template, shape_basis, exp_basis, shape_params, exp_params)

    # 渲染
    print(f"  渲染视角...", flush=True)
    images, K = render_views(verts, faces)

    # 保存图像
    for view_name, img_rgb in images.items():
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(subject_dir / "images" / f"{view_name}.jpg"), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 保存 ground truth mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(subject_dir / "ground_truth" / "mesh.obj"))

    # 保存 FLAME 参数
    np.savez(subject_dir / "ground_truth" / "flame_params.npz",
             shape_params=shape_params, exp_params=exp_params,
             v_template=v_template)

    # 保存相机参数
    save_cameras_json(subject_dir / "cameras.json", K)

    print(f"  保存到: {subject_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="生成 FLAME 合成测试数据")
    parser.add_argument("--subjects", type=int, default=3, help="生成几个 subject")
    parser.add_argument("--seed",     type=int, default=42, help="随机种子")
    parser.add_argument("--n-shape",  type=int, default=300)
    parser.add_argument("--n-exp",    type=int, default=100)
    parser.add_argument("--out-dir",  type=Path, default=ROOT / "output" / "test_data")
    args = parser.parse_args()

    from src import config as cfg

    print(f"加载 FLAME 模型: {cfg.FLAME_MODEL_PATH}", flush=True)
    v_template, shape_basis, exp_basis, faces = load_flame(
        cfg.FLAME_MODEL_PATH, args.n_shape, args.n_exp
    )
    print(f"  顶点: {v_template.shape[0]}, 面片: {faces.shape[0]}", flush=True)

    rng = np.random.default_rng(args.seed)

    for i in range(args.subjects):
        print(f"\n[Subject {i}]", flush=True)
        # shape: 以高斯分布采样，标准差 1.5（真实人脸范围约 ±2σ）
        shape_params = rng.normal(0, 1.5, size=args.n_shape).astype(np.float32)
        exp_params   = rng.normal(0, 0.5, size=args.n_exp  ).astype(np.float32)

        subject_dir = args.out_dir / f"subject_{i}"
        generate_subject(subject_dir, shape_params, exp_params,
                         v_template, shape_basis, exp_basis, faces)

    print(f"\n完成！共生成 {args.subjects} 个 subject 到 {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

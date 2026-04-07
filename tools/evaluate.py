"""
tools/evaluate.py — 计算重建 mesh 与 ground truth 的误差

用法:
    # 评估指定 session 对指定 subject 的误差
    python tools/evaluate.py \\
        --recon  output/sessions/4/meshes/face_mesh.obj \\
        --gt     output/test_data/subject_0/ground_truth/mesh.obj

    # 批量评估（自动匹配 session 和 subject）
    python tools/evaluate.py --batch

输出指标:
    Chamfer Distance (mm)  — 双向平均最近点距离，衡量整体形状相似度
    Hausdorff Distance (mm)— 单向最大误差，衡量最坏情况
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_mesh_vertices(obj_path: Path) -> np.ndarray:
    """从 .obj 文件读取顶点，返回 (N, 3) float64"""
    import trimesh
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if hasattr(mesh, "vertices"):
        return np.array(mesh.vertices, dtype=np.float64)
    raise ValueError(f"无法读取顶点: {obj_path}")


def chamfer_distance_mm(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Chamfer Distance（毫米）。
    对 pts_a 中每个点找 pts_b 最近点，反过来也做，取双向均值。
    使用 scipy 的 KDTree，速度够用（5000 点 < 1s）。
    """
    from scipy.spatial import KDTree

    tree_b = KDTree(pts_b)
    tree_a = KDTree(pts_a)

    dist_a2b, _ = tree_b.query(pts_a, k=1)
    dist_b2a, _ = tree_a.query(pts_b, k=1)

    chamfer = (dist_a2b.mean() + dist_b2a.mean()) / 2.0
    return float(chamfer * 1000)   # 米 → 毫米


def hausdorff_distance_mm(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Hausdorff Distance（毫米）：双向最大最近距离中的最大值。"""
    from scipy.spatial import KDTree

    tree_b = KDTree(pts_b)
    tree_a = KDTree(pts_a)

    dist_a2b, _ = tree_b.query(pts_a, k=1)
    dist_b2a, _ = tree_a.query(pts_b, k=1)

    h = max(dist_a2b.max(), dist_b2a.max())
    return float(h * 1000)


def align_meshes(recon_verts: np.ndarray, gt_verts: np.ndarray):
    """
    质心对齐 + 等比缩放对齐（排除位置/尺度差异，只评估形状）。
    返回对齐后的 recon_verts。
    """
    # 质心平移
    recon_center = recon_verts.mean(axis=0)
    gt_center    = gt_verts.mean(axis=0)
    recon_aligned = recon_verts - recon_center + gt_center

    # 尺度对齐（使用方差比）
    recon_scale = np.sqrt(((recon_aligned - gt_center) ** 2).sum(axis=1).mean())
    gt_scale    = np.sqrt(((gt_verts    - gt_center) ** 2).sum(axis=1).mean())
    if recon_scale > 1e-8:
        recon_aligned = (recon_aligned - gt_center) * (gt_scale / recon_scale) + gt_center

    return recon_aligned


def evaluate_pair(recon_path: Path, gt_path: Path, save_json: Path = None) -> dict:
    print(f"重建 mesh : {recon_path}")
    print(f"GT mesh   : {gt_path}")

    recon_verts = load_mesh_vertices(recon_path)
    gt_verts    = load_mesh_vertices(gt_path)

    print(f"  重建顶点数: {len(recon_verts)}")
    print(f"  GT 顶点数 : {len(gt_verts)}")

    # 对齐（消除绝对位置和尺度差异）
    recon_aligned = align_meshes(recon_verts, gt_verts)

    chamfer   = chamfer_distance_mm(recon_aligned, gt_verts)
    hausdorff = hausdorff_distance_mm(recon_aligned, gt_verts)

    # 评级
    def grade(v):
        if v < 2:   return "好 [<2mm]"
        if v < 5:   return "可接受 [2-5mm]"
        return "需改进 [>5mm]"

    result = {
        "recon_path":       str(recon_path),
        "gt_path":          str(gt_path),
        "recon_verts":      len(recon_verts),
        "gt_verts":         len(gt_verts),
        "chamfer_mm":       round(chamfer, 3),
        "hausdorff_mm":     round(hausdorff, 3),
        "chamfer_grade":    grade(chamfer),
        "hausdorff_grade":  grade(hausdorff),
    }

    print()
    print("=" * 45)
    print(f"  Chamfer Distance  : {chamfer:.2f} mm  [{grade(chamfer)}]")
    print(f"  Hausdorff Distance: {hausdorff:.2f} mm  [{grade(hausdorff)}]")
    print("=" * 45)
    print("  评级标准: 好 < 2mm | 可接受 2-5mm | 需改进 > 5mm")
    print()

    if save_json:
        with open(save_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果保存到: {save_json}")

    return result


def main():
    parser = argparse.ArgumentParser(description="评估 3D 人脸重建误差")
    parser.add_argument("--recon", type=Path,
                        help="重建 mesh 路径（.obj 或 .glb）")
    parser.add_argument("--gt",    type=Path,
                        help="Ground truth mesh 路径（.obj）")
    parser.add_argument("--save-json", type=Path,
                        help="保存评估结果到 JSON 文件")
    parser.add_argument("--batch", action="store_true",
                        help="批量评估：自动匹配 output/sessions/ 和 output/test_data/")
    args = parser.parse_args()

    if args.batch:
        # 批量模式：按 session id 顺序匹配 subject
        sessions_dir  = ROOT / "output" / "sessions"
        test_data_dir = ROOT / "output" / "test_data"

        sessions  = sorted(sessions_dir.glob("*/meshes/face_mesh.obj"))
        subjects  = sorted(test_data_dir.glob("*/ground_truth/mesh.obj"))

        if not sessions:
            print("没有找到重建结果（output/sessions/*/meshes/face_mesh.obj）")
            return
        if not subjects:
            print("没有找到测试数据（output/test_data/*/ground_truth/mesh.obj）")
            print("请先运行: python tools/generate_test_data.py")
            return

        all_results = []
        for i, (recon, gt) in enumerate(zip(sessions, subjects)):
            print(f"\n── Pair {i} ──────────────────────────────")
            r = evaluate_pair(recon, gt)
            all_results.append(r)

        # 汇总
        chamfers   = [r["chamfer_mm"]   for r in all_results]
        hausdorffs = [r["hausdorff_mm"] for r in all_results]
        print(f"\n汇总 ({len(all_results)} 对):")
        print(f"  Chamfer 均值   : {np.mean(chamfers):.2f} mm")
        print(f"  Hausdorff 均值 : {np.mean(hausdorffs):.2f} mm")

    elif args.recon and args.gt:
        evaluate_pair(args.recon, args.gt, save_json=args.save_json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

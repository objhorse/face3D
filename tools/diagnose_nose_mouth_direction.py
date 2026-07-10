from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.initializers.face_alignment_initializer import get_fa_per_view
from src.module0_intrinsics import undistort_images_with_calibration
from src.module1_preprocess import _resize_to_target, load_images
from src.module2_geometry import FLAMEModel, load_flame_landmark_mapping


NOSE_ALL = np.arange(27, 36, dtype=np.int64)
NOSE_BRIDGE = np.arange(27, 31, dtype=np.int64)
NOSE_BASE = np.arange(31, 36, dtype=np.int64)
MOUTH_OUTER = np.arange(48, 60, dtype=np.int64)
MOUTH_INNER = np.arange(60, 68, dtype=np.int64)
MOUTH_ALL = np.arange(48, 68, dtype=np.int64)
EYES = np.arange(36, 48, dtype=np.int64)

GROUPS = {
    "nose_all": NOSE_ALL,
    "nose_bridge": NOSE_BRIDGE,
    "nose_base_wings": NOSE_BASE,
    "mouth_outer": MOUTH_OUTER,
    "mouth_inner": MOUTH_INNER,
    "mouth_all": MOUTH_ALL,
    "eyes_reference": EYES,
}


def _read_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not verts:
        raise RuntimeError(f"No vertices found in {path}")
    return np.asarray(verts, dtype=np.float64)


def _project_vertices(vertices: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    v_cam = (R @ vertices.T + t[:, None]).T
    z = np.clip(v_cam[:, 2], 1e-6, None)
    v_hom = (K @ v_cam.T).T
    return np.stack([v_hom[:, 0] / z, v_hom[:, 1] / z], axis=1)


def _landmark_projection(
    proj_vertices: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    return (
        proj_vertices[lmk_tri_vidx[:, 0]] * bary[:, 0:1]
        + proj_vertices[lmk_tri_vidx[:, 1]] * bary[:, 1:2]
        + proj_vertices[lmk_tri_vidx[:, 2]] * bary[:, 2:3]
    )


def _subset_stats(errors: np.ndarray, idx: Iterable[int]) -> dict:
    values = errors[np.asarray(list(idx), dtype=np.int64)]
    return {
        "mean_px": round(float(values.mean()), 3),
        "max_px": round(float(values.max()), 3),
        "p75_px": round(float(np.percentile(values, 75)), 3),
    }


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _front_dimensions(target: np.ndarray, pred: np.ndarray) -> dict:
    def pair(name: str, i: int, j: int) -> dict:
        gt = _distance(target[i], target[j])
        pr = _distance(pred[i], pred[j])
        return {
            "name": name,
            "target_px": round(gt, 3),
            "model_px": round(pr, 3),
            "model_minus_target_px": round(pr - gt, 3),
            "ratio_model_to_target": round(pr / gt, 4) if gt > 1e-6 else None,
        }

    nose_base_center_gt = target[NOSE_BASE].mean(axis=0)
    nose_base_center_pr = pred[NOSE_BASE].mean(axis=0)
    mouth_center_gt = target[MOUTH_OUTER].mean(axis=0)
    mouth_center_pr = pred[MOUTH_OUTER].mean(axis=0)
    return {
        "nose_width_31_35": pair("nose_width_31_35", 31, 35),
        "mouth_width_48_54": pair("mouth_width_48_54", 48, 54),
        "nose_base_center_shift_px": [
            round(float(nose_base_center_pr[0] - nose_base_center_gt[0]), 3),
            round(float(nose_base_center_pr[1] - nose_base_center_gt[1]), 3),
        ],
        "mouth_center_shift_px": [
            round(float(mouth_center_pr[0] - mouth_center_gt[0]), 3),
            round(float(mouth_center_pr[1] - mouth_center_gt[1]), 3),
        ],
    }


def _draw_overlay(image_rgb: np.ndarray, target: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    img = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    colors = {
        "nose_bridge": (255, 180, 0),
        "nose_base_wings": (0, 140, 255),
        "mouth_outer": (255, 0, 180),
        "mouth_inner": (180, 0, 255),
    }
    draw_groups = {
        "nose_bridge": NOSE_BRIDGE,
        "nose_base_wings": NOSE_BASE,
        "mouth_outer": MOUTH_OUTER,
        "mouth_inner": MOUTH_INNER,
    }
    for name, idxs in draw_groups.items():
        color = colors[name]
        for idx in idxs:
            gt = tuple(np.round(target[idx]).astype(int))
            pr = tuple(np.round(pred[idx]).astype(int))
            cv2.circle(img, gt, 3, (0, 255, 0), -1)
            cv2.circle(img, pr, 3, (0, 0, 255), -1)
            cv2.line(img, gt, pr, color, 2, cv2.LINE_AA)
            cv2.putText(img, str(idx), (pr[0] + 3, pr[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    all_pts = np.vstack([target[NOSE_ALL], pred[NOSE_ALL], target[MOUTH_ALL], pred[MOUTH_ALL]])
    x0, y0 = np.floor(all_pts.min(axis=0) - 70).astype(int)
    x1, y1 = np.ceil(all_pts.max(axis=0) + 70).astype(int)
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (240, 240, 240), 2)
    cv2.putText(img, "green=target red=model lines=needed local correction", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    crop = img[y0:y1, x0:x1]
    if crop.size:
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_crop.jpg")), crop)


def _write_html(out_dir: Path, report: dict) -> None:
    rows = []
    for view, data in report["views"].items():
        for group, stats in data["groups"].items():
            rows.append(
                "<tr>"
                f"<td>{html.escape(view)}</td>"
                f"<td>{html.escape(group)}</td>"
                f"<td>{stats['mean_px']:.3f}</td>"
                f"<td>{stats['p75_px']:.3f}</td>"
                f"<td>{stats['max_px']:.3f}</td>"
                "</tr>"
            )

    front_dims = report.get("front_dimensions", {})
    dim_rows = []
    for key in ("nose_width_31_35", "mouth_width_48_54"):
        d = front_dims.get(key, {})
        dim_rows.append(
            "<tr>"
            f"<td>{html.escape(key)}</td>"
            f"<td>{d.get('target_px')}</td>"
            f"<td>{d.get('model_px')}</td>"
            f"<td>{d.get('model_minus_target_px')}</td>"
            f"<td>{d.get('ratio_model_to_target')}</td>"
            "</tr>"
        )

    cards = []
    for view in report["views"]:
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(view)}</h2>
              <img src="{view}_nose_mouth_overlay.jpg" />
              <img src="{view}_nose_mouth_overlay_crop.jpg" />
            </section>
            """
        )

    decision = report["decision"]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>鼻口局部形变方向检测</title>
<style>
body {{ margin: 0; padding: 28px; background: #101826; color: #e8f1ff; font-family: Arial, "Microsoft YaHei", sans-serif; }}
h1 {{ margin: 0 0 10px; }}
.muted {{ color: #9fb0c8; }}
.verdict {{ padding: 16px 18px; border-radius: 14px; background: {"#193d2a" if decision["reasonable"] else "#432323"}; margin: 18px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
.card {{ background: #172235; border: 1px solid #26364d; border-radius: 14px; padding: 14px; }}
img {{ max-width: 100%; background: #000; border-radius: 10px; margin: 8px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; background: #111b2a; }}
td, th {{ border: 1px solid #2b3d55; padding: 8px 10px; text-align: left; }}
th {{ background: #20304a; }}
.good {{ color: #84f2af; }}
.warn {{ color: #ffd37a; }}
code {{ color: #b9d7ff; }}
</style>
</head>
<body>
<h1>鼻口局部形变方向检测</h1>
<p class="muted">绿色点是检测目标，红色点是当前模型投影；彩色线表示鼻口局部需要修正的方向和大小。</p>
<div class="verdict">
  <h2>结论：{"方向合理，建议做鼻口局部 residual 小实验" if decision["reasonable"] else "证据不足，暂不建议做鼻口局部 residual"}</h2>
  <p>{html.escape(decision["reason"])}</p>
</div>
<h2>锚点锁定证据</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>stable anchor ratio</td><td>{report["anchor_summary"].get("anchor_ratio")}</td></tr>
<tr><td>anchor moved vertices</td><td>{report["anchor_summary"].get("anchor_moved_vertices")}</td></tr>
<tr><td>anchor max offset m</td><td>{report["anchor_summary"].get("anchor_max_offset_m")}</td></tr>
<tr><td>mouth mean before -> after</td><td>{report["anchor_summary"].get("mouth_before")} -> {report["anchor_summary"].get("mouth_after")}</td></tr>
</table>
<h2>鼻口/眼睛误差分组</h2>
<table>
<tr><th>视角</th><th>区域</th><th>mean px</th><th>p75 px</th><th>max px</th></tr>
{''.join(rows)}
</table>
<h2>正面比例检测</h2>
<table>
<tr><th>指标</th><th>照片目标 px</th><th>模型 px</th><th>模型-目标 px</th><th>模型/目标</th></tr>
{''.join(dim_rows)}
</table>
<p class="muted">center shift: nose_base={front_dims.get("nose_base_center_shift_px")}, mouth={front_dims.get("mouth_center_shift_px")}</p>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    out_dir = cfg.OUTPUT_DEBUG_DIR / "nose_mouth_direction_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = cfg.OUTPUT_MESH_DIR / "face_mesh.obj"
    camera_path = cfg.OUTPUT_MESH_DIR / "cameras.json"
    summary_path = cfg.OUTPUT_DEBUG_DIR / "personal_residual_deform" / "summary.json"

    vertices = _read_obj_vertices(mesh_path)
    cameras = json.loads(camera_path.read_text(encoding="utf-8"))["views"]

    lmk_data = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
    if lmk_data is None:
        raise RuntimeError("Missing FLAME landmark embedding; cannot run nose-mouth diagnosis")
    flame = FLAMEModel(cfg.FLAME_MODEL_PATH, n_shape=cfg.N_SHAPE_PARAMS, n_exp=cfg.N_EXP_PARAMS)
    flame_faces = flame.faces.cpu().numpy()
    lmk_tri_vidx = flame_faces[lmk_data["face_idx"]]
    bary = lmk_data["bary_coords"]

    raw_images = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        raw_images, _ = undistort_images_with_calibration(
            raw_images,
            calibration_path=cfg.CAMERA_CALIBRATION_PATH,
            alpha=cfg.UNDISTORT_ALPHA,
        )
    images: Dict[str, np.ndarray] = {
        view: _resize_to_target(img, cfg.WORK_IMAGE_SIZE)
        for view, img in raw_images.items()
    }
    for view, img in images.items():
        cv2.imwrite(str(out_dir / f"{view}_work_image.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    fa = get_fa_per_view(images, cfg.DEVICE)
    report = {
        "mesh": str(mesh_path),
        "views": {},
        "front_dimensions": {},
        "anchor_summary": {},
        "decision": {},
    }

    nose_mouth_means = []
    for view, image in images.items():
        target3 = fa.get(view)
        if target3 is None:
            continue
        target = target3[:, :2].astype(np.float64)
        cam = cameras[view]
        K = np.asarray(cam["K"], dtype=np.float64)
        R = np.asarray(cam["R"], dtype=np.float64)
        t = np.asarray(cam["t"], dtype=np.float64)
        proj_v = _project_vertices(vertices, K, R, t)
        pred = _landmark_projection(proj_v, lmk_tri_vidx, bary)
        errors = np.linalg.norm(pred - target, axis=1)
        groups = {name: _subset_stats(errors, idx) for name, idx in GROUPS.items()}
        nose_mouth_means.extend([groups["nose_base_wings"]["mean_px"], groups["mouth_outer"]["mean_px"]])
        report["views"][view] = {
            "overall_mean_px": round(float(errors.mean()), 3),
            "groups": groups,
        }
        if view == "front":
            report["front_dimensions"] = _front_dimensions(target, pred)
        _draw_overlay(image, target, pred, out_dir / f"{view}_nose_mouth_overlay.jpg")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        safety = summary.get("safety", {})
        anchors = summary.get("stable_anchors", {})
        before = summary.get("before_landmarks", {})
        after = summary.get("after_landmarks", {})
        report["anchor_summary"] = {
            "anchor_ratio": anchors.get("anchor_ratio"),
            "anchor_vertices": anchors.get("anchor_vertices"),
            "anchor_moved_vertices": safety.get("anchor_moved_vertices"),
            "anchor_max_offset_m": safety.get("anchor_max_offset_m"),
            "mouth_before": before.get("mouth_mean_px"),
            "mouth_after": after.get("mouth_mean_px"),
            "stable_before": before.get("stable_mean_px"),
            "stable_after": after.get("stable_mean_px"),
        }

    avg_nose_mouth = float(np.mean(nose_mouth_means)) if nose_mouth_means else 0.0
    anchor_locked = (
        report["anchor_summary"].get("anchor_moved_vertices") == 0
        and float(report["anchor_summary"].get("anchor_ratio") or 0.0) > 0.25
    )
    front_nose_ratio = report["front_dimensions"].get("nose_width_31_35", {}).get("ratio_model_to_target")
    nose_width_gap = abs(float(report["front_dimensions"].get("nose_width_31_35", {}).get("model_minus_target_px") or 0.0))
    reasonable = bool(avg_nose_mouth >= 8.0 and anchor_locked)
    reason = (
        f"鼻翼/嘴外轮廓平均误差约 {avg_nose_mouth:.2f}px；稳定锚点锁住 {report['anchor_summary'].get('anchor_ratio')} 的顶点且锚点移动为 "
        f"{report['anchor_summary'].get('anchor_moved_vertices')}。正面鼻宽模型/目标={front_nose_ratio}，鼻宽差={nose_width_gap:.2f}px。"
    )
    report["decision"] = {
        "reasonable": reasonable,
        "avg_nose_mouth_key_error_px": round(avg_nose_mouth, 3),
        "anchor_locked": anchor_locked,
        "reason": reason,
    }

    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir, report)
    print(json.dumps({
        "html": str(out_dir / "index.html"),
        "reasonable": reasonable,
        "avg_nose_mouth_key_error_px": round(avg_nose_mouth, 3),
        "front_nose_width_ratio": front_nose_ratio,
        "front_nose_width_gap_px": round(nose_width_gap, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

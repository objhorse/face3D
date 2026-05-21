"""Diagnose why chin texture can leak toward neck.

This is a read-only audit. It recreates the Phase 3 mask and mesh-selection
decisions, writes visual overlays, and builds a Chinese HTML report.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.module0_intrinsics import undistort_images_with_calibration
from src.module1_preprocess import (
    FACE_OVAL,
    create_face_mask_from_landmarks,
    detect_landmarks_mediapipe,
    load_images,
)
from src.module3_texture import (
    _expand_face_selection,
    _render_camera_depth,
    _strict_visible_face_filter,
    add_uv_hole_fill_faces_for_bake,
    load_cameras,
    load_mesh_obj,
    project_texture_points_to_image,
    rasterize_uv_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _phase3_grabcut_mask(image: np.ndarray, lmks: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    face_oval = create_face_mask_from_landmarks(lmks, image.shape)
    scale = 4
    sh, sw = h // scale, w // scale
    img_small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (sw, sh))
    oval_s = cv2.resize(face_oval, (sw, sh), interpolation=cv2.INTER_NEAREST)

    k_exp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    expanded = cv2.dilate(oval_s, k_exp)
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[expanded > 0] = cv2.GC_PR_BGD
    gc_mask[oval_s > 0] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(img_small, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        result_s = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception as exc:
        logger.warning("GrabCut failed, using face oval: %s", exc)
        result_s = oval_s

    mask = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return mask


def _pad_square(image: np.ndarray, fill: int = 0) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    side = max(h, w)
    if image.ndim == 2:
        canvas = np.full((side, side), fill, dtype=image.dtype)
    else:
        canvas = np.full((side, side, image.shape[2]), fill, dtype=image.dtype)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = image
    return canvas, (x_off, y_off)


def _chin_lines(lmks: np.ndarray, pad: Tuple[int, int], image_shape: Tuple[int, int]) -> Dict[str, int]:
    x_off, y_off = pad
    h, _ = image_shape[:2]
    chin_y = int(np.clip(lmks[152, 1] + y_off, 0, h - 1))
    oval_pts = lmks[FACE_OVAL].astype(np.int32)
    face_h = max(int(oval_pts[:, 1].max() - oval_pts[:, 1].min()), 1)
    neck_cut_y = min(h - 1, int(chin_y + max(4, int(0.04 * face_h))))
    strict_cut_y = min(h - 1, int(chin_y + max(3, int(0.03 * face_h))))
    return {"chin_y": chin_y, "neck_cut_y": neck_cut_y, "strict_cut_y": strict_cut_y, "face_h": face_h}


def _overlay_mask(image: np.ndarray, mask: np.ndarray, lines: Dict[str, int], title: str) -> np.ndarray:
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    out = np.where(mask[:, :, None] > 0, (out * 0.55 + green * 0.45).astype(np.uint8), out)
    cv2.line(out, (0, lines["chin_y"]), (out.shape[1] - 1, lines["chin_y"]), (255, 220, 0), 4)
    cv2.line(out, (0, lines["strict_cut_y"]), (out.shape[1] - 1, lines["strict_cut_y"]), (255, 150, 0), 4)
    cv2.line(out, (0, lines["neck_cut_y"]), (out.shape[1] - 1, lines["neck_cut_y"]), (255, 60, 60), 4)
    cv2.putText(out, title, (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def _overlay_leak(image: np.ndarray, phase3_mask: np.ndarray, strict_mask: np.ndarray, lines: Dict[str, int]) -> np.ndarray:
    out = image.copy()
    leak = (phase3_mask > 0) & (strict_mask == 0)
    below_strict = np.zeros_like(leak)
    below_strict[lines["strict_cut_y"]:, :] = True
    below_neck = np.zeros_like(leak)
    below_neck[lines["neck_cut_y"]:, :] = True
    red = np.zeros_like(out)
    red[:, :, 0] = 255
    orange = np.zeros_like(out)
    orange[:, :, 0] = 255
    orange[:, :, 1] = 150
    out = np.where(leak[:, :, None], (out * 0.45 + orange * 0.55).astype(np.uint8), out)
    out = np.where((leak & below_neck)[:, :, None], (out * 0.25 + red * 0.75).astype(np.uint8), out)
    cv2.line(out, (0, lines["strict_cut_y"]), (out.shape[1] - 1, lines["strict_cut_y"]), (255, 150, 0), 4)
    cv2.line(out, (0, lines["neck_cut_y"]), (out.shape[1] - 1, lines["neck_cut_y"]), (255, 60, 60), 4)
    cv2.putText(out, "orange=removed by chin cut, red=below neck cut", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3, cv2.LINE_AA)
    return out


def _chin_crop(image: np.ndarray, lines: Dict[str, int], mask: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = image.shape[:2]
    face_h = max(int(lines["face_h"]), 1)
    y0 = max(0, int(lines["chin_y"] - 0.22 * face_h))
    y1 = min(h, int(lines["neck_cut_y"] + 0.24 * face_h))
    if mask is not None and mask.any():
        ys, xs = np.where(mask > 0)
        x0 = max(0, int(xs.min() - 0.16 * face_h))
        x1 = min(w, int(xs.max() + 0.16 * face_h))
    else:
        x0, x1 = 0, w
    crop = image[y0:y1, x0:x1].copy()
    for key, color in [("chin_y", (255, 220, 0)), ("strict_cut_y", (255, 150, 0)), ("neck_cut_y", (255, 60, 60))]:
        yy = int(lines[key] - y0)
        if 0 <= yy < crop.shape[0]:
            cv2.line(crop, (0, yy), (crop.shape[1] - 1, yy), color, 4)
    return crop


def _save_thumb(path: Path, image: np.ndarray, max_side: int = 1200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _draw_points(image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], radius: int = 2) -> None:
    h, w = image.shape[:2]
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(image, (ix, iy), radius, color, -1, lineType=cv2.LINE_AA)


def _draw_triangles(image: np.ndarray, tris: np.ndarray, color: Tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    for tri in tris:
        pts = np.round(tri).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.polylines(image, [pts], True, color, 2, lineType=cv2.LINE_AA)


def _mesh_boundary_audit(
    vertices: np.ndarray,
    faces: np.ndarray,
    cameras: Dict[str, dict],
    square_images: Dict[str, np.ndarray],
    phase3_masks: Dict[str, np.ndarray],
    strict_masks: Dict[str, np.ndarray],
    lines_by_view: Dict[str, Dict[str, int]],
    out_dir: Path,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    face_centers = vertices[faces].mean(axis=1).astype(np.float32)
    keep = np.zeros(len(faces), dtype=bool)
    kept_by_view = {}
    stats = {}

    for view_name, cam in cameras.items():
        if view_name not in square_images or view_name not in phase3_masks:
            continue
        image = square_images[view_name]
        mask = phase3_masks[view_name]
        h, w = image.shape[:2]
        depth = _render_camera_depth(vertices, faces, cam["K"], cam["R"], cam["t"], (h, w))
        _, z, proj, front = project_texture_points_to_image(face_centers, cam["K"], cam["R"], cam["t"])
        in_img = front & (proj[:, 0] >= 0) & (proj[:, 0] < w - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h - 1)
        px = np.clip(proj[:, 0].astype(np.int32), 0, w - 1)
        py = np.clip(proj[:, 1].astype(np.int32), 0, h - 1)
        in_mask = mask[py, px] > 127
        z_ok = z <= (depth[py, px] + float(cfg.VISIBLE_FACE_CROP_Z_TOL))
        base_view_keep = in_img & in_mask & z_ok
        view_keep, strict_stats = _strict_visible_face_filter(
            vertices, faces, cam["K"], cam["R"], cam["t"], mask, proj, base_view_keep, (h, w)
        )
        keep |= view_keep
        kept_by_view[view_name] = view_keep

        # Boundary risk: kept face center is valid, but at least one projected vertex
        # crosses the explicit chin-cut mask. This catches triangles whose centers are
        # legal but whose lower edge can still sample neck pixels.
        v_cam, vz, vproj, vfront = project_texture_points_to_image(vertices, cam["K"], cam["R"], cam["t"])
        tri_proj = vproj[faces]
        tri_front = np.all(vfront[faces], axis=1)
        tri_px = np.clip(tri_proj[:, :, 0].astype(np.int32), 0, w - 1)
        tri_py = np.clip(tri_proj[:, :, 1].astype(np.int32), 0, h - 1)
        strict = strict_masks[view_name]
        vertex_strict = strict[tri_py, tri_px] > 127
        below_chin = np.any(tri_proj[:, :, 1] > lines_by_view[view_name]["chin_y"], axis=1)
        below_strict = np.any(tri_proj[:, :, 1] > lines_by_view[view_name]["strict_cut_y"], axis=1)
        below_neck = np.any(tri_proj[:, :, 1] > lines_by_view[view_name]["neck_cut_y"], axis=1)
        any_out_strict = ~np.all(vertex_strict, axis=1)
        risky = view_keep & tri_front & any_out_strict
        risky_below_strict = view_keep & tri_front & below_strict
        risky_below_neck = view_keep & tri_front & below_neck

        overlay = image.copy()
        _draw_points(overlay, proj[view_keep], (0, 230, 60), radius=1)
        _draw_points(overlay, proj[risky], (255, 80, 40), radius=2)
        _draw_triangles(overlay, tri_proj[risky][:800], (255, 120, 40))
        cv2.line(overlay, (0, lines_by_view[view_name]["strict_cut_y"]), (w - 1, lines_by_view[view_name]["strict_cut_y"]), (255, 150, 0), 5)
        cv2.line(overlay, (0, lines_by_view[view_name]["neck_cut_y"]), (w - 1, lines_by_view[view_name]["neck_cut_y"]), (255, 60, 60), 5)
        cv2.putText(overlay, "green=kept face center, orange=boundary-risk kept face", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        _save_thumb(out_dir / f"{view_name}_03_kept_face_centers_boundary_risk.jpg", overlay)
        _save_thumb(out_dir / f"{view_name}_06_mesh_boundary_chin_closeup.jpg", _chin_crop(overlay, lines_by_view[view_name], mask))

        stats[view_name] = {
            "kept_by_center_before_strict_filter": int(base_view_keep.sum()),
            "strict_filter_rejected": int(strict_stats.get("rejected", 0)),
            "strict_filter_rejected_lower": int(strict_stats.get("rejected_lower", 0)),
            "strict_filter_rejected_boundary": int(strict_stats.get("rejected_boundary", 0)),
            "kept_after_strict_filter": int(view_keep.sum()),
            "kept_boundary_risk": int(risky.sum()),
            "kept_cross_chin_line": int((view_keep & tri_front & below_chin).sum()),
            "kept_cross_strict_cut_line": int(risky_below_strict.sum()),
            "kept_cross_neck_cut_line": int(risky_below_neck.sum()),
            "risk_ratio": float(risky.sum() / max(view_keep.sum(), 1)),
        }

    keep_expanded = _expand_face_selection(faces, keep, int(cfg.VISIBLE_FACE_CROP_DILATE_RINGS))
    stats["combined"] = {
        "keep_before_rings": int(keep.sum()),
        "keep_after_rings": int(keep_expanded.sum()),
        "rings": int(cfg.VISIBLE_FACE_CROP_DILATE_RINGS),
    }
    return keep_expanded, stats


def _uv_stage_audit(vertices, faces, uv_verts, uv_faces, out_dir: Path) -> Dict[str, int]:
    tex_size = 2048
    tri_map_base, _ = rasterize_uv_map(uv_verts, uv_faces, tex_size)
    base_valid = tri_map_base >= 0
    holes_base = (~base_valid).sum()
    bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces, added = add_uv_hole_fill_faces_for_bake(
        vertices, faces, uv_verts, uv_faces, tex_size=tex_size
    )
    tri_map_bake, _ = rasterize_uv_map(bake_uv_verts, bake_uv_faces, tex_size)
    filled = tri_map_bake >= len(uv_faces) if added else np.zeros_like(base_valid, dtype=bool)
    repair_roi = cv2.dilate(
        filled.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=4,
    ).astype(bool) & (tri_map_bake >= 0)

    vis = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    vis[tri_map_bake >= 0] = (55, 55, 55)
    vis[base_valid] = (70, 180, 80)
    vis[repair_roi] = (255, 160, 0)
    vis[filled] = (255, 60, 60)
    _save_thumb(out_dir / "04_uv_hole_fill_and_repair_roi.png", vis)

    return {
        "base_valid_pixels": int(base_valid.sum()),
        "base_invalid_pixels": int(holes_base),
        "added_temporary_faces": int(added),
        "filled_uv_pixels": int(filled.sum()),
        "repair_roi_pixels": int(repair_roi.sum()),
    }


def _write_html(out_dir: Path, summary: dict) -> None:
    cards = []
    for view in ["front", "left", "right"]:
        cards.append(f"""
        <section>
          <h2>{view} 视角：mask、投影边界和下巴局部</h2>
          <div class="grid">
            <figure><img src="{view}_01_phase3_mask.jpg"><figcaption>Phase3 实际使用的高分辨率 GrabCut mask。黄色线=下巴点，橙色线=建议强制截断线，红色线=当前 neck cut 参考线。</figcaption></figure>
            <figure><img src="{view}_02_strict_mask.jpg"><figcaption>同一个 Phase3 mask，但在下巴下方强制截断后的对照。它用来判断“原始 mask 是否把脖子留下来了”。</figcaption></figure>
            <figure><img src="{view}_02b_phase3_extra_vs_strict.jpg"><figcaption>橙色=会被下巴截断移除的区域；红色=已经低于 neck cut 的区域。如果这里很多，问题来自 mask。</figcaption></figure>
            <figure><img src="{view}_05_mask_chin_closeup.jpg"><figcaption>下巴局部放大：专门看 Phase3 mask 在下巴到脖子之间留下了多少可采样区域。</figcaption></figure>
            <figure><img src="{view}_03_kept_face_centers_boundary_risk.jpg"><figcaption>绿色=visible crop 保留的三角面中心；橙色=中心合法、但三角面边缘越过下巴截断 mask 的风险面。</figcaption></figure>
            <figure><img src="{view}_06_mesh_boundary_chin_closeup.jpg"><figcaption>投影边界局部放大：如果这里橙色很多，说明不是 mask 本身，而是“只看三角面中心”的保留策略太宽。</figcaption></figure>
          </div>
        </section>
        """)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>下巴/脖子投影污染根因排查</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background:#111; color:#eee; }}
    h1, h2 {{ color:#fff; }}
    .summary {{ background:#1d1d1d; border:1px solid #333; padding:16px; border-radius:8px; }}
    .grid {{ display:grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap:16px; }}
    figure {{ margin:0; background:#191919; border:1px solid #333; padding:10px; border-radius:8px; }}
    img {{ width:100%; height:auto; display:block; }}
    figcaption {{ color:#ccc; font-size:14px; margin-top:8px; line-height:1.5; }}
    code, pre {{ color:#b8e986; }}
  </style>
</head>
<body>
  <h1>下巴/脖子投影污染根因排查</h1>
  <div class="summary">
    <h2>怎么看这份图</h2>
    <p><b>第一层：mask。</b>如果橙色/红色在下巴下方很多，说明 Phase3 的高分辨率 mask 本身把脖子区域放进来了。</p>
    <p><b>第二层：投影边界。</b>如果 mask 没有明显漏脖子，但橙色风险三角面集中在下巴边界，说明 visible face crop 只看三角面中心，三角面边缘仍可能跨到脖子。</p>
    <p><b>第三层：UV 修补。</b>如果前两层都很干净，但最终 albedo 仍被脖子颜色污染，就要重点看 UV hole fill / inpaint 是否把边缘颜色扩散进脸部。</p>
    <pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
  </div>
  {''.join(cards)}
  <section>
    <h2>UV 空洞修补 / inpaint 范围</h2>
    <figure><img src="04_uv_hole_fill_and_repair_roi.png"><figcaption>绿色=原始有效 UV；红色=为烘焙临时补出来的 UV 面；橙色=后续修补可能影响的 ROI。橙色范围越靠近下巴/脸周，越容易把边缘颜色扩散进去。</figcaption></figure>
  </section>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    out_dir = cfg.OUTPUT_DEBUG_DIR / "chin_root_cause_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        images, _ = undistort_images_with_calibration(images, calibration_path=cfg.CAMERA_CALIBRATION_PATH, alpha=cfg.UNDISTORT_ALPHA)

    square_images = {}
    phase3_masks = {}
    strict_masks = {}
    lines_by_view = {}
    mask_stats = {}

    for view_name, image in images.items():
        lmks, _ = detect_landmarks_mediapipe(image, view_name)
        if lmks is None:
            logger.warning("[%s] no landmarks, skipping mask audit", view_name)
            continue
        square_img, pad = _pad_square(image)
        phase3 = _phase3_grabcut_mask(image, lmks)
        phase3_sq, _ = _pad_square(phase3)
        lines = _chin_lines(lmks, pad, square_img.shape)
        strict_sq = phase3_sq.copy()
        strict_sq[lines["strict_cut_y"]:, :] = 0

        square_images[view_name] = square_img
        phase3_masks[view_name] = phase3_sq
        strict_masks[view_name] = strict_sq
        lines_by_view[view_name] = lines

        extra = (phase3_sq > 0) & (strict_sq == 0)
        below_chin = np.zeros_like(extra)
        below_chin[lines["chin_y"]:, :] = True
        below_strict = np.zeros_like(extra)
        below_strict[lines["strict_cut_y"]:, :] = True
        below_neck = np.zeros_like(extra)
        below_neck[lines["neck_cut_y"]:, :] = True
        mask_stats[view_name] = {
            **lines,
            "phase3_pixels": int((phase3_sq > 0).sum()),
            "chin_cut_pixels": int((strict_sq > 0).sum()),
            "phase3_pixels_below_chin": int(((phase3_sq > 0) & below_chin).sum()),
            "phase3_pixels_below_strict_cut": int(((phase3_sq > 0) & below_strict).sum()),
            "phase3_pixels_below_neck_cut": int(((phase3_sq > 0) & below_neck).sum()),
            "phase3_removed_by_chin_cut": int(extra.sum()),
        }

        _save_thumb(out_dir / f"{view_name}_01_phase3_mask.jpg", _overlay_mask(square_img, phase3_sq, lines, "Phase3 actual mask"))
        _save_thumb(out_dir / f"{view_name}_02_strict_mask.jpg", _overlay_mask(square_img, strict_sq, lines, "Phase3 + explicit chin cut"))
        _save_thumb(out_dir / f"{view_name}_02b_phase3_extra_vs_strict.jpg", _overlay_leak(square_img, phase3_sq, strict_sq, lines))
        _save_thumb(out_dir / f"{view_name}_05_mask_chin_closeup.jpg", _chin_crop(_overlay_leak(square_img, phase3_sq, strict_sq, lines), lines, phase3_sq))

    vertices, faces, uv_verts, uv_faces = load_mesh_obj(cfg.OUTPUT_MESH_DIR / "face_mesh.obj")
    cameras = load_cameras(cfg.OUTPUT_MESH_DIR / "cameras.json")
    scaled_cameras = {}
    for view_name, cam in cameras.items():
        if view_name not in square_images:
            continue
        side = square_images[view_name].shape[0]
        scale = side / float(cfg.WORK_IMAGE_SIZE)
        k = cam["K"].copy()
        k[0, :] *= scale
        k[1, :] *= scale
        scaled_cameras[view_name] = {"K": k, "R": cam["R"], "t": cam["t"]}
    kept_faces, mesh_stats = _mesh_boundary_audit(
        vertices, faces, scaled_cameras, square_images, phase3_masks, strict_masks, lines_by_view, out_dir
    )
    uv_stats = _uv_stage_audit(vertices, faces[kept_faces], uv_verts, uv_faces[kept_faces], out_dir)

    summary = {
        "dataset": str(cfg.DEFAULT_IMAGE_DIR),
        "views": cfg.DEFAULT_VIEW_NAMES,
        "mask_stats": mask_stats,
        "mesh_boundary_stats": mesh_stats,
        "uv_hole_fill_stats": uv_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir, summary)
    logger.info("Chin root cause audit saved: %s", out_dir / "index.html")


if __name__ == "__main__":
    main()

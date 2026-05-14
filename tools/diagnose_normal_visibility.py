"""Diagnose normal-based visibility rejection in texture baking.

This is diagnostic only. It mirrors the current phase3 texture sampling setup
and compares the active normal/view-direction convention against a few minimal
alternatives so we can tell direction errors from threshold/geometry issues.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.coordinates import (  # noqa: E402
    camera_center_for_texture_visibility,
    project_texture_points_to_image,
)
from src.module3_texture import (  # noqa: E402
    _render_camera_depth,
    add_uv_hole_fill_faces_for_bake,
    compute_face_normals,
    load_cameras,
    load_mesh_obj,
    rasterize_uv_map,
)
from tools.visualize_pipeline_audit import prepare_inputs, scale_cameras_for_padded_images  # noqa: E402


OUT_DIR = cfg.OUTPUT_DEBUG_DIR / "normal_visibility_diagnosis"
VIEW_ORDER = ("left", "front", "right")
TEX_SIZE = 2048
WEIGHT_THRESHOLD = 0.05
COSINE_THRESHOLD = float(np.sqrt(WEIGHT_THRESHOLD))


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def cosine_to_rgb(cosine: np.ndarray, valid_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:
    """Blue/purple for negative, black near zero, green/white for positive."""
    out = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    values = np.clip(cosine, -1.0, 1.0)
    neg = values < 0
    pos = ~neg
    rgb = np.zeros((len(values), 3), dtype=np.uint8)
    rgb[neg, 0] = np.clip((-values[neg]) * 120, 0, 255).astype(np.uint8)
    rgb[neg, 2] = np.clip((-values[neg]) * 255, 0, 255).astype(np.uint8)
    rgb[pos, 1] = np.clip(values[pos] * 255, 0, 255).astype(np.uint8)
    rgb[pos, 0] = np.clip(values[pos] * 120, 0, 255).astype(np.uint8)
    rgb[pos & (values >= COSINE_THRESHOLD), :] = np.maximum(
        rgb[pos & (values >= COSINE_THRESHOLD), :],
        np.array([210, 255, 210], dtype=np.uint8),
    )
    out[valid_y, valid_x] = rgb
    return out


def mask_rgb_outside(image: np.ndarray, selector: np.ndarray, valid_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:
    mask = np.zeros((TEX_SIZE, TEX_SIZE), dtype=bool)
    mask[valid_y[selector], valid_x[selector]] = True
    out = image.copy()
    out[~mask] = 0
    return out


def save_rgb_pair(paths: Dict[str, Path], label: str, image: np.ndarray, path: Path) -> None:
    save_rgb(path, image)
    paths[f"{label} raw UV"] = path
    flipped = path.with_name(path.stem + "_vflipped" + path.suffix)
    save_rgb(flipped, np.flipud(image))
    paths[f"{label} V-flipped"] = flipped


def make_contact_sheet(paths: Dict[str, Path], out_path: Path) -> None:
    panels = []
    for label, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (420, 420), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(image, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)
    if not panels:
        return
    cols = 3
    rows = (len(panels) + cols - 1) // cols
    sheet = np.full((rows * 420, cols * 420, 3), 245, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        sheet[r * 420 : (r + 1) * 420, c * 420 : (c + 1) * 420] = panel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def resize_max(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_normal_projection_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    proj: np.ndarray,
    in_img: np.ndarray,
    in_mask: np.ndarray,
    normal_ok: np.ndarray,
    out_path: Path,
    title: str,
    z_ok: Optional[np.ndarray] = None,
    max_points_per_class: int = 9000,
) -> None:
    overlay = image.copy()
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 230, 0), 8, cv2.LINE_AA)

    if z_ok is None:
        classes = [
            ("outside mask", in_img & ~in_mask, (255, 170, 40)),
            ("normal reject", in_mask & ~normal_ok, (255, 60, 210)),
            ("normal pass", in_mask & normal_ok, (90, 255, 120)),
        ]
        legend = [
            ("yellow contour: face mask", (255, 230, 0)),
            ("green: normal pass", (90, 255, 120)),
            ("pink: normal reject inside mask", (255, 60, 210)),
            ("orange: projected outside mask", (255, 170, 40)),
        ]
    else:
        classes = [
            ("outside mask", in_img & ~in_mask, (255, 170, 40)),
            ("occluded normal reject", in_mask & ~normal_ok & ~z_ok, (160, 70, 255)),
            ("visible normal reject", in_mask & ~normal_ok & z_ok, (255, 60, 210)),
            ("visible normal pass", in_mask & normal_ok & z_ok, (90, 255, 120)),
        ]
        legend = [
            ("yellow contour: face mask", (255, 230, 0)),
            ("green: visible + normal pass", (90, 255, 120)),
            ("pink: visible but normal reject", (255, 60, 210)),
            ("purple: occluded normal reject", (160, 70, 255)),
            ("orange: projected outside mask", (255, 170, 40)),
        ]
    for _, selector, color in classes:
        idx = np.where(selector)[0]
        if len(idx) == 0:
            continue
        step = max(1, len(idx) // max_points_per_class)
        for i in idx[::step]:
            x, y = int(proj[i, 0]), int(proj[i, 1])
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)

    cv2.rectangle(overlay, (0, 0), (880, 76 + 32 * len(legend)), (0, 0, 0), -1)
    cv2.putText(overlay, title, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, (text, color) in enumerate(legend):
        y = 68 + idx * 32
        cv2.circle(overlay, (24, y - 8), 8, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, text, (46, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    save_rgb(out_path, resize_max(overlay))


def sparse_projection_mask(
    vertices: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    image_shape: tuple[int, int],
    flip_y_for_projection: bool,
) -> np.ndarray:
    h, w = image_shape
    verts = vertices.copy()
    if flip_y_for_projection:
        verts[:, 1] *= -1
    v_cam = (r @ verts.T + t[:, None]).T
    z = v_cam[:, 2]
    valid = z > 1e-4
    proj = np.zeros((len(vertices), 2), dtype=np.float32)
    proj[valid, 0] = k[0, 0] * v_cam[valid, 0] / z[valid] + k[0, 2]
    proj[valid, 1] = k[1, 1] * v_cam[valid, 1] / z[valid] + k[1, 2]
    in_img = valid & (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    out = np.zeros((h, w), dtype=np.uint8)
    px = proj[in_img, 0].astype(np.int32)
    py = proj[in_img, 1].astype(np.int32)
    out[py, px] = 255
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)), iterations=1) > 0


def project_texels_with_convention(
    pts_3d: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    image_shape: tuple[int, int],
    flip_y_for_projection: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    pts = pts_3d.copy()
    if flip_y_for_projection:
        pts[:, 1] *= -1
    v_cam = (r @ pts.T + t[:, None]).T
    z = v_cam[:, 2]
    front = z > 1e-4
    proj = np.zeros((len(pts_3d), 2), dtype=np.float32)
    proj[front, 0] = k[0, 0] * v_cam[front, 0] / z[front] + k[0, 2]
    proj[front, 1] = k[1, 1] * v_cam[front, 1] / z[front] + k[1, 2]
    in_img = front & (proj[:, 0] >= 0) & (proj[:, 0] < w - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h - 1)
    return proj, in_img


def silhouette_iou(mask: np.ndarray, silhouette: np.ndarray) -> Dict[str, float | int]:
    mask_bool = mask > 127
    sil_bool = silhouette.astype(bool)
    intersection = int((mask_bool & sil_bool).sum())
    union = int((mask_bool | sil_bool).sum())
    return {
        "iou": float(intersection / max(union, 1)),
        "intersection_pixels": intersection,
        "mask_pixels": int(mask_bool.sum()),
        "silhouette_pixels": int(sil_bool.sum()),
        "mask_extra_pixels_outside_silhouette": int((mask_bool & ~sil_bool).sum()),
        "silhouette_pixels_missing_from_mask": int((sil_bool & ~mask_bool).sum()),
    }


def summarize_cosines(cosine: np.ndarray, selector: np.ndarray) -> Dict[str, float | int]:
    if not np.any(selector):
        return {
            "texels": 0,
            "cos_gt_0_percent": 0.0,
            "cos_gt_threshold_percent": 0.0,
            "mean": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
        }
    values = cosine[selector]
    return {
        "texels": int(values.size),
        "cos_gt_0_percent": float((values > 0.0).mean() * 100.0),
        "cos_gt_threshold_percent": float((values > COSINE_THRESHOLD).mean() * 100.0),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, padded, padded_masks = prepare_inputs()
    cameras = scale_cameras_for_padded_images(load_cameras(cfg.OUTPUT_MESH_DIR / "cameras.json"), padded)
    vertices, faces, uv_verts, uv_faces = load_mesh_obj(cfg.OUTPUT_MESH_DIR / "face_mesh.obj")

    enable_fill = bool(getattr(cfg, "ENABLE_UV_HOLE_FILL_FACES", True))
    if enable_fill:
        bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces, added_faces = add_uv_hole_fill_faces_for_bake(
            vertices, faces, uv_verts, uv_faces, TEX_SIZE
        )
    else:
        bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces, added_faces = vertices, faces, uv_verts, uv_faces, 0

    tri_map, bary_map = rasterize_uv_map(bake_uv_verts, bake_uv_faces, TEX_SIZE)
    valid_mask = tri_map >= 0
    valid_y, valid_x = np.where(valid_mask)
    valid_tri = tri_map[valid_y, valid_x]
    valid_bary = bary_map[valid_y, valid_x]
    g_faces = bake_faces[valid_tri]
    pts_3d = (
        valid_bary[:, 0:1] * bake_vertices[g_faces[:, 0]]
        + valid_bary[:, 1:2] * bake_vertices[g_faces[:, 1]]
        + valid_bary[:, 2:3] * bake_vertices[g_faces[:, 2]]
    )
    pt_normals = compute_face_normals(bake_vertices, bake_faces)[valid_tri]

    contact_paths: Dict[str, Path] = {}
    view_states: Dict[str, Dict[str, np.ndarray]] = {}
    summary: Dict[str, object] = {
        "tex_size": TEX_SIZE,
        "weight_threshold": WEIGHT_THRESHOLD,
        "cosine_threshold_equivalent": COSINE_THRESHOLD,
        "uv_hole_fill_faces_enabled": enable_fill,
        "uv_hole_fill_faces_added": int(added_faces),
        "valid_texels": int(len(valid_y)),
        "note": "normal_ok in the production pipeline is max(0, normal dot view_dir)^2 > 0.05, equivalent to cosine > 0.2236.",
        "views": {},
    }

    for view_name in VIEW_ORDER:
        cam = cameras[view_name]
        image = padded[view_name]
        mask = padded_masks[view_name]
        h_img, w_img = image.shape[:2]
        k, r, t = cam["K"], cam["R"], cam["t"]

        v_cam, z, proj, front = project_texture_points_to_image(pts_3d, k, r, t)
        in_img = front & (proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1)
        px_u = np.clip(proj[:, 0].astype(int), 0, w_img - 1)
        px_v = np.clip(proj[:, 1].astype(int), 0, h_img - 1)
        in_mask = in_img & (mask[px_v, px_u] > 127)

        proj_no_y_flip, in_img_no_y_flip = project_texels_with_convention(
            pts_3d, k, r, t, (h_img, w_img), flip_y_for_projection=False
        )
        px_u_no_y = np.clip(proj_no_y_flip[:, 0].astype(int), 0, w_img - 1)
        px_v_no_y = np.clip(proj_no_y_flip[:, 1].astype(int), 0, h_img - 1)
        in_mask_no_y_flip = in_img_no_y_flip & (mask[px_v_no_y, px_u_no_y] > 127)

        sparse_y_flip = sparse_projection_mask(bake_vertices, k, r, t, (h_img, w_img), True)
        sparse_no_y_flip = sparse_projection_mask(bake_vertices, k, r, t, (h_img, w_img), False)

        cam_center_current = camera_center_for_texture_visibility(r, t)
        cam_center_proj = -r.T @ t

        view_dirs_current = cam_center_current - pts_3d
        view_dirs_current /= np.clip(np.linalg.norm(view_dirs_current, axis=1, keepdims=True), 1e-8, None)
        cosine_current = np.sum(pt_normals * view_dirs_current, axis=1)
        current_normal_ok = cosine_current > COSINE_THRESHOLD
        depth_map = _render_camera_depth(bake_vertices, bake_faces, k, r, t, (h_img, w_img))
        z_ref = depth_map[px_v, px_u]
        z_ok = z <= (z_ref + 3e-3)

        view_dirs_no_y_flip = cam_center_proj - pts_3d
        view_dirs_no_y_flip /= np.clip(np.linalg.norm(view_dirs_no_y_flip, axis=1, keepdims=True), 1e-8, None)
        cosine_no_y_flip = np.sum(pt_normals * view_dirs_no_y_flip, axis=1)
        cosine_inverted_normals = -cosine_current

        cosine_img = cosine_to_rgb(cosine_current, valid_y, valid_x)
        save_rgb_pair(contact_paths, f"{view_name} cosine", cosine_img, OUT_DIR / f"{view_name}_cosine_uv.png")
        save_rgb_pair(
            contact_paths,
            f"{view_name} in-mask cosine",
            mask_rgb_outside(cosine_img, in_mask, valid_y, valid_x),
            OUT_DIR / f"{view_name}_cosine_in_mask_uv.png",
        )
        overlay_path = OUT_DIR / f"{view_name}_normal_projection_overlay_current_projection.jpg"
        draw_normal_projection_overlay(
            image,
            mask,
            proj,
            in_img,
            in_mask,
            current_normal_ok,
            overlay_path,
            f"{view_name}: current texture projection",
            z_ok=z_ok,
        )
        contact_paths[f"{view_name} current overlay"] = overlay_path

        summary["views"][view_name] = {
            "in_image_texels": int(in_img.sum()),
            "in_mask_texels": int(in_mask.sum()),
            "current_normals_in_mask": summarize_cosines(cosine_current, in_mask),
            "inverted_normals_in_mask": summarize_cosines(cosine_inverted_normals, in_mask),
            "no_camera_y_flip_in_mask": summarize_cosines(cosine_no_y_flip, in_mask),
            "current_normals_all_projected": summarize_cosines(cosine_current, in_img),
            "normal_ok_texels_current_threshold": int((in_mask & current_normal_ok).sum()),
            "normal_ok_texels_if_cos_gt_0": int((in_mask & (cosine_current > 0.0)).sum()),
            "normal_ok_texels_if_inverted": int((in_mask & (cosine_inverted_normals > COSINE_THRESHOLD)).sum()),
            "normal_ok_texels_if_no_camera_y_flip": int((in_mask & (cosine_no_y_flip > COSINE_THRESHOLD)).sum()),
            "normal_reject_texels": int((in_mask & ~current_normal_ok).sum()),
            "normal_reject_visible_zbuffer_ok_texels": int((in_mask & ~current_normal_ok & z_ok).sum()),
            "normal_reject_occluded_zbuffer_fail_texels": int((in_mask & ~current_normal_ok & ~z_ok).sum()),
            "normal_reject_visible_percent": float(
                (in_mask & ~current_normal_ok & z_ok).sum() * 100.0 /
                max(int((in_mask & ~current_normal_ok).sum()), 1)
            ),
            "zbuffer_ok_without_normal_filter_texels": int((in_mask & z_ok).sum()),
            "zbuffer_ok_but_normal_reject_texels": int((in_mask & z_ok & ~current_normal_ok).sum()),
            "texel_projection_in_mask_with_y_flip": int(in_mask.sum()),
            "texel_projection_in_mask_without_y_flip": int(in_mask_no_y_flip.sum()),
            "sparse_projection_iou_with_y_flip": silhouette_iou(mask, sparse_y_flip),
            "sparse_projection_iou_without_y_flip": silhouette_iou(mask, sparse_no_y_flip),
        }
        view_states[view_name] = {
            "in_mask": in_mask,
            "in_mask_no_y_flip_projection": in_mask_no_y_flip,
            "cosine_current": cosine_current,
            "cosine_no_camera_y_flip": cosine_no_y_flip,
            "cosine_inverted_normals": cosine_inverted_normals,
        }

        no_y_normal_ok = cosine_no_y_flip > COSINE_THRESHOLD
        no_y_overlay_path = OUT_DIR / f"{view_name}_normal_projection_overlay_no_y_projection.jpg"
        draw_normal_projection_overlay(
            image,
            mask,
            proj_no_y_flip,
            in_img_no_y_flip,
            in_mask_no_y_flip,
            no_y_normal_ok,
            no_y_overlay_path,
            f"{view_name}: no Y flip projection",
        )
        contact_paths[f"{view_name} no-y overlay"] = no_y_overlay_path

    any_in_mask = np.zeros(len(valid_y), dtype=bool)
    any_current_normal_ok = np.zeros(len(valid_y), dtype=bool)
    any_current_cos_positive = np.zeros(len(valid_y), dtype=bool)
    any_no_camera_y_flip_normal_ok = np.zeros(len(valid_y), dtype=bool)
    any_inverted_normal_ok = np.zeros(len(valid_y), dtype=bool)
    any_no_y_projection_and_normal_ok = np.zeros(len(valid_y), dtype=bool)
    for state in view_states.values():
        in_mask = state["in_mask"]
        in_mask_no_y = state["in_mask_no_y_flip_projection"]
        any_in_mask |= in_mask
        any_current_normal_ok |= in_mask & (state["cosine_current"] > COSINE_THRESHOLD)
        any_current_cos_positive |= in_mask & (state["cosine_current"] > 0.0)
        any_no_camera_y_flip_normal_ok |= in_mask & (state["cosine_no_camera_y_flip"] > COSINE_THRESHOLD)
        any_inverted_normal_ok |= in_mask & (state["cosine_inverted_normals"] > COSINE_THRESHOLD)
        any_no_y_projection_and_normal_ok |= in_mask_no_y & (state["cosine_no_camera_y_flip"] > COSINE_THRESHOLD)

    summary["combined"] = {
        "face_candidate_roi_texels_current_projection": int(any_in_mask.sum()),
        "current_normal_ok_texels": int(any_current_normal_ok.sum()),
        "current_normal_reject_texels": int((any_in_mask & ~any_current_normal_ok).sum()),
        "loose_cos_gt_0_normal_ok_texels": int(any_current_cos_positive.sum()),
        "loose_cos_gt_0_normal_reject_texels": int((any_in_mask & ~any_current_cos_positive).sum()),
        "no_camera_y_flip_normal_ok_texels": int(any_no_camera_y_flip_normal_ok.sum()),
        "no_camera_y_flip_normal_reject_texels": int((any_in_mask & ~any_no_camera_y_flip_normal_ok).sum()),
        "inverted_normal_ok_texels": int(any_inverted_normal_ok.sum()),
        "inverted_normal_reject_texels": int((any_in_mask & ~any_inverted_normal_ok).sum()),
        "no_y_projection_and_no_camera_y_flip_normal_ok_texels": int(any_no_y_projection_and_normal_ok.sum()),
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(contact_paths, OUT_DIR / "_contact_sheet.jpg")
    print(json.dumps(summary, indent=2))
    print(str((OUT_DIR / "_contact_sheet.jpg").resolve()))


if __name__ == "__main__":
    main()

"""Quantify why UV texels are rejected during multi-view texture sampling.

Diagnostic only. It mirrors the current phase3 camera scaling and visibility
tests, then writes JSON summaries and UV overlays under
output/debug/texture_rejection_diagnosis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

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
    _small_internal_invalid_uv_holes,
    _view_region_weight,
    add_uv_hole_fill_faces_for_bake,
    compute_face_normals,
    load_cameras,
    load_mesh_obj,
    rasterize_uv_map,
)
from tools.visualize_pipeline_audit import prepare_inputs, scale_cameras_for_padded_images  # noqa: E402


OUT_DIR = cfg.OUTPUT_DEBUG_DIR / "texture_rejection_diagnosis"
VIEW_ORDER = ("left", "front", "right")
TEX_SIZE = 2048

REASON_COLORS = {
    "sampled": (70, 220, 110),
    "uv_invalid": (255, 40, 40),
    "behind_camera": (110, 110, 110),
    "out_of_image": (40, 130, 255),
    "mask_reject": (255, 180, 60),
    "normal_reject": (255, 70, 210),
    "zbuffer_reject": (150, 70, 255),
    "blocked_by_view_rule": (80, 220, 220),
}


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_rgb_pair(
    paths: Dict[str, Path],
    label: str,
    image: np.ndarray,
    raw_path: Path,
    include_flipped: bool = True,
) -> None:
    save_rgb(raw_path, image)
    paths[f"{label} raw UV"] = raw_path
    if include_flipped:
        flipped_path = raw_path.with_name(raw_path.stem + "_vflipped" + raw_path.suffix)
        save_rgb(flipped_path, np.flipud(image))
        paths[f"{label} V-flipped"] = flipped_path


def save_gray_pair(
    paths: Dict[str, Path],
    label: str,
    image: np.ndarray,
    raw_path: Path,
    include_flipped: bool = True,
) -> None:
    save_gray(raw_path, image)
    paths[f"{label} raw UV"] = raw_path
    if include_flipped:
        flipped_path = raw_path.with_name(raw_path.stem + "_vflipped" + raw_path.suffix)
        save_gray(flipped_path, np.flipud(image))
        paths[f"{label} V-flipped"] = flipped_path


def atlas_from_valid(valid_y: np.ndarray, valid_x: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8)
    out[valid_y, valid_x] = values
    return out


def make_contact_sheet(paths: Dict[str, Path], out_path: Path) -> None:
    panels = []
    for label, path in paths.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (420, 420), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(img)
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


def draw_projected_reason_overlay(
    image: np.ndarray,
    face_mask: np.ndarray,
    mesh_silhouette: np.ndarray,
    proj: np.ndarray,
    reason: np.ndarray,
    reason_code: Dict[str, int],
    out_path: Path,
    max_points_per_reason: int = 8000,
) -> None:
    overlay = image.copy()
    mask_contours, _ = cv2.findContours((face_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mesh_contours, _ = cv2.findContours(mesh_silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, mask_contours, -1, (0, 255, 255), 8, cv2.LINE_AA)
    cv2.drawContours(overlay, mesh_contours, -1, (255, 255, 0), 6, cv2.LINE_AA)

    for key in ("sampled", "mask_reject", "normal_reject", "zbuffer_reject"):
        idx = np.where(reason == reason_code[key])[0]
        if len(idx) == 0:
            continue
        step = max(1, len(idx) // max_points_per_reason)
        color = REASON_COLORS[key]
        for i in idx[::step]:
            x, y = int(proj[i, 0]), int(proj[i, 1])
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)

    legend = [
        ("cyan contour: face mask", (0, 255, 255)),
        ("yellow contour: mesh silhouette", (255, 255, 0)),
        ("green: sampled", REASON_COLORS["sampled"]),
        ("orange: mask reject", REASON_COLORS["mask_reject"]),
        ("pink: normal reject", REASON_COLORS["normal_reject"]),
        ("purple: zbuffer reject", REASON_COLORS["zbuffer_reject"]),
    ]
    cv2.rectangle(overlay, (0, 0), (920, 46 + 34 * len(legend)), (0, 0, 0), -1)
    for idx, (text, color) in enumerate(legend):
        y = 34 + idx * 34
        cv2.circle(overlay, (24, y - 8), 9, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, text, (46, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    save_rgb(out_path, resize_max(overlay))


def colorize_reason(reason: np.ndarray, valid_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:
    palette_keys = [
        "sampled",
        "behind_camera",
        "out_of_image",
        "mask_reject",
        "normal_reject",
        "zbuffer_reject",
        "blocked_by_view_rule",
    ]
    out = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    for idx, key in enumerate(palette_keys):
        m = reason == idx
        out[valid_y[m], valid_x[m]] = REASON_COLORS[key]
    return out


def colorize_combined(combined: np.ndarray, valid_y: np.ndarray, valid_x: np.ndarray) -> np.ndarray:
    color_map = {
        "sampled": (70, 220, 110),
        "no_view_projects_inside_image": (40, 130, 255),
        "face_mask_rejects_all_candidate_views": (255, 180, 60),
        "normal_threshold_rejects_all_candidate_views": (255, 70, 210),
        "zbuffer_rejects_all_candidate_views": (150, 70, 255),
        "view_rule_or_zero_weight_blocks_remaining_candidates": (80, 220, 220),
    }
    out = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    for key, color in color_map.items():
        m = combined == key
        out[valid_y[m], valid_x[m]] = color
    return out


def mask_rgb_outside(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[~mask] = 0
    return out


def count_keys(values: np.ndarray, keys: list[str], selector: np.ndarray | None = None) -> Dict[str, int]:
    if selector is None:
        selector = np.ones(len(values), dtype=bool)
    return {key: int((selector & (values == key)).sum()) for key in keys}


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
    original_tri_map, _ = rasterize_uv_map(uv_verts, uv_faces, TEX_SIZE)
    original_valid_mask = original_tri_map >= 0
    internal_holes_original = _small_internal_invalid_uv_holes(original_valid_mask)

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

    x = pts_3d[:, 0].astype(np.float32)
    x_extent = max(float(np.max(np.abs(x))), 1e-6)
    center_face = np.abs(x / x_extent) < 0.35
    filled_uv_holes = (tri_map >= len(uv_faces)) if added_faces else _small_internal_invalid_uv_holes(valid_mask)
    uv_hole_repair_roi = cv2.dilate(
        filled_uv_holes.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=4,
    ).astype(bool) & valid_mask
    force_front = uv_hole_repair_roi[valid_y, valid_x]
    uv_hole_neighborhood_img = cv2.dilate(
        _small_internal_invalid_uv_holes(valid_mask).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=16,
    ).astype(bool) & valid_mask
    uv_hole_neighborhood = uv_hole_neighborhood_img[valid_y, valid_x]

    reason_code = {
        "sampled": 0,
        "behind_camera": 1,
        "out_of_image": 2,
        "mask_reject": 3,
        "normal_reject": 4,
        "zbuffer_reject": 5,
        "blocked_by_view_rule": 6,
    }

    accepted_by_view: Dict[str, np.ndarray] = {}
    reason_by_view: Dict[str, np.ndarray] = {}
    masks_by_view: Dict[str, Dict[str, np.ndarray]] = {}
    view_summary = {}
    contact_paths: Dict[str, Path] = {}

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
        in_mask = mask[px_v, px_u] > 127

        cam_center = camera_center_for_texture_visibility(r, t)
        view_dirs = cam_center - pts_3d
        view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
        cosines = np.sum(pt_normals * view_dirs, axis=1)
        weights = np.maximum(0.0, cosines) ** 2
        normal_ok = weights > 0.05

        depth_map = _render_camera_depth(bake_vertices, bake_faces, k, r, t, (h_img, w_img))
        mesh_silhouette = np.isfinite(depth_map)
        z_ref = depth_map[px_v, px_u]
        z_ok = z <= (z_ref + 3e-3)

        valid_pts = in_img & normal_ok & in_mask
        if view_name == "front":
            center_fallback = (uv_hole_neighborhood | force_front) & center_face & in_img & (np.abs(cosines) > 0.05)
            center_fallback &= in_mask
            accepted = (valid_pts & z_ok) | center_fallback
        else:
            accepted = valid_pts & z_ok & ~force_front

        reason = np.full(len(valid_y), reason_code["sampled"], dtype=np.uint8)
        reason[~front] = reason_code["behind_camera"]
        reason[front & ~in_img] = reason_code["out_of_image"]
        reason[in_img & ~in_mask] = reason_code["mask_reject"]
        reason[in_img & in_mask & ~normal_ok] = reason_code["normal_reject"]
        reason[in_img & in_mask & normal_ok & ~z_ok] = reason_code["zbuffer_reject"]
        if view_name != "front":
            reason[in_img & in_mask & normal_ok & z_ok & force_front] = reason_code["blocked_by_view_rule"]
        reason[accepted] = reason_code["sampled"]

        accepted_by_view[view_name] = accepted
        reason_by_view[view_name] = reason
        masks_by_view[view_name] = {
            "in_img": in_img,
            "in_mask": in_img & in_mask,
            "normal_ok": in_img & in_mask & normal_ok,
            "z_ok": in_img & in_mask & normal_ok & z_ok,
            "accepted": accepted,
            "force_front": force_front,
        }

        counts = {name: int((reason == code).sum()) for name, code in reason_code.items()}
        mask_bool = mask > 127
        sil_bool = mesh_silhouette
        mask_mesh_intersection = int((mask_bool & sil_bool).sum())
        mask_mesh_union = int((mask_bool | sil_bool).sum())
        view_summary[view_name] = {
            "reason_counts_all_valid_texels": counts,
            "accepted_texels": int(accepted.sum()),
            "accepted_percent_of_valid": float(accepted.sum() * 100.0 / max(len(valid_y), 1)),
            "in_image_texels": int(in_img.sum()),
            "in_mask_texels": int((in_img & in_mask).sum()),
            "normal_ok_texels": int((in_img & in_mask & normal_ok).sum()),
            "zbuffer_ok_texels": int((in_img & in_mask & normal_ok & z_ok).sum()),
            "face_mask_pixels": int(mask_bool.sum()),
            "mesh_silhouette_pixels": int(sil_bool.sum()),
            "face_mask_mesh_silhouette_iou": float(mask_mesh_intersection / max(mask_mesh_union, 1)),
            "face_mask_extra_pixels_outside_mesh": int((mask_bool & ~sil_bool).sum()),
            "mesh_pixels_missing_from_face_mask": int((sil_bool & ~mask_bool).sum()),
        }

        reason_img = colorize_reason(reason, valid_y, valid_x)
        save_rgb_pair(contact_paths, f"{view_name} reject", reason_img, OUT_DIR / f"{view_name}_reject_reason_uv.png")

        overlay_path = OUT_DIR / f"{view_name}_projection_reason_overlay.jpg"
        draw_projected_reason_overlay(image, mask, mesh_silhouette, proj, reason, reason_code, overlay_path)
        contact_paths[f"{view_name} on image"] = overlay_path

    accepted_any = np.zeros(len(valid_y), dtype=bool)
    for accepted in accepted_by_view.values():
        accepted_any |= accepted
    unsampled = ~accepted_any

    # Combined root-cause classification for texels that remain unsampled.
    any_in_img = np.zeros(len(valid_y), dtype=bool)
    any_in_mask = np.zeros(len(valid_y), dtype=bool)
    any_normal_ok = np.zeros(len(valid_y), dtype=bool)
    any_z_ok = np.zeros(len(valid_y), dtype=bool)
    for view_name in VIEW_ORDER:
        m = masks_by_view[view_name]
        any_in_img |= m["in_img"]
        any_in_mask |= m["in_mask"]
        any_normal_ok |= m["normal_ok"]
        any_z_ok |= m["z_ok"]

    face_candidate_roi = any_in_mask
    face_candidate_roi_img = np.zeros((TEX_SIZE, TEX_SIZE), dtype=bool)
    face_candidate_roi_img[valid_y[face_candidate_roi], valid_x[face_candidate_roi]] = True

    combined = np.full(len(valid_y), "sampled", dtype=object)
    combined[unsampled & ~any_in_img] = "no_view_projects_inside_image"
    combined[unsampled & any_in_img & ~any_in_mask] = "face_mask_rejects_all_candidate_views"
    combined[unsampled & any_in_mask & ~any_normal_ok] = "normal_threshold_rejects_all_candidate_views"
    combined[unsampled & any_normal_ok & ~any_z_ok] = "zbuffer_rejects_all_candidate_views"
    combined[unsampled & any_z_ok] = "view_rule_or_zero_weight_blocks_remaining_candidates"

    combined_counts = {}
    combined_keys = [
        "sampled",
        "no_view_projects_inside_image",
        "face_mask_rejects_all_candidate_views",
        "normal_threshold_rejects_all_candidate_views",
        "zbuffer_rejects_all_candidate_views",
        "view_rule_or_zero_weight_blocks_remaining_candidates",
    ]
    for key in combined_keys:
        combined_counts[key] = int((combined == key).sum())

    center_unsampled_selector = center_face & unsampled
    hole_roi_selector = force_front & unsampled
    face_candidate_unsampled_selector = face_candidate_roi & unsampled

    unsampled_atlas = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8)
    unsampled_atlas[valid_y[unsampled], valid_x[unsampled]] = 255
    save_gray_pair(contact_paths, "combined holes", unsampled_atlas, OUT_DIR / "combined_unsampled_holes.png")

    combined_reason_path = OUT_DIR / "combined_unsampled_root_cause_uv.png"
    combined_reason_img = colorize_combined(combined, valid_y, valid_x)
    save_rgb_pair(contact_paths, "combined reasons", combined_reason_img, combined_reason_path)

    roi_reason_path = OUT_DIR / "combined_unsampled_root_cause_face_candidate_roi.png"
    save_rgb_pair(
        contact_paths,
        "face ROI reasons",
        mask_rgb_outside(combined_reason_img, face_candidate_roi_img),
        roi_reason_path,
    )

    view_count = np.zeros(len(valid_y), dtype=np.uint8)
    for accepted in accepted_by_view.values():
        view_count += accepted.astype(np.uint8)
    save_gray_pair(
        contact_paths,
        "sample count",
        atlas_from_valid(valid_y, valid_x, np.clip(view_count * 85, 0, 255)),
        OUT_DIR / "combined_sample_count.png",
    )

    hole_img = np.zeros((TEX_SIZE, TEX_SIZE), dtype=np.uint8)
    hole_img[internal_holes_original] = 255
    save_gray_pair(contact_paths, "original UV holes", hole_img, OUT_DIR / "original_internal_uv_holes.png")

    summary = {
        "tex_size": TEX_SIZE,
        "display_note": "raw UV images use the exact texture/GLB coordinate convention; *_vflipped files are visual aids only and are not used by the pipeline.",
        "uv_hole_fill_faces_enabled": enable_fill,
        "uv_hole_fill_faces_added": int(added_faces),
        "valid_texels_current_bake": int(valid_mask.sum()),
        "valid_texels_original_uv": int(original_valid_mask.sum()),
        "original_internal_uv_hole_texels": int(internal_holes_original.sum()),
        "sampled_by_at_least_one_view": int(accepted_any.sum()),
        "unsampled_valid_texels": int(unsampled.sum()),
        "unsampled_percent_of_valid": float(unsampled.sum() * 100.0 / max(len(valid_y), 1)),
        "combined_unsampled_root_cause_counts": combined_counts,
        "combined_unsampled_root_cause_counts_center_face": count_keys(
            combined,
            combined_keys,
            center_unsampled_selector,
        ),
        "combined_unsampled_root_cause_counts_uv_hole_repair_roi": count_keys(
            combined,
            combined_keys,
            hole_roi_selector,
        ),
        "combined_unsampled_root_cause_counts_face_candidate_roi": count_keys(
            combined,
            combined_keys,
            face_candidate_unsampled_selector,
        ),
        "face_candidate_roi_texels": int(face_candidate_roi.sum()),
        "face_candidate_roi_unsampled_texels": int(face_candidate_unsampled_selector.sum()),
        "face_candidate_roi_unsampled_percent": float(
            face_candidate_unsampled_selector.sum() * 100.0 / max(int(face_candidate_roi.sum()), 1)
        ),
        "views": view_summary,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheet(contact_paths, OUT_DIR / "_contact_sheet.jpg")
    print(json.dumps(summary, indent=2))
    print(str((OUT_DIR / "_contact_sheet.jpg").resolve()))


if __name__ == "__main__":
    main()

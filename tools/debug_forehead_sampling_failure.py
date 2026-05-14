"""Diagnose why the forehead texture hole failed to sample originally.

This is diagnostic-only. It replays the strict visibility rules used before the
local forehead fallback and internal-UV-hole inpaint, then writes focused
visuals under output/debug/forehead_sampling_failure.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.module3_texture import (
    _bary_batch,
    _render_camera_depth,
    compute_face_normals,
    load_cameras,
    load_mesh_obj,
    rasterize_uv_map,
)
from tools.visualize_pipeline_audit import prepare_inputs, scale_cameras_for_padded_images


TEX_SIZE = 2048
OUT_DIR = Path("output/debug/forehead_sampling_failure")

REASON_COLORS = {
    "uv_invalid": (255, 0, 0),
    "out_of_image": (0, 120, 255),
    "mask_reject": (255, 170, 0),
    "normal_reject": (255, 0, 255),
    "zbuffer_reject": (120, 0, 255),
    "sample_ok": (0, 220, 0),
}


def _focus_bbox() -> Tuple[int, int, int, int]:
    summary_path = Path("output/debug/pipeline_audit/03_texture_visibility/focus_forehead/focus_summary.json")
    if summary_path.exists():
        focus = json.loads(summary_path.read_text(encoding="utf-8"))
        x, y, w, h = focus["focus_bbox_1024"]
        return x * 2, y * 2, w * 2, h * 2
    return 976, 796, 86, 98


def _make_contact_sheet(items: Dict[str, Path], out_path: Path) -> None:
    tiles = []
    for title, path in items.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(image)
    if tiles:
        cv2.imwrite(str(out_path), np.concatenate(tiles, axis=1))


def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int], pad: int = 120) -> np.ndarray:
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    cv2.rectangle(crop, (x - x0, y - y0), (x + w - 1 - x0, y + h - 1 - y0), (0, 255, 255), 2)
    return crop


def _raster_invalid_components(valid_mask: np.ndarray) -> Tuple[np.ndarray, list]:
    invalid = (~valid_mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
    components = []
    h, w = valid_mask.shape
    for label in range(1, num_labels):
        x, y, cw, ch, area = stats[label]
        touches_border = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        components.append(
            {
                "label": int(label),
                "area": int(area),
                "bbox": [int(x), int(y), int(cw), int(ch)],
                "touches_border": bool(touches_border),
            }
        )
    return labels, components


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, padded, masks = prepare_inputs()
    cameras = scale_cameras_for_padded_images(load_cameras(cfg.OUTPUT_MESH_DIR / "cameras.json"), padded)
    vertices, faces, uv_verts, uv_faces = load_mesh_obj(cfg.OUTPUT_MESH_DIR / "face_mesh.obj")
    tri_map, bary_map = rasterize_uv_map(uv_verts, uv_faces, tex_size=TEX_SIZE)
    valid_mask = tri_map >= 0
    valid_y, valid_x = np.where(valid_mask)
    valid_tri = tri_map[valid_y, valid_x]
    valid_bary = bary_map[valid_y, valid_x]
    g_faces = faces[valid_tri]
    pts_3d = (
        valid_bary[:, 0:1] * vertices[g_faces[:, 0]]
        + valid_bary[:, 1:2] * vertices[g_faces[:, 1]]
        + valid_bary[:, 2:3] * vertices[g_faces[:, 2]]
    )
    pt_normals = compute_face_normals(vertices, faces)[valid_tri]

    bbox = _focus_bbox()
    x, y, w, h = bbox
    bbox_mask = np.zeros((TEX_SIZE, TEX_SIZE), dtype=bool)
    bbox_mask[y : y + h, x : x + w] = True
    in_focus_valid = bbox_mask[valid_y, valid_x]

    labels, components = _raster_invalid_components(valid_mask)
    focus_invalid_labels, counts = np.unique(labels[bbox_mask & ~valid_mask], return_counts=True)
    focus_components = []
    by_label = {c["label"]: c for c in components}
    for label, count in zip(focus_invalid_labels.tolist(), counts.tolist()):
        if label == 0:
            continue
        item = dict(by_label[int(label)])
        item["overlap_focus"] = int(count)
        focus_components.append(item)

    uv_valid_vis = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    uv_valid_vis[valid_mask] = (170, 170, 170)
    uv_valid_vis[~valid_mask] = REASON_COLORS["uv_invalid"]
    cv2.imwrite(str(OUT_DIR / "01_uv_validity_focus.png"), _crop(uv_valid_vis, bbox))

    summary = {
        "focus_bbox_2048": list(map(int, bbox)),
        "focus_pixels": int(w * h),
        "uv_valid_pixels": int((bbox_mask & valid_mask).sum()),
        "uv_invalid_pixels": int((bbox_mask & ~valid_mask).sum()),
        "invalid_components_overlapping_focus": focus_components,
        "views": {},
    }

    overlay_paths = {"uv validity": OUT_DIR / "01_uv_validity_focus.png"}

    for view_name, cam in cameras.items():
        K, R, t = cam["K"], cam["R"], cam["t"]
        image = padded[view_name]
        h_img, w_img = image.shape[:2]

        pts_proj = pts_3d.copy()
        pts_proj[:, 1] *= -1
        v_cam = (R @ pts_proj.T + t[:, None]).T
        z = v_cam[:, 2]
        front = z > 1e-4
        proj = np.zeros((len(valid_y), 2), dtype=np.float32)
        proj[front, 0] = K[0, 0] * v_cam[front, 0] / z[front] + K[0, 2]
        proj[front, 1] = K[1, 1] * v_cam[front, 1] / z[front] + K[1, 2]
        in_img = front & (proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1)

        cam_center_proj = -R.T @ t
        cam_center = cam_center_proj.copy()
        cam_center[1] *= -1
        view_dirs = cam_center - pts_3d
        view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
        cosines = np.sum(pt_normals * view_dirs, axis=1)
        normal_ok = np.maximum(0, cosines) ** 2 > 0.05

        px_u = np.clip(proj[:, 0].astype(int), 0, w_img - 1)
        px_v = np.clip(proj[:, 1].astype(int), 0, h_img - 1)
        in_mask = masks[view_name][px_v, px_u] > 127
        depth = _render_camera_depth(vertices, faces, K, R, t, (h_img, w_img))
        z_ref = depth[px_v, px_u]
        visible = z <= (z_ref + 3e-3)

        reason = np.full(len(valid_y), "sample_ok", dtype=object)
        reason[~in_img] = "out_of_image"
        reason[in_img & ~in_mask] = "mask_reject"
        reason[in_img & in_mask & ~normal_ok] = "normal_reject"
        reason[in_img & in_mask & normal_ok & ~visible] = "zbuffer_reject"

        focus_reason = reason[in_focus_valid]
        reason_counts = {key: int((focus_reason == key).sum()) for key in REASON_COLORS if key != "uv_invalid"}
        focus_cos = cosines[in_focus_valid]
        summary["views"][view_name] = {
            "reason_counts_for_uv_valid_focus_pixels": reason_counts,
            "cosine_min_mean_max": [
                float(focus_cos.min()) if len(focus_cos) else None,
                float(focus_cos.mean()) if len(focus_cos) else None,
                float(focus_cos.max()) if len(focus_cos) else None,
            ],
            "visible_pixels": int((in_focus_valid & visible).sum()),
            "normal_ok_pixels": int((in_focus_valid & normal_ok).sum()),
            "mask_ok_pixels": int((in_focus_valid & in_mask).sum()),
            "in_image_pixels": int((in_focus_valid & in_img).sum()),
        }

        atlas = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
        atlas[bbox_mask & ~valid_mask] = REASON_COLORS["uv_invalid"]
        for key, color in REASON_COLORS.items():
            if key == "uv_invalid":
                continue
            mask = np.zeros((TEX_SIZE, TEX_SIZE), dtype=bool)
            selected = in_focus_valid & (reason == key)
            mask[valid_y[selected], valid_x[selected]] = True
            atlas[mask] = color
        path = OUT_DIR / f"02_{view_name}_strict_reject_reason_focus.png"
        cv2.imwrite(str(path), _crop(atlas, bbox))
        overlay_paths[f"{view_name} reject"] = path

        raw_overlay = image.copy()
        selected_indices = np.where(in_focus_valid)[0]
        for key, color in REASON_COLORS.items():
            if key == "uv_invalid":
                continue
            pts = selected_indices[reason[selected_indices] == key]
            if len(pts) == 0:
                continue
            step = max(1, len(pts) // 2500)
            for idx in pts[::step]:
                cv2.circle(raw_overlay, (int(px_u[idx]), int(px_v[idx])), 2, color[::-1], -1, cv2.LINE_AA)
        raw_path = OUT_DIR / f"03_{view_name}_raw_projection_overlay.jpg"
        cv2.imwrite(str(raw_path), raw_overlay)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _make_contact_sheet(overlay_paths, OUT_DIR / "_forehead_sampling_failure_contact_sheet.jpg")
    print(json.dumps(summary, indent=2))
    print(str((OUT_DIR / "_forehead_sampling_failure_contact_sheet.jpg").resolve()))


if __name__ == "__main__":
    main()

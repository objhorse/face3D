"""
Generate an ordered visual audit of the face reconstruction pipeline.

This script is diagnostic-only: it reads the current inputs and pipeline outputs,
then writes images under output/debug/pipeline_audit without changing the core
pipeline results.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.coordinates import (  # noqa: E402
    camera_center_for_texture_visibility,
    project_texture_points_to_image,
)
from src.module1_preprocess import (  # noqa: E402
    create_face_mask_from_landmarks,
    detect_landmarks_mediapipe,
    load_images,
    preprocess_all_views,
)
from src.module0_intrinsics import undistort_images_with_calibration  # noqa: E402
from src.module3_texture import (  # noqa: E402
    _bary_batch,
    _bilinear_sample,
    _render_camera_depth,
    bake_texture,
    compute_face_normals,
    load_cameras,
    load_mesh_obj,
    poisson_seam_fix,
    rasterize_uv_map,
)


AUDIT_DIR = cfg.OUTPUT_DEBUG_DIR / "pipeline_audit"
VIEW_ORDER = ("left", "front", "right")
DIAG_TEX_SIZE = 1024


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resize_for_display(image: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def colorize_mask(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    base = image.copy()
    color_img = colorize_mask(mask.astype(np.uint8), color)
    m = mask.astype(bool)
    base[m] = (0.55 * base[m] + 0.45 * color_img[m]).astype(np.uint8)
    return base


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_legend(path: Path, entries: Iterable[Tuple[str, Tuple[int, int, int]]]) -> None:
    entries = list(entries)
    width = 560
    height = 48 + 42 * len(entries)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "reject reason legend", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    for idx, (label, color) in enumerate(entries):
        y = 58 + idx * 42
        cv2.rectangle(canvas, (22, y - 20), (58, y + 12), color[::-1], -1)
        cv2.rectangle(canvas, (22, y - 20), (58, y + 12), (40, 40, 40), 1)
        cv2.putText(canvas, label, (74, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def make_contact_sheet(items: Iterable[Tuple[str, Path]], out_path: Path, thumb_w: int = 420) -> None:
    thumbs = []
    for label, path in items:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb = cv2.resize(img, (thumb_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(draw_label(thumb, label))
    if not thumbs:
        return

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * cell_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y, x = r * cell_h, c * thumb_w
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def save_orientation_check(
    front_single_path: Path,
    pre_poisson_path: Path,
    current_path: Path,
    out_path: Path,
) -> None:
    """Compare raw UV display, V-flipped display, and final texture orientation."""
    entries = [
        ("front sample raw UV", front_single_path, False),
        ("front sample V-flipped", front_single_path, True),
        ("pre poisson final orientation", pre_poisson_path, False),
        ("current texture final orientation", current_path, False),
    ]
    panels = []
    for label, path, flip_v in entries:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (1024, 1024):
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        img = np.flipud(img).copy() if flip_v else img.copy()

        cv2.rectangle(img, (485, 390), (535, 455), (255, 0, 0), 4)
        cv2.putText(img, "forehead hole box", (360, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (485, 1024 - 455), (535, 1024 - 390), (255, 220, 0), 3)
        cv2.putText(img, "mirrored box", (360, 1024 - 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (1024, 45), (0, 0, 0), -1)
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if panels:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet = np.concatenate([cv2.resize(p, (512, 512), interpolation=cv2.INTER_AREA) for p in panels], axis=1)
        cv2.imwrite(str(out_path), sheet)


def mask_to_rgb(mask: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    out[mask.astype(bool)] = color
    return out


def crop_around_mask(image: np.ndarray, mask: np.ndarray, pad: int = 80) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    h, w = image.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    return image[y0:y1, x0:x1]


def save_zoom(path: Path, image: np.ndarray, scale: int = 4) -> None:
    if image.size == 0:
        return
    zoom = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    save_rgb(path, zoom)


def square_pad_rgb(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = image
    return canvas, (x_off, y_off)


def square_pad_mask(mask: np.ndarray, shape: Tuple[int, int], offset: Tuple[int, int]) -> np.ndarray:
    size = max(shape)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = offset
    h, w = shape
    canvas[y_off : y_off + h, x_off : x_off + w] = mask
    return canvas


def make_run_phase3_mask(view_name: str, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    lmks, _ = detect_landmarks_mediapipe(image, view_name)
    if lmks is None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, fallback = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return fallback

    face_oval = create_face_mask_from_landmarks(lmks, image.shape)
    scale = 4
    sh, sw = h // scale, w // scale
    img_small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (sw, sh))
    oval_s = cv2.resize(face_oval, (sw, sh), interpolation=cv2.INTER_NEAREST)
    expanded = cv2.dilate(oval_s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[expanded > 0] = cv2.GC_PR_BGD
    gc_mask[oval_s > 0] = cv2.GC_PR_FGD
    try:
        cv2.grabCut(
            img_small,
            gc_mask,
            None,
            np.zeros((1, 65), dtype=np.float64),
            np.zeros((1, 65), dtype=np.float64),
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        result_s = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        result_s = oval_s

    mask = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def copy_existing_debug(section_dir: Path) -> None:
    section_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "init_*.png",
        "init_*.json",
        "landmark_reproj_*.*",
        "mesh_projection_front.jpg",
        "depth_*.png",
    ]
    copied = []
    for pattern in patterns:
        for src in cfg.OUTPUT_DEBUG_DIR.glob(pattern):
            dst = section_dir / src.name
            shutil.copy2(src, dst)
            if dst.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                copied.append((dst.stem, dst))
    make_contact_sheet(copied, section_dir / "_contact_sheet.jpg")


def prepare_inputs() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        raw, _ = undistort_images_with_calibration(
            raw,
            calibration_path=getattr(cfg, "CAMERA_CALIBRATION_PATH", None),
            alpha=getattr(cfg, "UNDISTORT_ALPHA", 0.0),
        )
    padded = {}
    padded_masks = {}
    raw_masks = {}
    for view_name in VIEW_ORDER:
        image = raw[view_name]
        padded_img, offset = square_pad_rgb(image)
        padded[view_name] = padded_img
        mask = make_run_phase3_mask(view_name, image)
        raw_masks[view_name] = mask
        padded_masks[view_name] = square_pad_mask(mask, image.shape[:2], offset)
    return raw, padded, padded_masks


def save_raw_and_preprocess(raw: Dict[str, np.ndarray], padded: Dict[str, np.ndarray]) -> None:
    raw_dir = AUDIT_DIR / "00_raw"
    prep_dir = AUDIT_DIR / "01_preprocess"
    for view_name in VIEW_ORDER:
        save_rgb(raw_dir / f"{view_name}_raw.jpg", raw[view_name])
        save_rgb(raw_dir / f"{view_name}_square_padded.jpg", padded[view_name])
    make_contact_sheet(
        [(f"{v} raw", raw_dir / f"{v}_raw.jpg") for v in VIEW_ORDER]
        + [(f"{v} padded", raw_dir / f"{v}_square_padded.jpg") for v in VIEW_ORDER],
        raw_dir / "_contact_sheet.jpg",
    )

    preprocess_all_views(raw, debug_dir=prep_dir, target_size=getattr(cfg, "WORK_IMAGE_SIZE", 512))
    items = []
    for view_name in VIEW_ORDER:
        for suffix in ("landmarks", "face_mask", "bg_mask", "parser_mask", "masked"):
            path = prep_dir / f"{view_name}_{suffix}.png"
            if path.exists():
                items.append((f"{view_name} {suffix}", path))
    make_contact_sheet(items, prep_dir / "_contact_sheet.jpg")


def scale_cameras_for_padded_images(cameras: Dict[str, dict], images: Dict[str, np.ndarray]) -> Dict[str, dict]:
    scaled = {}
    base_size = float(getattr(cfg, "WORK_IMAGE_SIZE", 512))
    for view_name, image in images.items():
        cam = cameras[view_name]
        scale = image.shape[0] / base_size
        k = cam["K"].copy()
        k[0, :] *= scale
        k[1, :] *= scale
        scaled[view_name] = {"K": k, "R": cam["R"], "t": cam["t"]}
    return scaled


def render_texture_visibility(
    padded: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    tex_dir = AUDIT_DIR / "03_texture_visibility"
    fusion_dir = AUDIT_DIR / "04_texture_fusion"
    final_dir = AUDIT_DIR / "05_final"
    tex_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    vertices, faces, uv_verts, uv_faces = load_mesh_obj(cfg.OUTPUT_MESH_DIR / "face_mesh.obj")
    cameras = scale_cameras_for_padded_images(load_cameras(cfg.OUTPUT_MESH_DIR / "cameras.json"), padded)
    tri_map, bary_map = rasterize_uv_map(uv_verts, uv_faces, DIAG_TEX_SIZE)
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

    pass_count = np.zeros(len(valid_y), dtype=np.uint8)
    best_weight = np.full(len(valid_y), -1.0, dtype=np.float32)
    winner = np.full(len(valid_y), 255, dtype=np.uint8)
    single_view_items = []
    visibility_items = []
    stats = {}
    failure_masks: Dict[str, Dict[str, np.ndarray]] = {}

    for view_idx, view_name in enumerate(VIEW_ORDER):
        cam = cameras[view_name]
        image = padded[view_name]
        mask = masks[view_name]
        h_img, w_img = image.shape[:2]
        k, r, t = cam["K"], cam["R"], cam["t"]

        depth_map = _render_camera_depth(vertices, faces, k, r, t, (h_img, w_img))
        v_cam, z, proj, front = project_texture_points_to_image(pts_3d, k, r, t)
        in_img = front & (proj[:, 0] >= 0) & (proj[:, 0] < w_img - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h_img - 1)

        px_u = np.clip(proj[:, 0].astype(int), 0, w_img - 1)
        px_v = np.clip(proj[:, 1].astype(int), 0, h_img - 1)
        in_mask = mask[px_v, px_u] > 127
        cam_center = camera_center_for_texture_visibility(r, t)
        view_dirs = cam_center - pts_3d
        view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
        weights = np.maximum(0.0, np.sum(pt_normals * view_dirs, axis=1)) ** 2
        normal_ok = weights > 0.05
        z_ref = depth_map[px_v, px_u]
        z_ok = z <= (z_ref + 3e-3)
        sample_ok = in_img & in_mask & normal_ok & z_ok
        failure_masks[view_name] = {
            "in_img": in_img,
            "in_mask": in_img & in_mask,
            "normal_ok": in_img & in_mask & normal_ok,
            "z_ok": in_img & in_mask & normal_ok & z_ok,
            "sample_ok": sample_ok,
            "mask_fail": in_img & ~in_mask,
            "normal_fail": in_img & in_mask & ~normal_ok,
            "z_fail": in_img & in_mask & normal_ok & ~z_ok,
        }

        reason = np.zeros(len(valid_y), dtype=np.uint8)
        reason[~front] = 1
        reason[front & ~in_img] = 2
        reason[in_img & ~in_mask] = 3
        reason[in_img & in_mask & ~normal_ok] = 4
        reason[in_img & in_mask & normal_ok & ~z_ok] = 5
        reason[sample_ok] = 6

        normal_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE), dtype=np.uint8)
        normal_img[valid_y, valid_x] = np.clip(weights * 255.0, 0, 255).astype(np.uint8)
        normal_color = cv2.applyColorMap(normal_img, cv2.COLORMAP_TURBO)
        cv2.imwrite(str(tex_dir / f"{view_name}_normal_weight_uv.png"), normal_color)

        z_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE), dtype=np.uint8)
        z_img[valid_y[z_ok], valid_x[z_ok]] = 255
        save_gray(tex_dir / f"{view_name}_z_visible_uv.png", z_img)

        sample_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE), dtype=np.uint8)
        sample_img[valid_y[sample_ok], valid_x[sample_ok]] = 255
        save_gray(tex_dir / f"{view_name}_sample_valid_uv.png", sample_img)

        palette = np.array(
            [
                [0, 0, 0],
                [80, 80, 80],
                [70, 120, 255],
                [255, 180, 70],
                [255, 80, 80],
                [170, 70, 255],
                [80, 230, 120],
            ],
            dtype=np.uint8,
        )
        reason_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE, 3), dtype=np.uint8)
        reason_img[valid_y, valid_x] = palette[reason]
        save_rgb(tex_dir / f"{view_name}_reject_reason_uv.png", reason_img)

        single = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE, 3), dtype=np.uint8)
        if sample_ok.any():
            colors = _bilinear_sample(image, proj[sample_ok, 0], proj[sample_ok, 1])
            single[valid_y[sample_ok], valid_x[sample_ok]] = np.clip(colors, 0, 255).astype(np.uint8)
            candidate = sample_ok & (weights > best_weight)
            winner[candidate] = view_idx
            best_weight[candidate] = weights[candidate]
        save_rgb(fusion_dir / f"{view_name}_single_view_bake.png", single)

        pass_count += sample_ok.astype(np.uint8)
        visibility_items.extend(
            [
                (f"{view_name} normal", tex_dir / f"{view_name}_normal_weight_uv.png"),
                (f"{view_name} z visible", tex_dir / f"{view_name}_z_visible_uv.png"),
                (f"{view_name} sample", tex_dir / f"{view_name}_sample_valid_uv.png"),
                (f"{view_name} reject reason", tex_dir / f"{view_name}_reject_reason_uv.png"),
            ]
        )
        single_view_items.append((f"{view_name} single bake", fusion_dir / f"{view_name}_single_view_bake.png"))
        stats[view_name] = {
            "front": int(front.sum()),
            "in_img": int(in_img.sum()),
            "in_mask": int((in_img & in_mask).sum()),
            "normal_ok": int((in_img & in_mask & normal_ok).sum()),
            "z_ok": int((in_img & in_mask & normal_ok & z_ok).sum()),
            "sample_ok": int(sample_ok.sum()),
        }

    draw_legend(
        tex_dir / "reject_reason_legend.png",
        [
            ("black: invalid UV / no triangle", (0, 0, 0)),
            ("gray: behind camera", (80, 80, 80)),
            ("blue: projected outside image", (70, 120, 255)),
            ("orange: outside face mask", (255, 180, 70)),
            ("red: normal threshold failed", (255, 80, 80)),
            ("purple: z-buffer visibility failed", (170, 70, 255)),
            ("green: accepted sample", (80, 230, 120)),
        ],
    )
    visibility_items.insert(0, ("legend", tex_dir / "reject_reason_legend.png"))

    # Detailed forehead region panels. This is the area that currently becomes
    # the visible black hole, so split every predicate into a comparable crop.
    front_sample = failure_masks["front"]["sample_ok"]
    front_missing = valid_mask.copy()
    for view_name in VIEW_ORDER:
        front_missing[valid_y[failure_masks[view_name]["sample_ok"]], valid_x[failure_masks[view_name]["sample_ok"]]] = False
    forehead_roi = front_missing & (np.indices(valid_mask.shape)[0] < int(valid_mask.shape[0] * 0.48))
    if forehead_roi.any():
        detail_dir = tex_dir / "forehead_detail"
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_items = []
        for view_name in VIEW_ORDER:
            masks_for_view = failure_masks[view_name]
            for name in ("in_img", "in_mask", "normal_ok", "z_ok", "sample_ok", "mask_fail", "normal_fail", "z_fail"):
                img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE, 3), dtype=np.uint8)
                img[valid_y[masks_for_view[name]], valid_x[masks_for_view[name]]] = (255, 255, 255)
                crop = crop_around_mask(img, forehead_roi, pad=50)
                out_path = detail_dir / f"{view_name}_{name}_forehead_zoom.png"
                save_zoom(out_path, crop, scale=5)
                detail_items.append((f"{view_name} {name}", out_path))
        roi_img = mask_to_rgb(forehead_roi, (255, 255, 255))
        roi_path = detail_dir / "forehead_missing_roi.png"
        save_zoom(roi_path, crop_around_mask(roi_img, forehead_roi, pad=50), scale=5)
        detail_items.insert(0, ("missing ROI", roi_path))
        make_contact_sheet(detail_items, detail_dir / "_contact_sheet.jpg", thumb_w=320)

    pass_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE), dtype=np.uint8)
    pass_img[valid_y, valid_x] = np.clip(pass_count * 85, 0, 255)
    save_gray(fusion_dir / "sample_count_uv.png", pass_img)

    winner_palette = np.array([[40, 170, 255], [80, 235, 100], [255, 90, 80]], dtype=np.uint8)
    winner_img = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE, 3), dtype=np.uint8)
    for idx in range(len(VIEW_ORDER)):
        m = winner == idx
        winner_img[valid_y[m], valid_x[m]] = winner_palette[idx]
    save_rgb(fusion_dir / "winner_view_uv.png", winner_img)

    unsampled = np.zeros((DIAG_TEX_SIZE, DIAG_TEX_SIZE), dtype=np.uint8)
    unsampled[valid_y[pass_count == 0], valid_x[pass_count == 0]] = 255
    save_gray(fusion_dir / "unsampled_holes_uv.png", unsampled)

    make_contact_sheet(visibility_items, tex_dir / "_contact_sheet.jpg")
    make_contact_sheet(
        single_view_items
        + [
            ("sample count", fusion_dir / "sample_count_uv.png"),
            ("winner view", fusion_dir / "winner_view_uv.png"),
            ("unsampled holes", fusion_dir / "unsampled_holes_uv.png"),
        ],
        fusion_dir / "_contact_sheet.jpg",
    )

    pre = bake_texture(vertices, faces, uv_verts, uv_faces, tri_map, bary_map, cameras, padded, DIAG_TEX_SIZE, face_masks=masks)
    save_rgb(final_dir / "pre_poisson_texture_1024.png", pre)
    post = poisson_seam_fix(pre.copy(), valid_mask)
    save_rgb(final_dir / "post_poisson_texture_1024.png", post)

    current_path = cfg.OUTPUT_TEXTURE_DIR / "albedo_white.png"
    if current_path.exists():
        current = cv2.cvtColor(cv2.imread(str(current_path)), cv2.COLOR_BGR2RGB)
        save_rgb(final_dir / "current_albedo_white.png", current)
        current_small = cv2.resize(current, (DIAG_TEX_SIZE, DIAG_TEX_SIZE), interpolation=cv2.INTER_AREA)
        valid_small = valid_mask
        dark = valid_small & (current_small.mean(axis=2) < 12)
        overlay = overlay_mask(current_small, dark, (255, 0, 0))
        save_rgb(final_dir / "current_dark_holes_overlay.png", overlay)

        gray = current_small.mean(axis=2).astype(np.float32)
        seam = np.zeros_like(valid_small, dtype=bool)
        seam[:, 1:] |= np.abs(gray[:, 1:] - gray[:, :-1]) > 35
        seam[1:, :] |= np.abs(gray[1:, :] - gray[:-1, :]) > 35
        seam &= valid_small
        save_rgb(final_dir / "current_seam_overlay.png", overlay_mask(current_small, seam, (255, 220, 0)))
        save_orientation_check(
            fusion_dir / "front_single_view_bake.png",
            final_dir / "pre_poisson_texture_1024.png",
            final_dir / "current_albedo_white.png",
            tex_dir / "orientation_check_front_sample_vs_final.jpg",
        )
        visibility_items.insert(1, ("orientation check", tex_dir / "orientation_check_front_sample_vs_final.jpg"))

    make_contact_sheet(visibility_items, tex_dir / "_contact_sheet.jpg")

    make_contact_sheet(
        [
            ("pre poisson 1024", final_dir / "pre_poisson_texture_1024.png"),
            ("post poisson 1024", final_dir / "post_poisson_texture_1024.png"),
            ("current texture", final_dir / "current_albedo_white.png"),
            ("dark holes", final_dir / "current_dark_holes_overlay.png"),
            ("seams", final_dir / "current_seam_overlay.png"),
        ],
        final_dir / "_contact_sheet.jpg",
    )

    (AUDIT_DIR / "texture_visibility_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return pre, post


def build_index() -> None:
    sections = [
        ("00 Raw", AUDIT_DIR / "00_raw"),
        ("01 Preprocess", AUDIT_DIR / "01_preprocess"),
        ("02 Geometry", AUDIT_DIR / "02_geometry"),
        ("03 Texture Visibility", AUDIT_DIR / "03_texture_visibility"),
        ("04 Texture Fusion", AUDIT_DIR / "04_texture_fusion"),
        ("05 Final", AUDIT_DIR / "05_final"),
    ]
    body = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Pipeline Audit</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f1e8;color:#211b16}"
        "h1{margin-bottom:4px}.section{margin:28px 0;padding:18px;background:white;border-radius:14px}"
        "img{max-width:100%;border:1px solid #ddd;border-radius:8px}a{color:#8b3f1f}</style>",
        "<h1>Face3D Pipeline Audit</h1>",
        "<p>Ordered visual outputs from raw images through preprocessing, geometry, texture visibility, fusion, and final texture.</p>",
    ]
    for title, directory in sections:
        body.append(f"<div class='section'><h2>{html.escape(title)}</h2>")
        sheet = directory / "_contact_sheet.jpg"
        if sheet.exists():
            body.append(f"<p><a href='{sheet.relative_to(AUDIT_DIR).as_posix()}'>contact sheet</a></p>")
            body.append(f"<img src='{sheet.relative_to(AUDIT_DIR).as_posix()}'>")
        images = sorted(p for p in directory.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        body.append("<ul>")
        for image_path in images:
            if image_path.name == "_contact_sheet.jpg":
                continue
            rel = image_path.relative_to(AUDIT_DIR).as_posix()
            body.append(f"<li><a href='{rel}'>{html.escape(rel)}</a></li>")
        body.append("</ul></div>")
    (AUDIT_DIR / "index.html").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    ensure_clean_dir(AUDIT_DIR)
    raw, padded, padded_masks = prepare_inputs()
    save_raw_and_preprocess(raw, padded)
    copy_existing_debug(AUDIT_DIR / "02_geometry")
    render_texture_visibility(padded, padded_masks)
    build_index()
    print(f"Audit written to: {AUDIT_DIR}")
    print(f"Open index: {AUDIT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()

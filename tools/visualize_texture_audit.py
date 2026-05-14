"""Visual audit for the final baked texture.

This reads the exported GLB and albedo texture, then renders the textured mesh
back into each calibrated camera view. The output is meant for human inspection:
photo, rendered texture, overlay, and a V-flipped comparison.
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.module0_intrinsics import undistort_images_with_calibration  # noqa: E402
from src.module1_preprocess import load_images  # noqa: E402
from src.module3_texture import load_cameras  # noqa: E402


OUT_DIR = cfg.OUTPUT_DEBUG_DIR / "texture_visual_audit"
VIEW_ORDER = ("left", "front", "right")
DISPLAY_SIZE = 1200
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
VIEW_LABELS = {
    "left": "左侧",
    "front": "正面",
    "right": "右侧",
}


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def square_pad_rgb(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    size = max(h, w)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def resize_square(image: np.ndarray, size: int = DISPLAY_SIZE) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def barycentric(points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.full((len(points), 3), -1.0, dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u, v, w], axis=1).astype(np.float32)


def sample_texture(texture: np.ndarray, uv: np.ndarray, flip_v: bool = False) -> np.ndarray:
    h, w = texture.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(1.0 - uv[:, 1] if flip_v else uv[:, 1], 0.0, 1.0)
    x = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    y = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    return texture[y, x]


def render_projected_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture: np.ndarray,
    camera: dict,
    image_shape: Tuple[int, int],
    flip_v: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    k = camera["K"]
    r = camera["R"]
    t = camera["t"].reshape(3)

    cam = (r @ vertices.T).T + t
    z = cam[:, 2]
    projected = np.empty((len(vertices), 2), dtype=np.float32)
    projected[:, 0] = k[0, 0] * (cam[:, 0] / np.clip(z, 1e-8, None)) + k[0, 2]
    projected[:, 1] = k[1, 1] * (cam[:, 1] / np.clip(z, 1e-8, None)) + k[1, 2]

    out = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    zbuf = np.full((h, w), np.inf, dtype=np.float32)

    for face in faces:
        if np.any(z[face] <= 1e-6):
            continue
        pts = projected[face]
        x0 = max(0, int(np.floor(np.min(pts[:, 0]))))
        x1 = min(w - 1, int(np.ceil(np.max(pts[:, 0]))))
        y0 = max(0, int(np.floor(np.min(pts[:, 1]))))
        y1 = min(h - 1, int(np.ceil(np.max(pts[:, 1]))))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
        bary = barycentric(pix, pts[0], pts[1], pts[2])
        inside = np.all(bary >= -1e-5, axis=1)
        if not inside.any():
            continue

        px = pix[inside, 0].astype(np.int32)
        py = pix[inside, 1].astype(np.int32)
        b = bary[inside]
        depth = b @ z[face]
        update = depth < zbuf[py, px]
        if not update.any():
            continue

        px = px[update]
        py = py[update]
        b = b[update]
        depth = depth[update]
        uv_pix = b @ uv[face]
        out[py, px] = sample_texture(texture, uv_pix, flip_v=flip_v)
        zbuf[py, px] = depth
        mask[py, px] = True

    return out, mask


def make_overlay(photo: np.ndarray, render: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = photo.copy()
    m = mask.astype(bool)
    out[m] = (0.48 * photo[m] + 0.52 * render[m]).astype(np.uint8)
    return out


def font(size: int) -> ImageFont.FreeTypeFont:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def draw_text_rgb(image: np.ndarray, text: str, xy: Tuple[int, int], size: int = 28) -> np.ndarray:
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=(255, 255, 255), font=font(size))
    return np.asarray(pil)


def draw_label_bgr(image: np.ndarray, label: str) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb[:46, :] = 0
    rgb = draw_text_rgb(rgb, label, (12, 8), size=26)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_contact_sheet(panels: Dict[str, Path], out_path: Path, thumb: int = 360) -> None:
    images = []
    for label, path in panels.items():
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img = cv2.resize(bgr, (thumb, thumb), interpolation=cv2.INTER_AREA)
        images.append(draw_label_bgr(img, label))
    if not images:
        return
    cols = 3
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * thumb, cols * thumb, 3), 245, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * thumb : (r + 1) * thumb, c * thumb : (c + 1) * thumb] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def load_exported_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(cfg.OUTPUT_MESH_DIR / "face.glb", force="scene")
    if hasattr(loaded, "geometry"):
        mesh = next(iter(loaded.geometry.values()))
    else:
        mesh = loaded
    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        uv,
    )


def load_display_inputs() -> Dict[str, np.ndarray]:
    images = load_images(cfg.DEFAULT_IMAGE_DIR, cfg.DEFAULT_VIEW_NAMES)
    if getattr(cfg, "UNDISTORT_IMAGES", True):
        images, _ = undistort_images_with_calibration(
            images,
            calibration_path=getattr(cfg, "CAMERA_CALIBRATION_PATH", None),
            alpha=getattr(cfg, "UNDISTORT_ALPHA", 0.0),
        )
    return {name: resize_square(square_pad_rgb(images[name]), DISPLAY_SIZE) for name in VIEW_ORDER}


def scaled_cameras() -> Dict[str, dict]:
    cameras = load_cameras(cfg.OUTPUT_MESH_DIR / "cameras.json")
    scale = DISPLAY_SIZE / float(getattr(cfg, "WORK_IMAGE_SIZE", 1024))
    out = {}
    for name, cam in cameras.items():
        k = cam["K"].copy()
        k[0, :] *= scale
        k[1, :] *= scale
        out[name] = {"K": k, "R": cam["R"], "t": cam["t"]}
    return out


def write_texture_reference(texture: np.ndarray, out_path: Path) -> None:
    ref = texture.copy()
    h, w = ref.shape[:2]
    for y in (0, h // 4, h // 2, 3 * h // 4, h - 1):
        cv2.line(ref, (0, y), (w - 1, y), (255, 255, 255), 2)
    for x in (0, w // 4, w // 2, 3 * w // 4, w - 1):
        cv2.line(ref, (x, 0), (x, h - 1), (255, 255, 255), 2)
    cv2.rectangle(ref, (0, 0), (920, 82), (0, 0, 0), -1)
    ref = draw_text_rgb(ref, "最终 albedo UV 展开图，不是相机照片", (24, 18), size=38)
    save_rgb(out_path, ref)


def write_html(paths: Dict[str, Path], stats: dict) -> None:
    rows = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>纹理可视化审计</title>",
        "<style>body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:24px;background:#f5f5f5;color:#111} img{max-width:100%;border:1px solid #ccc;background:#000} .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:white;padding:14px;border-radius:6px} code{white-space:pre-wrap}</style>",
        "<h1>纹理可视化审计</h1>",
        "<p>用“叠加检查图”判断最终贴图是否和标定照片对齐。“V 方向翻转对照”是故意做的错误版本，用来一眼排查上下反的问题。</p>",
        "<div class='card'><h2>统计摘要</h2><code>",
        html.escape(json.dumps(stats, indent=2, ensure_ascii=False)),
        "</code></div>",
        "<div class='grid'>",
    ]
    for label, path in paths.items():
        rel = path.relative_to(OUT_DIR).as_posix()
        rows.append(f"<div class='card'><h2>{html.escape(label)}</h2><img src='{html.escape(rel)}'></div>")
    rows.append("</div>")
    (OUT_DIR / "index.html").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vertices, faces, uv = load_exported_mesh()
    texture_bgr = cv2.imread(str(cfg.OUTPUT_TEXTURE_DIR / "albedo_white.png"), cv2.IMREAD_COLOR)
    if texture_bgr is None:
        raise FileNotFoundError(cfg.OUTPUT_TEXTURE_DIR / "albedo_white.png")
    texture = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)

    photos = load_display_inputs()
    cameras = scaled_cameras()
    paths: Dict[str, Path] = {}
    stats = {
        "mesh_顶点数": int(len(vertices)),
        "mesh_面片数": int(len(faces)),
        "贴图尺寸_RGB": list(texture.shape),
        "可视化画布尺寸": DISPLAY_SIZE,
        "各视角": {},
    }

    tex_ref = OUT_DIR / "00_texture_atlas_with_grid.jpg"
    write_texture_reference(texture, tex_ref)
    paths["最终 albedo UV 展开图"] = tex_ref

    crop_map = cfg.OUTPUT_DEBUG_DIR / "visible_face_crop" / "visible_face_crop_uv_keep_delete.png"
    if crop_map.exists():
        crop_copy = OUT_DIR / "01_visible_crop_map_green_keep_red_delete.png"
        shutil.copy2(crop_map, crop_copy)
        paths["可见脸部裁剪图：绿色保留，红色删除"] = crop_copy

    contact_panels: Dict[str, Path] = {}
    for view in VIEW_ORDER:
        photo = photos[view]
        render, mask = render_projected_texture(vertices, faces, uv, texture, cameras[view], photo.shape[:2], flip_v=True)
        flipped, flipped_mask = render_projected_texture(vertices, faces, uv, texture, cameras[view], photo.shape[:2], flip_v=False)
        overlay = make_overlay(photo, render, mask)
        flipped_overlay = make_overlay(photo, flipped, flipped_mask)
        view_label = VIEW_LABELS.get(view, view)

        view_paths = {
            f"{view_label} 原始照片": OUT_DIR / f"{view}_01_photo.jpg",
            f"{view_label} 当前 UV 贴图渲染": OUT_DIR / f"{view}_02_render_pipeline_uv.jpg",
            f"{view_label} 当前 UV 叠加检查": OUT_DIR / f"{view}_03_overlay_pipeline_uv.jpg",
            f"{view_label} V 方向翻转对照渲染": OUT_DIR / f"{view}_04_render_vflipped_check.jpg",
            f"{view_label} V 方向翻转叠加对照": OUT_DIR / f"{view}_05_overlay_vflipped_check.jpg",
        }
        save_rgb(view_paths[f"{view_label} 原始照片"], photo)
        save_rgb(view_paths[f"{view_label} 当前 UV 贴图渲染"], render)
        save_rgb(view_paths[f"{view_label} 当前 UV 叠加检查"], overlay)
        save_rgb(view_paths[f"{view_label} V 方向翻转对照渲染"], flipped)
        save_rgb(view_paths[f"{view_label} V 方向翻转叠加对照"], flipped_overlay)
        paths.update(view_paths)

        contact_panels[f"{view_label} 原始照片"] = view_paths[f"{view_label} 原始照片"]
        contact_panels[f"{view_label} 当前UV渲染"] = view_paths[f"{view_label} 当前 UV 贴图渲染"]
        contact_panels[f"{view_label} 当前UV叠加"] = view_paths[f"{view_label} 当前 UV 叠加检查"]
        contact_panels[f"{view_label} V翻转渲染"] = view_paths[f"{view_label} V 方向翻转对照渲染"]
        contact_panels[f"{view_label} V翻转叠加"] = view_paths[f"{view_label} V 方向翻转叠加对照"]
        stats["各视角"][view_label] = {
            "当前UV渲染像素数": int(mask.sum()),
            "V方向翻转对照渲染像素数": int(flipped_mask.sum()),
        }

    contact = OUT_DIR / "_contact_sheet.jpg"
    make_contact_sheet(contact_panels, contact)
    paths = {"总览拼图": contact, **paths}

    write_html(paths, stats)
    (OUT_DIR / "summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"纹理可视化审计已写入: {OUT_DIR / 'index.html'}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

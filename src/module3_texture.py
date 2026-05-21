"""
模块3：多视角纹理融合 + GLB 打包

流程：
  1. UV 光栅化：为每个纹理像素找到对应的3D三角面片和重心坐标
  2. 多视角颜色采样：将3D点投影到每张照片，采样颜色
  3. 加权混合：用面法线·相机方向夹角余弦值作为权重
  4. 泊松接缝修补：消除视角切换处的色差
  5. GLB 打包：Mesh + 纹理 + lighting元数据，可选Draco压缩

输入：
  - output/meshes/face_mesh.obj
  - output/meshes/cameras.json
  - 原始3张照片
  - 光照类型配置

输出：
  - output/textures/albedo_white.png   — 纹理贴图
  - output/meshes/face.glb             — 最终 GLB
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.coordinates import (
    camera_center_for_texture_visibility,
    image_uv_to_obj_uv,
    obj_uv_to_image_uv,
    project_texture_points_to_image,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_mesh_obj(obj_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 face_mesh.obj，返回：
      vertices: (N, 3) 几何顶点
      faces:    (F, 3) 几何面片索引 (0-based)
      uv_verts: (T, 2) UV 坐标 [0,1]（V 轴已翻转，重新翻回来）
      uv_faces: (F, 3) UV 面片索引
    """
    vertices, uv_verts, faces, uv_faces = [], [], [], []
    with open(str(obj_path)) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                u = float(parts[1])
                v = float(parts[2])
                uv_verts.append([u, v])
            elif parts[0] == "f":
                gf, uf = [], []
                for token in parts[1:4]:
                    segs = token.split("/")   # v/vt 或 v/vt/vn
                    gf.append(int(segs[0]) - 1)
                    uf.append(int(segs[1]) - 1)
                faces.append(gf)
                uv_faces.append(uf)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        obj_uv_to_image_uv(np.array(uv_verts, dtype=np.float32)),
        np.array(uv_faces, dtype=np.int32),
    )


def load_cameras(cameras_json: Path) -> Dict[str, dict]:
    """加载 cameras.json，返回 {view_name: {K, R, t}}（numpy arrays）"""
    with open(str(cameras_json)) as f:
        data = json.load(f)
    result = {}
    for name, cam in data["views"].items():
        result[name] = {
            "K": np.array(cam["K"], dtype=np.float32),
            "R": np.array(cam["R"], dtype=np.float32),
            "t": np.array(cam["t"], dtype=np.float32),
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# UV 光栅化
# ══════════════════════════════════════════════════════════════════════════════

def rasterize_uv_map(
    uv_verts: np.ndarray,   # (T, 2) UV 坐标 [0,1]
    uv_faces: np.ndarray,   # (F, 3) UV 面片索引
    tex_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 UV 空间光栅化到纹理图像坐标。

    Returns:
        tri_map:  (H, W) int32  — 每像素所属三角面片索引（-1=空）
        bary_map: (H, W, 3) float32 — 重心坐标
    """
    H = W = tex_size
    tri_map  = np.full((H, W), -1, dtype=np.int32)
    bary_map = np.zeros((H, W, 3), dtype=np.float32)

    # UV [0,1] → 像素坐标 [0, tex_size)
    # FLAME/DECA UV 使用图像坐标约定：v=0 在顶部，v=1 在底部（与 PNG 图像 y 轴一致）
    # 不需要翻转 V 轴
    uvs_px = np.stack([
        uv_verts[:, 0] * tex_size,   # u → x (不变)
        uv_verts[:, 1] * tex_size,   # v → y (不翻转)
    ], axis=1)  # (T, 2)

    logger.info(f"  UV 光栅化 {len(uv_faces)} 个三角形到 {tex_size}×{tex_size} 纹理...")

    for tri_idx in range(len(uv_faces)):
        i0, i1, i2 = uv_faces[tri_idx]
        p0 = uvs_px[i0]
        p1 = uvs_px[i1]
        p2 = uvs_px[i2]

        # 包围盒（加1像素边距防止边缘漏点）
        x_min = max(0, int(min(p0[0], p1[0], p2[0])) - 1)
        x_max = min(W - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        y_min = max(0, int(min(p0[1], p1[1], p2[1])) - 1)
        y_max = min(H - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        if x_max < x_min or y_max < y_min:
            continue

        # 生成包围盒内的所有像素中心
        xs = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)

        # 批量计算重心坐标
        bary = _bary_batch(pts, p0, p1, p2)  # (N, 3)
        inside = np.all(bary >= -1e-5, axis=1)

        if not inside.any():
            continue

        px = pts[inside, 0].astype(int)
        py = pts[inside, 1].astype(int)
        # 只覆盖还未被占用的像素（避免重叠三角形互相覆盖）
        unset = tri_map[py, px] == -1
        tri_map[py[unset], px[unset]] = tri_idx
        bary_map[py[unset], px[unset]] = bary[inside][unset]

    filled = (tri_map >= 0).sum()
    logger.info(f"  UV 光栅化完成: {filled}/{H*W} 像素有效 ({filled*100/(H*W):.1f}%)")
    return tri_map, bary_map


def add_uv_hole_fill_faces_for_bake(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_verts: np.ndarray,
    uv_faces: np.ndarray,
    tex_size: int = 2048,
    max_area: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Add temporary bake-only fan triangles over small enclosed UV holes."""
    tri_map, bary_map = rasterize_uv_map(uv_verts, uv_faces, tex_size)
    valid_mask = tri_map >= 0
    holes = _small_internal_invalid_uv_holes(valid_mask, max_area=max_area)
    if not holes.any():
        return vertices, faces, uv_verts, uv_faces, 0

    valid_y, valid_x = np.where(valid_mask)
    valid_tri = tri_map[valid_y, valid_x]
    valid_bary = bary_map[valid_y, valid_x]
    g_faces = faces[valid_tri]
    pts_3d = (
        valid_bary[:, 0:1] * vertices[g_faces[:, 0]]
        + valid_bary[:, 1:2] * vertices[g_faces[:, 1]]
        + valid_bary[:, 2:3] * vertices[g_faces[:, 2]]
    )
    point_lookup = {(int(x), int(y)): p for x, y, p in zip(valid_x, valid_y, pts_3d)}

    new_vertices = [vertices]
    new_faces = [faces]
    new_uv_verts = [uv_verts]
    new_uv_faces = [uv_faces]
    vertex_offset = len(vertices)
    uv_offset = len(uv_verts)
    added_faces = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= 0 or area > max_area:
            continue
        hole_mask = labels == label
        ring_mask = cv2.dilate(hole_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        ring_mask &= valid_mask
        ring_y, ring_x = np.where(ring_mask)
        if len(ring_x) < 3:
            continue
        ring_xy = np.stack([ring_x, ring_y], axis=1).astype(np.float32)
        ring_vertices = np.asarray([point_lookup[(int(x), int(y))] for x, y in zip(ring_x, ring_y)], dtype=np.float32)

        hole_y, hole_x = np.where(hole_mask)
        hole_xy = np.stack([hole_x, hole_y], axis=1).astype(np.float32)
        verts_add = np.empty((len(hole_xy), 3), dtype=np.float32)
        for idx, xy in enumerate(hole_xy):
            d2 = np.sum((ring_xy - xy[None, :]) ** 2, axis=1)
            nearest = np.argsort(d2)[:8]
            weights = 1.0 / np.maximum(d2[nearest], 1e-3)
            weights = weights / weights.sum()
            verts_add[idx] = (ring_vertices[nearest] * weights[:, None]).sum(axis=0)
        uvs_add = np.stack(
            [(hole_x.astype(np.float32) + 0.5) / tex_size, (hole_y.astype(np.float32) + 0.5) / tex_size],
            axis=1,
        ).astype(np.float32)

        local_index = -np.ones(hole_mask.shape, dtype=np.int32)
        local_index[hole_y, hole_x] = np.arange(len(hole_x), dtype=np.int32)
        faces_add = []
        uv_faces_add = []
        ys, xs = np.where(hole_mask[:-1, :-1] & hole_mask[:-1, 1:] & hole_mask[1:, :-1] & hole_mask[1:, 1:])
        for yy, xx in zip(ys, xs):
            i00 = int(local_index[yy, xx])
            i10 = int(local_index[yy, xx + 1])
            i01 = int(local_index[yy + 1, xx])
            i11 = int(local_index[yy + 1, xx + 1])
            faces_add.append([vertex_offset + i00, vertex_offset + i10, vertex_offset + i01])
            faces_add.append([vertex_offset + i10, vertex_offset + i11, vertex_offset + i01])
            uv_faces_add.append([uv_offset + i00, uv_offset + i10, uv_offset + i01])
            uv_faces_add.append([uv_offset + i10, uv_offset + i11, uv_offset + i01])
        if not faces_add:
            continue
        faces_add = np.asarray(faces_add, dtype=np.int32)
        uv_faces_add = np.asarray(uv_faces_add, dtype=np.int32)
        face_vertices = verts_add[faces_add - vertex_offset]
        fan_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        if float(fan_normals[:, 2].mean()) < 0.0:
            faces_add[:, [1, 2]] = faces_add[:, [2, 1]]

        new_vertices.append(verts_add)
        new_uv_verts.append(uvs_add)
        new_faces.append(faces_add)
        new_uv_faces.append(uv_faces_add)
        vertex_offset += len(verts_add)
        uv_offset += len(uvs_add)
        added_faces += len(faces_add)

    if added_faces == 0:
        return vertices, faces, uv_verts, uv_faces, 0

    return (
        np.vstack(new_vertices).astype(np.float32),
        np.vstack(new_faces).astype(np.int32),
        np.vstack(new_uv_verts).astype(np.float32),
        np.vstack(new_uv_faces).astype(np.int32),
        added_faces,
    )


def _bary_batch(pts: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    批量计算点 pts(N,2) 相对于三角形 (a,b,c) 的重心坐标。
    Returns: (N, 3) — [w_a, w_b, w_c]
    """
    v0 = c - a
    v1 = b - a
    v2 = pts - a   # (N, 2)

    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0  # (N,)
    d21 = v2 @ v1  # (N,)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.full((len(pts), 3), -1.0, dtype=np.float32)

    v = (d11 * d20 - d01 * d21) / denom  # (N,)
    w = (d00 * d21 - d01 * d20) / denom  # (N,)
    u = 1.0 - v - w

    return np.stack([u, w, v], axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 面法线计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """返回 (F, 3) 单位面法线（指向相机方向，即与 FLAME 法线方向一致）"""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)
    return normals.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 多视角颜色采样
# ══════════════════════════════════════════════════════════════════════════════

def _bilinear_sample(image: np.ndarray, u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
    """
    双线性采样图像。
    image: (H, W, 3) uint8
    u_px, v_px: (N,) float32 像素坐标
    Returns: (N, 3) float32 [0,255]
    """
    H, W = image.shape[:2]
    u0 = np.floor(u_px).astype(int)
    v0 = np.floor(v_px).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    # 边界裁剪
    u0 = np.clip(u0, 0, W - 1)
    u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)

    wu = (u_px - np.floor(u_px)).reshape(-1, 1)
    wv = (v_px - np.floor(v_px)).reshape(-1, 1)

    c00 = image[v0, u0].astype(np.float32)
    c10 = image[v0, u1].astype(np.float32)
    c01 = image[v1, u0].astype(np.float32)
    c11 = image[v1, u1].astype(np.float32)

    return (c00 * (1 - wu) * (1 - wv) +
            c10 * wu       * (1 - wv) +
            c01 * (1 - wu) * wv       +
            c11 * wu       * wv)


def _render_camera_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Render a coarse camera-space z-buffer for visibility filtering."""
    H, W = image_shape
    depth = np.full((H, W), np.inf, dtype=np.float32)

    v_cam, z, proj, valid = project_texture_points_to_image(vertices, K, R, t)
    if not np.any(valid):
        return depth

    for tri in faces:
        tri_z = z[tri]
        if np.any(tri_z <= 1e-4):
            continue
        pts = proj[tri]
        x_min = max(0, int(np.floor(np.min(pts[:, 0]))))
        x_max = min(W - 1, int(np.ceil(np.max(pts[:, 0]))))
        y_min = max(0, int(np.floor(np.min(pts[:, 1]))))
        y_max = min(H - 1, int(np.ceil(np.max(pts[:, 1]))))
        if x_min > x_max or y_min > y_max:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
        bary = _bary_batch(pix, pts[0], pts[1], pts[2])
        inside = np.all(bary >= -1e-5, axis=1)
        if not np.any(inside):
            continue

        pix_in = pix[inside]
        depth_in = (bary[inside] @ tri_z.astype(np.float32)).astype(np.float32)
        px = pix_in[:, 0].astype(np.int32)
        py = pix_in[:, 1].astype(np.int32)
        np.minimum.at(depth, (py, px), depth_in)

    return depth


def _expand_face_selection(faces: np.ndarray, keep_faces: np.ndarray, rings: int) -> np.ndarray:
    expanded = keep_faces.copy()
    for _ in range(max(0, int(rings))):
        keep_vertices = np.zeros(int(faces.max()) + 1, dtype=bool)
        keep_vertices[faces[expanded].ravel()] = True
        expanded |= np.any(keep_vertices[faces], axis=1)
    return expanded


def _visible_face_crop_strict_settings(image_shape: Tuple[int, int]) -> Tuple[bool, int, float]:
    try:
        from src import config as cfg
        enabled = bool(getattr(cfg, "VISIBLE_FACE_CROP_STRICT_BOUNDARY", True))
        base_erode_px = float(getattr(cfg, "VISIBLE_FACE_CROP_ERODE_PX_AT_1024", 3))
        lower_start = float(getattr(cfg, "VISIBLE_FACE_CROP_LOWER_STRICT_START", 0.78))
        base_size = float(getattr(cfg, "WORK_IMAGE_SIZE", 1024))
    except Exception:
        enabled, base_erode_px, lower_start, base_size = True, 3.0, 0.78, 1024.0

    h, w = image_shape[:2]
    erode_px = int(round(base_erode_px * max(h, w) / max(base_size, 1.0)))
    return enabled, max(0, erode_px), float(np.clip(lower_start, 0.0, 1.0))


def _erode_binary_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if erode_px <= 0:
        return mask
    k = max(3, int(erode_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask, kernel, iterations=1)


def _mask_lower_strict_y(mask: np.ndarray, lower_start: float) -> int:
    ys = np.where(mask > 127)[0]
    if len(ys) == 0:
        return mask.shape[0]
    y_min, y_max = int(ys.min()), int(ys.max())
    return int(round(y_min + (y_max - y_min) * float(lower_start)))


def _strict_visible_face_filter(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    center_proj: np.ndarray,
    base_keep: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Tighten visible-crop decisions near mask boundaries and the chin."""
    enabled, erode_px, lower_start = _visible_face_crop_strict_settings(image_shape)
    if not enabled or not np.any(base_keep):
        return base_keep, {"enabled": int(enabled), "rejected": 0}

    h_img, w_img = image_shape[:2]
    eroded = _erode_binary_mask(mask, erode_px)
    lower_y = _mask_lower_strict_y(mask, lower_start)

    tri = vertices[faces].astype(np.float32)
    edge_mid = (tri + np.roll(tri, -1, axis=1)) * 0.5
    samples = np.concatenate([tri, edge_mid], axis=1)
    _, _, sample_proj_flat, sample_front_flat = project_texture_points_to_image(
        samples.reshape(-1, 3), K, R, t
    )
    sample_proj = sample_proj_flat.reshape(len(faces), 6, 2)
    sample_front = sample_front_flat.reshape(len(faces), 6)

    sx = np.clip(sample_proj[:, :, 0].astype(np.int32), 0, w_img - 1)
    sy = np.clip(sample_proj[:, :, 1].astype(np.int32), 0, h_img - 1)
    sample_in_img = (
        sample_front
        & (sample_proj[:, :, 0] >= 0)
        & (sample_proj[:, :, 0] < w_img - 1)
        & (sample_proj[:, :, 1] >= 0)
        & (sample_proj[:, :, 1] < h_img - 1)
    )
    sample_in_mask = sample_in_img & (mask[sy, sx] > 127)
    sample_in_eroded = sample_in_img & (eroded[sy, sx] > 127)

    cx = np.clip(center_proj[:, 0].astype(np.int32), 0, w_img - 1)
    cy = np.clip(center_proj[:, 1].astype(np.int32), 0, h_img - 1)
    center_in_eroded = eroded[cy, cx] > 127
    lower_face = cy >= lower_y
    boundary_face = ~center_in_eroded

    all_samples_in_mask = np.all(sample_in_mask, axis=1)
    all_samples_in_eroded = np.all(sample_in_eroded, axis=1)
    strict_needed = base_keep & (boundary_face | lower_face)
    strict_ok = all_samples_in_mask & (~lower_face | all_samples_in_eroded)
    keep = base_keep & (~strict_needed | strict_ok)

    rejected = base_keep & ~keep
    stats = {
        "enabled": 1,
        "erode_px": int(erode_px),
        "lower_y": int(lower_y),
        "base_keep": int(base_keep.sum()),
        "strict_needed": int(strict_needed.sum()),
        "rejected": int(rejected.sum()),
        "rejected_lower": int((rejected & lower_face).sum()),
        "rejected_boundary": int((rejected & boundary_face).sum()),
    }
    return keep, stats


def _save_visible_face_crop_debug(
    uv_verts: np.ndarray,
    uv_faces: np.ndarray,
    keep_faces: np.ndarray,
    out_dir: Path,
    tex_size: int = 1024,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tri_map, _ = rasterize_uv_map(uv_verts, uv_faces, tex_size)
    image = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    valid = tri_map >= 0
    image[valid & keep_faces[np.clip(tri_map, 0, len(keep_faces) - 1)]] = (70, 220, 110)
    image[valid & ~keep_faces[np.clip(tri_map, 0, len(keep_faces) - 1)]] = (255, 70, 70)
    cv2.imwrite(str(out_dir / "visible_face_crop_uv_keep_delete.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def crop_mesh_to_visible_face(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_verts: np.ndarray,
    uv_faces: np.ndarray,
    cameras: Dict[str, dict],
    images: Dict[str, np.ndarray],
    face_masks: Optional[Dict[str, np.ndarray]],
    debug_dir: Path,
    dilate_rings: int = 2,
    z_tol: float = 0.006,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep mesh faces that are visible through at least one face-mask view."""
    if face_masks is None:
        return faces, uv_faces, np.ones(len(faces), dtype=bool)

    face_centers = vertices[faces].mean(axis=1).astype(np.float32)
    face_keep = np.zeros(len(faces), dtype=bool)

    for view_name, cam in cameras.items():
        if view_name not in images or view_name not in face_masks:
            continue
        image = images[view_name]
        mask = face_masks[view_name]
        h_img, w_img = image.shape[:2]
        k, r, t = cam["K"], cam["R"], cam["t"]

        depth_map = _render_camera_depth(vertices, faces, k, r, t, (h_img, w_img))
        _, z, proj, front = project_texture_points_to_image(face_centers, k, r, t)
        in_img = (
            front
            & (proj[:, 0] >= 0)
            & (proj[:, 0] < w_img - 1)
            & (proj[:, 1] >= 0)
            & (proj[:, 1] < h_img - 1)
        )
        px = np.clip(proj[:, 0].astype(np.int32), 0, w_img - 1)
        py = np.clip(proj[:, 1].astype(np.int32), 0, h_img - 1)
        in_mask = mask[py, px] > 127
        z_ref = depth_map[py, px]
        z_ok = z <= (z_ref + float(z_tol))
        view_keep = in_img & in_mask & z_ok
        view_keep, strict_stats = _strict_visible_face_filter(
            vertices, faces, k, r, t, mask, proj, view_keep, (h_img, w_img)
        )
        if strict_stats.get("enabled"):
            logger.info(
                f"  [{view_name}] visible crop strict: "
                f"need={strict_stats.get('strict_needed', 0)}, "
                f"reject={strict_stats.get('rejected', 0)} "
                f"(lower={strict_stats.get('rejected_lower', 0)}, "
                f"boundary={strict_stats.get('rejected_boundary', 0)}), "
                f"erode={strict_stats.get('erode_px', 0)}px"
            )
        face_keep |= view_keep

    if not np.any(face_keep):
        logger.warning("Visible face crop found no faces; keeping the full mesh")
        return faces, uv_faces, np.ones(len(faces), dtype=bool)

    face_keep = _expand_face_selection(faces, face_keep, dilate_rings)
    kept = int(face_keep.sum())
    logger.info(
        f"  Visible face crop: keep {kept}/{len(faces)} faces "
        f"({kept * 100.0 / max(len(faces), 1):.1f}%), delete {len(faces) - kept}"
    )
    _save_visible_face_crop_debug(uv_verts, uv_faces, face_keep, debug_dir)
    return faces[face_keep], uv_faces[face_keep], face_keep


def _view_region_weight(view_name: str, pts_3d: np.ndarray) -> np.ndarray:
    """Softly blend view regions while keeping the front view dominant."""
    x = pts_3d[:, 0].astype(np.float32)
    x_extent = max(float(np.max(np.abs(x))), 1e-6)
    x_norm = x / x_extent
    abs_x = np.abs(x_norm)

    if view_name == "front":
        center = 1.0 - np.clip((abs_x - 0.20) / 0.35, 0.0, 1.0)
        edge_fade = 1.0 - 0.65 * np.clip((abs_x - 0.35) / 0.45, 0.0, 1.0)
        return (1.6 + 2.4 * center) * edge_fade

    if view_name == "left":
        side_preference = np.clip((-x_norm + 0.15) / 0.55, 0.0, 1.0)
    elif view_name == "right":
        side_preference = np.clip((x_norm + 0.15) / 0.55, 0.0, 1.0)
    else:
        side_preference = np.ones_like(x_norm, dtype=np.float32)
    edge_boost = np.clip((abs_x - 0.10) / 0.45, 0.0, 1.0)
    return 0.18 + 1.25 * edge_boost * side_preference


def _masked_color_stats(image: np.ndarray, mask: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if mask is None:
        pixels = image.reshape(-1, 3)
    else:
        pixels = image[mask > 127]
    if len(pixels) < 100:
        return None
    brightness = pixels.mean(axis=1)
    keep = brightness > 25.0
    if keep.sum() >= 100:
        pixels = pixels[keep]
    pixels = pixels.astype(np.float32)
    return pixels.mean(axis=0), pixels.std(axis=0) + 1e-6


def _match_color_stats(
    colors: np.ndarray,
    src_stats: Optional[Tuple[np.ndarray, np.ndarray]],
    ref_stats: Optional[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if src_stats is None or ref_stats is None or len(colors) == 0:
        return colors
    src_mean, src_std = src_stats
    ref_mean, ref_std = ref_stats
    matched = (colors.astype(np.float32) - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(matched, 0.0, 255.0)


def _rgb_to_lab_float(colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return colors.astype(np.float32)
    rgb = np.clip(colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def _lab_to_rgb_float(colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return colors.astype(np.float32)
    lab = np.clip(colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)


def _robust_color_transform_lab(
    side_colors: np.ndarray,
    front_colors: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    if len(side_colors) < 500:
        return None

    side_rgb = side_colors.astype(np.float32)
    front_rgb = front_colors.astype(np.float32)
    side_brightness = side_rgb.mean(axis=1)
    front_brightness = front_rgb.mean(axis=1)
    rgb_delta = np.linalg.norm(side_rgb - front_rgb, axis=1)
    keep = (
        (side_brightness > 35.0) & (side_brightness < 245.0) &
        (front_brightness > 35.0) & (front_brightness < 245.0) &
        (rgb_delta < 95.0)
    )
    if keep.sum() < 500:
        keep = (
            (side_brightness > 25.0) & (side_brightness < 250.0) &
            (front_brightness > 25.0) & (front_brightness < 250.0)
        )
    if keep.sum() < 500:
        return None

    side_lab = _rgb_to_lab_float(side_rgb[keep])
    front_lab = _rgb_to_lab_float(front_rgb[keep])

    side_mean = np.median(side_lab, axis=0)
    front_mean = np.median(front_lab, axis=0)
    side_std = np.percentile(side_lab, 75, axis=0) - np.percentile(side_lab, 25, axis=0)
    front_std = np.percentile(front_lab, 75, axis=0) - np.percentile(front_lab, 25, axis=0)
    # Keep this as a gentle white-balance/exposure correction. The overlap can
    # include brows, hair, shadows, or slightly mismatched anatomy, so a strong
    # affine match easily creates yellow/orange side patches.
    raw_scale = np.clip(front_std / np.maximum(side_std, 1.0), 0.90, 1.10)
    scale = 1.0 + (raw_scale - 1.0) * 0.45
    bias_limits = np.array([10.0, 5.0, 5.0], dtype=np.float32)
    raw_bias = front_mean - side_mean * scale
    bias = np.clip(raw_bias, -bias_limits, bias_limits) * 0.45
    scale[1:] = 1.0
    bias[1:] = 0.0
    return scale.astype(np.float32), bias.astype(np.float32), int(keep.sum())


def _apply_lab_transform(colors: np.ndarray, transform: Optional[Tuple[np.ndarray, np.ndarray, int]]) -> np.ndarray:
    if transform is None or len(colors) == 0:
        return colors.astype(np.float32)
    scale, bias, _ = transform
    lab = _rgb_to_lab_float(colors)
    matched = lab * scale[None, :] + bias[None, :]
    return np.clip(_lab_to_rgb_float(matched), 0.0, 255.0)


def bake_texture(
    vertices: np.ndarray,       # (N, 3)
    faces: np.ndarray,          # (F, 3)
    uv_verts: np.ndarray,       # (T, 2)
    uv_faces: np.ndarray,       # (F, 3)
    tri_map: np.ndarray,        # (H, W) int32
    bary_map: np.ndarray,       # (H, W, 3)
    cameras: Dict[str, dict],
    images: Dict[str, np.ndarray],
    tex_size: int = 2048,
    face_masks: Optional[Dict[str, np.ndarray]] = None,  # {view: (H,W) uint8 mask 0/255}
    force_front_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将3张照片的颜色烘焙到 UV 纹理图。

    Returns: (H, W, 3) uint8 RGB 纹理图
    """
    H = W = tex_size
    face_normals = compute_face_normals(vertices, faces)  # (F, 3)

    # 有效像素的面片索引和重心坐标
    valid_mask = tri_map >= 0
    valid_y, valid_x = np.where(valid_mask)
    valid_tri  = tri_map[valid_y, valid_x]    # (M,)
    valid_bary = bary_map[valid_y, valid_x]   # (M, 3)

    # 计算每个有效纹理像素对应的3D坐标
    g_faces = faces[valid_tri]                # (M, 3) 几何顶点索引
    v0 = vertices[g_faces[:, 0]]              # (M, 3)
    v1 = vertices[g_faces[:, 1]]
    v2 = vertices[g_faces[:, 2]]
    pts_3d = (valid_bary[:, 0:1] * v0 +
              valid_bary[:, 1:2] * v1 +
              valid_bary[:, 2:3] * v2)        # (M, 3)

    # 对应的面法线
    pt_normals = face_normals[valid_tri]      # (M, 3)

    # 累积颜色与权重
    color_acc  = np.zeros((len(valid_y), 3), dtype=np.float64)
    weight_acc = np.zeros(len(valid_y), dtype=np.float64)
    x_coords = pts_3d[:, 0].astype(np.float32)
    x_extent = max(float(np.max(np.abs(x_coords))), 1e-6)
    center_face = np.abs(x_coords / x_extent) < 0.35
    internal_uv_holes = _small_internal_invalid_uv_holes(valid_mask, max_area=20000)
    uv_hole_neighborhood_img = cv2.dilate(
        internal_uv_holes.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=16,
    ).astype(bool) & valid_mask
    uv_hole_neighborhood = uv_hole_neighborhood_img[valid_y, valid_x]
    force_front = force_front_mask[valid_y, valid_x] if force_front_mask is not None else np.zeros(len(valid_y), dtype=bool)

    view_names = list(cameras.keys())
    ref_stats = _masked_color_stats(
        images["front"], face_masks.get("front") if face_masks is not None else None
    ) if "front" in images else None
    color_stats = {
        name: _masked_color_stats(
            images[name], face_masks.get(name) if face_masks is not None else None
        )
        for name in view_names
        if name in images
    }
    logger.info(f"  对 {len(view_names)} 个视角进行颜色采样（{len(valid_y)} 个有效纹理像素）...")

    view_samples = {}
    for view_name in view_names:
        cam    = cameras[view_name]
        K, R, t = cam["K"], cam["R"], cam["t"]
        image  = images[view_name]              # (H_img, W_img, 3) RGB
        H_img, W_img = image.shape[:2]
        depth_map = _render_camera_depth(vertices, faces, K, R, t, (H_img, W_img))
        v_cam, z, proj, front = project_texture_points_to_image(pts_3d, K, R, t)

        # 在图像范围内的点
        in_img = (front &
                  (proj[:, 0] >= 0) & (proj[:, 0] < W_img - 1) &
                  (proj[:, 1] >= 0) & (proj[:, 1] < H_img - 1))

        # 计算权重：面法线 · 相机方向（使用 Y 翻转后的坐标系）
        # 还原回 FLAME 坐标系（翻转 Y）
        cam_center = camera_center_for_texture_visibility(R, t)
        view_dirs  = cam_center - pts_3d        # (M, 3) 点→相机方向
        norms_vd   = np.linalg.norm(view_dirs, axis=1, keepdims=True)
        view_dirs  /= np.clip(norms_vd, 1e-8, None)
        cosines    = np.sum(pt_normals * view_dirs, axis=1)  # (M,)
        weights    = np.maximum(0, cosines) ** 2               # 余弦平方：强调正对视角，减少多视图混影

        # 如有 face mask，检查投影点是否落在 mask 内
        if face_masks is not None and view_name in face_masks:
            mask_img = face_masks[view_name]   # (H_img, W_img) uint8
            mask_H, mask_W = mask_img.shape[:2]
            # 采样 mask 值（最近邻）
            px_u_int = proj[:, 0].astype(int)
            px_v_int = proj[:, 1].astype(int)
            px_u_int = np.clip(px_u_int, 0, mask_W - 1)
            px_v_int = np.clip(px_v_int, 0, mask_H - 1)
            in_mask = mask_img[px_v_int, px_u_int] > 127
            valid_pts = in_img & (weights > 0.05) & in_mask
        else:
            valid_pts = in_img & (weights > 0.05)

        effective_weights = weights.copy()
        if valid_pts.any():
            px_u_int = np.clip(proj[:, 0].astype(int), 0, W_img - 1)
            px_v_int = np.clip(proj[:, 1].astype(int), 0, H_img - 1)
            z_ref = depth_map[px_v_int, px_u_int]
            visible = z <= (z_ref + 3e-3)
            if view_name == "front":
                # FLAME's UV atlas contains small internal holes/seams. Faces around
                # those seams can have inward normals even when they project to skin,
                # so keep normal as a soft weight instead of a hard reject there.
                center_fallback = (uv_hole_neighborhood | force_front) & center_face & in_img & (np.abs(cosines) > 0.05)
                if face_masks is not None and view_name in face_masks:
                    center_fallback &= in_mask
                effective_weights[center_fallback] = np.maximum(
                    effective_weights[center_fallback],
                    np.abs(cosines[center_fallback]) ** 2,
                )
                valid_pts = (valid_pts & visible) | center_fallback
            else:
                valid_pts &= visible & ~force_front

        if valid_pts.sum() == 0:
            continue

        colors = _bilinear_sample(
            image,
            proj[valid_pts, 0],
            proj[valid_pts, 1],
        )  # (K, 3)

        # 额外过滤极暗像素（残余背景）
        vp_idx = np.where(valid_pts)[0]
        view_weights = _view_region_weight(view_name, pts_3d)[vp_idx]
        sample_weights = effective_weights[vp_idx] * view_weights

        view_samples[view_name] = {
            "idx": vp_idx,
            "colors": colors.astype(np.float32),
            "weights": sample_weights.astype(np.float32),
        }
        fg_mask = np.ones(len(vp_idx), dtype=bool)

        logger.info(f"    [{view_name}] 采样 {fg_mask.sum()} 个前景像素（共{valid_pts.sum()}有效）")

    # 归一化
    front_present = None
    front_colors_full = None
    front_weights_full = None
    if "front" in view_samples:
        front_present = np.zeros(len(valid_y), dtype=bool)
        front_colors_full = np.zeros((len(valid_y), 3), dtype=np.float32)
        front_weights_full = np.zeros(len(valid_y), dtype=np.float32)
        front_idx = view_samples["front"]["idx"]
        front_present[front_idx] = True
        front_colors_full[front_idx] = view_samples["front"]["colors"]
        front_weights_full[front_idx] = view_samples["front"]["weights"]

    for view_name in view_names:
        sample = view_samples.get(view_name)
        if sample is None:
            continue
        vp_idx = sample["idx"]
        colors = sample["colors"]
        sample_weights = sample["weights"]

        local_transform = None
        if view_name != "front" and front_present is not None:
            overlap = (
                front_present[vp_idx] &
                (sample_weights > 1e-4) &
                (front_weights_full[vp_idx] > 1e-4)
            )
            if overlap.any():
                side_w = sample_weights[overlap]
                front_w = front_weights_full[vp_idx[overlap]]
                blend_ratio = side_w / np.maximum(side_w + front_w, 1e-6)
                seam_overlap = overlap.copy()
                overlap_positions = np.where(overlap)[0]
                seam_positions = overlap_positions[(blend_ratio > 0.08) & (blend_ratio < 0.92)]
                if len(seam_positions) >= 500:
                    seam_overlap[:] = False
                    seam_overlap[seam_positions] = True

                local_transform = _robust_color_transform_lab(
                    colors[seam_overlap],
                    front_colors_full[vp_idx[seam_overlap]],
                )
                if local_transform is not None:
                    scale, bias, n_used = local_transform
                    logger.info(
                        f"    [{view_name}] local LAB match to front: "
                        f"n={n_used}, scale={scale.round(3).tolist()}, bias={bias.round(2).tolist()}"
                    )
                else:
                    logger.info(f"    [{view_name}] local LAB match skipped: overlap={int(overlap.sum())}")

        if local_transform is not None:
            matched_colors = _apply_lab_transform(colors, local_transform)
        else:
            matched_colors = _match_color_stats(colors, color_stats.get(view_name), ref_stats)

        color_acc[vp_idx]  += matched_colors * sample_weights[:, None]
        weight_acc[vp_idx] += sample_weights

    has_color = weight_acc > 0
    texture = np.zeros((H, W, 3), dtype=np.float32)
    texture[valid_y[has_color], valid_x[has_color]] = (
        color_acc[has_color] / weight_acc[has_color, None]
    )

    # 对无颜色的有效区域做 inpainting 填充（遮挡区域）
    texture_uint8 = texture.clip(0, 255).astype(np.uint8)
    has_color_img = weight_acc_img(texture_uint8, valid_y, valid_x, has_color, H, W)
    missing_valid_mask = (valid_mask & ~has_color_img).astype(np.uint8)
    if missing_valid_mask.any():
        # 用有效像素的中位肤色预填充空洞，避免 TELEA 将边界污染色向内扩散
        sampled_colors = texture_uint8[valid_y[has_color], valid_x[has_color]]  # (K, 3)
        if len(sampled_colors) > 0:
            median_skin = np.median(sampled_colors, axis=0).astype(np.uint8)
        else:
            median_skin = np.array([180, 140, 120], dtype=np.uint8)
        hole_y, hole_x = np.where(missing_valid_mask)
        texture_uint8[hole_y, hole_x] = median_skin
        # 小半径 inpaint 仅用于平滑预填充边界（不再需要跨越大距离）
        texture_uint8 = cv2.inpaint(texture_uint8, missing_valid_mask * 255, 3, cv2.INPAINT_TELEA)
    if internal_uv_holes.sum() > 8:
        texture_uint8 = cv2.inpaint(texture_uint8, internal_uv_holes.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)

    return texture_uint8


def _small_internal_invalid_uv_holes(valid_mask: np.ndarray, max_area: int = 20000) -> np.ndarray:
    """Find small enclosed UV holes; keep the exterior transparent/black."""
    invalid = (~valid_mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
    holes = np.zeros_like(valid_mask, dtype=bool)
    h, w = valid_mask.shape
    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + comp_w >= w or y + comp_h >= h
        if not touches_border and area <= max_area:
            holes[labels == label] = True
    return holes


def weight_acc_img(texture, vy, vx, has_color, H, W):
    """生成有有效颜色的像素 bool 图"""
    m = np.zeros((H, W), dtype=bool)
    m[vy[has_color], vx[has_color]] = True
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 单视角纹理烘焙（使用预融合统一纹理）
# ══════════════════════════════════════════════════════════════════════════════

def bake_texture_single(
    vertices: np.ndarray,       # (N, 3)
    faces: np.ndarray,          # (F, 3)
    uv_verts: np.ndarray,       # (T, 2)
    uv_faces: np.ndarray,       # (F, 3)
    tri_map: np.ndarray,        # (H, W) int32
    bary_map: np.ndarray,       # (H, W, 3)
    camera: dict,               # {"K": ..., "R": ..., "t": ...}
    unified_img: np.ndarray,    # (Hi, Wi, 3) uint8 RGB 预融合统一纹理
    tex_size: int = 2048,
    face_mask: Optional[np.ndarray] = None,  # (Hi, Wi) uint8, 0=背景 255=人脸
) -> np.ndarray:
    """
    从单张预融合统一纹理图烘焙到 UV 纹理图。
    每个 UV 纹素通过重心插值得到 3D 点，再用正面相机投影到统一纹理图采色。
    face_mask 用于排除背景/支架等干扰区域。

    Returns: (tex_size, tex_size, 3) uint8 RGB 纹理图
    """
    H = W = tex_size
    H_img, W_img = unified_img.shape[:2]

    valid_mask = tri_map >= 0
    valid_y, valid_x = np.where(valid_mask)

    if len(valid_y) == 0:
        logger.warning("UV 光栅化结果为空，无有效纹理像素")
        return np.zeros((H, W, 3), dtype=np.uint8)

    valid_tri  = tri_map[valid_y, valid_x]    # (M,)
    valid_bary = bary_map[valid_y, valid_x]   # (M, 3)

    # 重心插值 → 3D 坐标
    g_faces = faces[valid_tri]
    v0 = vertices[g_faces[:, 0]]
    v1 = vertices[g_faces[:, 1]]
    v2 = vertices[g_faces[:, 2]]
    pts_3d = (valid_bary[:, 0:1] * v0 +
              valid_bary[:, 1:2] * v1 +
              valid_bary[:, 2:3] * v2)  # (M, 3)

    K, R, t = camera["K"], camera["R"], camera["t"]

    # ── 法线可见性剔除（排除后脑、侧面背对相机的面片）────────────────
    face_normals_all = compute_face_normals(vertices, faces)  # (F, 3)
    pt_normals = face_normals_all[valid_tri]                  # (M, 3)

    # 相机中心在 FLAME 坐标系（还原 Y 翻转）

    # 视线方向（点 → 相机），在 FLAME 坐标系中计算
    cam_center = camera_center_for_texture_visibility(R, t)
    view_dirs = cam_center - pts_3d  # (M, 3)
    view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
    cosines = np.sum(pt_normals * view_dirs, axis=1)          # (M,)
    front_facing = cosines > 0.05   # 法线朝向相机（小阈值留擦边余量）

    logger.info(
        f"  法线可见性: {front_facing.sum()} / {len(valid_tri)} 面片朝向相机 "
        f"({front_facing.mean()*100:.1f}%)"
    )
    v_cam, z, proj, front = project_texture_points_to_image(pts_3d, K, R, t)

    in_img = (front &
              front_facing &
              (proj[:, 0] >= 0) & (proj[:, 0] < W_img - 1) &
              (proj[:, 1] >= 0) & (proj[:, 1] < H_img - 1))

    # face_mask 排除背景/支架：投影落点必须在 mask 内（人脸区域）
    if face_mask is not None:
        px_u = np.clip(proj[:, 0].astype(int), 0, face_mask.shape[1] - 1)
        px_v = np.clip(proj[:, 1].astype(int), 0, face_mask.shape[0] - 1)
        in_face = face_mask[px_v, px_u] > 127
        before = in_img.sum()
        in_img = in_img & in_face
        logger.info(f"  face_mask 过滤: {before} → {in_img.sum()} 像素（排除 {before - in_img.sum()} 背景点）")

    texture = np.zeros((H, W, 3), dtype=np.uint8)

    if in_img.sum() > 0:
        colors = _bilinear_sample(unified_img, proj[in_img, 0], proj[in_img, 1])
        colors_u8 = colors.clip(0, 255).astype(np.uint8)

        # 过滤极暗像素（暗背景/黑色区域被错误投影进来）
        brightness = colors_u8.mean(axis=1)
        bright_enough = brightness > 20.0
        in_img_indices = np.where(in_img)[0]
        good_indices   = in_img_indices[bright_enough]

        texture[valid_y[good_indices], valid_x[good_indices]] = colors_u8[bright_enough]
        logger.info(
            f"  单视角烘焙: {len(good_indices)} / {len(valid_y)} 个有效 UV 像素已着色"
            f"（过滤暗像素 {in_img.sum() - len(good_indices)} 个）"
        )
    else:
        good_indices = np.array([], dtype=int)

    # Inpainting：填充未能投影到的有效 UV 区域（法线背面、越界、遮挡、暗像素）
    has_color = np.zeros((H, W), dtype=bool)
    if len(good_indices) > 0:
        has_color[valid_y[good_indices], valid_x[good_indices]] = True
    inpaint_needed = valid_mask & ~has_color
    if inpaint_needed.any():
        n_holes = int(inpaint_needed.sum())
        logger.info(f"  Inpainting 填充 {n_holes} 个空洞像素（多尺度）...")
        inpaint_mask_u8 = inpaint_needed.astype(np.uint8) * 255

        # Step 1: 计算有效皮肤区域的平均颜色
        skin_pixels = texture[has_color]
        if len(skin_pixels) > 0:
            avg_skin = np.median(skin_pixels, axis=0).astype(np.uint8)
        else:
            avg_skin = np.array([200, 170, 150], dtype=np.uint8)
        logger.info(f"  平均肤色: RGB={avg_skin}")

        # Step 2: 关键——将 mesh 外背景临时填为平均肤色
        # 防止 TELEA 从纯黑背景取色覆盖后脑等边界区域
        bg_mask = ~valid_mask
        texture[bg_mask] = avg_skin
        texture[inpaint_needed] = avg_skin  # 空洞也预填肤色，提供传播起点

        # Step 3: 缩小到 256×256 做多尺度 TELEA（传播大范围颜色梯度）
        SCALE = 8
        tex_s    = cv2.resize(texture,        (W // SCALE, H // SCALE), interpolation=cv2.INTER_AREA)
        mask_s   = cv2.resize(inpaint_mask_u8,(W // SCALE, H // SCALE), interpolation=cv2.INTER_NEAREST)
        tex_s_ip = cv2.inpaint(tex_s, mask_s, inpaintRadius=20, flags=cv2.INPAINT_TELEA)

        # Step 4: 上采样回 2048，替换空洞区域（保留直采区域不动）
        tex_seed = cv2.resize(tex_s_ip, (W, H), interpolation=cv2.INTER_LINEAR)
        texture[inpaint_needed] = tex_seed[inpaint_needed]

        # Step 5: 原尺寸细化（修复边界锯齿）
        texture = cv2.inpaint(texture, inpaint_mask_u8, inpaintRadius=8, flags=cv2.INPAINT_TELEA)

        # Step 6: 还原 mesh 外背景为黑色（UI 透明显示需要）
        texture[bg_mask] = 0
        logger.info("  Inpainting 完成")

    return texture


# ══════════════════════════════════════════════════════════════════════════════
# 泊松接缝修复
# ══════════════════════════════════════════════════════════════════════════════

def poisson_seam_fix(
    texture: np.ndarray,      # (H, W, 3) uint8
    valid_mask: np.ndarray,   # (H, W) bool — 有效纹理区域
) -> np.ndarray:
    """
    在纹理有效区域内做拉普拉斯平滑，消除视角切换处的色差。
    使用 OpenCV 的 inpaint 对边界过渡区做平滑处理。
    """
    # 检测颜色跳变边界：与相邻像素色差大的区域
    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 计算局部色差
    diff_x = np.abs(np.roll(lab, 1, axis=1) - lab).sum(axis=2)
    diff_y = np.abs(np.roll(lab, 1, axis=0) - lab).sum(axis=2)
    seam_mask = ((diff_x + diff_y) > 30) & valid_mask

    # 膨胀接缝区域后做 inpaint
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seam_dilated = cv2.dilate(seam_mask.astype(np.uint8) * 255, kernel)
    seam_dilated = (seam_dilated > 0) & valid_mask

    if seam_dilated.any():
        result = cv2.inpaint(
            texture,
            seam_dilated.astype(np.uint8) * 255,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA,
        )
        n_seam = seam_dilated.sum()
        logger.info(f"  泊松接缝修复: {n_seam} 个接缝像素已修复")
        return result
    return texture


# ══════════════════════════════════════════════════════════════════════════════
# GLB 打包
# ══════════════════════════════════════════════════════════════════════════════

def poisson_seam_fix_local(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    repair_roi: np.ndarray,
) -> np.ndarray:
    """Apply seam smoothing only inside a caller-provided repair region."""
    roi = repair_roi & valid_mask
    if not roi.any():
        return texture
    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff_x = np.abs(np.roll(lab, 1, axis=1) - lab).sum(axis=2)
    diff_y = np.abs(np.roll(lab, 1, axis=0) - lab).sum(axis=2)
    seam_mask = ((diff_x + diff_y) > 30) & roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seam_dilated = cv2.dilate(seam_mask.astype(np.uint8) * 255, kernel)
    seam_dilated = (seam_dilated > 0) & roi
    if not seam_dilated.any():
        return texture
    result = cv2.inpaint(texture, seam_dilated.astype(np.uint8) * 255, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    logger.info(f"  Local seam repair: {int(seam_dilated.sum())} pixels")
    return result


def export_glb(
    vertices: np.ndarray,       # (N, 3)
    faces: np.ndarray,          # (F, 3)
    uv_verts: np.ndarray,       # (T, 2)
    uv_faces: np.ndarray,       # (F, 3)
    texture: np.ndarray,        # (H, W, 3) uint8 RGB
    output_path: Path,
    lighting_type: str = "white",
    lighting_display_name: str = "白光",
):
    """
    将 Mesh + 纹理打包为 GLB 文件。
    使用 trimesh 导出，extras 元数据携带图层信息（用于前端图层切换）。
    """
    try:
        import trimesh
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("请安装 trimesh 和 Pillow: pip install trimesh pillow")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    geom_used, geom_inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[geom_used]
    faces = geom_inverse.reshape(faces.shape).astype(np.int32)

    uv_used, uv_inverse = np.unique(uv_faces.reshape(-1), return_inverse=True)
    uv_verts = uv_verts[uv_used]
    uv_faces = uv_inverse.reshape(uv_faces.shape).astype(np.int32)

    # module2 已完成 Loop Subdivision（~160K faces）+ Laplacian，此处仅做最终精修
    # ── Step 1：轻度 Laplacian 平滑（2次，修复 per-face-vertex 展开前的微小锯齿）
    shared_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(shared_mesh, iterations=2, lamb=0.3)
    smooth_verts = np.array(shared_mesh.vertices)

    # ── Step 2：展开为 per-face-vertex（UV 必须独立寻址）────────────────────
    flat_geom_idx = faces.flatten()
    flat_uv_idx   = uv_faces.flatten()
    exp_verts = smooth_verts[flat_geom_idx]
    exp_uv    = image_uv_to_obj_uv(uv_verts[flat_uv_idx])
    exp_faces = np.arange(len(exp_verts)).reshape(-1, 3)

    # ── Step 3：重算平滑顶点法线 ────────────────────────────────────────────
    sub_mesh    = trimesh.Trimesh(vertices=exp_verts, faces=exp_faces, process=False)
    exp_normals = np.array(sub_mesh.vertex_normals)

    # 创建带法线的 trimesh Mesh
    mesh = trimesh.Trimesh(
        vertices=exp_verts,
        faces=exp_faces,
        vertex_normals=exp_normals,
        process=False,
    )

    # 创建材质
    tex_pil = PILImage.fromarray(texture)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_pil,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        name=f"skin_{lighting_type}",
    )

    # 绑定 UV
    visual = trimesh.visual.texture.TextureVisuals(
        uv=exp_uv,
        material=material,
    )
    mesh.visual = visual

    # GLB extras 元数据（前端图层切换用）
    scene = trimesh.Scene(
        geometry={f"face_{lighting_type}": mesh},
        metadata={
            "face3d_layers": [
                {
                    "name": lighting_display_name,
                    "type": lighting_type,
                    "mesh_name": f"face_{lighting_type}",
                }
            ]
        },
    )

    # 尝试 Draco 压缩（减少文件大小 5-10x），失败时回退标准 GLB
    glb_bytes = None
    try:
        import subprocess, tempfile, os
        # trimesh >= 4.x 支持通过 draco 命令行压缩
        std_bytes = scene.export(file_type="glb")
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp_in:
            tmp_in.write(std_bytes)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path.replace(".glb", "_draco.glb")
        result = subprocess.run(
            ["draco_encoder", "-i", tmp_in_path, "-o", tmp_out_path,
             "-qp", "11", "-qt", "10"],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(tmp_out_path):
            with open(tmp_out_path, "rb") as df:
                glb_bytes = df.read()
            logger.info(f"Draco 压缩成功: {len(std_bytes)/1024/1024:.1f} MB → {len(glb_bytes)/1024/1024:.1f} MB")
        os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)
    except Exception as _draco_err:
        logger.debug(f"Draco 压缩不可用（{_draco_err}），使用标准 GLB")

    if glb_bytes is None:
        glb_bytes = scene.export(file_type="glb")

    with open(str(output_path), "wb") as f:
        f.write(glb_bytes)

    size_mb = len(glb_bytes) / 1024 / 1024
    logger.info(f"GLB 已导出: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def run_texture_pipeline(
    mesh_dir: Path,
    images: Dict[str, np.ndarray],
    output_texture_dir: Path,
    output_mesh_dir: Path,
    tex_size: int = 2048,
    lighting_type: str = "white",
    lighting_display_name: str = "白光",
    face_masks: Optional[Dict[str, np.ndarray]] = None,
    unified_texture: Optional[np.ndarray] = None,       # 预融合统一纹理（fallback）
    hires_front_image: Optional[np.ndarray] = None,     # 原始高清正面图（仅正面，旧接口）
    hires_scale_factor: float = 1.0,                     # K 缩放系数（原始分辨率 / 512）
    hires_images: Optional[Dict[str, np.ndarray]] = None,  # 多视角高清图（最优先）

    working_image_size: Optional[int] = None,
) -> Path:
    """
    完整纹理融合流程。

    Args:
        mesh_dir: 包含 face_mesh.obj 和 cameras.json 的目录
        images:   {view_name: RGB numpy array}
        output_texture_dir: 纹理输出目录
        output_mesh_dir:    GLB 输出目录
        hires_front_image:  原始高清正面照片（优先于 unified_texture）
        hires_scale_factor: 原始分辨率 / 512，用于缩放 K 矩阵

    Returns:
        GLB 文件路径
    """
    output_texture_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    if working_image_size is None:
        try:
            from src import config as cfg
            working_image_size = int(getattr(cfg, "WORK_IMAGE_SIZE", 512))
        except Exception:
            working_image_size = 512

    # ── 加载 Mesh 和相机 ──────────────────────────────────────────────────
    logger.info("加载 Mesh 和相机参数...")
    vertices, faces, uv_verts, uv_faces = load_mesh_obj(mesh_dir / "face_mesh.obj")
    cameras = load_cameras(mesh_dir / "cameras.json")
    try:
        from src import config as cfg
        enable_visible_face_crop = bool(getattr(cfg, "ENABLE_VISIBLE_FACE_CROP", False))
        visible_face_crop_rings = int(getattr(cfg, "VISIBLE_FACE_CROP_DILATE_RINGS", 2))
        visible_face_crop_z_tol = float(getattr(cfg, "VISIBLE_FACE_CROP_Z_TOL", 0.006))
    except Exception:
        enable_visible_face_crop = False
        visible_face_crop_rings = 2
        visible_face_crop_z_tol = 0.006
    if enable_visible_face_crop:
        crop_cameras = cameras
        crop_images = images
        crop_masks = face_masks
        if hires_images is not None:
            crop_cameras = {}
            crop_images = {}
            crop_masks = {} if face_masks is not None else None
            for view_name, hires_img in hires_images.items():
                if view_name not in cameras:
                    continue
                h, w = hires_img.shape[:2]
                max_sz = max(h, w)
                canvas = np.zeros((max_sz, max_sz, 3), dtype=np.uint8)
                y_off = (max_sz - h) // 2
                x_off = (max_sz - w) // 2
                canvas[y_off:y_off + h, x_off:x_off + w] = hires_img
                crop_images[view_name] = canvas

                view_sf = max_sz / float(working_image_size)
                cam = cameras[view_name]
                k_crop = cam["K"].copy()
                k_crop[0, :] *= view_sf
                k_crop[1, :] *= view_sf
                crop_cameras[view_name] = {"K": k_crop, "R": cam["R"], "t": cam["t"]}

                if crop_masks is not None and view_name in face_masks:
                    crop_masks[view_name] = cv2.resize(
                        face_masks[view_name],
                        (max_sz, max_sz),
                        interpolation=cv2.INTER_NEAREST,
                    )
        faces, uv_faces, _ = crop_mesh_to_visible_face(
            vertices,
            faces,
            uv_verts,
            uv_faces,
            crop_cameras,
            crop_images,
            crop_masks,
            output_texture_dir.parent / "debug" / "visible_face_crop",
            dilate_rings=visible_face_crop_rings,
            z_tol=visible_face_crop_z_tol,
        )
    try:
        from src import config as cfg
        enable_uv_hole_fill_faces = bool(getattr(cfg, "ENABLE_UV_HOLE_FILL_FACES", False))
    except Exception:
        enable_uv_hole_fill_faces = False
    if enable_uv_hole_fill_faces:
        bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces, added_uv_faces = add_uv_hole_fill_faces_for_bake(
            vertices, faces, uv_verts, uv_faces, tex_size
        )
        if added_uv_faces:
            logger.info(f"  UV hole fill bake mesh: added {added_uv_faces} temporary faces")
    else:
        bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces = vertices, faces, uv_verts, uv_faces
        added_uv_faces = 0
        logger.info("  UV hole fill bake mesh disabled; internal holes will use texture inpaint")
    logger.info(f"  Mesh: {len(vertices)} 顶点, {len(faces)} 面片, {len(uv_verts)} UV点")

    # ── UV 光栅化 ─────────────────────────────────────────────────────────
    logger.info("UV 光栅化...")
    tri_map, bary_map = rasterize_uv_map(bake_uv_verts, bake_uv_faces, tex_size)
    valid_mask = tri_map >= 0
    filled_uv_holes = (tri_map >= len(uv_faces)) if added_uv_faces else _small_internal_invalid_uv_holes(valid_mask)
    uv_hole_repair_roi = cv2.dilate(
        filled_uv_holes.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=4,
    ).astype(bool) & valid_mask

    # ── 纹理烘焙（高清直采 > 统一纹理 > 多视角） ─────────────────────────
    if hires_images is not None:
        # ── 多视角高清烘焙（覆盖全部 UV，包括侧面耳朵等区域）──────────
        logger.info(f"多视角高清烘焙模式（{len(hires_images)} 个视角）...")
        scaled_cameras: Dict[str, dict] = {}
        processed_hires: Dict[str, np.ndarray] = {}
        scaled_masks: Dict[str, np.ndarray] = {}

        for view_name, hires_img in hires_images.items():
            if view_name not in cameras:
                logger.warning(f"  [{view_name}] 无对应相机，跳过")
                continue

            # 等比缩放 + 居中填充到正方形（与 _resize_to_target 逻辑一致）
            h, w = hires_img.shape[:2]
            max_sz = max(h, w)
            new_w, new_h = w, h  # max side already == max_sz, no scaling needed
            canvas = np.zeros((max_sz, max_sz, 3), dtype=np.uint8)
            y_off = (max_sz - new_h) // 2
            x_off = (max_sz - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = hires_img
            processed_hires[view_name] = canvas

            # 缩放 K 矩阵匹配高清正方形分辨率
            view_sf = max_sz / float(working_image_size)
            cam = cameras[view_name]
            K_h = cam["K"].copy()
            K_h[0, :] *= view_sf   # fx, cx
            K_h[1, :] *= view_sf   # fy, cy
            scaled_cameras[view_name] = {"K": K_h, "R": cam["R"], "t": cam["t"]}

            # 缩放 face_mask 到高清分辨率（INTER_NEAREST 保持硬边界）
            if face_masks is not None and view_name in face_masks:
                scaled_masks[view_name] = cv2.resize(
                    face_masks[view_name], (max_sz, max_sz),
                    interpolation=cv2.INTER_NEAREST,
                )

            logger.info(
                f"  [{view_name}] {w}×{h} → {max_sz}×{max_sz}, K_scale={view_sf:.3f}"
            )

        texture = bake_texture(
            bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces,
            tri_map, bary_map,
            scaled_cameras,
            processed_hires,
            tex_size,
            face_masks=scaled_masks if scaled_masks else None,
            force_front_mask=uv_hole_repair_roi,
        )
        logger.info("接缝修复...")
        if added_uv_faces:
            debug_dir = output_texture_dir.parent / "debug" / "uv_hole_fill"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "before_poisson.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        texture = poisson_seam_fix_local(texture, valid_mask, uv_hole_repair_roi)
        if added_uv_faces:
            cv2.imwrite(str(debug_dir / "after_poisson.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

    elif hires_front_image is not None:
        # ── 高清直采模式（绕过 512×512 分辨率瓶颈）────────────────────
        logger.info(f"高清直采模式 (scale={hires_scale_factor:.3f})...")
        front_cam = cameras.get("front", next(iter(cameras.values())))

        # 缩放 K 矩阵匹配高清图分辨率
        K_hires = front_cam["K"].copy()
        K_hires[0, :] *= hires_scale_factor  # fx, cx
        K_hires[1, :] *= hires_scale_factor  # fy, cy
        hires_cam = {"K": K_hires, "R": front_cam["R"], "t": front_cam["t"]}

        # 等比缩放 + 居中黑边填充到正方形（与 _resize_to_target 逻辑相同）
        max_size = max(hires_front_image.shape[:2])
        h, w = hires_front_image.shape[:2]
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(hires_front_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((max_size, max_size, 3), dtype=np.uint8)
        y_off = (max_size - new_h) // 2
        x_off = (max_size - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        hires_img = canvas

        logger.info(
            f"  高清图: {hires_front_image.shape[1]}×{hires_front_image.shape[0]}"
            f" → {max_size}×{max_size} (正方形填充)"
        )

        # 将 512×512 face_mask 按相同倍率缩放到 max_size×max_size
        # 原理：K_hires = K_512 × scale，投影坐标同比放大，mask 同步放大即可对齐
        hires_face_mask = None
        if face_masks is not None and "front" in face_masks:
            mask_512 = face_masks["front"]
            hires_face_mask = cv2.resize(
                mask_512, (max_size, max_size),
                interpolation=cv2.INTER_NEAREST,
            )
            logger.info(f"  face_mask 放大: 512×512 → {max_size}×{max_size}")

        texture = bake_texture_single(
            bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces,
            tri_map, bary_map,
            hires_cam, hires_img,
            tex_size,
            face_mask=hires_face_mask,
        )
        logger.info("跳过接缝修复（单视角无拼接）")

    elif unified_texture is not None:
        # ── 预融合统一纹理 fallback ────────────────────────────────────
        logger.info("使用预融合统一纹理（单视角正面投影）...")
        front_cam = cameras.get("front", next(iter(cameras.values())))
        texture = bake_texture_single(
            bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces,
            tri_map, bary_map,
            front_cam, unified_texture,
            tex_size,
        )
        logger.info("跳过接缝修复（单视角无拼接）")

    else:
        logger.info("多视角颜色采样（烘焙）...")
        for view_name in cameras:
            if view_name not in images:
                logger.warning(f"  相机 [{view_name}] 无对应图像，跳过")
        view_images = {k: images[k] for k in cameras if k in images}
        texture = bake_texture(
            bake_vertices, bake_faces, bake_uv_verts, bake_uv_faces,
            tri_map, bary_map,
            cameras, view_images,
            tex_size,
            face_masks=face_masks,
            force_front_mask=uv_hole_repair_roi,
        )
        # ── 泊松接缝修复 ─────────────────────────────────────────────────
        logger.info("接缝修复...")
        if added_uv_faces:
            debug_dir = output_texture_dir.parent / "debug" / "uv_hole_fill"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "before_poisson.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        texture = poisson_seam_fix_local(texture, valid_mask, uv_hole_repair_roi)
        if added_uv_faces:
            cv2.imwrite(str(debug_dir / "after_poisson.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

    # ── 保存纹理图 ────────────────────────────────────────────────────────
    tex_path = output_texture_dir / f"albedo_{lighting_type}.png"
    import cv2 as _cv
    _cv.imwrite(str(tex_path), _cv.cvtColor(texture, _cv.COLOR_RGB2BGR))
    logger.info(f"纹理已保存: {tex_path}")

    # ── GLB 打包 ──────────────────────────────────────────────────────────
    logger.info("打包 GLB...")
    glb_path = output_mesh_dir / "face.glb"
    export_glb(
        vertices, faces, uv_verts, uv_faces,
        texture, glb_path,
        lighting_type=lighting_type,
        lighting_display_name=lighting_display_name,
    )

    return glb_path

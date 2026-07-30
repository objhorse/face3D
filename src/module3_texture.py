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

from src.appearance.projective_sampling import (
    assert_strict_sampling_coordinates,
    project_points_strict,
    render_camera_depth,
    sample_projected_attributes,
)
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
    return render_camera_depth(vertices, faces, K, R, t, image_shape)


def _expand_face_selection(faces: np.ndarray, keep_faces: np.ndarray, rings: int) -> np.ndarray:
    expanded = keep_faces.copy()
    for _ in range(max(0, int(rings))):
        keep_vertices = np.zeros(int(faces.max()) + 1, dtype=bool)
        keep_vertices[faces[expanded].ravel()] = True
        expanded |= np.any(keep_vertices[faces], axis=1)
    return expanded


def _filter_small_face_components(
    faces: np.ndarray,
    keep_faces: np.ndarray,
    min_component_faces: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    min_component_faces = int(max(0, min_component_faces))
    selected = np.flatnonzero(keep_faces)
    stats = {
        "enabled": min_component_faces > 1,
        "min_component_faces": min_component_faces,
        "before_faces": int(selected.size),
        "after_faces": int(selected.size),
        "removed_faces": 0,
        "component_count": 0,
        "removed_component_count": 0,
        "components": [],
    }
    if min_component_faces <= 1 or selected.size == 0:
        return keep_faces, stats

    vertex_to_local_faces: Dict[int, List[int]] = {}
    for local_idx, face_idx in enumerate(selected):
        for vertex_idx in faces[face_idx]:
            vertex_to_local_faces.setdefault(int(vertex_idx), []).append(local_idx)

    seen = np.zeros(selected.size, dtype=bool)
    component_faces: List[np.ndarray] = []
    for start in range(selected.size):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component = []
        while stack:
            local_idx = stack.pop()
            component.append(local_idx)
            for vertex_idx in faces[selected[local_idx]]:
                for neighbor in vertex_to_local_faces.get(int(vertex_idx), []):
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        stack.append(neighbor)
        component_faces.append(selected[np.asarray(component, dtype=np.int64)])

    component_faces.sort(key=len, reverse=True)
    keep_filtered = keep_faces.copy()
    keep_filtered[:] = False
    kept_components = 0
    removed_components = 0
    removed_faces = 0
    for component_id, comp in enumerate(component_faces):
        comp_size = int(len(comp))
        keep_component = comp_size >= min_component_faces
        if component_id == 0 and not keep_component:
            keep_component = True
        if keep_component:
            keep_filtered[comp] = True
            kept_components += 1
        else:
            removed_components += 1
            removed_faces += comp_size
        stats["components"].append({
            "component": int(component_id),
            "faces": comp_size,
            "kept": bool(keep_component),
        })

    stats.update({
        "after_faces": int(keep_filtered.sum()),
        "removed_faces": int(removed_faces),
        "component_count": int(len(component_faces)),
        "kept_component_count": int(kept_components),
        "removed_component_count": int(removed_components),
    })
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "visible_face_crop_components.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return keep_filtered, stats


def _trim_side_ear_faces(
    face_centers: np.ndarray,
    keep_faces: np.ndarray,
    enabled: bool,
    x_abs_min: float,
    y_min: float,
    y_max: float,
    z_max: float,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    stats = {
        "enabled": bool(enabled),
        "x_abs_min": float(x_abs_min),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_max": float(z_max),
        "removed_faces": 0,
        "before_faces": int(np.count_nonzero(keep_faces)),
        "after_faces": int(np.count_nonzero(keep_faces)),
    }
    if not enabled or len(face_centers) == 0:
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            with open(debug_dir / "visible_face_crop_side_ear_trim.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        return keep_faces, stats

    centers = np.asarray(face_centers, dtype=np.float32)
    side_ear = (
        (np.abs(centers[:, 0]) >= float(x_abs_min)) &
        (centers[:, 1] >= float(y_min)) &
        (centers[:, 1] <= float(y_max)) &
        (centers[:, 2] <= float(z_max))
    )
    remove = keep_faces & side_ear
    if not np.any(remove):
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            with open(debug_dir / "visible_face_crop_side_ear_trim.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        return keep_faces, stats

    trimmed = keep_faces.copy()
    trimmed[remove] = False
    stats.update({
        "removed_faces": int(np.count_nonzero(remove)),
        "after_faces": int(np.count_nonzero(trimmed)),
    })
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "visible_face_crop_side_ear_trim.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return trimmed, stats


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
    min_component_faces: int = 0,
    side_ear_trim: bool = False,
    side_ear_x_abs_min: float = 0.066,
    side_ear_y_min: float = -0.060,
    side_ear_y_max: float = 0.060,
    side_ear_z_max: float = 0.004,
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
    face_keep, side_ear_stats = _trim_side_ear_faces(
        face_centers,
        face_keep,
        enabled=side_ear_trim,
        x_abs_min=side_ear_x_abs_min,
        y_min=side_ear_y_min,
        y_max=side_ear_y_max,
        z_max=side_ear_z_max,
        debug_dir=debug_dir,
    )
    if side_ear_stats.get("removed_faces", 0) > 0:
        logger.info(
            "  Visible face crop side-ear trim: removed "
            f"{side_ear_stats['removed_faces']} rear-side faces "
            f"(abs(x)>={side_ear_stats['x_abs_min']:.3f}, "
            f"z<={side_ear_stats['z_max']:.3f})"
        )
    face_keep, component_stats = _filter_small_face_components(
        faces,
        face_keep,
        min_component_faces=min_component_faces,
        debug_dir=debug_dir,
    )
    if component_stats.get("removed_faces", 0) > 0:
        logger.info(
            "  Visible face crop components: removed "
            f"{component_stats['removed_faces']} faces from "
            f"{component_stats['removed_component_count']} small components "
            f"(<{component_stats['min_component_faces']} faces)"
        )
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
    keep = _overlap_skin_samples(side_rgb, front_rgb)
    if keep.sum() < 500:
        side_brightness = side_rgb.mean(axis=1)
        front_brightness = front_rgb.mean(axis=1)
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
    shift_limits = np.array([10.0, 6.0, 6.0], dtype=np.float32)
    mean_shift = np.clip(front_mean - side_mean, -shift_limits, shift_limits) * 0.60
    bias = side_mean + mean_shift - side_mean * scale
    return scale.astype(np.float32), bias.astype(np.float32), int(keep.sum())


def _overlap_skin_samples(side_rgb: np.ndarray, front_rgb: np.ndarray) -> np.ndarray:
    side = np.asarray(side_rgb, dtype=np.float32)
    front = np.asarray(front_rgb, dtype=np.float32)
    side_brightness = side.mean(axis=1)
    front_brightness = front.mean(axis=1)
    rgb_delta = np.linalg.norm(side - front, axis=1)
    return (
        (side_brightness > 35.0) & (side_brightness < 245.0) &
        (front_brightness > 35.0) & (front_brightness < 245.0) &
        (rgb_delta < 95.0)
    )


def _apply_lab_transform(colors: np.ndarray, transform: Optional[Tuple[np.ndarray, np.ndarray, int]]) -> np.ndarray:
    if transform is None or len(colors) == 0:
        return colors.astype(np.float32)
    scale, bias, _ = transform
    lab = _rgb_to_lab_float(colors)
    matched = lab * scale[None, :] + bias[None, :]
    return np.clip(_lab_to_rgb_float(matched), 0.0, 255.0)


def _multiband_blend(
    layers: List[np.ndarray],
    weights: List[np.ndarray],
    levels: int = 5,
) -> np.ndarray:
    """Blend registered view textures with Gaussian weights and Laplacian colors."""
    if len(layers) != len(weights) or not layers:
        raise ValueError("layers and weights must be non-empty and have equal length")
    if len(layers) == 1:
        return np.asarray(layers[0], dtype=np.float32)
    color_pyramids, weight_pyramids = [], []
    for layer, weight in zip(layers, weights):
        gaussian_color = [np.asarray(layer, dtype=np.float32)]
        gaussian_weight = [np.asarray(weight, dtype=np.float32)]
        for _ in range(max(1, int(levels))):
            if min(gaussian_color[-1].shape[:2]) <= 16:
                break
            gaussian_color.append(cv2.pyrDown(gaussian_color[-1]))
            gaussian_weight.append(cv2.pyrDown(gaussian_weight[-1]))
        laplacian = []
        for index in range(len(gaussian_color) - 1):
            up = cv2.pyrUp(
                gaussian_color[index + 1],
                dstsize=(gaussian_color[index].shape[1], gaussian_color[index].shape[0]),
            )
            laplacian.append(gaussian_color[index] - up)
        laplacian.append(gaussian_color[-1])
        color_pyramids.append(laplacian)
        weight_pyramids.append(gaussian_weight)

    blended_levels = []
    for level in range(len(color_pyramids[0])):
        numerator = np.zeros_like(color_pyramids[0][level], dtype=np.float32)
        denominator = np.zeros(color_pyramids[0][level].shape[:2], dtype=np.float32)
        for colors, view_weights in zip(color_pyramids, weight_pyramids):
            weight = np.maximum(view_weights[level], 0.0)
            numerator += colors[level] * weight[:, :, None]
            denominator += weight
        blended_levels.append(numerator / np.maximum(denominator[:, :, None], 1e-6))

    result = blended_levels[-1]
    for level in range(len(blended_levels) - 2, -1, -1):
        result = cv2.pyrUp(
            result,
            dstsize=(blended_levels[level].shape[1], blended_levels[level].shape[0]),
        ) + blended_levels[level]
    return result


def _feather_view_weight(weight: np.ndarray, radius_px: float = 72.0) -> np.ndarray:
    values = np.asarray(weight, dtype=np.float32)
    support = values > 1e-6
    if not support.any():
        return values.copy()
    distance = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
    feather = np.clip(distance / max(float(radius_px), 1.0), 0.0, 1.0)
    feather = feather * feather * (3.0 - 2.0 * feather)
    return values * feather


def _apply_front_feature_ownership(
    texture: np.ndarray,
    valid_y: np.ndarray,
    valid_x: np.ndarray,
    protected_features: np.ndarray,
    front_present: Optional[np.ndarray],
    front_colors: Optional[np.ndarray],
) -> Tuple[np.ndarray, dict]:
    """Restore exact front samples in protected facial-feature UV pixels."""
    result = np.asarray(texture, dtype=np.float32).copy()
    protected = np.asarray(protected_features, dtype=bool)
    if front_present is None or front_colors is None:
        return result, {
            "protected_pixels": int(protected.sum()),
            "front_owned_pixels": 0,
            "front_ownership_ratio": 0.0,
        }
    owned = protected & np.asarray(front_present, dtype=bool)
    if np.any(owned):
        result[np.asarray(valid_y)[owned], np.asarray(valid_x)[owned]] = np.asarray(front_colors)[owned]
    protected_count = int(protected.sum())
    owned_count = int(owned.sum())
    return result, {
        "protected_pixels": protected_count,
        "front_owned_pixels": owned_count,
        "front_ownership_ratio": (
            float(owned_count / protected_count) if protected_count else 1.0
        ),
    }


def _local_overlap_color_correction(
    colors: np.ndarray,
    indices: np.ndarray,
    front_present: np.ndarray,
    front_colors_full: np.ndarray,
    valid_y: np.ndarray,
    valid_x: np.ndarray,
    protected_features: np.ndarray,
    texture_shape: Tuple[int, int],
    sigma_px: float = 42.0,
    strength: float = 0.75,
    max_rgb_shift: float = 18.0,
) -> Tuple[np.ndarray, dict]:
    """Correct spatially varying low-frequency color differences in UV overlap."""
    corrected = np.asarray(colors, dtype=np.float32).copy()
    overlap = front_present[indices] & ~protected_features[indices]
    if int(overlap.sum()) < 500:
        return corrected, {"support_pixels": int(overlap.sum()), "applied": False}
    reference = front_colors_full[indices[overlap]]
    keep = _overlap_skin_samples(corrected[overlap], reference)
    overlap_positions = np.flatnonzero(overlap)[keep]
    if len(overlap_positions) < 500:
        return corrected, {"support_pixels": int(len(overlap_positions)), "applied": False}

    height, width = texture_shape
    residual = np.zeros((height, width, 3), dtype=np.float32)
    support = np.zeros((height, width), dtype=np.float32)
    global_indices = indices[overlap_positions]
    yy, xx = valid_y[global_indices], valid_x[global_indices]
    residual[yy, xx] = front_colors_full[global_indices] - corrected[overlap_positions]
    support[yy, xx] = 1.0
    smooth_support = cv2.GaussianBlur(support, (0, 0), sigmaX=float(sigma_px), sigmaY=float(sigma_px))
    smooth_residual = cv2.GaussianBlur(residual, (0, 0), sigmaX=float(sigma_px), sigmaY=float(sigma_px))
    field = smooth_residual / np.maximum(smooth_support[:, :, None], 1e-4)
    field = np.clip(field, -float(max_rgb_shift), float(max_rgb_shift))
    sample_field = field[valid_y[indices], valid_x[indices]]
    confidence = np.clip(smooth_support[valid_y[indices], valid_x[indices]] / 0.12, 0.0, 1.0)
    corrected += float(strength) * sample_field * confidence[:, None]
    corrected = np.clip(corrected, 0.0, 255.0)
    return corrected, {
        "support_pixels": int(len(overlap_positions)),
        "applied": True,
        "max_rgb_shift": float(np.max(np.abs(float(strength) * sample_field * confidence[:, None]))),
    }


def _write_side_ear_repair_debug(
    debug_dir: Optional[Path],
    before: np.ndarray,
    after: np.ndarray,
    roi_mask: np.ndarray,
    repair_mask: np.ndarray,
    stats: dict,
) -> None:
    if debug_dir is None:
        return
    debug_dir.mkdir(parents=True, exist_ok=True)
    with open(debug_dir / "side_ear_texture_repair.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    cv2.imwrite(str(debug_dir / "roi_mask.png"), (roi_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(debug_dir / "repair_mask.png"), (repair_mask.astype(np.uint8) * 255))

    overlay = before.copy()
    overlay[roi_mask] = (overlay[roi_mask].astype(np.float32) * 0.55 + np.array([40, 170, 255]) * 0.45).astype(np.uint8)
    overlay[repair_mask] = (overlay[repair_mask].astype(np.float32) * 0.25 + np.array([255, 60, 40]) * 0.75).astype(np.uint8)
    cv2.imwrite(str(debug_dir / "roi_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    focus = roi_mask | repair_mask
    if not np.any(focus):
        return
    ys, xs = np.where(focus)
    pad = 32
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(before.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(before.shape[1], int(xs.max()) + pad + 1)
    cv2.imwrite(str(debug_dir / "before_crop.png"), cv2.cvtColor(before[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "after_crop.png"), cv2.cvtColor(after[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / "overlay_crop.png"), cv2.cvtColor(overlay[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))


def repair_side_ear_texture_patch(
    texture: np.ndarray,
    valid_mask: np.ndarray,
    tri_map: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    enabled: bool,
    x_abs_min: float,
    y_min: float,
    y_max: float,
    z_max: float,
    lab_delta_threshold: float,
    grad_lab_threshold: float,
    dilate_px: int,
    inpaint_radius: int,
    smooth_alpha: float,
    debug_dir: Optional[Path] = None,
) -> np.ndarray:
    stats = {
        "enabled": bool(enabled),
        "x_abs_min": float(x_abs_min),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "z_max": float(z_max),
        "roi_faces": 0,
        "roi_pixels": 0,
        "repair_pixels": 0,
        "reference_pixels": 0,
        "ring_pixels": 0,
        "median_rgb": None,
    }
    empty_mask = np.zeros(valid_mask.shape, dtype=bool)
    if not enabled or len(faces) == 0 or texture.size == 0:
        _write_side_ear_repair_debug(debug_dir, texture, texture, empty_mask, empty_mask, stats)
        return texture

    centers = vertices[faces].mean(axis=1).astype(np.float32)
    side_faces = (
        (np.abs(centers[:, 0]) >= float(x_abs_min)) &
        (centers[:, 1] >= float(y_min)) &
        (centers[:, 1] <= float(y_max)) &
        (centers[:, 2] <= float(z_max))
    )
    stats["roi_faces"] = int(np.count_nonzero(side_faces))

    face_index_ok = (tri_map >= 0) & (tri_map < len(faces))
    roi_lookup = np.zeros(valid_mask.shape, dtype=bool)
    roi_lookup[face_index_ok] = side_faces[tri_map[face_index_ok]]
    roi_mask = valid_mask.astype(bool) & roi_lookup
    stats["roi_pixels"] = int(np.count_nonzero(roi_mask))
    before = texture.copy()
    if stats["roi_pixels"] < 32:
        _write_side_ear_repair_debug(debug_dir, before, texture, roi_mask, empty_mask, stats)
        return texture

    ring_px = max(17, int(dilate_px) * 5)
    if ring_px % 2 == 0:
        ring_px += 1
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    ring_mask = cv2.dilate(roi_mask.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
    ring_mask = ring_mask & valid_mask.astype(bool) & ~roi_mask
    if np.count_nonzero(ring_mask) < 100:
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1))
        ring_mask = cv2.dilate(roi_mask.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
        ring_mask = ring_mask & valid_mask.astype(bool) & ~roi_mask

    inner_band = 0.024
    reference_faces = (
        (np.abs(centers[:, 0]) >= max(0.0, float(x_abs_min) - inner_band)) &
        (np.abs(centers[:, 0]) < float(x_abs_min)) &
        (centers[:, 1] >= float(y_min) - 0.012) &
        (centers[:, 1] <= float(y_max) + 0.012) &
        (centers[:, 2] <= float(z_max) + 0.030)
    )
    reference_lookup = np.zeros(valid_mask.shape, dtype=bool)
    reference_lookup[face_index_ok] = reference_faces[tri_map[face_index_ok]]
    reference_mask = valid_mask.astype(bool) & reference_lookup
    stats["reference_pixels"] = int(np.count_nonzero(reference_mask))

    sample_mask = reference_mask if stats["reference_pixels"] >= 1000 else ring_mask
    if np.count_nonzero(sample_mask) < 100:
        sample_mask = valid_mask.astype(bool) & ~roi_mask
    sample_pixels = texture[sample_mask]
    if len(sample_pixels) == 0:
        sample_pixels = texture[valid_mask.astype(bool)]

    def skin_reference_pixels(pixels: np.ndarray) -> np.ndarray:
        if len(pixels) == 0:
            return pixels
        p = pixels.astype(np.float32)
        brightness = p.mean(axis=1)
        r, g, b = p[:, 0], p[:, 1], p[:, 2]
        skin_like = (
            (brightness > 105.0) & (brightness < 245.0) &
            (r > g * 0.78) & (g > b * 0.78) & (r > b * 0.86)
        )
        if np.count_nonzero(skin_like) < 500:
            return pixels
        filtered = pixels[skin_like]
        filtered_brightness = filtered.astype(np.float32).mean(axis=1)
        bright_cut = np.percentile(filtered_brightness, 45)
        brighter = filtered[filtered_brightness >= bright_cut]
        return brighter if len(brighter) >= 500 else filtered

    skin_pixels = skin_reference_pixels(sample_pixels)
    if len(skin_pixels) < 500:
        broader_sample = texture[valid_mask.astype(bool) & ~roi_mask]
        broader_skin = skin_reference_pixels(broader_sample)
        if len(broader_skin) >= 500:
            skin_pixels = broader_skin
    if len(skin_pixels) >= 100:
        sample_pixels = skin_pixels
    median_rgb = np.median(sample_pixels, axis=0).astype(np.uint8)
    stats["ring_pixels"] = int(np.count_nonzero(sample_mask))
    stats["median_rgb"] = [int(v) for v in median_rgb.tolist()]

    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB).astype(np.float32)
    median_lab = cv2.cvtColor(median_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
    lab_delta = lab - median_lab[None, None, :]
    lab_delta[:, :, 1:] *= 1.35
    color_delta = np.linalg.norm(lab_delta, axis=2)

    lab_smooth = cv2.GaussianBlur(lab, (0, 0), 1.2)
    grad_x = cv2.Sobel(lab_smooth[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lab_smooth[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    roi_delta = color_delta[roi_mask]
    roi_grad = grad_mag[roi_mask]
    delta_cut = max(float(lab_delta_threshold), float(np.percentile(roi_delta, 68)))
    grad_cut = max(float(grad_lab_threshold), float(np.percentile(roi_grad, 72)))
    repair_core = roi_mask & ((color_delta >= delta_cut) | (grad_mag >= grad_cut))
    repair_mask = repair_core
    min_repair = max(64, int(stats["roi_pixels"] * 0.015))
    if np.count_nonzero(repair_mask) < min_repair:
        delta_cut = max(float(lab_delta_threshold) * 0.75, float(np.percentile(roi_delta, 58)))
        repair_core = roi_mask & ((color_delta >= delta_cut) | (grad_mag >= grad_cut))
        repair_mask = repair_core

    max_repair = int(stats["roi_pixels"] * 0.42)
    if np.count_nonzero(repair_mask) > max_repair:
        delta_cut = max(float(lab_delta_threshold), float(np.percentile(roi_delta, 82)))
        grad_cut = max(float(grad_lab_threshold), float(np.percentile(roi_grad, 86)))
        repair_core = roi_mask & ((color_delta >= delta_cut) | (grad_mag >= grad_cut))
        repair_mask = repair_core

    dilate_px = int(max(0, dilate_px))
    if dilate_px > 0 and np.any(repair_mask):
        k = dilate_px * 2 + 1
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(repair_mask.astype(np.uint8), repair_kernel, iterations=1).astype(bool) & roi_mask
        repair_mask = dilated if np.count_nonzero(dilated) <= max_repair else repair_core
    stats["repair_pixels"] = int(np.count_nonzero(repair_mask))

    result = texture.copy()
    if np.any(repair_mask):
        tmp = texture.copy()
        tmp[~valid_mask.astype(bool)] = median_rgb
        tmp[repair_mask] = median_rgb
        inpainted = cv2.inpaint(tmp, repair_mask.astype(np.uint8) * 255, int(max(1, inpaint_radius)), cv2.INPAINT_TELEA)
        result[repair_mask] = inpainted[repair_mask]

    smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
    if smooth_alpha > 0:
        tmp = result.copy()
        tmp[~valid_mask.astype(bool)] = median_rgb
        smooth = cv2.bilateralFilter(tmp, 9, 18, 7)
        blended = result.astype(np.float32)
        blended[roi_mask] = (
            blended[roi_mask] * (1.0 - smooth_alpha) +
            smooth[roi_mask].astype(np.float32) * smooth_alpha
        )
        result = np.clip(blended, 0, 255).astype(np.uint8)

    logger.info(
        "  Side-ear texture repair: "
        f"roi_faces={stats['roi_faces']}, roi_px={stats['roi_pixels']}, "
        f"repair_px={stats['repair_pixels']}"
    )
    _write_side_ear_repair_debug(debug_dir, before, result, roi_mask, repair_mask, stats)
    return result


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
    sampling_warps: Optional[Dict[str, object]] = None,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
    diagnostics: Optional[dict] = None,
    alpha_mask_out: Optional[dict] = None,
    transparent_bottom_quantile: Optional[float] = None,
    sampling_mode: str = "legacy_registered",
    sampling_debug_out: Optional[dict] = None,
) -> np.ndarray:
    """
    将3张照片的颜色烘焙到 UV 纹理图。

    Returns: (H, W, 3) uint8 RGB 纹理图
    """
    H = W = tex_size
    if sampling_mode not in {"legacy_registered", "strict_projective"}:
        raise ValueError(
            "sampling_mode must be 'legacy_registered' or 'strict_projective'"
        )
    strict_projective = sampling_mode == "strict_projective"
    if strict_projective and sampling_warps:
        raise ValueError(
            "strict_projective sampling rejects sampling_warps; geometry and "
            "camera projection must define the sampled pixel"
        )
    if diagnostics is not None:
        diagnostics["sampling_mode"] = sampling_mode
        diagnostics.setdefault("sampling_coordinates", {})
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
    blend_layers = []
    blend_weights = []
    blend_view_names = []
    for view_name in view_names:
        cam    = cameras[view_name]
        K, R, t = cam["K"], cam["R"], cam["t"]
        image  = images[view_name]              # (H_img, W_img, 3) RGB
        H_img, W_img = image.shape[:2]
        depth_map = _render_camera_depth(vertices, faces, K, R, t, (H_img, W_img))
        projective_sample = None
        if strict_projective:
            projection = project_points_strict(pts_3d, K, R, t)
            projective_sample = sample_projected_attributes(
                projection,
                image,
                mask=(
                    face_masks[view_name]
                    if face_masks is not None and view_name in face_masks
                    else None
                ),
                depth_map=depth_map,
                semantic_map=(
                    feature_masks[view_name]
                    if feature_masks is not None and view_name in feature_masks
                    else None
                ),
            )
            v_cam = projection.camera_points
            z = projection.depth
            proj = projection.pixel_xy
            front = projection.front_facing
            pixel_xy = projection.pixel_xy
            if diagnostics is not None:
                diagnostics["sampling_coordinates"][view_name] = (
                    assert_strict_sampling_coordinates(
                        proj,
                        pixel_xy,
                        projective_sample.in_bounds,
                    )
                )
        else:
            v_cam, z, proj, front = project_texture_points_to_image(
                pts_3d, K, R, t
            )
            pixel_xy = proj
            if sampling_warps is not None and view_name in sampling_warps:
                pixel_xy = sampling_warps[view_name].apply(
                    proj, (H_img, W_img)
                )

        # 在图像范围内的点
        if strict_projective:
            in_img = projective_sample.in_bounds
        else:
            in_img = (front &
                      (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < W_img - 1) &
                      (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < H_img - 1))

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
            if strict_projective:
                in_mask = projective_sample.mask
            else:
                mask_img = face_masks[view_name]   # (H_img, W_img) uint8
                mask_H, mask_W = mask_img.shape[:2]
            # 采样 mask 值（最近邻）
                px_u_int = pixel_xy[:, 0].astype(int)
                px_v_int = pixel_xy[:, 1].astype(int)
                px_u_int = np.clip(px_u_int, 0, mask_W - 1)
                px_v_int = np.clip(px_v_int, 0, mask_H - 1)
                in_mask = mask_img[px_v_int, px_u_int] > 127
            valid_pts = in_img & (weights > 0.05) & in_mask
        else:
            valid_pts = in_img & (weights > 0.05)

        effective_weights = weights.copy()
        if valid_pts.any():
            if strict_projective:
                z_ref = projective_sample.depth
            else:
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
                # UV seam fallback may relax the normal test, but it must never
                # bypass depth visibility and sample an occluded eye/nose surface.
                center_fallback &= visible
                valid_pts = (valid_pts & visible) | center_fallback
            else:
                valid_pts &= visible & ~force_front

        if valid_pts.sum() == 0:
            continue

        if strict_projective:
            colors = projective_sample.rgb[valid_pts]
        else:
            colors = _bilinear_sample(
                image,
                pixel_xy[valid_pts, 0],
                pixel_xy[valid_pts, 1],
            )  # (K, 3)

        # 额外过滤极暗像素（残余背景）
        vp_idx = np.where(valid_pts)[0]
        view_weights = _view_region_weight(view_name, pts_3d)[vp_idx]
        sample_weights = effective_weights[vp_idx] * view_weights

        view_samples[view_name] = {
            "idx": vp_idx,
            "colors": colors.astype(np.float32),
            "weights": sample_weights.astype(np.float32),
            "pixel_xy": pixel_xy[valid_pts].astype(np.float32),
        }
        if strict_projective and projective_sample.semantic is not None:
            view_samples[view_name]["feature"] = (
                projective_sample.semantic[valid_pts] > 0
            )
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

    protected_feature_full = np.zeros(len(valid_y), dtype=bool)
    if "front" in view_samples and feature_masks is not None and "front" in feature_masks:
        front_sample = view_samples["front"]
        if strict_projective and "feature" in front_sample:
            protected = front_sample["feature"]
        else:
            feature_mask = feature_masks["front"]
            pixel_xy = front_sample["pixel_xy"]
            px = np.clip(pixel_xy[:, 0].astype(np.int32), 0, feature_mask.shape[1] - 1)
            py = np.clip(pixel_xy[:, 1].astype(np.int32), 0, feature_mask.shape[0] - 1)
            protected = feature_mask[py, px] > 0
        protected_feature_full[front_sample["idx"][protected]] = True
        if diagnostics is not None:
            diagnostics["protected_feature_uv_pixels"] = int(protected.sum())

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

        if view_name != "front" and front_present is not None:
            correction_protection = (
                np.zeros_like(protected_feature_full)
                if strict_projective
                else protected_feature_full
            )
            matched_colors, local_color_report = _local_overlap_color_correction(
                matched_colors,
                vp_idx,
                front_present,
                front_colors_full,
                valid_y,
                valid_x,
                correction_protection,
                (H, W),
            )
            if diagnostics is not None:
                diagnostics.setdefault("local_color_field", {})[view_name] = local_color_report

        if diagnostics is not None and view_name != "front" and front_present is not None:
            overlap = front_present[vp_idx] & (sample_weights > 1e-4) & (front_weights_full[vp_idx] > 1e-4)
            if int(overlap.sum()) >= 100:
                reference = front_colors_full[vp_idx[overlap]]
                before_lab = _rgb_to_lab_float(colors[overlap])
                after_lab = _rgb_to_lab_float(matched_colors[overlap])
                reference_lab = _rgb_to_lab_float(reference)
                skin_keep = _overlap_skin_samples(colors[overlap], reference)
                skin_before = np.linalg.norm(before_lab[skin_keep] - reference_lab[skin_keep], axis=1)
                skin_after = np.linalg.norm(after_lab[skin_keep] - reference_lab[skin_keep], axis=1)
                diagnostics.setdefault("overlap_color", {})[view_name] = {
                    "pixels": int(overlap.sum()),
                    "lab_delta_before": float(np.median(np.linalg.norm(before_lab - reference_lab, axis=1))),
                    "lab_delta_after": float(np.median(np.linalg.norm(after_lab - reference_lab, axis=1))),
                    "skin_pixels": int(skin_keep.sum()),
                    "skin_lab_delta_before": float(np.median(skin_before)) if len(skin_before) else None,
                    "skin_lab_delta_after": float(np.median(skin_after)) if len(skin_after) else None,
                }

        color_acc[vp_idx]  += matched_colors * sample_weights[:, None]
        weight_acc[vp_idx] += sample_weights

        layer = np.zeros((H, W, 3), dtype=np.float32)
        weight_layer = np.zeros((H, W), dtype=np.float32)
        layer[valid_y[vp_idx], valid_x[vp_idx]] = matched_colors
        weight_layer[valid_y[vp_idx], valid_x[vp_idx]] = sample_weights
        weight_layer = _feather_view_weight(weight_layer, radius_px=72.0)
        blend_layers.append(layer)
        blend_weights.append(weight_layer)
        blend_view_names.append(view_name)

    has_color = weight_acc > 0
    texture = np.zeros((H, W, 3), dtype=np.float32)
    if len(blend_layers) >= 2:
        texture = _multiband_blend(blend_layers, blend_weights, levels=5)
        texture[~valid_mask] = 0.0
    else:
        texture[valid_y[has_color], valid_x[has_color]] = (
            color_acc[has_color] / weight_acc[has_color, None]
        )

    if sampling_debug_out is not None:
        source_view = np.full((H, W), -1, dtype=np.int16)
        source_weight = np.zeros((H, W), dtype=np.float32)
        if blend_weights:
            stacked_weights = np.stack(blend_weights, axis=0)
            source_view = np.argmax(stacked_weights, axis=0).astype(np.int16)
            source_weight = np.max(stacked_weights, axis=0).astype(np.float32)
            source_view[source_weight <= 0.0] = -1
            source_view[~valid_mask] = -1
            source_weight[~valid_mask] = 0.0
        sampling_debug_out.update(
            {
                "view_names": tuple(blend_view_names),
                "source_view": source_view,
                "source_weight": source_weight,
                "observed": source_weight > 0.0,
            }
        )

    texture, ownership_report = _apply_front_feature_ownership(
        texture,
        valid_y,
        valid_x,
        protected_feature_full,
        front_present,
        front_colors_full,
    )
    if diagnostics is not None:
        diagnostics["front_feature_ownership"] = ownership_report

    # 对无颜色的有效区域做 inpainting 填充（遮挡区域）
    texture_uint8 = texture.clip(0, 255).astype(np.uint8)
    has_color_img = weight_acc_img(texture_uint8, valid_y, valid_x, has_color, H, W)
    if alpha_mask_out is not None:
        geometry_keep = None
        if transparent_bottom_quantile is not None:
            quantile = float(np.clip(transparent_bottom_quantile, 0.0, 0.25))
            y_floor = float(np.quantile(vertices[:, 1], quantile))
            geometry_keep = np.zeros((H, W), dtype=bool)
            keep_samples = pts_3d[:, 1] >= y_floor
            geometry_keep[valid_y, valid_x] = keep_samples
            if diagnostics is not None:
                diagnostics["transparent_bottom"] = {
                    "vertex_y_quantile": quantile,
                    "y_floor": y_floor,
                    "hidden_uv_pixels": int((valid_mask & ~geometry_keep).sum()),
                }
        visible_alpha, observation_confidence = texture_alpha_masks(
            valid_mask,
            has_color_img,
            geometry_keep=geometry_keep,
        )
        alpha_mask_out["mask"] = visible_alpha
        alpha_mask_out["observation_confidence"] = observation_confidence
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
        small_missing = _small_connected_regions(missing_valid_mask, max_area=12000)
        if small_missing.any():
            texture_uint8 = cv2.inpaint(texture_uint8, small_missing * 255, 3, cv2.INPAINT_TELEA)
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


def _small_connected_regions(mask: np.ndarray, max_area: int = 12000) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= int(max_area):
            result[labels == label] = 1
    return result


def _texture_alpha_from_observation(
    valid_mask: np.ndarray,
    observed_mask: np.ndarray,
    geometry_keep: Optional[np.ndarray] = None,
) -> np.ndarray:
    allowed = np.asarray(valid_mask, dtype=bool)
    if geometry_keep is not None:
        allowed &= np.asarray(geometry_keep, dtype=bool)
    observed = ((np.asarray(observed_mask) > 0) & allowed).astype(np.uint8) * 255
    min_component_area = max(8, int(observed.size * 0.0036))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((observed > 0).astype(np.uint8), 8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_component_area:
            observed[labels == label] = 0
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    observed = cv2.morphologyEx(observed, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    observed = cv2.dilate(
        observed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    observed[~allowed] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats((observed > 0).astype(np.uint8), 8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_component_area:
            observed[labels == label] = 0
    observed[~allowed] = 0
    return observed


def texture_alpha_masks(
    valid_mask: np.ndarray,
    observed_mask: np.ndarray,
    geometry_keep: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return visible filled alpha and independent observation confidence."""
    allowed = np.asarray(valid_mask, dtype=bool)
    if geometry_keep is not None:
        allowed &= np.asarray(geometry_keep, dtype=bool)
    visible_alpha = allowed.astype(np.uint8) * 255
    observation_confidence = (
        (np.asarray(observed_mask) > 0) & allowed
    ).astype(np.uint8) * 255
    return visible_alpha, observation_confidence


def transparent_bottom_face_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_y_quantile: float,
) -> Tuple[np.ndarray, float]:
    """Select low-neck faces for a separate transparent GLB primitive."""
    quantile = float(np.clip(vertex_y_quantile, 0.0, 0.25))
    y_floor = float(np.quantile(np.asarray(vertices)[:, 1], quantile))
    face_mask = np.asarray(vertices)[np.asarray(faces), 1].mean(axis=1) <= y_floor
    return face_mask, y_floor


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
    smooth_geometry: bool = True,
    transparent_face_mask: Optional[np.ndarray] = None,
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

    # Legacy export optionally smooths geometry before UV expansion. Stable
    # reconstruction disables this because texture packaging must not deform mesh.
    if smooth_geometry:
        shared_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.smoothing.filter_laplacian(shared_mesh, iterations=2, lamb=0.3)
        smooth_verts = np.array(shared_mesh.vertices)
    else:
        smooth_verts = np.asarray(vertices).copy()

    # ── Step 2：展开为 per-face-vertex（UV 必须独立寻址）────────────────────
    flat_geom_idx = faces.flatten()
    flat_uv_idx   = uv_faces.flatten()
    exp_verts = smooth_verts[flat_geom_idx]
    exp_uv    = image_uv_to_obj_uv(uv_verts[flat_uv_idx])
    exp_faces = np.arange(len(exp_verts)).reshape(-1, 3)

    # ── Step 3：重算平滑顶点法线 ────────────────────────────────────────────
    transparent_faces = np.zeros(len(exp_faces), dtype=bool)
    if transparent_face_mask is not None:
        transparent_faces = np.asarray(transparent_face_mask, dtype=bool)
        if transparent_faces.shape != (len(exp_faces),):
            raise ValueError("transparent_face_mask must match the face count")
    opaque_faces = ~transparent_faces
    if not np.any(opaque_faces):
        raise ValueError("At least one opaque face is required")

    opaque_texture = texture[..., :3] if texture.ndim == 3 else texture
    opaque_material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=PILImage.fromarray(opaque_texture),
        metallicFactor=0.0,
        roughnessFactor=0.9,
        alphaMode="OPAQUE",
        name=f"skin_{lighting_type}",
    )

    def _textured_face_subset(face_mask: np.ndarray, material) -> "trimesh.Trimesh":
        corner_indices = exp_faces[face_mask].reshape(-1)
        subset_vertices = exp_verts[corner_indices]
        subset_uv = exp_uv[corner_indices]
        subset_faces = np.arange(len(subset_vertices)).reshape(-1, 3)
        normal_source = trimesh.Trimesh(
            vertices=subset_vertices,
            faces=subset_faces,
            process=False,
        )
        subset = trimesh.Trimesh(
            vertices=subset_vertices,
            faces=subset_faces,
            vertex_normals=np.asarray(normal_source.vertex_normals),
            process=False,
        )
        subset.visual = trimesh.visual.texture.TextureVisuals(
            uv=subset_uv,
            material=material,
        )
        return subset

    geometry = {
        f"face_{lighting_type}": _textured_face_subset(opaque_faces, opaque_material)
    }
    if np.any(transparent_faces):
        transparent_texture = np.zeros((2, 2, 4), dtype=np.uint8)
        transparent_material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=PILImage.fromarray(transparent_texture),
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode="BLEND",
            name=f"hidden_bottom_{lighting_type}",
        )
        geometry[f"hidden_bottom_{lighting_type}"] = _textured_face_subset(
            transparent_faces,
            transparent_material,
        )

    # 创建带法线的 trimesh Mesh
    # 创建材质
    # 绑定 UV
    # GLB extras 元数据（前端图层切换用）
    scene = trimesh.Scene(
        geometry=geometry,
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
    smooth_geometry_on_export: bool = True,
    face_masks: Optional[Dict[str, np.ndarray]] = None,
    unified_texture: Optional[np.ndarray] = None,       # 预融合统一纹理（fallback）
    hires_front_image: Optional[np.ndarray] = None,     # 原始高清正面图（仅正面，旧接口）
    hires_scale_factor: float = 1.0,                     # K 缩放系数（原始分辨率 / 512）
    hires_images: Optional[Dict[str, np.ndarray]] = None,  # 多视角高清图（最优先）

    working_image_size: Optional[int] = None,
    sampling_warps: Optional[Dict[str, object]] = None,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
    diagnostics: Optional[dict] = None,
    transparent_unobserved: bool = False,
    transparent_bottom_quantile: Optional[float] = None,
    sampling_mode: str = "legacy_registered",
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
        visible_face_crop_min_component_faces = int(getattr(cfg, "VISIBLE_FACE_CROP_MIN_COMPONENT_FACES", 0))
        visible_face_crop_side_ear_trim = bool(getattr(cfg, "VISIBLE_FACE_CROP_SIDE_EAR_TRIM", False))
        visible_face_crop_side_ear_x_abs_min = float(getattr(cfg, "VISIBLE_FACE_CROP_SIDE_EAR_X_ABS_MIN", 0.066))
        visible_face_crop_side_ear_y_min = float(getattr(cfg, "VISIBLE_FACE_CROP_SIDE_EAR_Y_MIN", -0.060))
        visible_face_crop_side_ear_y_max = float(getattr(cfg, "VISIBLE_FACE_CROP_SIDE_EAR_Y_MAX", 0.060))
        visible_face_crop_side_ear_z_max = float(getattr(cfg, "VISIBLE_FACE_CROP_SIDE_EAR_Z_MAX", 0.004))
    except Exception:
        enable_visible_face_crop = False
        visible_face_crop_rings = 2
        visible_face_crop_z_tol = 0.006
        visible_face_crop_min_component_faces = 0
        visible_face_crop_side_ear_trim = False
        visible_face_crop_side_ear_x_abs_min = 0.066
        visible_face_crop_side_ear_y_min = -0.060
        visible_face_crop_side_ear_y_max = 0.060
        visible_face_crop_side_ear_z_max = 0.004
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
            min_component_faces=visible_face_crop_min_component_faces,
            side_ear_trim=visible_face_crop_side_ear_trim,
            side_ear_x_abs_min=visible_face_crop_side_ear_x_abs_min,
            side_ear_y_min=visible_face_crop_side_ear_y_min,
            side_ear_y_max=visible_face_crop_side_ear_y_max,
            side_ear_z_max=visible_face_crop_side_ear_z_max,
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
    alpha_mask_out = {}
    sampling_debug_out = {}

    # ── 纹理烘焙（高清直采 > 统一纹理 > 多视角） ─────────────────────────
    if hires_images is not None:
        # ── 多视角高清烘焙（覆盖全部 UV，包括侧面耳朵等区域）──────────
        logger.info(f"多视角高清烘焙模式（{len(hires_images)} 个视角）...")
        scaled_cameras: Dict[str, dict] = {}
        processed_hires: Dict[str, np.ndarray] = {}
        scaled_masks: Dict[str, np.ndarray] = {}
        scaled_feature_masks: Dict[str, np.ndarray] = {}

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
            if feature_masks is not None and view_name in feature_masks:
                scaled_feature_masks[view_name] = cv2.resize(
                    feature_masks[view_name], (max_sz, max_sz),
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
            sampling_warps=sampling_warps,
            feature_masks=scaled_feature_masks if scaled_feature_masks else None,
            diagnostics=diagnostics,
            alpha_mask_out=alpha_mask_out,
            transparent_bottom_quantile=transparent_bottom_quantile,
            sampling_mode=sampling_mode,
            sampling_debug_out=sampling_debug_out,
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
            sampling_warps=sampling_warps,
            feature_masks=feature_masks,
            diagnostics=diagnostics,
            alpha_mask_out=alpha_mask_out,
            transparent_bottom_quantile=transparent_bottom_quantile,
            sampling_mode=sampling_mode,
            sampling_debug_out=sampling_debug_out,
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

    try:
        from src import config as cfg
        enable_side_ear_texture_repair = bool(getattr(cfg, "ENABLE_SIDE_EAR_TEXTURE_REPAIR", False))
        side_ear_texture_repair_x_abs_min = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_X_ABS_MIN", 0.062))
        side_ear_texture_repair_y_min = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_Y_MIN", -0.060))
        side_ear_texture_repair_y_max = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_Y_MAX", 0.060))
        side_ear_texture_repair_z_max = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_Z_MAX", 0.006))
        side_ear_texture_repair_lab_delta = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_LAB_DELTA", 14.0))
        side_ear_texture_repair_grad_lab = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_GRAD_LAB", 18.0))
        side_ear_texture_repair_dilate_px = int(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_DILATE_PX", 2))
        side_ear_texture_repair_inpaint_radius = int(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_INPAINT_RADIUS", 5))
        side_ear_texture_repair_smooth_alpha = float(getattr(cfg, "SIDE_EAR_TEXTURE_REPAIR_SMOOTH_ALPHA", 0.16))
    except Exception:
        enable_side_ear_texture_repair = False
        side_ear_texture_repair_x_abs_min = 0.062
        side_ear_texture_repair_y_min = -0.060
        side_ear_texture_repair_y_max = 0.060
        side_ear_texture_repair_z_max = 0.006
        side_ear_texture_repair_lab_delta = 14.0
        side_ear_texture_repair_grad_lab = 18.0
        side_ear_texture_repair_dilate_px = 2
        side_ear_texture_repair_inpaint_radius = 5
        side_ear_texture_repair_smooth_alpha = 0.16

    texture = repair_side_ear_texture_patch(
        texture,
        valid_mask,
        tri_map,
        vertices,
        faces,
        enabled=enable_side_ear_texture_repair,
        x_abs_min=side_ear_texture_repair_x_abs_min,
        y_min=side_ear_texture_repair_y_min,
        y_max=side_ear_texture_repair_y_max,
        z_max=side_ear_texture_repair_z_max,
        lab_delta_threshold=side_ear_texture_repair_lab_delta,
        grad_lab_threshold=side_ear_texture_repair_grad_lab,
        dilate_px=side_ear_texture_repair_dilate_px,
        inpaint_radius=side_ear_texture_repair_inpaint_radius,
        smooth_alpha=side_ear_texture_repair_smooth_alpha,
        debug_dir=output_texture_dir.parent / "debug" / "side_ear_texture_repair",
    )

    valid_texture_mask = alpha_mask_out.get("mask")
    observation_confidence = alpha_mask_out.get("observation_confidence")
    if valid_texture_mask is not None:
        cv2.imwrite(
            str(output_texture_dir / f"texture_valid_{lighting_type}.png"),
            valid_texture_mask,
        )
    if observation_confidence is not None:
        confidence_path = output_texture_dir / f"texture_observation_{lighting_type}.png"
        cv2.imwrite(str(confidence_path), observation_confidence)
    if sampling_debug_out:
        view_names = tuple(sampling_debug_out.get("view_names", ()))
        source_view = sampling_debug_out.get("source_view")
        source_weight = sampling_debug_out.get("source_weight")
        if source_view is not None:
            palette = np.array(
                [
                    [67, 151, 232],
                    [59, 190, 118],
                    [222, 108, 142],
                ],
                dtype=np.uint8,
            )
            source_rgb = np.zeros((*source_view.shape, 3), dtype=np.uint8)
            for view_index, _view_name in enumerate(view_names):
                source_rgb[source_view == view_index] = palette[
                    view_index % len(palette)
                ]
            source_path = output_texture_dir / f"texture_source_{lighting_type}.png"
            cv2.imwrite(str(source_path), cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
        if source_weight is not None:
            observed_path = (
                output_texture_dir / f"texture_sampling_valid_{lighting_type}.png"
            )
            cv2.imwrite(
                str(observed_path),
                (np.asarray(source_weight) > 0.0).astype(np.uint8) * 255,
            )
        if diagnostics is not None:
            diagnostics["sampling_debug"] = {
                "view_names": list(view_names),
                "source_map": str(
                    output_texture_dir / f"texture_source_{lighting_type}.png"
                ),
                "valid_map": str(
                    output_texture_dir / f"texture_sampling_valid_{lighting_type}.png"
                ),
            }
    if transparent_unobserved and valid_texture_mask is not None:
        texture = np.dstack((texture, valid_texture_mask))
    if diagnostics is not None:
        diagnostics["opaque_uv_pixels"] = int(
            (valid_texture_mask > 0).sum()
        ) if valid_texture_mask is not None else int(texture.shape[0] * texture.shape[1])
        diagnostics["observed_uv_pixels"] = int(
            (observation_confidence > 0).sum()
        ) if observation_confidence is not None else 0
        diagnostics["inpainted_valid_opaque"] = True
        diagnostics["transparent_unobserved"] = bool(transparent_unobserved)
        diagnostics["face_material_alpha_mode"] = (
            "BLEND" if transparent_unobserved else "OPAQUE"
        )

    transparent_face_mask = None
    if not transparent_unobserved and transparent_bottom_quantile is not None:
        quantile = float(np.clip(transparent_bottom_quantile, 0.0, 0.25))
        transparent_face_mask, y_floor = transparent_bottom_face_mask(
            vertices,
            faces,
            quantile,
        )
        if diagnostics is not None:
            diagnostics.setdefault("transparent_bottom", {}).update(
                {
                    "vertex_y_quantile": quantile,
                    "y_floor": y_floor,
                    "hidden_faces": int(transparent_face_mask.sum()),
                    "material_split": True,
                }
            )

    # ── 保存纹理图 ────────────────────────────────────────────────────────
    tex_path = output_texture_dir / f"albedo_{lighting_type}.png"
    import cv2 as _cv
    color_code = _cv.COLOR_RGBA2BGRA if texture.ndim == 3 and texture.shape[2] == 4 else _cv.COLOR_RGB2BGR
    _cv.imwrite(str(tex_path), _cv.cvtColor(texture, color_code))
    logger.info(f"纹理已保存: {tex_path}")

    # ── GLB 打包 ──────────────────────────────────────────────────────────
    logger.info("打包 GLB...")
    glb_path = output_mesh_dir / "face.glb"
    export_glb(
        vertices, faces, uv_verts, uv_faces,
        texture, glb_path,
        lighting_type=lighting_type,
        lighting_display_name=lighting_display_name,
        smooth_geometry=smooth_geometry_on_export,
        transparent_face_mask=transparent_face_mask,
    )

    return glb_path

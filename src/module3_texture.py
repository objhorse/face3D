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
                uv_verts.append([u, 1.0 - v])   # 翻转 V 轴还原（导出时做了 1-v）
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
        np.array(uv_verts, dtype=np.float32),
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

    view_names = list(cameras.keys())
    logger.info(f"  对 {len(view_names)} 个视角进行颜色采样（{len(valid_y)} 个有效纹理像素）...")

    for view_name in view_names:
        cam    = cameras[view_name]
        K, R, t = cam["K"], cam["R"], cam["t"]
        image  = images[view_name]              # (H_img, W_img, 3) RGB
        H_img, W_img = image.shape[:2]

        # PnP 时翻转了 FLAME 的 Y 轴，投影时需保持一致
        pts_3d_proj = pts_3d.copy()
        pts_3d_proj[:, 1] *= -1

        # 投影到图像空间
        v_cam = (R @ pts_3d_proj.T + t[:, None]).T  # (M, 3)
        z     = v_cam[:, 2]
        front = z > 1e-4                        # 只处理在相机前方的点

        proj = np.zeros((len(valid_y), 2), dtype=np.float32)
        proj[front, 0] = (K[0,0] * v_cam[front,0] / z[front] + K[0,2])
        proj[front, 1] = (K[1,1] * v_cam[front,1] / z[front] + K[1,2])

        # 在图像范围内的点
        in_img = (front &
                  (proj[:, 0] >= 0) & (proj[:, 0] < W_img - 1) &
                  (proj[:, 1] >= 0) & (proj[:, 1] < H_img - 1))

        # 计算权重：面法线 · 相机方向（使用 Y 翻转后的坐标系）
        cam_center_proj = -R.T @ t              # (3,) 相机中心在投影坐标系
        # 还原回 FLAME 坐标系（翻转 Y）
        cam_center = cam_center_proj.copy()
        cam_center[1] *= -1
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

        if valid_pts.sum() == 0:
            continue

        colors = _bilinear_sample(
            image,
            proj[valid_pts, 0],
            proj[valid_pts, 1],
        )  # (K, 3)

        # 额外过滤极暗像素（残余背景）
        brightness = colors.mean(axis=1)
        fg_mask    = brightness > 15.0

        vp_idx = np.where(valid_pts)[0]
        fg_idx = vp_idx[fg_mask]

        color_acc[fg_idx]  += colors[fg_mask] * weights[fg_idx, None]
        weight_acc[fg_idx] += weights[fg_idx]

        logger.info(f"    [{view_name}] 采样 {fg_mask.sum()} 个前景像素（共{valid_pts.sum()}有效）")

    # 归一化
    has_color = weight_acc > 0
    texture = np.zeros((H, W, 3), dtype=np.float32)
    texture[valid_y[has_color], valid_x[has_color]] = (
        color_acc[has_color] / weight_acc[has_color, None]
    )

    # 对无颜色的有效区域做 inpainting 填充（遮挡区域）
    texture_uint8 = texture.clip(0, 255).astype(np.uint8)
    inpaint_mask  = (valid_mask & ~(weight_acc_img(texture_uint8, valid_y, valid_x, has_color, H, W))).astype(np.uint8)
    if inpaint_mask.any():
        texture_uint8 = cv2.inpaint(texture_uint8, inpaint_mask * 255, 3, cv2.INPAINT_TELEA)

    return texture_uint8


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
    cam_center_proj = -R.T @ t       # (3,) 投影坐标系
    cam_center = cam_center_proj.copy()
    cam_center[1] *= -1              # 还原 Y 翻转 → FLAME 坐标系

    # 视线方向（点 → 相机），在 FLAME 坐标系中计算
    view_dirs = cam_center - pts_3d  # (M, 3)
    view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
    cosines = np.sum(pt_normals * view_dirs, axis=1)          # (M,)
    front_facing = cosines > 0.05   # 法线朝向相机（小阈值留擦边余量）

    logger.info(
        f"  法线可见性: {front_facing.sum()} / {len(valid_tri)} 面片朝向相机 "
        f"({front_facing.mean()*100:.1f}%)"
    )

    # Y 轴翻转（与 bake_texture 保持一致）
    pts_proj = pts_3d.copy()
    pts_proj[:, 1] *= -1

    v_cam = (R @ pts_proj.T + t[:, None]).T  # (M, 3)
    z     = v_cam[:, 2]
    front = z > 1e-4

    proj = np.zeros((len(valid_y), 2), dtype=np.float32)
    proj[front, 0] = K[0, 0] * v_cam[front, 0] / z[front] + K[0, 2]
    proj[front, 1] = K[1, 1] * v_cam[front, 1] / z[front] + K[1, 2]

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

    # module2 已完成 Loop Subdivision（~160K faces）+ Laplacian，此处仅做最终精修
    # ── Step 1：轻度 Laplacian 平滑（2次，修复 per-face-vertex 展开前的微小锯齿）
    shared_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(shared_mesh, iterations=2, lamb=0.3)
    smooth_verts = np.array(shared_mesh.vertices)

    # ── Step 2：展开为 per-face-vertex（UV 必须独立寻址）────────────────────
    flat_geom_idx = faces.flatten()
    flat_uv_idx   = uv_faces.flatten()
    exp_verts = smooth_verts[flat_geom_idx]
    exp_uv    = uv_verts[flat_uv_idx]
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
    hires_front_image: Optional[np.ndarray] = None,     # 原始高清正面图（优先）
    hires_scale_factor: float = 1.0,                     # K 缩放系数（原始分辨率 / 512）
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

    # ── 加载 Mesh 和相机 ──────────────────────────────────────────────────
    logger.info("加载 Mesh 和相机参数...")
    vertices, faces, uv_verts, uv_faces = load_mesh_obj(mesh_dir / "face_mesh.obj")
    cameras = load_cameras(mesh_dir / "cameras.json")
    logger.info(f"  Mesh: {len(vertices)} 顶点, {len(faces)} 面片, {len(uv_verts)} UV点")

    # ── UV 光栅化 ─────────────────────────────────────────────────────────
    logger.info("UV 光栅化...")
    tri_map, bary_map = rasterize_uv_map(uv_verts, uv_faces, tex_size)
    valid_mask = tri_map >= 0

    # ── 纹理烘焙（高清直采 > 统一纹理 > 多视角） ─────────────────────────
    if hires_front_image is not None:
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
            vertices, faces, uv_verts, uv_faces,
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
            vertices, faces, uv_verts, uv_faces,
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
            vertices, faces, uv_verts, uv_faces,
            tri_map, bary_map,
            cameras, view_images,
            tex_size,
            face_masks=face_masks,
        )
        # ── 泊松接缝修复 ─────────────────────────────────────────────────
        logger.info("接缝修复...")
        texture = poisson_seam_fix(texture, valid_mask)

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

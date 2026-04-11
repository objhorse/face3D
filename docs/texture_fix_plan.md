# 纹理质量修复计划：高清纹理 + 法线可见性剔除

## Context

当前 3D 重建结果存在三类纹理问题：

**问题 A — 皮肤质感极差（最严重）**
原始照片 2160×3264（700万像素），但 pipeline 在预处理阶段统一缩放到 512×512（26万像素，**缩小27倍**），然后 unified_texture 也是 512×512。这个 512×512 源图要投射到 2048×2048 UV纹理 → 4倍放大 → 严重模糊。再叠加 TPS 变形插值 + 三视角混合平均，毛孔/眉毛/胡茬等细节全部丢失，变成"泥塑"效果。

**问题 B — 后脑勺出现五官幽灵**
`bake_texture_single` 无法线可见性判断，后脑面片法线背对相机仍被投影采色。

**问题 C — 额头纹理缺失**
额头顶点投影到 `proj_y < 0`（图像边界之上），无法采色，出现黑块。

---

## 总预期效果

修复后的 GLB 模型应达到：
- **正面五官区域**：皮肤质感清晰，毛孔/眉毛/胡茬/痘印可分辨，与原始照片接近
- **正面区域颜色**：连续无拼接，色温与原始照片一致
- **后脑勺**：干净均匀肤色，无五官幽灵（inpainting 填充）
- **额头**：有颜色覆盖，无黑块
- **侧面过渡区域**：从清晰正面自然过渡到 inpainting 肤色

---

## 修复方案：两阶段

### Phase 1: 高清直采（解决问题 A — 皮肤质感）

**核心思路**：绕过 512×512 的 unified_texture，直接用**原始高分辨率正面照片**进行纹理烘焙。

**为什么 unified_texture 融合方案在这里是反效果**：
1. 512×512 分辨率瓶颈 → 丢失 96% 像素
2. TPS 薄板样条变形 → 插值模糊（相当于低通滤波）
3. 三视角加权平均 → 进一步平滑细节
4. Reinhard 颜色迁移 → 可能偏移色调

对于医疗级术前/术后对比，正面区域的皮肤细节远比侧面覆盖完整性重要。正面相机已能覆盖 ~75-85% 的可见面部区域，剩余用 inpainting 补即可。

**数据流变化**：

```
旧：原始照片 → 缩放512 → TPS融合512 → bake_texture_single(512源) → UV 2048
新：原始照片(2160×3264) → bake_texture_single(高清源 + 缩放K) → UV 2048
```

**修改文件与内容**：

#### 1. `src/api/pipeline_runner.py`
- 保留原始高分辨率正面图像，传给纹理阶段
- 计算从 512→原始分辨率 的 K 缩放系数

```python
# 在 _run_pipeline 中，保存原始正面图像（不缩放）
original_front = images["front"]  # 原始分辨率

# ... 现有 preprocess / geometry 流程不变 ...

# 纹理阶段：传入原始高清图 + 缩放系数
scale_factor = max(original_front.shape[:2]) / 512  # ≈ 6.375
glb_path = run_texture_pipeline(
    ...
    hires_front_image=original_front,   # 新参数：原始分辨率正面图
    hires_scale_factor=scale_factor,     # 新参数：K 缩放系数
)
```

#### 2. `src/module3_texture.py` — `run_texture_pipeline`
- 新增 `hires_front_image` 和 `hires_scale_factor` 参数
- 当提供高清图时，缩放 front camera 的 K 矩阵，直接用高清图烘焙

```python
def run_texture_pipeline(
    ...
    hires_front_image: Optional[np.ndarray] = None,   # 新增
    hires_scale_factor: float = 1.0,                    # 新增
) -> Path:
```

在纹理烘焙分支中：
```python
if hires_front_image is not None:
    # 缩放 K 到匹配高清图分辨率
    front_cam = cameras.get("front", next(iter(cameras.values())))
    K_hires = front_cam["K"].copy()
    K_hires[0, :] *= hires_scale_factor  # fx, cx
    K_hires[1, :] *= hires_scale_factor  # fy, cy
    hires_cam = {"K": K_hires, "R": front_cam["R"], "t": front_cam["t"]}

    # 将原始图 resize 到等比 + 黑边（与预处理时相同逻辑，保持居中）
    from src.module1_preprocess import _resize_to_target
    hires_img = _resize_to_target(hires_front_image, 
                                   target=int(512 * hires_scale_factor))

    texture = bake_texture_single(
        ..., hires_cam, hires_img, tex_size,
    )
```

> **关键点**：`_resize_to_target` 做的是等比缩放 + 居中黑边填充。原始照片 2160×3264 在缩放到 512 时，是按 `scale = 512/3264 ≈ 0.157` 缩放后居中。我们需要保持同样的居中逻辑，只是在更高分辨率下。实际上更简单的做法是：把原始图用同样的 `_resize_to_target` 居中到 `max_size×max_size`（如 3264×3264），然后 K 按 `3264/512` 缩放即可。

#### 3. `src/module3_texture.py` — `bake_texture_single`
- 函数本身**不需修改**，它已经用 `unified_img.shape[:2]` 自适应图像尺寸
- 传入 3264×3264 的高清图 + 缩放后的 K，自动在高清图上采样

**预期效果**：
- UV 纹理从 3264×3264 源采样（而非 512×512）→ 每个 UV 纹素有 ~2.5 个源像素支撑（而非 0.06 个）
- 毛孔、眉毛细节、胡茬点、痘印 全部可见
- 无 TPS 变形模糊，无三视角混合平均

---

### Phase 2: 法线可见性剔除（解决问题 B + C — 后脑幽灵 + 额头缺失）

**修改文件**: `src/module3_texture.py` — `bake_texture_single` 函数（line 377-451）

**新增逻辑（在投影之前）**：

```python
# 1. 计算面法线（复用已有函数 compute_face_normals, line 187）
face_normals = compute_face_normals(vertices, faces)
pt_normals = face_normals[valid_tri]  # (M, 3)

# 2. 计算相机中心（FLAME 坐标系，与 bake_texture line 305-308 一致）
cam_center_proj = -R.T @ t
cam_center = cam_center_proj.copy()
cam_center[1] *= -1  # 还原 Y 翻转

# 3. 可见性判断
view_dirs = cam_center - pts_3d
view_dirs /= np.clip(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8, None)
cosines = np.sum(pt_normals * view_dirs, axis=1)
front_facing = cosines > 0.05  # 法线朝向相机（小阈值避免擦边）
```

**修改投影有效性条件**：

```python
# 旧: in_img = front & (proj in bounds)
# 新: in_img = front & front_facing & (proj in bounds)
```

**不修改的部分**：
- inpainting 逻辑（line 442-449）保持不变 → 自动填充后脑 + 额头空洞
- `bake_texture`（多视角版）→ 已有法线权重
- `module1b_texture_fusion.py` → 不再使用但保留（作为 fallback）

---

## 修改文件清单

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `src/module3_texture.py` | 修改 `bake_texture_single` 加法线检查；修改 `run_texture_pipeline` 加高清图参数 | ~30行 |
| `src/api/pipeline_runner.py` | 保存原始高清图 + 计算缩放系数 → 传给纹理流程 | ~10行 |

**不改**：`module1_preprocess.py`（512预处理用于几何拟合不变）、`module2_geometry.py`、`module1b_texture_fusion.py`（保留但不调用）

---

## 完整 Pipeline 及各步骤预期效果

### Step 1: 预处理 (`module1_preprocess.py`) — 不变
- **输入**: 3张原始照片（left/front/right）
- **输出**: 每个视角 → 512×512 图像 + 468 关键点 + face_mask
- **预期效果**: 人脸居中，关键点覆盖五官，face_mask 排除支架/背景
- **验证**: `debug/{view}_landmarks.png`、`debug/{view}_face_mask.png`

### Step 2: 三视角纹理预融合 (`module1b_texture_fusion.py`) — **跳过**
- 不再调用（高清直采方案下此步骤是反效果）
- 保留代码作为 fallback

### Step 3: 3DMM 几何重建 (`module2_geometry.py`) — 不变
- **输出**: `face_mesh.obj` + `cameras.json`（K 仍基于 512×512）
- **验证**: MeshLab 打开 .obj 检查几何

### Step 4: 纹理烘焙 — **核心修改**
- **输入**: mesh + front camera K（按原始分辨率缩放）+ 原始高清正面图
- **输出**: 2048×2048 UV 纹理图
- **预期效果**:
  - 正面区域：**清晰皮肤细节**（毛孔/胡茬/痘印可分辨），从高清图直接采色
  - 法线检查排除后脑面片 → 标记为空洞
  - 投影越界像素（额头顶部）→ 标记为空洞
  - 空洞区域 → inpainting 填充均匀肤色
  - 有效直采像素约占 50-60%（排除后脑 + 少量越界）
- **验证**: `textures/albedo_white.png` 正面区域应能看到毛孔细节

### Step 5: GLB 打包 — 不变

---

## 分层验证计划

### Layer 1: K 缩放正确性验证（快速脚本，不需要完整重建）

用已有 session 的 `cameras.json` + `face_mesh.obj`，对比 512 和高清两种 K 下的投影：

```
检查项:
□ 加载 cameras.json → front K → 缩放到原始分辨率
□ 取 mesh 鼻尖顶点 → 分别用 K_512 和 K_hires 投影 → 检查像素坐标比例一致
  预期: proj_hires / proj_512 ≈ scale_factor（±1像素误差）
□ _resize_to_target(原始图, target=3264) 后检查人脸居中位置与 512 版一致
```

### Layer 2: 法线可见性数值验证

```
检查项:
□ 计算 face_normals → 统计 cosine > 0 的面片比例
  预期: ~50% 正面，~50% 背面
□ 按 UV v 坐标分段，查看各段被排除率
  预期: 头顶/后脑排除率高，正面五官排除率 ≈ 0
□ 对比加法线检查前后 in_img 像素数
  预期: 减少 ~30-50%
```

### Layer 3: 纹理图视觉检查（提交完整重建）

```
检查项:
□ albedo_white.png 正面区域能看到毛孔/眉毛/胡茬细节 ← 核心指标
□ 后脑勺区域无五官幽灵
□ 额头有颜色（inpainting），无黑块
□ 正面→侧面过渡自然
```

### Layer 4: 3D 模型旋转检查 + 新旧对比

```
检查项:
□ 正面: 皮肤细节清晰，比旧版有质的飞跃
□ 45° 侧面: 纹理连续
□ 90° 侧面: inpainting 过渡自然
□ 后脑: 干净均匀
□ 与旧 session 并排对比: 正面清晰度提升明显
```

---

## 关于技术路线能否满足项目目标的评估

**项目目标**: 医疗级 3D 人脸重建，用于术前/术后对比。

**修复后能力评估**:

| 区域 | 质量 | 说明 |
|------|------|------|
| 正面五官（90% 关注度）| **高** | 原始高清照片直接采色，毛孔级细节 |
| 侧面过渡 | 中等 | 正面相机可见部分清晰，边缘 inpainting |
| 额头 | 中等 | 部分直采 + inpainting 补全 |
| 后脑/头顶 | 低 | 完全 inpainting，但临床不关注 |

**结论**: 高清直采方案直接解决了皮肤质感最核心的问题（分辨率瓶颈），法线剔除解决了后脑伪影。对于术前/术后**正面对比**这一核心场景，修复后的技术路线可以满足医疗级需求。

**潜在后续优化（不在本次范围）**:
- 多光照条件拍摄（cross-polarized）进一步减少高光反射
- 使用侧面相机高清图补充侧面纹理（而非 inpainting）
- 超分辨率网络增强 inpainting 区域质感

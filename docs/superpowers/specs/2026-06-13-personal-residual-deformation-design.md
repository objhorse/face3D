# Personal Residual Deformation 设计

日期：2026-06-13

## 背景

当前管线已经完成三视角 FLAME 拟合、shape-only 微调、相机姿态微调、深度置换和 UV 纹理融合。姿态微调后，左右侧脸相机对齐有所改善，但用户仍认为脸型不像。当前指标显示：

- 左侧密集轮廓误差约 43.7 px。
- 右侧密集轮廓误差约 41.9 px。
- 侧脸稀疏轮廓关键点均值约 25-26 px。
- 正面五官区域相对稳定，主要残差集中在侧脸脸颊、下颌和外轮廓。

这说明剩余误差主要不是相机姿态问题，而是 FLAME 低维 identity space 无法充分表达该个体脸型。因此需要在 FLAME 基础网格上增加一层受约束的个体残差形变。

## 目标

实现一个实验性的 `personal residual deformation` 阶段：

```text
最终个体几何 = FLAME identity/expression geometry + per-vertex residual offsets
```

第一版目标不是彻底解决真实扫描级重建，而是验证这层残差是否能显著改善侧脸脸型：

- 将左右侧脸密集轮廓误差从约 42-44 px 降到 25-32 px。
- 保持正面眼鼻嘴稳定，稳定区域误差不能明显恶化。
- 输出可视化审计页面，能清楚判断变形是否改善脸型或把脸拉坏。
- 若验收门控失败，自动回滚，不污染最终 mesh。

## 非目标

- 不改 FLAME 模型文件或 FLAME identity 参数维度。
- 不做完全自由的无约束 mesh 优化。
- 不解决眼球、口腔、鼻孔和纹理破碎问题；这些属于后续器官/纹理专项。
- 不引入新训练模型或网络依赖。

## 插入点

新增阶段放在 `shape_only + pose_refine` 之后、深度置换之前。

原因：

- 此时 FLAME identity 和相机姿态已相对稳定。
- 深度置换后再做大尺度脸型形变会破坏深度局部细节。
- UV 纹理阶段依赖最终 mesh 和 camera，因此 residual 需要先于纹理烘焙完成。

## 设计方案

### 1. 形变变量

为当前 FLAME/subdivision 前的基础人脸顶点计算 residual offset：

```text
offset[v] = (dx, dy, dz)
```

第一版采用直接几何求解/平滑传播，而不是高维可微优化器。这样更容易审计，也更容易回滚。

### 2. 可动区域

优先允许移动：

- 左右脸颊
- 下颌线
- 侧脸外轮廓
- 额头边缘

强保护区域：

- 眼睛和眼眶中心
- 鼻梁、鼻尖核心区
- 嘴唇和口腔附近
- 鼻孔附近

保护区域通过 2D landmark 半径投影到 3D 顶点集合，并在所有视角合并。

### 3. 约束来源

侧脸密集轮廓是主约束：

- 从每个侧脸 mask 提取逐行左右边界。
- 找到当前 mesh 投影在对应行的外轮廓候选顶点。
- 将候选顶点沿相机图像横向移动，使投影靠近 mask 边界。
- 左右侧视角权重大于正面。

正脸作为稳定约束：

- 正面可用于限制脸宽和下颌整体漂移。
- 正面眼鼻嘴附近不直接参与大位移 residual。

稀疏 landmark 作为辅助：

- 用于防止下颌点、脸颊点偏离过大。
- 不作为唯一目标，因为 68 点不足以定义真实脸型。

### 4. 平滑与正则

Residual offset 必须满足：

- 邻接顶点平滑，避免局部尖刺。
- 单顶点最大位移默认限制在 12-18 mm。
- 保护区最大位移接近 0。
- 移动顶点比例受限，避免整张脸被整体拉扯。
- 边跳变受限，避免产生折皱或断层。

### 5. 验收门控

形变完成后计算 before/after：

- 左右侧脸密集轮廓误差。
- 正面密集轮廓误差。
- 稳定五官 landmark 误差。
- 全局 landmark 均值。
- 最大位移、p95 位移、移动顶点比例、边跳变。

第一版验收规则：

- 至少一个侧脸密集轮廓改善超过 8 px。
- 两个侧脸平均密集轮廓改善为正。
- 正面稳定五官不能恶化超过 0.5-1.0 px。
- 全局 landmark 均值不能明显恶化。
- 最大位移不超过配置阈值。
- 安全指标失败则回滚。

### 6. 输出与审计

新增输出目录：

```text
output/debug/personal_residual_deform/
```

输出文件：

- `summary.json`：记录配置、before/after 指标、验收原因。
- `index.html`：中文审计页面。
- `*_before_contour.png` / `*_after_contour.png`：轮廓叠加。
- `*_after_mesh.png`：变形后投影。
- `residual_heatmap_*.png`：残差位移热力图。

主评测 HTML 后续读取该 summary，将脸型残差改善写入报告。

## 配置

新增配置项：

```python
ENABLE_PERSONAL_RESIDUAL_DEFORM = True
PERSONAL_RESIDUAL_MAX_OFFSET_M = 0.018
PERSONAL_RESIDUAL_TARGET_SIDE_IMPROVE_PX = 8.0
PERSONAL_RESIDUAL_MAX_STABLE_WORSEN_PX = 0.75
PERSONAL_RESIDUAL_VIEW_WEIGHT_FRONT = 0.35
PERSONAL_RESIDUAL_VIEW_WEIGHT_SIDE = 1.0
PERSONAL_RESIDUAL_SMOOTH_ITER = 30
PERSONAL_RESIDUAL_SMOOTH_ALPHA = 0.30
PERSONAL_RESIDUAL_CONSTRAINT_KEEP = 0.55
```

这些阈值用于第一版实验，后续根据审计结果调参。

## 失败与回滚

该阶段必须是保守的：

- 任一硬安全门控失败，返回原始 `verts_base` 和 `verts_displaced`。
- summary 中记录失败原因。
- Debug 图仍保留，便于判断是约束方向错、保护区太大，还是平滑/门控太严。

## 测试计划

1. 运行语法检查：

```powershell
python -m py_compile src\module2_geometry.py src\config.py tools\generate_face_evaluation_report.py
```

2. 运行 Phase 1：

```powershell
python -u run_phase1.py
```

3. 检查：

- `output/debug/personal_residual_deform/summary.json`
- `output/debug/personal_residual_deform/index.html`
- `output/debug/optimized_shape.json`

4. 若 residual 被接受，继续运行 Phase 3 和评测报告：

```powershell
python -u run_phase3.py
python tools\visualize_texture_audit.py
python tools\visualize_pipeline_audit.py
python tools\generate_face_evaluation_report.py
```

5. 浏览器验证主报告和 residual 审计页。

## 风险

- 三视角约束不足，侧脸 residual 可能改善轮廓但破坏真实 3D 体积。
- mask 边界若包含头发或阴影，会把脸型拉向错误边界。
- 过强平滑会吃掉收益，过弱平滑会产生局部鼓包。
- 该阶段不能解决眼睛、口腔、鼻孔纹理缺陷。

## 决策

采用受约束 personal residual deformation 作为下一步实验。它保留 FLAME 的稳定拓扑和表情基础，同时用可回滚的 per-vertex residual 补偿个体脸型差异。第一版重点验证侧脸轮廓是否能明显下降，而不是追求最终完美人脸。

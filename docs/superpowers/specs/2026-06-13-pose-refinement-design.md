# 相机姿态微调设计：缩小 UV 渲染与原图差异

## 目标

在不修改纹理融合、不直接拉扯最终 mesh 的前提下，先修正三视图相机姿态，使纯 UV 渲染和中性材质渲染投影回原图时更贴近真实脸型与五官位置。

本阶段优化目标是“三视图均衡”，不是只让正面好看。成功标准是正面、左侧、右侧的重投影与轮廓指标整体下降，并且任何一个视角不能明显变差。

## 当前问题依据

最新完整管线评测显示：

- 正面 68 点重投影平均误差约 17 px，左右侧约 14 px。
- 自由轮廓变形主要改善正面，左右侧最终稠密轮廓误差仍约 86.8 px。
- 纯 UV 渲染比 UV 叠加更不像原图，原因之一是叠加图保留了原始照片的真实脸型边界，而纯渲染暴露了相机/几何对齐偏差。
- 纹理阶段已经能生成可用 GLB，因此下一步应优先减少投影偏差，再处理眼部、口鼻和发际纹理问题。

## 推荐方案

新增一个 `pose_refinement` 步骤，只优化每个视角的相机姿态 `R/t`，不优化 FLAME shape、不修改顶点、不修改 UV。

插入位置：

1. 初始化得到每个视角 `R/t`。
2. 完成现有 LBFGS 与 shape-only fine tune。
3. 执行 pose refinement，更新 `per_view_results[view]["R"]` 和 `["t"]`。
4. 保存 before/after 投影调试图与 JSON 摘要。
5. 继续后续深度置换、自由修脸、导出 mesh 和 `cameras.json`。

这样后续纹理烘焙会使用更准确的相机位姿，但 mesh 几何本身不会被这个步骤直接改变。

## 损失函数

每个视角独立优化 6 个小参数：

- 旋转增量 `delta_rvec`
- 平移增量 `delta_t`

基础投影采用现有 `project_vertices` 逻辑。损失由四部分组成：

| 损失项 | 作用 |
| --- | --- |
| 68 点 landmark reprojection loss | 约束五官、下颌和基础脸型 |
| 稳定关键点加权 loss | 鼻梁、眼区、嘴内轮廓不能漂移 |
| dense contour loss | 约束侧脸脸缘、下颌、颧骨外轮廓 |
| 姿态正则 | 限制相机旋转和平移偏离原始估计 |

首版不引入光度误差和纹理误差，避免把光照、表情、眼球纹理问题误当成姿态问题。

## 接受与回退条件

每个视角独立判断是否接受新姿态。默认接受条件：

- 当前视角 68 点平均误差至少下降 1.0 px，或 dense contour 误差至少下降 3.0 px。
- 当前视角 68 点最大误差不能增加超过 5.0 px。
- 稳定关键点平均误差不能增加超过 1.0 px。
- 旋转增量不超过 6 度。
- 平移增量不超过当前深度尺度的 8%。

全局多视角验证：

- 三视图平均 landmark 误差不能变差。
- 正面平均误差不能变差超过 0.5 px。
- 任一侧脸 dense contour 不能变差超过 2 px。
- 如果全局验证失败，全部回退到原始相机姿态。

## 配置项

新增配置集中放在 `src/config.py`：

```python
ENABLE_POSE_REFINEMENT = True
POSE_REFINE_MAX_ITER = 80
POSE_REFINE_LR = 0.01
POSE_REFINE_LMK_WEIGHT = 1.0
POSE_REFINE_STABLE_WEIGHT = 1.4
POSE_REFINE_DENSE_CONTOUR_WEIGHT = 0.35
POSE_REFINE_ROT_REG = 0.02
POSE_REFINE_TRANS_REG = 0.02
POSE_REFINE_MIN_MEAN_IMPROVE_PX = 1.0
POSE_REFINE_MIN_DENSE_IMPROVE_PX = 3.0
POSE_REFINE_MAX_STABLE_WORSEN_PX = 1.0
POSE_REFINE_MAX_MAXERR_WORSEN_PX = 5.0
POSE_REFINE_MAX_ROT_DEG = 6.0
POSE_REFINE_MAX_TRANS_REL = 0.08
```

默认开启，但所有阈值都要保守。若实验中发现姿态微调会牺牲侧脸或引入纹理错位，应立即回退。

## 输出产物

新增目录：

```text
output/debug/pose_refinement/
```

其中包含：

- `summary.json`：每个视角 before/after 指标、接受状态、回退原因。
- `{view}_before_reprojection.png`
- `{view}_after_reprojection.png`
- `{view}_before_mesh.png`
- `{view}_after_mesh.png`
- `index.html`：三视图 before/after 可视化对比。

后续评测报告读取这些产物，将 UV 渲染和原图差异缩小情况可视化。

## 验证流程

实施后跑完整管线：

1. `run_phase1.py`
2. `run_phase3.py`
3. `tools/visualize_texture_audit.py`
4. `tools/visualize_pipeline_audit.py`
5. `tools/generate_face_evaluation_report.py`

关键验收指标：

- 正面平均重投影误差目标从约 17 px 降到 10-12 px。
- 左右侧不允许明显变差，理想情况下降到 10-12 px。
- 侧脸 dense contour 至少有一个视角明显改善。
- 纯 UV 渲染的脸型边界、下颌和五官位置比当前报告更接近原图。
- 中性材质视图不能出现新的脸型扭曲。

## 不做的事情

本阶段不处理：

- 眼球/眼眶黑洞。
- 嘴腔、牙齿、鼻孔纹理破碎。
- 发际、耳侧裁剪边界。
- 深度置换负尺度回退。
- 直接增大自由修脸位移。

这些问题应在相机/几何对齐稳定后单独处理。

## 风险

- 相机姿态优化可能把投影指标变好，但让后续纹理采样视角发生偏移。
- 侧脸 dense contour 与 68 点 landmark 可能互相冲突。
- 如果姿态正则太弱，模型会出现“相机迁就错误 shape”的假改善。

因此第一版必须带完整回退逻辑和 before/after 可视化，不能只凭最终 GLB 判断。

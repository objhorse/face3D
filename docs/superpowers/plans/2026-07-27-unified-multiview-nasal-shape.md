# 统一多视角鼻部形状优化实施计划

> 对应设计：
> `docs/superpowers/specs/2026-07-27-unified-multiview-nasal-shape-design.md`

## 实施目标

在不改变三视角采集、固定 rig 外参、FLAME 拓扑、表情和纹理的前提下，
使用一套统一目标函数同时优化：

- 鼻翼总宽度。
- 左右鼻翼轻微不对称。
- 鼻头前后深度。
- 鼻头高度和圆度。
- 鼻头到鼻翼的过渡形状。

新实验从与 `profile_shape_v2` 相同的
`protected_expression_depth_v3` 基线开始，不叠加旧的单一鼻尖深度结果。
这样可以直接比较统一优化与旧方法，而不会把两次形变混在一起。

## 实施原则

- 先验证图像证据，再优化几何。
- 使用一个连续目标函数，不按部位逐项添加硬阈值。
- 纹理不进入优化目标。
- FLAME shape 与低维鼻部语义参数联合求解。
- 不允许逐顶点自由残差。
- 两个数据集使用相同配置，禁止单独调权重。
- 第一轮只做独立实验，不替换 stable pipeline 默认结果。
- 每个批次先写测试，再实现，再生成可检查产物。
- 当前 dirty worktree 中的其他实验改动保持不动。

## Batch A：统一鼻部观测

### A1. 定义数据结构和坐标约定

新增 `src/geometry/nasal_observations.py`：

- `NasalObservationConfig`
  - 工作分辨率。
  - mask 扰动尺度。
  - 图像梯度窗口。
  - 正脸与侧脸 ROI 范围。
  - 距离场截断距离。
- `NasalViewObservation`
  - 视角名称和被摄者左右语义。
  - 原图尺寸和工作尺寸。
  - 二值观测边界。
  - signed/unsigned distance field。
  - 连续 confidence map。
  - 相机引用和坐标变换元数据。
- `NasalObservationBundle`
  - front、subject-left、subject-right 三个观测。
  - 统一相机命名和数据集映射。

复用：

- `src.cross_view_geometry.Camera`
- `src.geometry.profile_silhouette_extrema` 中的图像坐标转换。
- stable pipeline 已缓存的 face mask 和 semantic mask。
- 已确认的固定 joint-rig 标定文件。

新增 `tests/test_nasal_observations.py`：

- camera1、camera2、camera3 正确映射到被摄者左右语义。
- 原图、letterbox canvas、工作分辨率往返误差小于数值容差。
- 空 mask、越界 ROI 和尺寸不一致时给出明确错误。

### A2. 正脸鼻翼边界观测

在 `nasal_observations.py` 中实现：

- 从正脸 semantic nose 区域提取左右外侧鼻翼边界。
- 使用鼻部中心线将边界分为 subject-left 和 subject-right。
- 将边界转换为距离场。
- 计算 mask erosion/base/dilation 的边界稳定度。
- 结合局部图像梯度，形成连续 confidence。
- 鼻孔暗区、强高光和语义不稳定位置只降低 confidence，不删除整段边界。

测试：

- 合成鼻部 mask 的左右鼻翼位置可被正确识别。
- 左右不对称不会被错误镜像或强制对称。
- mask 边界扰动越大，confidence 越低。
- 图像梯度与 mask 边界一致时，confidence 连续提高。

### A3. 双侧鼻头轮廓观测

在 `nasal_observations.py` 中实现：

- 使用 front 鼻部锚点和 rig 外参限定侧脸 epipolar ROI。
- 在侧脸 face silhouette 上提取鼻头局部曲线。
- 保留上缘、最前点、下缘和鼻翼过渡区的多点轮廓。
- 对两侧分别生成距离场与 confidence，不先合并成一个标量。
- 记录各 mask 扰动版本的轮廓和稳定性。

测试：

- 合成侧脸轮廓能恢复鼻头上缘、顶点和下缘。
- 同一曲线的小范围扰动产生连续误差，不发生点编号跳变。
- 一侧低置信不会污染另一侧观测。
- 轮廓搜索不会落到嘴唇、眼睑或脸颊外轮廓。

### A4. 只读观测诊断

新增 `run_nasal_observation_audit.py`：

```powershell
python run_nasal_observation_audit.py `
  --capture-dir D:\face3D\captures_20260612_135253 `
  --source-output output\experiments\captures_20260612_135253_protected_expression_depth_v3 `
  --output output\experiments\captures_20260612_135253_nasal_observation_v1
```

输出：

- `nasal_observations.json`
- `debug/nasal_observations/index.html`
- 三视角原图、边界、confidence 和 baseline 几何投影叠加图。

对 `135253` 和 `140210` 都运行。进入 Batch B 前人工确认：

- 正脸边界确实位于鼻翼外缘。
- 两侧轮廓确实沿鼻头皮肤外轮廓。
- 没有把鼻孔阴影、嘴唇或贴图颜色当作几何边界。

## Batch B：低维鼻部生成模型

### B1. 构建语义支持区域

新增 `src/geometry/nasal_semantic_basis.py`：

- 复用 `build_default_nose_mouth_control_seeds()` 的鼻梁、鼻头和左右鼻翼种子。
- 使用 mesh geodesic 距离构建紧支撑、平滑衰减的权重。
- 明确排除嘴、人中、眼睛、下巴和 jaw 种子。
- 输出每个语义参数的 vertex support、方向和归一化尺度。

测试：

- 所有 basis 有限且维度一致。
- support 只落在鼻部拓扑邻域。
- 零参数严格复现 baseline。
- 嘴、眼、jaw 的直接 basis 位移为零。

### B2. 构建可解释参数 basis

生成以下固定语义模式：

- `alar_width_shared`
- `alar_width_asymmetry`
- `alar_depth_shared`
- `alar_depth_asymmetry`
- `tip_depth`
- `tip_vertical`
- `tip_roundness`
- `tip_alar_fullness`

实现要求：

- 宽度沿 subject-left/right 横向方向运动。
- 深度沿 front camera 的前后方向运动。
- 圆度通过鼻头局部法线和拓扑位置形成平滑凸度变化。
- asymmetry 模式对左右鼻翼符号相反。
- basis 在鼻部加权内积下做正交化和尺度归一化。
- 不通过扩大 support 来获得更大变化。

测试：

- 每个参数的正负方向符合语义。
- shared 模式保持左右同向，asymmetry 模式保持左右反向。
- roundness 不等价于单纯 tip depth。
- 任意小参数组合产生连续、无折痕位移。

### B3. 构建可观测 FLAME 子空间

不直接对全部 FLAME shape 参数做有限差分。新增：

- 根据 FLAME shape basis 在鼻部与受保护区域的响应比筛选候选模式。
- 对三视角鼻部投影 Jacobian 做 SVD。
- 保留可由当前观测解释的低秩 FLAME 子空间。
- 子空间选择规则和奇异值相对阈值固定，不按数据集手调。
- 保存选择的模式、奇异值和可观测度报告。

测试：

- 无鼻部响应的 shape 模式不会进入子空间。
- 合成可观测模式能被保留。
- rank-deficient 输入会稳定降秩，而不是产生巨大系数。
- 两个数据集使用相同的选择规则。

## Batch C：统一多视角目标函数

### C1. 投影鼻部模型边界

新增 `src/geometry/multiview_nasal_objective.py`：

- 将候选 FLAME + semantic basis 生成的 mesh 投影到固定三相机。
- 正脸提取模型鼻翼语义边界采样。
- 侧脸提取 view-dependent 鼻头 silhouette 采样。
- 使用 z-buffer/朝向过滤被遮挡点。
- 返回每个残差对应的视角、像素位置、confidence 和语义来源。

测试：

- 已知相机与合成 mesh 投影准确。
- 被遮挡的后侧鼻翼不会进入残差。
- subject-left/right 语义不会随图像左右翻转。
- 候选 mesh 不会修改 camera、expression 或 topology。

### C2. 实现统一残差

参数：

```text
theta = [observable_flame_coefficients, semantic_nasal_coefficients]
```

残差向量一次性包含：

- 正脸 confidence-weighted 鼻翼边界距离。
- subject-left 侧脸鼻头轮廓距离。
- subject-right 侧脸鼻头轮廓距离。
- FLAME/MICA 高斯先验。
- semantic 参数零均值先验。
- 鼻部曲率和平滑性残差。
- 由观测置信度调节的弱对称残差。

要求：

- 使用 robust loss 降低局部错误边界的影响。
- 每项残差按有效观测数量归一化。
- objective 权重来自固定配置，不允许 runner 覆盖为数据集专用值。
- 没有逐部位 pass/fail 阈值。
- 缺少证据时，对应图像项自然减弱，先验接管。

测试：

- 合成已知鼻翼变宽时，正确参数使总 objective 下降。
- 合成鼻头变圆时，roundness 比单纯 depth 得分更低。
- 只提供单侧可靠证据时，弱对称先验能抑制无依据的另一侧畸变。
- 对称和非对称真实目标均可恢复。
- confidence 为零的像素不贡献梯度。

### C3. 优化器

新增：

- `NasalOptimizationConfig`
- `fit_multiview_nasal_shape()`
- iteration trace 和 objective term trace。

首版使用 SciPy robust least-squares 与有限差分 Jacobian：

- 初始化所有更新为零。
- 低分辨率和全分辨率使用同一 objective，只改变采样密度。
- 参数尺度由 basis 归一化决定。
- 优化器停止条件只依据数值收敛，不依据某部位人工阈值。
- 失败时返回完整诊断，不覆盖 baseline 产物。

测试：

- 零残差输入保持零更新。
- 合成目标从不同小初始化收敛到相近解。
- objective trace 单调或在 robust solver 容差内稳定收敛。
- NaN、奇异 Jacobian 和无有效观测产生清晰失败原因。

## Batch D：实验导出与双数据集验证

### D1. 独立实验 runner

新增 `run_multiview_nasal_shape_experiment.py`：

参数：

- `--capture-dir`
- `--source-output`
- `--output`
- `--rig-calibration`
- `--viewer-template`

流程：

1. 加载与 `profile_shape_v2` 相同的 baseline。
2. 构建三视角观测。
3. 构建 FLAME 子空间与 semantic basis。
4. 执行统一优化。
5. 使用冻结 expression 生成候选。
6. 执行最小有效性检查。
7. 使用 baseline 的原始 UV 和纹理导出候选 GLB。
8. 生成几何诊断与离线 A/B viewer。

### D2. 最小有效性检查

只保留：

- 参数和顶点无 NaN/Inf。
- 顶点、面和 UV 拓扑不变。
- 无新增退化面、翻面或非流形边。

边界拟合、对称、平滑和参数幅度都属于 objective 或报告，不再建立
逐部位硬拒绝门禁。

### D3. 输出产物

每个数据集输出：

- `nasal_observations.json`
- `nasal_fit_report.json`
- `meshes/face_same_texture.glb`
- `debug/nasal_geometry/index.html`
- `nasal_shape_compare.html`
- front/subject-left/subject-right 离线截图。

viewer：

- baseline 和 candidate 使用完全相同纹理。
- 初始相机姿态一致。
- 支持同步旋转或固定正面/左右侧预设。
- 页面明确标注当前数据集和模型角色。
- 所有 GLB 资源嵌入 HTML，可通过 `file:///` 打开。

### D4. 双数据集运行

使用完全相同配置运行：

1. `D:\face3D\captures_20260612_135253`
2. `D:\face3D\captures_20260612_140210`

分别从各自的 `protected_expression_depth_v3` 输出开始。

检查：

- `135253` 在证据已接近 baseline 时不会出现无依据的大幅变形。
- `140210` 同时改善鼻翼宽度和鼻头轮廓，不只增加鼻尖深度。
- 两个结果都没有鼻头折痕、鼻翼断裂或嘴部联动。
- 纯几何投影改善与带纹理视觉改善方向一致。

### D5. 与旧实验对照

报告同时记录：

- 原 baseline。
- `profile_shape_v2` 的单鼻尖标量结果。
- 新统一优化 candidate。

正式 A/B viewer 仍只比较 baseline 与新 candidate。旧实验只作为报告中的
参考指标，避免三栏 viewer 干扰直观判断。

## Batch E：回归与是否接入主流程

### E1. 聚焦测试

```powershell
D:\Anaconda\envs\gaussian\python.exe -B -m pytest `
  tests\test_nasal_observations.py `
  tests\test_nasal_semantic_basis.py `
  tests\test_multiview_nasal_objective.py -q
```

### E2. 相关回归

```powershell
D:\Anaconda\envs\gaussian\python.exe -B -m pytest `
  tests\test_profile_silhouette_extrema.py `
  tests\test_profile_shape_fit.py `
  tests\test_expression_depth.py `
  tests\test_mesh_quality.py `
  tests\test_semantic_epipolar_refinement.py `
  tests\test_joint_rig_refinement.py -q
```

### E3. 人工验收后再决定集成

第一轮实验完成后暂停在两个 A/B viewer，由用户判断真人相似度。只有新方法
在两个数据集上都优于 baseline，才另开集成批次：

- 在 `stable_three_view.py` 增加可选 nasal optimization stage。
- 默认 pipeline 是否启用由用户确认。
- 保留旧 pipeline 和 profile 实验作为回退。

本实施计划不直接修改默认 pipeline。

## 提交策略

- 计划文档独立提交。
- Batch A、B、C、D 分别提交，避免把观测错误和优化错误混在一起。
- 每次只暂存该批次文件，不带入现有 dirty worktree 的无关修改。
- 大型 GLB、图片和 HTML 实验产物不提交。
- 在用户看过最终 A/B viewer 前，不提交默认 pipeline 切换。

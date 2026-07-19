# 三视角观测驱动的口鼻侧貌深度实施计划

设计依据：`docs/superpowers/specs/2026-07-19-observation-driven-profile-depth-design.md`

目标：在固定三相机 rig 下，从当前三张照片恢复个体 nose/lip/chin 前后关系，先约束 expression 不再把闭合嘴部向前推，再用可靠三维观测修正 MICA-anchored shape。全程保持 FLAME 固定拓扑，不使用自由顶点残差。

## Batch A：侧貌深度基线与质量度量

### 文件

- 新增 `src/geometry/profile_depth_quality.py`
- 新增 `tests/test_profile_depth_quality.py`
- 新增 `run_profile_depth_audit.py`

### 实现

1. 从 FLAME semantic regions 与同拓扑 neutral/final mesh 中提取 nose tip、mouth、chin 的前后坐标。
2. 计算 nose-tip-to-upper-lip、nose-tip-to-lower-lip、subnasale-to-lip、lip-to-chin 深度及 expression mouth-depth displacement。
3. 支持绝对单位和按 face width 归一化的指标。
4. 将当前结果、MICA anchor、optimized neutral、optimized expression 分阶段输出到 `profile_depth_baseline.json`。
5. 质量函数只测量，不使用统一人群比例修改模型。

### 测试

- 全局刚体平移不改变相对深度。
- 嘴部整体前移能被 expression displacement 指标检出。
- semantic region 缺失、顶点数不一致和非有限顶点明确失败。
- 当前 `captures_20260612_140210` 能复现约 8 mm neutral、约 4 mm final 的阶段差异。

## Batch B：固定 Rig 三角化内核

### 文件

- 新增 `src/geometry/profile_triangulation.py`
- 新增 `tests/test_profile_triangulation.py`

### 实现

1. 读取当前 calibration schema，统一 left/front/right 相机语义。
2. 将相机投影矩阵统一到 front-camera reference frame。
3. 对两视图和三视图 observation 实现加权 DLT 三角化。
4. 计算正深度、视线夹角、逐视图重投影、pair 间三维一致性和置信度。
5. 明确记录标定平移单位；单位未知时同时输出 raw 和 face-width-normalized 指标。
6. rig 或 camera naming 不一致时直接失败，不回退到独立相机姿态。

### 测试

- 合成三相机与已知三维点可在亚像素噪声下恢复。
- camera1/camera3 交换会被正深度或重投影门禁拒绝。
- 低视差、负深度、单视图离群点和错误尺度均被识别。
- 结果与输入视图字典顺序无关。

## Batch C：固定语义点观测与只诊断实验

### 文件

- 新增 `src/geometry/profile_observations.py`
- 新增 `tests/test_profile_observations.py`
- 扩展 `run_profile_depth_audit.py`

### 实现

1. 从预处理结果收集 MediaPipe dense landmarks 与 face_alignment 68 landmarks。
2. 建立 nose tip、subnasale、upper lip、lower lip、mouth center、chin 的固定语义定义。
3. 在小范围 semantic ROI 与 epipolar line 内做局部 refinement；禁止把任意纹理匹配直接当作解剖点。
4. 为每点输出 detector agreement、边界响应、视角可见性和最终 observation confidence。
5. 调用 Batch B 三角化，生成 PLY、三视角重投影图和 JSON。
6. 先在 `captures_20260612_140210` 与 `captures_20260612_135253` 上运行，不修改 mesh。

### 输出

- `debug/profile_depth/observations_left.png`
- `debug/profile_depth/observations_front.png`
- `debug/profile_depth/observations_right.png`
- `debug/profile_depth/triangulated_points.ply`
- `debug/profile_depth/reprojection_<view>.png`
- `debug/profile_depth/profile_depth_quality.json`
- `debug/profile_depth/index.html`

### 验收

- 至少 nose tip、一个 lip point 和 chin 有效。
- 至少四个中线语义点通过三角化。
- 三视角重投影 P90 不高于 5 px。
- 两个数据集的 nose/lip/chin 深度顺序与输入侧脸视觉一致。

若 Batch C 不通过，停止后续几何修改，优先修相机语义、标定或语义点定义。

## Batch D：Expression 口周深度约束

### 文件

- 新增 `src/geometry/expression_depth.py`
- 新增 `tests/test_expression_depth.py`
- 修改 `src/geometry/expression_fidelity.py`
- 修改 `src/module2_geometry.py`
- 修改 `src/config.py`

### 实现

1. 预计算每个 FLAME expression basis 对 mouth ROI 前后位移、嘴缝闭合和眼睑闭合的影响。
2. closed-mouth 数据对高 mouth-depth influence 模式施加强正则。
3. expression fidelity 同时检查二维嘴缝和三维 mouth-depth displacement。
4. 闭眼几何优先交给 semantic eyelid rig，通用 expression 不再以改变口鼻深度换取眼睑闭合。
5. 当三维观测有效时，expression final profile 必须匹配观测；无观测时采用 neutral-mouth preservation 回退。
6. 门禁失败时回退到受限 expression，不改变 neutral identity mesh。

### 测试

- 合成 expression 向前推嘴但保持嘴缝闭合时必须失败。
- 仅闭合眼睑、口周不动的 expression 可以通过。
- 受限 expression 不改变 nose/lip/chin semantic ordering。
- 旧二维 eye/mouth gap 测试继续通过。

### 验收

- `captures_20260612_140210` 的 expression mouth forward displacement 从约 3 mm 降到 0.5 mm 内，或匹配可靠三维观测。
- 不新增退化面、翻面或非流形边。

## Batch E：MICA-Anchored Shape 三维侧貌拟合

### 文件

- 新增 `src/geometry/profile_depth_fit.py`
- 新增 `tests/test_profile_depth_fit.py`
- 修改 `src/module2_geometry.py`
- 修改 `src/config.py`

### 实现

1. 固定 rig relative extrinsics，只优化一个全局 head pose。
2. 在 neutral expression 下优化 FLAME shape coefficients。
3. 损失包含稳定内部二维 landmarks、三维 semantic points、相对 profile depth、MICA anchor 和 mean-shape prior。
4. 每个三维点按 observation confidence 加权并使用 Huber loss。
5. 候选必须通过 profile improvement、二维 interior preservation、identity drift 和 mesh quality 四类门禁。
6. 三维观测不足时精确保留 MICA baseline，禁止启用自由 residual。

### 测试

- 合成 shape target 可恢复预设 nose/lip/chin 相对深度。
- 全局 pose/scale 变化不会被错误吸收到局部口周形状。
- 低置信三维点不会主导 shape。
- 门禁拒绝后顶点和参数与 baseline 完全一致。

### 验收

- 两个数据集 final profile 与各自三角化目标误差不高于 2 mm 或等价归一化容差。
- 不是简单把所有样本拉向 FLAME mean face。
- 正面脸宽、眼鼻口二维内部投影不发生明显恶化。

## Batch F：主 Pipeline、报告与回归

### 文件

- 修改 `src/pipeline/stable_three_view.py`
- 修改 `src/geometry/template_fit.py`
- 修改 `src/reports/stable_reconstruction_report.py`
- 修改 pipeline 相关测试

### 实现

1. 在 stable fit 前检查固定 rig 可用性与 camera naming。
2. 将 Batch C observation、Batch E shape 和 Batch D expression 按固定顺序接入。
3. 纹理仍只在最终通过 profile-depth gate 的 mesh 上烘焙。
4. session metadata 增加 rig、triangulation、neutral profile、expression displacement 和 final profile 摘要。
5. 质量报告并列显示输入侧脸、neutral 纯几何和 final 纯几何，不让纹理影响形状判断。
6. 保留配置开关用于 A/B，但 stable 默认只有在两个数据集验收后切换到新路径。

### 回归

- `captures_20260612_140210` 全流程重跑。
- `captures_20260612_135253` 全流程重跑。
- 固定拓扑约 79,936 顶点、159,616 面。
- 无新增 NaN/Inf、退化面、非流形边或删面。
- 旧 stable baseline 可由配置复现。
- API、GLB 路径和离线 viewer 保持兼容。

## 提交策略

- 每个 Batch 独立提交，避免与当前工作树中的纹理、眼睑和旧 silhouette 实验混合。
- Batch A-C 只增加诊断能力，不修改最终 `face.glb`。
- Batch C 验收前不开始 Batch D-E。
- Batch D 与 Batch E 分别提供 neutral/expression A/B viewer；最终是否接入主 pipeline 以输入照片的侧貌对比为准。

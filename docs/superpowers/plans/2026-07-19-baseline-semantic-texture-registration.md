# Baseline 正脸五官分区语义纹理配准实施计划

## 目标

在不运行几何 pipeline、不修改 Baseline 顶点/面片/UV/相机的前提下，为 `captures_20260612_135253_controlled_identity_v1` 重新烘焙正脸五官纹理。第一轮先完成鼻部，验证通过后再启用嘴和眼睛。

## Batch A：冻结 Baseline

### 文件

- 新增 `src/appearance/baseline_texture_lock.py`
- 新增 `tests/test_baseline_texture_lock.py`
- 新增 `run_baseline_texture_registration_experiment.py`

### 工作

1. 定义冻结清单：OBJ、相机、UV 拓扑和原始 GLB。
2. 计算并校验 SHA-256；哈希不一致立即停止。
3. 实验入口只复制/读取冻结 mesh，不调用 `run_stable_template_fit`。
4. 烘焙后重新读取 OBJ，比较顶点、面片和 UV 数组完全一致。

### 验证

- 正确文件通过。
- 任意顶点、面片、UV 或相机变化均被拒绝。
- 输出目录不覆盖 Baseline。

## Batch B：有序鼻部观测

### 文件

- 新增 `src/appearance/semantic_feature_registration.py`
- 新增 `tests/test_semantic_feature_registration.py`
- 修改 `src/appearance/stable_texture_registration.py`

### 工作

1. 从投影 FLAME 68 点构造鼻翼左右、鼻头下缘的有序模型曲线。
2. 从 `nose_mask` 和鼻部 ROI 构造有序照片边界。
3. 按左右侧和归一化弧长配对，禁止最近邻自由匹配。
4. 鼻孔候选按人脸中心线划分左右连通区域，记录中心、宽高和置信度。
5. 输出配准前叠加图、对应线和分项 JSON；本批次不改纹理。

### 验证

- 左侧模型点只能匹配左侧照片点，右侧同理。
- 每条曲线的对应顺序单调，不交叉。
- 观测不足时明确返回低置信，不伪造控制点。

## Batch C：整体微调与鼻部局部场

### 文件

- 扩展 `src/appearance/texture_registration.py`
- 扩展 `tests/test_texture_registration.py`
- 修改 `src/appearance/stable_texture_registration.py`

### 工作

1. 使用眼角、鼻中心、嘴角估计有界二维相似变换。
2. 在相似变换后，用有序鼻部控制点拟合鼻区局部场。
3. 局部场在鼻区核心生效，在过渡环衰减为零。
4. 计算位移 P95、最大值和 Jacobian；失败时回退到整体微调。
5. 组合成兼容现有 `SamplingWarp.apply()` 的分层采样 warp。

### 验证

- 平移、旋转、尺度和局部位移满足设计上限。
- 鼻区之外位移接近零。
- Jacobian 门禁能拒绝折叠场。

## Batch D：正脸五官纹理所有权

### 文件

- 修改 `src/module3_texture.py`
- 扩展 `tests/test_texture_blending.py`

### 工作

1. 将单一 `feature_masks` 扩展为核心权重图。
2. 鼻部核心由正脸照片至少占 `95%`，侧视图权重归零。
3. 在过渡环恢复多视图混合，避免硬边。
4. 记录各五官核心的来源比例。

### 验证

- 侧视图不能覆盖鼻孔和鼻翼核心。
- 过渡区权重连续。
- 现有无删面和透明度门禁继续通过。

## Batch E：只重跑纹理与 A/B

### 文件

- 完成 `run_baseline_texture_registration_experiment.py`
- 复用 `tools/embed_glb_assets.py`

### 工作

1. 加载冻结 Baseline 和三张原图。
2. 运行鼻部观测、分层 warp 和纹理烘焙。
3. 输出到 `output/experiments/captures_20260612_135253_baseline_texture_nose_v2`。
4. 生成 Baseline/候选离线 Viewer、正面截图和鼻部近景。
5. 输出 `semantic_feature_report.json` 与几何哈希证明。

### 验收

- 几何与相机哈希完全不变。
- 鼻部语义边界误差改善至少 `20%`。
- 鼻孔左右顺序正确，P90 锚点误差不高于 `4 px`。
- 最大局部位移不高于 `16 px`，P95 不高于 `10 px`。
- 最小 Jacobian 大于 `0.35`。
- 用户在相同几何的带纹理 A/B 中判断候选不低于 Baseline。

## 后续批次

鼻部验收通过后，复用 Batch B-C 的接口依次加入嘴唇和眼睑。未通过前不并行启用，避免无法判断是哪一个局部场造成退化。

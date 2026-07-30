# 三视角语义眼睑 Rig 实施计划

## Batch A：眼睛状态共识

### 文件

- 新增 `src/geometry/eye_state.py`
- 新增 `tests/test_eye_state.py`

### 实现

1. 从每视角 MediaPipe 478 点分别计算左右眼 aspect ratio。
2. 输出每只眼的 `open/closed/uncertain` 和置信度。
3. 以正脸优先、三视角多数一致的方式汇总共享状态。
4. 冲突且无法形成共识时返回 `uncertain`，禁止局部形变。

### 验证

- 双眼睁开、双眼闭合、单侧闭合、侧视图冲突和低置信输入。
- 汇总结果与视角输入顺序无关。

## Batch B：固定拓扑语义控制带

### 文件

- 新增 `src/geometry/semantic_eyelid_rig.py`
- 新增 `tests/test_semantic_eyelid_rig.py`

### 实现

1. 使用 FLAME 68 点三角形映射建立左右眼上睑、下睑、内外眼角种子。
2. 通过 mesh 邻接和测地环生成核心带与眼窝过渡带。
3. 建立 8 个控制柄的紧支撑平滑权重。
4. 生成低维形变基：闭合、高度、倾斜和法向隆起。
5. 提供眼部之外逐点不变检查。

### 验证

- 权重有限、非负、连续并在支持边界归零。
- 左右眼控制区不串联。
- 过渡带外所有顶点位移严格为零。
- 相同固定拓扑对不同人物复用相同语义索引。

## Batch C：三视角受控拟合

### 文件

- 新增 `src/geometry/eyelid_fit.py`
- 新增 `tests/test_eyelid_fit.py`

### 实现

1. 冻结 baseline mesh、相机和非眼部顶点。
2. 优化 8 个控制柄的低维三维偏移，不开放逐顶点变量。
3. 损失包含三视角有序眼睑点、眼角、眼缝、左右软一致和位移正则。
4. 闭眼状态增加上眼睑连续隆起先验；睁眼状态保留眼睛开口。
5. 使用现有 `local_mesh_quality` 回退到最大安全形变幅度。
6. 非眼部变化、退化面、法线翻转或观测恶化时回退 baseline。

### 验证

- 合成三视角目标能够恢复预设低维参数。
- 危险控制偏移会缩小或拒绝。
- 非眼部 landmark 和顶点保持不变。

## Batch D：冻结 Baseline 的实验入口

### 文件

- 新增 `run_semantic_eyelid_experiment.py`

### 实现

1. 加载 `captures_20260612_135253_controlled_identity_v1` 的冻结 OBJ、相机和哈希。
2. 加载并预处理三张原图，汇总眼睛状态。
3. 加载 FLAME landmark 映射并构建眼睑 Rig。
4. 运行受控拟合和质量门禁。
5. 输出 baseline/candidate OBJ、纯几何 GLB、状态和质量 JSON。
6. 不调用 texture pipeline，不覆盖源实验目录。

## Batch E：纯几何报告与验收

### 输出目录

`output/experiments/captures_20260612_135253_semantic_eyelid_v1`

### 输出

- `meshes/baseline_geometry.glb`
- `meshes/eyelid_candidate.glb`
- `debug/eye_state.json`
- `debug/eyelid_controls.json`
- `debug/eyelid_quality.json`
- `semantic_eyelid_compare.html`
- 正面、左右侧同机位纯几何截图

### 验收

- 当前样本可靠识别为双眼闭合。
- 三视角眼睑投影整体改善。
- 眼部之外顶点逐点不变。
- 无新增退化面、翻面、尖坑或边界异常。
- Candidate 眼睑弧度在纯几何 Viewer 中比 baseline 更自然。
- 如视觉未改善，保留 baseline 并将 Candidate 标为实验失败，不追逐单项分数。

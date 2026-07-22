# 几何优先严格投影纹理实施计划

> 对应设计：`docs/superpowers/specs/2026-07-22-geometry-first-projective-texture-design.md`

## 实施原则

- 先建立真实、可审计的投影链，再替换纹理烘焙。
- 每个批次独立产出报告和测试，不覆盖用户认可的 baseline。
- strict 路径不允许 TPS/RBF/nonrigid warp，也不允许纹理阶段修改 mesh 或 camera。
- 现有 dirty worktree 中的旧实验改动原样保留；每次只提交本批次明确涉及的文件。

## A1：Truth Diagnostic

### 任务 1：建立严格投影基础类型

新增 `src/appearance/projective_sampling.py`：

- `ProjectionCoordinates`
  - 保存 camera-space point、depth、原始像素坐标和 front-facing 状态。
- `project_points_strict(points, K, R, t)`
  - 只做一次标准透视投影。
  - 拒绝 NaN/Inf 和非法矩阵形状。
- `compare_sampling_coordinates(projected, sampled)`
  - 统计 mean/P50/P90/P95/max 位移。
  - 输出 exact-coordinate ratio 和超过 1/5/12 px 的比例。
- `assert_strict_sampling_coordinates(projected, sampled)`
  - strict 模式下任何非零位移均失败。

新增 `tests/test_projective_sampling.py`：

- 已知相机的投影结果准确。
- 输入 shape、NaN/Inf、零深度处理明确。
- 相同坐标通过，warp 后坐标被拒绝。
- 指标统计对空样本和部分有效样本行为稳定。

### 任务 2：生成未变形几何证据

新增 `src/reports/projective_texture_report.py`：

- 加载 `face_mesh.obj`、`cameras.json` 和三视角原图。
- 将 mesh 顶点和抽样面投影到原图，不使用 registration warp。
- 生成每视角：
  - 原图 + 未变形 wireframe/轮廓。
  - 深度可见点叠加。
  - mesh 语义锚点叠加。
  - 若存在 legacy warp，则生成 `proj -> sample_proj` 位移箭头，仅作对照。
- 生成 `truth_metrics.json`：
  - camera/image 尺寸。
  - 投影有效比例。
  - 正深度比例。
  - legacy sampling displacement。
  - geometry/camera SHA-256。
  - 明确记录 `final_sampling_mode=unwarped`。
- 生成静态 `index.html`，不依赖本地服务器。

新增 `tests/test_projective_texture_report.py`：

- 报告缺少 warp 时仍可生成。
- 报告明确区分真实投影与 legacy 采样坐标。
- hash、视角、图片路径和关键指标被写入 JSON。
- 报告生成不修改输入文件。

### 任务 3：增加可重复运行入口

新增 `run_projective_texture_truth_diagnostic.py`：

- 参数：
  - `--capture-dir`
  - `--mesh-dir`
  - `--output-dir`
  - `--with-legacy-warp`，默认关闭。
- 自动按被摄者语义映射：
  - camera1 -> left（被摄者左脸数据）
  - camera2 -> front
  - camera3 -> right（被摄者右脸数据）
- 原图先使用与 stable pipeline 相同的 undistort、intrinsics 和 preprocess 链。
- 开启 `--with-legacy-warp` 时调用现有 registration，但其结果只能进入比较图和指标。
- 运行前后验证 `face_mesh.obj` 与 `cameras.json` hash 不变。

### 任务 4：双数据集验证

运行：

1. `captures_20260612_135253`
   - mesh：`captures_20260612_135253_controlled_identity_v1/meshes`
2. `captures_20260612_140210`
   - mesh：`captures_20260612_140210_full_rerun_20260719/meshes`

输出到各自实验目录下的 `debug/projective_texture_truth_a1`，不覆盖 GLB 或纹理。

验收：

- 三视角 HTML 可直接通过 `file:///` 打开。
- 每个视角均显示未变形几何投影。
- 开启 legacy 对照时，位移指标非零且不会被标记为 strict 采样。
- 输入 mesh/camera hash 前后一致。
- 诊断报告能直接指出鼻部错位发生在原始投影还是 warp 之后。

## A2：Strict Projective Sampler

### 任务 1：统一采样对象

- 扩展 `ProjectionCoordinates` 为批量 `ProjectionSample`。
- 同一对象携带 RGB、mask、depth、semantic 的唯一像素坐标。
- 把 `_render_camera_depth` 和 bilinear/nearest sampling 收敛到新模块。

### 任务 2：新增 strict bake 分支

- `module3_texture.bake_texture()` 增加明确的 `sampling_mode`。
- strict 模式不接收 `sampling_warps`；传入即报错。
- RGB、face mask 和 z-buffer 均在 `pixel_xy` 读取。
- 保存逐 texel source-view 和 valid/confidence 中间产物。

### 任务 3：实验接入

- `stable_texture.py` 支持 `STABLE_TEXTURE_MODE=strict_projective`。
- 初期只在独立实验配置开启，不替换默认 baseline。
- 生成 legacy/strict 同几何对比 GLB。

验收：

- `sample_proj` 在 strict 调用链中不存在。
- 任意坐标位移测试必然失败。
- 纹理前后 mesh/camera hash 一致。

## A3：Semantic Ownership And Gates

### 任务 1：统一语义接口

- 新增 `src/appearance/semantic_visibility.py`。
- 定义 mesh region 与 image parser label 的兼容矩阵。
- 对鼻头、鼻翼、鼻孔、眼睑、眼球、嘴唇、口腔建立硬边界。

### 任务 2：视角归属

- 正面负责中央五官。
- 左右侧面仅覆盖同侧外周和侧鼻翼。
- 冲突时降 confidence，不跨语义平均。

### 任务 3：质量产物

- 输出 source-view ownership map。
- 输出 semantic reject map 和各区域 reject 比例。
- 鼻孔暗像素落到鼻头皮肤时必须被拒绝。

## A4：Intrinsic Appearance Correction

### 任务 1：低频照明估计

- 新增 `src/appearance/intrinsic_appearance.py`。
- 在高置信 skin 区域拟合每视角低频 luminance field。
- 排除五官、头发、背景、高光和深阴影。

### 任务 2：跨视角颜色一致化

- 只校正低频亮度和色偏。
- 保留高频毛孔、痣和局部肤色。
- 输出校正前后图及 estimated shading map。

验收：

- 鼻翼/鼻孔烘焙阴影减弱。
- PBR 查看器中不再出现明显双重阴影。
- 高频纹理锐度不出现显著下降。

## A5：Integration And Acceptance

### 任务 1：pipeline 集成

- `stable_three_view.py` 在纹理前后执行 geometry/camera lock。
- 默认模式切换前保留显式 legacy 配置。
- API、GLB 路径和 viewer 不变。

### 任务 2：质量报告整合

- stable report 加入 truth overlay、ownership、confidence、semantic reject 和 shading。
- 未变形残差作为主指标，warp 后残差仅作 legacy 对照。

### 任务 3：最终回归

- 两个数据集完成 legacy/strict A/B。
- mesh 面数、顶点和 hash 不变。
- 鼻头黑环、鼻孔纹理爬升和跨视角色差较 baseline 减少。
- 所有未解决错位都能被归到 geometry/camera/semantic/appearance 之一。

## 测试命令

每批次先运行对应测试，再运行相关回归：

```powershell
python -m pytest tests/test_projective_sampling.py tests/test_projective_texture_report.py -q
python -m pytest tests/test_texture_blending.py tests/test_texture_registration.py tests/test_baseline_texture_lock.py -q
```

完整 pipeline 接入后再运行 stable pipeline smoke test；A1 只生成只读诊断，不需要完整重建。

## 提交策略

- 计划文档独立提交。
- A1 基础投影与测试独立提交。
- A1 双数据集报告为 gitignored 运行产物，不提交大型图片。
- A2-A5 每个批次通过测试和人工截图检查后分别提交。

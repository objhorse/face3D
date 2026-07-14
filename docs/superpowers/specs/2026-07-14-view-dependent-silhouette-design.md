# 三视角视角相关剪影优化设计

日期：2026-07-14  
基线提交：`bad7d53`（`7.14`）  
目标分支：`codex/restore-7-12-full`

## 1. 问题与目标

当前优化把 68 点检测器中的 `0-16` 脸周点，与 FLAME 上固定的 17 个三维表面位置绑定，并在正面、左侧和右侧视图中重复使用同一组位置。

这在几何上不成立。图像中的最外轮廓是相机视线与三维表面的切线位置；视角改变时，形成轮廓的表面位置会沿脸颊和下颌滑动。因此，左右侧视图和正视图中的脸周点不能被解释为同一块皮肤。

本实验只修复这一观测模型错误：

- `0-16` 点不再参与任何固定三维点对应损失。
- 每个视角使用当前模型实时渲染出的 soft silhouette，与该视角的人脸 mask 比较。
- 相机姿态在剪影精修阶段冻结，使改进只能来自共享 FLAME shape，而不能由相机移动代偿。
- 眼、鼻、嘴等内部稳定关键点继续作为形状保护锚点。
- 不修改纹理、UV、自由残差、Depth Anything 或 mesh 拓扑。

## 2. 方案选择

采用方案 C：低分辨率可微整 mask 渲染。

使用 NVIDIA `nvdiffrast` 在 GPU 上将每个视角的 FLAME mesh 渲染为 soft alpha mask。当前环境具备 PyTorch 2.4.1、CUDA 11.8、RTX 4060 和 Visual Studio C++ 工具链；项目目前没有可微渲染依赖，因此 `nvdiffrast` 作为本实验的显式可选依赖安装。

不选择自写 rasterizer，因为 159,616 个三角面会使纯 PyTorch soft rasterization 速度和显存不可控。也不把现有逐行 soft-min 轮廓作为方案 C，因为它仍只描述局部边界，并非完整的模型投影区域。

## 3. 架构与数据流

新增 `src/geometry/differentiable_silhouette.py`，负责：

1. 把相机内参和当前 `R/t` 转换为光栅器需要的投影坐标。
2. 在固定的低分辨率空间渲染 full mesh soft alpha mask。
3. 将预处理的人脸 mask 缩放到相同分辨率并生成可靠性权重。
4. 计算可微 mask loss 和仅用于报告的像素指标。

在现有 landmark 联合拟合完成后增加一个独立的 silhouette shape refinement：

1. 输入 `shape_opt`、每视角共享 expression、固定 `K/R/t` 和三张 shape mask。
2. 只优化 FLAME shape 参数。
3. 每次迭代为三个视角重新生成完整预测 mask，因此每个视角会自然选择自己的可见轮廓表面。
4. loss 由 weighted soft Dice、边界距离和 shape delta 正则组成。
5. 内部稳定关键点只用于限制精修不能破坏眼、鼻、嘴的位置。
6. 输出候选 shape，由新的 mask 指标和既有 mesh quality gate 决定是否接受。

`0-16` 固定 landmark 误差仍写入报告，名称标记为 `legacy_fixed_contour_diagnostic`，但不参与 loss，也不参与候选接受或拒绝。

## 4. Mask 可靠性

三视角照片的头发、耳朵和脖子边界并不等于 FLAME 面部表面，不能平均施加同等约束。目标 mask 由现有 `shape_mask` 产生，但使用可靠性图：

- 左右脸颊、下颌和下巴轮廓：高权重。
- 额头两侧：中等权重。
- 发际线顶部、耳朵外缘和脖子：零权重。
- mask 边界置信度过低或断裂的位置：零权重，并记录到报告。

该权重规则由图像语义和相对脸框位置自动生成，不为当前被摄者手工画分区，因此可以迁移到其他三视角数据集。

## 5. 优化与门禁

默认在 256x256 或按人脸框等比例的近似分辨率优化，最终指标换算回原图像素。

候选必须同时满足：

- 三视角加权 silhouette loss 总体下降。
- 至少两个视角的可靠边界距离下降，第三个视角不得显著恶化。
- 内部稳定 landmark mean 不得恶化超过 1 px。
- shape 参数改变量不超过配置上限。
- 顶点、面片、退化面和非流形质量门禁全部通过。

门禁不再读取固定 `0-16` contour mean。若候选未通过，则保留 `7.14` 几何并在报告中说明具体失败项。

## 6. 产物与诊断

实验输出单独写入，不覆盖基线：

- `output/experiments/view_dependent_silhouette_c/baseline/`
- `output/experiments/view_dependent_silhouette_c/candidate/`
- `output/experiments/view_dependent_silhouette_c/summary.json`
- `output/experiments/view_dependent_silhouette_c/index.html`

每个视角保存：原图、目标 mask、预测 mask、边界叠加图和误差热图。Viewer 同时提供 baseline 与 candidate 的贴图和 clay 模式，避免纹理视觉误差被误认为几何变化。

## 7. 失败处理

- `nvdiffrast` 安装或加载失败：实验明确失败，不改变 pipeline 默认结果。
- CUDA 光栅化失败或显存不足：依次降低到 192 和 128 分辨率；仍失败则停止。
- mask 可靠区域不足：该视角退出 loss，但至少需要两个有效视角才能开始优化。
- loss 出现 NaN/Inf：丢弃候选并保留 `7.14`。
- 候选门禁失败：保留 baseline，报告失败原因，不修改 `face.glb`。

## 8. 测试与验收

单元测试：

- 投影矩阵与现有 `project_vertices` 在多个三维测试点上保持像素一致。
- 同一椭球在不同 yaw 下产生不同的轮廓表面位置，但预测 mask 均正确。
- `0-16` 点权重为零，改变这些检测点不会改变优化 loss。
- mask loss 对 shape 参数存在有限且非零梯度。
- 低置信区域不产生梯度。

数据集验收使用 `captures_20260612_140210`：

- 完整 pipeline 和方案 C 实验均能运行。
- 三视角可靠剪影误差总体优于 `7.14`。
- 不再发生“dense silhouette 明显改善，却因固定 contour landmark 变差而拒绝”的冲突。
- 内部五官投影和 mesh 质量不退化。
- 提供 baseline/candidate viewer，由用户最终判断脸颊、下颌和整体脸型是否更像。

本实验只验证视角相关剪影约束，不以一次实验解决鼻翼宽度、纹理错位或三相机外参误差。

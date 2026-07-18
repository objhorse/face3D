# MICA 身份锚定联合优化设计

日期：2026-07-15
分支：`codex/restore-7-12-full`
视觉基线：`captures_20260612_140210_silhouette_c_v2`
失败反例：`captures_20260612_140210_silhouette_metric_v5`

## 1. 问题与证据

v5 通过了轮廓、内部关键点和网格质量门禁，但人工观察确认其身份已经明显改变。进一步对比证明：

- C v2 与 v5 使用相同照片、mask、landmark 和逐项完全相同的 MICA shape。
- 两版最终 FLAME shape 的参数距离为 `10.10`，方向余弦相似度仅为 `0.73`。
- C v2 最终 shape 相对 MICA 的参数漂移为 `5.60`，v5 为 `11.04`。
- C v2 相对 MICA 中性网格的平均/P95/最大位移分别为脸宽的 `1.12% / 1.99% / 2.77%`。
- v5 对应数值为 `3.76% / 5.66% / 8.02%`。
- 两个独立进程的初始化器只有约 `0.0002` 的 DECA expression 数值差和正脸约 `0.57` 度的姿态差，但联合优化将该扰动放大成完全不同的 identity shape。

根因位于 `JointFLAMEOptimizer`：shape、expression 和 camera pose 同时优化，而 shape 正则为 `mean(shape ** 2)`，它把参数拉向 FLAME 平均脸，并未保护 MICA 给出的本人身份。三视角二维观测本身不足以唯一分解相机、表情和身份，小幅初始化变化因此可能被放大为不同的局部解。

## 2. 目标与边界

本阶段保留联合优化结构，不改为完全解耦的多阶段拟合。目标是：

1. MICA shape 明确成为身份锚点。
2. 联合优化可以在 MICA 附近修正脸型，但不能通过改变身份来偿还二维误差。
3. 轮廓优化前后都执行身份门禁，任何“投影更准但换了人”的候选都不能输出。
4. 同一数据重复运行时，最终 identity shape 应稳定落在同一可信邻域。

本阶段不修改纹理烘焙、不恢复自由顶点残差，也不把 C v2 参数硬编码为当前人物模板。C v2 只作为视觉和数值回归样本。

## 3. 联合优化中的 MICA 锚定

`JointFLAMEOptimizer.optimize` 在接收 `init_shape` 后保存不可训练的 `identity_anchor`。shape 正则改为两项：

1. 主要身份正则：`mean((shape_param - identity_anchor) ** 2)`。
2. 极弱平均脸先验：`mean(shape_param ** 2)`，只用于阻止异常系数，不再主导优化方向。

新增独立配置：

- `JOINT_IDENTITY_ANCHOR_WEIGHTS`：三档递增锚定强度。
- `JOINT_MEAN_SHAPE_PRIOR_WEIGHT`：极弱平均脸先验。
- `JOINT_MAX_IDENTITY_ATTEMPTS = 3`。

初始 smoke test 使用 `5e-4 / 2e-3 / 8e-3`，但三档的中性网格平均漂移分别为 `2.75% / 1.91% / 1.68%`，均未通过 `1.5%` 硬门槛。第一次校准到 `8e-3 / 1.6e-2 / 3.2e-2` 后，第二个三视角数据集的最强档仍以 `1.62% / 2.52% / 4.08%` 略超平均/P95/最大位移门槛。因此默认三档整体校准为 `1.6e-2 / 3.2e-2 / 6.4e-2`，不放宽身份门禁。极弱平均脸先验保持 `5e-6`。这些值是 pipeline 级默认配置，不能为单个人脸动态改写；实验只能通过显式配置覆盖，并必须写入报告。

每次尝试都必须从同一份 MICA shape、同一组初始 expression 和同一组初始 pose 重新开始，不能从失败候选继续优化。这样每次只改变一个变量：身份锚定强度。

## 4. 身份漂移指标

新增独立的 identity drift evaluator，同时计算参数空间和几何空间指标。

### 4.1 参数空间

- `coefficient_delta_l2 = ||shape_candidate - shape_mica||2`
- `coefficient_delta_max = max(abs(shape_candidate - shape_mica))`
- shape 必须全部为有限值。

参数 L2 默认硬门槛为 `7.0`。该阈值允许 C v2 的 `5.60`，并拒绝 v5 的 `11.04`。最大单参数变化同时记录，用于发现少数系数异常，但第一版不单独硬编码人物相关上限。

### 4.2 几何空间

分别用 MICA shape 和候选 shape，在零 expression 下生成固定拓扑中性 FLAME mesh。逐顶点欧氏位移除以 MICA mesh 的 X 方向脸宽，得到跨尺度可比较指标：

- 平均位移不超过 `1.5% face width`。
- P95 位移不超过 `2.5% face width`。
- 最大位移不超过 `4.0% face width`。

参数和几何门禁必须同时通过。固定拓扑 mesh quality 门禁继续检查 NaN/Inf、退化面、非流形和边界变化。

## 5. 三档联合优化与候选选择

每档锚定强度执行一次完整联合优化，并产生独立记录：

1. 优化后的 shape、expression 和 per-view camera pose。
2. 内部 landmark 重投影指标。
3. 相对 MICA 的 identity drift 指标。
4. 中性 mesh quality。

候选选择顺序：

1. 首先过滤 identity drift 或 mesh quality 失败的候选。
2. 在通过身份门禁的候选中，选择内部 landmark 和可信轮廓综合表现最好的候选。
3. 若两个候选的可信轮廓差异不超过当前 `256` 分辨率轮廓测量的一个渲染像素，并且内部 landmark 差异也不超过一个原图像素，则视为观测质量相同，选择身份漂移更小的候选。
4. 若三档都失败，禁止使用联合 shape；回退到 MICA shape，并保留最强锚定尝试中通过质量门禁的 expression/camera 仅用于诊断，不直接宣称为成功最终参数。

报告必须记录每档尝试及被拒绝原因，不能只保存最终候选。

## 6. 后续 shape-only 轮廓优化

shape-only silhouette refinement 仍可运行，但它必须同时接收：

- 当前联合优化 shape，作为局部优化起点。
- 原始 MICA shape，作为全流程身份锚点。

每个 checkpoint 除现有可信轮廓、内部五官和网格门禁外，再执行 identity drift 门禁。平均可信轮廓改善还必须至少达到一个当前渲染像素；低于渲染分辨率的变化不允许修改 shape。候选选择只能发生在全部门禁通过的 checkpoint 中。若多个合格 checkpoint 的可信轮廓差异不超过一个当前渲染像素，则选择最早的合格 checkpoint；测量分辨率无法支持的后续形变不作为额外改进。该规则避免微小数值抖动在长优化轨迹中决定最终脸型。

如果轮廓优化没有合格 checkpoint，则保留通过身份门禁的联合优化 shape。轮廓分数不得覆盖身份门禁。

## 7. 数据流与产物

数据流为：

`MICA shape -> 三档联合优化 -> identity gate -> 联合候选选择 -> shape-only silhouette checkpoints -> identity gate -> 最终 shape`

新增调试产物：

- `debug/joint_identity_anchor/summary.json`
- `debug/joint_identity_anchor/attempt_1.json`
- `debug/joint_identity_anchor/attempt_2.json`
- `debug/joint_identity_anchor/attempt_3.json`
- `debug/shape_only_fine_tune/summary.json` 中的 `identity_drift` 字段

稳定质量报告新增独立的 `Identity Preservation` 章节，展示 MICA、C v2 回归参考和当前候选的参数/网格漂移。纹理指标与身份指标继续完全分离。

## 8. 失败处理

- MICA shape 缺失或包含 NaN/Inf：停止稳定 pipeline，不允许退回零 shape 后继续宣称身份重建成功。
- 某档联合优化出现 NaN/Inf：丢弃该档并尝试更强锚定。
- 身份门禁失败：丢弃候选，不进入轮廓优化。
- mesh quality 失败：丢弃候选。
- 三档均失败：输出明确失败报告并保留 MICA baseline 调试模型，不覆盖已存在的成功 `face.glb`。
- shape-only 候选身份漂移失败：保留联合优化结果。

## 9. 测试与验收

单元测试：

- 身份正则的最小值位于 MICA anchor，而不是零 shape。
- C v2 的已保存 shape 通过默认 identity drift gate。
- v5 的已保存 shape 被 identity drift gate 拒绝。
- 参数门禁通过但几何位移超限时仍拒绝。
- 三档候选只从 identity 和 mesh gate 均通过的集合中选择。
- shape-only checkpoint 不能绕过 identity gate。
- identity 指标不读取纹理或旧 `0-16` landmark。

数据集验收使用 `captures_20260612_140210`：

- 重新完整运行至少两次。
- 两次最终 coefficient delta 应显著小于历史 C v2/v5 的 `10.10` 差异，并默认要求两次 shape 距离小于 `1.0`。
- 两次均通过 identity、可信轮廓、内部五官和 mesh quality 门禁。
- 最终 mesh 保持 `159,616` 面、零新增退化面和零新增非流形。
- 新结果与 C v2 并排 viewer 供人工复核；人工确认“仍是同一个人”是提交前必要条件。

## 10. 回滚与提交策略

当前 v5 结果和相关未提交实现不得作为成功版本提交。实施在现有 `7.14`/C v2 安全基线上继续，输出使用新的实验目录，不覆盖 C v2。只有用户完成视觉复核后，才提交身份锚定实现。

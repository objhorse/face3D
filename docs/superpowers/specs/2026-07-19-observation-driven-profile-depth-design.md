# 三视角观测驱动的口鼻侧貌深度修复设计

状态：已批准

日期：2026-07-19

## 1. 背景

当前稳定 pipeline 的拓扑已经不再破碎，但多个数据集持续出现口周整体前突、鼻尖与嘴唇几乎齐平的问题。二维关键点、嘴缝闭合和 mesh quality gate 仍可通过，因此现有指标无法识别这种三维侧貌错误。

在 `captures_20260612_140210_full_rerun_20260719` 上隔离各阶段后，鼻尖最前点相对嘴唇最前点的领先距离为：

| 阶段 | 鼻尖领先嘴唇 |
| --- | ---: |
| FLAME 平均脸 | 13.72 mm |
| MICA 身份初始化 | 8.00 mm |
| 身份形状优化后的 neutral mesh | 7.77 mm |
| 加入共享 expression 后 | 3.97 mm |

历史输出也普遍从 neutral mesh 的约 8-10 mm 降至最终 mesh 的约 3-7 mm。这说明问题由两部分叠加：MICA 对口鼻前后比例的身份估计偏差，以及 expression 在未受深度约束时继续把嘴部向前推。

另外，当前主 runner 默认没有启用固定 rig 外参，而是允许每个视角独立优化相机姿态。独立相机姿态可以吸收错误的面部深度，使二维重投影分数看起来正常。

## 2. 目标

在不改变三视角采集条件的前提下：

- 从标定相机和三视角图像中恢复被摄者自己的鼻尖、鼻下点、上下唇和下巴前后关系。
- 在 FLAME 参数空间内纠正 MICA 身份形状，不进行自由顶点推拉。
- 将身份深度与表情分开，防止闭嘴 expression 将口周整体向前推。
- 新增真正反映三维侧貌的质量指标，使错误模型不能仅凭二维误差和拓扑完整性通过。
- 保持固定拓扑、UV、纹理和现有 API/GLB 接口兼容。

## 3. 非目标

- 不使用统一的“正常鼻唇距离”把所有人修成同一种脸。
- 不恢复自由 residual deformation 或 Depth-Anything 顶点置换。
- 不在本阶段解决纹理接缝、头发或不可见区域补全。
- 不声明医疗级测量精度；本阶段先建立可审计的个体侧貌约束。

## 4. 方案比较

### 方案 A：固定 rig + 个体三维侧貌观测（采用）

使用标定内外参，在去畸变图像中检测固定解剖语义点，进行鲁棒三角化。将观测到的个体三维点作为 shape 和 expression 的约束。

优点：直接利用现有三相机条件；结果随被摄者变化；能够解决二维投影无法约束深度的问题。

代价：需要严格处理相机语义、三角化置信度和错误检测点。

### 方案 B：增加人类侧貌统计先验

对鼻尖、嘴唇和下巴加入经验比例范围或 E-line 类约束。

优点：实现快，即使三角化失败也能抑制极端前突。

缺点：会把不同个体拉向平均脸，不能作为主要身份恢复依据。仅适合作为宽松异常门禁，不作为优化目标。

### 方案 C：更换 MICA/FLAME 底层身份模型

替换为更强的多视角身份模型或更高表达能力的头模。

优点：长期上限更高。

缺点：改动面大，不能解释和修复当前 expression 深度泄漏，也会破坏已稳定的接口和拓扑基础。本阶段不采用。

## 5. 总体架构

新增一条只作用于参数化模型的 profile-depth 路径：

1. 固定相机 rig，优化单一全局头部位姿。
2. 从三视角提取固定语义侧貌点及其像素置信度。
3. 在 front-camera 参考坐标系中鲁棒三角化。
4. 先用三维 profile loss 修正 MICA shape。
5. 再优化 expression，同时保持口周前后深度与观测一致。
6. 通过三维侧貌门禁后才允许导出和烘焙纹理。

建议新增模块：

- `src/geometry/profile_observations.py`：语义点提取、跨视角匹配和置信度。
- `src/geometry/profile_triangulation.py`：固定 rig 三角化、重投影验证和坐标统一。
- `src/geometry/profile_depth_fit.py`：shape/profile loss 与 expression depth-preservation loss。
- `src/geometry/profile_depth_quality.py`：三维侧貌指标和 acceptance gate。
- `tests/test_profile_observations.py`
- `tests/test_profile_triangulation.py`
- `tests/test_profile_depth_fit.py`
- `tests/test_profile_depth_quality.py`

## 6. 固定 Rig 与坐标系

### 6.1 相机约束

- 使用已验证的 `camera_calibration.json` 内参、畸变和相对外参。
- front camera 作为 rig reference。
- 每个 session 只优化一个全局头部刚体位姿 `R_head, t_head`。
- left/front/right 相机之间的相对位姿固定，不再各自独立吸收形状误差。
- v1 不允许 per-camera 自由 pose residual；如重投影无法通过，报告标定或观测失败，而不是放松相机。

### 6.2 坐标与单位

- 所有二维点先在原始分辨率上去畸变，再映射到统一工作画布。
- 三角化输出统一到 front-camera 参考坐标系。
- 标定平移单位必须在运行前审计并写入 metadata；若无法确认毫米尺度，则同时输出绝对单位和按眼间距归一化的相对深度。

## 7. 三维侧貌观测

### 7.1 v1 语义点

优先使用在三个视角中语义稳定、对侧貌有直接意义的中线点：

- nose tip
- subnasale
- upper-lip vermilion midpoint
- lower-lip vermilion midpoint
- mouth center
- soft-tissue chin / pogonion

鼻翼宽度和口角暂时继续作为二维宽度约束，不混入第一版侧貌深度损失。

### 7.2 观测来源

- MediaPipe dense landmarks 提供初始位置与闭眼/闭嘴状态。
- face_alignment 68 点提供与现有 FLAME landmark mapping 兼容的语义锚点。
- 在初始点附近使用局部边界、颜色梯度和 epipolar line 做小范围 refinement。
- LoFTR/已有 cross-view surface observations 只作为局部匹配证据，不直接把任意纹理点当作固定解剖点。

### 7.3 鲁棒三角化

每个语义点执行：

- 三对相机两两三角化并计算三视角重投影误差。
- 检查正深度、视线夹角、跨 pair 三维一致性和语义 ROI。
- 使用 Huber/加权中值融合有效候选。
- 输出 observation confidence；低置信点不进入强约束。

三角化失败必须保留 MICA baseline，而不是生成猜测深度。

## 8. Shape 优化

shape 阶段仍只优化 FLAME shape coefficients，不允许自由顶点形变。

损失由以下部分组成：

- 原有稳定内部 landmark 重投影损失。
- 固定 rig 下的多视角投影损失。
- 三维 profile point loss：FLAME 语义点与三角化目标的鲁棒三维距离。
- 相对侧貌 loss：nose-tip-to-lip、subnasale-to-lip、lip-to-chin 的前后深度差。
- MICA identity anchor 与 mean-shape prior。

三维绝对点和相对深度同时使用：绝对点约束整体形状，相对深度降低全局尺度和轻微位姿误差的影响。

shape 候选只有在三维 profile 误差改善、二维内部点未显著变差、identity drift 和 mesh quality 均通过时才能接受。

## 9. Expression 优化

当前 expression fidelity 只检查眼睑和嘴缝的二维垂直间距，无法发现嘴唇整体向前移动。新设计增加：

- `mouth_depth_displacement`：expression mesh 相对 neutral mesh 的口周法向/前后位移。
- `profile_target_error`：加入 expression 后，鼻尖、上下唇和下巴仍需匹配三维观测。
- `closed_mouth_depth_preservation`：闭嘴样本不得仅为缩小嘴缝而把整个口周向前推。

闭眼样本优先使用语义眼睑 rig 完成眼睑闭合。通用 FLAME expression 不再为了闭眼而自由改变鼻子、嘴部和下巴。

v1 推荐策略：

- shape 使用 neutral expression 求解身份。
- 对 expression basis 预计算 mouth-depth influence。
- 对显著改变口周前后深度的 expression 模式施加强正则。
- 允许眼睑和必要的嘴缝闭合变化，但最终 profile depth 必须保持在观测容差内。

## 10. 质量指标与门禁

新增 `profile_depth_quality.json`，至少包含：

- 每个语义点的三角化状态、置信度和三视角重投影误差。
- `nose_tip_minus_upper_lip_depth`
- `nose_tip_minus_lower_lip_depth`
- `subnasale_minus_lip_depth`
- `lip_minus_chin_depth`
- neutral 与 expression 的 mouth-depth displacement。
- shape 前后、expression 前后的观测误差变化。

第一版 acceptance gate：

- 至少四个中线语义点通过三角化，其中必须包含 nose tip、一个 lip point 和 chin。
- 有效点三视角重投影 P90 不高于 5 px；目标值在完成标定审计后收紧。
- 最终 nose/lip/chin 相对深度误差不高于 2 mm 或归一化脸宽的等价容差。
- 闭嘴时口周因 expression 产生的平均前向位移不高于 0.5 mm，除非三维观测明确支持该变化。
- 原有 topology gate、identity drift gate 和内部 landmark preservation gate 继续生效。

不得仅使用统一人群阈值强行修改个体；统计范围只用于发现明显异常或观测故障。

## 11. 调试产物

每次 pipeline 保存：

- `debug/profile_depth/observations_<view>.png`
- `debug/profile_depth/epipolar_matches.png`
- `debug/profile_depth/triangulated_points.ply`
- `debug/profile_depth/reprojection_<view>.png`
- `debug/profile_depth/neutral_profile.png`
- `debug/profile_depth/final_profile.png`
- `debug/profile_depth/profile_depth_quality.json`
- `debug/profile_depth/index.html`

报告必须把输入侧脸、neutral 几何侧貌、final 几何侧貌并列展示，默认不显示纹理，避免贴图影响形状判断。

## 12. 失败处理

- rig 数据缺失或相机语义不一致：停止 profile-depth 优化，明确报错，不回退到独立相机优化并伪装成功。
- 单个语义点不可靠：降低权重或剔除该点，保留其他有效点。
- 有效点不足：保留 MICA neutral baseline，标记 `profile_depth_unresolved`。
- expression 造成深度恶化：回退到受限 expression 或 neutral-mouth 版本。
- shape 候选改善侧貌但破坏其他区域：拒绝候选并保留 baseline。

## 13. 测试计划

### 单元测试

- 已知相机和三维点可在噪声下稳定三角化。
- 错误 camera1/camera3 语义会被正深度或重投影门禁识别。
- 低视差、错误匹配和离群点不会生成高置信三维目标。
- profile relative-depth loss 对全局平移不敏感。
- closed-mouth expression 前推口周时质量门禁失败。
- shape/expression 候选拒绝后精确保留 baseline 顶点。

### 回归测试

- `captures_20260612_140210`：最终 nose-tip-to-lip 深度接近三角化观测，明显高于当前 3.97 mm 的错误结果。
- `captures_20260612_135253`：验证算法不是只针对单个人。
- 原有固定拓扑、无删面、纹理和 GLB 导出测试继续通过。
- 未启用 profile-depth 功能时，旧 baseline 可复现以便 A/B 对比。

## 14. 验收标准

- 三视角使用固定 rig 相对外参，只优化一个全局头部位姿。
- 输入照片中的 nose/lip/chin 中线点能够生成可审计三维观测。
- neutral 和 final mesh 都报告口鼻前后关系，表情不能再无审计地把嘴部向前推。
- 两个数据集的侧脸视觉均不再出现鼻尖与嘴唇近乎齐平的系统性前突。
- 最终模型继续保持 79,936 顶点、159,616 面附近的固定拓扑，无新增破洞、退化面或非流形边。
- 三维侧貌不合格时 pipeline 必须失败或回退，不得只凭二维误差与 topology gate 标记成功。

## 15. 实施顺序

1. 建立 profile-depth 诊断，不改变几何，验证固定 rig 三角化是否可信。
2. 增加 expression mouth-depth audit，隔离并阻断表情造成的口周前推。
3. 将三维 profile loss 接入 MICA-anchored shape 优化。
4. 增加 acceptance gate、报告和两个数据集的 A/B 回归。
5. 验证通过后，再把语义眼睑 rig 串回主 pipeline。

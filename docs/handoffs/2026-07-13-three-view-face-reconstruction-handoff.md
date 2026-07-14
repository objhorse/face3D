# 三视角人脸重建项目交接记录

更新时间：2026-07-13  
项目根目录：`D:\face3D`  
当前主要分支：`codex/three-view-stable-template`

## 1. 当前目标

在不改变三视角 RGB 采集条件的前提下，生成：

- 不破洞、不撕裂、固定拓扑的人脸模型；
- 鼻、嘴、下巴、脸宽等身份形状尽可能接近真人；
- 可作为未来医美交互编辑基底的参数化模型；
- 当前阶段不声明医疗级毫米精度。

稳定路线已经禁用自由顶点残差和 Depth Anything 几何置换。真实感优先依靠稳定几何、正确相机和贴图，而不是激进拉顶点。

## 2. 当前最大问题

最大问题不是某个鼻翼权重，而是：

> 三台相机、跨视角观测和人脸参数还没有形成可信的统一三维坐标系统。

具体表现：

1. 原稳定拟合允许三视角相机姿态独立优化。每台相机单独投影尚可，但同一个图像点的空间射线不能稳定相交。
2. 早期跨视角实验把左右三角点云分别旋转、缩放、平移到当前模型附近，掩盖了相机与尺度错误。
3. 原流程三个视角使用不同 expression，允许嘴、眼、鼻周围形状分别代偿相机或身份误差。
4. LoFTR 的有效观测主要分布在脸颊；鼻部仅少量点，嘴部几乎没有可靠三维点。
5. 左右视角夹角大、共同可见区域少，不能指望普通纹理匹配产生大量真正的三视角闭环点。

因此当前“像素误差下降”不必然代表三维脸型更像，甚至可能是相机、expression 或点云对齐在替错误几何背锅。

## 3. 已确认的重要数据

主要测试数据集：

`D:\face3D\captures_20260612_140210`

视角语义：

- `camera1`：被拍摄者左脸，对应程序 `left`
- `camera2`：正脸，对应程序 `front`
- `camera3`：被拍摄者右脸，对应程序 `right`

正常稳定重跑基线：

`D:\face3D\output\experiments\captures_20260612_140210_normal_rerun`

基线关键投影指标：

| 视角 | mean | contour | nose | eyes | mouth |
|---|---:|---:|---:|---:|---:|
| left | 17.753 px | 28.352 | 6.386 | 14.803 | 6.485 |
| front | 16.640 px | 26.750 | 12.178 | 7.883 | 9.790 |
| right | 19.380 px | 29.041 | 6.573 | 18.340 | 5.890 |

这些分数只能说明二维投影，不等于三维身份相似度。

## 4. 精确相机诊断结论

不要再用任意旋转角度截图判断模型。已经新增：

`D:\face3D\render_calibrated_model_views.py`

它使用输出中的 `K/R/t`，分别生成：

- 原图；
- 纯几何 clay；
- 贴图模型；
- 原图与模型叠加。

正常基线诊断目录：

`D:\face3D\output\experiments\captures_20260612_140210_normal_rerun\camera_view_diagnostics`

人工观察结论：

- 当前几何仍明显偏 FLAME 平均脸；
- 最大视觉差异是鼻底、人中、嘴唇、下巴的侧面深度关系；
- 其次是鼻尖、鼻翼、鼻梁曲线；
- 下颌与脸颊厚度仍偏模板化；
- 贴图会掩盖几何问题。

## 5. 跨视角实验及发现

实验入口：

`D:\face3D\run_fitted_camera_identity_experiment.py`

主要实验目录：

- 旧的双视角/独立归一化实验：  
  `D:\face3D\output\experiments\captures_20260612_140210_geometry_identity_v1`
- 新的统一坐标/三视角闭环实验：  
  `D:\face3D\output\experiments\captures_20260612_140210_geometry_identity_v2`

### 5.1 原独立相机

- 左右各有 51 条可信双向匹配；
- 原相机三角化后仅左侧 19 条通过，右侧 0 条；
- 右侧三角化重投影误差中位数约 6.54 px。

### 5.2 两两 Essential Matrix 相机候选

相对原拟合相机的差异：

- 左：旋转约 9.14 度，平移方向约 9.35 度；
- 右：旋转约 7.88 度，平移方向约 5.04 度。

两两修正后，双视角有效点可到 `50/50`，重投影 P90 约 1.05 px，但这不等于三台相机共同一致。

### 5.3 曾经的错误归一化

`normalize_observation_groups()` 曾分别把左右点云对齐到模型：

- 左侧 scale 约 0.918，平移约 18 mm；
- 右侧单独旋转约 12.1 度。

这会破坏统一尺度。目前新的实验路径已经禁用该步骤。不要恢复。

### 5.4 真正三视角闭环

- 左右 LoFTR 共同可见区域太少；
- 左右直接匹配只留下 8 条双向点；
- 没有 LoFTR 点能闭合 `front-left-right` 三视角轨迹；
- 使用稳定 MediaPipe 语义点得到 10 条三视角候选，但严格门禁最初 0 条通过。

新增三相机 bundle adjustment 后：

- 联合代价从约 40.70 降到 1.40；
- 左相机相对两两候选再变约 2.30 度、6.75 mm；
- 右相机再变约 8.15 度、10.92 mm；
- 严格门禁最终仍仅接受 1 个鼻部三视角点。

结论：当前人脸图像提供的三视角共同点仍不足以独立替代固定 rig 标定。

## 6. 共享 Expression 修改

已修改：

- `src/config.py`：新增 `STABLE_SHARED_EXPRESSION = True`
- `src/module2_geometry.py`：`JointFLAMEOptimizer` 支持三个同步视角共享一个 expression 参数向量。

共享 expression 完整重跑：

`D:\face3D\output\experiments\captures_20260612_140210_shared_expression_v1`

结果：

- 原三个视角 expression 最大差异约 3.01；
- 新版本三个视角 expression 差异精确为 0；
- 正面鼻部像素误差从 12.18 px 上升到 17.60 px；
- 其他整体误差也略升。

这不是单纯退步，而是证明旧版一部分低误差来自独立 expression 代偿。共享版本暂未设为默认视觉基线。

精确截图目录：

`D:\face3D\output\experiments\captures_20260612_140210_shared_expression_v1\camera_diagnostics`

## 7. Balanced 候选的正确定位

旧实验曾生成：

`D:\face3D\output\experiments\captures_20260612_140210_geometry_identity_v1\balanced_textured\model_viewer.html`

它在经过左右点云分别归一化后，三维附着误差改善约 3.83%，拓扑门禁通过。

但是由于它依赖分别归一化后的点云，不能作为可信新基线。只保留作历史实验对照，不应继续在其上放大形变。

## 8. 纹理现状

已完成的纹理改进包括：

- 语义纹理配准；
- 局部 LAB 色彩匹配；
- 多频段/接缝混合；
- 未观测 UV 透明或低置信处理；
- no-delete 几何模式。

主要相关文件：

- `src/appearance/texture_registration.py`
- `src/appearance/photometric_normalization.py`
- `src/appearance/stable_texture_registration.py`
- `src/appearance/stable_texture.py`
- `src/module3_texture.py`
- `run_stable_retexture.py`

纹理仍有局部错位与拼接问题，但当前优先级低于统一几何和相机。

## 9. 踩过的坑

### 9.1 只看二维 landmark 总分

二维分数更低不代表三维更像。FLAME landmark 与检测器 landmark 对鼻翼等位置定义可能不同，贴图错位也会误导肉眼判断。

### 9.2 用任意角度 viewer 截图比较

任意角度不能对应真实相机。必须使用精确 `K/R/t` 渲染原图视角。

### 9.3 把左右相机分别拟合得很好

每台相机单独投影好，不代表它们组成一致的三维相机架。独立 pose 可以掩盖错误脸型。

### 9.4 左右点云分别做 similarity alignment

这是目前最重要的错误之一。它会把标定和尺度误差先消掉，再把残差伪装成身份形状证据。

### 9.5 放宽自由形变或保护区

直接放宽会再次造成鼻尖、人中、鼻翼折叠或破碎。不能靠“大胆拉顶点”解决观测不准。

### 9.6 Depth Anything 直接置换顶点

单目深度没有可靠公制尺度，曾出现负尺度 fallback 和局部破碎。稳定 v1 中保持禁用。未来只能作为相对深度或法线弱约束。

### 9.7 用 LoFTR 强求左右大角度闭环

左右视角共同可见纹理太少。LoFTR 适合作为局部弱证据，不能单独承担三视角标定。

### 9.8 Viewer 复制后的相对路径

曾发生 Three.js vendor 路径少退一层，同时旧中文乱码破坏 HTML/JavaScript。生成 viewer 后必须验证：

- Three.js 路径；
- GLTFLoader 路径；
- `meshes/face.glb`；
- HTML/JS 语法。

### 9.9 camera1/camera3 语义混乱

必须始终使用被拍摄者左右：`camera1=left`，`camera3=right`。不要再按观察者左右解释。

### 9.10 工具超时不等于任务失败

若命令输出显示 GLB、报告已生成，应检查产物文件和日志尾部。多次长任务实际完成，但 shell 工具返回 timeout。

## 10. 推荐的后续技术顺序

严格按以下顺序，不要跳到局部拉脸：

1. 回到标定数据集，过滤坏帧，固定内参，三台相机和全部标定板姿态联合 bundle adjustment。
2. 用不同好帧子集重复标定，输出外参旋转/平移置信区间。
3. 固定 rig 外参。人脸照片只允许很小的整体 rig 修正，不能左右相机独立漂移。
4. 三视角共享 identity、expression 和一个整体人脸姿态。
5. 使用可微渲染联合比较三视角：轮廓、语义边界、mask、局部特征、遮挡关系。
6. 单目深度只作为尺度不变的相对深度/法线弱约束。
7. 先优化 FLAME identity，再优化鼻、唇、下巴语义控制器，最后才允许低频形变图。
8. 每个候选必须通过拓扑、面片翻转、边长变化、局部曲率和跨视角一致性硬门禁。

如果固定 rig 仍无法同时解释三张脸图，应先检查标定板角点、去畸变坐标、裁剪/缩放后的 K、相机映射和同步采集脚本，不能让人脸形状吸收这些误差。

## 11. 测试与环境

`gaussian` 环境当前没有 pytest。通过代理安装时长时间无输出，已终止。

`face3d` 环境已有 `pytest 9.0.3`，已运行：

```text
D:\Anaconda\envs\face3d\python.exe -m pytest tests\test_mesh_quality.py -q
4 passed
```

另外 9 个纹理相关 unittest 已通过。pytest 缓存目录可能因权限出现 warning，不影响测试结果。

pytest 对模型运行不是必需依赖，但对几何质量门禁和回归测试非常必要。

## 12. Git 与工作区注意事项

当前根工作区有未提交修改，包含纹理、共享 expression、跨视角诊断和 runner 改动。不要 reset、checkout 或清理用户文件。

近期关键未提交文件包括：

- `run_fitted_camera_identity_experiment.py`
- `run_stable_pipeline.py`
- `src/config.py`
- `src/module2_geometry.py`
- 多个 `src/appearance/*` 纹理模块
- 多个 `src/geometry/*` 跨视角与质量模块
- `render_calibrated_model_views.py`

`.worktrees/medical-resolution-calibration` 也是 dirty worktree，包含此前大量实验。不要覆盖或回滚。

开始新对话后，第一步应执行：

```powershell
git status --short
git branch --show-current
```

然后阅读本文和目标代码，再决定提交切点。

## 13. 一句话结论

当前项目已经从“容易破碎的自由残差路线”转向稳定模板路线；下一阶段的关键不是继续调鼻子，而是先得到一套能同时解释三张照片、具有统一尺度和置信区间的固定三相机 rig，再用多证据可微渲染恢复身份几何。

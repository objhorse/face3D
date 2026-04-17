# Face3D 模型升级迁移计划

## 目标

当前项目已经验证出两个事实：

- 现有 `landmark + 规则式多视图贴图` 链路可以跑通，但真实性上限有限
- 当前主要瓶颈不再是单个 bug，而是方法本身对真实身份建模和稳定纹理融合的能力不足

本计划的目标不是继续做局部修补，而是把项目逐步升级到一条更适合“高真实性人脸重建”的路线，同时尽量复用现有工程。

目标优先级如下：

1. 明显提升灰模对本人身份特征的保留
2. 降低侧脸、下颌缘、鼻翼等区域对稀疏关键点的过度依赖
3. 让多视图纹理融合建立在更可信的几何基础上
4. 保持现有输出格式、调试能力和前端展示链路尽量不变

## 为什么现在要换模型

当前方案的问题已经比较清楚：

- 关键点拟合只能保证稀疏对齐，不能保证高真实性几何
- 多视图纹理融合需要大量 hand-crafted 规则，边缘和接缝不稳定
- 即使 reprojection error 不高，也不等于模型真实像本人
- 继续在旧方法上做小修，收益会越来越小

一句话判断：

- 如果目标只是“能看出是这个人”，当前路线还能继续修
- 如果目标是“越真实越好”，现在应该把主要投入转向更强的初值模型和更强的重建范式

## 总体策略

建议采用“分层替换、逐步验收”的迁移方式，而不是一次性推翻整个项目。

总体架构调整为：

1. 用更强的 identity 初始化替换当前弱初值
2. 用更强的 per-view 几何/表情初始化替换当前单纯 landmark 驱动的起点
3. 保留现有 FLAME 优化、相机、导出和前端链路
4. 在新的几何基础上再重构纹理融合

建议优先路线：

- `MICA` 负责 shared identity shape 初值
- `DECA` 负责 per-view pose / expression / 局部几何补偿
- 必要时保留 `EMOCA` 作为可选后端

## 分阶段计划

### 阶段 1：统一初始化后端

目标：

- 清理当前初始化入口
- 为 `legacy / deca / mica / mica_deca / mica_emoca` 建立统一的 backend 抽象

要做的事：

- 在 `src/initializers/` 下整理初始化模块
- 让 `src/module2_geometry.py` 不再直接绑定单一初始化逻辑
- 在 `src/config.py` 中明确初始化后端配置
- 保留旧链路作为 fallback

建议新增或整理的文件：

- `src/initializers/__init__.py`
- `src/initializers/face_alignment_initializer.py`
- `src/initializers/deca_initializer.py`
- `src/initializers/mica_initializer.py`
- `src/initializers/emoca_initializer.py`

验收标准：

- 可以通过配置切换不同初始化后端
- 旧 pipeline 不被破坏
- 缺模型或缺权重时能给出可执行的报错信息

### 阶段 2：让 MICA 成为默认 identity 初值

目标：

- 让项目默认不再依赖弱 landmark-only identity 起点
- 输出更像本人的 shared shape

要做的事：

- 完整接入 `MICA`
- 明确 `MICA` 输出是 embedding、shape code 还是可直接转 FLAME 参数
- 把多视图的 identity 信息融合成一份 shared shape init
- 保存调试产物，方便与旧初始化对比

建议调试输出：

- `output/debug/init_mica_shape.json`
- `output/debug/init_view_left.json`
- `output/debug/init_view_front.json`
- `output/debug/init_view_right.json`
- `output/debug/init_reproj_left.png`
- `output/debug/init_reproj_front.png`
- `output/debug/init_reproj_right.png`

验收标准：

- 只看灰模，身份相似度已经优于旧方案
- 正脸和侧脸轮廓更稳定
- reprojection 不明显恶化

### 阶段 3：引入 DECA 做 per-view 几何增强

目标：

- 在 shared identity 的基础上补强局部几何
- 改善鼻、嘴、眼周、下巴等细节区域

要做的事：

- 接入 `DECA` 编码路径
- 明确 `shape / expression / pose / detail` 哪些进入当前工程
- 决定 `DECA` 只做初始化还是参与更深一层几何生成
- 输出每视图初始化来源说明

推荐策略：

- identity 仍以 `MICA` 为主
- per-view expression / pose 由 `DECA` 提供
- 局部 detail 先作为可选增强，不要一开始就强耦合

验收标准：

- 鼻翼、嘴唇、下巴转折较纯 `MICA` 更自然
- 左右视图拟合稳定性提升
- side profile 比旧方案更可信

### 阶段 4：基于新几何重构多视图纹理融合

目标：

- 不再让当前贴图模块单靠规则硬撑全局
- 让贴图逻辑与“可信几何 + 明确可见性”对齐

要做的事：

- 重构 `src/module3_texture.py`
- 明确区分真实采样区域与 inpaint 区域
- 前脸仅负责中央高可信区域
- 左右脸仅负责各自近侧区域
- 输出主来源视图图、采样置信图、遮挡拒绝图

建议保留/新增诊断图：

- `uv_view_source_map.png`
- `uv_view_confidence.png`
- `uv_view_count.png`
- `left_glb_reproject_compare.png`
- `front_glb_reproject_compare.png`
- `right_glb_reproject_compare.png`

验收标准：

- 侧脸不再被 front 大面积污染
- 接缝缩小到局部区域
- 右脸和左脸差距显著收敛

### 阶段 5：决定是否进一步切换到更强范式

目标：

- 判断当前 FLAME 框架是否还值得继续投入

判断标准：

- 如果灰模已明显像本人，纹理问题也能控制在局部范围，则继续沿当前项目优化
- 如果灰模仍不够像，或者纹理总是脆弱，则不要继续打补丁，应考虑更强的 dense reconstruction / photometric fitting 路线

这一步先不动代码，先做决策。

## 推荐实施顺序

推荐按下面顺序推进，不要并行大改：

1. 完成初始化后端抽象
2. 让 `MICA` 成为默认 identity 初值
3. 接入 `DECA`
4. 重新评估灰模质量
5. 再重构纹理融合
6. 最后决定是否继续留在当前框架

## 模块改动范围

优先会涉及这些文件：

- `src/config.py`
- `src/module2_geometry.py`
- `src/module3_texture.py`
- `src/initializers/*.py`

可能新增：

- `docs/` 下的依赖说明
- checkpoint 配置说明
- 调试输出说明

尽量不要动这些部分，除非确实必要：

- 前端加载逻辑
- GLB 导出接口
- FastAPI 服务层

## 配置建议

建议在 `src/config.py` 中显式支持以下配置：

- `INIT_BACKEND`
- `MICA_DIR`
- `MICA_CKPT`
- `DECA_DIR`
- `DECA_CKPT`
- `EMOCA_DIR`
- `EMOCA_CKPT`
- `INIT_DEVICE`
- `ALLOW_LEGACY_FALLBACK`
- `SAVE_INIT_DEBUG`

推荐后端枚举：

- `legacy`
- `deca`
- `mica`
- `mica_deca`
- `mica_emoca`

## 验收指标

不要只看最终 `glb` 主观感觉，建议每阶段都固定看这几类指标：

### 几何指标

- 初始 reprojection error
- 优化后 reprojection error
- 左/前/右三视图误差是否均衡
- side profile 是否更稳定

### 视觉指标

- 灰模是否更像本人
- 鼻梁、鼻尖、下颌、面颊转折是否更自然
- 左右脸轮廓是否更接近照片

### 纹理指标

- 左右脸是否仍被 front 污染
- 接缝是否缩小
- 空洞 / 纹理漂移是否下降

## 风险与应对

### 风险 1：MICA / DECA 接入复杂，短期反馈变慢

应对：

- 保留 `legacy` fallback
- 每阶段都要求可独立运行
- 不要一次同时重写 geometry 和 texture

### 风险 2：初值换强了，但纹理模块仍拉胯

应对：

- 把 geometry 提升和 texture 重构分阶段验收
- 先确认灰模提升，再进入贴图层

### 风险 3：引入新模型后依赖环境不稳定

应对：

- 把 checkpoint / external repo / import path 统一配置化
- 优先做 encode 路径最小闭环
- 避免一开始就引入不必要的大依赖

## 给 Claude 的执行要求

如果把代码任务交给 Claude，建议明确要求它按以下原则执行：

1. 先做初始化后端抽象，再接模型，不要反过来
2. 每一步都保留 fallback，不允许把旧链路一次删掉
3. 每引入一个新后端，都必须有调试输出
4. 不允许把大量逻辑直接堆进 `module2_geometry.py`
5. 在灰模明显提升之前，不要开始大改贴图逻辑

## 当前建议结论

基于目前项目状态，建议立即执行的决策是：

- 停止把主要精力放在旧纹理链路的细节修补上
- 先把 `MICA + DECA` 这一层接稳
- 用新几何重新评估贴图问题

一句话总结：

先换强初值，再谈高质量贴图；否则会一直在“把真人照片贴到不够像的头模上”这个问题里打转。

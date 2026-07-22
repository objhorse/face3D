# 几何优先的严格投影纹理设计

## 1. 背景与问题定义

当前模型鼻头附近的黑影、鼻孔纹理爬到鼻头、五官纹理与几何边界不重合，并不是一个孤立的贴图参数问题，而是现有纹理架构允许不同证据使用不同坐标造成的系统性问题。

当前流程先把 mesh 点投影到原图坐标 `proj`，用该坐标进行深度和遮挡判断；随后纹理注册又通过 TPS/RBF 等非刚性图像变形得到 `sample_proj`，实际 RGB 和 mask 却在 `sample_proj` 采样。这意味着系统可能使用原射线通过可见性检查，却从另一条射线取得颜色。稀疏 landmark 残差可以因此显著下降，但像素不再必然对应同一个三维表面点。

此外，当前所谓 albedo 主要做色度归一化，没有可靠分离低频光照。原图中的鼻孔阴影、鼻翼阴影和高光会直接烘焙进纹理，查看器再次施加 PBR 光照后形成双重明暗，进一步放大“黑环”和几何错觉。

因此第一阶段采用路线 A：先建立一个**几何与相机冻结、所有证据使用同一投影射线、禁止非刚性挪动像素**的严格纹理基线。它的首要价值是让错误变得诚实、可定位，再在这个基线上改进几何、相机和外观分离。

## 2. 目标

1. 建立 `strict_projective` 纹理模式，RGB、mask、深度、可见性和语义均使用同一个未经变形的原图像素坐标。
2. 纹理阶段不得修改 mesh、相机参数或投影坐标，不得用图像变形掩盖几何/相机误差。
3. 将真实投影不一致显式记录为低置信区域，并在质量报告中定位到视角和语义部位。
4. 为鼻子、眼睛、嘴部等高风险区域建立语义一致性和视角归属规则，防止错误纹理跨器官传播。
5. 分离或抑制输入图像中的低频光照，减少烘焙阴影与查看器光照叠加。
6. 保持现有固定拓扑、API 返回路径和 GLB 查看方式兼容。

## 3. 非目标

1. 本阶段不修改嘴部前突、鼻形、眼睑等几何形状。
2. 本阶段不实现几何、相机、材质和光照的联合可微优化。
3. 本阶段不使用神经纹理、生成式补脸或跨语义 inpaint。
4. 本阶段不重新采集数据，也不新增深度图输入。
5. 本阶段不声明医疗级几何或颜色精度。
6. 本阶段不删除 legacy 注册代码，但它不得进入 strict 模式的最终烘焙路径。

## 4. 核心架构决策

### 4.1 单一投影契约

每个 UV texel 在每个视角上只允许产生一份不可变的 `ProjectionSample`：

```text
ProjectionSample
  world_point          mesh 上的三维点
  camera_point         相机坐标系三维点
  pixel_xy             K/R/t 直接投影得到的原图坐标
  expected_depth       camera_point.z
  observed_depth       同一 pixel_xy 的 z-buffer 深度
  visible              深度与朝向联合判断结果
  normal_cosine        表面法线与视线夹角
  semantic_region      mesh 语义区域
  image_semantic       同一 pixel_xy 的图像语义
  source_confidence    该视角对该 texel 的置信度
```

以下数据必须全部在同一个 `pixel_xy` 读取：

- RGB
- face/skin mask
- semantic mask
- depth buffer
- occlusion 判定
- feature/landmark 距离场
- 颜色与光照估计样本

strict 模式中不允许出现 `sample_proj != proj`。若需要保留 warp 结果做对照，只能写入 debug 产物，不能进入最终纹理。

### 4.2 几何与相机冻结

纹理阶段输入以下只读对象：

- stable mesh 顶点、面和 UV
- 每视角内参 `K`
- 每视角外参 `R/t`
- 相机命名与被摄者左右语义
- 图像尺寸和预处理坐标变换

纹理开始时记录这些对象的 hash，完成后再次校验。任何变化均视为实现错误并使任务失败。

相机投影必须经过唯一公共函数，禁止 module 内部各自复制坐标翻转、缩放或 crop 逻辑。原图坐标、预处理图坐标和纹理坐标必须使用显式命名的数据结构转换，不能依赖隐含约定。

### 4.3 Truth Mode

新增 truth mode，直接暴露几何和相机在原始观测上的真实误差：

- 禁用 TPS/RBF/local similarity 等非刚性采样变形。
- 注册后的 landmark 残差只作为对照，不作为 strict 模式成功指标。
- 报告必须展示未变形投影轮廓、语义边界和原图叠加。
- 若五官不重合，优先归因到几何、相机、坐标变换或语义定义，不允许纹理阶段移动像素补偿。

### 4.4 语义兼容门禁

几何语义与图像语义必须兼容后才允许采样。例如：

| mesh 区域 | 允许的图像语义 | 禁止来源 |
| --- | --- | --- |
| 鼻梁/鼻头皮肤 | skin、nose skin | nostril、mouth、background |
| 鼻翼外侧 | nose skin、相邻 cheek skin | nostril interior、lip |
| 鼻孔/鼻小柱下方 | nostril、nose underside | nose tip highlight、cheek |
| 眼睑皮肤 | eyelid、periocular skin | eyeball、eyelash dark band |
| 嘴唇 | upper/lower lip 对应语义 | skin、mouth cavity |
| 人中/口周皮肤 | skin | lip、mouth cavity |

若 FLAME 当前拓扑没有可靠表示鼻孔内腔，鼻孔暗色只能投到明确的鼻底/鼻孔语义区域；无法匹配的暗像素应降低置信度，不得被拉伸到鼻头表面。

### 4.5 视角归属与融合

多视角融合不再只依赖法线夹角，而是结合语义归属：

- 正面视角优先负责鼻头、鼻小柱、正面鼻翼边界、眼睑、嘴唇和人中。
- 侧面视角优先负责侧脸、太阳穴、耳朵、外侧脸颊和对应一侧鼻翼。
- 侧面视角不得覆盖鼻头正面、对侧鼻孔或跨越鼻梁中线。
- 融合只能在同一语义区域内部进行，禁止 multiband 或 inpaint 跨越鼻孔/鼻皮肤、嘴唇/皮肤等边界。
- 视角间证据冲突时降低 confidence，而不是平均出一块位置错误的纹理。

### 4.6 外观与光照分离

第一阶段采用受限的低频光照校正，不追求完整 inverse rendering：

1. 在高置信皮肤区域估计每视角低频 luminance/shading field。
2. 排除眼睛、眉毛、鼻孔、嘴唇、头发、背景和高光异常点。
3. 将输入颜色分为低频照明与保留身份细节的高频反射成分。
4. 跨视角只对低频颜色和亮度做稳健匹配，不抹除毛孔、痣和局部肤色细节。
5. 输出名称必须准确：未完成光照分离时称 `baked_color`，只有经过校正的结果才称 `albedo`。

这样可以减少鼻翼阴影被永久烘焙后又被查看器灯光重复加深的问题。

### 4.7 置信度是一等输出

每个 texel 生成 confidence，至少包含：

- visibility confidence
- view-angle confidence
- semantic compatibility confidence
- projection/feature-distance confidence
- photometric consistency confidence
- source-view identity

低置信区域允许使用同一语义连通域内的保守补色，但不得跨五官边界。最终 GLB 可以生成，但质量报告必须明确标黄；关键区域低置信超过阈值时不得标记为高质量成功。

## 5. 数据流

```text
三视角原图 + masks + semantic maps
                |
                v
冻结的 mesh + K/R/t + 坐标变换
                |
                v
UV rasterization：texel -> 三维表面点
                |
                v
唯一 project()：三维点 -> 各视角 pixel_xy
                |
                v
同一 pixel_xy 完成 depth / visibility / semantic gate / RGB 采样
                |
                v
低频照明校正 + 同语义视角归属与融合
                |
                v
baked_color/albedo + confidence + source-view map
                |
                v
GLB + strict projective quality report
```

## 6. 模块边界

### 6.1 新模块

- `src/appearance/projective_sampling.py`
  - 定义投影数据结构。
  - 统一 UV texel 到原图像素的映射。
  - 保证所有采样证据共享同一坐标。

- `src/appearance/semantic_visibility.py`
  - 负责深度、遮挡、法线、语义兼容和视角归属。
  - 输出逐视角的 valid/confidence/source 信息。

- `src/appearance/intrinsic_appearance.py`
  - 负责低频照明估计和受限颜色一致化。
  - 不负责几何或像素位置调整。

- `src/reports/projective_texture_report.py`
  - 生成原图投影叠加、source-view map、confidence map、鼻/眼/嘴局部图及指标。

### 6.2 修改模块

- `src/appearance/stable_texture.py`
  - 增加 strict projective 烘焙入口。
  - 只消费 `ProjectionSample`，不接收任意二维 warp。

- `src/module3_texture.py`
  - 根据配置选择 strict 或 legacy。
  - strict 路径中彻底移除 `sample_proj` 分支。

- `src/pipeline/stable_three_view.py`
  - 在纹理前冻结并记录 geometry/camera hash。
  - 在纹理后执行一致性门禁和报告生成。

### 6.3 降级为 legacy/diagnostic

- `src/appearance/stable_texture_registration.py`
- `src/appearance/texture_registration.py`

它们可以继续生成“如果允许变形，二维 landmark 能下降多少”的诊断，但不得给 strict 最终 GLB 提供采样坐标。

## 7. 配置设计

```python
STABLE_TEXTURE_MODE = "strict_projective"
STABLE_ENABLE_NONRIGID_TEXTURE_WARP = False
STABLE_TEXTURE_TRUTH_REPORT = True
STABLE_TEXTURE_ENABLE_INTRINSIC_CORRECTION = True
STABLE_TEXTURE_FEATURE_WARNING_PX = 5.0
STABLE_TEXTURE_FEATURE_REJECT_PX = 12.0
STABLE_TEXTURE_SEMANTIC_GATE = True
```

规则：

- strict 模式强制 `STABLE_ENABLE_NONRIGID_TEXTURE_WARP=False`；配置冲突应直接报错，不能静默 fallback。
- 特征残差超过 warning 时降低局部置信度。
- 超过 reject 时该视角不得为对应语义区域提供纹理，但整张模型可使用其他可信视角或同语义补色继续输出。
- 若所有视角对关键区域均被 reject，模型仍可输出诊断 GLB，但 session 不得标记为高质量成功。

## 8. 失败行为

以下情况必须失败，不得回退到非刚性 warp：

1. 缺少或无法解析相机参数。
2. 相机命名与被摄者左右语义不明确。
3. 原图与预处理图坐标变换不可逆或尺寸不一致。
4. 投影产生 NaN/Inf、负深度或越界比例异常。
5. RGB、mask、depth 使用了不同采样坐标。
6. 纹理阶段 geometry/camera hash 发生变化。
7. strict 调用栈中出现 TPS/RBF/nonrigid warp。

以下情况允许输出，但必须降级质量状态：

1. 鼻、眼、嘴等关键区域投影残差偏高。
2. 某些区域只有单视角有效纹理。
3. 低频光照无法稳定估计。
4. 不可见区域需要同语义补色。

## 9. 质量报告

报告必须同时展示“好看程度”和“证据是否真实一致”，至少包含：

- 三视角原图与未变形几何轮廓叠加。
- mesh 语义投影与图像语义边界叠加。
- RGB/visibility/semantic 实际采样坐标一致性检查。
- 每个视角的 source-view ownership map。
- confidence map 和被 reject 的区域。
- 鼻头、鼻翼、鼻孔、眼睑、嘴唇局部截图。
- 低频照明校正前后对比。
- 未变形投影残差；不得把 warp 后残差作为 strict 成功指标。
- 最终几何与输入几何的 hash、顶点数和面数一致性。

## 10. 测试策略

### 10.1 单元测试

1. `ProjectionSample` 中 RGB、mask、depth、semantic 查询必须共享完全相同的浮点坐标和取整规则。
2. 构造遮挡样例，验证 visibility 在实际 RGB 采样坐标上判断。
3. 构造鼻孔像素投向鼻头皮肤的样例，验证 semantic gate 拒绝该样本。
4. strict 模式传入 warp 时必须抛错。
5. 纹理前后 mesh 和 camera hash 必须相同。
6. source-view 融合不得跨语义边界。
7. 合成光照样例中，低频明暗被削弱而高频皮肤细节被保留。

### 10.2 集成测试

使用至少两个现有三视角数据集：

- `captures_20260612_135253`
- `captures_20260612_140210`

每个数据集都生成 legacy 与 strict 对比：

- textured GLB
- clay GLB
- source-view map
- confidence map
- 三视角投影报告
- 鼻/眼/嘴局部对比

### 10.3 回归测试

- 固定拓扑不删面，顶点数和面数不变。
- API 和 viewer 仍从约定路径读取 GLB。
- legacy 模式可显式启用，但不是默认模式。
- 原有坏样例中的鼻头黑环和鼻孔爬升必须在报告中被语义门禁识别。

## 11. 验收标准

### 11.1 结构性硬指标

- strict 最终路径中 `max(abs(sample_proj - proj)) == 0`，或不存在 `sample_proj` 概念。
- RGB、mask、depth、semantic 的采样坐标逐样本一致。
- strict 调用栈不包含 TPS、RBF 或其他非刚性二维 warp。
- 纹理阶段 mesh/camera hash 完全不变。
- 最终面数等于输入稳定 mesh 面数，不通过删面修贴图。
- 鼻孔/口腔暗像素不能落到不兼容的皮肤语义区域。

### 11.2 视觉与报告指标

- 鼻头不再出现由鼻孔纹理上移形成的连续黑环。
- 原图阴影在 PBR 查看器中不再形成明显双重阴影。
- 正面中央五官在有效区域优先来自正面相机，来源可在 ownership map 中验证。
- 多视角交界色差下降，且过渡不跨语义边界。
- 任何未解决的五官错位在 truth report 中可直接定位，不能由 warp 后数字掩盖。

### 11.3 第一阶段成功定义

第一阶段不以“立即比 legacy 更像”为唯一标准，而以建立可信观测链为硬前提。在此基础上，鼻头黑影和明显器官纹理错位应减少；若仍存在错位，报告必须能够证明它来自几何、相机或语义模型，并为下一阶段提供可优化的真实残差。

## 12. 实施批次

### A1：Truth Diagnostic

- 复用现有 mesh 和相机，生成完全未变形的三视角几何/语义投影。
- 输出实际 sampling source 和 confidence，不改变当前默认 GLB。
- 验证相机命名、坐标转换和鼻部不一致的真实位置。

### A2：Strict Projective Sampler

- 实现统一 `ProjectionSample`。
- 将 depth、mask、semantic、RGB 收敛到同一坐标。
- 生成第一个无 warp 的对比 GLB。

### A3：Semantic Ownership And Gates

- 增加鼻、眼、嘴等语义兼容规则。
- 增加视角归属和冲突降置信机制。
- 输出 ownership/confidence map。

### A4：Intrinsic Appearance Correction

- 估计并削弱每视角低频照明。
- 做稳健跨视角亮度/色彩一致化。
- 保留高频皮肤身份细节。

### A5：Integration And Acceptance

- 接入 stable pipeline 配置选择。
- 完成两个数据集的 legacy/strict A/B。
- 通过结构性门禁后再讨论下一阶段几何或联合优化。

## 13. 兼容与回滚

- 现有 `face.glb`、纹理路径、API 和 viewer 接口不变。
- strict 模式经过验收前，可由实验配置启用；通过验收后再设为默认。
- legacy 注册保留为显式模式和诊断工具，不删除历史能力。
- 所有 A/B 结果写入独立实验目录，不覆盖用户认可的 baseline。

## 14. 风险与假设

- 输入仍只有三视角 RGB，没有真实深度图；未观测区域只能低置信补全。
- 固定外参仍可能存在小误差，strict 模式会更直接地暴露它们。
- FLAME 拓扑对鼻孔内腔和眼睑接触关系表达有限，语义门禁只能避免错误扩散，不能凭空补出几何。
- 第一版 strict 结果可能局部比 warp 版本更朴素，但它提供的是可验证的底层基线。
- 若 strict 投影在多个数据集上都系统偏移，下一阶段应优化统一相机/几何坐标链，而不是恢复纹理 warp。

## 15. 决策记录

- 2026-07-22：确认路线 A 为第一阶段。
- 第一阶段采用几何优先、严格投影、无非刚性纹理 warp 的架构。
- 现有 TPS/RBF 注册只保留作 legacy 和诊断，不作为最终真实纹理链。
- 完成路线 A 并取得可信残差后，再评估路线 B 的联合可微优化。

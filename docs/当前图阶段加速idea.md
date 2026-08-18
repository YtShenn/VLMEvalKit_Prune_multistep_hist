现在的项目中，RUN_BASH/run_android_control_history_state_packet_qwen3_5.sh这个已经实现了对历史图的裁剪加速，以及decode阶段的序列解码加速，但对当前图片current screenshot还没有做任何加速处理，有没有什么好的创新可以发文章的思路？


**方向一**
`Predict-Then-Perceive` 的核心思想是：当前图最贵，不该先“看全”再决定动作，而应该先用任务和历史预测“下一步大概率发生在哪”，再把视觉预算花到这些区域。

现在你们的 current screenshot 是直接整张进 prompt 的，[android_control_curated.py (line 704)](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/android_control_curated.py:704)。可以把它改成两阶段：

1. `Action Prior Predictor`
   输入只用文字历史和历史 action packet，不看当前图或只看极低分辨率 thumbnail。
   输出：
   * `action_type` 先验分布
   * 1 到 K 个候选 ROI
   * 每个 ROI 的置信度
   * 是否需要 full-screen fallback

这个 predictor 不一定要重训练大模型，最轻量的实现是：

* 从历史动作序列提取状态转移模式
* 用小 MLP / 小 transformer / 甚至一个额外 prompt head，预测下一步关注区
* ROI 可以定义成 normalized box，比如 `[cx, cy, w, h]`

2. `Adaptive Perception Builder`
   对当前图只构造：
   * 一张低分辨率 global thumbnail
   * K 个高分辨率 ROI crop
   * 可选一个 uncertainty crop，比如 top-1 和 top-2 候选的并集区域

最后 prompt 不是“1张完整 current screenshot”，而是：

* `Current Global View`
* `Predicted Region 1`
* `Predicted Region 2`
* `...`
* 并明确要求 bbox 仍输出在原图坐标系

3. `Verification / Refinement`
   模型先基于这些局部图输出粗 bbox 和 action。
   如果置信度低，或者输出不合法，再触发第二轮：
   * 扩大 ROI
   * 增加一个邻域 crop
   * 最坏情况回退 full screenshot

这样就形成一个 budgeted inference。论文点不只是 speedup，而是“GUI action prediction can guide visual acquisition”。

**怎么训练**
最自然的监督信号来自当前 step 的 GT bbox / point。
你可以把 predictor 训练成：

* 给当前 step 预测一个包含 GT 的 ROI
* 优化 `IoU(ROI, GT bbox)` 或 `point-in-box`
* 再加一个 cost regularizer，惩罚过大的 ROI

损失可以是：

* `L_action_prior`
* `L_roi_regression`
* `L_budget = lambda * visual_tokens`

这会变成一个 accuracy-cost tradeoff 学习问题，很像“learned adaptive compute”。

**为什么这条线有论文味**
它和纯 token pruning 不一样。不是“删 token”，而是“先决策，再感知”。这比 ShowUI/FocusUI 那类 instruction-conditioned selection 更像 agent setting，因为它显式利用了 action history 和 temporal policy。

---

**方向二**
`Delta Packet for Current Screenshot` 的核心思想是：当前图的信息不是绝对的，而是相对上一帧“哪里变了、怎么变了”。GUI 多步任务里，真正重要的往往是新弹窗、列表滚动后的新增区域、按钮状态变化，而不是整张图。

这个方向可以把当前图表示成一个“变化包”而不是一张完整图：

1. `Reference Frame`
   拿最近一张历史图 `I_{t-1}` 和当前图 `I_t` 做对比。
   由于 AndroidControl 有 scroll、swipe、页面跳转，不能直接做像素 diff，需要更鲁棒一点的差分：
   * 低成本版：thumbnail 后做 SSIM / abs diff / edge diff
   * 稍强版：patch-level feature diff
   * 更论文版：instruction-conditioned change map
2. `Delta Packet Builder`
   从差分图里提取若干块：
   * `global current thumbnail`
   * `changed region crops`
   * `persistent anchor crop`
   * `motion strip`

这里有几个细节很关键：

* 对 `click/tap` 类动作，变化往往局部，适合小 ROI
* 对 `scroll/swipe` 类动作，变化是条带式，适合保留竖条或横条，不要只切正方形
* 对 `open_app / navigate_home / page jump`，变化可能是全局语义变化，这时 delta 不够，要保留更大的 global view

所以 delta packet 最好是 action-aware 的：

* `tap packet = thumbnail + local changed boxes`
* `scroll packet = thumbnail + vertical strip + edge anchors`
* `page transition packet = larger thumbnail + top changed box`

3. `Temporal Alignment`
   真正难点在于 scroll 后内容整体平移，简单 diff 会把整页都判成 change。
   一个很好的创新点是先做粗对齐，再取 delta：
   * 估计 vertical / horizontal shift
   * 或者在 patch grid 上做最大相关匹配
   * 对齐后再找 residual change

这一步如果做出来，论文会更扎实，因为它体现了 GUI temporal structure，不是普通图像变化检测。

4. `Prompt Packing`
   当前图不再只是一张 image，而是：
   * `Current thumbnail`
   * `Delta region 1: newly appeared / changed`
   * `Delta region 2`
   * `Anchor region: unchanged context`
   * 可选 `Last screenshot thumbnail for alignment reference`

模型既看到“现在长什么样”，也看到“哪里和上一步不一样”。

## 详细方案

**总体定义**

这条线可以正式定义成：

`Delta Packet for Current Screenshot`
给定历史截图 `I_{t-1}`、当前截图 `I_t`、任务指令 `x`、历史动作 `a_{<t}`，不再把 `I_t` 作为一张完整图送进 VLM，而是构造一个紧凑的当前帧视觉包：

- 一个低分辨率全局图 `global thumbnail`
- 若干个高价值变化区域 `delta ROIs`
- 可选少量稳定锚点 `anchor ROIs`

目标是在尽量少的 visual tokens 下，保留完成当前动作最需要的信息。

你这篇 paper 的核心 claim 可以写成：

1. GUI 多步决策中，当前帧的信息需求高度不均匀。
2. 相比静态压缩，利用相邻帧的时间差分可以更有效分配视觉预算。
3. 对 scroll / popup / page transition 等 GUI 变化模式，动作感知的 delta packet 比均匀缩放或纯 ROI 裁剪更优。

---

**方法总览**

建议把方法分成 5 个模块：

1. `Reference Frame Selection`
2. `Temporal Alignment`
3. `Delta Saliency Estimation`
4. `Packet Construction`
5. `Budgeted Fallback`

整体流程是：

- 输入最近历史帧 `I_{t-1}` 和当前帧 `I_t`
- 先做粗对齐，消除滚动和平移
- 在对齐后空间中估计变化显著图 `delta map`
- 从 `delta map` 中选 K 个高价值 ROI
- 和一个全局缩略图一起组成 current delta packet
- 若变化图不可靠，则退化到更保守的 packet 或 full image

---

**1. Reference Frame Selection**

最简单先用最近一帧历史图 `I_{t-1}` 当参考帧。

后面可以扩展成两种设置：

- `single-ref`
  只用最近一帧，最快，最干净。
- `multi-ref`
  在最近 `m` 帧里选和当前动作最相关的一帧做参考。
  例如上一帧是 scroll 前，前两帧是点击前，可能前两帧更稳定。

论文第一版建议先做 `single-ref`，因为：

- 更容易实现
- 更容易讲清楚
- 更像“current screenshot delta packet”而不是复杂 memory system

在你们 repo 里，这个参考帧就是 `history_image_paths[-1]`，当前已经在 [android_control_curated.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/android_control_curated.py:647) 拿到了 history 列表。

---

**2. Temporal Alignment**

这是方法里非常关键的一步。GUI 里很多变化不是“内容变了”，而是“内容整体挪了”，尤其是 scroll。

如果不对齐，简单 diff 会把整屏都当成 changed region，delta packet 就失效了。

建议从轻到重做三档：

**2.1 Shift-only 对齐**
假设相邻帧主要是竖直或水平平移。

做法：

- 先把两张图缩到小分辨率 thumbnail
- 在有限位移窗口内搜索 `(dx, dy)`，找到使相似度最高的平移
- 用这个平移把 `I_{t-1}` 对齐到 `I_t`

相似度可以用：

- `L1` / `L2`
- `SSIM`
- edge map correlation

这个版本对 scroll 特别有效，而且实现便宜。

**2.2 Patch-grid 对齐**
把图切成 patch grid，在 patch 层面估计局部位移。
这能处理局部内容滑动和部分区域变化。

做法：

- 将 thumbnail 划分成 `h x w` 个小 patch
- 对每个 patch 找最优局部匹配
- 得到粗糙的 patch flow
- 再把参考帧 warp 到当前帧坐标

**2.3 Action-aware 对齐**
利用上一步历史动作类型约束对齐形式：

- `scroll up/down` 时只搜索纵向位移
- `swipe left/right` 时只搜索横向位移
- `click` 时优先假设小位移或无位移
- `navigate_home/open_app` 时降低对齐强度，保留更多 global context

论文第一版建议主方法用 `2.1 + action constraint`。
这已经足够有新意，而且更稳。

---

**3. Delta Saliency Estimation**

对齐之后，估计哪里真正重要。

建议把 `delta saliency` 分成三部分，再融合：

`S_delta = alpha * S_pixel + beta * S_structure + gamma * S_action`

**3.1 Pixel-level change**
最直接：

- RGB abs diff
- 灰度 diff
- 边缘 diff

作用：

- 找到明显新弹窗
- 找到按钮状态变化
- 找到新出现的列表项

**3.2 Structure-level change**
GUI 很多关键变化是边界、文本块、按钮轮廓变化。
建议额外加：

- Sobel / Canny edge diff
- 局部纹理或 patch embedding diff

作用：

- 过滤纯颜色噪声
- 强化控件级变化

**3.3 Action prior modulation**
虽然这篇先主打 delta packet，但仍然建议轻量引入动作先验，不然会漏掉“没明显变化但很重要”的区域。

例如：

- 上一步是 `scroll down`，那么当前目标更可能出现在中下部新增内容区域
- 上一步是 `open_app`，顶部标题栏和中部主内容区域都重要
- 上一步是 `click search box`，键盘区域可能重要

所以可以把 delta saliency 乘一个软 prior：
`S = S_delta * (1 + lambda * S_prior)`

但论文叙事上仍说主角是 delta，不把它写成第二个方法。

---

**4. Packet Construction**

这是最核心的输出格式设计。建议 current packet 不只是一堆变化块，而是有结构的。

**4.1 必备组件**
每个 current packet 至少包含：

- `G`: current global thumbnail
- `D1 ... Dk`: top-k delta ROIs

仅这两个就能跑起来。

**4.2 可选 anchor 组件**
为了避免模型失去坐标系和整体语义，建议加少量 anchor：

- `A_top`: 顶部状态栏 / 标题区域
- `A_bottom`: 底部导航栏 / 输入栏
- `A_center`: 中央稳定上下文

不是每次都加，建议在这些情况加：

- delta 很稀疏
- 发生 page transition
- 目标动作依赖全局布局

**4.3 ROI 选择策略**
从 saliency map 选 box 时不要只取 top pixel，建议：

1. 对 saliency map threshold
2. 连通域聚类
3. 对每个连通域生成 bbox
4. 按 score 排序
5. 做 NMS
6. 取 top-k

每个 ROI 的 score 可以是：

- saliency sum
- saliency mean
- area regularized score

推荐：
`score(box) = sum_saliency / area^tau`
这样能防止过大 box 抢分。

**4.4 ROI 形状**
这里是 GUI 场景里很值得强调的点：

- `tap/click` 类变化：局部方框
- `scroll` 类变化：长条带 ROI
- `popup` 类变化：中大矩形
- `keyboard` 类变化：底部横条

所以 ROI proposal 不应只有正方形。
建议支持三种模板：

- square
- horizontal strip
- vertical strip

这会成为 paper 的一个很好的 domain-specific insight。

**4.5 分辨率分配**
每个 ROI 不一定同分辨率。
可以做简单预算分配：

- global thumbnail: 最低分辨率
- top-1 delta ROI: 最高分辨率
- top-2/3 ROI: 中等分辨率
- anchor ROI: 低到中分辨率

这就形成 token budget schedule。

---

**5. Budgeted Fallback**

delta packet 最怕两种情况：

- 当前帧和参考帧差太大，导致 diff 几乎全屏
- 当前帧和参考帧差太小，但目标区域实际上很重要

所以一定要设计 fallback，不然方法不稳。

建议三种 fallback：

**5.1 Delta-too-large**
如果变化面积占比过大，比如超过 `r_large`：

- 不走细 delta ROI
- 改成 `larger thumbnail + 1 or 2 big ROIs`
- 甚至直接 full screenshot

这常发生在 app 切换、整页跳转。

**5.2 Delta-too-small**
如果变化太少，比如小于 `r_small`：

- 保留 top changed ROI
- 再补一个 center / prior-guided ROI
- 防止漏掉“静态但需要点击”的目标

**5.3 Low-confidence fallback**
如果第一轮 packet 推理后输出格式不合法、bbox 异常、或模型 self-confidence 低：

- 第二轮扩大 ROI
- 再不行就 full screenshot

论文里可以把这叫 `progressive recovery`.

---

**Prompt 形式**

现在你们 current 图只是一句 `Current Screenshot:` 加一张图，[android_control_curated.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/android_control_curated.py:704)。

delta packet 版建议改成：

- `Current Global Thumbnail:`
- `Current Delta Region 1:`
- `Current Delta Region 2:`
- `Current Anchor Region:`
- `The final prediction bbox_2d must still be on the original current screenshot.`

要点是让模型知道：

- 这些图都来自当前帧
- 有的图是变化区域，有的是全局参考
- 输出坐标必须映射回原图

如果你想更进一步，可以在文本里加入每个 ROI 的来源说明：

- `newly changed region`
- `scroll-exposed region`
- `stable context anchor`

这能帮助模型形成角色感知。

---

**坐标映射**

这是实现上必须做稳的地方。

每个 ROI 都要记录：

- 原图中的 `crop_xyxy`
- resize 前尺寸
- resize 后尺寸

模型如果在某个 ROI 上预测局部 bbox，需要映射回 current screenshot 原图坐标。

历史 packet 里你们已经有 `crop_xyxy` 的元数据，[state_packet.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/state_packet.py:254)。current delta packet 也沿用同样设计最合适。

如果还是让模型直接输出原图 bbox，也建议内部保留这些映射信息，后面做调试和 failure analysis 很重要。

---

**训练与推理两种路线**

**路线 A：零训练 / 训练无关**
这是最适合先出结果的。

做法：

- 对齐 + diff + heuristic ROI proposal
- 直接替换 current full screenshot 为 delta packet
- 不训练额外模块

优点：

- 实现快
- 适合先做 strong baseline
- 容易证明方法本身有效

缺点：

- ceiling 可能有限

**路线 B：轻量学习增强**
在路线 A 跑通后，加一个轻量模块学习：

- alignment parameter selection
- delta ROI ranking
- action-conditioned fusion weights

这时候你论文里可以写：

- training-free base variant
- learned enhanced variant

很适合做主表 + ablation。

---

**建议的算法版本**

为了方便写 paper，我建议方法分三版：

**DP-Base**

- nearest previous frame
- shift alignment
- abs diff + edge diff
- top-k connected-component ROIs
- global thumbnail + delta ROIs

**DP-Action**

- DP-Base
- 加 action-aware search constraint
- 加 action-specific ROI templates

**DP-Hybrid**

- DP-Action
- 加一个轻量 prior 或 uncertainty fallback

投稿时主方法可以写 `DP-Hybrid`，但实验里必须保留 `DP-Base`，这样更有说服力。

---

**实验设计**

**主比较对象**

1. Full current screenshot
2. Resized current screenshot
3. Naive current thumbnail + center crop
4. Random crop packet
5. Delta Packet Base
6. Delta Packet Action
7. Delta Packet Hybrid

**核心指标**

- task success
- action_type accuracy
- bbox / point accuracy
- prefill latency
- total latency
- current-frame visual tokens
- end-to-end token budget

**特别重要的分析维度**
按动作类型分开报：

- click / tap
- long_press
- swipe / scroll
- navigate_home / back
- open_app
- input_text

我很建议再按 GUI change pattern 分组：

- local popup
- list scroll
- full-page transition
- keyboard emergence
- subtle state change

这会让你的 paper 更像在研究 GUI 时序结构，而不是普通 VLM compression。

---

**论文贡献点可以怎么写**

可以写成这 3 条：

1. 提出 `Delta Packet for Current Screenshot`，利用时序变化而非静态压缩来构造当前帧紧凑表示。
2. 提出 action-aware temporal alignment 与 packet construction，使方法适应 scroll、popup、transition 等 GUI 特有变化模式。
3. 在 Android GUI agent setting 中，实现更优的 accuracy-latency-token tradeoff。

---

**在你们代码里怎么落地**

最合适的切入点有两个。

第一，在 [state_packet.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/state_packet.py:222) 旁边新增 current packet 构建函数，比如：

- `build_current_delta_packet(...)`

输入建议是：

- `current_image_path`
- `reference_image_path`
- `history_action_packets`
- `sample_index`

输出复用现有风格：

- `list[PacketImage]`
- `meta dict`

第二，在 [android_control_curated.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/android_control_curated.py:704) 和 [android_control_curated.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/AndroidControl_Curated/android_control_curated.py:738) 附近，把 current image 从：

- 一张完整图

替换成：

- `Current Global Thumbnail`
- `Current Delta Region i`
- optional `Current Anchor Region`

这样你几乎不需要改模型主干，只改 dataset prompt builder 和 packet builder，就能做第一版实验。

---

**我建议你先做的最小可发表版本**

如果我们要先做一个最稳的版本，我建议是：

1. 参考帧只用最近一帧
2. 只做 shift-based alignment
3. diff 用 `abs diff + edge diff`
4. current packet = `1 global thumbnail + 2 delta ROIs`
5. scroll 动作额外允许 `vertical strip ROI`
6. 加 `delta-too-large / too-small` fallback

这个版本已经足够形成一篇有清晰方法定义的 work，而且实现复杂度可控。

如果你愿意，我下一步可以直接继续给你出：

- 一版更“论文格式”的 Method 小节草稿
  或
- 一版严格对齐你们 repo 的模块设计清单，细到每个函数该新增什么参数、返回什么字段。

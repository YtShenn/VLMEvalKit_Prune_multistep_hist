
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

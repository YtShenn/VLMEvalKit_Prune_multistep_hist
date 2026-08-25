
**第二条很有论文味**
`Uncertainty-Aware Coarse-to-Fine Attention Budgeting`

很多 top-k attn 方法的问题，不是“找到错地方”，而是“太自信地只留一个地方”。

你可以改成两阶段：

* 第一阶段只做粗网格级判断，比如 `3x3` 或 `4x4` 的 attention routing。
* 第二阶段不是固定 top-k crop，而是看不确定性来分配预算。
  例如：
  * top1 和 top2 分差很小，就两个区域都保留
  * attention 很分散，就保留一个更大的 union crop
  * attention 很集中，才 aggressively prune

这样方法的关键词就不是 `top-k`，而是：
`adaptive compute allocation based on attention uncertainty`

这比普通 attn 剪枝更容易讲创新，因为你的贡献点变成：

* GUI grounding 里的错误很多来自 early over-pruning
* 不确定性感知的预算调度比固定 top-k 更稳
* 同样 token budget 下，adaptive allocation 比 static allocation 更优

这个方向很适合做漂亮 ablation：

* fixed top4
* fixed top8
* uncertainty-aware top4/top8 mixed
* full image fallback ratio

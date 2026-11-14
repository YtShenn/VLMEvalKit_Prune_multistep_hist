实现 GUIKV-SFD 的过程里，主要经历了这几类问题。
1. 一开始静态模板是逐 token 追加的
   - 原因：GUIKV 原先把任何 q_len > 1 的输入都当作一次新的 prefill；若把完整静态模板片段送入，会触发二次 KV 压缩，破坏已有缓存。
   - 临时方案：把静态模板拆成逐 token continuation，所以能稳定运行，但静态部分会有十几次 forward，导致总 decode 步数偏高、耗时变长。
   - 问题：这不符合论文 SFD“已知静态片段一次 KV-cache forward”的设计，因此此前 GUIKV+SFD 的 30 多步并不是论文中约 25 步的同类实现。
2. GUIKV 有“逻辑长度”和“物理 KV 长度”两套坐标
   - GUIKV 剪枝后，实际缓存只保留压缩后的 KV，例如逻辑 prompt 2892 token，但物理 KV 可能只有 1157。
   - RoPE 位置编码仍必须使用原始逻辑位置；而 attention mask / KV 索引必须使用物理压缩后的长度。
   - 初版片段级实现沿用了逻辑长度的 attention mask，导致张量维度不匹配，例如 mask 宽度是 2899、实际 KV 宽度是 1164。
解决方式：
- 静态片段 continuation 时，构造 key 维度为“物理 KV 长度 + 当前片段长度”的 4D causal mask；
- cache_position 采用物理位置；
- position_ids 保持逻辑位置，以确保 RoPE 与原始上下文对齐。
3. 4D mask 的 dtype 不匹配
   - 初版 mask 使用 float32，而模型 attention 的 query 是 bfloat16。
   - Flash/SDPA attention 要求 bias 和 query dtype 一致，因此报错 invalid dtype for bias。
解决方式：
- 4D mask 改为继承模型实际 dtype（本实验中为 bfloat16）。
4. GUIKV prefill 在后续层被误判为 continuation
   - 最隐蔽的问题：第 0 层完成 prefill 压缩后，立刻在共享 cache 上写入 _guikv_original_seq_length。
   - 后续 layer 只要看到这个全局属性，就错误地认为自己在做 continuation，于是只有 layer 0 被压缩，其他层仍保留完整 KV。
   - 这会使不同层的 KV 长度不一致，继续导致片段 mask 错配，也会让后续样本的逻辑长度异常累积。
解决方式：
- 判断是否 continuation 时，不再只看全局 metadata；
- 改为同时检查“当前 layer 是否已有物理 KV”：
  - 当前层缓存为空：这是本轮 prefill，应执行 GUIKV 压缩；
  - 当前层已有缓存：才是静态模板/动态 token 的 continuation。
最终小样本验证结果：
- 1 个 task，9 个连续样本；
- SFD 成功 9/9，零 fallback；
- 静态模板确认均为 mode=chunk；
- 平均静态模板 forward：2.78 步；
- 平均动态未知字段 forward：15.33 步；
- 平均总 decode：18.11 步。
和论文原文的 SFD 是否一致：
现在在 SFD 协议层面已经一致：
1. 多模态 prompt 只做一次 prefill；
2. 已知静态模板按“片段”一次性追加到已有 KV cache；
3. 只对 action、bbox 等未知字段进行自回归动态解码；
4. 不再重建完整 prompt，也不再把静态文本逐 token decode。
与论文方法唯一必然不同的是底层缓存实现：论文 PACT-GUI 使用其自身的 KV 选择/压缩机制；GUIKV 使用 GUIKV 的物理压缩 cache。因此 GUIKV 需要额外维护“逻辑 RoPE 位置”和“物理 KV 索引”两套坐标。但这属于 GUIKV 后端适配，不改变 SFD 的解码流程与统计口径。
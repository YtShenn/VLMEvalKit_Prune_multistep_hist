可以同步一部分，但不能把主要提升“原样同步”到真正的不加 decode prefill 路径里。

这次提升来自三类东西：

1. **当前截图尺寸反归一化**
   这个已经同步到普通路径了，因为它在 [model.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/vlm/qwen3_vl/model.py:104) 的 `generate_inner` 后处理里，和 template prefill 无关。只要 `QWEN3VL_ANDROID_DENORM_ON_INFER=1`，普通推理也会用。
2. **bbox 不再被截断**
   这个只属于 streaming constrained decoding。普通 `model.generate()` 是完整生成完再 decode 文本，不会中途用 parser 判断“bbox 是否结束”，所以没有同类 bug 可以同步。也就是说普通路径不会出现“parser 把 `2040` 截成 `2`”这个问题，因为普通路径没有这个 parser。
3. **action_type / bbox slot 约束**
   这是提升最大的部分，但它本质上就是 constrained decoding。比如先把输出限制在合法动作集合里，只有 `click/long_press` 再继续生成 bbox。这种能力如果同步到“不加 decode prefill”，那它就不再是纯普通 decode 了，只是换了名字的 constrained decode。

所以准确说：

- **能同步的**：坐标反归一化、输出 JSON 规范化、当前截图尺寸选择。
- **不能在纯普通 decode 中同步的**：action slot 约束、bbox slot 约束、结构 token 强制生成。
- **可以另做一个中间实验条件**：`no_template_prefill + android_constrained_decode`，不做静态模板预填充，但仍加 logits processor 约束动作/bbox。这个可以帮助区分“速度优化”与“约束带来的精度提升”。

如果你的目标是做公平消融，我建议保留三组：

```bash
# 纯普通 decode
QWEN3VL_ENABLE_TEMPLATE_PREFILL=0

# constrained decode，但不计入 template prefill 加速实验
QWEN3VL_ENABLE_ANDROID_CONSTRAINED_DECODE=1
QWEN3VL_ENABLE_TEMPLATE_PREFILL=0

# 当前 template prefill + constrained decode
QWEN3VL_ENABLE_TEMPLATE_PREFILL=1
```

这样论文/报告里就不会把“约束提升精度”和“decode prefill 加速”混在一起。


普通生成：
模型自己写整句
约束生成：
程序先把句子骨架钉住，模型只填空
比如 AndroidControl 里，现在更像是在做：
<answer>{"action_type": " [模型填动作] "如果动作是 click/long_press：, "bbox_2d": [ [模型填四个坐标] ]}</answer>



关键点是：`plain decode` 里模型不是在“7个动作标签里做一次分类”，而是在“整个词表的下一个 token”上做自回归生成；`constrained decoding` 则把这个问题改成了“只在合法动作集合里选”。这两者差别很大，所以动作本身也会变得更准，不只是 bbox 更准。

具体说，有几个原因：

1. 普通生成时，动作不是一个干净的分类任务模型每一步都在几万词表里算 logits。它要同时决定：
   `{"action_type": "navigate_back"}`
   里的结构 token、引号、逗号、字段名、动作词本身。所以它不是只比较 `swipe:up` 和 `navigate_back` 哪个分高，而是在和很多无关 token 一起竞争。
2. constrained decode 会把搜索空间压到“合法答案子空间”
   比如到动作槽位时，我们可以只允许：
   `click / long_press / input_text / swipe:up / swipe:down / swipe:left / swipe:right / navigate_back / navigate_home / wait`
   这时模型的分数会在这些合法动作上重新归一化。
   原来 plain decode 里，可能有一部分概率质量浪费在：

- 错的 JSON 结构
- 多余解释文字
- 半截字段名
- 先去生成 bbox
- 其他不合法 continuation

这些都会干扰动作选择。约束后，这些干扰全被砍掉了。

3. 动作词未必是“一个 token”
   是的，很多动作字符串很可能是多个 token，尤其像：

- `swipe:up`
- `navigate_back`

它们不一定是单 token，可能拆成类似：

- `swipe`
- `:`
- `up`

或者：

- `navigate`
- `_`
- `back`

所以 plain decode 里，即使第一步走对了，后面也可能漂掉。而 constrained decode 不是只约束第一个 token，而是约束“整个合法字符串前缀”，这样 `navigate_...` 之后只能往合法动作名继续补，不会中途拐到别的地方。

4. 约束实际上引入了“先验结构”
   对 AndroidControl 来说，动作集合很小，而且有明显条件分支：

- 如果是 `click / long_press`，才需要 bbox
- 如果是 `navigate_back / swipe:up`，就不该再生成 bbox

plain decode 需要模型自己学会这个条件逻辑。constrained decode 则是代码真的在执行这个逻辑，所以动作和后续结构都更稳定。

5. 你看到的提升，不等于“模型本身更懂了”，而是“解码器帮它少犯格式/路径错误”
   模型原始偏好可能已经差不多了，但 plain decode 时经常输在：

- schema 漂移
- 多 token 动作串生成偏航
- 不该生成 bbox 时还往 bbox 方向走
- 应该停的时候不停

constrained decode 把这些错误路径剪掉以后，最终落到正确动作上的概率就更高了。

一句话总结：

普通生成时，模型是在“全词表里一步一步瞎走”；
constrained decoding 时，我们把路栏起来，只让它在“合法动作/合法结构”里走。
所以即使是 `swipe:up`、`navigate_back` 这种看起来不需要 bbox 的动作，预测也会更正确。

如果你愿意，我下一条可以直接画一个 very concrete 的例子，对比：

- `plain decode` 下 `navigate_back`
- `constrained decode` 下 `navigate_back`

分别每一步 token 是怎么选的。

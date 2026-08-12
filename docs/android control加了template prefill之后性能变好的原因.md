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




# 8月12日新的正确的代码

你的怀疑是对的：**如果只是把“模型本来一定会生成的固定 token”并行喂进去，理论上不应该显著提高任务能力。**

但我们现在这个实现并不完全等价于 baseline，它其实引入了一个很强的 **constrained decoding / output prior**，所以准确率变好是可能的。

主要原因有几个：

1. **输出顺序变了：action-first**

原始 prompt 写的是：

```text
{"bbox_2d": [...], "action_type": ACTION_TYPE}
```

但我新实现走的是：

```text
{"action_type": "...", "bbox_2d": [...]}
```

这不是纯加速了。它改变了自回归决策顺序。

baseline 是先生成 bbox，再生成 action。  
structured 是先生成 action，再根据 action 决定要不要 bbox。

对 AndroidControl 来说，这很可能更合理，因为：
- `wait` / `navigate_back` / `swipe` 本来不需要 bbox
- 先确定动作类型，再决定是否预测坐标，更符合任务结构
- 先 bbox 再 action 时，前面坐标生成错了会污染后面的 action token

所以 Type 和 SR 提升，很可能有一部分来自 **action-first 结构化约束**，不是来自并行 decode 本身。

2. **固定 JSON 被 teacher-forcing 了**

baseline 需要模型自己生成：

```text
<answer>
{
"bbox_2d"
:
...
}
</answer>
```

这些格式 token 如果有任何偏移，比如 key 名错、引号错、标签错、额外解释、提前结束，都会影响 parser 或后续生成。

structured decode 等于强制告诉模型：

```text
现在你就在这个 JSON 结构里，只需要填 action 和 bbox
```

这会减少格式错误和解码漂移。它不是提升视觉理解，而是减少输出协议错误。

3. **减少固定模板 token 的贪心误差传播**

即使固定 token 很简单，baseline 也是一步步生成。某一步如果生成了不同空格、不同 key 顺序、额外 token，后续上下文就变了。

structured path 直接把模板作为已知上下文喂进去，后续 slot 的预测上下文更干净：

```text
<answer>{"action_type": "click", "bbox_2d": [
```

这对 bbox 数字生成会更稳定。

4. **官方 eval 对格式很敏感**

AndroidControl 的指标不是只看语义，还要 parse 出：

```text
action_type
bbox_2d
```

structured 输出天然更容易被 parse 对。SR 提升十几个点，可能有一部分来自“模型本来意思对，但 baseline 格式/顺序/解析失败或类型错”的样本被救回来了。

所以结论是：

**现在这个 structured decode 不是纯粹的无损加速实验，它同时是一个结构化约束解码实验。**

如果要验证“纯加速是否无损”，需要做 ablation：

1. 做一个 **bbox-first structured decode**，严格匹配原 prompt 的 key 顺序：
   ```text
   {"bbox_2d": [...], "action_type": "..."}
   ```

2. 或者把 baseline prompt 也改成 action-first，然后不用 structured decode 跑一次。

3. 对比 per-sample detail，看提升来自：
   - Type 纠正
   - bbox 纠正
   - parser 成功率提升
   - 非 click 动作不再强行带 bbox

我现在倾向的判断是：SR 提升主要不是“序列并行”带来的，而是 **action-first + JSON 模板约束** 带来的。加速本身减少 wall time；结构化约束改变了解码分布，因此可能提高或降低 accuracy。

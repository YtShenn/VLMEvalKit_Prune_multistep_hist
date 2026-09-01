# 实验结果简报

本文档汇总当前在 AndroidControl 多步 GUI 控制任务上完成的几类创新操作：`State Packet`、`Structured Fast Decode`、当前帧 `Attention Prune`，并整理对应的 `Type / Grounding / SR`、运行时间、decode 步数与 FLOPs。

实验对象主要是 `AndroidControl_Curated_High_Task_Improved`，backbone 为 `Qwen3-VL-4B-Instruct` 或对应的 `Qwen3-VL-4B-Instruct-AttnPrune` 实现。

## 一、方法简要介绍

### 1. State Packet：历史帧输入级压缩

`State Packet` 针对多步 GUI 任务里昂贵的历史截图输入做压缩。原始做法是把最多 4 张历史截图和当前截图一起送入 VLM，导致 image encode 与 LLM prefill 都很重。现在的做法是保留当前截图的完整视觉信息，同时把历史截图改写成紧凑视觉包：

- 每张历史图保留一张低分辨率 thumbnail，维持全局页面布局和状态记忆；
- 从历史动作、点击区域或关键交互区域裁出 ROI crop，保留局部细节；
- 将历史图从完整截图 token 压缩为 `thumbnail + ROI` 组合；
- 不需要训练，属于 inference-time / plug-in 式历史视觉压缩。

主要收益来自两部分：历史图 image encode token 变少，LLM prefill 序列长度也显著下降。8/12 的完整 40 样本消融中，平均 prompt token 从 `7522.8` 降到 `2997.1`，历史 packet 估算 token 从原始 `183376` 压到 `4317`。

### 2. Structured Fast Decode：结构化约束与模板化解码




`Structured Fast Decode` 针对 AndroidControl 输出格式固定这一特点，将普通自由生成改成结构化 slot 生成。核心输出从自由文本变为 action-first JSON：

```text
<answer>{"action_type": "...", "bbox_2d": [...]}</answer>
```

实现上，固定 JSON 模板由程序提供，模型主要填动作类型和坐标槽位。它的作用有两层：

- 速度层面：减少模型逐 token 生成固定 JSON 结构的串行 decode 步数；
- 精度层面：把生成空间限制在合法 action / bbox schema 内，减少格式漂移、动作串跑偏和不必要 bbox 输出。

需要在汇报里说明：这不是纯无损加速，它同时改变了解码分布，相当于引入了 AndroidControl 输出协议先验。因此 Type、Grounding、SR 的提升不应只归因于速度优化。

### 3. 当前帧 Attention Prune：指令引导的当前截图 token 剪枝

当前帧 prune 针对仍然完整输入的 current screenshot。方法在 LLM 早期层使用“任务指令 token 到当前图 visual token”的 attention 作为重要性信号，只保留高注意力的当前帧视觉 token，并加入安全保留策略：

- 在指定 decoder layer 后触发，例如 `layer=1/3/5`；
- 通过 `QWEN3VL_ATTN_PRUNE_KEEP_RATIO` 控制保留比例；
- query 侧使用当前 step instruction 的 token；
- visual 侧只对当前截图 token 做剪枝，不剪历史 State Packet；
- safety keep 额外保留顶部、底部、边缘、中心和文本密集区域，降低 UI 关键元素被误删的风险；
- 支持 side attention 额外计算或 eager attention 抓取。

当前 sweep 结果显示：当前帧 prune 可以进一步压低一部分 LLM FLOPs，但对 Grounding/SR 比较敏感，尤其在高压缩比例下会伤定位精度。它更适合作为 accuracy-cost tradeoff 模块，而不是默认无损加速。

## 二、实验对比

### 2.1 40 样本主消融：Baseline -> State Packet -> State Packet + Fast Decode

数据来源：`OUTPUT_0/ablation_android_control_*_0812(only_HTI)_ratio2/.../summary.json` 与 `compare_table_structure_decode_android.md`。

| Method                   | Type (%) | Grounding (%) | SR (%) | Total (s) | Encode (s) | Prefill (s) | Decode (s) | Decode steps | Prompt tokens | Vision TFLOPs | LLM TFLOPs | LM-head TFLOPs | E2E TFLOPs |
| ------------------------ | -------: | ------------: | -----: | --------: | ---------: | ----------: | ---------: | -----------: | ------------: | ------------: | ---------: | -------------: | ---------: |
| Baseline                 |     77.5 |          63.0 |   52.5 |     5.777 |      0.530 |       0.833 |      3.316 |       38.225 |        7522.8 |        120.96 |      96.88 |           5.88 |     223.71 |
| + State Packet           |     75.0 |          77.8 |   60.0 |     4.769 |      0.256 |       0.354 |      3.554 |       38.450 |        2997.1 |         17.55 |      27.43 |           2.36 |      47.33 |
| + Structured Fast Decode |     80.0 |          85.2 |   72.5 |     3.058 |      0.274 |       0.392 |      1.932 |       19.325 |        2997.1 |         17.55 |      27.37 |           2.36 |      47.28 |

主要观察：

- `State Packet` 把 prompt tokens 降到 `39.8%`，E2E FLOPs 约降到 baseline 的 `21.1%`，即约 `4.73x` FLOPs 降低。
- `State Packet` 对 encode 和 prefill 最明显：encode `0.530s -> 0.256s`，prefill `0.833s -> 0.354s`。
- `Structured Fast Decode` 主要减少 decode：decode steps `38.45 -> 19.33`，decode time `3.554s -> 1.932s`。
- 精度上，最终 `SR` 从 `52.5%` 提升到 `72.5%`；这部分包含结构化约束带来的输出合法性和 action-first 先验收益。

### 2.2 最新 23 样本 quick run：Baseline vs State Packet + Fast Decode

数据来源：`OUTPUT/android_control_hist4_keep_system_prompt_*_0828/.../summary.json` 与对应 `*_android_control_detail_official.json`。该组用于和当前帧 prune sweep 对齐，样本数为 23。

| Method                       | Type (%) | Grounding (%) | SR (%) | Total (s) | Encode (s) | Prefill (s) | Decode (s) | Decode steps | Prompt tokens | Vision TFLOPs | LLM TFLOPs | LM-head TFLOPs | E2E TFLOPs |
| ---------------------------- | -------: | ------------: | -----: | --------: | ---------: | ----------: | ---------: | -----------: | ------------: | ------------: | ---------: | -------------: | ---------: |
| Baseline                     |     87.0 |          76.5 |   73.9 |     3.022 |      0.403 |       0.566 |      1.563 |       39.130 |        6587.7 |         92.37 |      80.29 |           5.15 |     177.82 |
| + State Packet + Fast Decode |     87.0 |          88.2 |   82.6 |     1.879 |      0.212 |       0.275 |      1.028 |       25.609 |        3027.9 |         17.28 |      31.44 |           2.70 |      51.42 |

主要观察：

- 在相同 23 个样本上，`State Packet + Fast Decode` 将 total wall time 从 `3.022s` 降到 `1.879s`，约 `1.61x` 加速。
- E2E FLOPs 从 `177.82` TFLOPs 降到 `51.42` TFLOPs，约 `3.46x` 降低。
- Grounding 从 `76.5%` 到 `88.2%`，SR 从 `73.9%` 到 `82.6%`。

### 2.3 当前帧 Attention Prune sweep

以下结果均在 `State Packet + Structured Fast Decode` 基础上增加当前帧 attention prune。样本数为 23。需要特别标注：0830 这批是修正后的实现，旧 sweep 里误把视觉/序列剪枝口径处理错的问题已经调整回来；新日志中当前帧视觉 token 为 `2550`，剪枝后只改变当前帧 token 数，历史 `State Packet` 不参与剪枝，且这批关闭了 safety keep 额外保留开销。

数据来源：`OUTPUT/android_control_hist4_keep_system_prompt_state_packet_attn_prune_structured_fast_official_prune_*_0830_high/.../summary.json`、`OUTPUT/android_control_hist4_keep_system_prompt_state_packet_attn_conf_observe_layers0123_0830_high/.../summary.json` 及对应 0830 run log。

| Current-frame prune setting | Type (%) | Grounding (%) | SR (%) | Total (s) | Encode (s) | Prefill (s) | Decode (s) | Decode steps | Prompt tokens | Current visual tokens after | LLM TFLOPs | E2E TFLOPs |
| --------------------------- | -------: | ------------: | -----: | --------: | ---------: | ----------: | ---------: | -----------: | ------------: | --------------------------: | ---------: | ---------: |
| No current prune / observe  |     87.0 |          88.2 |   82.6 |     1.888 |      0.217 |       0.302 |      1.017 |       25.609 |        3027.9 |                        2550 |      31.44 |      51.42 |
| layer1, keep 0.55           |     91.3 |          82.4 |   78.3 |     1.726 |      0.216 |       0.188 |      0.971 |       24.043 |        3027.9 |                        1403 |      30.30 |      50.18 |
| layer1, keep 0.60           |     91.3 |          88.2 |   82.6 |     1.831 |      0.225 |       0.202 |      1.021 |       25.652 |        3027.9 |                        1530 |      31.58 |      51.57 |
| layer3, keep 0.40           |     87.0 |          88.2 |   78.3 |     1.488 |      0.189 |       0.159 |      0.802 |       20.217 |        3027.9 |                        1020 |      27.69 |      47.34 |
| layer3, keep 0.50           |     87.0 |          82.4 |   73.9 |     1.654 |      0.215 |       0.182 |      0.942 |       23.739 |        3027.9 |                        1275 |      30.15 |      50.02 |
| layer3, keep 0.55           |     91.3 |          88.2 |   82.6 |     1.695 |      0.217 |       0.184 |      0.943 |       24.000 |        3027.9 |                        1403 |      30.30 |      50.18 |
| layer3, keep 0.60           |     91.3 |          94.1 |   87.0 |     1.874 |      0.217 |       0.208 |      1.106 |       25.609 |        3027.9 |                        1530 |      31.58 |      51.57 |
| layer3, keep 0.80           |     87.0 |          88.2 |   82.6 |     2.064 |      0.232 |       0.261 |      1.071 |       26.826 |        3027.9 |                        2040 |      32.75 |      52.84 |
| layer3, keep 0.95           |     87.0 |          88.2 |   82.6 |    23.354 |      1.119 |       2.062 |     19.214 |       26.783 |        3027.9 |                        2423 |      32.76 |      52.85 |

主要观察：

- 修正实现后，当前帧 prune 的 FLOPs 下降幅度很有限：no-prune 为 `51.42` TFLOPs，最激进的 `layer3, keep 0.40` 为 `47.34` TFLOPs，只降低约 `7.9%`。这是因为当前帧只占压缩后序列的一部分，且视觉 encode FLOPs 仍然不变。
- wall time 的最好点是 `layer3, keep 0.40`，从 no-prune/observe 的 `1.888s` 降到 `1.488s`，约 `1.27x` 加速，但 SR 从 `82.6%` 降到 `78.3%`。
- 兼顾精度的点是 `layer3, keep 0.60`：Type/Grounding/SR 为 `91.3/94.1/87.0`，但 total `1.874s` 与 no-prune `1.888s` 基本持平，实际加速只有约 `1.01x`。
- `layer1, keep 0.60` 和 `layer3, keep 0.55/0.80` 能维持 no-prune 的 `82.6%` SR，但 E2E FLOPs 与 wall time 只小幅变化，不能作为强加速证据。
- `layer3, keep 0.95` 出现异常慢的 wall time（`23.354s`），但 decode steps 并未相应增加，说明更可能是运行时抖动或异常样本拖慢；不应把它作为方法趋势解读。
- 因此当前更稳妥的表述是：修正后的当前帧 attention prune 可以验证“当前截图 token 还能被裁掉一部分”，但由于 current-frame prune 发生在 LLM 内部、不能减少 vision encode，且剩余 decode 开销仍占主导，整体端到端加速有限。它目前更适合作为探索性 Pareto 附表，而不是主加速路径。

## 三、汇报结论

当前 AndroidControl 上最清晰的创新链路是：

1. `State Packet` 解决历史帧冗余，显著降低 visual token、encode、prefill 和 FLOPs；
2. `Structured Fast Decode` 利用固定动作 schema 减少 decode 步数，同时提升输出合法性；
3. `Current-frame Attention Prune` 尝试继续压缩当前截图 token；0830 修正实现后，精度可以在部分配置下维持甚至小幅波动上升，但端到端加速仍然有限。

从当前结果看，最适合放主表的是：

- `Baseline`
- `+ State Packet`
- `+ State Packet + Structured Fast Decode`

当前帧 prune 建议作为 ablation / Pareto sweep 放在附表，重点说明它是后续可优化方向：修正后的结果不再支持“显著加速”的强结论，只能说明当前帧 token 有一定冗余；若后续要转化为主收益，需要把剪枝前移到 vision encode 或 image token 构造阶段，或者结合更稳定的 layout-aware / action-conditioned saliency 与低置信度 fallback。

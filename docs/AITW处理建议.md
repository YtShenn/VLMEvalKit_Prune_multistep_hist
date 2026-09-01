
**那既然这样，我能不能复用[https://github.com/Daiqiang-Li/HistPrune-GUI](https://github.com/Daiqiang-Li/HistPrune-GUI)这里的AITW处理逻辑，包括数据处理、prompt构造、输出格式、测评逻辑等等**

可以，而且我觉得这是**合理的起点**。因为 AITW 官方没有规定 prompt / 输出格式 / 推理协议，复用 HistPrune-GUI 的 AITW 处逻辑能减少很多“协议自造”的不确定性。

但我建议你**复用协议，不直接照搬整套脚本结构**。

可以复用的部分：

- AITW 预处理思路：原始 AITW -> `images/*.png` + `aitw_data_test.json`
- episode/step 顺序组织方式
- 历史缓存方式：每个 episode 内维护 `previous_actions`、`previous_imgs`
- 默认历史长度：`his_num=4`
- prompt 组织：

```text
Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {goal}.
Image_0:<image>
Step_0: {previous_action}
...
Image_k:<image>
```

- 输出格式，例如：

```json
{"action_type": 4, "click_point": (x,y)}
```

```json
{"action_type": 3, "typed_text": "xxx"}
```

- action type 映射和 `action_matching` 测评逻辑
- no-history 设置：只去掉历史截图，但保留历史动作文本

需要谨慎处理的部分：

- 它的路径是硬编码的，要改成你当前项目里的 dataset root/env var。
- 它的 `AITW_eval.py` 是一个大脚本，数据、推理、剪枝、profile、eval 混在一起，不适合直接塞进你的项目。
- 它用的是预处理后的 JSON，不是原始 TFRecord，所以你要确认 `aitw_data_test.json` 的来源和 split 是否与你论文想声明的一致。
- 它的 prompt/output 格式如果用于你方法，也应该**同样用于 baseline**，否则对比不公平。
- 它的模型输出解析比较脆弱：`ast.literal_eval(response)`，你最好在当前项目里做更稳的解析和 wrong-format 记录。
- 它默认历史帧顺序是旧到新，最后一张是当前帧；迁移时要保持这一点。

我会建议论文实现上这样写：

1. **在当前 VLMEvalKit 里新增 AITW dataset adapter**
2. adapter 内复用 HistPrune-GUI 的：
   - 数据 JSON schema
   - prompt template
   - `action2step`
   - 输出/action format
   - action matching eval
3. 模型推理继续走你当前的 `Qwen3VLChat.generate_inner`
4. 你的方法和 FastV/SparseVLM baseline 都在这个 adapter 下统一跑

这样你可以在论文里比较自然地说：**We follow the AITW preprocessing, prompt format, and action-matching protocol used by HistPrune-GUI, and evaluate all methods in our unified inference framework.**

一句话：**能复用，而且应该复用；但最好复用它的 AITW 协议和评测逻辑，别把整个 `AITW_eval.py` 当成你的主实验框架。**

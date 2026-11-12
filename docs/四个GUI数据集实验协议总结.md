# 四个 GUI 数据集实验协议总结

本文总结当前四数据集联合实验的实际调用口径。联合脚本默认调用：

```text
AITW_all
Mind2Web_test_task
GUIOdyssey_high_task_split
AndroidControl_Curated_High_Task_Improved
```

对应入口见 `RUN_BASH/run_qwen3vl_sparsevlm_4datasets_4B.sh` 和 `RUN_BASH/run_qwen3vl_divprune_4B.sh`。

## 共同实验设置

- 默认保留最近 **4 步**历史截图：`AITW_HIS_NUM=4`、`MIND2WEB_HIS_NUM=4`、`GUI_ODYSSEY_MAX_HISTORY_IMAGES=4`、`ANDROID_CONTROL_MAX_HISTORY_IMAGES=4`。
- 默认历史动作来自标注（GT / teacher forcing），不是模型此前预测；只有显式设置各数据集的 `*_HISTORY_ACTION_SOURCE=prediction` 时才使用模型预测历史。
- 当前截图始终保留。
- SparseVLM 联合脚本默认 `VLM_EVAL_SAMPLE_MODE=off`，即不做抽样、评测全量数据。其他启动脚本可通过该环境变量改变抽样策略，比较结果时需确认设置一致。

## 数据集对照表

| 数据集 | 当前调用的划分 | 数据与输入协议 | 输出和动作匹配 | 主报告指标 |
| --- | --- | --- | --- | --- |
| **AndroidControl-Curated** | `AndroidControl_Curated_High_Task_Improved` | 读取 `benchmark_resource/android_control_high_task-improved.json`。输入为当前截图和最多 4 步历史截图/历史动作。 | 输出动作类型和坐标。模型输出的 0--1000 坐标会反归一化到当前原图像素空间。该划分会将主 GT 动作和 `candidate_actions` 都视为可接受答案，任一候选满足即可。 | 官方口径：`Type (%)`、`Grounding (%)`、`SR (%)`。`Grounding` 仅以 click/long-press 样本为分母；`SR` 是单步类型与位置同时正确的比例。另输出 task-level `Task_Success_Rate` 供参考。 |
| **AITW** | `AITW_all`，`AITW_SPLIT=test` | 读取 `aitw_data_test.json`，覆盖 `general`、`single`、`webshopping`、`install`、`googleapps` 五个类别；每条轨迹按 step 展开。输入为当前截图和最多 4 步历史。 | 输出 JSON 动作。Tap：预测点命中候选框（带归一化边界松弛）或与 GT 点距离不超过 0.14；文本输入：token F1 不低于 0.5；scroll/drag：类型及方向一致。 | 每类报 `Action Acc`、`Step SR`、`Type Acc`、`Text/Click/Scroll Acc` 等。`AITW_all` 额外报告五个类别的宏平均 `Avg Action Acc` 和 `Avg Step SR`。**这不是 AP / mAP，而是按类别平均的动作准确率和轨迹成功率。** |
| **Mind2Web** | `Mind2Web_test_task` | 读取 `mind2web_data_test_task.json`；每条 web trajectory 按 action step 展开。输入为当前截图和最多 4 步历史。 | 要求只输出 Python dict；坐标是当前图的 0--1000 归一化坐标。操作类别需匹配；Element 正确要求预测点击点位于 GT bbox 内；对 TYPE / SELECT 还将值并入操作 F1。 | `Operation_F1`（操作类别宏平均）、`Element_Accuracy`、`Operation_Accuracy`、`Step_Success_Rate`、`Episode_Success_Rate`，以及 Macro 版本的 element / operation / step 指标。 |
| **GUI-Odyssey** | `GUIOdyssey_high_task_split` | 读取 `test_anno/high_task_split.json`，当前截图加最多 4 步历史。 | 要求单条动作命令，如 `CLICK: (x, y)`。动作类型必须一致；CLICK/LONG_PRESS 命中 SAM2 bbox 或与 GT 归一化坐标距离不超过 0.14；TYPE 使用 Levenshtein 相似度不低于 0.5；SCROLL 方向必须一致。 | 该划分配置为 `macro`：`AMS` 是 step 级动作匹配率，`SR` 是一条完整 episode 每一步均正确的成功率；另附 action-type 与 text 的统计。 |

## AndroidControl 的 Hard 口径

当前联合实验中的 `AndroidControl_Curated_High_Task_Improved` **不是**完整的官方 `AndroidControl-Curated-Hard`。

官方 Hard 需要同时评测：

1. `AndroidControl_Curated_High_Point`
2. `AndroidControl_Curated_High_Task_Improved`

然后使用两子集的计数做加权合并，而不是直接平均百分比：

- `Type (%) = (两个子集 Type_Correct 之和) / (Effective_Total 之和)`
- `Grounding (%) = (两个子集 Grounding_Correct 之和) / (BBox_Total 之和)`
- `SR (%) = (两个子集 SR_Correct 之和) / (Effective_Total 之和)`

运行时应设置：

```bash
export ANDROID_CONTROL_CURATED_EVAL_MODE=official
export QWEN3VL_ANDROID_DENORM_ON_INFER=1
export QWEN3VL_ANDROID_DENORM_BASE=1000
```

完成两个子集的推理后，可用 `RUN_BASH/run_android_control_official_eval_from_xlsx.sh` 生成每个子集的官方结果及 Hard 汇总。

## 实现入口

- 四数据集联合启动：`RUN_BASH/run_qwen3vl_sparsevlm_4datasets_4B.sh`
- AITW：`vlmeval/dataset/AITW/aitw.py`、`vlmeval/dataset/AITW/action_matching.py`
- Mind2Web：`vlmeval/dataset/Mind2Web/mind2web.py`
- GUI-Odyssey：`vlmeval/dataset/GUI_Odyssey/gui_odyssey.py`
- AndroidControl：`vlmeval/dataset/AndroidControl_Curated/android_control_curated.py`、`vlmeval/dataset/AndroidControl_Curated/official_eval.py`

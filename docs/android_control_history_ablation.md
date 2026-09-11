# AndroidControl 历史信息消融

`ANDROID_CONTROL_HISTORY_ABLATION_MODE` 默认是 `legacy`，不设置时完全维持原有行为。
下列模式只影响 AndroidControl Curated 的历史输入；当前截图、任务指令、输出格式和评测流程均不变。

| 模式 | 历史动作文本 | 历史视觉输入 | 自动行为 |
| --- | --- | --- | --- |
| `none` | 不保留（prompt 中移除整个 `Past_Actions` 字段） | 不保留 | 忽略旧的 history/image 开关 |
| `text` | 保留 | 不保留 | 忽略历史截图开关 |
| `thumbnail` | 保留 | 每个历史步骤仅全局缩略图 | 自动启用 state packet |
| `roi` | 保留 | 每个历史步骤仅动作 ROI | 自动启用 state packet |
| `legacy`（默认） | 维持原逻辑 | 维持原逻辑 | 兼容已有脚本 |

缩略图和 ROI 模式中，历史动作文本被保留以维持“截图—已执行动作”配对；这与现有 state-packet 协议一致。ROI 是以该历史步骤已执行的动作及其标注坐标构造的，而不是当前待预测步骤的标注。

所有视觉模式仍需要设置要保留的历史步数，例如四步：

```bash
export ANDROID_CONTROL_MAX_HISTORY_IMAGES=4
```

建议分别运行：

```bash
# 无历史信息
export ANDROID_CONTROL_HISTORY_ABLATION_MODE=none

# 仅历史动作文本
export ANDROID_CONTROL_HISTORY_ABLATION_MODE=text

# 历史动作文本 + 每步一张缩略图
export ANDROID_CONTROL_HISTORY_ABLATION_MODE=thumbnail

# 历史动作文本 + 每步一个动作 ROI
export ANDROID_CONTROL_HISTORY_ABLATION_MODE=roi
```

`thumbnail` / `roi` 模式会覆盖 `ANDROID_CONTROL_STATE_PACKET_ENABLE`，确保视觉包被启用；并分别覆盖为单图输出。若只想在原有 history 协议下单独控制 state packet，也可以使用：

```bash
export ANDROID_CONTROL_STATE_PACKET_ENABLE=1
export ANDROID_CONTROL_STATE_PACKET_IMAGE_MODE=thumbnail  # thumbnail | roi | both（默认）
```

打开 `ANDROID_CONTROL_DEBUG_HISTORY_PROMPT=1` 后，日志会打印 `history_ablation_mode`。打开 `ANDROID_CONTROL_STATE_PACKET_DEBUG=1` 后，packet 元数据中的 `packet_image_mode` 可确认每步实际输入的视觉类型。

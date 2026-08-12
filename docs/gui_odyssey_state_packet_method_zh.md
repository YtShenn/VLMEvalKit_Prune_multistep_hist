# GUIOdyssey 历史 State Packet 方法说明

本文档说明当前为 `GUIOdyssey` 历史多图输入新增的 `State Packet` 方案实现，包括：

- 方法设计
- 代码文件与函数说明
- 运行脚本
- 调试输出
- `summary.json` 新增字段

## 一、目标

该实现面向 `RUN_BASH/run_gui_odyssey_history_ablation_qwen3_5.sh` 中的 `hist4_keep_system_prompt` 设定做最小侵入式扩展。

目标如下：

- 尽量少改现有代码
- 不启用原有 ROI prune
- 通过开关控制是否启用历史 `State Packet`
- 开关关闭时，尽量保持与原 `hist4_keep_system_prompt` 一致
- 额外输出：
  - 历史图裁剪坐标
  - 历史图裁剪前后尺寸
  - 历史图裁剪前后估算 token 数
  - State Packet 各阶段耗时
  - 现有阶段时间统计
  - FLOPs 汇总

## 二、方法设计

### 1. 原始历史图输入

原始 `hist4_keep_system_prompt` 中，每张历史图会完整作为一张图片输入模型。

### 2. State Packet 输入

开启 `State Packet` 后，每张历史图不再直接整图输入，而会被替换为两张历史图像：

- 一张超小缩略图 `thumbnail`
- 一张动作邻域裁剪图 `action ROI`

即每个历史 step 对应：

- `thumbnail`
- `action_roi`
- 历史动作文本

其中：

- `thumbnail` 用于保留页面整体布局
- `action_roi` 用于保留与该历史动作最相关的局部细节

### 3. 当前支持的动作邻域裁剪规则

- `CLICK / LONG_PRESS`

  - 使用动作坐标作为中心
  - 从原图中裁一个方形邻域
- `SCROLL`

  - 没有明确点击点时，使用基于方向的启发式区域
  - 上下滚动使用中间纵向区域
  - 左右滚动使用中间横向区域
- 其他动作如 `TYPE / PRESS_BACK / COMPLETE`

  - 回退到中心区域裁剪

### 4. token 数说明

当前输出的是**估算 token 数**，不是 processor 真正返回的精确 token 数。

估算方式：

- patch size 默认按 `16`
- merge size 默认按 `2`
- 估算公式近似为：
  `estimated_tokens = (width / 16) * (height / 16) / (merge_size^2)`

这个估算主要用于：

- 调试
- 对比裁剪前后 token 规模
- 汇总到 `summary.json`

### 改进思路

**增加 packet 的“变化信息”而不是只保留静态内容**
如果当前 packet 只是从单帧里抽 thumbnail/ROI，它对“上一帧到这一帧哪里变了”表达不够强。
你可以补两类信息：

* 当前 history frame 相对上一 history frame 的差异区域
* 当前 frame 中与最后 action 相关的局部区域
  也就是把 ROI 选择从“显著区域”升级为“变化区域 + 交互相关区域”。

## 三、代码文件与函数说明

### 1. 新增文件

#### [state_packet.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/GUI_Odyssey/state_packet.py)

作用：

- 提供 GUIOdyssey 历史 `State Packet` 的独立实现
- 封装裁剪、缩略图、尺寸与 token 估算、临时图片保存、调试输出

主要函数：

- `state_packet_enabled()`

  - 读取环境变量，判断是否启用 State Packet
- `state_packet_debug_enabled()`

  - 读取环境变量，判断是否输出详细调试日志
- `build_state_packet(...)`

  - 对单张历史图生成：
    - `thumbnail`
    - `action_roi`
  - 返回：
    - 两个 `PacketImage`
    - 一份统计元信息

主要数据结构：

- `PacketImage`
  - 表示一张参与后续 encode / LLM 的历史 packet 图像
  - 包含：
    - 图片路径
    - 宽高
    - 估算 token 数
    - 裁剪框坐标

### 2. 修改文件

#### [gui_odyssey.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/dataset/GUI_Odyssey/gui_odyssey.py)

新增能力：

- 在 `GUIOdyssey` 数据集类中加入 `State Packet` 开关式接入
- 开关关闭时尽量保持原有 prompt 组织方式
- 开关开启时，把每个历史 step 展开为：
  - `thumbnail`
  - `action_roi`
  - 历史动作文本

新增或扩展的关键部分：

- `self.use_history_state_packet`

  - 是否启用 State Packet
- `self._state_packet_records`

  - 用于积累每张历史图的 packet 统计，供 summary 汇总
- `_build_history_visual_entries(...)`

  - 构建历史 step 的图像输入列表
  - 开关关闭时返回原始历史图
  - 开关开启时调用 `build_state_packet(...)`
- `summarize_state_packet_records()`

  - 对所有历史 packet 记录做聚合统计
  - 最终写入 `summary.json`

#### [inference.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/inference.py)

新增能力：

- 在原有 `summary.json` 基础上追加 `State Packet` 汇总信息
- 保留现有：
  - wall time
  - stage timing
  - generate timing
  - prune timing

#### [patch.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/vlm/qwen3_vl_flops/patch.py)

新增能力：

- 在每次样本生成完成后，把该样本的 FLOPs 汇总缓存到模型对象上

#### [model.py](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/vlmeval/vlm/qwen3_vl/model.py)

新增能力：

- 把样本级 FLOPs 信息并入 `_vlmeval_generate_timing_records`
- 支持 `summary.json` 汇总：
  - `vision_flops`
  - `llm_flops`
  - `lm_head_flops`
  - `e2e_flops`

## 四、运行脚本

新增脚本：

#### [run_gui_odyssey_history_state_packet_qwen3_5.sh](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep_hist/RUN_BASH/run_gui_odyssey_history_state_packet_qwen3_5.sh)

特点：

- 基于原 `hist4_keep_system_prompt` 配置
- 默认：
  - 开启 history
  - `max_history_images=4`
  - `keep_system_prompt=1`
  - `GUI_ODYSSEY_STATE_PACKET_ENABLE=1`
  - `QWEN3VL_ENABLE_ROI_PRUNE=0`
  - `QWEN3VL_PROFILE_FLOPS=1`

如果希望回退到原 `hist4_keep_system_prompt` 行为，可在运行前设：

```bash
export GUI_ODYSSEY_STATE_PACKET_ENABLE=0
```

此时会尽量保持原始 prompt 行为。

## 五、环境变量开关

### 1. 核心开关

- `GUI_ODYSSEY_STATE_PACKET_ENABLE`

  - `1`：启用 State Packet
  - `0`：关闭，回退到原历史图输入
- `GUI_ODYSSEY_STATE_PACKET_DEBUG`

  - `1`：输出详细 State Packet 调试信息
  - `0`：关闭调试日志

### 2. State Packet 参数

- `GUI_ODYSSEY_STATE_PACKET_CACHE_DIR`

  - 临时 packet 图像缓存目录
- `GUI_ODYSSEY_STATE_PACKET_PATCH_SIZE`

  - token 估算使用的 patch size
- `GUI_ODYSSEY_STATE_PACKET_MERGE_SIZE`

  - token 估算使用的 merge size
- `GUI_ODYSSEY_STATE_PACKET_THUMB_LONG_EDGE`

  - thumbnail 的长边大小
- `GUI_ODYSSEY_STATE_PACKET_ROI_LONG_EDGE`

  - ROI 图像缩放后的长边大小
- `GUI_ODYSSEY_STATE_PACKET_ROI_SHORT_SIDE_RATIO`

  - 基于原图短边决定 ROI 大小的比例
- `GUI_ODYSSEY_STATE_PACKET_ROI_MIN_SIDE_PX`

  - ROI 最小边长像素

## 六、调试输出说明

当 `GUI_ODYSSEY_STATE_PACKET_DEBUG=1` 时，会输出如下调试信息：

### 1. 单张历史图 packet 生成日志

前缀：

- `[GUIOdysseyStatePacket]`

包含：

- 样本 index
- 历史图 index
- 原图尺寸
- 原图估算 token 数
- thumbnail 尺寸与估算 token 数
- ROI 裁剪坐标 `left, top, right, bottom`
- ROI 图尺寸与估算 token 数
- packet 总估算 token 数
- 各阶段耗时：
  - 打开图像
  - 生成 thumbnail
  - 生成 ROI
  - 总耗时

### 2. prompt 组装阶段日志

前缀：

- `[GUIOdysseyStatePacketPrompt]`

包含：

- 样本 index
- 历史图 index
- 原图与 packet token 数估算
- thumbnail / ROI 尺寸
- ROI 坐标
- packet 生成耗时

说明：

- 即使 `GUI_ODYSSEY_STATE_PACKET_DEBUG=0`，估算 token 数与各阶段耗时仍然会正常计算，并参与最终 `summary.json` 聚合
- `裁剪框坐标`、`裁剪前后尺寸` 只在 `GUI_ODYSSEY_STATE_PACKET_DEBUG=1` 时逐样本打印
- `DEBUG` 关闭时，不再依赖这些坐标/尺寸做常规 summary 聚合

## 七、summary.json 新增字段

原有 summary 字段保留不变，包括：

- 原先的 wall time 统计
- stage timing 统计
- generate timing 统计
- prune timing 统计

新增 `State Packet` 相关字段：

- `state_packet_enabled`
- `state_packet_history_image_count`
- `avg_state_packet_original_estimated_tokens`
- `avg_state_packet_packet_estimated_tokens`
- `avg_state_packet_thumbnail_estimated_tokens`
- `avg_state_packet_roi_estimated_tokens`
- `state_packet_total_original_estimated_tokens`
- `state_packet_total_packet_estimated_tokens`
- `state_packet_avg_compression_ratio`
- `avg_state_packet_open_image_s`
- `avg_state_packet_thumbnail_build_s`
- `avg_state_packet_roi_build_s`
- `avg_state_packet_total_s`
- `total_state_packet_open_image_s`
- `total_state_packet_thumbnail_build_s`
- `total_state_packet_roi_build_s`
- `total_state_packet_total_s`

新增 FLOPs 相关字段：

- `flops_profiled_samples`
- `avg_vision_flops`
- `avg_llm_flops`
- `avg_lm_head_flops`
- `avg_e2e_flops`
- `total_vision_flops`
- `total_llm_flops`
- `total_lm_head_flops`
- `total_e2e_flops`

## 八、设计原则

这次实现遵循以下原则：

- 尽量少改现有文件
- 新功能优先写成独立模块
- 原有 ROI prune 不启用
- 通过环境变量开关控制行为
- 开关关闭时尽量回退到原始 `hist4_keep_system_prompt`
- 保留原有阶段时间统计，并把新增 State Packet 与 FLOPs 统计合并到同一个 `summary.json`

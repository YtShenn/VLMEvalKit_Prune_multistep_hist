
要拿到官方 `AndroidControl-Curated-Hard` 结果，你需要跑这两个子集，然后再做一次官方口径汇总：

- `AndroidControl_Curated_High_Point`
- `AndroidControl_Curated_High_Task_Improved`

也就是说，`Hard` 不是单独一个数据集，而是这两个结果合并出来的。

最直接的跑法是自己调用 `run.py`，只跑这两个数据集，并打开官方评测开关：

```bash
export ANDROID_CONTROL_CURATED_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated"
export ANDROID_CONTROL_CURATED_IMAGE_ROOT="/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images"

export ANDROID_CONTROL_CURATED_EVAL_MODE=official

export QWEN3VL_ANDROID_DENORM_ON_INFER=1
export QWEN3VL_ANDROID_DENORM_BASE=1000

python -u run.py \
  --data \
    AndroidControl_Curated_High_Point \
    AndroidControl_Curated_High_Task_Improved \
  --model Qwen3-VL-4B-Instruct \
  --work-dir OUTPUT/outputs_android_control_hard_official \
  --mode all \
  --reuse
```

如果你想沿用现有脚本思路，也可以参考 [run_qwen3vl_android_control_curated.sh](/home/ytshen/storage_net2/VLMEvalKit_Prune_multistep/RUN_BASH/run_qwen3vl_android_control_curated.sh:46)，只是把数据集收缩到上面两个。

跑完以后，再把这两个 `xlsx` 做官方 Hard 汇总。用我新加的脚本最方便：

```bash
bash RUN_BASH/run_android_control_official_eval_from_xlsx.sh \
  OUTPUT/outputs_android_control_hard_official/AndroidControl_Curated_High_Point/Qwen3-VL-4B-Instruct/*/Qwen3-VL-4B-Instruct_AndroidControl_Curated_High_Point.xlsx \
  OUTPUT/outputs_android_control_hard_official/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/*/Qwen3-VL-4B-Instruct_AndroidControl_Curated_High_Task_Improved.xlsx
```

最后看生成的：

- `*_official_metrics.json`：每个子集自己的官方口径结果
- `android_control_curated_official_summary.json`：里面会有 `AndroidControl-Curated-Hard` 的合并结果，字段就是官方表里的 `Type (%) / Grounding (%) / SR (%)`

如果你愿意，我可以下一步直接给你新建一个专门的 `RUN_BASH/run_qwen3vl_android_control_curated_hard_official.sh`，把这两步串起来一键跑。

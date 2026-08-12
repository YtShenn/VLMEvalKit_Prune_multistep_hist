| field                                        |     baseline | state_packet | state_packet_structured_decode | speedup(state_packet/baseline) | speedup(state_packet_structured_decode/baseline) | speedup(state_packet_structured_decode/state_packet) |
| -------------------------------------------- | -----------: | -----------: | -----------------------------: | -----------------------------: | -----------------------------------------------: | ---------------------------------------------------: |
| Type%                                        |         77.5 |         75.0 |                           80.0 |                           -2.5 |                                             +2.5 |                                                   +5 |
| Grounding%                                   |         63.0 |         77.8 |                           85.2 |                          +14.8 |                                            +22.2 |                                                 +7.4 |
| SR%                                          |         52.5 |         60.0 |                           72.5 |                           +7.5 |                                              +20 |                                                +12.5 |
| avg_total_wall_s                             |     5.777268 |     4.768839 |                       3.057781 |                       1.211462 |                                         1.889366 |                                             1.559575 |
| avg_infer_wall_s                             |     5.588128 |     4.617162 |                       2.901329 |                       1.210295 |                                         1.926058 |                                             1.591396 |
| avg_llm_stage_s                              |     4.348655 |     4.120358 |                       2.391354 |                       1.055407 |                                         1.818491 |                                             1.723023 |
| avg_encode_s                                 |     0.529653 |     0.256403 |                       0.274362 |                       2.065704 |                                         1.930489 |                                             0.934543 |
| avg_prefill_s                                |     0.833044 |     0.354149 |                        0.39178 |                       2.352241 |                                         2.126305 |                                             0.903948 |
| avg_decode_s                                 |     3.316361 |     3.554439 |                       1.932164 |                        0.93302 |                                         1.716397 |                                             1.839615 |
| avg_decode_tokens                            |       38.225 |        38.45 |                           16.6 |                       0.994148 |                                         2.302711 |                                             2.316265 |
| avg_decode_steps                             |       38.225 |        38.45 |                         19.325 |                       0.994148 |                                         1.978008 |                                             1.989651 |
| avg_prompt_seq_tokens                        |       7522.8 |      2997.05 |                        2997.05 |                       2.510068 |                                         2.510068 |                                                    1 |
| avg_vision_flops                             | 1.209576e+14 | 1.754504e+13 |                   1.754504e+13 |                       6.894118 |                                         6.894118 |                                                    1 |
| avg_llm_flops                                | 9.687525e+13 | 2.742712e+13 |                   2.737485e+13 |                       3.532097 |                                         3.538842 |                                              1.00191 |
| avg_lm_head_flops                            | 5.881037e+12 | 2.360575e+12 |                   2.356082e+12 |                       2.491358 |                                         2.496108 |                                             1.001907 |
| avg_e2e_flops                                | 2.237139e+14 | 4.733274e+13 |                   4.727597e+13 |                       4.726409 |                                         4.732084 |                                             1.001201 |
| state_packet_total_packet_estimated_tokens   |          N/A |         4317 |                           4317 |                            N/A |                                              N/A |                                                    1 |
| state_packet_total_original_estimated_tokens |          N/A |       183376 |                         183376 |                            N/A |                                              N/A |                                                    1 |




python utils/compare_summary_json.py
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_baseline_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json"
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_state_packet_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json"
  "OUTPUT/ablation_android_control_hist4_keep_system_prompt_state_packet_structured_fast_official_xiabanqian_node5_0812(only_HTI)_ratio2/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260812_G1e13089b/summary.json"
  --labels baseline state_packet state_packet_structured_decode
  --output compare_table_structure_decode.md

8月12日结果

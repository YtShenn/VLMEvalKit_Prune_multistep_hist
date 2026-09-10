
那就现在的这些所有的复现的方法，能否梳理一下各个flops得到这个最后具体的数，由来是什么、估算过程是什么样的



可以。现在的最终数值可概括为：每个样本先按实际运行 shape 算一次，再对样本取平均。

$F_{\text{sample}}=F_{\text{vision}}+F_{\text{LLM-prefill}}+F_{\text{LLM-decode}}+F_{\text{lm-head}}$

`summary.json` 中：

$\texttt{avg\_e2e\_flops}=\frac{1}{N}\sum_i F_{\text{sample},i}$

对应的 `avg_vision_flops`、`avg_llm_flops`、`avg_lm_head_flops` 也是各自逐样本平均，三项相加就是 e2e。

## 公共计算公式

设 decoder hidden size 为 \(H\)，MLP intermediate size 为 \(I\)，attention heads 为 \(A\)，KV heads 为 \(A_{kv}\)，head dim 为 \(d\)，decoder 层数为 \(L\)。

对于某个 decoder layer，query 长度为 \(q\)、KV 长度为 \(k\)：

$D(q,k)=2qH(2Ad+2A_{kv}d)+4Aqkd+6qHI$

含义分别是 Q/K/V/O 投影、attention 两次矩阵乘、MLP 三个线性层。代码见 [model.py](/mnt/storage2/users/ytshen_data/VLMEvalKit_Prune_multistep_hist/vlmeval/vlm/qwen3_vl/model.py:487)。

Vision encoder 用本次 processor 实际得到的每张图/每帧 grid。若第 \(j\) 个独立视觉 attention 序列有 \(n_j\) 个 token，则二次项是：

$\sum_j n_j^2$

而不是所有图片 token 合并后的平方。线性投影与 MLP 则按总 token 数 \(\sum_j n_j\) 计算。见 [model.py](/mnt/storage2/users/ytshen_data/VLMEvalKit_Prune_multistep_hist/vlmeval/vlm/qwen3_vl/model.py:520)。

lm_head 按实际进入 output projection 的 tensor token 数 \(u\) 计：

$F_{\text{lm-head}}=2uH|\mathcal V|$

现在通过 hook 读取 `lm_head` 真正收到的 `hidden_states.shape[-2]`；生成通常 prefill 为 1 token、每个 decode step 也是 1 token。

## 各复现方法的差别

记：

- \(S\)：进入 decoder 的原始 prefill prompt 长度；
- \(S'\)：某次实际裁剪后的长度；
- \(K_t\)：第 \(t\) 个 decode step 时真实物理 KV 长度；
- `D(q,k;r)`：上式的 decoder FLOPs，累计 \(r\) 层。

| 方法         | Vision FLOPs 的输入                                 | Prefill LLM FLOPs                                              | Decode LLM FLOPs                                    |
| ------------ | --------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| Baseline     | 实际全部输入图/帧 grid                              | `D(S,S; L)`                                                  | 每步`D(1,K_t; L)`                                 |
| GUI-KV       | 实际全部输入图/帧 grid                              | `D(S,S; L)`                                                  | KV 已压缩，按实际`K_t`                            |
| ST-Lite      | 实际全部输入图/帧 grid                              | `D(S,S; L)`                                                  | KV 已压缩，按实际`K_t`                            |
| GUIPruner    | TAR 后实际送入 processor 的图像 grid                | 前 2 层`D(S,S;2)`；后续 `D(S',S';L-2)`                     | 按 SSP 后实际 KV 长度                               |
| HistPrune    | 实际全部输入图/帧 grid                              | 剪枝层前`D(S,S;r)`；之后 `D(S',S';L-r)`                    | 按实际裁后 KV 长度                                  |
| FastV        | 实际全部输入图/帧 grid                              | 前`K` 层 `D(S,S;K)`；后续 `D(S',S';L-K)`                 | 按实际裁后 KV 长度                                  |
| SparseVLM    | 实际全部输入图/帧 grid                              | 每条`layer_prunes` 后，从下一层起切换到该条记录的真实 `S'` | 按最后实际 KV 长度                                  |
| PruMerge     | 实际全部输入图/帧 grid                              | 合并发生在 layer 0 前：`D(S',S';L)`                          | 按合并后 KV 长度                                    |
| DivPrune     | 实际全部输入图/帧 grid                              | 筛选发生在 layer 0 前：`D(S',S';L)`                          | 按筛选后 KV 长度                                    |
| State Packet | State packet 实际保留下来并送入 processor 的图/文本 | 普通 baseline 公式，但\(S\) 是改写后的实际 prompt              | 按真实运行的 KV                                     |
| Fast Decode  | 实际输入图/帧 grid                                  | hook 到每次真实 language-model forward 的\(q,k\)               | 静态模板 token 和动态 token 的真实 forward 都会累计 |

几个容易混淆的点：

- GUI-KV / ST-Lite 的 prefill 不会因为“prefill 后 KV 被压缩”而减少。因为各层 prefill attention 已经在完整 \(S\) 上执行；省下的是后续 autoregressive decode。
- FastV、HistPrune、SparseVLM、GUIPruner 是 decoder 内部裁序列，所以要按层边界分段算。
- PruMerge、DivPrune 是进入 layer 0 前就物理缩短 hidden sequence，所以所有 decoder 层都使用 \(S'\)。
- GUIPruner 的 TAR 会在视觉编码前改变实际图像尺寸/网格，因此 Vision 项也会变；其他多数 token-pruning 方法是在 Vision 之后裁 token，Vision 项通常不变。
- State Packet 的 `estimated_tokens` 是方法内部诊断字段，不参与最终 FLOPs；最终 FLOPs 只看真正送入模型的输入及实际 forward。

最终汇总写入 summary 的路径是：

1. runtime hooks 记录每次实际 forward 的 `q_len`、cache 长度、剪枝审计信息及 lm_head 输入 shape；
2. 生成结束写出该样本的四项 FLOPs；
3. [inference.py](/mnt/storage2/users/ytshen_data/VLMEvalKit_Prune_multistep_hist/vlmeval/inference.py:883) 对所有有效样本求和、除以 `flops_profiled_samples`。

## 当前统一口径及不计入项

当前口径计入 Vision Transformer 主层、LLM decoder、lm_head；不计 token scoring、Top-k、索引/拼接、KV 重排、图像预处理等辅助操作的理论 FLOPs。这些操作会进入真实 latency，但不进入论文中常见的 backbone theoretical FLOPs。

要保证可比性，所有方法和 baseline 应当使用：

- `QWEN3VL_RUNTIME_TRACKING=1`；
- 相同数据样本、history 设置、最大生成 token、解码策略；
- 不启用旧的 `QWEN3VL_PROFILE_FLOPS=1` profiler 路径来混合统计。

GUIPruner 是唯一额外覆盖公共 tracker 的方法：它把公共 tracker 看到的“外层整段 prefill”替换为真实的两段 decoder 执行；lm_head 则保留 hook 到的真实输入长度。

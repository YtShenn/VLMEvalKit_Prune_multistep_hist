可以。设：

```text
N = prompt + image + history 的初始序列长度
D = baseline 总输出 token 数
S = 固定模板 token 数
U = 动态未知 token 数
D = S + U

L = transformer 层数
H = hidden size
I = MLP intermediate size
A = attention heads
d = head dim
V = vocab size
```

为了简化，只看 decode 阶段，不看最初 prefill。

**Baseline：逐 token decode**

第 `t` 个输出 token，`t=0...D-1`，它 attend 的上下文长度约是：

```text
N + t
```

每个 decode token 的主要 FLOPs 近似：

```text
C_linear = L * O(H^2 + H*I)
C_attn(t) = L * O(A * d * (N + t))
C_lm = O(H * V)
```

所以 baseline decode FLOPs：

```text
F_base
≈ Σ_{t=0}^{D-1} [ C_linear + C_attn(t) + C_lm ]

≈ D * C_linear
 + L * O(A*d * Σ_{t=0}^{D-1}(N+t))
 + D * C_lm

≈ D * C_linear
 + L * O(A*d * (D*N + D(D-1)/2))
 + D * C_lm
```

---

**Structured/template decode**

动态 token 仍然逐 token decode，共 `U` 个。  
固定模板 token 分成 `B` 个连续 bulk block，总 token 数 `S`。

设第 `b` 个模板 block 长度是 `s_b`：

```text
Σ s_b = S
B = 模板 block 数
```

动态 token 的 FLOPs 仍类似逐 token：

```text
F_dyn
≈ U * C_linear
 + L * O(A*d * (U*N + dynamic growth terms))
 + U * C_lm
```

模板 block 的 FLOPs：

每个 block 长度 `s_b`，它不是逐 token loop，而是一次 forward。它仍然要算每个模板 token 的 hidden/KV/attention，所以：

```text
F_tpl_block_b
≈ s_b * C_linear
 + L * O(A*d * (s_b * N_b + s_b(s_b-1)/2))
 + lm_head_cost
```

其中 `N_b` 是这个模板 block 前已有上下文长度，约等于 `N + 前面已生成 token 数`。

如果实现里对模板 block 的所有位置都算 logits，那么：

```text
lm_head_cost ≈ s_b * C_lm
```

如果优化成只算最后一个位置 logits，那么：

```text
lm_head_cost ≈ 1 * C_lm
```

所以 structured 总 FLOPs：

```text
F_struct
≈ F_dyn + Σ_b F_tpl_block_b
```

把大项展开：

```text
F_struct
≈ (U + S) * C_linear
 + L * O(A*d * [所有输出 token 对上下文的 attention 总量])
 + U * C_lm
 + B_or_S_lm_head
```

其中：

```text
U + S = D
```

所以前两项大体仍然是：

```text
D * C_linear
+ L * O(A*d * (D*N + D(D-1)/2))
```

也就是和 baseline 很接近。

真正差别主要在 lm_head 和调度开销：

```text
F_base - F_struct
≈ 如果模板 block 只算最后 logits:
   (S - B) * C_lm

否则:
   FLOPs 近似不省，主要省 wall time / kernel launch / 串行依赖
```

---

**如果只看理论 FLOPs，可得结论**

如果你的实现对模板 block 所有 token 都算 logits：

```text
F_struct ≈ F_base
```

如果模板 block 只对每个 block 最后一个 token 算 logits：

```text
F_struct ≈ F_base - (S - B) * O(H*V)
```

注意，这个节省项和 `N` 无关；而 attention 大项是：

```text
O(D * N)
```

所以当 `N` 很大时，e2e FLOPs 降幅会被长上下文 attention 淹没。

---

**代入你说的例子**

```text
D = 30
S = 10
U = 20
B = 2
N 很大，比如 3000
```

attention 主项：

```text
baseline ≈ 30N
structured ≈ 20N + 10N = 30N
```

所以 attention FLOPs 几乎不降。

如果只省 lm_head：

```text
省 ≈ (S - B) * O(HV)
   = 8 * O(HV)
```

总体降幅大概是：

```text
(F_base - F_struct) / F_base
≈ 8*H*V / [30*(C_linear + A*d*N + H*V)]
```

如果 `N` 很大，分母被 attention 和 transformer 层计算撑大，降幅就不会接近 `10/30 = 33%`。
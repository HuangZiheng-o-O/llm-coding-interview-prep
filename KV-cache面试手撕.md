# 交叉注意力KV缓存解释

**Created:** 9/13/2026 23:31:58  
**Updated:** 9/13/2026 23:36:21  
**Exported:** 9/13/2026 23:36:41  
**Link:** [https://chatgpt.com/c/6aa76aac-f984-83ea-b345-6664202f2f63](https://chatgpt.com/c/6aa76aac-f984-83ea-b345-6664202f2f63)  

## Prompt:
你深度搜索面试题考手写KV cache到底是些什么代码？给我问题和最优雅简洁的回答

## Response:
我会把“手写 KV cache”拆成面试里最常见的几类：**单层缓存、增量解码、多头注意力、batch/beam、cross-attention 固定缓存**。我会给你每题一个极简标准答案，并说明面试官真正想看什么。

搜到的公开题型里，最核心的“手写 KV cache”并不是让你写 vLLM。常见要求其实集中在：**把普通 self-attention 改成增量 decode，正确维护每层 K/V，再解释 shape、复杂度和边界条件**。公开题库甚至直接把 `prefill/decode + per-layer cache + beam reorder + 预分配` 放在同一道高级题里。

我搜了 2026 年公开的 LLM inference 面试题库、代码题和 Hugging Face / PyTorch 当前实现。结论很明确：

**面试里所谓“手写 KV cache”，80% 的核心就是：把普通 causal self-attention 改成 `prefill + incremental decode`。** 更高级才会追问 static cache、Grouped Query Attention（分组查询注意力）、beam reorder、encoder-decoder cross cache。一个公开题库甚至把 “Implement KV cache for inference” 归到 Machine Learning Engineer technical screen，并明确要求 prefill/decode、per-layer cache、batch、beam search、预分配等。([prachub.com](https://prachub.com/interview-questions/implement-kv-cache-for-inference?utm_source=chatgpt.com))

你如果准备面试，下面这些代码值得直接练到能默写。

---

## 题 1：手写最基本 KV Cache

**面试官：**

> Implement multi-head causal self-attention with KV cache.  
> `x: [B, T, C]`，cache 中 K/V 为 `[B, H, S, D]`。支持 prefill 和单 token decode。

这是最核心的一题。

### 最简洁标准答案

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.h, self.d)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        # [B, H, T, D]

        if cache is not None:
            assert T == 1          # incremental decode
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        new_cache = (k, v)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(cache is None and T > 1)
        )

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y), new_cache
```

这段基本就是你要背的 **KV cache 母代码**。

核心只有：

```python
if cache is not None:
    k = torch.cat([old_k, new_k], dim=2)
    v = torch.cat([old_v, new_v], dim=2)
```

然后：

```python
attention(q_new, k_all, v_all)
```

公开 interview prep 的代码题也是围绕这个 incremental decode 展开的。([Sebastian Raschka's AI Magazine](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms?utm_source=chatgpt.com))

---

# 题 2：为什么只 cache K/V，不 cache Q？

这是几乎必追问。

**最优雅回答：**

在第 $t$ 个 decode step，只需要当前 token 的 query：

$$
O_t
=
\operatorname{softmax}
\left(
\frac{q_tK_{1:t}^{T}}{\sqrt d}
\right)
V_{1:t}
$$

历史的 $q_1,q_2,\dots,q_{t-1}$ 以后不会再使用。

而历史的：

$$
K_{1:t},V_{1:t}
$$

每个后续 token 都要访问，所以缓存 K/V。

一句话：

> **Q is consumed once; K and V are reused by every future query.**

这个回答非常标准。([PracHub](https://prachub.com/interview-questions/explain-kv-cache-in-transformer-inference?utm_source=chatgpt.com))

---

# 题 3：一个 Transformer 是一套 cache，还是每层一套？

**面试官：**

> How do you maintain KV cache across transformer layers?

答案：

**每个 attention layer 都有自己独立的 K/V cache。**

例如 32 层：

```text
layer 0:  K0, V0
layer 1:  K1, V1
layer 2:  K2, V2
...
layer 31: K31, V31
```

最简单的数据结构：

```python
past_key_values = [
    (k0, v0),
    (k1, v1),
    ...
]
```

模型 forward：

```python
def forward(self, x, past_key_values=None):
    if past_key_values is None:
        past_key_values = [None] * len(self.blocks)

    new_cache = []

    for block, cache in zip(self.blocks, past_key_values):
        x, cache = block(x, cache)
        new_cache.append(cache)

    return x, new_cache
```

### 面试官真正检查的点

千万别说：

```text
整个模型只有一个 K cache 和 V cache
```

因为不同 layer 有不同的 hidden state 和不同的 $W_K,W_V$，所以必须：

$$
(K^{(1)},V^{(1)}),
(K^{(2)},V^{(2)}),
\dots
$$

---

# 题 4：`torch.cat` 每一步是不是很垃圾？

这就是从普通算法题升级成 inference engineer 的地方。

面试官可能说：

> Your implementation calls `torch.cat` every decoding step. Can you improve it?

正确答案：

**提前分配固定大小的 cache，然后原地写入。**

```python
class StaticKVCache:
    def __init__(self, B, H, max_len, D, device, dtype):
        self.k = torch.empty(
            B, H, max_len, D,
            device=device, dtype=dtype
        )
        self.v = torch.empty_like(self.k)
        self.pos = 0

    def append(self, k, v):
        n = k.size(2)
        end = self.pos + n

        self.k[:, :, self.pos:end].copy_(k)
        self.v[:, :, self.pos:end].copy_(v)

        self.pos = end

        return (
            self.k[:, :, :end],
            self.v[:, :, :end]
        )
```

decode：

```python
k_all, v_all = cache.append(k_new, v_new)

y = F.scaled_dot_product_attention(
    q_new,
    k_all,
    v_all
)
```

这比反复：

```python
torch.cat(...)
```

更接近 production。

Hugging Face 当前的 `StaticCache` 正是预分配固定最大长度，然后原地更新；官方也明确指出这样更适合 `torch.compile`，代价是预留更多内存。([Hugging Face](https://huggingface.co/docs/transformers/v5.0.0/kv_cache?utm_source=chatgpt.com))

### 面试回答一句话

> `torch.cat` is fine for a reference implementation; production inference normally preallocates KV memory and writes the new K/V into the current cache position.

非常好用。

---

# 题 5：prefill 和 decode 到底有什么区别？

这个也极高频。

### Prefill

prompt：

```text
"I love machine learning"
```

一次送进去：

```python
x.shape == [B, S_prompt, C]
```

一次性得到：

```text
K_cache:
[B, H, S_prompt, D]

V_cache:
[B, H, S_prompt, D]
```

### Decode

以后每次只有一个 token：

```python
x_new.shape == [B, 1, C]
```

得到：

```text
q_new: [B, H, 1, D]
k_new: [B, H, 1, D]
v_new: [B, H, 1, D]
```

追加：

```python
K = cat([K_cache, k_new], dim=2)
V = cat([V_cache, v_new], dim=2)
```

attention：

```python
q_new @ K.transpose(-2, -1)
```

shape：

```text
[B,H,1,D] @ [B,H,D,S]
         ↓
[B,H,1,S]
```

因此 decode 时当前一个 query 会看**整个历史 KV**。

这就是 KV cache。

---

# 题 6：KV cache 把 attention 从 $O(n^2)$ 变成 $O(1)$ 吗？

**陷阱题。**

回答：

> No. At one decode step, the new query still attends to all previous keys, so attention work and KV reads grow linearly with context length.

假设现在已经有 $t$ 个 token：

$$
q_t \in \mathbb{R}^{1\times d}
$$

$$
K_{1:t}\in\mathbb{R}^{t\times d}
$$

依然需要：

$$
q_tK_{1:t}^{T}
$$

因此当前 step 大约随 $t$ 线性增长。

KV cache 省掉的是：

> **重新计算旧 token 的 K/V，以及重新运行整个历史 prefix。**

对于逐 token 生成整个长度 $T$，朴素地每次重新跑完整 prefix 会产生大量重复计算；cached incremental decoding 将每一步 attention 降到一行 query 对历史 K/V。([PracHub](https://prachub.com/concepts/llm-inference-optimization-and-kv-cache?utm_source=chatgpt.com))

---

# 题 7：Grouped Query Attention 下 KV cache shape 是什么？

现在非常值得准备。

假设：

```text
query heads = 32
KV heads    = 8
head dim    = 128
```

则：

```text
Q:
[B, 32, 1, 128]

K cache:
[B, 8, S, 128]

V cache:
[B, 8, S, 128]
```

**千万不要把 cache 存成 32 heads。**

Grouped Query Attention（GQA）的目的之一就是减少 KV cache。

当前 PyTorch 甚至可以直接：

```python
y = F.scaled_dot_product_attention(
    q,
    k_cache,
    v_cache,
    enable_gqa=True
)
```

PyTorch 当前文档规定 query head 数必须能被 KV head 数整除。([PyTorch Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention?utm_source=chatgpt.com))

如果要求手写 reference implementation：

```python
repeat = q.size(1) // k.size(1)

k_attn = k.repeat_interleave(repeat, dim=1)
v_attn = v.repeat_interleave(repeat, dim=1)

scores = q @ k_attn.transpose(-2, -1)
```

注意：

**cache 里依旧只存 8 个 KV heads。**

不要提前把它膨胀成 32 个再存。

这是一个非常好的加分点。

---

# 题 8：Beam Search 后 cache 怎么办？

高级一点的面试会问。

假设 beam search 选完以后：

```python
beam_idx = tensor([2, 0, 0, 3])
```

你的 KV cache 也必须跟着重新排列。

最漂亮的代码：

```python
def reorder_cache(past_key_values, beam_idx):
    return [
        (
            k.index_select(0, beam_idx),
            v.index_select(0, beam_idx)
        )
        for k, v in past_key_values
    ]
```

为什么？

因为 cache 第一维通常是：

```text
batch / beam
```

beam 的顺序发生改变，历史 K/V 必须跟对应 beam 一起改变，否则：

> 当前 token 属于 beam A，历史 KV 却可能来自 beam B。

公开的高级 KV-cache interview task 也明确要求 beam expand/reorder 和 EOS handling。([PracHub](https://prachub.com/interview-questions/implement-kv-cache-for-inference?utm_source=chatgpt.com))

---

# 题 9：Encoder-Decoder 的 Cross-Attention Cache 怎么写？

这个正好解决你刚才一直纠结的地方。

**面试官：**

> Implement KV caching for decoder cross-attention.

答案反而更简单：

```python
def forward(self, x, encoder_output, cache=None):
    q = self.q_proj(x)

    if cache is None:
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)
        cache = (k, v)
    else:
        k, v = cache

    out = attention(q, k, v)

    return out, cache
```

看到了吗？

这里**根本没有**：

```python
torch.cat(...)
```

因为：

```text
Q ← decoder 当前 hidden state

K ← encoder_output
V ← encoder_output
```

encoder output 整个 decoding 期间不变。

所以：

```python
if cache is None:
    算一次 K/V
else:
    直接复用
```

Hugging Face 当前甚至专门有 `EncoderDecoderCache`，内部明确分成：

```text
self_attention_cache
cross_attention_cache
```

两套。([GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py?utm_source=chatgpt.com))

这就是你上一轮问题对应到代码后的样子。

---

# 题 10：让你写 KV cache 显存公式

这个也建议直接背。

假设：

- $L$：Transformer layers
- $B$：batch size
- $S$：cached sequence length
- $H_{kv}$：KV heads
- $D$：head dimension
- $b$：每个元素 bytes

那么：

$$
\text{KV bytes}
=
2LB S H_{kv}Db
$$

前面的 $2$：

```text
K + V
```

例如 `float16/bfloat16`：

$$
b=2
$$

所以一个非常漂亮的面试回答是：

> KV-cache memory scales linearly with batch size, sequence length, layer count and number of KV heads.

这也是为什么 GQA / Multi-Query Attention（多查询注意力）能明显减少 cache。([AIInfraInterviews](https://aiinfrainterviews.com/blog/kv-cache-interview-questions?utm_source=chatgpt.com))

---

## 如果面试官只给你 15 分钟

你优先写这一版，真的就够了：

```python
def attention_with_cache(x, cache=None):
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    if cache is not None:
        old_k, old_v = cache
        k = torch.cat([old_k, k], dim=-2)
        v = torch.cat([old_v, v], dim=-2)

    new_cache = (k, v)

    scores = q @ k.transpose(-2, -1)
    scores /= q.size(-1) ** 0.5

    attn = scores.softmax(dim=-1)
    out = attn @ v

    return out, new_cache
```

然后嘴上补四句话：

> **1. Cache is per transformer layer.**  
> **2. During prefill I cache all prompt K/V.**  
> **3. During decode I compute Q/K/V only for the new token, append K/V, and let the new Q attend over all cached K/V.**  
> **4. In production I would preallocate the cache instead of calling `torch.cat` every step.**

如果还能再补一句：

> **For GQA, the cache uses `n_kv_heads`, not `n_query_heads`.**

基本已经是一个很完整的 inference 面试回答。

### 你现在最值得练的版本

如果目标是 **MLE / LLM / inference 面试**，我建议你真的只死磕这三个手写题：

```text
① CachedMultiHeadAttention
   Dynamic KV cache + prefill/decode

② StaticKVCache
   不用 cat，预分配 + position 写入

③ CrossAttentionWithKVCache
   encoder K/V 只算一次，不增长
```

把这三个能在白板上 **10 分钟无提示写出来**，KV cache 的手写题主体基本就覆盖了。高级面再补 GQA 和 beam reorder。当前公开的 frontier-lab / inference interview prep 也基本沿着这条路线展开。([Jiaxin Zhang](https://jxzhangjhu.github.io/blog/2026/frontier-lab-interview/?utm_source=chatgpt.com))

如果你愿意，我下一条可以直接给你一套 **“KV Cache 面试现场 30 分钟模拟题”**：我先只出题，你手写，我像面试官一样逐行挑 bug，包括 shape、causal mask、prefill/decode 和 `torch.cat` 的坑。

---

**Sources:**

- [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms?utm_source=chatgpt.com)
- [Explain KV cache in Transformer inference | OpenAI](https://prachub.com/interview-questions/explain-kv-cache-in-transformer-inference?utm_source=chatgpt.com)
- [Cache strategies · Hugging Face](https://huggingface.co/docs/transformers/v5.0.0/kv_cache?utm_source=chatgpt.com)
- [torch.nn.functional.scaled_dot_product_attention — PyTorch 2.14 documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention?utm_source=chatgpt.com)
- [transformers/src/transformers/cache_utils.py at main · huggingface/transformers · GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py?utm_source=chatgpt.com)
- [KV Cache Interview Questions: The Arithmetic That Decides Your Serving Answer | AIInfraInterviews](https://aiinfrainterviews.com/blog/kv-cache-interview-questions?utm_source=chatgpt.com)
- [Three Gates and Seven Buckets: What Frontier Labs Actually Test | Jiaxin Zhang](https://jxzhangjhu.github.io/blog/2026/frontier-lab-interview/?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
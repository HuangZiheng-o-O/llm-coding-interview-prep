阅读排序：
| 优先级                         | 题目                                                         | 我的建议                         |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| **S：必须闭卷写**              | Softmax、Cross-Entropy、Linear Layer、LayerNorm、Softmax Attention、MHA、Causal Attention、KV Cache、RoPE | 核心中的核心                     |
| **S−：现代 LLM 高频**          | RMSNorm、GQA、LoRA、Top-k/Top-p                              | 2026 面试价值很高                |
| **A：应熟练**                  | ReLU、Dropout、Embedding、GELU、SwiGLU、Cross-Attention、GPT-2 Block、Adam、Beam Search | 高频基础/组合题                  |
| **A−：理论很热，完整手写略少** | FlashAttention、MoE、DPO、PPO、GRPO、BPE                     | 至少能够闭卷写核心版本           |
| **B：有时间练**                | BatchNorm、Kaiming、Gradient Clip、Gradient Accumulation、Cosine LR、Sliding Window、Speculative Decoding、INT8 | 第二轮                           |
| **C：岗位相关**                | Linear Attention、Conv2D、ViT Patch Embedding、Linear Regression | LLM 岗可后放；CV/通用 MLE 则上调 |

> **合并说明**：本文档已并入《补充文档-LLM面试手撕代码补充题单》。每个级别章节内部先列主文档原有题目（第一部分），再列补充题单新增题目（第二部分）；章节顺序与原主文档一致。补充题单的参考文献见文末。

补充题单部分的编写原则：实现目标是“面试现场可手推、可手写、可解释”，优先保留核心计算图、shape、mask、索引和梯度语义；不追求 CUDA kernel、分布式容错、量化库兼容等工业细节。所有 NumPy 版本与 PyTorch 版本保持同一数学对象和计算逻辑。本版进一步与公开 LLM 手撕题库、PyTorch 官方实现说明及相关论文逐题交叉核对：只有在不损失数学语义与正确性的前提下，才吸收更短、更适合白板手写的写法。

补充题单新增题目一览：
| 级别 | 补充题单新增题目 |
| ---- | ---------------- |
| **S：必须闭卷写** | S1. AdamW Optimizer（Adam with Decoupled Weight Decay）、S2. Causal Language Modeling / Pretraining Loss（Next-Token Loss）、S3. Manual Backpropagation（Two-Layer MLP）、S4. Decoder-Only Transformer Block（Generic Pre-Norm） |
| **S−：现代 LLM 高频** | S−1. Multi-Query Attention（MQA）、S−2. Multi-Head Latent Attention（MLA，面试简化版）、S−3. RoPE with Position Offset / KV-Cache RoPE、S−4. Generalized Advantage Estimation（GAE）、S−5. SFT Masked Loss（Assistant-Only Loss）、S−6. Modern LLaMA-Style Block（RMSNorm + RoPE + GQA + SwiGLU） |
| **A：应熟练** | A1. Reward Model Pairwise Loss（Bradley-Terry / Preference Loss）、A2. KL Divergence / KL Penalty（Policy vs Reference）、A3. InfoNCE Loss（In-Batch Contrastive Loss）、A4. Pre-Norm vs Post-Norm Transformer、A5. Sinusoidal Positional Encoding、A6. ALiBi（Attention with Linear Biases）、A7. Attention Padding Mask + Causal Mask、A8. QLoRA（4-bit Frozen Base + LoRA，面试简化版）、A9. Gradient Checkpointing（Activation Checkpointing）、A10. Mixed Precision Training（FP16/BF16 + Loss Scaling）、A11. Standard FFN / MLP（Position-wise Feed-Forward）、A12. Knowledge Distillation Loss（Teacher-Student） |
| **A−：理论很热，完整手写略少** | A−1. PagedAttention（Simplified Page-Table Version）、A−2. Continuous Batching Scheduler（Iteration-Level Scheduling）、A−3. DAPO Objective / Loss（Decoupled Clip and Dynamic Sampling Policy Optimization）、A−4. GSPO Objective / Loss（Group Sequence Policy Optimization）、A−5. DDP Gradient All-Reduce（Simplified） |

### S：必须闭卷写

> 本章分两部分：**第一部分**为主文档原有题目，**第二部分**为《补充文档-LLM面试手撕代码补充题单》新增题目。

#### ▍第一部分 · 主文档原有题目

> [跳转到通用概率版 BCE](#probability-bce)

#### 👀 Softmax / Cross-Entropy / MSE

分类输出层的标准三件套。核心对象：softmax 把 logits 映射为概率，cross-entropy 衡量概率分布距离，MSE 用于回归。

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

$$
L_{\text{CE}}^{\text{multi}}
= -\frac{1}{N}\sum_{n=1}^{N}
\log \frac{e^{z_{n,y_n}}}{\sum_c e^{z_{n,c}}},
\quad z=\text{logits}
$$

$$
L_{\text{CE}}^{\text{bin}} = -\frac{1}{N}\sum_n \big[ y_n \log \hat{y}_n + (1-y_n)\log(1-\hat{y}_n) \big]
$$

$$
L_{\text{MSE}} = \frac{1}{N}\sum_n (y_n - \hat{y}_n)^2
$$

减最大值是 numerical stability（数值稳定，防 `exp` 上溢）。

#### （A）PyTorch 版本

<span style="color:red">**`dim=0` 是最外层；`dim=-1` 是最里面那一层。**</span>

以 attention scores 为例：shape 是 `(B, H, Tq, Tk)`——`B` batch，`H` head，`Tq` query 位置，**最后一维 `Tk` 是 key 位置**。因此 `softmax(scores, dim=-1)` 是对每个 query 沿全部 key 归一化（每行和为 1）。自注意力时常写成 `(B,H,T,T)`，那只是 `Tq=Tk=T` 的特例。


**① Softmax**

$$
\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

代码对应：减 `max` → `exp` → 沿 `dim` 求和再除。

```python
import torch

def softmax(x, dim=-1):
    # x: (..., C) -> 同形状概率，沿 dim 归一化
    x = x - x.max(dim=dim, keepdim=True).values # (...,1) 广播回 (...,C)：减 max 防 exp 上溢
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)  # 沿 dim 求和得 (...,1)，广播相除
```

**② Cross-Entropy（多分类，logits + class index）**

神经网络通常直接输出：

$$
z_1,z_2,\dots,z_C
$$

我们需要先通过 Softmax 转成概率：

$$
p_i=
\frac{e^{z_i}}
{\sum_{j=1}^{C}e^{z_j}}
$$

因此正确类别 $y$ 的概率为：

$$
p_y=
\frac{e^{z_y}}
{\sum_{j=1}^{C}e^{z_j}}
$$

再代入 Cross-Entropy：

$$
L=-\log p_y
$$

>从完整公式看最清楚。假设有 $C$ 个类别，真实标签分布为 $q$，模型预测概率为 $p$，交叉熵定义为
>
>$$
>L=-\sum_{i=1}^{C} q_i\log p_i
>$$
>
>如果真实类别是 $y$，那么 one-hot 标签就是
>
>$$
>q_i=
>\begin{cases}
>1,& i=y\\
>0,& i\neq y
>\end{cases}
>$$
>
>代进去：
>
>$$
>L
>=
>-\left(
>0\log p_1+\cdots+1\log p_y+\cdots+0\log p_C
>\right)
>$$
>
>所以只剩下
>
>$$
>\boxed{L=-\log p_y}
>$$
>
>直觉上，它只关心：**你给正确答案分了多大的概率。**
>

得到：
$$
L=
-\log
\left(
\frac{e^{z_y}}
{\sum_{j=1}^{C}e^{z_j}}
\right)
$$

利用对数公式：

$$
\log\frac ab=\log a-\log b
$$

得到：这就是 **logits + class index** 时最重要的 Cross-Entropy 公式

$$
L
=
-z_y+
\log
\left(
\sum_{j=1}^{C}e^{z_j}
\right)
$$


$$
L_{\text{CE}}^{\text{multi}}
= -\frac{1}{N}\sum_{n=1}^{N}
\log \frac{e^{z_{n,y_n}}}{\sum_c e^{z_{n,c}}}
= -\frac{1}{N}\sum_{n=1}^{N}
\big(z_{n,y_n} - \operatorname{logsumexp}_c(z_{n,c})\big),
\quad z=\text{logits}
$$

代码对应：`logits - logsumexp` 得 log-softmax，再取目标类下标求均值。

```python
def cross_entropy(logits, targets):
    # logits: (N, C) 未归一化分数；targets: (N,) 类别下标
    # log_softmax = logits - logsumexp(logits)，避免先 softmax 再 log
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # (N,1) 广播 -> (N,C)
    return -log_probs[torch.arange(targets.size(0)), targets].mean()  # 逐样本取目标类：(N,C)->(N,)
```

**③ BCE（二分类）**

$$
L_{\text{CE}}^{\text{bin}} = -\frac{1}{N}\sum_n \big[ y_n \log \hat{y}_n + (1-y_n)\log(1-\hat{y}_n) \big]
$$

代码对应：`y_pred` 先 clamp 防 `log(0)`，再按公式逐元素算后 `.mean()`。

```python
def bce(y_pred, y_true, eps=1e-12):
    # y_pred, y_true: (N,) 或 (N, 1) 概率/0-1标签；返回标量
    y_pred = torch.clamp(y_pred, eps, 1 - eps)   # 截到 [eps,1-eps]，防 log(0) = -inf
    return -(y_true * torch.log(y_pred)
             + (1 - y_true) * torch.log(1 - y_pred)).mean()  # 逐元素后对所有位置取均值
```

**④ MSE（回归）**
$$
L_{\text{MSE}} = \frac{1}{N}\sum_n (y_n - \hat{y}_n)^2
$$

代码对应：逐元素平方差再 `.mean()`。

```python
def mse(y_pred, y_true): return ((y_pred - y_true) ** 2).mean()  # 同 shape 逐元素平方 -> 标量
```

```
x = torch.tensor([
    [                    # 第1个样本
        [2.0, 1.0, 0.1],
        [0.5, 3.0, 2.0]
    ],
    [                    # 第2个样本
        [1.2, 0.3, 2.5],
        [4.0, 2.0, 1.0]
    ]
])

第1个样本
[
    位置1 -> [2.0, 1.0, 0.1]
    位置2 -> [0.5, 3.0, 2.0]
]

第2个样本
[
    位置1 -> [1.2, 0.3, 2.5]
    位置2 -> [4.0, 2.0, 1.0]
]
```

```
tensor([
    [
        [2.0],
        [3.0]
    ],
    [
        [2.5],
        [4.0]
    ]
])
```

位置1

```
[2.0,1.0,0.1]
```

最大值

```
2.0
```

位置2

```
[0.5,3.0,2.0]
```

最大值

```
3.0
```

第一步

```Python
# x - x.max(dim=-1, keepdim=True) 的结果：(2,2,3)，每行最大值被平移为 0
tensor([
    [
        [ 0.0,-1.0,-1.9],
        [-2.5, 0.0,-1.0]
    ],
    [
        [-1.3,-2.2, 0.0],
        [ 0.0,-2.0,-3.0]
    ]
])

```

https://chatgpt.com/share/6a3e2307-d208-83ea-91ba-e507e593865b

#### （B）NumPy 版本

**① Softmax**

```python
import numpy as np

def softmax(x, axis=-1):
    # x: (..., C) -> 同形状概率；减 max 防 exp 溢出
    x = x - np.max(x, axis=axis, keepdims=True)   # (...,1) 广播回 (...,C)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)  # 沿 axis 聚合成 (...,1) 后广播相除
```

**② Cross-Entropy（多分类，logits + class index）**

```python
def cross_entropy(logits, targets):
    # logits: (N, C) 未归一化分数；targets: (N,) 类别下标 -> 标量 loss
    x = logits - logits.max(axis=-1, keepdims=True)   # 平移不改 softmax 结果，仅为数值稳定
    log_probs = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))  # log-softmax，仍是 (N,C)
    return -log_probs[np.arange(len(targets)), targets].mean()  # 花式索引取对角元素：(N,C)->(N,)
```

**③ BCE（二分类）**

```python
def bce(y_pred, y_true, eps=1e-12):
    # y_pred: 预测概率；y_true: 0/1 标签，同 shape -> 标量
    y_pred = np.clip(y_pred, eps, 1 - eps)   # 截断到 (0,1) 开区间，防 log(0)
    return -np.mean(y_true * np.log(y_pred)
                    + (1 - y_true) * np.log(1 - y_pred))
```

**④ MSE（回归）**

```python
def mse(y_pred, y_true): return np.mean((y_pred - y_true) ** 2)  # 同 shape 逐元素 -> 标量
```

**Demo**

```python
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])   # (N=2, C=3)：2 个样本 3 个类别
targets = np.array([0, 1])                              # (N,)：样本 0 属类 0，样本 1 属类 1
print('p:', softmax(logits))                            # (2,3)，每行和为 1
print('CE:', cross_entropy(logits, targets))            # 标量
```

#### Step-by-step（softmax + CE 为例）

```python
def cross_entropy(logits, targets): pass   # logits:(N,C)，targets:(N,) 类别下标 -> 标量
```

（i）稳定 log-softmax：$$\log p = z - \operatorname{logsumexp}(z)$$
（ii）`log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)`
（iii）直接在 log 域计算；无需显式 softmax，也无需手动 `clamp`。

（i）取真实类别：$$\log p_{y_n}$$
（ii）`log_probs[torch.arange(N), targets]`
（iii）高级索引把 `(N,C)` 压成 `(N,)`，每个样本只取目标类别。

（i）交叉熵：$$L = -\frac{1}{N}\sum_n \log p_{n,y_n}$$
（ii）`return -log_probs[torch.arange(N), targets].mean()`
（iii）这是 `F.cross_entropy(logits, targets)` 的核心计算图。

#### 👀 LayerNorm（层归一化）

沿最后一维做归一化，含可学习缩放偏置。核心对象：标准化后的 $$\hat{x}$$。

$$
\mu = \frac{1}{d}\sum_i x_i,\quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},\quad y = \gamma \odot \hat{x} + \beta
$$

注意方差用有偏估计（除以 $$d$$ 而非 $$d-1$$），PyTorch 中 `unbiased=False`。

#### （A）PyTorch 版本

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        # gamma, beta: 形状 (d_model,) 可学习；初始化为恒等变换
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps                                 # 防止方差为 0 时除零

    def forward(self, x):
        # x: (..., d_model) -> 同形状
        mean = x.mean(-1, keepdim=True)                # 沿特征维聚合: (...,D) -> (...,1)
        var = x.var(-1, keepdim=True, unbiased=False)  # 有偏方差（除 D），同为 (...,1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # (...,1) 广播回 (...,D)：逐 token 标准化
        return self.gamma * x_hat + self.beta          # (D,) 广播到 (...,D)，逐通道缩放平移
```

这里用 `unbiased=False`，也就是方差除以 $D$，因为 LayerNorm 的目的只是对当前 token 的 $D$ 个特征做标准化，并不是用这 $D$ 个值去估计某个更大总体的方差；因此直接计算平均平方偏差 $ \sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2 $ 最自然，这样归一化后 $ \hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} $，忽略 $\epsilon$ 时，归一化后的特征均值为 $0$、方差为 $1$；而 `unbiased=True` 会除以 $D-1$，这是统计学中为了让“样本方差”更好地估计“总体方差”而做的 Bessel 校正（它适合这样的场景：你从某个未知概率分布中抽取 *D* 个样本，希望根据这些样本去估计那个分布真正的总体方差。），但 LayerNorm 并不存在这种总体参数估计需求，所以 PyTorch 的 `nn.LayerNorm` 也采用 `unbiased=False`。



#### （B）NumPy 版本

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: (..., D)   gamma, beta: (D,) -> 返回 (..., D)
    mean = x.mean(-1, keepdims=True)                   # 沿最后一维: (...,D) -> (...,1)
    var = x.var(-1, keepdims=True)                     # numpy 默认有偏（ddof=0）
    x_hat = (x - mean) / np.sqrt(var + eps)            # 广播标准化，逐 token 独立
    return gamma * x_hat + beta                        # (D,) broadcast 到 (...,D)

# Demo
x = np.random.randn(2, 4, 8)                           # (B=2, T=4, D=8)
g, b = np.ones(8), np.zeros(8)                         # 恒等仿射参数，各 (D,)
print(layer_norm(x, g, b).shape)                       # (2, 4, 8)：shape 不变
```

#### Step-by-step

```python
def layernorm(x, gamma, beta, eps=1e-5): pass   # x:(...,D)，gamma/beta:(D,) -> (...,D)
```

（i）按最后一维求均值 $$\mu$$
（ii）`mean = x.mean(-1, keepdim=True)`
（iii）每个 token 单独归一，跨 token 不混。

（i）有偏方差 $$\sigma^2$$
（ii）`var = x.var(-1, keepdim=True, unbiased=False)`
（iii）`unbiased=False` 对应除以 $$d$$。

（i）标准化 $$\hat{x}$$
（ii）`x_hat = (x - mean) / torch.sqrt(var + eps)`
（iii）`+eps` 防止 0 方差除零。

（i）仿射变换 $$y = \gamma \hat{x} + \beta$$
（ii）`return self.gamma * x_hat + self.beta`
（iii）让网络保留缩放/平移的表达能力。

#### 👀 MHA（Multi-Head Attention，含/不含 mask）

#### 👀 MultiHeadAttention
```py
import math
import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    普通通用 MHA：
    forward(q_source, k_source, v_source, attn_mask=None)

    适用：
    1. Self-Attention:
       q_source = k_source = v_source = x
    2. Cross-Attention:
       q_source = decoder_x
       k_source = v_source = encoder_output
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, S, D)
        view: (B, S, H, d_head)
        transpose: (B, H, S, d_head)

        作用：
        把一个 d_model 向量拆成 H 个 attention head。
        """
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, H, S, d_head)
        transpose: (B, S, H, d_head)
        reshape: (B, S, D)

        作用：
        把多个 head 的结果重新拼回 d_model。
        """
        B, H, S, d_head = x.shape

        return x.transpose(1, 2).contiguous().view(
            B, S, H * d_head
        )

    def forward(
        self,
        q_source: Tensor,
        k_source: Tensor,
        v_source: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        q_source: (B, S_q, D)
        k_source: (B, S_k, D)
        v_source: (B, S_k, D)

        attn_mask:
        - None，表示不 mask
        - 或 shape 可 broadcast 到 (B, H, S_q, S_k)
        - True 表示禁止 attend 的位置
        """

        B, S_q, _ = q_source.shape
        S_k = k_source.shape[1]

        # q: (B, S_q, D) -> (B, H, S_q, d_head)
        q = self._split_heads(self.W_q(q_source))

        # k: (B, S_k, D) -> (B, H, S_k, d_head)
        k = self._split_heads(self.W_k(k_source))

        # v: (B, S_k, D) -> (B, H, S_k, d_head)
        v = self._split_heads(self.W_v(v_source))

        # k.transpose(-2, -1): (B, H, d_head, S_k)
        # scores: (B, H, S_q, S_k)
        # 作用：每个 query token 对每个 key token 打分
        scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if attn_mask is not None:
            # attn_mask broadcast 到 scores
            # True 的位置被置为 -inf，softmax 后概率接近 0
            scores = scores.masked_fill(
                attn_mask, float("-inf")
            )

        # weights: (B, H, S_q, S_k)
        # dim=-1 表示每个 query 对所有 key 做 softmax
        weights = torch.softmax(scores, dim=-1)

        # attn: (B, H, S_q, d_head)
        # 作用：用 attention weights 对 value 加权求和
        attn = torch.matmul(weights, v)

        # out: (B, S_q, D)
        out = self._merge_heads(attn)

        return self.W_o(out)
```
#### 👀 DecoderSelfAttentionWithKVCache
```py
class DecoderSelfAttentionWithKVCache(nn.Module):
    """
    Decoder Self-Attention + KV cache：
    forward(x_new, cache=None)

    因为 Q/K/V 都来自 decoder 的 x_new，
    所以每一步只对新 token 计算 q_new/k_new/v_new。

    cache:
    - None
    - 或 (past_k, past_v)

    past_k: (B, H, S_past, d_head)
    past_v: (B, H, S_past, d_head)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, S, D)
        return: (B, H, S, d_head)
        """
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, H, S, d_head)
        return: (B, S, D)
        """
        B, H, S, d_head = x.shape

        return x.transpose(1, 2).contiguous().view(
            B, S, H * d_head
        )

    def forward(
        self,
        x_new: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        x_new: (B, S_new, D)

        训练 / prefill:
        - S_new 可以大于 1
        - cache 通常为 None

        增量 decode:
        - S_new 通常等于 1
        - cache 保存过去所有 token 的 K/V
        """

        B, S_new, _ = x_new.shape

        # q_new: (B, S_new, D) -> (B, H, S_new, d_head)
        q_new = self._split_heads(self.W_q(x_new))

        # k_new: (B, S_new, D) -> (B, H, S_new, d_head)
        k_new = self._split_heads(self.W_k(x_new))

        # v_new: (B, S_new, D) -> (B, H, S_new, d_head)
        v_new = self._split_heads(self.W_v(x_new))

        if cache is None:
            # 没有历史 token，total 就是当前 new token
            k_total = k_new
            v_total = v_new
        else:
            past_k, past_v = cache

            # past_k: (B, H, S_past, d_head)
            # k_new:  (B, H, S_new,  d_head)
            # cat dim=2，也就是 sequence 维度
            # k_total: (B, H, S_past + S_new, d_head)
            k_total = torch.cat([past_k, k_new], dim=2)
            v_total = torch.cat([past_v, v_new], dim=2)

        new_cache = (k_total, v_total)

        S_total = k_total.shape[2]
        S_past = S_total - S_new

        # scores: (B, H, S_new, S_total)
        # 每个新 query 可以看向 cache 中的历史 K，
        # 以及当前 chunk 内允许看的 K。
        scores = torch.matmul(
            q_new, k_total.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if S_new > 1:
            # prefill 或 chunk decode 时需要 causal mask。
            #
            # query 位置范围：
            # 当前 chunk 的第 i 个 query，
            # 它在完整序列中的绝对位置是 S_past + i。
            #
            # key 位置范围：
            # 0 到 S_total - 1。
            #
            # 如果 key_pos > query_abs_pos，则不能 attend。
            #
            # mask: (S_new, S_total)
            # True 表示未来位置，需要屏蔽。
            q_pos = torch.arange(
                S_past, S_total, device=x_new.device
            ).unsqueeze(1)               # (S_new, 1)

            k_pos = torch.arange(
                S_total, device=x_new.device
            ).unsqueeze(0)               # (1, S_total)

            mask = k_pos > q_pos         # (S_new, S_total)

            # mask broadcast:
            # (S_new, S_total)
            # -> (1, 1, S_new, S_total)
            # -> 匹配 scores: (B, H, S_new, S_total)
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        # decode 单 token 时 S_new == 1：
        # 当前 token 的 query 可以 attend 所有 past + current，
        # 不需要额外 causal mask。
        weights = torch.softmax(scores, dim=-1)

        # attn: (B, H, S_new, d_head)
        attn = torch.matmul(weights, v_total)

        # out: (B, S_new, D)
        out = self._merge_heads(attn)

        return self.W_o(out), new_cache
```
#### 👀 CrossAttentionWithKVCache

```py

class CrossAttentionWithKVCache(nn.Module):
    """
    Cross-Attention + KV cache：
    forward(decoder_x, encoder_output, cache=None)

    Q 来源：decoder_x
    K/V 来源：encoder_output

    encoder_output 在整个 decoding 过程中不变，
    所以 cross-attention 的 K/V 可以缓存。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, S, D)
        return: (B, H, S, d_head)
        """
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, H, S, d_head)
        return: (B, S, D)
        """
        B, H, S, d_head = x.shape

        return x.transpose(1, 2).contiguous().view(
            B, S, H * d_head
        )

    def forward(
        self,
        decoder_x: Tensor,
        encoder_output: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
        encoder_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        decoder_x: (B, S_q, D)
        encoder_output: (B, S_enc, D)

        cache:
        - None：第一次 cross-attention，计算 encoder K/V
        - (enc_k, enc_v)：后续 decode 复用 encoder K/V

        encoder_mask:
        - None
        - 或 shape 可 broadcast 到 (B, H, S_q, S_enc)
        - True 表示 encoder padding，需要屏蔽
        """

        B, S_q, _ = decoder_x.shape

        # q: (B, S_q, D) -> (B, H, S_q, d_head)
        # 注意：Q 来自 decoder 当前 hidden states。
        q = self._split_heads(self.W_q(decoder_x))

        if cache is None:
            # 第一次调用时，从 encoder_output 计算 K/V。
            # encoder_output: (B, S_enc, D)
            #
            # enc_k: (B, H, S_enc, d_head)
            # enc_v: (B, H, S_enc, d_head)
            enc_k = self._split_heads(self.W_k(encoder_output))
            enc_v = self._split_heads(self.W_v(encoder_output))
        else:
            # 后续 decode 复用 encoder K/V。
            # 因为 encoder_output 不随 decoder step 改变。
            enc_k, enc_v = cache

        new_cache = (enc_k, enc_v)

        # scores: (B, H, S_q, S_enc)
        # 每个 decoder query attend 所有 encoder key。
        scores = torch.matmul(
            q, enc_k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if encoder_mask is not None:
            # encoder_mask 通常用于屏蔽 padding token。
            # broadcast 到 scores 后，True 的位置不可见。
            scores = scores.masked_fill(
                encoder_mask, float("-inf")
            )

        # weights: (B, H, S_q, S_enc)
        weights = torch.softmax(scores, dim=-1)

        # attn: (B, H, S_q, d_head)
        attn = torch.matmul(weights, enc_v)

        # out: (B, S_q, D)
        out = self._merge_heads(attn)

        return self.W_o(out), new_cache


# Demo
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, H = 2, 32, 4
    S_q, S_k, S_new, S_enc = 3, 5, 1, 6

    # 1. 普通 MHA
    mha = MultiHeadAttention(D, H)
    q_src = torch.randn(B, S_q, D)
    k_src = torch.randn(B, S_k, D)
    v_src = torch.randn(B, S_k, D)
    out = mha(q_src, k_src, v_src)
    print("MHA:", out.shape)  # (B, S_q, D)

    # 2. Decoder Self-Attention + KV cache
    self_attn = DecoderSelfAttentionWithKVCache(D, H)
    x1 = torch.randn(B, 4, D)
    x2 = torch.randn(B, S_new, D)

    out1, cache = self_attn(x1, cache=None)
    out2, cache = self_attn(x2, cache=cache)

    print("Self out1:", out1.shape)          # (B, 4, D)
    print("Self out2:", out2.shape)          # (B, 1, D)
    print("Self cache K:", cache[0].shape)   # (B, H, 5, d_head)

    # 3. Cross-Attention + KV cache
    cross_attn = CrossAttentionWithKVCache(D, H)
    decoder_x = torch.randn(B, S_new, D)
    encoder_output = torch.randn(B, S_enc, D)

    cross_out1, cross_cache = cross_attn(
        decoder_x,
        encoder_output,
        cache=None,
    )

    cross_out2, cross_cache = cross_attn(
        decoder_x,
        encoder_output,
        cache=cross_cache,
    )

    print("Cross out:", cross_out1.shape)        # (B, 1, D)
    print("Cross cache K:", cross_cache[0].shape) # (B, H, S_enc, d_head)
```



将 Q/K/V 投到 $$h$$ 个子空间并行做 scaled dot-product attention，最后拼回并线性变换。核心对象：注意力分数 $$\text{softmax}(QK^\top/\sqrt{d_k})$$，shape 为 `(B, H, Tq, Tk)`——最后一维 `Tk` 是 key 位置，softmax 沿该维归一化。

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

$$
\text{MHA}(X) = W_O \cdot \text{Concat}\big[\text{head}_1, \dots, \text{head}_h\big],\quad
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

其中 $$d_k = d_{\text{model}}/h$$，$$M$$ 是可选 mask（causal：上三角置 $$-\infty$$）。

#### （A）PyTorch 版本

```
Q: (B, H, T_q, d_q)
K: (B, H, T_k, d_k)
V: (B, H, T_k, d_v)
```



```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0                # d_model 必须能被 head 数整除
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads           # 每个 head 的维度 d_k
        self.Wq = nn.Linear(d_model, d_model)         
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
#‼️Cross Attention 中 Q 来自一边，K,V 来自另一边；d_q=d_k 必须满足，d_v 可以不同，但实际 Transformer 通常也设成一样。
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (B, T, d_model)；mask 可选 -> 返回 (B, T_q, d_model)
        B = Q.size(0)                                  # batch size
        # 投影 + 切头: (B, T, d_model) -> (B, H, T, d_k)；view 拆 head，transpose 把 H 提到前面
        Q = self.Wq(Q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(K).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(V).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # scores: (B,H,T_q,d_k) @ (B,H,d_k,T_k) -> (B,H,T_q,T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 除 sqrt(d_k) 防 softmax 饱和；因为 Q\cdot K 的方差随 d_k 增长，而标准差随 \sqrt{d_k} 增长
        if mask is not None:
            # mask: (1,1,T_q,T_k) 或可广播; 0 处屏蔽（causal 用下三角全 1）
            scores = scores.masked_fill(mask == 0, float('-inf'))  # -inf 经 softmax 后权重为 0；scores.masked_fill(条件, 值) 的意思是：在 scores 里，凡是条件为 True 的位置，替换成指定的值。
        attn = F.softmax(scores, dim=-1)               # dim=-1 = Tk（key 位置）：(B,H,Tq,Tk) 每行对全部 key 归一化
        # out: (B,H,T_q,T_k) @ (B,H,T_k,d_v) -> (B,H,T_q,d_v)：按权重聚合 value
        out = torch.matmul(attn, V)
        # 合头: (B,H,T_q,d_v) -> (B,T_q,d_model)；contiguous 是 transpose 后 view 的前提
        # view() 通常要求连续内存，所以先调用 contiguous()。
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.Wo(out)                            # 融合各 head 信息，shape 不变
```

#### （B）NumPy 版本（苛刻面试版）

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)            # 减 max 防溢出，shape 不变
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def mha_numpy(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    # X: (B, T, D)   W*: (D, D) -> 返回 (B, T, D)
    B, T, D = X.shape
    H = num_heads; d_k = D // H                        # H: head 数；d_k: 每 head 维度
    # 投影 + reshape + 转置: (B, T, D) -> (B, H, T, d_k)
    def split(x, W):
        y = x @ W                                      # (B,T,D) @ (D,D) -> (B, T, D)
        return y.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # 把 D 拆成 (H,d_k) 再把 H 换到轴 1
    Q, K, V = split(X, Wq), split(X, Wk), split(X, Wv)
    # scores: (B,H,T,d_k) @ (B,H,d_k,T) -> (B,H,T,T)；自注意力下 Tq=Tk=T，最后一维仍是 key 位置
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)     # 用大负数而非 -inf，避免整行被屏蔽时出 NaN
    attn = softmax(scores, axis=-1)                    # 沿 Tk（最后一维 / key 位置）归一化，每行和为 1
    out = attn @ V                                     # (B,H,T,T) @ (B,H,T,d_k) -> (B, H, T, d_k)
    # 合头: (B, H, T, d_k) -> (B, T, D)，把各 head 沿特征维拼回
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ Wo

# Demo
B, T, D, H = 2, 4, 16, 4                               # batch / 序列长 / 模型维 / head 数
X = np.random.randn(B, T, D)
Wq, Wk, Wv, Wo = [np.random.randn(D, D) * 0.1 for _ in range(4)]  # 小尺度初始化，防 scores 过大
mask = np.tril(np.ones((T, T)))[None, None]            # 下三角 causal，(T,T) -> (1,1,T,T) 便于广播
print(mha_numpy(X, Wq, Wk, Wv, Wo, H, mask).shape)     # (2, 4, 16)
```

#### Step-by-step

```python
def mha(Q, K, V, mask=None): pass   # Q/K/V:(B,T,D)，mask 可广播 -> (B,T_q,D)
```

（i）线性投影 + 多头切分
$$
Q' = QW^Q \in \mathbb{R}^{B \times T \times D} \to \mathbb{R}^{B \times H \times T \times d_k}
$$
（ii）`Q = self.Wq(Q).view(B, -1, H, d_k).transpose(1, 2)`
（iii）`view` 拆 head，`transpose` 把 head 提到 batch 后，方便 4D matmul 并行。

（i）相似度矩阵
$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times T_q \times T_k}
$$
（ii）`scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)`
（iii）除 $$\sqrt{d_k}$$ 防止 logits 过大导致 softmax 饱和（gradient vanishing）。

（i）mask：$$S_{ij} \leftarrow -\infty$$ 若不可见
（ii）`scores = scores.masked_fill(mask == 0, float('-inf'))`
（iii）`-inf` 经 softmax 自动归零；causal mask 用下三角全 1 矩阵。

（i）softmax + 加权求和
$$
A = \text{softmax}(S), \quad O = AV
$$
（ii）`attn = F.softmax(scores, dim=-1); out = torch.matmul(attn, V)`
（iii）每个 query 加权聚合所有 key 对应的 value。

（i）合头 + 输出投影
（ii）`out = out.transpose(1,2).reshape(B, -1, D); return self.Wo(out)`
（iii）拼接 head 维度，最后过 $$W_O$$ 融合不同 head 的信息。

#### 👀 MHA + KV Cache

自回归解码时只输入新 token，把历史 K/V 缓存复用，避免每步重算前面所有位置。核心对象：cache 中拼接的 $$K_{\text{cache}}, V_{\text{cache}}$$，以及带 cache offset 的 causal mask。

$$
K_t = \text{Concat}(K_{<t}, k_t),\quad
V_t = \text{Concat}(V_{<t}, v_t)
$$

$$
\text{Attn}(Q,K,V)
= \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{causal}}\right)V
$$

prefill 时 `T_new > 1`，必须加 causal mask 防止当前位置看到未来 token；单 token decode 时 `T_new=1`，同一套 offset mask 会自然退化为“所有已有 KV 都可见”。

#### （A）PyTorch 版本

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
 
    def _split(self, x):
        # (B, T, D) -> (B, H, T, d_k)：拆 head 并把 H 提到 batch 之后
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        # (B, H, T, d_k) -> (B, T, D)：_split 的逆操作，各 head 沿特征维拼接
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def forward(self, x, kv_cache=None):
        # x: (B, T_new, D) 只含本步新增 token；返回 (输出, 更新后的 cache)
        Q = self._split(self.Wq(x))                    # (B,H,T_new,d_k)
        K = self._split(self.Wk(x))
        V = self._split(self.Wv(x))

        past_len = 0                                   # 已缓存的历史 token 数
        if kv_cache is not None and kv_cache.get("k") is not None:
            past_len = kv_cache["k"].size(2)           # cache 的时间维长度
            K = torch.cat([kv_cache["k"], K], dim=2)  # 沿时间维拼: (B,H,T_total,d_k)，历史在前
            V = torch.cat([kv_cache["v"], V], dim=2)
        new_cache = {"k": K, "v": V}                   # 含历史+新增，供下一步复用

        # Q @ K^T: (B,H,T_new,d_k) @ (B,H,d_k,T_total)
        # -> scores: (B,H,T_new,T_total)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # cache-aware causal mask: query i 的绝对位置是 past_len + i
        T_new, T_total = Q.size(2), K.size(2)          # 本步 query 数 / 全部可见 key 数
        mask = torch.triu(                             # (T_new,T_total) bool，True = 需屏蔽的未来位置
            torch.ones(T_new, T_total, device=x.device, dtype=torch.bool),
            diagonal=past_len + 1,                     # 偏移 past_len 使 query i 只能看到 key <= past_len+i
        )
        scores = scores.masked_fill(mask, float("-inf"))  # (T_new,T_total) 广播到 (B,H,T_new,T_total)

        attn = F.softmax(scores, dim=-1)               # 沿 T_total 归一化
        out = self._merge(torch.matmul(attn, V))       # (B,H,T_new,d_k) -> (B,T_new,D)
        return self.Wo(out), new_cache
```

#### （B）NumPy 版本

```python
import numpy as np

def mha_kv_cache_numpy(x, Wq, Wk, Wv, Wo, H, cache=None):
    # x: (B,T_new,D)；cache['k'/'v']: (B,H,T_cache,d_k) -> 返回 (输出, 新 cache)
    B, T, D = x.shape                                  # T 即 T_new，本步新 token 数
    d_k = D // H

    def split(y, W):
        z = y @ W                                      # (B,T,D) @ (D,D) -> (B,T,D)
        return z.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # -> (B,H,T,d_k)

    Q, K, V = split(x, Wq), split(x, Wk), split(x, Wv)
    past_len = 0                                       # 历史 token 数，决定 mask 偏移
    if cache is not None and cache.get("k") is not None:
        past_len = cache["k"].shape[2]
        K = np.concatenate([cache["k"], K], axis=2)   # 沿时间维拼接: (B,H,T_total,d_k)
        V = np.concatenate([cache["v"], V], axis=2)
    new_cache = {"k": K, "v": V}

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (B,H,T,T_total)
    mask = np.triu(                                    # (T,T_total) bool，True 表示未来位置
        np.ones((T, K.shape[2]), dtype=bool),
        k=past_len + 1,                                # query i 的绝对位置是 past_len+i
    )
    scores = np.where(mask[None, None], -np.inf, scores)  # mask 升维到 (1,1,T,T_total) 后广播

    a = scores - scores.max(-1, keepdims=True)         # 手写 softmax：减行最大值保数值稳定
    attn = np.exp(a) / np.exp(a).sum(-1, keepdims=True)
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, D)  # (B,H,T,d_k) -> 合头 (B,T,D)
    return out @ Wo, new_cache

# Demo: 单 token decode 时 cache 长度逐步增长
B, D, H = 1, 16, 4                                     # 逐 token 解码，batch=1
Wq, Wk, Wv, Wo = [np.random.randn(D, D) * 0.1 for _ in range(4)]
cache = None                                           # 首步无历史，past_len=0
for t in range(3):
    x_new = np.random.randn(B, 1, D)                   # (B, T_new=1, D)：每步只喂一个新 token
    out, cache = mha_kv_cache_numpy(x_new, Wq, Wk, Wv, Wo, H, cache)
    print(f"step {t}, cache K len = {cache['k'].shape[2]}")  # 依次为 1, 2, 3
```

#### Step-by-step

```python
def mha_kv(x, cache=None): pass   # x:(B,T_new,D)，cache 存 (B,H,T_cache,d_k) -> (输出, 新 cache)
```

（i）新 token 投影：`Q,K,V = split(Wq(x)), split(Wk(x)), split(Wv(x))`。

（ii）记录历史长度并拼 cache：`past_len = cache['k'].size(2)`，随后 `torch.cat(..., dim=2)`；shape 从 `(B,H,T_new,d_k)` 变为 `(B,H,T_total,d_k)`。

（iii）attention scores：`scores = Q @ K.transpose(-2,-1) / sqrt(d_k)`，得到 `(B,H,T_new,T_total)`。

（iv）关键 causal offset：`mask = triu(ones(T_new,T_total), diagonal=past_len+1)`；prefill 与 chunked decode 都不会看到未来位置。

（v）`softmax -> @V -> merge -> Wo`，同时返回 `new_cache` 供下一步复用。

---

#### ▍第二部分 · 补充题单新增题目

#### S1. AdamW Optimizer（Adam with Decoupled Weight Decay）

##### 1️⃣ Dense Opening

AdamW 解决 Adam 中 L2 regularization 与真正 weight decay 不等价的问题：先用 Adam 的一阶矩、二阶矩做自适应梯度更新，再对参数单独做 decoupled weight decay。核心对象是 $$m_t$$、$$v_t$$、bias correction 与独立的参数衰减项 $$\lambda\theta$$。面试里应能从 Adam 公式直接写出 AdamW 单步更新。[参考 1]

##### 2️⃣ Global Equations

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t
=
(1-\eta\lambda)\theta_{t-1}
-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中 $$\eta$$ 是 learning rate，$$\lambda$$ 是 weight decay。关键点：衰减直接作用于参数，不进入 $$g_t$$ 的一阶/二阶矩统计。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准面试实现）**

```python
import torch
from torch import Tensor

def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step: int,                       # 从 1 开始的迭代计数，仅用于 bias correction
    lr: float = 1e-3,
    beta1: float = 0.9,              # 一阶矩衰减率
    beta2: float = 0.999,            # 二阶矩衰减率
    eps: float = 1e-8,               # 分母防除零
    weight_decay: float = 0.01,
):
    # param, grad, m, v: 同 shape，例如 (D,) / (M, N)；全程逐元素运算，shape 不变
    m = beta1 * m + (1.0 - beta1) * grad           # 梯度的 EWMA（动量方向）
    v = beta2 * v + (1.0 - beta2) * grad.square()  # 梯度平方的 EWMA（逐坐标尺度）

    # bias correction：训练初期 m、v 从 0 开始会偏小
    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)

    # decoupled weight decay：直接衰减参数，不进入 m/v 统计
    param = param * (1.0 - lr * weight_decay)

    # Adam 自适应更新；shape 不变
    param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return param, m, v                             # 新参数与新矩状态，供下一步传入
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def adamw_step_numpy(
    param,
    grad,
    m,
    v,
    step,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01,
):
    # param/grad/m/v 同 shape，逐元素更新；step 从 1 开始
    m = beta1 * m + (1.0 - beta1) * grad               # 一阶矩：梯度 EWMA
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)        # 二阶矩：逐元素平方，不可用 norm 聚合

    m_hat = m / (1.0 - beta1 ** step)                  # bias correction，抵消零初始化偏差
    v_hat = v / (1.0 - beta2 ** step)

    param = param * (1.0 - lr * weight_decay)          # decoupled weight decay
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)  # 按坐标自适应步长
    return param, m, v
```

**最小可运行测试**

```python
# Demo
p = torch.tensor([1.0, -2.0])      # 待优化参数，(D=2,)
g = torch.tensor([0.1, -0.2])      # 该步梯度，与 p 同 shape
m = torch.zeros_like(p)            # 一阶矩初值全 0
v = torch.zeros_like(p)            # 二阶矩初值全 0

p, m, v = adamw_step(
    p, g, m, v,
    step=1,                        # 首步，bias correction 分母为 1-beta
    lr=1e-2,
    weight_decay=0.1,
)
print("param:", p)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def adamw_step(
    param: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    step: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    pass   # param/grad/m/v 同 shape；返回 (新 param, 新 m, 新 v)
```

**Step 1：更新一阶矩**

（i）公式

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

（ii）代码

```python
m = beta1 * m + (1.0 - beta1) * grad   # 逐元素 EWMA，m 与 grad 同 shape
```

（iii）$$m_t$$ 是梯度的指数滑动平均（EWMA, Exponentially Weighted Moving Average），shape 与参数完全一致。

**Step 2：更新二阶矩**

（i）公式

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

（ii）代码

```python
v = beta2 * v + (1.0 - beta2) * grad.square()   # square 是逐元素平方，不做任何维度聚合
```

（iii）逐元素统计平方梯度尺度；不能写成 `grad.norm() ** 2`，否则会把所有参数维聚合成一个标量。

**Step 3：Bias correction**

（i）公式

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

（ii）代码

```python
m_hat = m / (1.0 - beta1 ** step)   # step 为标量，除数广播；shape 不变
v_hat = v / (1.0 - beta2 ** step)
```

（iii）因为 $$m_0=v_0=0$$，训练早期矩估计向 0 偏，需要校正。

**Step 4：Decoupled weight decay**

（i）公式

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

（ii）代码

```python
param = param * (1.0 - lr * weight_decay)   # 只缩放参数本身，不污染 m/v 的梯度统计
```

（iii）这是 AdamW 与“把 $$\lambda\theta$$ 加进 gradient”最关键的区别。

**Step 5：Adam update**

（i）公式

$$
\theta
\leftarrow
\theta-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

（ii）代码

```python
param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)   # 逐坐标自适应步长，eps 防除零
```

（iii）$$\sqrt{\hat v_t}$$ 对每个参数坐标自适应缩放更新幅度。

#### S2. Causal Language Modeling / Pretraining Loss（Next-Token Loss）

##### 1️⃣ Dense Opening

Decoder-only LLM 的预训练目标是 next-token prediction：位置 $$t$$ 的 logits 预测位置 $$t+1$$ 的 token。核心操作只有两个：sequence shift 与 masked cross-entropy。最容易写错的是 logits/labels 没有错一位，以及 padding token 被计入 loss。[参考 2]

##### 2️⃣ Global Equations

给定 token 序列 $$x_1,\ldots,x_T$$：

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T-1}
p_\theta(x_{t+1}\mid x_{\le t})
$$

$$
\mathcal{L}_{\mathrm{CLM}}
=
-\frac{1}{|\mathcal M|}
\sum_{(b,t)\in\mathcal M}
\log
p_\theta
\left(
x_{b,t+1}\mid x_{b,\le t}
\right)
$$

其中 $$\mathcal M$$ 是非 padding 的有效 target 位置。

shape 对齐：

$$
\text{logits}:(B,T,V)
\rightarrow
(B,T-1,V)
$$

$$
\text{labels}:(B,T)
\rightarrow
(B,T-1)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    # logits: (B, T, V)
    # labels: (B, T)，padding 可预先写成 -100
    # logits[:, t] 预测 labels[:, t+1]
    shift_logits = logits[:, :-1, :]       # 丢掉最后位置（无 target）：(B,T,V) -> (B, T-1, V)
    shift_labels = labels[:, 1:]           # 左移一位对齐 next token：(B,T) -> (B, T-1)

    V = shift_logits.size(-1)              # vocabulary size，必须保留为分类维
    return F.cross_entropy(
        shift_logits.reshape(-1, V),       # 合并 batch 与时间维: (B*(T-1), V)
        shift_labels.reshape(-1),          # (B*(T-1),)
        ignore_index=ignore_index,         # 该值的位置不计入分母，用于 padding
    )
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def causal_lm_loss_numpy(
    logits,
    labels,
    ignore_index=-100,
):
    # logits: (B, T, V), labels: (B, T) -> 标量 loss
    x = logits[:, :-1, :]                  # 去掉末位: (B, T-1, V)
    y = labels[:, 1:]                      # 左移一位: (B, T-1)，y[t] 是 x[t] 的 target

    B, Tm1, V = x.shape
    x = x.reshape(-1, V)                   # (B*(T-1), V)
    y = y.reshape(-1)                      # (B*(T-1),)

    mask = y != ignore_index               # bool (B*(T-1),)：True 为有效 target
    x = x[mask]                            # 布尔索引筛掉无效行: (M, V)
    y = y[mask]                            # (M,)

    # stable log-softmax
    x = x - x.max(axis=-1, keepdims=True)  # (M,1) 广播，防 exp 上溢
    log_z = np.log(np.exp(x).sum(axis=-1))  # 沿 V 聚合: (M,V) -> (M,)
    nll = log_z - x[np.arange(len(y)), y]  # 花式索引取目标类分数 -> (M,)
    return nll.mean()                      # 仅在 M 个有效位置上取均值
```

**最小可运行测试**

```python
# Demo
B, T, V = 2, 5, 10                              # batch / 序列长 / 词表大小
logits = torch.randn(B, T, V)                   # (B,T,V) 模型输出分数
labels = torch.randint(0, V, (B, T))            # (B,T) token id
labels[0, -1] = -100                            # 标记为 padding，该 target 不计入 loss

print("CLM loss:", causal_lm_loss(logits, labels).item())
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    pass   # logits:(B,T,V)，labels:(B,T) -> 标量；核心是错位对齐 + masked CE
```

**Step 1：错位对齐**

（i）公式

$$
\hat y_t=\text{logits}_t,
\qquad
y_t=x_{t+1}
$$

（ii）代码

```python
shift_logits = logits[:, :-1, :]   # (B,T,V) -> (B,T-1,V)：去掉无 target 的末位
shift_labels = labels[:, 1:]       # (B,T) -> (B,T-1)：整体左移一位
```

（iii）最后一个 logits 没有下一个 target；第一个 token 通常作为上下文，不由序列内部前一个位置预测。

**Step 2：展平 batch 与 sequence**

（i）shape

$$
(B,T-1,V)\rightarrow(B(T-1),V)
$$

$$
(B,T-1)\rightarrow(B(T-1))
$$

（ii）代码

```python
V = shift_logits.size(-1)                  # 词表维大小，保留为分类维
flat_logits = shift_logits.reshape(-1, V)  # (B,T-1,V) -> (B*(T-1), V)
flat_labels = shift_labels.reshape(-1)     # (B,T-1) -> (B*(T-1),)，与上面逐行对应
```

（iii）只合并样本维和时间维，vocabulary 维必须保留为分类维。

**Step 3：Cross-Entropy**

（i）公式

$$
\ell_i
=
-\log
\frac{e^{z_{i,y_i}}}
{\sum_{v=1}^{V}e^{z_{i,v}}}
$$

（ii）代码

```python
return F.cross_entropy(
    flat_logits,                 # (N, V) 未归一化 logits，内部自带 log_softmax
    flat_labels,                 # (N,) 类别下标
    ignore_index=ignore_index,   # 命中该值的位置不参与均值（分子分母都排除）
)
```

（iii）`ignore_index=-100` 的 token 不参与均值，适合 padding 或显式屏蔽位置。

#### S3. Manual Backpropagation（Two-Layer MLP）

##### 1️⃣ Dense Opening

手写反向传播最适合用两层 MLP：Linear $$\rightarrow$$ ReLU $$\rightarrow$$ Linear $$\rightarrow$$ MSE。核心对象是链式法则中每个局部 Jacobian；面试时要能从 $$d\hat Y$$ 一路反推 $$dW_2, db_2, dW_1, db_1$$，同时保持矩阵 shape 对齐。

##### 2️⃣ Global Equations

前向：

$$
Z_1=XW_1+b_1
$$

$$
H=\operatorname{ReLU}(Z_1)
$$

$$
\hat Y=HW_2+b_2
$$

$$
L
=
\frac{1}{2B}
\|\hat Y-Y\|_F^2
$$

反向：

$$
d\hat Y
=
\frac{\hat Y-Y}{B}
$$

$$
dW_2=H^\top d\hat Y,
\qquad
db_2=\sum_b d\hat Y_b
$$

$$
dH=d\hat YW_2^\top
$$

$$
dZ_1=dH\odot\mathbb{1}[Z_1>0]
$$

$$
dW_1=X^\top dZ_1,
\qquad
db_1=\sum_b dZ_{1,b}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（禁用 autograd，手写梯度）**

```python
import torch
from torch import Tensor

def mlp_forward_backward(
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
):
    # X: (B, D) 输入批次；Y: (B, O) 回归目标
    # W1: (D, H), b1: (H,)；W2: (H, O), b2: (O,)
    # 返回 loss(标量)、Y_hat:(B,O)、grads(每项与对应参数同 shape)
    Z1 = X @ W1 + b1                    # (B,D)@(D,H) -> (B,H)；b1 沿 batch 广播
    H = torch.relu(Z1)                  # (B, H)：负半轴置 0，得到隐藏激活
    Y_hat = H @ W2 + b2                 # (B,H)@(H,O) -> (B,O) 预测值

    B = X.size(0)                       # batch size，loss 与梯度都按它取平均
    diff = Y_hat - Y                    # (B, O) 残差
    loss = 0.5 * diff.square().sum() / B  # 1/2 与平方求导的 2 相消，简化 dY

    dY = diff / B                       # (B, O)：dL/dY_hat
    dW2 = H.T @ dY                      # (H,B)@(B,O) -> (H,O)，与 W2 对齐
    db2 = dY.sum(dim=0)                 # 沿 batch 维聚合 -> (O,)

    dH = dY @ W2.T                      # (B,O)@(O,H) -> (B,H)：回传到隐藏层
    dZ1 = dH * (Z1 > 0).to(X.dtype)     # (B,H)：ReLU 局部导数是 0/1 掩码
    dW1 = X.T @ dZ1                     # (D,B)@(B,H) -> (D,H)
    db1 = dZ1.sum(dim=0)                # 沿 batch 维聚合 -> (H,)

    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }
    return loss, Y_hat, grads
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def mlp_forward_backward_numpy(
    X, Y, W1, b1, W2, b2,
):
    # X: (B,D), Y: (B,O)；W1: (D,H), W2: (H,O)，返回 loss / Y_hat / grads
    Z1 = X @ W1 + b1                    # (B, H)：b1 广播到每个样本
    H = np.maximum(Z1, 0.0)             # (B, H)：ReLU
    Y_hat = H @ W2 + b2                 # (B, O)

    B = X.shape[0]                      # batch size
    diff = Y_hat - Y                    # (B, O) 残差
    loss = 0.5 * np.sum(diff ** 2) / B

    dY = diff / B                       # (B, O)：dL/dY_hat
    dW2 = H.T @ dY                      # (H,B)@(B,O) -> (H,O)
    db2 = dY.sum(axis=0)                # 沿 batch 聚合 -> (O,)

    dH = dY @ W2.T                      # (B,O)@(O,H) -> (B,H)
    dZ1 = dH * (Z1 > 0)                 # bool 掩码广播相乘：Z1<=0 处梯度归零
    dW1 = X.T @ dZ1                     # (D,B)@(B,H) -> (D,H)
    db1 = dZ1.sum(axis=0)               # 沿 batch 聚合 -> (H,)

    grads = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }
    return loss, Y_hat, grads
```

**最小可运行测试**

```python
# Demo
B, D, H, O = 4, 3, 5, 2   # batch / 输入维 / 隐藏维 / 输出维
X = torch.randn(B, D)     # (4,3) 输入
Y = torch.randn(B, O)     # (4,2) 目标
W1 = torch.randn(D, H)    # (3,5)
b1 = torch.zeros(H)       # (5,)
W2 = torch.randn(H, O)    # (5,2)
b2 = torch.zeros(O)       # (2,)

loss, pred, grads = mlp_forward_backward(
    X, Y, W1, b1, W2, b2
)
print("loss:", loss.item())
print("dW1:", grads["W1"].shape)   # 应为 (3,5)，与 W1 同 shape
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def mlp_forward_backward(
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
):
    # 目标：一次前向 (X:(B,D) -> Y_hat:(B,O)) + 手写反向，返回 loss 与四个参数梯度
    pass
```

**Step 1：Forward**

（i）公式

$$
Z_1=XW_1+b_1,
\qquad
H=\operatorname{ReLU}(Z_1),
\qquad
\hat Y=HW_2+b_2
$$

（ii）代码

```python
Z1 = X @ W1 + b1      # (B,D)@(D,H) -> (B,H)；b1:(H,) 沿 batch 广播
H = torch.relu(Z1)    # (B,H) 隐藏激活
Y_hat = H @ W2 + b2   # (B,H)@(H,O) -> (B,O) 预测
```

（iii）`b1: (H,)` 广播到 `(B,H)`；`b2: (O,)` 广播到 `(B,O)`。

**Step 2：从 MSE 得到 $$d\hat Y$$**

（i）公式

$$
L
=
\frac{1}{2B}\sum_{b,o}
(\hat Y_{bo}-Y_{bo})^2
$$

$$
\frac{\partial L}{\partial \hat Y}
=
\frac{\hat Y-Y}{B}
$$

（ii）代码

```python
diff = Y_hat - Y   # (B,O) 残差
dY = diff / B      # dL/dY_hat：除以 B 对应 loss 按样本平均
```

（iii）前面的 $$1/2$$ 与平方求导产生的 2 抵消。

**Step 3：第二层 Linear**

（i）公式

$$
dW_2=H^\top d\hat Y,
\qquad
db_2=\sum_b d\hat Y_b
$$

（ii）代码

```python
dW2 = H.T @ dY        # (H,B)@(B,O) -> (H,O)，与 W2 同 shape
db2 = dY.sum(dim=0)   # 沿 batch 维聚合 -> (O,)
```

（iii）`(H,B) @ (B,O) -> (H,O)`，与 `W2` shape 完全一致。

**Step 4：穿过 ReLU**

（i）公式

$$
dH=d\hat YW_2^\top
$$

$$
dZ_1=dH\odot\mathbb{1}[Z_1>0]
$$

（ii）代码

```python
dH = dY @ W2.T                    # (B,O)@(O,H) -> (B,H)
dZ1 = dH * (Z1 > 0).to(X.dtype)   # 0/1 掩码：Z1<=0 的位置梯度截为 0
```

（iii）ReLU 在 $$Z_1\le0$$ 的位置梯度为 0。

**Step 5：第一层 Linear**

（i）公式

$$
dW_1=X^\top dZ_1,
\qquad
db_1=\sum_b dZ_{1,b}
$$

（ii）代码

```python
dW1 = X.T @ dZ1        # (D,B)@(B,H) -> (D,H)
db1 = dZ1.sum(dim=0)   # 沿 batch 维聚合 -> (H,)
```

（iii）这是完整链式法则终点；若继续求 $$dX$$，则 `dX = dZ1 @ W1.T`。

#### S4. Decoder-Only Transformer Block（Generic Pre-Norm）

##### 1️⃣ Dense Opening

Decoder-only block 的最小计算图是 causal self-attention + FFN，两者都包 residual；现代训练通常采用 Pre-Norm。核心对象是因果 attention 分数矩阵与两次 residual update。该题考察能否把 MHA、causal mask、LayerNorm 和 FFN 拼成一个正确 block，而非单独背算子。[参考 2, 10]

##### 2️⃣ Global Equations

Pre-Norm decoder block：

$$
X_1
=
X+\operatorname{MHA}
\left(
\operatorname{LN}(X)
\right)
$$

$$
X_2
=
X_1+\operatorname{FFN}
\left(
\operatorname{LN}(X_1)
\right)
$$

causal attention：

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}
\right)
$$

$$
M_{ij}
=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准面试实现）**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()
        assert d_model % num_heads == 0     # 保证 D 能被 H 整除，才能均分 head
        self.H = num_heads                  # head 数
        self.dh = d_model // num_heads      # 每个 head 的维度 d_h
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model)    # attention 前的 Pre-Norm
        self.ln2 = nn.LayerNorm(d_model)    # FFN 前的 Pre-Norm

        self.qkv = nn.Linear(d_model, 3 * d_model)   # 一次投影出 Q/K/V，D -> 3D
        self.out_proj = nn.Linear(d_model, d_model)  # 合头后的输出投影

        self.fc1 = nn.Linear(d_model, d_ff)  # FFN 升维 D -> d_ff
        self.fc2 = nn.Linear(d_ff, d_model)  # FFN 降维 d_ff -> D

    def _attention(self, x):
        # x: (B, T, D) 已归一化的输入；返回 (B, T, D)
        B, T, D = x.shape

        # 投影得 (B,T,3D)，沿最后一维等分成三份 -> 三个 (B,T,D)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # (B,T,D) -> (B,T,H,dh) -> (B,H,T,dh)：把 head 拆成独立 batch 维
        def split(z):
            return z.view(B, T, self.H, self.dh).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)

        # scores: (B,H,T,dh) @ (B,H,dh,T) -> (B,H,T,T)
        # 每个 query 对所有 key 的相似度；除 sqrt(dh) 防 softmax 饱和
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.dh)

        # causal: (T,T) bool，diagonal=1 取严格上三角，即未来位置 j>i 记 True
        causal = torch.triu(
            torch.ones(
                T, T,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal[None, None],           # (T,T) -> (1,1,T,T)，广播到所有 batch/head
            float("-inf"),                # 未来 token 置 -inf，softmax 后权重为 0
        )

        attn = F.softmax(scores, dim=-1)  # 沿 key 维归一化，每行和为 1
        out = torch.matmul(attn, v)       # (B,H,T,T)@(B,H,T,dh) -> (B,H,T,dh)

        # 合头: (B,H,T,dh) -> (B,T,H,dh) -> (B,T,D)；contiguous 保证 view 可用
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, D)
        return self.out_proj(out)

    def forward(self, x):
        # x: (B,T,D)；Pre-Norm：先 LN 再进子层，输出与输入残差相加
        x = x + self._attention(self.ln1(x))
        x = x + self.fc2(                 # FFN：D -> d_ff -> D，只在特征维变换
            F.gelu(self.fc1(self.ln2(x)))
        )
        return x
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: (..., D)，仅在最后一维做统计；gamma/beta: (D,) 广播
    mean = x.mean(axis=-1, keepdims=True)   # (...,1)，keepdims 便于广播回原 shape
    var = ((x - mean) ** 2).mean(
        axis=-1,
        keepdims=True,
    )                                       # (...,1)
    x_hat = (x - mean) / np.sqrt(var + eps)  # eps 防除零
    return gamma * x_hat + beta

def gelu(x):
    # 精确 GELU：0.5x(1+erf(x/√2))，逐元素运算不改变 shape
    return 0.5 * x * (
        1.0 + np.vectorize(np.math.erf)(
            x / np.sqrt(2.0)
        )
    )

def decoder_block_numpy(x, p, H):
    # x: (B,T,D)；p: 参数字典；H: head 数。返回 (B,T,D)
    B, T, D = x.shape
    dh = D // H                             # 每个 head 的维度

    h = layer_norm(
        x, p["g1"], p["b1"]
    )                                       # (B,T,D)
    qkv = h @ p["Wqkv"] + p["bqkv"]        # (B,T,D)@(D,3D) -> (B,T,3D)
    q, k, v = np.split(qkv, 3, axis=-1)    # 沿最后一维三等分 -> 各 (B,T,D)

    # (B,T,D) -> (B,T,H,dh) -> (B,H,T,dh)：head 维前置以便批量 matmul
    def split(z):
        return z.reshape(
            B, T, H, dh
        ).transpose(0, 2, 1, 3)

    q, k, v = split(q), split(k), split(v)

    scores = (
        q @ k.transpose(0, 1, 3, 2)         # (B,H,T,dh)@(B,H,dh,T)
    ) / np.sqrt(dh)                         # (B,H,T,T)

    causal = np.triu(
        np.ones((T, T), dtype=bool),
        k=1,                                # 严格上三角为 True，标记未来位置
    )
    scores = np.where(
        causal[None, None],                 # (T,T) -> (1,1,T,T) 广播到 batch/head
        -1e9,                               # 用大负数替代 -inf，避免 exp 出 NaN
        scores,
    )
    scores = scores - scores.max(           # 逐行减最大值，softmax 数值稳定技巧
        axis=-1,
        keepdims=True,
    )
    a = np.exp(scores)
    a = a / a.sum(axis=-1, keepdims=True)   # 沿 key 维归一化 -> (B,H,T,T)

    out = a @ v                             # (B,H,T,T)@(B,H,T,dh) -> (B,H,T,dh)
    out = out.transpose(
        0, 2, 1, 3
    ).reshape(B, T, D)                      # 合头 (B,H,T,dh) -> (B,T,D)
    x = x + out @ p["Wo"] + p["bo"]        # 输出投影 + attention residual

    h = layer_norm(
        x, p["g2"], p["b2"]
    )
    h = gelu(h @ p["W1"] + p["bf1"])       # (B,T,D) -> (B,T,d_ff)
    return x + h @ p["W2"] + p["bf2"]      # 降回 (B,T,D) 并做 FFN residual
```

**最小可运行测试**

```python
# Demo
block = DecoderBlock(
    d_model=32,
    num_heads=4,
    d_ff=128,
)
x = torch.randn(2, 6, 32)  # (B=2 序列数, T=6 token 数, D=32 特征维)
y = block(x)
print("output:", y.shape)  # (2, 6, 32)：block 不改变 shape
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()
        # 需要准备：ln1/ln2、qkv 投影(D->3D)、out_proj(D->D)、fc1(D->d_ff)、fc2(d_ff->D)
        pass

    def forward(self, x):
        # x: (B,T,D) -> 返回 (B,T,D)：Pre-Norm 的 attention residual + FFN residual
        pass
```

**Step 1：第一次 Pre-Norm**

（i）公式

$$
\tilde X=\operatorname{LN}(X)
$$

（ii）代码

```python
h = self.ln1(x)   # (B,T,D) -> (B,T,D)：逐 token 在特征维归一化
```

（iii）LayerNorm 逐 token 在最后一维 $$D$$ 上归一化；shape 保持 `(B,T,D)`。

**Step 2：QKV 投影并切 head**

（i）shape

$$
(B,T,D)
\rightarrow
(B,T,3D)
\rightarrow
3\times(B,H,T,d_h)
$$

（ii）代码

```python
q, k, v = self.qkv(h).chunk(3, dim=-1)    # (B,T,3D) 沿末维三等分 -> 各 (B,T,D)
q = q.view(B, T, H, dh).transpose(1, 2)   # (B,T,D) -> (B,T,H,dh) -> (B,H,T,dh)
```

（iii）head 维移到 sequence 前，便于四维 batched matmul。

**Step 3：Causal attention**

（i）公式

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}
$$

（ii）代码

```python
scores = torch.matmul(
    q, k.transpose(-2, -1)     # (B,H,T,dh)@(B,H,dh,T) -> (B,H,T,T)
) / math.sqrt(self.dh)         # 缩放，避免 dh 大时点积方差过大

scores = scores.masked_fill(
    causal[None, None],        # (T,T) -> (1,1,T,T) 广播到全部 batch/head
    float("-inf"),             # 未来位置屏蔽，softmax 后概率为 0
)
```

（iii）`scores: (B,H,T,T)`；每一行 query 只能看到自身和过去 token。

**Step 4：Attention residual**

（i）公式

$$
X_1=X+\operatorname{Attention}(\operatorname{LN}(X))
$$

（ii）代码

```python
x = x + self._attention(self.ln1(x))   # 残差两端均为 (B,T,D)，梯度可直通
```

（iii）residual 两端 shape 都必须是 `(B,T,D)`。

**Step 5：FFN residual**

（i）公式

$$
X_2
=
X_1
+
W_2\operatorname{GELU}
\left(
W_1\operatorname{LN}(X_1)
\right)
$$

（ii）代码

```python
x = x + self.fc2(                  # d_ff -> D，恢复到残差可相加的 shape
    F.gelu(self.fc1(self.ln2(x)))  # (B,T,D) -> (B,T,d_ff)，逐元素 GELU
)
```

（iii）FFN 只在最后一维变换：`D -> d_ff -> D`，不会混合 token 维。

### S−：现代 LLM 高频

> 本章分两部分：**第一部分**为主文档原有题目，**第二部分**为《补充文档-LLM面试手撕代码补充题单》新增题目。

#### ▍第一部分 · 主文档原有题目

#### 👀 RMSNorm（Root Mean Square Norm）

LayerNorm 的简化：去掉均值中心化，只按均方根（RMS, Root Mean Square）缩放。LLaMA、Mistral 等现代 LLM 广泛使用。

$$
\text{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon},\qquad
 y=\gamma\odot\frac{x}{\text{RMS}(x)}
$$

注意：标准 RMSNorm 把 $$\epsilon$$ 放在平方根内部。

#### （A）PyTorch 版本

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # 逐维缩放 γ，(d_model,)
        self.eps = eps                                   # 防除零，放在 sqrt 内部

    def forward(self, x):
        # x: (..., d_model)；半精度下先用 float32 做平方/mean/rsqrt，更稳，再 cast 回输入 dtype
        dtype = x.dtype
        x_f = x.float()
        rms_inv = torch.rsqrt(x_f.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_f * rms_inv).to(dtype)  # γ:(D,) 广播到 (...,D)，shape 不变
```

#### （B）NumPy 版本

```python
import numpy as np

def rms_norm(x, gamma, eps=1e-6):
    # x: (..., D)；gamma: (D,) 可学习缩放。输出与 x 同 shape
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)  # (...,1)，keepdims 便于广播
    return gamma * x / rms   # 无均值中心化、无 bias，仅按 magnitude 归一

# Demo
x = np.random.randn(2, 4, 8)   # (B=2, T=4, D=8)
gamma = np.ones(8)             # (D,)：初始化为 1 即恒等缩放
print(rms_norm(x, gamma).shape)
```

#### Step-by-step

```python
# x: (...,D) -> 同 shape；三步：均方 -> 逆 RMS -> 乘 gamma
def rmsnorm(x, gamma, eps=1e-6): pass
```

（i）均方：$$m_2=\frac{1}{d}\sum_i x_i^2$$
（ii）`x_f = x.float(); mean_sq = x_f.square().mean(-1, keepdim=True)`
（iii）只度量 magnitude，不减均值；半精度输入先提升到 `float32` 做 reduction，可减少平方溢出和累计误差。

（i）逆 RMS：$$1/\sqrt{m_2+\epsilon}$$
（ii）`rms_inv = torch.rsqrt(mean_sq + eps)`
（iii）`rsqrt` 直接得到逆平方根，代码比 `sqrt` 后再除更短。

（i）可学习缩放：$$y=\gamma\odot x\cdot\text{rms\_inv}$$
（ii）`return gamma * (x_f * rms_inv).to(dtype)`
（iii）无 bias；核心参数只有逐维缩放 $$\gamma$$，归一化结果最后 cast 回输入 dtype。

#### 👀 GQA（Grouped-Query Attention）

MHA 与 MQA 的折中：H 个 Q-head 只配 G 个 K/V-head，多组 Query 共享同一组 K/V。$$G=H$$ 退化为 MHA；$$G=1$$ 退化为 MQA。

$$
K,V\in\mathbb{R}^{B\times G\times T\times d_k},\qquad
\text{每个 KV head 对应 } H/G \text{ 个 Q head}
$$

#### （A）PyTorch 版本

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0   # H 必须能被 G 整除，才能均匀分组
        self.num_heads = num_heads             # H：Query head 数
        self.num_kv_heads = num_kv_heads       # G：KV head 数（G<H 即共享）
        self.head_dim = d_model // num_heads   # d_k
        self.repeat = num_heads // num_kv_heads  # 每个 KV head 服务多少个 Q head
        self.Wq = nn.Linear(d_model, d_model)                       # D -> H*d_k
        self.Wk = nn.Linear(d_model, num_kv_heads * self.head_dim)  # D -> G*d_k
        self.Wv = nn.Linear(d_model, num_kv_heads * self.head_dim)  # KV 参数量随 G 缩小
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B,T,D) -> 返回 (B,T,D)
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # -> (B,H,T,d_k)
        K = self.Wk(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # -> (B,G,T,d_k)
        V = self.Wv(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # -> (B,G,T,d_k)
        # (B,G,T,d_k) -> (B,H,T,d_k)：每个 KV head 连续复制 repeat 次，与 Q head 顺序对齐
        K = K.repeat_interleave(self.repeat, dim=1)
        V = V.repeat_interleave(self.repeat, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        attn = F.softmax(scores, dim=-1)               # 沿 key 维归一化
        out = torch.matmul(attn, V)                    # (B,H,T,T)@(B,H,T,d_k) -> (B,H,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # 合头 -> (B,T,H*d_k)=(B,T,D)
        return self.Wo(out)                            # (B,T,D)
```

#### （B）NumPy 版本

```python
import numpy as np

def gqa_numpy(X, Wq, Wk, Wv, Wo, H, G):
    # X: (B,T,D)；Wq/Wo: (D,D)，Wk/Wv: (D,G*d_k)；H: Q head 数，G: KV head 数
    B, T, D = X.shape
    d_k, rep = D // H, H // G     # d_k 每头维度；rep 每个 KV head 被几个 Q head 共享
    Q = (X @ Wq).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)   # -> (B,H,T,d_k)
    K = (X @ Wk).reshape(B, T, G, d_k).transpose(0, 2, 1, 3)   # -> (B,G,T,d_k)
    V = (X @ Wv).reshape(B, T, G, d_k).transpose(0, 2, 1, 3)   # -> (B,G,T,d_k)
    K = np.repeat(K, rep, axis=1)                     # 沿 head 维逐个复制 -> (B,H,T,d_k)
    V = np.repeat(V, rep, axis=1)

    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)   # (B,H,T,d_k)@(B,H,d_k,T) -> (B,H,T,T)
    a = scores - scores.max(-1, keepdims=True)            # 逐行减最大值，softmax 数值稳定
    attn = np.exp(a) / np.exp(a).sum(-1, keepdims=True)   # 沿 key 维归一化
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, D)  # (B,H,T,d_k) -> 合头 (B,T,D)
    return out @ Wo

# Demo
B, T, D, H, G = 2, 5, 16, 4, 2   # 4 个 Q head 共享 2 个 KV head（每组 2 个）
X = np.random.randn(B, T, D)
Wq = np.random.randn(D, D) * 0.1
Wk = np.random.randn(D, G * (D // H)) * 0.1   # (16, 2*4=8)
Wv = np.random.randn(D, G * (D // H)) * 0.1
Wo = np.random.randn(D, D) * 0.1
print(gqa_numpy(X, Wq, Wk, Wv, Wo, H, G).shape)   # (2,5,16)
```

#### Step-by-step

```python
# x: (B,T,D) -> (B,T,D)；关键在 K/V 只投影出 G 个 head，再沿 head 维复制到 H 个
def gqa(x): pass
```

（i）Q 有 H 个头，KV 只有 G 个头：`Wq: D->D`，`Wk/Wv: D->G*d_k`。

（ii）KV 沿 head 维复制 $$H/G$$ 次：`K = K.repeat_interleave(repeat, dim=1)`，shape `(B,G,T,d_k) -> (B,H,T,d_k)`。

（iii）后续 attention 与 MHA 相同：`scores -> softmax -> @V`。

（iv）合并 heads 后必须经过 output projection：`return self.Wo(out)`。公开面试题库的 MHA/GQA/MQA 标准实现通常都保留这一步。

（v）模型例子要写准确：LLaMA-2 70B、Mistral、LLaMA 3 等采用 GQA；并非所有 LLaMA-2 尺寸都使用 GQA。

#### 👀 LoRA（Low-Rank Adaptation）

冻结原权重 $$W$$，新增低秩增量。本文采用行向量输入（row-vector）记法：$$\Delta W = \frac{\alpha}{r}AB$$，其中 $$A \in \mathbb{R}^{d_{\text{in}}\times r}$$、$$B \in \mathbb{R}^{r\times d_{\text{out}}}$$；与论文/常见 PyTorch 权重记法中的 $$BA$$ 只是矩阵存储方向不同。

$$
h = xW + \frac{\alpha}{r}\, x A B,\quad A \sim \mathcal{N}(0,\sigma^2),\ B = 0 \text{（初始时增量为 0）}
$$

$$r \ll \min(d_{\text{in}}, d_{\text{out}})$$ 时参数量从 $$d_{\text{in}} d_{\text{out}}$$ 降到 $$(d_{\text{in}}+d_{\text{out}}) r$$。

#### （A）PyTorch 版本

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        for p in self.linear.parameters(): p.requires_grad = False  # 冻结主干
        # 低秩矩阵: A: (in_dim, r)   B: (r, out_dim)，r << min(in_dim,out_dim)
        self.A = nn.Parameter(torch.randn(in_dim, r) * 0.01)        # 小随机初始化，打破对称
        self.B = nn.Parameter(torch.zeros(r, out_dim))              # B=0 保证起点等价于原模型
        self.scale = alpha / r   # 缩放系数，使不同 r 下增量幅度与学习率可比

    def forward(self, x):
        # x: (..., in_dim) -> (..., out_dim)
        # 冻结主干输出 + 低秩增量：x@A 先降到 r 维，再 @B 升回 out_dim
        return self.linear(x) + self.scale * (x @ self.A @ self.B)
```

#### （B）NumPy 版本（推理前向）

```python
import numpy as np

def lora_forward(x, W, A, B, alpha, r):
    # x: (..., in)   W: (in, out)   A: (in, r)   B: (r, out) -> 输出 (..., out)
    return x @ W + (alpha / r) * (x @ A @ B)   # 先降到 r 维再升回，避免构造 (in,out) 大矩阵

# Demo
in_d, out_d, r = 16, 32, 4     # 秩 r=4，增量参数量 (16+32)*4 远小于 16*32
x = np.random.randn(2, 5, in_d)   # (B=2, T=5, in=16)
W = np.random.randn(in_d, out_d) * 0.1
A = np.random.randn(in_d, r) * 0.01
B = np.zeros((r, out_d))          # B=0 -> 初始增量为 0，输出等同原模型
print(lora_forward(x, W, A, B, alpha=8, r=r).shape)   # (2,5,32)
```

#### Step-by-step

```python
class LoRALinear(nn.Module):
    # 需要：冻结的 nn.Linear(in_dim,out_dim) + 可训练 A:(in_dim,r)、B:(r,out_dim) + scale=alpha/r
    def __init__(self, in_dim, out_dim, r, alpha): pass
```

（i）主干冻结 + 低秩可训练
（ii）`A = nn.Parameter(torch.randn(in_dim, r) * 0.01); B = nn.Parameter(torch.zeros(r, out_dim))`
（iii）`B = 0` 让初始 $$\Delta W = 0$$，训练起点等于原模型。

（i）增量 $$\Delta h = \frac{\alpha}{r} x A B$$
（ii）`lora = x @ self.A @ self.B`
（iii）低秩分支只新增两次小矩阵乘；冻结的主干 $$xW$$ 仍保留。可训练参数量为 $$O((d_{in}+d_{out})r)$$。

（i）尺度因子 $$\alpha/r$$
（ii）`return self.linear(x) + self.scale * lora`
（iii）让不同 $$r$$ 下学习率/初始幅度相对一致，便于超参迁移。

#### 👀 LLM Decode（Greedy / Beam Search / Top-K / Top-p）

给定当前前缀，模型输出下一 token 的 logits。Greedy / Top-k / Top-p 只决定“一步怎么选”；Beam Search 需要对每个候选序列重新得到条件分布，因此 logits 必须依赖当前 `seq`。

$$
p_t=\text{softmax}(\text{logits}_t/\tau)
$$

温度 $$\tau>1$$ 更平坦，$$\tau<1$$ 更尖锐。

#### （A）PyTorch 版本（四种合一）

```python
import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> int:
    # logits: (V,) 全词表未归一化分数 -> 返回得分最高的 token id（标量）
    return logits.argmax().item()


def top_k_sampling(logits: torch.Tensor,
                   k: int = 5,
                   temperature: float = 1.0) -> int:
    # 只对 top-k logits 做 softmax；与全词表 softmax 后再截断归一化等价
    # temperature<1 更尖锐、>1 更平坦
    top_logits, top_idx = torch.topk(logits / temperature, k)  # 值 (k,)；top_idx (k,) 是词表 id
    probs = F.softmax(top_logits, dim=-1)   # (k,)，在候选集内重新归一化
    pick = torch.multinomial(probs, 1)      # (1,)：候选集内的局部下标，非词表 id
    return top_idx[pick].item()             # 局部下标映射回词表 id


def top_p_sampling(logits: torch.Tensor,
                   p: float = 0.9,
                   temperature: float = 1.0) -> int:
    probs = F.softmax(logits / temperature, dim=-1)     # (V,) 全词表概率
    sorted_p, sorted_idx = probs.sort(descending=True)  # 各 (V,)；sorted_idx: 排名 -> 词表 id
    cum = sorted_p.cumsum(-1)                           # (V,) 概率前缀和，单调递增

    # 保留“第一个使累计概率 >= p”的 token；其后的才删除
    remove = cum > p                # (V,) bool 掩码
    remove[1:] = remove[:-1].clone()  # 掩码整体右移一位，把临界 token 留在核内
    remove[0] = False                 # 概率最大的 token 永不删除，防止候选集为空
    sorted_p[remove] = 0.0            # 核外概率置零
    sorted_p /= sorted_p.sum()        # 核内重新归一化为合法分布

    pick = torch.multinomial(sorted_p, 1)   # (1,)：排序后的名次下标
    return sorted_idx[pick].item()          # 名次映射回词表 id


def beam_search(next_log_probs_fn,
                start_token: int,
                beam_width: int = 3,
                max_new_tokens: int = 10,
                eos_token: int = 9):
    # next_log_probs_fn(seq) -> (V,) log 概率；必须依赖各自候选 seq
    beams = [([start_token], 0.0)]   # 每个 beam 是 (token 序列, 累计 log 概率)

    for _ in range(max_new_tokens):
        candidates = []              # 本轮所有扩展结果，规模最多 beam_width^2
        for seq, score in beams:
            if seq[-1] == eos_token:
                candidates.append((seq, score))        # 已结束 beam 原样保留
                continue

            log_probs = next_log_probs_fn(seq)         # (V,)
            top_lp, top_idx = log_probs.topk(beam_width)  # 各 (beam_width,)：值与词表 id
            for lp, idx in zip(top_lp, top_idx):
                candidates.append(
                    (seq + [idx.item()], score + lp.item())  # log 概率相加 = 联合概率相乘
                )

        candidates.sort(key=lambda x: x[1], reverse=True)  # 按累计 log 概率降序
        beams = candidates[:beam_width]                    # 剪枝，只留前 beam_width 条
        if all(seq[-1] == eos_token for seq, _ in beams):
            break

    return beams
```

#### 👀 （A-补充）Top-p 面试快速版（Hot-100：`cumsum <= p`）

下面保留一份网上 Hot-100 题库常见的 **Top-p 快速手写版**。它的优点是代码非常短，面试现场容易迅速写出；需要明确它与上面的严格 nucleus sampling 版本存在一个边界语义差异：`cumsum <= p` 只保留“累计概率仍未超过 p”的 token，因此会丢掉**第一个把累计概率推过 p 的临界 token**。例如排序后概率为 `[0.5, 0.3, 0.2]`、`p=0.7` 时，这个快速版只保留 `0.5`；上面的严格版会保留 `0.5 + 0.3`，从而得到累计概率至少达到 `p` 的最小候选集合。

```python
def top_p_sampling_simple_hot100(logits: torch.Tensor,
                                 p: float = 0.9,
                                 temperature: float = 1.0) -> int:
    """
    Top-p 面试快速手写版：核心就是 `cumsum <= p`。

    Args:
        logits: (V,) 当前一步对整个词表的未归一化 logits。
        p: nucleus 阈值，常见如 0.9 / 0.95。
        temperature: 温度；<1 让分布更尖锐，>1 让分布更平坦。

    Returns:
        采样得到的 token id（Python int）。

    面试时要主动说明：
        这个版本追求“短、容易写”。`keep = cumsum <= p` 会排除第一个
        使累计概率超过 p 的临界 token，因此和严格的 nucleus sampling
        定义有轻微边界差异。严格版本见上面的 `top_p_sampling`。
    """

    # Step 1：先做 temperature scaling，再按 logits 从大到小排序。
    # softmax 是单调的，因此按 logits 排序与按最终概率排序的顺序一致；
    # 直接排序 logits 可以少写一次“先 softmax、再排序概率”的中间变量。
    sorted_logits, sorted_idx = torch.sort(
        logits / temperature, descending=True
    )  # sorted_logits: (V,)，sorted_idx: (V,)；排名 -> 原词表 token id

    # Step 2：只在“排序后的 logits”上做 softmax，得到从大到小排列的概率。
    # probs[i] 与 sorted_idx[i] 一一对应，后面采样到的是排序空间里的位置，
    # 最终还需要借助 sorted_idx 映射回原词表 id。
    probs = F.softmax(sorted_logits, dim=-1)            # (V,)，sum(probs)=1

    # Step 3：计算前缀累计概率。
    # 例如 probs = [0.50, 0.30, 0.15, 0.05]
    # 则 cumsum = [0.50, 0.80, 0.95, 1.00]。
    cumsum = torch.cumsum(probs, dim=-1)                # (V,)，单调递增

    # Step 4：Hot-100 快速版的核心写法。
    # 只保留累计概率 <= p 的位置。
    # 注意：如果某个 token 让累计概率第一次从 <p 变成 >p，
    # 那么它在这里会被 False 掉；这正是它与上面严格版本的差异。
    keep = cumsum <= p                                  # (V,) bool

    # 极端情况下，最大概率本身就 > p，此时上面的 keep 会全 False。
    # 强制保留排序后的第 0 个 token，保证候选集合永远非空，
    # 也避免后续归一化出现 0/0 -> NaN。
    keep[0] = True

    # Step 5：把核外概率直接乘成 0。
    # keep.to(probs.dtype) 只是把 bool mask 转成 0/1，方便与概率逐元素相乘。
    filtered_probs = probs * keep.to(probs.dtype)       # (V,)，核外位置为 0

    # 截断后概率和通常 < 1，因此必须重新归一化。
    # 经过 keep[0] = True 后，分母一定 > 0。
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Step 6：在“排序后的候选空间”采样一个局部下标。
    # sampled_rank 是概率排名位置，并不是原始词表 token id。
    sampled_rank = torch.multinomial(filtered_probs, 1) # (1,)

    # sorted_idx[rank] 才是原词表 id；最后转成 Python int，
    # 与上面正式版 `top_p_sampling` 的返回类型保持一致。
    return sorted_idx[sampled_rank].item()
```

> **记忆方式**：快速版可记成 `sort -> softmax -> cumsum -> <= p -> 至少留 1 个 -> renorm -> sample`。如果面试官追问“标准 Top-p 是否应包含跨过阈值的那个 token”，应回答：严格 nucleus sampling 通常保留该临界 token，因此使用上面正式版的“`cum > p` 后右移删除 mask”更准确。

#### （B）NumPy 版本

```python
import numpy as np


def _softmax(x):
    # x: (n,) 一维 logits；返回同 shape 的概率
    x = x - x.max()                                    # 减最大值防 exp 溢出，不改变 softmax 结果
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()                         # 沿唯一维归一化，和为 1


def greedy(logits):
    # logits: (V,) 词表打分；返回概率最大的 token id
    return int(np.argmax(logits))


def top_k_sampling(logits, k=5, temperature=1.0):
    scaled = logits / temperature                      # temperature<1 拉大差距更确定，>1 更随机
    idx = np.argsort(scaled)[-k:]                      # 升序排序取末 k 个：(V,) -> (k,) 的 top-k 下标
    probs = _softmax(scaled[idx])                      # 只在 k 个候选上重新归一化：(k,)
    return int(np.random.choice(idx, p=probs))         # 按 probs 从候选下标里抽一个 token id


def top_p_sampling(logits, p=0.9, temperature=1.0):
    probs = _softmax(logits / temperature)             # (V,) 全词表概率
    order = np.argsort(-probs)                         # 取负数实现降序：(V,) 概率从大到小的下标
    sorted_p = probs[order]                            # 按 order 重排概率，与 order 逐位对齐
    cum = np.cumsum(sorted_p)                          # 前缀累计概率，cum[i]=前 i+1 个候选的概率和

    remove = cum > p                                   # 累计已超过 p 的位置标记为待丢弃
    remove[1:] = remove[:-1]                           # 右移一位：保留“刚把累计推过 p 的那个 token”
    remove[0] = False                                  # 至少保留概率最高的 token
    sorted_p[remove] = 0.0                             # 掩掉核外候选（概率置 0）
    sorted_p /= sorted_p.sum()                         # 在 nucleus 内重新归一化
    return int(np.random.choice(order, p=sorted_p))    # 采样下标空间是 order，概率是对齐后的 sorted_p


def beam_search(next_log_probs_fn, start_token,
                beam_width=3, max_new_tokens=10, eos_token=9):
    # beams: [(token 序列, 累计 log-prob)]，用 log 累加代替概率连乘以防下溢
    beams = [([start_token], 0.0)]
    for _ in range(max_new_tokens):
        candidates = []                                 # 本轮所有 beam 扩展出的候选序列
        for seq, score in beams:
            if seq[-1] == eos_token:
                candidates.append((seq, score))         # 已结束的 beam 原样保留，不再扩展
                continue
            log_probs = next_log_probs_fn(seq)          # 每个 beam 独立前向：(V,) 下一 token 的 log-prob
            idx = np.argsort(log_probs)[-beam_width:][::-1]   # 升序末 w 个再反转：(V,) -> (w,) 降序 top-w 下标
            for j in idx:
                candidates.append((seq + [int(j)], score + float(log_probs[j])))   # 序列加长 1，score 累加
        candidates.sort(key=lambda x: x[1], reverse=True)     # 全局按累计 log-prob 降序
        beams = candidates[:beam_width]                 # 剪枝回 beam_width 条
        if all(seq[-1] == eos_token for seq, _ in beams):
            break
    return beams

# Demo：一步采样
np.random.seed(0)
logits = np.random.randn(10)                            # (10,) 模拟词表大小 V=10 的一步 logits
print('greedy:', greedy(logits))
print('top-k :', top_k_sampling(logits, k=3))
print('top-p :', top_p_sampling(logits, p=0.8))
```

#### Step-by-step（Top-p 最容易写错）

```python
# logits: (V,) 或 (B,V)；p 为核概率阈值；返回采样得到的 token id
def top_p(logits, p, temperature): pass
```

（i）`probs = F.softmax(logits / temperature, dim=-1)`。

（ii）降序排序并累计：`sorted_p, sorted_idx = probs.sort(descending=True); cum = sorted_p.cumsum(-1)`。

（iii）关键截断：`remove = cum > p; remove[1:] = remove[:-1].clone(); remove[0] = False`。右移一位是为了保留“刚好把累计概率推过 p 的第一个 token”，符合 nucleus sampling 的最小候选集合定义。

（iv）归一化后 `torch.multinomial` 采样。

#### Step-by-step（Beam Search 易错点）

（i）维护 `beams = [(seq, cumulative_log_prob)]`，score 使用 log-prob 累加避免连乘下溢。

（ii）**每个 beam 必须独立调用模型**：`log_probs = next_log_probs_fn(seq)`。如果所有 beam 共用一份固定的 `logits_t`，就没有真正建模 $$P(x_t\mid x_{<t})$$，不是标准自回归 Beam Search。

（iii）每个活跃 beam 只扩展当前 top-`beam_width` token，再全局排序剪回 `beam_width`。

（iv）EOS beam 原样保留，不再继续扩展。长度惩罚（length penalty）属于常见扩展，面试基础版可先省略。

---

#### ▍第二部分 · 补充题单新增题目

#### 👀 S−1. Multi-Query Attention（MQA）

##### 1️⃣ Dense Opening

MQA 让所有 query heads 共享同一组 K/V，从而把 decode 阶段 KV cache 的 head 维从 $$H$$ 压到 1。核心对象仍是 scaled dot-product attention；结构变化只发生在 K/V projection 与 head broadcasting。面试重点是明确 $$Q:(B,H,T,d_h)$$，而 $$K,V:(B,1,T,d_h)$$。[参考 3]

##### 2️⃣ Global Equations

$$
Q=XW_Q
\in
\mathbb{R}^{B\times H\times T\times d_h}
$$

$$
K=XW_K,\quad
V=XW_V
\in
\mathbb{R}^{B\times 1\times T\times d_h}
$$

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)
$$

$$
O=AV
$$

K/V 在 head 维为 1，通过 broadcasting 与 $$H$$ 个 Q-head 计算 attention。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.H = num_heads                # query head 数
        self.dh = d_model // num_heads    # 每个 head 的维度
        self.D = d_model

        self.Wq = nn.Linear(d_model, d_model)     # Q 仍是 H 个 head：输出 H*dh = D
        self.Wk = nn.Linear(d_model, self.dh)     # MQA 关键：K 只输出 1 个 head 的 dh
        self.Wv = nn.Linear(d_model, self.dh)     # V 同上，decode 时 KV cache 的 head 维压到 1
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B,T,D)
        B, T, D = x.shape

        # Q: (B,T,D) -> (B,H,T,dh)：拆出 head 维并前移，便于 4D 并行 matmul
        q = self.Wq(x)
        q = q.view(
            B, T, self.H, self.dh
        ).transpose(1, 2)

        # K/V: (B,T,dh) -> (B,1,T,dh)：插入长度 1 的 head 维用于后续 broadcast
        k = self.Wk(x).unsqueeze(1)
        v = self.Wv(x).unsqueeze(1)

        # q:(B,H,T,dh) @ k^T:(B,1,dh,T) -> scores:(B,H,T,T)
        # k 的 head 维 1 自动 broadcast 到 H；除 sqrt(dh) 防 softmax 饱和
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.dh)

        attn = F.softmax(scores, dim=-1)  # 沿 key 维归一化：每个 query 对所有 key 的权重和为 1
        out = torch.matmul(attn, v)       # (B,H,T,T) @ (B,1,T,dh) -> (B,H,T,dh)：按权重聚合 value

        # (B,H,T,dh) -> (B,T,D)：head 维换回并展平，合头
        out = out.transpose(
            1, 2
        ).contiguous().view(B, T, D)
        return self.Wo(out)
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def mqa_numpy(
    x, Wq, Wk, Wv, Wo, num_heads,
):
    # x: (B,T,D)；Wq:(D,D)，Wk/Wv:(D,dh) 只出一个 head，Wo:(D,D)
    B, T, D = x.shape
    H = num_heads
    dh = D // H                           # 每个 head 的维度

    q = (x @ Wq).reshape(
        B, T, H, dh
    ).transpose(0, 2, 1, 3)               # (B,T,D) -> (B,T,H,dh) -> (B,H,T,dh)：拆头并把 head 维前移

    k = (x @ Wk)[:, None, :, :]           # (B,T,dh) -> (B,1,T,dh)：插入 head 维待 broadcast
    v = (x @ Wv)[:, None, :, :]           # (B,1,T,dh)

    scores = (
        q @ k.transpose(0, 1, 3, 2)       # k 转置为 (B,1,dh,T)，head 维 1 广播到 H
    ) / np.sqrt(dh)                        # (B,H,T,T)：每个 query 对所有 key 的缩放点积

    scores = scores - scores.max(          # 沿 key 维减最大值，防 exp 溢出
        axis=-1,
        keepdims=True,
    )
    attn = np.exp(scores)
    attn = attn / attn.sum(                # 沿 key 维归一化，得到 attention 权重 (B,H,T,T)
        axis=-1,
        keepdims=True,
    )

    out = attn @ v                         # (B,H,T,T) @ (B,1,T,dh) -> (B,H,T,dh)：加权聚合 value
    out = out.transpose(
        0, 2, 1, 3
    ).reshape(B, T, D)                     # (B,H,T,dh) -> (B,T,H,dh) -> (B,T,D)：合头
    return out @ Wo
```

**最小可运行测试**

```python
# Demo
mqa = MQA(d_model=32, num_heads=4)   # dh = 32/4 = 8，K/V 只有 1 个 head
x = torch.randn(2, 5, 32)            # (B=2, T=5, D=32)
print("output:", mqa(x).shape)       # 输出形状与输入一致 (2,5,32)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        # 待实现：Wq 输出 D 维（H 个 head），Wk/Wv 只输出 dh 维（1 个 head）
        super().__init__()
        pass

    def forward(self, x):
        # x: (B,T,D) -> 返回 (B,T,D)
        pass
```

**Step 1：Q 多头，K/V 单头**

（i）shape

$$
Q:(B,H,T,d_h),
\qquad
K,V:(B,1,T,d_h)
$$

（ii）代码

```python
q = self.Wq(x).view(          # (B,T,D) -> (B,T,H,dh) -> (B,H,T,dh)
    B, T, self.H, self.dh
).transpose(1, 2)

k = self.Wk(x).unsqueeze(1)   # (B,T,dh) -> (B,1,T,dh)：单 head K
v = self.Wv(x).unsqueeze(1)   # (B,T,dh) -> (B,1,T,dh)：单 head V
```

（iii）MQA 的核心就这一处：`Wk/Wv` 只输出一个 head 的维度。

**Step 2：Head broadcasting**

（i）公式

$$
S_{b,h}
=
\frac{
Q_{b,h}K_b^\top
}{
\sqrt{d_h}
}
$$

（ii）代码

```python
scores = torch.matmul(            # q:(B,H,T,dh) @ k^T:(B,1,dh,T) -> (B,H,T,T)
    q,
    k.transpose(-2, -1),          # 交换最后两维，把 K 变成 (B,1,dh,T)
) / math.sqrt(self.dh)            # 缩放防 softmax 饱和
```

（iii）`k` 的 head 维为 1，PyTorch 自动广播到 $$H$$；无需手动 `repeat`。

**Step 3：聚合并合头**

（i）公式

$$
O_h
=
\operatorname{softmax}(S_h)V
$$

（ii）代码

```python
out = torch.matmul(
    F.softmax(scores, dim=-1),    # 沿 key 维归一化：(B,H,T,T)
    v,                            # (B,1,T,dh) 广播到 H，聚合后得 (B,H,T,dh)
)
out = out.transpose(1, 2).reshape(B, T, D)   # (B,H,T,dh) -> (B,T,H,dh) -> (B,T,D)：合头
```

（iii）输出仍包含 $$H$$ 个 query head，因此最后仍合回 $$D=H d_h$$。

#### S−2. Multi-Head Latent Attention（MLA，面试简化版）

##### 1️⃣ Dense Opening

MLA 的核心不是“少几个 KV heads”，而是把 K/V 的内容信息先压进低维 latent $$c^{KV}$$，decode 时优先缓存这个 latent；需要 attention 时再通过 up-projection 恢复 per-head K/V 内容部分。DeepSeek-V2 还将位置部分与内容压缩解耦，使 RoPE 只作用于独立的 positional subspace。下面保留 **KV latent compression + decoupled RoPE** 这两个面试核心，不展开 kernel fusion、矩阵吸收与完整工程 cache layout。[参考 4]

##### 2️⃣ Global Equations

KV latent compression：

$$
c_t^{KV}
=
x_tW^{DKV}
\in
\mathbb{R}^{r}
$$

$$
k_{t,h}^{C}
=
c_t^{KV}W_{h}^{UK},
\qquad
v_{t,h}
=
c_t^{KV}W_{h}^{UV}
$$

独立位置 key：

$$
k_t^{R}
=
\operatorname{RoPE}
\left(
x_tW^{KR},
t
\right)
$$

query 拆为 content 与 position：

$$
q_{t,h}
=
\left[
q_{t,h}^{C};
\operatorname{RoPE}(q_{t,h}^{R},t)
\right]
$$

$$
k_{t,h}
=
\left[
k_{t,h}^{C};
k_t^{R}
\right]
$$

$$
\operatorname{Attn}_{t,h}
=
\operatorname{softmax}
\left(
\frac{
q_{t,h}k_{:,h}^{\top}
}{
\sqrt{d_c+d_r}
}
\right)
v_{:,h}
$$

面试 cache 核心：

$$
\text{Cache}
=
\left\{
c_{1:t}^{KV},
k_{1:t}^{R}
\right\}
$$

而不是直接缓存完整 per-head $$K,V$$。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（简化但计算图完整）**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rope_part(x, positions):
    # x: (B,H,T,d), d 为偶数；positions: (T,) 绝对位置
    # 返回: 同 shape；位置/频率用 float32 算角度，最后 cast 回 x.dtype
    d = x.size(-1)
    assert d % 2 == 0
    half = d // 2                 # 前后半各作为复数的实部/虚部，共 half 个旋转平面

    freq = 1.0 / (                # (half,) 每个旋转平面的角频率，10000 为 RoPE base
        10000.0
        ** (
            torch.arange(
                0, half,
                device=x.device,
                dtype=torch.float32,
            ) / half
        )
    )
    angle = positions.float()[:, None] * freq[None, :]       # (T,half)，避免 fp16 position/frequency 精度损失
    cos = angle.cos()[None, None, :, :].to(x.dtype)          # -> (1,1,T,half)，再广播到 B/H
    sin = angle.sin()[None, None, :, :].to(x.dtype)

    x1 = x[..., :half]            # (B,H,T,d) -> (B,H,T,half)：实部
    x2 = x[..., half:]            # (B,H,T,half)：虚部
    return torch.cat(             # 复数乘法后沿最后一维拼回 (B,H,T,d)
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        dim=-1,
    )

class SimpleMLA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        latent_dim,
        content_dim,
        rope_dim,
    ):
        super().__init__()
        assert rope_dim % 2 == 0   # RoPE 需要成对旋转，故位置子空间维度必须为偶数
        self.H = num_heads
        self.dc = content_dim      # 每个 head 的 content 子空间维度
        self.dr = rope_dim         # 位置子空间维度（所有 head 共享同一份 k_r）
        self.dv = content_dim      # V 的 per-head 维度，这里与 dc 取相同
        self.D = d_model

        # Q 仍按 head 输出 content + rope 两部分
        self.Wq = nn.Linear(
            d_model,
            num_heads * (content_dim + rope_dim),
        )

        # KV 先压缩为低维 latent：D -> R(latent_dim)，cache 只存这个
        self.W_dkv = nn.Linear(
            d_model,
            latent_dim,
            bias=False,
        )
        self.W_uk = nn.Linear(     # up-projection：R -> H*dc，恢复 per-head content K
            latent_dim,
            num_heads * content_dim,
            bias=False,
        )
        self.W_uv = nn.Linear(     # up-projection：R -> H*dv，恢复 per-head V
            latent_dim,
            num_heads * self.dv,
            bias=False,
        )

        # 共享 positional key：D -> dr，与 latent bottleneck 解耦
        self.W_kr = nn.Linear(
            d_model,
            rope_dim,
            bias=False,
        )
        self.Wo = nn.Linear(       # 合头后 H*dv -> D
            num_heads * self.dv,
            d_model,
        )

    def forward(self, x, start_pos=0):
        # x: (B,T,D)；start_pos 为已解码历史长度，用于 RoPE 位置偏移
        # 返回: 输出 (B,T,D)、latent cache c_kv (B,T,R)、共享 rope key (B,1,T,dr)
        B, T, _ = x.shape
        pos = torch.arange(        # (T,) 本次这批 token 的绝对位置
            start_pos,
            start_pos + T,
            device=x.device,
        )

        # q: (B,T,H,dc+dr) -> (B,H,T,dc+dr)：拆头并前移 head 维
        q = self.Wq(x).view(
            B, T, self.H, self.dc + self.dr
        ).transpose(1, 2)
        q_c, q_r = torch.split(    # 沿最后一维切成 content/position 两块
            q,
            [self.dc, self.dr],    # -> q_c:(B,H,T,dc)，q_r:(B,H,T,dr)
            dim=-1,
        )
        q_r = rope_part(q_r, pos)  # 只对 position 子空间加 RoPE

        # c_kv: (B,T,R)，KV cache 主体；另缓存共享 RoPE key k_r
        c_kv = self.W_dkv(x)

        # latent -> per-head content K/V：(B,T,R) -> (B,T,H,dc) -> (B,H,T,dc)
        k_c = self.W_uk(c_kv).view(
            B, T, self.H, self.dc
        ).transpose(1, 2)
        v = self.W_uv(c_kv).view(
            B, T, self.H, self.dv
        ).transpose(1, 2)          # (B,H,T,dv)

        # k_r: (B,T,dr) -> (B,1,T,dr)
        # 在 H 个 heads 间共享 positional key
        k_r = self.W_kr(x)[:, None, :, :]
        k_r = rope_part(k_r, pos)
        k_r = k_r.expand(          # (B,1,T,dr) -> (B,H,T,dr)：只是视图展开，不复制显存
            B, self.H, T, self.dr
        )

        # 拼接 content / position 子空间：最后一维 dc+dr
        q = torch.cat([q_c, q_r], dim=-1)   # (B,H,T,dc+dr)
        k = torch.cat([k_c, k_r], dim=-1)   # (B,H,T,dc+dr)

        scores = torch.matmul(     # q @ k^T:(B,H,dc+dr,T) -> (B,H,T,T)
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.dc + self.dr)

        # decoder-only causal mask：上三角(不含对角)为 True，即未来位置
        causal = torch.triu(
            torch.ones(
                T, T,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal[None, None],    # (T,T) -> (1,1,T,T) 广播到 (B,H,T,T)
            float("-inf"),         # 屏蔽未来 key，softmax 后权重为 0
        )

        attn = F.softmax(scores, dim=-1)  # 沿 key 维归一化
        out = torch.matmul(attn, v)       # (B,H,T,T) @ (B,H,T,dv) -> (B,H,T,dv)
        out = out.transpose(
            1, 2
        ).contiguous().view(
            B, T, self.H * self.dv        # (B,H,T,dv) -> (B,T,H,dv) -> (B,T,H*dv)：合头
        )
        return self.Wo(out), c_kv, k_r[:, :1]   # k_r[:, :1] 取回单份共享 key：(B,1,T,dr)
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def rope_part_numpy(x, positions):
    # x: (B,H,T,d) d 为偶数；positions: (T,)；返回同 shape
    d = x.shape[-1]
    half = d // 2                 # 旋转平面数
    freq = 1.0 / (                # (half,) 角频率，base=10000
        10000.0
        ** (
            np.arange(half) / half
        )
    )
    angle = positions[:, None] * freq[None, :]   # (T,1)*(1,half) -> (T,half)
    cos = np.cos(angle)[None, None]              # (T,half) -> (1,1,T,half) 便于广播
    sin = np.sin(angle)[None, None]

    x1 = x[..., :half]            # (B,H,T,half) 实部
    x2 = x[..., half:]            # (B,H,T,half) 虚部
    return np.concatenate(        # 沿最后一维拼回 (B,H,T,d)
        [
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ],
        axis=-1,
    )

def simple_mla_numpy(
    x,
    Wq,
    W_dkv,
    W_uk,
    W_uv,
    W_kr,
    Wo,
    H,
    dc,
    dr,
    start_pos=0,
):
    # x: (B,T,D)；Wq:(D,H*(dc+dr))，W_dkv:(D,R)，W_uk/W_uv:(R,H*dc)，W_kr:(D,dr)
    B, T, _ = x.shape
    pos = np.arange(                    # (T,) 绝对位置，带历史偏移
        start_pos,
        start_pos + T,
    )

    q = (x @ Wq).reshape(               # (B,T,H*(dc+dr)) -> (B,T,H,dc+dr)
        B, T, H, dc + dr
    ).transpose(0, 2, 1, 3)             # -> (B,H,T,dc+dr)：head 维前移
    q_c, q_r = q[..., :dc], q[..., dc:] # 沿最后一维切分 content:(B,H,T,dc) / position:(B,H,T,dr)
    q_r = rope_part_numpy(q_r, pos)     # 仅 position 子空间旋转

    c_kv = x @ W_dkv                    # (B,T,R)

    k_c = (c_kv @ W_uk).reshape(        # latent 上投影：(B,T,H*dc) -> (B,T,H,dc)
        B, T, H, dc
    ).transpose(0, 2, 1, 3)             # -> (B,H,T,dc)
    v = (c_kv @ W_uv).reshape(
        B, T, H, dc
    ).transpose(0, 2, 1, 3)             # (B,H,T,dc)

    k_r = (x @ W_kr)[:, None, :, :]     # (B,T,dr) -> (B,1,T,dr)：所有 head 共享
    k_r = rope_part_numpy(k_r, pos)
    k_r = np.broadcast_to(              # (B,1,T,dr) -> (B,H,T,dr)：只读视图，不复制数据
        k_r,
        (B, H, T, dr),
    )

    q = np.concatenate([q_c, q_r], axis=-1)   # (B,H,T,dc+dr)
    k = np.concatenate([k_c, k_r], axis=-1)   # (B,H,T,dc+dr)

    scores = (
        q @ k.transpose(0, 1, 3, 2)     # k^T:(B,H,dc+dr,T) -> scores:(B,H,T,T)
    ) / np.sqrt(dc + dr)

    causal = np.triu(                   # (T,T) 上三角(不含对角)为 True = 未来位置
        np.ones((T, T), dtype=bool),
        k=1,
    )
    scores = np.where(
        causal[None, None],             # (T,T) -> (1,1,T,T) 广播到 (B,H,T,T)
        -1e9,                           # 用大负数而非 -inf，避免后续出现 NaN
        scores,
    )
    scores = scores - scores.max(       # 沿 key 维减最大值，防 exp 溢出
        axis=-1,
        keepdims=True,
    )
    a = np.exp(scores)
    a /= a.sum(axis=-1, keepdims=True)  # 沿 key 维归一化得权重 (B,H,T,T)

    out = a @ v                         # (B,H,T,T) @ (B,H,T,dc) -> (B,H,T,dc)
    out = out.transpose(
        0, 2, 1, 3
    ).reshape(B, T, H * dc)             # (B,H,T,dc) -> (B,T,H,dc) -> (B,T,H*dc)：合头
    return out @ Wo, c_kv, k_r[:, :1]   # 同时返回 latent cache 与单份共享 rope key (B,1,T,dr)
```

**最小可运行测试**

```python
# Demo
mla = SimpleMLA(
    d_model=32,
    num_heads=4,
    latent_dim=8,       # R=8，远小于 H*dc=24，cache 因此变小
    content_dim=6,
    rope_dim=2,
)
x = torch.randn(2, 5, 32)   # (B=2, T=5, D=32)
y, c_kv, k_rope = mla(x)    # 分别是输出、latent cache、共享 rope key

print("output:", y.shape)       # (2,5,32)
print("latent:", c_kv.shape)    # (2,5,8)
print("rope K:", k_rope.shape)  # (2,1,5,2)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class SimpleMLA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        latent_dim,
        content_dim,
        rope_dim,
    ):
        # 待实现：Wq、KV 下投影 W_dkv 与上投影 W_uk/W_uv、共享 W_kr、输出 Wo
        super().__init__()
        pass

    def forward(self, x, start_pos=0):
        # x: (B,T,D) -> 返回 (B,T,D) 与需要缓存的 (c_kv, k_r)
        pass
```

**Step 1：压缩 KV**

（i）公式

$$
c^{KV}=XW^{DKV}
$$

（ii）代码

```python
c_kv = self.W_dkv(x)   # (B,T,D) -> (B,T,R)：把 K/V 内容压进低维 latent
```

（iii）shape 从 `(B,T,D)` 压到 `(B,T,R)`；$$R\ll H d_h$$ 时，cache 主体显著变小。

**Step 2：从 latent 恢复 content K/V**

（i）公式

$$
K^C=c^{KV}W^{UK},
\qquad
V=c^{KV}W^{UV}
$$

（ii）代码

```python
k_c = self.W_uk(c_kv).view(   # (B,T,R) -> (B,T,H*dc) -> (B,T,H,dc)
    B, T, H, dc
).transpose(1, 2)             # -> (B,H,T,dc)

v = self.W_uv(c_kv).view(
    B, T, H, dv
).transpose(1, 2)             # -> (B,H,T,dv)
```

（iii）这是 low-rank bottleneck 的核心：历史 token 优先保存 latent，而不是直接保存所有 head 的完整 K/V。

**Step 3：位置子空间独立 RoPE**

（i）公式

$$
q_h=[q_h^C;\operatorname{RoPE}(q_h^R)],
\qquad
k_h=[k_h^C;\operatorname{RoPE}(k^R)]
$$

（ii）代码

```python
q_r = rope_part(q_r, pos)          # (B,H,T,dr) 位置子空间加 RoPE
k_r = rope_part(k_r, pos)          # (B,1,T,dr) 共享 positional key，同样加 RoPE
q = torch.cat([q_c, q_r], dim=-1)  # (B,H,T,dc)+(B,H,T,dr) -> (B,H,T,dc+dr)
k = torch.cat([k_c, k_r], dim=-1)  # 同上，点积因此自然拆成内容项 + 位置项
```

（iii）位置部分不经过 KV latent bottleneck；这就是 decoupled RoPE 的面试核心。

**Step 4：Attention**

（i）公式

$$
S=\frac{QK^\top}{\sqrt{d_c+d_r}}
$$

（ii）代码

```python
scores = torch.matmul(              # q:(B,H,T,dc+dr) @ k^T:(B,H,dc+dr,T) -> (B,H,T,T)
    q,
    k.transpose(-2, -1),
) / math.sqrt(self.dc + self.dr)    # 缩放因子用拼接后的总维度
```

（iii）从这里开始与普通多头 attention 相同。

**Step 5：面试中的 cache 表述**

（i）核心对象

$$
\text{cache}
=
(c^{KV}_{1:t},k^R_{1:t})
$$

（ii）代码对应

```python
return self.Wo(out), c_kv, k_r[:, :1]   # 只需缓存 c_kv:(B,T,R) 与 k_r:(B,1,T,dr)
```

（iii）完整 DeepSeek MLA 还有进一步矩阵吸收（absorption）和更复杂的投影结构；面试手写版抓住 latent cache 与 decoupled positional key 即可。这里的 `start_pos` 只负责 RoPE 绝对位置偏移；示例返回可缓存的 `c_kv/k_r`，但没有把历史 cache 拼回本次 attention，因此它展示的是 cache 内容与位置语义，不是完整增量 decode 接口。

#### S−3. RoPE with Position Offset / KV-Cache RoPE

##### 1️⃣ Dense Opening

RoPE 在增量 decode 时最常见的 bug 是每次新 token 都从 position 0 重新旋转。正确做法是新 chunk 使用绝对位置 $$p=\text{past\_len},\ldots,\text{past\_len}+T_{\text{new}}-1$$。核心对象是 position-dependent rotation；只要 offset 正确，prefill 一次计算与分块增量计算应得到同样的旋转结果。[参考 5]

##### 2️⃣ Global Equations

对每个二维 pair：

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(p\theta_i)&-\sin(p\theta_i)\\
\sin(p\theta_i)&\cos(p\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}
$$

$$
\theta_i
=
10000^{-2i/d}
$$

decode chunk 的 position：

$$
p_t
=
\text{start\_pos}+t
$$

因此历史 cache 长度为 $$L$$ 时，新 token 的 `start_pos = L`。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
from torch import Tensor

def apply_rope_offset(
    x: Tensor,
    start_pos: int = 0,
) -> Tensor:
    # x: (B,H,T,D)，D 必须为偶数；start_pos 为本 chunk 首 token 的绝对位置
    # 返回: (B,H,T,D)，shape 不变，只旋转每个二维 pair
    B, H, T, D = x.shape
    assert D % 2 == 0

    # 每个 pair 使用一个频率：theta_i = 10000^(-2i/D)；角度计算固定用 float32
    freq = 1.0 / (
        10000.0
        ** (
            torch.arange(
                0, D, 2,
                device=x.device,
                dtype=torch.float32,
            ) / D                          # 偶数下标 0,2,...,D-2 归一化为 2i/D
        )
    )                                     # (D/2,)

    # decode 时从 past_len 开始；position 也用 float32，避免 fp16 长位置精度损失
    pos = torch.arange(
        start_pos,
        start_pos + T,
        device=x.device,
        dtype=torch.float32,
    )                                     # (T,)

    angle = pos[:, None] * freq[None, :]   # (T,D/2)：先以 float32 计算角度
    cos = angle.cos()[None, None, :, :].to(x.dtype)  # cast 回输入 dtype，再广播到 (B,H,T,D/2)
    sin = angle.sin()[None, None, :, :].to(x.dtype)

    even = x[..., 0::2]                    # (B,H,T,D/2)
    odd = x[..., 1::2]                     # 偶/奇下标各取一半，(x_2i, x_2i+1) 构成一个旋转平面

    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin  # 逐 pair 做 2x2 旋转，再交错写回原下标
    out[..., 1::2] = even * sin + odd * cos
    return out
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def apply_rope_offset_numpy(
    x,
    start_pos=0,
):
    # x: (B,H,T,D)，D 偶数；返回同 shape 的旋转结果
    B, H, T, D = x.shape
    assert D % 2 == 0

    freq = 1.0 / (
        10000.0
        ** (
            np.arange(0, D, 2) / D         # 2i/D，i = 0..D/2-1
        )
    )                                     # (D/2,)
    pos = np.arange(
        start_pos,
        start_pos + T,
    )                                     # (T,)：绝对位置，从历史长度 start_pos 续接

    angle = pos[:, None] * freq[None, :]   # (T,1)*(1,D/2) -> (T,D/2)
    cos = np.cos(angle)[None, None]        # (T,D/2) -> (1,1,T,D/2)，广播到 batch/head
    sin = np.sin(angle)[None, None]

    even = x[..., 0::2]                    # (B,H,T,D) -> (B,H,T,D/2)
    odd = x[..., 1::2]

    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin  # 2x2 旋转后交错写回，恢复 (B,H,T,D)
    out[..., 1::2] = even * sin + odd * cos
    return out
```

**最小可运行测试**

```python
# Demo：full 与分块 RoPE 应一致
x = torch.randn(1, 4, 6, 8)            # (B=1, H=4, T=6, D=8)

full = apply_rope_offset(
    x,                                 # 一次性 prefill：位置 0..5
    start_pos=0,
)

part1 = apply_rope_offset(
    x[:, :, :4],                       # 前 4 个 token，(1,4,4,8)，位置 0..3
    start_pos=0,
)
part2 = apply_rope_offset(
    x[:, :, 4:],                       # 后 2 个 token，(1,4,2,8)，位置必须续接为 4..5
    start_pos=4,
)

chunked = torch.cat([part1, part2], dim=2)  # 沿 T 拼回 (1,4,6,8)
print(torch.allclose(full, chunked, atol=1e-6))
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def apply_rope_offset(
    x: Tensor,
    start_pos: int = 0,
) -> Tensor:
    # x: (B,H,T,D) -> (B,H,T,D)；start_pos = 已缓存的历史 token 数
    pass
```

**Step 1：构造频率**

（i）公式

$$
\theta_i=10000^{-2i/D}
$$

（ii）代码

```python
freq = 1.0 / (
    10000.0
    ** (
        torch.arange(
            0, D, 2,                # 偶数下标 0,2,...,D-2，共 D/2 个
            device=x.device,
            dtype=torch.float32,    # RoPE 角度用 float32 计算更稳
        ) / D                       # -> 2i/D
    )
)                                   # (D/2,)：每个二维平面一个旋转频率
```

（iii）`freq: (D/2,)`，每两个 embedding 坐标形成一个二维旋转平面。

**Step 2：加 position offset**

（i）公式

$$
p_t=\text{start\_pos}+t
$$

（ii）代码

```python
pos = torch.arange(
    start_pos,                      # decode 时等于已缓存历史长度 L
    start_pos + T,
    device=x.device,
    dtype=torch.float32,
)                                   # (T,)：本 chunk 各 token 的绝对位置；避免 fp16 position 精度损失
```

（iii）KV cache decode 时 `start_pos` 就是已有历史 token 数。

**Step 3：广播 angle**

（i）shape

$$
(T)\times(D/2)
\rightarrow
(T,D/2)
\rightarrow
(1,1,T,D/2)
$$

（ii）代码

```python
angle = pos[:, None] * freq[None, :]   # (T,1)*(1,D/2) 外积 -> (T,D/2)
cos = angle.cos()[None, None].to(x.dtype)  # float32 算三角函数，再 cast 回输入 dtype
sin = angle.sin()[None, None].to(x.dtype)
```

（iii）batch/head 共享同一组位置频率，因此只需在前两维补 1。

**Step 4：二维旋转**

（i）公式

$$
x'_{\mathrm{even}}
=
x_{\mathrm{even}}\cos
-
x_{\mathrm{odd}}\sin
$$

$$
x'_{\mathrm{odd}}
=
x_{\mathrm{even}}\sin
+
x_{\mathrm{odd}}\cos
$$

（ii）代码

```python
out[..., 0::2] = even * cos - odd * sin  # (B,H,T,D/2) 结果写回偶数下标
out[..., 1::2] = even * sin + odd * cos  # 奇数下标；交错还原为 (B,H,T,D)
```

（iii）rotation 保持每个二维 pair 的 L2 norm；offset 只改变角度，不改变 shape。


#### 👀 S−4. Generalized Advantage Estimation（GAE）

##### 1️⃣ Dense Opening

GAE 用 TD residual 的指数加权和估计 advantage，在 bias 与 variance 之间用 $$\lambda$$ 调节。核心对象是 $$\delta_t$$ 与反向递推的 $$A_t$$。面试实现只需一个 reverse loop，并正确处理终止状态的 bootstrap mask。[参考 6]

##### 2️⃣ Global Equations

TD residual：

$$
\delta_t
=
r_t
+
\gamma(1-d_t)V_{t+1}
-
V_t
$$

其中 $$d_t=1$$ 表示该步后 episode 终止。

GAE：

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l}
$$

反向递推：

$$
A_t
=
\delta_t
+
\gamma\lambda(1-d_t)A_{t+1}
$$

value target 常写为：

$$
R_t^{V}
=
A_t+V_t
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
from torch import Tensor

def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    # rewards: (T,)
    # values:  (T+1,)；最后一个是 bootstrap V_T
    # dones:   (T,)；1 表示该 transition 后终止
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)  # (T,)，反向循环里逐位置填入

    gae = torch.zeros(
        (),                                 # 0-dim 标量：递推累加器 A_{t+1}，初值对应 A_T = 0
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in reversed(range(T)):            # 必须从 T-1 倒推到 0，才能复用 A_{t+1}
        not_done = 1.0 - dones[t]           # 终止步取 0，切断跨 episode 的 bootstrap

        delta = (
            rewards[t]                      # TD residual：r_t + γ(1-d_t)V_{t+1} - V_t
            + gamma * not_done * values[t + 1]
            - values[t]
        )

        gae = (
            delta
            + gamma * lam * not_done * gae  # λ 衰减地累加未来 residual
        )
        advantages[t] = gae

    returns = advantages + values[:-1]      # values[:-1]: (T+1,) -> (T,) 丢掉 bootstrap 项，对齐 advantages
    return advantages, returns
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def compute_gae_numpy(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95,
):
    # rewards/dones: (T,)；values: (T+1,)，末位为 bootstrap V_T
    T = len(rewards)
    advantages = np.zeros_like(
        rewards,
        dtype=np.float64,                   # 强制 float64，避免整型 reward 把递推截断
    )

    gae = 0.0                               # 累加器，等价于边界条件 A_T = 0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]           # 终止步归零，阻断 V_{t+1} 与 A_{t+1}

        delta = (
            rewards[t]
            + gamma * not_done * values[t + 1]
            - values[t]
        )

        gae = (
            delta
            + gamma * lam * not_done * gae
        )
        advantages[t] = gae

    returns = advantages + values[:-1]      # (T+1,) -> (T,)：critic 的回归目标 A_t + V_t
    return advantages, returns
```

**最小可运行测试**

```python
# Demo
rewards = torch.tensor([1.0, 0.0, 1.0])       # (T=3,)
values = torch.tensor([0.5, 0.4, 0.2, 0.0])   # (T+1=4,)，末位 0.0 是 bootstrap V_3
dones = torch.tensor([0.0, 0.0, 1.0])         # (3,)：最后一步终止，故 t=2 不 bootstrap

adv, ret = compute_gae(
    rewards,
    values,
    dones,
)
print("adv:", adv)
print("returns:", ret)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
):
    # rewards/dones: (T,)；values: (T+1,)
    # 返回: advantages (T,) 与 returns (T,)
    pass
```

**Step 1：TD residual**

（i）公式

$$
\delta_t
=
r_t
+
\gamma(1-d_t)V_{t+1}
-
V_t
$$

（ii）代码

```python
not_done = 1.0 - dones[t]           # 标量掩码：终止步为 0

delta = (
    rewards[t]
    + gamma * not_done * values[t + 1]  # 索引 t+1 取下一状态的 value，故 values 长度需为 T+1
    - values[t]
)
```

（iii）终止状态必须把 $$V_{t+1}$$ bootstrap 切断，否则会把下一个 episode 的 value 混进当前轨迹。

**Step 2：反向递推**

（i）公式

$$
A_t
=
\delta_t
+
\gamma\lambda(1-d_t)A_{t+1}
$$

（ii）代码

```python
gae = (
    delta
    + gamma * lam * not_done * gae  # 右侧 gae 是上一轮算出的 A_{t+1}
)
advantages[t] = gae                 # 写入 (T,) 的第 t 位
```

（iii）从 $$T-1$$ 往 0 迭代，一次循环即可实现完整加权和。

**Step 3：Value target**

（i）公式

$$
R_t^V=A_t+V_t
$$

（ii）代码

```python
returns = advantages + values[:-1]  # values 去掉末位 bootstrap，(T+1,) -> (T,) 与 advantages 对齐
```

（iii）PPO 常用 `returns` 训练 critic，而 `advantages` 训练 policy。

#### 👀 S−5. SFT Masked Loss（Assistant-Only Loss）

##### 1️⃣ Dense Opening

SFT 的核心仍是 causal next-token cross-entropy，但 instruction tuning 通常只让 assistant/response token 产生梯度，prompt、system、padding 位置全部屏蔽。面试里最短且标准的写法是先把不训练的位置 label 设为 `-100`，再做 next-token shift，最后交给 `cross_entropy(ignore_index=-100)`；关键仍是 mask 必须作用在 **target token** 上。[参考 7, 21]

##### 2️⃣ Global Equations

设 $$m_{b,t}\in\{0,1\}$$ 表示 target token $$y_{b,t}$$ 是否属于 assistant response：

$$
\mathcal L_{\mathrm{SFT}}
=
-
\frac{
\sum_{b,t}
m_{b,t}
\log
p_\theta
\left(
y_{b,t}
\mid
y_{b,<t}
\right)
}{
\sum_{b,t}m_{b,t}
}
$$

shift 后：

$$
\text{logits}_{:,0:T-1}
\leftrightarrow
\text{labels}_{:,1:T}
$$

$$
\widetilde y_{b,t}
=
\begin{cases}
y_{b,t}, & m_{b,t}=1\\
-100, & m_{b,t}=0
\end{cases}
$$

shift 后直接忽略 `-100`：

$$
\mathcal L_{\mathrm{SFT}}
=
\operatorname{CE}
\left(
\text{logits}_{:,0:T-1},
\widetilde y_{:,1:T};
\text{ignore\_index}=-100
\right)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def sft_masked_loss(
    logits: Tensor,
    labels: Tensor,
    response_mask: Tensor,
) -> Tensor:
    # logits:        (B,T,V)
    # labels:        (B,T)
    # response_mask: (B,T)，assistant token=1
    masked_labels = labels.masked_fill(
        response_mask == 0,                # prompt/system/pad 位置
        -100,
    )                                      # (B,T)，prompt/pad -> -100

    # logits_t 预测 label_{t+1}
    shift_logits = logits[:, :-1, :]       # (B,T,V) -> (B,T-1,V)：丢掉最后一步（无 target）
    shift_labels = masked_labels[:, 1:]    # (B,T) -> (B,T-1)：左移一位，target 从第 1 个 token 起

    V = shift_logits.size(-1)              # 词表大小
    return F.cross_entropy(
        shift_logits.reshape(-1, V),       # (B,T-1,V) -> (B*(T-1),V)：把 batch/time 压平成样本维
        shift_labels.reshape(-1),          # (B,T-1) -> (B*(T-1),)
        ignore_index=-100,                 # -100 位置既不算 loss 也不参与均值分母
    )
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def sft_masked_loss_numpy(
    logits,
    labels,
    response_mask,
):
    # logits: (B,T,V)；labels/response_mask: (B,T)；返回标量 loss
    masked_labels = np.where(
        response_mask.astype(bool),        # 0/1 转 bool 才能当条件掩码
        labels,
        -100,
    )                                      # (B,T)

    x = logits[:, :-1, :]                  # (B,T-1,V)
    y = masked_labels[:, 1:]               # (B,T-1)：next-token shift 后的 target
    valid = y != -100                      # (B,T-1) bool：需要计入 loss 的位置
    safe_y = np.where(valid, y, 0)         # 把 -100 换成合法下标 0，避免后面 gather 越界（其值会被 valid 丢掉）

    # stable log-softmax
    x = x - x.max(axis=-1, keepdims=True)  # 沿词表维减最大值防 exp 上溢，shape 不变
    log_probs = x - np.log(
        np.exp(x).sum(axis=-1, keepdims=True)  # (B,T-1,V) -> (B,T-1,1)：logsumexp 归一化项
    )

    B, Tm1 = y.shape
    nll = -log_probs[
        np.arange(B)[:, None],             # (B,1) 行索引，与下面两个索引广播成 (B,T-1)
        np.arange(Tm1)[None, :],           # (1,T-1) 时间索引
        safe_y,                            # (B,T-1) 目标 token id：沿 V 维取出对应 log-prob
    ]                                      # (B,T-1)
    return nll[valid].mean()               # bool 掩码展平成 1D，只对有效位置求均值
```

**最小可运行测试**

```python
# Demo
B, T, V = 2, 6, 20                       # batch=2，序列长 6，词表 20
logits = torch.randn(B, T, V)            # (2,6,20)
labels = torch.randint(0, V, (B, T))     # (2,6)，token id ∈ [0,V)

# 前 3 个位置视为 prompt，后面是 assistant
response_mask = torch.tensor([
    [0, 0, 0, 1, 1, 1],                  # 样本 0：前 3 个 prompt token 不算 loss
    [0, 0, 1, 1, 1, 1],                  # (B,T)=(2,6)，每行 prompt 长度可不同
])

print(
    "SFT loss:",
    sft_masked_loss(
        logits,
        labels,
        response_mask,
    ).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def sft_masked_loss(
    logits: Tensor,
    labels: Tensor,
    response_mask: Tensor,
) -> Tensor:
    # logits: (B,T,V)；labels: (B,T)；response_mask: (B,T)，1=assistant token
    # 返回: 标量 loss，只在 assistant target 上平均
    pass
```

**Step 1：先把非 response target 标成 ignore index**

（i）公式

$$
\widetilde y_{b,t}
=
\begin{cases}
y_{b,t}, & m_{b,t}=1\\
-100, & m_{b,t}=0
\end{cases}
$$

（ii）代码

```python
masked_labels = labels.masked_fill(
    response_mask == 0,             # (B,T) bool：非 assistant 位置
    -100,
)                                   # (B,T)，shape 不变，只把不训练的 target 换成 ignore index
```

（iii）`-100` 是 PyTorch `cross_entropy` 默认的 ignore index；prompt、system、padding 都可以统一通过 label mask 屏蔽。

**Step 2：Next-token shift**

（i）对齐关系

$$
\text{logits}_t
\rightarrow
\widetilde y_{t+1}
$$

（ii）代码

```python
shift_logits = logits[:, :-1, :]    # (B,T,V) -> (B,T-1,V)：位置 0..T-2 的预测
shift_labels = masked_labels[:, 1:] # (B,T) -> (B,T-1)：对应 target 位置 1..T-1
```

（iii）先 mask label 再 shift，可以直接保证 mask 与 target token 对齐。

**Step 3：一次 Cross-Entropy 完成 masked reduction**

（i）公式

$$
L
=
-\frac{1}{N_{\mathrm{valid}}}
\sum_{(b,t):\widetilde y_{b,t}\neq -100}
\log p_\theta(\widetilde y_{b,t})
$$

（ii）代码

```python
V = shift_logits.size(-1)
return F.cross_entropy(
    shift_logits.reshape(-1, V),    # (B,T-1,V) -> (B*(T-1),V)
    shift_labels.reshape(-1),       # (B,T-1) -> (B*(T-1),)
    ignore_index=-100,              # 默认 reduction='mean'，分母只数有效 token
)
```

（iii）与手工 `token_loss * mask` 完全同一目标，但代码更短，也与常见 LLM SFT 数据管线一致。

#### S−6. Modern LLaMA-Style Block（RMSNorm + RoPE + GQA + SwiGLU）

##### 1️⃣ Dense Opening

现代 LLaMA-style decoder block 可以压缩成一条固定计算链：Pre-RMSNorm $$\rightarrow$$ RoPE-GQA causal attention $$\rightarrow$$ residual $$\rightarrow$$ Pre-RMSNorm $$\rightarrow$$ SwiGLU $$\rightarrow$$ residual。核心对象是 GQA 中 Q-head 与 KV-head 的分组映射，以及 RoPE 只作用于 Q/K。下面是面试可写版本，不包含 fused kernels、tensor parallel 或特殊 cache layout。[参考 5, 8]

##### 2️⃣ Global Equations

RMSNorm：

$$
\operatorname{RMSNorm}(x)
=
\gamma
\odot
\frac{x}{
\sqrt{
\frac{1}{D}
\sum_{j=1}^{D}x_j^2+\epsilon
}
}
$$

GQA：

$$
Q
\in
\mathbb{R}^{B\times H_q\times T\times d_h}
$$

$$
K,V
\in
\mathbb{R}^{B\times H_{kv}\times T\times d_h}
$$

$$
r
=
\frac{H_q}{H_{kv}}
$$

每个 KV head 被 $$r$$ 个 query heads 共享。

SwiGLU：

$$
\operatorname{SwiGLU}(x)
=
W_{\mathrm{down}}
\left[
\operatorname{SiLU}(xW_g)
\odot
(xW_u)
\right]
$$

完整 block：

$$
X_1
=
X+
\operatorname{GQA}_{\mathrm{RoPE}}
\left(
\operatorname{RMSNorm}(X)
\right)
$$

$$
X_2
=
X_1+
\operatorname{SwiGLU}
\left(
\operatorname{RMSNorm}(X_1)
\right)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准面试实现）**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(d_model)            # 每个通道一个可学习缩放 gamma，(D,)
        )
        self.eps = eps                     # 防除零，放在 sqrt 内部

    def forward(self, x):
        # x: (...,D) -> (...,D)；float32 reduction 后 cast 回原 dtype
        dtype = x.dtype
        x_f = x.float()
        rms_inv = torch.rsqrt(
            x_f.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_f * rms_inv).to(dtype) * self.weight

def apply_rope(x):
    # x: (B,H,T,dh), dh even -> 同 shape，位置固定从 0 开始
    B, H, T, D = x.shape
    freq = 1.0 / (
        10000.0
        ** (
            torch.arange(
                0, D, 2,                   # D/2 个偶数下标，每个对应一个旋转平面
                device=x.device,
                dtype=torch.float32,
            ) / D
        )
    )                                      # (D/2,)
    pos = torch.arange(
        T,
        device=x.device,
        dtype=torch.float32,
    )                                      # (T,)
    angle = pos[:, None] * freq[None, :]   # float32: (T,D/2)
    cos = angle.cos()[None, None].to(x.dtype)  # 三角函数后 cast 回输入 dtype
    sin = angle.sin()[None, None].to(x.dtype)

    even = x[..., 0::2]                    # (B,H,T,D) -> (B,H,T,D/2)
    odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin  # 2x2 旋转后交错写回，还原 (B,H,T,D)
    out[..., 1::2] = even * sin + odd * cos
    return out

class LlamaStyleBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,
        d_ff,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0   # Hq 必须能整分给 Hkv 组，否则 GQA 分组不均

        self.Hq = num_heads                # query head 数
        self.Hkv = num_kv_heads            # KV head 数（< Hq，多个 Q head 共享一个 KV head）
        self.dh = d_model // num_heads     # 每个 head 的维度
        self.D = d_model

        self.norm1 = RMSNorm(d_model)      # attention 前的 pre-norm
        self.norm2 = RMSNorm(d_model)      # FFN 前的 pre-norm

        self.Wq = nn.Linear(
            d_model,
            num_heads * self.dh,           # = d_model，Q 保留全部 head
            bias=False,
        )
        self.Wk = nn.Linear(
            d_model,
            num_kv_heads * self.dh,        # KV 投影维度更小，这是 GQA 省 cache 的来源
            bias=False,
        )
        self.Wv = nn.Linear(
            d_model,
            num_kv_heads * self.dh,
            bias=False,
        )
        self.Wo = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.gate = nn.Linear(
            d_model,                       # SwiGLU 的门控分支：D -> d_ff，过 SiLU
            d_ff,
            bias=False,
        )
        self.up = nn.Linear(
            d_model,                       # 线性分支：D -> d_ff，与门控逐元素相乘
            d_ff,
            bias=False,
        )
        self.down = nn.Linear(
            d_ff,                          # 降回 D
            d_model,
            bias=False,
        )

    def _attn(self, x):
        # x: (B,T,D)
        B, T, _ = x.shape

        q = self.Wq(x).view(               # (B,T,D) -> (B,T,Hq*dh) -> (B,T,Hq,dh)
            B, T, self.Hq, self.dh
        ).transpose(1, 2)                  # (B,Hq,T,dh)

        k = self.Wk(x).view(               # KV 只有 Hkv 个 head
            B, T, self.Hkv, self.dh
        ).transpose(1, 2)                  # (B,Hkv,T,dh)

        v = self.Wv(x).view(
            B, T, self.Hkv, self.dh
        ).transpose(1, 2)                  # (B,Hkv,T,dh)

        # RoPE 只作用 Q/K：相对位置由 q·k 的角度差体现，V 是被加权的内容
        q = apply_rope(q)
        k = apply_rope(k)

        # GQA: (B,Hkv,T,dh) -> (B,Hq,T,dh)
        repeat = self.Hq // self.Hkv       # 每个 KV head 被 repeat 个 Q head 共享
        k = k.repeat_interleave(
            repeat,                        # interleave 而非 tile：head i 的副本相邻，
            dim=1,                         # 才能与 q 的 head 分组 [0..r-1],[r..2r-1] 对齐
        )
        v = v.repeat_interleave(
            repeat,
            dim=1,
        )

        scores = torch.matmul(
            q,                             # q:(B,Hq,T,dh) @ k^T:(B,Hq,dh,T) -> (B,Hq,T,T)
            k.transpose(-2, -1),           # 交换最后两维得到 key 的转置
        ) / math.sqrt(self.dh)             # (B,Hq,T,T)

        causal = torch.triu(
            torch.ones(
                T, T,                      # (T,T) 全 1
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,                    # 严格上三角为 True = 未来位置（对角线保留自身可见）
        )
        scores = scores.masked_fill(
            causal[None, None],            # (T,T) -> (1,1,T,T)，广播到所有 batch/head
            float("-inf"),                 # 未来 token 屏蔽为 -inf，softmax 后权重恰为 0
        )

        a = F.softmax(scores, dim=-1)      # 沿 key 维归一化，每行 attention 权重和为 1
        out = torch.matmul(a, v)           # (B,Hq,T,T) @ (B,Hq,T,dh) -> (B,Hq,T,dh)：按权重聚合 value
        out = out.transpose(
            1, 2                           # (B,Hq,T,dh) -> (B,T,Hq,dh)
        ).contiguous().view(B, T, self.D)  # 合并 head 维 -> (B,T,D)；transpose 后需 contiguous 才能 view
        return self.Wo(out)

    def _mlp(self, x):
        # (...,D) -> (...,d_ff) -> (...,D)
        return self.down(
            F.silu(self.gate(x))           # 门控分支过 SiLU
            * self.up(x)                   # 与线性分支逐元素相乘（gating），仍是 (...,d_ff)
        )

    def forward(self, x):
        x = x + self._attn(self.norm1(x))  # pre-norm + residual，(B,T,D) 全程不变
        x = x + self._mlp(self.norm2(x))
        return x
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def rms_norm_numpy(x, weight, eps=1e-6):
    # x: (...,D)；weight: (D,)；返回 (...,D)
    rms = np.sqrt(
        np.mean(
            x ** 2,
            axis=-1,                       # 沿特征维聚合 -> (...,1)
            keepdims=True,
        )
        + eps                              # 防除零
    )
    return x / rms * weight                # 广播回 (...,D)，逐通道乘 gamma

def rope_numpy(x):
    # x: (B,H,T,D) -> (B,H,T,D)
    B, H, T, D = x.shape
    freq = 1.0 / (
        10000.0
        ** (
            np.arange(0, D, 2) / D
        )
    )                                      # (D/2,)
    pos = np.arange(T)                     # (T,)
    angle = pos[:, None] * freq[None, :]   # (T,1)*(1,D/2) -> (T,D/2)
    cos = np.cos(angle)[None, None]        # -> (1,1,T,D/2)
    sin = np.sin(angle)[None, None]

    even = x[..., 0::2]                    # (B,H,T,D/2)
    odd = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out

def silu_numpy(x):
    return x / (1.0 + np.exp(-x))          # x * sigmoid(x)，shape 不变

def llama_block_numpy(
    x,
    p,
    Hq,
    Hkv,
):
    # x: (B,T,D)；p 是权重字典；Hq/Hkv 为 query/KV head 数
    B, T, D = x.shape
    dh = D // Hq                           # 单 head 维度

    h = rms_norm_numpy(
        x,
        p["norm1"],
    )                                      # pre-norm，(B,T,D) 不变

    q = (h @ p["Wq"]).reshape(             # (B,T,D)@(D,Hq*dh) -> (B,T,Hq*dh) -> (B,T,Hq,dh)
        B, T, Hq, dh
    ).transpose(0, 2, 1, 3)                # -> (B,Hq,T,dh)：head 提到前面便于批量 matmul

    k = (h @ p["Wk"]).reshape(             # KV 投影到 Hkv*dh，比 Q 窄
        B, T, Hkv, dh
    ).transpose(0, 2, 1, 3)                # -> (B,Hkv,T,dh)

    v = (h @ p["Wv"]).reshape(
        B, T, Hkv, dh
    ).transpose(0, 2, 1, 3)                # -> (B,Hkv,T,dh)

    q = rope_numpy(q)                      # RoPE 只旋转 Q/K
    k = rope_numpy(k)

    repeat = Hq // Hkv                     # 每个 KV head 服务的 Q head 数
    k = np.repeat(k, repeat, axis=1)       # 沿 head 维逐元素重复 -> (B,Hq,T,dh)，与 q 的 head 分组对齐
    v = np.repeat(v, repeat, axis=1)

    scores = (
        q @ k.transpose(0, 1, 3, 2)        # (B,Hq,T,dh) @ (B,Hq,dh,T) -> (B,Hq,T,T)
    ) / np.sqrt(dh)                        # 缩放防 softmax 饱和

    causal = np.triu(
        np.ones((T, T), dtype=bool),
        k=1,                               # 严格上三角 = 未来位置
    )
    scores = np.where(
        causal[None, None],                # (T,T) -> (1,1,T,T) 广播到 batch/head
        -1e9,                              # 用大负数而非 -inf，避免后续减 max 出 NaN
        scores,
    )
    scores -= scores.max(
        axis=-1,                           # 沿 key 维取行最大值，(B,Hq,T,T) -> (B,Hq,T,1)
        keepdims=True,
    )
    a = np.exp(scores)
    a /= a.sum(axis=-1, keepdims=True)     # 手写 softmax：每行权重归一化为 1

    out = a @ v                            # (B,Hq,T,T) @ (B,Hq,T,dh) -> (B,Hq,T,dh)
    out = out.transpose(
        0, 2, 1, 3                         # -> (B,T,Hq,dh)
    ).reshape(B, T, D)                     # 合并所有 head -> (B,T,D)
    x = x + out @ p["Wo"]                  # 输出投影 + 第一条 residual

    h = rms_norm_numpy(
        x,
        p["norm2"],
    )
    gate = silu_numpy(h @ p["Wg"])         # (B,T,D) -> (B,T,d_ff)，门控分支
    up = h @ p["Wu"]                       # (B,T,d_ff)，线性分支
    x = x + (gate * up) @ p["Wd"]          # 逐元素相乘后降回 (B,T,D)，第二条 residual
    return x
```

**最小可运行测试**

```python
# Demo
block = LlamaStyleBlock(
    d_model=64,
    num_heads=8,                 # Hq=8，dh = 64/8 = 8
    num_kv_heads=2,              # Hkv=2，每个 KV head 被 4 个 Q head 共享
    d_ff=128,
)
x = torch.randn(2, 6, 64)        # (B=2, T=6, D=64)，输出同 shape
print("output:", block(x).shape)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class LlamaStyleBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_heads,              # < num_heads，且需整除 num_heads
        d_ff,
    ):
        super().__init__()
        pass

    def forward(self, x):
        # x: (B,T,d_model) -> (B,T,d_model)
        pass
```

**Step 1：Pre-RMSNorm**

（i）公式

$$
\tilde X
=
\operatorname{RMSNorm}(X)
$$

（ii）代码

```python
h = self.norm1(x)   # (B,T,D) -> (B,T,D)：只按 RMS 重缩放，不改 shape
```

（iii）不减均值，只按 RMS 重缩放；shape 不变。

**Step 2：GQA 投影**

（i）shape

$$
Q:(B,H_q,T,d_h)
$$

$$
K,V:(B,H_{kv},T,d_h)
$$

（ii）代码

```python
q = self.Wq(h).view(
    B, T, self.Hq, self.dh      # (B,T,D) -> (B,T,Hq,dh)：拆出 head 维
).transpose(1, 2)               # -> (B,Hq,T,dh)

k = self.Wk(h).view(
    B, T, self.Hkv, self.dh     # K 只有 Hkv 个 head
).transpose(1, 2)               # -> (B,Hkv,T,dh)
```

（iii）$$H_{kv}<H_q$$ 是 GQA 的结构核心。

**Step 3：RoPE + KV head replication**

（i）公式

$$
Q\leftarrow\operatorname{RoPE}(Q),
\qquad
K\leftarrow\operatorname{RoPE}(K)
$$

$$
r=H_q/H_{kv}
$$

（ii）代码

```python
q = apply_rope(q)
k = apply_rope(k)                       # shape 不变，只按绝对位置旋转

repeat = self.Hq // self.Hkv            # 每个 KV head 的共享倍数 r
k = k.repeat_interleave(repeat, dim=1)  # (B,Hkv,T,dh) -> (B,Hq,T,dh)：同一 head 的副本相邻
v = v.repeat_interleave(repeat, dim=1)
```

（iii）RoPE 不作用于 V；repeat 只为面试代码直观，工业 kernel 通常避免真正复制 KV。

**Step 4：Causal attention residual**

（i）公式

$$
X_1
=
X+
\operatorname{Attention}
\left(
\operatorname{RMSNorm}(X)
\right)
$$

（ii）代码

```python
x = x + self._attn(self.norm1(x))   # attention 分支输出 (B,T,D)，与输入同 shape 才能相加
```

（iii）attention 输出最终回到 `(B,T,D)` 才能做 residual。

**Step 5：SwiGLU residual**

（i）公式

$$
X_2
=
X_1
+
W_d
\left[
\operatorname{SiLU}(X_1W_g)
\odot
(X_1W_u)
\right]
$$

（ii）代码

```python
x = x + self._mlp(self.norm2(x))    # SwiGLU 分支 D -> d_ff -> D，第二条 residual
```

（iii）这一步完成现代 decoder block 的第二条 residual branch。

### A：应熟练

> 本章分两部分：**第一部分**为主文档原有题目，**第二部分**为《补充文档-LLM面试手撕代码补充题单》新增题目。

#### ▍第一部分 · 主文档原有题目

#### 👀 SwiGLU

LLaMA、PaLM 等使用的门控 FFN：用 SiLU 门控一个并行支路。核心对象：$$\text{SiLU}(W_1 x) \odot (W_2 x)$$。

$$
\text{SiLU}(x) = x \cdot \sigma(x),\quad
\text{SwiGLU}(x) = W_3 \big( \text{SiLU}(W_1 x) \odot W_2 x \big)
$$

相比 ReLU FFN，多了 50% 参数（三个矩阵），但效果显著更好。

#### （A）PyTorch 版本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # 门控支路升维：D -> d_ff
        self.w2 = nn.Linear(d_model, d_ff, bias=False)   # 数值支路升维：D -> d_ff
        self.w3 = nn.Linear(d_ff, d_model, bias=False)   # 降维回残差流：d_ff -> D

    def forward(self, x):
        # x: (..., D) -> 同形状
        # silu(w1 x) 与 w2 x 都是 (..., d_ff)，逐元素相乘做 gating 后再投影回 D
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

#### （B）NumPy 版本

```python
import numpy as np

def silu(x): return x / (1 + np.exp(-x))           # SiLU(x)=x·σ(x)，逐元素平滑非线性

def swiglu_numpy(x, W1, W2, W3):
    # x: (..., D)；W1/W2: (D,d_ff)，W3: (d_ff,D)；返回 (..., D)
    gate = silu(x @ W1)                                # (..., d_ff)：门控值，控制各通道开合
    up   = x @ W2                                      # (..., d_ff)：被门控的数值支路
    return (gate * up) @ W3                            # 逐元素 gating 后 (..., d_ff) -> (..., D)

# Demo
x = np.random.randn(2, 4, 8)                       # (B=2, T=4, D=8)
W1, W2 = np.random.randn(8, 32), np.random.randn(8, 32)   # D=8 -> d_ff=32
W3 = np.random.randn(32, 8)                        # d_ff=32 -> D=8
print(swiglu_numpy(x, W1, W2, W3).shape)           # 期望 (2,4,8)，与输入同形
```

#### Step-by-step

```python
def swiglu(x): pass                # 待实现：x (...,D) -> (...,D)，门控支路 × 数值支路后投影回 D
```

（i）门控分支 + 数值分支
（ii）`gate = F.silu(self.w1(x)); up = self.w2(x)`
（iii）`SiLU` 提供平滑非线性，逐元素与 `up` 相乘形成 gating。

（i）门控乘积
（ii）`return self.w3(gate * up)`
（iii）gating 选择性放大/抑制 hidden 维各通道。

---

#### ▍第二部分 · 补充题单新增题目

#### A1. Reward Model Pairwise Loss（Bradley-Terry / Preference Loss）

##### 1️⃣ Dense Opening

Reward Model（RM）接收同一个 prompt 下的 chosen / rejected response，输出两个标量 reward，并要求 chosen reward 更高。核心对象是 reward margin $$\Delta r=r_c-r_r$$，用 logistic preference likelihood 转成 pairwise loss。面试手写通常只考这一行稳定的 `logsigmoid`。[参考 7]

##### 2️⃣ Global Equations

Bradley-Terry preference probability：

$$
P(y_c\succ y_r\mid x)
=
\sigma
\left(
r_\phi(x,y_c)
-
r_\phi(x,y_r)
\right)
$$

loss：

$$
\mathcal L_{\mathrm{RM}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\sigma
\left(
r_i^c-r_i^r
\right)
$$

若加入 margin $$m$$：

$$
\mathcal L
=
-\log\sigma(r_c-r_r-m)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def reward_pairwise_loss(
    chosen_rewards: Tensor,
    rejected_rewards: Tensor,
    margin: float = 0.0,
) -> Tensor:
    # chosen/rejected: (B,) 每个 pair 的标量 reward；margin: 期望的最小 reward 间隔
    # 返回: 标量 loss
    diff = (
        chosen_rewards
        - rejected_rewards
        - margin
    )                                      # (B,)：reward margin Δr，broadcast 减去标量 margin

    # -log(sigmoid(diff)) 即 softplus(-diff)，logsigmoid 避免 exp 溢出
    return -F.logsigmoid(diff).mean()      # (B,) -> 标量：对 batch 内所有 pair 取平均
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def reward_pairwise_loss_numpy(
    chosen_rewards,
    rejected_rewards,
    margin=0.0,
):
    # chosen/rejected: (B,)；返回标量 loss
    diff = (
        chosen_rewards
        - rejected_rewards
        - margin
    )                                      # (B,)：reward margin Δr

    # softplus(-diff) = log(1 + exp(-diff))，即 -log σ(Δr) 的稳定写法
    return np.logaddexp(
        0.0,
        -diff,
    ).mean()                               # (B,) -> 标量
```

**最小可运行测试**

```python
# Demo
chosen = torch.tensor([2.0, 1.0, 3.0])     # (B=3,) chosen response 的 reward
rejected = torch.tensor([0.5, 1.2, 1.5])   # (B=3,) 同 prompt 下 rejected 的 reward，第 2 对为逆序样本

print(
    "RM loss:",
    reward_pairwise_loss(
        chosen,
        rejected,
    ).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def reward_pairwise_loss(
    chosen_rewards: Tensor,
    rejected_rewards: Tensor,
    margin: float = 0.0,
) -> Tensor:
    # 待实现：两个 (B,) reward -> 标量 pairwise loss
    pass
```

**Step 1：Reward difference**

（i）公式

$$
\Delta r=r_c-r_r-m
$$

（ii）代码

```python
diff = (
    chosen_rewards
    - rejected_rewards
    - margin
)                                      # (B,)：逐 pair 的 reward margin Δr
```

（iii）只关心相对排序，不要求 reward 具有绝对标定尺度。

**Step 2：Preference likelihood**

（i）公式

$$
P(c\succ r)=\sigma(\Delta r)
$$

（ii）代码语义

```python
F.logsigmoid(diff)                     # (B,) -> (B,)：log σ(Δr)，即 chosen 被偏好的 log-likelihood
```

（iii）$$\Delta r$$ 越大，chosen 被偏好的概率越接近 1。

**Step 3：Negative log-likelihood**

（i）公式

$$
L=-\log\sigma(\Delta r)
$$

（ii）代码

```python
return -F.logsigmoid(diff).mean()      # 取负后沿 batch 平均：(B,) -> 标量
```

（iii）不要手写 `-torch.log(torch.sigmoid(diff))`；`logsigmoid` 数值更稳定。

#### 👀 A2. KL Divergence / KL Penalty（Policy vs Reference）

##### 1️⃣ Dense Opening

RLHF / preference optimization 中 KL penalty 用来限制 policy 远离 reference policy。核心对象是 token distribution 上的 $$D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$$。面试最稳妥的实现是直接从两个 logits 做 `log_softmax`，再按 policy probability 加权求和；若做 sampled-KL，才会退化成采样 token 上的 log-ratio。[参考 7]

##### 2️⃣ Global Equations

精确 categorical KL：

$$
D_{\mathrm{KL}}
\left(
\pi_\theta\|
\pi_{\mathrm{ref}}
\right)
=
\sum_{v=1}^{V}
\pi_\theta(v)
\left[
\log\pi_\theta(v)
-
\log\pi_{\mathrm{ref}}(v)
\right]
$$

带 token mask：

$$
L_{\mathrm{KL}}
=
\frac{
\sum_{b,t}
m_{b,t}
D_{\mathrm{KL}}^{(b,t)}
}{
\sum_{b,t}m_{b,t}
}
$$

采样 token 上常见 log-ratio：

$$
\Delta\log\pi_t
=
\log\pi_\theta(y_t)
-
\log\pi_{\mathrm{ref}}(y_t)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（精确 vocabulary KL）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def kl_penalty(
    policy_logits: Tensor,
    ref_logits: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # logits: (B,T,V)；mask: (B,T) 1 表示该 token 参与统计（通常只算 response 部分）
    # 返回: 标量 KL penalty
    log_p = F.log_softmax(
        policy_logits,
        dim=-1,
    )                                      # (B,T,V)：policy 的 log π_θ，沿 vocab 归一化
    log_q = F.log_softmax(
        ref_logits,
        dim=-1,
    )                                      # (B,T,V)：reference 的 log π_ref

    p = log_p.exp()                        # (B,T,V)：policy 概率，作为 KL 的加权权重

    # 对 vocabulary 求和 -> 每个 token 一个 KL
    token_kl = (
        p * (log_p - log_q)
    ).sum(dim=-1)                          # (B,T,V) -> (B,T)：聚合掉 vocab 维

    if mask is None:
        return token_kl.mean()             # 无 mask 时所有位置等权平均

    mask = mask.to(token_kl.dtype)         # (B,T)：int mask 转成浮点，才能参与乘法与求和
    return (
        (token_kl * mask).sum()
        / mask.sum().clamp_min(1.0)        # 除以有效 token 数；clamp 防全零 mask 除零
    )
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def log_softmax_numpy(x, axis=-1):
    # x: 任意 shape，沿 axis 归一化；返回同 shape 的 log-probability
    x = x - x.max(
        axis=axis,
        keepdims=True,
    )                                      # 减去逐行最大值（keepdims 保持可 broadcast），防 exp 溢出
    return x - np.log(
        np.exp(x).sum(
            axis=axis,
            keepdims=True,
        )
    )                                      # x - logsumexp(x)

def kl_penalty_numpy(
    policy_logits,
    ref_logits,
    mask=None,
):
    # policy_logits/ref_logits: (B,T,V)；mask: (B,T) 或 None
    log_p = log_softmax_numpy(
        policy_logits,
        axis=-1,
    )                                      # (B,T,V)
    log_q = log_softmax_numpy(
        ref_logits,
        axis=-1,
    )                                      # (B,T,V)

    p = np.exp(log_p)                      # (B,T,V)：policy 概率权重
    token_kl = np.sum(
        p * (log_p - log_q),
        axis=-1,
    )                                      # (B,T)

    if mask is None:
        return token_kl.mean()

    mask = mask.astype(np.float64)         # (B,T)：0/1 转 float 以便加权求和
    return (
        (token_kl * mask).sum()
        / max(mask.sum(), 1.0)             # 只对有效 token 平均，下界 1.0 防除零
    )
```

**最小可运行测试**

```python
# Demo
policy = torch.randn(2, 4, 10)             # (B=2, T=4, V=10) policy logits
ref = torch.randn(2, 4, 10)                # 同 shape 的 reference logits
mask = torch.tensor([                      # (B=2, T=4)：两条序列的有效长度分别为 3 和 2
    [1, 1, 1, 0],
    [1, 1, 0, 0],
])

print(
    "KL:",
    kl_penalty(
        policy,
        ref,
        mask,
    ).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def kl_penalty(
    policy_logits: Tensor,
    ref_logits: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    # 待实现：两组 (B,T,V) logits + (B,T) mask -> 标量 KL
    pass
```

**Step 1：转成 log-probability**

（i）公式

$$
\log p_v
=
z_v-\operatorname{logsumexp}(z)
$$

（ii）代码

```python
log_p = F.log_softmax(
    policy_logits,
    dim=-1,
)                                      # (B,T,V)：沿 vocab 维归一化的 log π_θ
log_q = F.log_softmax(
    ref_logits,
    dim=-1,
)                                      # (B,T,V)：log π_ref
```

（iii）直接对 logits 做稳定 log-softmax，不需要先 softmax 再 log。

**Step 2：Vocabulary KL**

（i）公式

$$
D_{\mathrm{KL}}
=
\sum_v
p_v(\log p_v-\log q_v)
$$

（ii）代码

```python
p = log_p.exp()                        # (B,T,V)：加权用的 policy 概率

token_kl = (
    p * (log_p - log_q)
).sum(dim=-1)                          # (B,T,V) -> (B,T)：每个位置一个 KL
```

（iii）`(B,T,V) -> (B,T)`，只在 vocabulary 维聚合。

**Step 3：Masked reduction**

（i）公式

$$
L
=
\frac{
\sum m_{bt}\mathrm{KL}_{bt}
}{
\sum m_{bt}
}
$$

（ii）代码

```python
return (
    (token_kl * mask).sum()            # (B,T) 逐元素屏蔽 padding 后求和为标量
    / mask.sum().clamp_min(1.0)        # 除以有效 token 数，clamp 防除零
)
```

（iii）RLHF 中通常只对 response token 计算 KL。

#### A3. InfoNCE Loss（In-Batch Contrastive Loss）

##### 1️⃣ Dense Opening

InfoNCE 把每个 query 与 batch 中对应的 positive key 配对，其余 key 作为 negatives。核心对象是温度缩放后的相似度矩阵 $$S=QK^\top/\tau$$；若正样本位于对角线，整个损失直接等价于对 $$S$$ 做 cross-entropy，targets 为 `arange(B)`。[参考 9]

##### 2️⃣ Global Equations

L2 normalize 后：

$$
\tilde q_i
=
\frac{q_i}{\|q_i\|_2},
\qquad
\tilde k_j
=
\frac{k_j}{\|k_j\|_2}
$$

similarity：

$$
s_{ij}
=
\frac{
\tilde q_i^\top\tilde k_j
}{
\tau
}
$$

loss：

$$
\mathcal L
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
e^{s_{ii}}
}{
\sum_{j=1}^{B}e^{s_{ij}}
}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def infonce_loss(
    queries: Tensor,
    keys: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    # queries, keys: (B,D)，keys[i] 是 queries[i] 的正样本；temperature 越小分布越尖锐
    q = F.normalize(
        queries,
        dim=-1,
    )                                      # (B,D)：按行做 L2 归一化，使点积等价于 cosine
    k = F.normalize(
        keys,
        dim=-1,
    )                                      # (B,D)

    # (B,D) @ (D,B) -> (B,B)
    # 第 i 行第 i 列为正样本
    scores = (
        q @ k.T                            # k.T: (B,D) -> (D,B)，得到 query×key 全对相似度
    ) / temperature

    targets = torch.arange(
        queries.size(0),
        device=queries.device,
    )                                      # (B,)：第 i 行的正类索引就是 i（对角线）

    return F.cross_entropy(
        scores,
        targets,
    )                                      # 沿每行 softmax 后取对角项负 log，平均成标量
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def infonce_loss_numpy(
    queries,
    keys,
    temperature=0.07,
):
    # queries, keys: (B,D)；返回标量 loss
    q = queries / np.linalg.norm(
        queries,
        axis=-1,
        keepdims=True,                     # 保留 (B,1) 以便与 (B,D) broadcast 相除
    )
    k = keys / np.linalg.norm(
        keys,
        axis=-1,
        keepdims=True,
    )

    scores = (
        q @ k.T
    ) / temperature                        # (B,D)@(D,B) -> (B,B)

    scores = scores - scores.max(
        axis=-1,
        keepdims=True,
    )                                      # 逐行减最大值，稳定后续 exp
    log_probs = scores - np.log(
        np.exp(scores).sum(
            axis=-1,
            keepdims=True,
        )
    )                                      # (B,B)：行内 log-softmax

    B = queries.shape[0]
    return -log_probs[
        np.arange(B),
        np.arange(B),                      # 双 arange 花式索引取对角线：(B,B) -> (B,)
    ].mean()
```

**最小可运行测试**

```python
# Demo
q = torch.randn(4, 16)                     # (B=4, D=16) query 向量
k = q + 0.1 * torch.randn(4, 16)           # 正样本 = query 加小扰动，故对角线相似度应最高

print(
    "InfoNCE:",
    infonce_loss(q, k).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def infonce_loss(
    queries: Tensor,
    keys: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    # 待实现：两组 (B,D) 向量 -> 标量 InfoNCE loss（对角线为正样本）
    pass
```

**Step 1：Normalize**

（i）公式

$$
\tilde q_i=q_i/\|q_i\|_2
$$

（ii）代码

```python
q = F.normalize(queries, dim=-1)       # (B,D) -> (B,D)：每行单位化
k = F.normalize(keys, dim=-1)
```

（iii）归一化后点积就是 cosine similarity。

**Step 2：构造全 batch similarity matrix**

（i）shape

$$
(B,D)(D,B)\rightarrow(B,B)
$$

（ii）代码

```python
scores = (
    q @ k.T                            # (B,D)@(D,B) -> (B,B) 相似度矩阵
) / temperature                        # 温度缩放，控制 softmax 尖锐度
```

（iii）第 $$i$$ 行表示 query $$i$$ 与全部 keys 的相似度。

**Step 3：对角线为正类**

（i）公式

$$
y_i=i
$$

（ii）代码

```python
targets = torch.arange(
    queries.size(0),
    device=queries.device,             # 与 scores 同 device，否则 cross_entropy 报错
)                                      # (B,)：类别标签 = 自身行号
```

（iii）这就是 in-batch negatives 的索引核心。

**Step 4：Cross-Entropy**

（i）公式

$$
L_i
=
-\log
\frac{e^{s_{ii}}}
{\sum_j e^{s_{ij}}}
$$

（ii）代码

```python
return F.cross_entropy(
    scores,                            # (B,B) 当作 B 类 logits
    targets,                           # (B,) 正类索引
)
```

（iii）无需显式写 softmax；cross-entropy 内部使用稳定 log-softmax。

#### A4. Pre-Norm vs Post-Norm Transformer

##### 1️⃣ Dense Opening

Pre-Norm 与 Post-Norm 的区别只在 LayerNorm 相对 residual branch 的位置。Pre-Norm 先 norm 再 sublayer，residual 主干保持直接恒等路径；Post-Norm 先 residual 相加再 norm。核心对象是 residual recurrence。面试通常要求写公式、指出训练稳定性差异，并能写出两个 forward。[参考 10]

##### 2️⃣ Global Equations

Pre-Norm：

$$
X_1
=
X+
F(\operatorname{LN}(X))
$$

$$
X_2
=
X_1+
G(\operatorname{LN}(X_1))
$$

Post-Norm：

$$
X_1
=
\operatorname{LN}
\left(
X+F(X)
\right)
$$

$$
X_2
=
\operatorname{LN}
\left(
X_1+G(X_1)
\right)
$$

其中 $$F$$ 通常是 attention，$$G$$ 是 FFN。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（两种 forward 并列）**

```python
import torch
import torch.nn as nn

class NormPlacementBlock(nn.Module):
    def __init__(
        self,
        d_model,
        attn,
        ffn,
    ):
        super().__init__()
        self.attn = attn                   # sublayer F：(B,T,D) -> (B,T,D)
        self.ffn = ffn                     # sublayer G：(B,T,D) -> (B,T,D)
        self.ln1 = nn.LayerNorm(d_model)   # 沿最后一维 D 归一化，attention 分支用
        self.ln2 = nn.LayerNorm(d_model)   # FFN 分支用

    def forward_pre_norm(self, x):
        # x: (B,T,D) -> shape 不变
        x = x + self.attn(self.ln1(x))     # 先 norm 再进 sublayer，residual 主干是纯恒等路径
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_post_norm(self, x):
        # residual 后再 LayerNorm
        x = self.ln1(
            x + self.attn(x)               # 恒等路径也被 LN 缩放，深层易梯度不稳
        )
        x = self.ln2(
            x + self.ffn(x)
        )
        return x
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def layer_norm_numpy(
    x, gamma, beta, eps=1e-5,   # x:(B,T,D)；gamma/beta:(D,) 逐通道仿射；eps 防除零
):
    mean = x.mean(
        axis=-1,                # 沿特征维 D 聚合 -> (B,T,1)，keepdims 便于广播回 (B,T,D)
        keepdims=True,
    )
    var = ((x - mean) ** 2).mean(
        axis=-1,
        keepdims=True,          # (B,T,1)
    )
    return (
        gamma
        * (x - mean)
        / np.sqrt(var + eps)    # 标准化到零均值单位方差，再做仿射；输出仍是 (B,T,D)
        + beta
    )

def pre_norm_block_numpy(
    x,                          # (B,T,D)
    attn_fn,
    ffn_fn,
    g1,                         # ln1 的 gamma/beta
    b1,
    g2,                         # ln2 的 gamma/beta
    b2,
):
    h = layer_norm_numpy(x, g1, b1)   # 只对进入 sublayer 的支路做 norm
    x = x + attn_fn(h)                # residual 加的是未归一化的 x，主干保持恒等

    h = layer_norm_numpy(x, g2, b2)
    x = x + ffn_fn(h)
    return x

def post_norm_block_numpy(
    x,
    attn_fn,
    ffn_fn,
    g1,
    b1,
    g2,
    b2,
):
    x = layer_norm_numpy(
        x + attn_fn(x),         # 先相加再归一化：LN 位于 residual 之后
        g1,
        b1,
    )
    x = layer_norm_numpy(
        x + ffn_fn(x),
        g2,
        b2,
    )
    return x
```

**最小可运行测试**

```python
# Demo
D = 8                              # d_model：特征维
attn = nn.Linear(D, D, bias=False) # 用线性层假扮 attention sublayer F
ffn = nn.Sequential(
    nn.Linear(D, 16),              # 中间维放大 2 倍，(…,D) -> (…,16)
    nn.ReLU(),
    nn.Linear(16, D),              # 再投回 D，保证 residual 可加
)

block = NormPlacementBlock(
    D,
    attn,
    ffn,
)
x = torch.randn(2, 4, D)           # (B=2, T=4, D=8)

print(
    block.forward_pre_norm(x).shape,
    block.forward_post_norm(x).shape,
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class NormPlacementBlock(nn.Module):
    def __init__(
        self,
        d_model,   # 特征维 D，决定两个 LayerNorm 的归一化长度
        attn,      # sublayer F：(B,T,D) -> (B,T,D)
        ffn,       # sublayer G：(B,T,D) -> (B,T,D)
    ):
        super().__init__()
        pass       # 待填：保存 attn/ffn 并建 ln1、ln2
```

**Step 1：Pre-Norm attention branch**

（i）公式

$$
X_1=X+F(\operatorname{LN}(X))
$$

（ii）代码

```python
x = x + self.attn(
    self.ln1(x)   # 只归一化支路输入，(B,T,D) -> (B,T,D)
)                 # residual 加的是原始 x，梯度可无阻塞回传
```

（iii）residual identity path 不穿过 LayerNorm。

**Step 2：Pre-Norm FFN branch**

（i）公式

$$
X_2=X_1+G(\operatorname{LN}(X_1))
$$

（ii）代码

```python
x = x + self.ffn(
    self.ln2(x)   # 第二个 LN 用独立参数，(B,T,D) -> (B,T,D)
)
```

（iii）深层网络中这条 identity path 有利于梯度直接传播。

**Step 3：Post-Norm**

（i）公式

$$
X_1=\operatorname{LN}(X+F(X))
$$

（ii）代码

```python
x = self.ln1(
    x + self.attn(x)   # 先做 residual 相加，再整体归一化 (B,T,D)
)
```

（iii）原始 Transformer 使用 Post-Norm；训练很深网络时通常对 warmup / initialization 更敏感。


#### A5. Sinusoidal Positional Encoding

##### 1️⃣ Dense Opening

Sinusoidal positional encoding 直接用不同频率的 sine/cosine 给每个绝对位置生成一个 $$D$$ 维向量，再与 token embedding 相加。核心对象是位置 $$pos$$ 与维度频率 $$10000^{-2i/D}$$。面试实现重点是偶数维用 sin、奇数维用 cos，以及广播 shape。[参考 2]

##### 2️⃣ Global Equations

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}{
10000^{2i/D}
}
\right)
$$

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{
10000^{2i/D}
}
\right)
$$

输入 embedding：

$$
X'
=
X+PE
$$

其中：

$$
X:(B,T,D),
\qquad
PE:(T,D)
$$

通过 batch 维广播相加。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
from torch import Tensor

def sinusoidal_position_encoding(
    seq_len: int,
    d_model: int,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    # 返回 (T,D) 的绝对位置编码表，可与 (B,T,D) embedding 广播相加
    assert d_model % 2 == 0                # sin/cos 成对占用，D 必须是偶数

    pos = torch.arange(
        seq_len,
        device=device,
        dtype=dtype,
    )[:, None]                             # (T,1)

    freq = 10000.0 ** (
        -torch.arange(
            0,
            d_model,
            2,                             # 步长 2 取偶数下标 2i -> (D/2,)
            device=device,
            dtype=dtype,
        ) / d_model
    )                                      # (D/2,)

    angle = pos * freq[None, :]             # (T,1)*(1,D/2) 广播 -> (T,D/2)

    pe = torch.zeros(
        seq_len,
        d_model,
        device=device,
        dtype=dtype,
    )
    pe[:, 0::2] = torch.sin(angle)         # 偶数列写 sin，(T,D/2) 填入 D 的偶数位
    pe[:, 1::2] = torch.cos(angle)         # 奇数列写 cos，与偶数列共享同一频率
    return pe
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def sinusoidal_position_encoding_numpy(
    seq_len,
    d_model,
):
    assert d_model % 2 == 0                # sin/cos 成对占列

    pos = np.arange(seq_len)[:, None]       # (T,1)：位置轴，补一维便于广播
    freq = 10000.0 ** (
        -np.arange(0, d_model, 2)          # 偶数下标 2i，共 D/2 个频率
        / d_model
    )                                      # (D/2,)

    angle = pos * freq[None, :]             # (T,1)*(1,D/2) -> (T,D/2)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle)            # 偶数列 <- sin
    pe[:, 1::2] = np.cos(angle)            # 奇数列 <- cos，交错写满 (T,D)
    return pe
```

**最小可运行测试**

```python
# Demo
pe = sinusoidal_position_encoding(
    seq_len=6,                  # T=6 个位置
    d_model=8,                  # D=8 -> 4 组 sin/cos
)
x = torch.randn(2, 6, 8)        # token embedding: (B=2, T=6, D=8)

# (B,T,D) + (T,D) -> (B,T,D)
y = x + pe
print(y.shape)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def sinusoidal_position_encoding(
    seq_len: int,   # T：需要编码的位置个数
    d_model: int,   # D：特征维，必须为偶数
) -> Tensor:
    pass            # 待填：造 pos(T,1)、freq(D/2,)，交错写入 pe(T,D)
```

**Step 1：Position axis**

（i）shape

$$
pos:(T,1)
$$

（ii）代码

```python
pos = torch.arange(
    seq_len,
    dtype=torch.float32,   # 后续要参与浮点乘法，直接建成 float
)[:, None]                 # (T,) -> (T,1)：末尾补一维用于广播
```

（iii）补一维后可以与频率向量直接广播相乘。

**Step 2：Frequency axis**

（i）公式

$$
\omega_i=10000^{-2i/D}
$$

（ii）代码

```python
freq = 10000.0 ** (
    -torch.arange(
        0, d_model, 2,        # 偶数下标 2i：0,2,...,D-2 -> (D/2,)
        dtype=torch.float32,
    ) / d_model
)                             # 频率随维度指数衰减，波长从 2π 递增到 10000·2π
```

（iii）只需要 $$D/2$$ 个频率，因为 sin/cos 共享同一频率；幂函数形式与公式直接一一对应，也更适合面试手写。

**Step 3：交错写入**

（i）公式

$$
PE_{:,0::2}=\sin(angle),
\qquad
PE_{:,1::2}=\cos(angle)
$$

（ii）代码

```python
pe[:, 0::2] = torch.sin(angle)   # angle:(T,D/2) 写入偶数列
pe[:, 1::2] = torch.cos(angle)   # 同频率的 cos 写入相邻奇数列
```

（iii）最终 `pe: (T,D)`，与 `(B,T,D)` embedding 自动广播。

#### A6. ALiBi（Attention with Linear Biases）

##### 1️⃣ Dense Opening

ALiBi 不向 embedding 加位置向量，而是直接给 attention score 加一个与 query-key 距离线性相关的 head-specific bias。核心对象是 $$-m_h(i-j)$$：距离越远，分数惩罚越大。面试通常无需背完整 slope 生成算法，重点是 bias 的 shape、符号和加入 scores 的位置。[参考 11]

##### 2️⃣ Global Equations

对 causal decoder：

$$
S_{h,i,j}
=
\frac{
q_{h,i}^{\top}k_{h,j}
}{
\sqrt{d_h}
}
-
m_h(i-j),
\qquad
j\le i
$$

其中每个 head 有独立正 slope $$m_h>0$$。

定义距离：

$$
D_{i,j}
=
i-j
$$

$$
B_{h,i,j}
=
-m_hD_{i,j}
$$

因此：

$$
S
=
QK^\top/\sqrt{d_h}
+
B
+
M_{\mathrm{causal}}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准面试实现）**

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def alibi_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    slopes: Tensor,
) -> Tensor:
    # q,k,v: (B,H,T,D)
    # slopes: (H,)；每个 head 一个正斜率 m_h，面试中可由题目给定
    B, H, T, D = q.shape                   # D 为 head_dim

    scores = torch.matmul(
        q,                                 # (B,H,T,D) @ (B,H,D,T)
        k.transpose(-2, -1),               # 交换末两维得到 k^T
    ) / math.sqrt(D)                       # (B,H,T,T)

    pos = torch.arange(
        T,
        device=q.device,
    )                                      # (T,)：token 的绝对下标

    # distance[i,j] = i-j
    distance = (
        pos[:, None] - pos[None, :]        # (T,1)-(1,T) 广播成相对距离矩阵
    )                                      # (T,T)

    # bias: (H,1,1) * (1,T,T)
    # -> (H,T,T)
    bias = (
        -slopes[:, None, None]             # 距离越远惩罚越大，故取负号
        * distance[None].to(q.dtype)       # int -> q 的浮点 dtype，避免类型不匹配
    )

    causal = distance < 0                  # j > i
    scores = scores + bias[None]           # (H,T,T) 补 batch 维后广播到 (B,H,T,T)
    scores = scores.masked_fill(
        causal[None, None],                # (T,T) -> (1,1,T,T)，屏蔽未来 token
        float("-inf"),
    )

    attn = F.softmax(scores, dim=-1)       # 沿 key 维归一化，每行 query 权重和为 1
    return torch.matmul(attn, v)           # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def alibi_attention_numpy(
    q,
    k,
    v,
    slopes,
):
    # q,k,v: (B,H,T,D)；slopes: (H,)
    B, H, T, D = q.shape

    scores = (
        q @ k.transpose(0, 1, 3, 2)        # 只换末两维 -> (B,H,D,T)，得 (B,H,T,T)
    ) / np.sqrt(D)

    pos = np.arange(T)                     # (T,)
    distance = (
        pos[:, None] - pos[None, :]        # 广播相减：distance[i,j]=i-j
    )                                      # (T,T)

    bias = (
        -slopes[:, None, None]             # (H,) -> (H,1,1)
        * distance[None]                   # (T,T) -> (1,T,T)
    )                                      # (H,T,T)

    causal = distance < 0                  # j>i 即未来位置
    scores = scores + bias[None]           # 广播到 (B,H,T,T)
    scores = np.where(
        causal[None, None],                # (1,1,T,T) 广播选择
        -1e9,                              # 用大负数而非 -inf，避免 exp 出 NaN
        scores,
    )

    scores -= scores.max(
        axis=-1,                           # 沿 key 维取行最大值 (B,H,T,1)
        keepdims=True,                     # 数值稳定：减最大值不改变 softmax 结果
    )
    a = np.exp(scores)
    a /= a.sum(axis=-1, keepdims=True)     # 归一化成注意力概率 (B,H,T,T)
    return a @ v                           # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
```

**最小可运行测试**

```python
# Demo
B, H, T, D = 2, 4, 5, 8        # batch / heads / 序列长 / head_dim
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

# 面试里重点是不同 head 使用不同 slope
slopes = torch.tensor([        # (H,)=(4,)，一般取 2^{-8k/H} 的等比数列
    1.0,
    0.5,
    0.25,
    0.125,
])

print(
    alibi_attention(
        q, k, v, slopes
    ).shape
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def alibi_attention(
    q: Tensor,        # (B,H,T,D)
    k: Tensor,        # (B,H,T,D)
    v: Tensor,        # (B,H,T,D)
    slopes: Tensor,   # (H,)：每个 head 的正斜率 m_h
) -> Tensor:
    pass              # 待填：scores -> 距离 bias -> causal mask -> softmax @ v
```

**Step 1：普通 attention scores**

（i）公式

$$
S=QK^\top/\sqrt{d_h}
$$

（ii）代码

```python
scores = torch.matmul(
    q,                     # (B,H,T,D)
    k.transpose(-2, -1),   # (B,H,D,T)：末两维转置
) / math.sqrt(D)           # 缩放防止点积过大导致 softmax 饱和
```

（iii）shape 是 `(B,H,T,T)`。

**Step 2：构造相对距离**

（i）公式

$$
D_{ij}=i-j
$$

（ii）代码

```python
distance = (
    pos[:, None] - pos[None, :]   # (T,1)-(1,T) 广播 -> (T,T)，元素为 i-j
)
```

（iii）过去 token $$j<i$$ 的距离为正；未来 token $$j>i$$ 为负并由 causal mask 屏蔽。

**Step 3：Head-specific linear bias**

（i）公式

$$
B_{hij}=-m_hD_{ij}
$$

（ii）代码

```python
bias = (
    -slopes[:, None, None]   # (H,) -> (H,1,1)
    * distance[None]         # (T,T) -> (1,T,T)，广播成 (H,T,T)
)
scores = scores + bias[None] # 补 batch 维 (1,H,T,T)，加到 (B,H,T,T) 上
```

（iii）`bias: (H,T,T)` 在 batch 维广播。

**Step 4：Causal softmax**

（i）代码

```python
scores = scores.masked_fill(
    causal[None, None],   # (T,T) -> (1,1,T,T)，True 处（未来 token）被填充
    float("-inf"),
)
attn = F.softmax(scores, dim=-1)   # 沿 key 维归一化，被屏蔽位置权重为 0
```

（iii）ALiBi 是 score bias，不替代 causal mask。

#### A7. Attention Padding Mask + Causal Mask

##### 1️⃣ Dense Opening

实际 batch 中 attention mask 通常同时处理两件事：causal 约束保证不能看未来，padding mask 保证不能把 `<pad>` 当作 key。核心对象是一个可广播到 `(B,H,T_q,T_k)` 的 boolean visibility mask。面试最容易写错的是 padding mask 只屏蔽 key 维，或 causal / padding 的 0/1 语义相反。

##### 2️⃣ Global Equations

causal 可见条件：

$$
C_{i,j}
=
\mathbb{1}[j\le i]
$$

key padding 可见条件：

$$
P_{b,j}
=
\mathbb{1}
[\text{token}_{b,j}\text{ valid}]
$$

组合：

$$
M_{b,i,j}
=
C_{i,j}
\land
P_{b,j}
$$

attention：

$$
S_{b,h,i,j}
\leftarrow
-\infty
\quad
\text{if }M_{b,i,j}=0
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def masked_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    padding_mask: Tensor,
) -> Tensor:
    # q,k,v: (B,H,T,D)
    # padding_mask: (B,T)，1=valid, 0=pad
    B, H, T, D = q.shape

    scores = torch.matmul(
        q,                                 # (B,H,T,D)
        k.transpose(-2, -1),               # (B,H,D,T)：末两维转置
    ) / math.sqrt(D)                       # (B,H,T,T)

    # causal: (1,1,T,T)
    causal = torch.tril(
        torch.ones(
            T, T,
            dtype=torch.bool,              # 直接建 bool，便于后面按位与
            device=q.device,
        )
    )[None, None]                          # 下三角 True=可见，(T,T) -> (1,1,T,T)

    # key padding: (B,T)
    # -> (B,1,1,T)，只在 key 维屏蔽
    key_valid = padding_mask[
        :, None, None, :                   # 插入 head 维与 query 维，末维对齐 T_k
    ].bool()                               # 0/1 int -> bool

    # broadcast -> (B,1,T,T)
    visible = causal & key_valid           # 同时满足「不看未来」与「key 非 pad」

    scores = scores.masked_fill(
        ~visible,                          # 取反：True 处为不可见，需填负无穷
        float("-inf"),
    )

    attn = F.softmax(scores, dim=-1)       # 沿 key 维归一化 (B,H,T,T)
    return torch.matmul(attn, v)           # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
```

**（B）NumPy 版本（苛刻面试版）**

```python
import numpy as np

def masked_attention_numpy(
    q,
    k,
    v,
    padding_mask,
):
    # q,k,v: (B,H,T,D)；padding_mask: (B,T)，1=valid
    B, H, T, D = q.shape

    scores = (
        q @ k.transpose(0, 1, 3, 2)       # 换末两维 -> (B,H,D,T)，得 (B,H,T,T)
    ) / np.sqrt(D)

    causal = np.tril(
        np.ones((T, T), dtype=bool)       # 下三角 True：query i 只能看 j<=i
    )[None, None]                         # (1,1,T,T)

    key_valid = padding_mask[
        :, None, None, :                  # 补 head 维与 query 维，只约束 key 维
    ].astype(bool)                        # (B,1,1,T)

    visible = causal & key_valid          # (B,1,T,T)

    scores = np.where(
        visible,                          # 可见处保留原分数
        scores,
        -1e9,                             # 不可见处置大负数，softmax 后趋近 0
    )
    scores -= scores.max(
        axis=-1,                          # 行最大值 (B,H,T,1)，数值稳定用
        keepdims=True,
    )
    a = np.exp(scores)
    a /= a.sum(axis=-1, keepdims=True)    # 沿 key 维归一化成概率
    return a @ v                          # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
```

**最小可运行测试**

```python
# Demo
B, H, T, D = 2, 4, 5, 8        # batch / heads / 序列长 / head_dim
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

padding_mask = torch.tensor([  # (B,T)=(2,5)，1=真实 token，0=右侧 <pad>
    [1, 1, 1, 1, 0],           # 样本 0：有效长度 4
    [1, 1, 1, 0, 0],           # 样本 1：有效长度 3
])

print(
    masked_attention(
        q, k, v, padding_mask
    ).shape
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def masked_attention(
    q: Tensor,              # (B,H,T,D)
    k: Tensor,              # (B,H,T,D)
    v: Tensor,              # (B,H,T,D)
    padding_mask: Tensor,   # (B,T)：1=valid，0=pad
) -> Tensor:
    pass                    # 待填：scores -> causal & key_valid -> masked softmax @ v
```

**Step 1：Causal visibility**

（i）shape

$$
C:(1,1,T,T)
$$

（ii）代码

```python
causal = torch.tril(
    torch.ones(
        T, T,               # (T_q, T_k) 全 1
        dtype=torch.bool,
        device=q.device,
    )
)[None, None]               # 取下三角后补 batch/head 维：(T,T) -> (1,1,T,T)
```

（iii）下三角为 True 表示可见。

**Step 2：Padding visibility**

（i）shape

$$
(B,T)
\rightarrow
(B,1,1,T)
$$

（ii）代码

```python
key_valid = padding_mask[
    :, None, None, :   # (B,T) -> (B,1,1,T)：只对齐 key 维，不限制 query 维
].bool()               # 0/1 -> bool，便于与 causal 做按位与
```

（iii）padding 是 key 是否可被关注，因此放在最后一维 $$T_k$$。

**Step 3：逻辑与**

（i）公式

$$
M=C\land P
$$

（ii）代码

```python
visible = causal & key_valid   # (1,1,T,T) & (B,1,1,T) -> (B,1,T,T)，两条件同时成立才可见
```

（iii）broadcast 后得到 `(B,1,T,T)`，再广播到所有 heads。

**Step 4：Mask scores**

（i）代码

```python
scores = scores.masked_fill(
    ~visible,          # 取反后 True 表示需要屏蔽的位置
    float("-inf"),
)
```

（iii）不可见位置经过 softmax 后概率为 0。

#### A8. QLoRA（4-bit Frozen Base + LoRA，面试简化版）

##### 1️⃣ Dense Opening

真实 QLoRA 的关键是：冻结 4-bit quantized base model，只训练 LoRA adapters；原论文使用 NF4（NormalFloat 4-bit）、double quantization 和 paged optimizers。面试手写不适合复刻 bitsandbytes kernel，因此下面用**对称 4-bit per-row quantization**替代 NF4，只保留真正需要手写的计算图：`quantize frozen W -> dequantize for matmul -> add LoRA low-rank update`。必须能明确指出该简化与真实 QLoRA 的差异。[参考 12]

##### 2️⃣ Global Equations

LoRA update：

$$
\Delta W
=
\frac{\alpha}{r}BA
$$

$$
y
=
x\widetilde W^\top
+
\frac{\alpha}{r}
xA^\top B^\top
$$

其中 $$\widetilde W$$ 是 4-bit 权重反量化后的近似。

面试版对称 4-bit quantization：

$$
s_i
=
\frac{
\max_j|W_{ij}|
}{7}
$$

$$
Q_{ij}
=
\operatorname{clip}
\left(
\operatorname{round}
\left(
\frac{W_{ij}}{s_i}
\right),
-8,
7
\right)
$$

$$
\widetilde W_{ij}
=
s_iQ_{ij}
$$

真实 QLoRA 则使用 NF4 codebook，而不是上述 uniform int4。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（面试简化实现）**

```python
import torch
import torch.nn as nn

def quantize_int4_per_row(weight):
    # weight: (O,I)，O 为输出通道数、I 为输入维度
    # 每个输出通道独立 scale: (O,1)；返回 q:(O,I) int4 码值 + scale
    scale = (
        weight.abs().amax(
            dim=1,                         # 沿输入维取每行绝对值最大：(O,I) -> (O,1)
            keepdim=True,
        )
        / 7.0                              # 正半轴最多 7 级，故步长 = rowmax/7
    ).clamp_min(1e-8)                      # 防全零行使 scale=0 导致除零

    q = torch.round(
        weight / scale                     # 按行归一到 int4 量级，scale:(O,1) broadcast 到 (O,I)
    ).clamp(-8, 7).to(torch.int8)          # int8 容器存 4-bit 值
    return q, scale

class SimpleQLoRALinear(nn.Module):
    def __init__(
        self,
        weight,
        rank=4,
        alpha=8.0,
    ):
        # weight: (O,I) 预训练权重，量化后冻结；rank r、alpha 决定 LoRA 容量与缩放
        super().__init__()
        out_features, in_features = weight.shape

        q, scale = quantize_int4_per_row(
            weight.detach()                # detach：base 权重不参与梯度
        )
        self.register_buffer(              # 注册为 buffer 而非 Parameter，故不进 optimizer
            "qweight",
            q,
        )
        self.register_buffer(
            "scale",
            scale,
        )

        # 只训练 LoRA A/B
        self.A = nn.Parameter(
            0.01 * torch.randn(            # A: (r,I) 小随机初始化，负责降维投影
                rank,
                in_features,
            )
        )
        self.B = nn.Parameter(
            torch.zeros(                   # B: (O,r) 零初始化，保证训练起点 ΔW=0
                out_features,
                rank,
            )
        )
        self.scaling = alpha / rank        # α/r：让 LoRA 幅度与 rank 解耦

    def forward(self, x):
        # x: (...,I)；返回 (...,O)
        # qweight: (O,I) -> dequant W: (O,I)
        W = (
            self.qweight.to(x.dtype)       # int8 -> 浮点，才能与 scale 相乘
            * self.scale.to(x.dtype)       # scale:(O,1) broadcast 逐行还原量级
        )

        base = x @ W.T                    # (...,I) @ (I,O) -> (...,O)：冻结基座分支
        lora = (
            x @ self.A.T @ self.B.T        # (...,I)->(...,r)->(...,O)：低秩旁路
        ) * self.scaling                  # (...,O)
        return base + lora
```

**（B）NumPy 版本（同一简化计算图）**

```python
import numpy as np

def quantize_int4_per_row_numpy(weight):
    # weight: (O,I)；返回 q:(O,I) int4 码值、scale:(O,1) 每行步长
    scale = (
        np.max(
            np.abs(weight),
            axis=1,                        # 沿输入维聚合掉列：(O,I) -> (O,1)
            keepdims=True,
        )
        / 7.0
    )
    scale = np.maximum(scale, 1e-8)        # 下限保护，防除零

    q = np.clip(
        np.round(weight / scale),          # scale broadcast 到 (O,I) 后取整
        -8,
        7,                                 # 对称 int4 表示范围
    ).astype(np.int8)
    return q, scale

def qlora_linear_numpy(
    x,
    qweight,
    scale,
    A,
    B,
    alpha,
):
    # x: (...,I)；qweight:(O,I)；scale:(O,1)；A:(r,I)；B:(O,r)。返回 (...,O)
    rank = A.shape[0]                      # r：LoRA 秩，用于恢复 α/r 缩放
    W = qweight.astype(
        np.float32                         # int8 -> float32 才能反量化
    ) * scale                              # (O,I)

    base = x @ W.T                         # (...,O)
    lora = (
        x @ A.T @ B.T                      # (...,I)->(...,r)->(...,O)
    ) * (alpha / rank)
    return base + lora
```

**最小可运行测试**

```python
# Demo
weight = torch.randn(8, 16)        # (O=8, I=16) 假装是预训练权重
layer = SimpleQLoRALinear(
    weight,
    rank=4,
    alpha=8.0,
)
x = torch.randn(2, 16)             # (B=2, I=16)

y = layer(x)                       # (2,16) -> (2,8)
print("output:", y.shape)
print(
    "trainable:",
    sum(                           # 只统计 requires_grad 的参数，即 A/B，应为 r*I + O*r
        p.numel()
        for p in layer.parameters()
        if p.requires_grad
    ),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class SimpleQLoRALinear(nn.Module):
    def __init__(
        self,
        weight,
        rank=4,
        alpha=8.0,
    ):
        # weight: (O,I) 待量化冻结的基座权重；rank/alpha 为 LoRA 超参
        super().__init__()
        pass                       # 量化 weight 存为 buffer + 建 A/B 两个可训练参数

    def forward(self, x):
        # x: (...,I) -> 返回 (...,O)
        pass                       # dequant 基座分支 + LoRA 低秩分支相加
```

**Step 1：量化冻结 base weight**

（i）面试版公式

$$
Q
=
\operatorname{round}(W/s)
$$

（ii）代码

```python
q, scale = quantize_int4_per_row(
    weight.detach()                # (O,I) -> q:(O,I) int8 码值 + scale:(O,1)，detach 断梯度
)
```

（iii）`qweight` 不参与梯度；真实 QLoRA 的量化格式是 NF4，这里只用 uniform int4 表达“冻结量化基座”计算图。

**Step 2：反量化**

（i）公式

$$
\widetilde W=sQ
$$

（ii）代码

```python
W = (
    self.qweight.to(x.dtype)       # int8 码值 -> x 的浮点 dtype
    * self.scale.to(x.dtype)       # (O,1) broadcast 乘 (O,I)，还原近似权重
)
```

（iii）面试逻辑上是 dequantize-on-the-fly；工业实现不会每次显式生成完整 FP 权重。

**Step 3：Base branch**

（i）公式

$$
y_{\mathrm{base}}
=
x\widetilde W^\top
$$

（ii）代码

```python
base = x @ W.T                     # (...,I) @ (I,O) -> (...,O)：W.T 转置以匹配最后一维
```

（iii）冻结基座没有 trainable gradient。

**Step 4：LoRA branch**

（i）公式

$$
y_{\mathrm{LoRA}}
=
\frac{\alpha}{r}
xA^\top B^\top
$$

（ii）代码

```python
lora = (
    x @ self.A.T @ self.B.T        # (...,I)->(...,r) 压缩到低秩，再 ->(...,O) 还原
) * self.scaling                   # 乘 α/r
```

（iii）只有 $$A,B$$ 更新；这就是 QLoRA 的可训练部分。

**Step 5：合并输出**

（i）公式

$$
y=y_{\mathrm{base}}+y_{\mathrm{LoRA}}
$$

（ii）代码

```python
return base + lora                 # 两分支同为 (...,O)，逐元素相加
```

（iii）面试追问真实 QLoRA 时补充：NF4、double quantization、paged optimizer。


#### A9. Gradient Checkpointing（Activation Checkpointing）

##### 1️⃣ Dense Opening

Gradient checkpointing 用额外 forward compute 换 activation memory：前向时只保留 checkpoint 边界输入，不保存内部 activation；反向需要这些 activation 时重新执行该 block 的 forward。核心对象是“save fewer activations, recompute on backward”。PyTorch 用 `torch.utils.checkpoint.checkpoint`；NumPy 版用同一个两层 MLP 手动展示“只保存输入、反向时重算 $$Z,H$$”这一机制。[参考 13]

##### 2️⃣ Global Equations

普通反向需要保存：

$$
Z_1=XW_1+b_1,
\qquad
H=\phi(Z_1)
$$

checkpoint 后前向只保留边界 $$X$$，反向重新计算：

$$
(X)
\xrightarrow{\text{recompute}}
Z_1
\xrightarrow{}
H
\xrightarrow{}
Y
$$

内存/计算权衡可概括为：

$$
\text{less saved activation memory}
\Longleftrightarrow
\text{extra forward recomputation}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
    ):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model,
            d_ff,
        )
        self.fc2 = nn.Linear(
            d_ff,
            d_model,                       # 降回 D，保证 block 输入输出同形
        )

    def _core(self, x):
        # x: (...,D)
        # 中间 h: (...,d_ff)
        h = torch.relu(
            self.fc1(x)                    # (...,D) -> (...,d_ff)
        )
        return self.fc2(h)                 # (...,D)

    def forward(self, x):
        # forward 时不保存 _core 内部全部 activation；
        # backward 时会重新执行 _core
        return checkpoint(
            self._core,                    # 被包裹的纯前向函数，反向时按需重算
            x,                             # 唯一需要保存的 checkpoint 边界输入
            use_reentrant=False,           # 新版实现，兼容 autograd 的非重入路径
        )
```

**（B）NumPy 版本（手写 recompute 机制）**

```python
import numpy as np

def checkpointed_mlp_forward_numpy(
    x,
    W1,
    b1,
    W2,
    b2,
):
    # x:(B,D)；W1:(D,d_ff)、b1:(d_ff,)；W2:(d_ff,D)、b2:(D,)
    # 前向输出需要算完整图，但“checkpoint”
    # 只把边界输入 x 留给 backward
    z = x @ W1 + b1                       # (B,D)@(D,d_ff) -> (B,d_ff)，b1 沿 batch broadcast
    h = np.maximum(z, 0.0)                # ReLU，shape 不变
    y = h @ W2 + b2                       # (B,d_ff)@(d_ff,D) -> (B,D)

    saved = x                             # 不保存 z / h
    return y, saved

def checkpointed_mlp_backward_numpy(
    dy,
    saved_x,
    W1,
    b1,
    W2,
    b2,
):
    # dy:(B,D) 上游梯度；saved_x:(B,D) 是前向唯一保留的边界
    # backward 时从 saved_x 重算 activation
    z = saved_x @ W1 + b1                # recompute
    h = np.maximum(z, 0.0)               # recompute

    dW2 = h.T @ dy                       # (d_ff,B)@(B,D) -> (d_ff,D)：沿 batch 累加
    db2 = dy.sum(axis=0)                 # 聚合掉 batch 维：(B,D) -> (D,)

    dh = dy @ W2.T                       # (B,D)@(D,d_ff) -> (B,d_ff)
    dz = dh * (z > 0)                    # ReLU 反向：mask 掉 z<=0 位置的梯度

    dW1 = saved_x.T @ dz                 # (D,B)@(B,d_ff) -> (D,d_ff)
    db1 = dz.sum(axis=0)                 # (B,d_ff) -> (d_ff,)
    dx = dz @ W1.T                       # (B,d_ff)@(d_ff,D) -> (B,D)，回传给上一层

    return dx, dW1, db1, dW2, db2
```

**最小可运行测试**

```python
# Demo
model = CheckpointedMLP(
    d_model=16,
    d_ff=64,                       # 中间维放大 4 倍，是 FFN 的常见比例
)
x = torch.randn(
    4, 16,                         # (B=4, D=16)
    requires_grad=True,            # 需要梯度，才能验证 checkpoint 的反向重算能打通
)

loss = model(x).square().mean()    # (4,16) -> 标量，聚合掉所有维
loss.backward()                    # 触发 _core 重算并回传

print("loss:", loss.item())
print("dx:", x.grad.shape)         # 与 x 同形 (4,16)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedMLP(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,                      # d_model: 输入/输出维 D；d_ff: 中间放大维
    ):
        super().__init__()
        pass                       # 建 fc1:(D->d_ff)、fc2:(d_ff->D)，前向再用 checkpoint 包裹
```

**Step 1：定义纯 block function**

（i）计算图

$$
X\rightarrow Z\rightarrow H\rightarrow Y
$$

（ii）代码

```python
def _core(self, x):
    h = torch.relu(
        self.fc1(x)                # (...,D) -> (...,d_ff)
    )
    return self.fc2(h)             # (...,d_ff) -> (...,D)
```

（iii）checkpoint 的 function 应尽量是确定性的纯前向区域。

**Step 2：只保存边界**

（i）代码

```python
return checkpoint(
    self._core,                    # 反向时会被重新执行的前向区域
    x,                             # 只有这个边界输入被保存下来
    use_reentrant=False,
)
```

（iii）PyTorch 不再像普通 eager autograd 那样保存该区域内所有中间 activation。

**Step 3：Backward recomputation**

（i）机制

$$
X
\rightarrow
\text{recompute}(Z,H)
\rightarrow
\nabla
$$

（ii）NumPy 对应

```python
z = saved_x @ W1 + b1              # (B,D)@(D,d_ff) -> (B,d_ff)，反向前重算
h = np.maximum(z, 0.0)             # ReLU 激活，供 dW2 使用
```

（iii）反向前重算中间量，因此节省 activation memory，但增加计算。

#### A10. Mixed Precision Training（FP16/BF16 + Loss Scaling）

##### 1️⃣ Dense Opening

Mixed precision 用低精度执行适合的前向算子，同时保留必要的 FP32 数值范围；FP16 训练还常配合 loss scaling 防止小梯度 underflow。核心对象是缩放损失 $$L_s=sL$$：先让 backward 得到放大后的 gradient，再除以 $$s$$ 恢复真实梯度。PyTorch 标准组合是 `autocast + GradScaler`。[参考 14]

##### 2️⃣ Global Equations

loss scaling：

$$
L_s=sL
$$

$$
\nabla_\theta L_s
=
s\nabla_\theta L
$$

optimizer step 前 unscale：

$$
g
=
\frac{g_s}{s}
$$

因此数学更新不变：

$$
\theta
\leftarrow
\theta-\eta g
$$

BF16 exponent 范围接近 FP32，通常比 FP16 更少依赖 loss scaling；FP16 更需要关注 underflow。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准 FP16 AMP 训练 step）**

```python
import torch
import torch.nn.functional as F

def amp_train_step(
    model,
    optimizer,
    scaler,
    x,
    y,
):
    # x: (B,D) 输入；y: (B,O) 目标；scaler 为 GradScaler，返回标量 loss
    optimizer.zero_grad(
        set_to_none=True               # 置 None 而非 0，省显存且避免残留累积
    )

    use_cuda = x.is_cuda               # FP16 autocast 只在 GPU 上有意义

    # CUDA 上启用 FP16 autocast；
    # CPU demo 时自动关闭，函数仍可运行
    with torch.autocast(
        device_type=x.device.type,
        dtype=torch.float16,           # GEMM 等算子降到 FP16，敏感 op 仍保持高精度
        enabled=use_cuda,
    ):
        pred = model(x)                # (B,D) -> (B,O)
        loss = F.mse_loss(
            pred,
            y,                         # 逐元素平方差再 mean 成标量
        )

    # scale -> backward -> unscale/step -> update scale
    scaler.scale(loss).backward()      # loss 乘 s 再回传，把小梯度拉出 FP16 underflow 区
    scaler.step(optimizer)             # 内部先 unscale（除 s），检测到 inf/NaN 则跳过本步
    scaler.update()                    # 按最近是否 overflow 动态调整 s
    return loss.detach()               # detach 断开计算图，只留数值
```

**（B）NumPy 版本（手写 loss scaling 示意）**

```python
import numpy as np

def mixed_precision_linear_step_numpy(
    x,
    y,
    W_fp32,
    lr=1e-2,
    loss_scale=1024.0,
):
    # x:(B,D)；y:(B,O)；W_fp32:(D,O) 是 master weight；loss_scale 即公式里的 s
    # master weight 保持 FP32
    # forward 使用 FP16 副本
    x16 = x.astype(np.float16)             # 类型转换，只用于低精度前向
    W16 = W_fp32.astype(np.float16)        # FP16 副本，master 权重本身不降精度

    # (B,D) @ (D,O) -> (B,O)
    pred16 = x16 @ W16
    pred = pred16.astype(np.float32)       # 回到 FP32 做 loss/梯度，避免累积误差

    B = x.shape[0]                         # batch size，用于把 loss 归一化为均值
    diff = pred - y.astype(np.float32)     # (B,O) 残差，也正是 MSE 对 pred 的梯度方向
    loss = 0.5 * np.sum(diff ** 2) / B     # 聚合掉 (B,O) 全部维度 -> 标量

    # scaled loss 的梯度：
    # dL_scaled/dpred = scale * dL/dpred
    d_pred_scaled = (
        loss_scale * diff / B              # (B,O)，整体放大 s 倍
    ).astype(np.float32)

    # 用 FP32 累积 weight gradient
    grad_scaled = (
        x.astype(np.float32).T             # (B,D) -> (D,B)
        @ d_pred_scaled                    # (D,B)@(B,O) -> (D,O)，沿 batch 累加
    )

    # optimizer 前 unscale
    grad = grad_scaled / loss_scale        # 除回 s，恢复真实梯度量级

    W_fp32 = W_fp32 - lr * grad            # SGD 更新只作用在 FP32 master weight 上
    return W_fp32, loss
```

**最小可运行测试**

```python
# Demo：CPU 也可以运行，CUDA 时才真正启用 FP16 AMP
model = torch.nn.Linear(4, 2)      # D=4 -> O=2
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available(),   # 无 GPU 时退化为恒等缩放（s=1）
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model = model.to(device)           # 参数搬到目标设备，需与输入同设备

x = torch.randn(8, 4, device=device)     # (B=8, D=4)
y = torch.randn(8, 2, device=device)     # (B=8, O=2)

loss = amp_train_step(
    model,
    optimizer,
    scaler,
    x,
    y,
)
print("loss:", loss.item())        # 标量取 Python float
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch

def amp_train_step(
    model,
    optimizer,
    scaler,                        # GradScaler：负责 loss scaling 与 inf/NaN 检查
    x,
    y,                             # x:(B,D) 输入，y:(B,O) 目标
):
    pass                           # zero_grad -> autocast forward -> scale backward -> step/update
```

**Step 1：Autocast forward**

（i）代码

```python
with torch.autocast(
    device_type=x.device.type,     # 由输入设备决定 autocast 后端
    dtype=torch.float16,           # 低精度算子的目标 dtype
    enabled=x.is_cuda,             # CPU 上关闭，保证 demo 可跑
):
    pred = model(x)                # (B,D) -> (B,O)
    loss = F.mse_loss(pred, y)     # (B,O) 聚合为标量
```

（iii）适合低精度的 GEMM 等算子由 autocast 选择较低精度；部分 reduction / sensitive op 会保持更高精度。

**Step 2：Scale loss**

（i）公式

$$
L_s=sL
$$

（ii）代码

```python
scaler.scale(loss).backward()      # loss * s 后回传，链式法则使所有梯度同样放大 s 倍
```

（iii）backward 得到的梯度整体乘 $$s$$，把很小的 FP16 gradient 拉出 underflow 区域。

**Step 3：Unscale + optimizer step**

（i）公式

$$
g=g_s/s
$$

（ii）代码

```python
scaler.step(optimizer)             # 先把梯度除回 s，再执行 optimizer.step()
```

（iii）`GradScaler` 会在 step 前处理 unscale，并在检测到 inf/NaN 时跳过危险更新。

**Step 4：动态更新 scale**

（i）代码

```python
scaler.update()                    # overflow 则调小 s，长期平稳则调大 s
```

（iii）根据最近是否出现 overflow 自动调整 scale。

#### 👀 A11. Standard FFN / MLP（Position-wise Feed-Forward）

##### 1️⃣ Dense Opening

Transformer FFN 对每个 token 独立应用两层 Linear：先从 $$D$$ 升到 $$D_{ff}$$，经过非线性，再降回 $$D$$。核心对象是最后一维的 `D -> D_ff -> D`；sequence 维完全不混合。面试时通常写 ReLU/GELU 版即可，若问现代 LLM 再切到 SwiGLU。[参考 2]

##### 2️⃣ Global Equations

$$
H
=
\phi(XW_1+b_1)
$$

$$
Y
=
HW_2+b_2
$$

shape：

$$
X:(B,T,D)
$$

$$
W_1:(D,D_{ff}),
\qquad
W_2:(D_{ff},D)
$$

$$
Y:(B,T,D)
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
    ):
        super().__init__()
        self.fc1 = nn.Linear(
            d_model,
            d_ff,
        )
        self.fc2 = nn.Linear(
            d_ff,
            d_model,                       # 降回 D，使输出可直接与 residual 相加
        )

    def forward(self, x):
        # x: (...,D)，前面所有维（B,T）都当作独立 token，逐 token 独立变换
        h = F.gelu(
            self.fc1(x)                    # 升维 (...,D) -> (...,d_ff)
        )                                  # (...,d_ff)
        return self.fc2(h)                 # (...,D)
```

**（B）NumPy 版本（苛刻面试版）**

```python
import math
import numpy as np

def gelu_numpy(x):
    # 精确版 GELU：0.5x(1+erf(x/√2))；shape 不变
    erf = np.vectorize(math.erf)           # math.erf 只吃标量，包一层做逐元素运算
    return 0.5 * x * (
        1.0
        + erf(
            x / np.sqrt(2.0)               # 标准正态 CDF 的自变量
        )
    )

def ffn_numpy(
    x,
    W1,
    b1,
    W2,
    b2,
):
    # x: (...,D)；W1:(D,d_ff)、b1:(d_ff,)；W2:(d_ff,D)、b2:(D,)
    h = gelu_numpy(
        x @ W1 + b1                        # 升维到 d_ff，b1 沿前置维 broadcast
    )                                      # (...,d_ff)
    return h @ W2 + b2                    # (...,D)
```

**最小可运行测试**

```python
# Demo
ffn = FFN(
    d_model=16,
    d_ff=64,
)
x = torch.randn(2, 5, 16)          # (B=2, T=5, D=16)
print(ffn(x).shape)                # 仍是 (2,5,16)：T 维不混合
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,                      # d_model: token 表示维 D；d_ff: 中间放大维（常取 4D）
    ):
        super().__init__()
        pass                       # 建 fc1:(D->d_ff) 与 fc2:(d_ff->D)
```

**Step 1：升维**

（i）公式

$$
Z=XW_1+b_1
$$

（ii）代码

```python
z = self.fc1(x)                    # (...,D) -> (...,d_ff)，只改最后一维
```

（iii）只改最后一维：`(...,D) -> (...,d_ff)`。

**Step 2：Activation**

（i）公式

$$
H=\operatorname{GELU}(Z)
$$

（ii）代码

```python
h = F.gelu(z)                      # 逐元素非线性，shape 保持 (...,d_ff)
```

（iii）shape 不变。

**Step 3：降维**

（i）公式

$$
Y=HW_2+b_2
$$

（ii）代码

```python
return self.fc2(h)                 # (...,d_ff) -> (...,D)
```

（iii）恢复到 `(...,D)`，从而可以与 residual 相加。

#### A12. Knowledge Distillation Loss（Teacher-Student）

##### 1️⃣ Dense Opening

Knowledge distillation 让 student 同时学习 hard labels 与 teacher 的 softened distribution。核心对象是 temperature $$T$$：teacher/student logits 都除以 $$T$$，soft loss 使用 KL divergence，并乘 $$T^2$$ 补偿温度对 gradient scale 的缩小。面试最标准答案是 `alpha * T^2 * KL + (1-alpha) * CE`；teacher 作为固定 target，应在这条 loss 路径上停止梯度。[参考 15, 24]

##### 2️⃣ Global Equations

soft distributions：

$$
p_T
=
\operatorname{softmax}
\left(
z_T/T
\right)
$$

$$
p_S
=
\operatorname{softmax}
\left(
z_S/T
\right)
$$

distillation KL：

$$
L_{\mathrm{soft}}
=
T^2
D_{\mathrm{KL}}
\left(
p_T\|p_S
\right)
$$

hard-label loss：

$$
L_{\mathrm{hard}}
=
\operatorname{CE}
(z_S,y)
$$

总目标：

$$
L
=
\alpha L_{\mathrm{soft}}
+
(1-\alpha)L_{\mathrm{hard}}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准实现）**

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def distillation_loss(
    student_logits: Tensor,       # (B,C) student 未归一化 logits，梯度只从这里回传
    teacher_logits: Tensor,       # (B,C) teacher logits，作为固定监督目标
    labels: Tensor,               # (B,) 每个样本的 ground-truth 类别下标
    temperature: float = 2.0,     # T>1 软化分布，放大非 top-1 类别的相对信息
    alpha: float = 0.5,           # soft loss 与 hard loss 的加权系数
) -> Tensor:                      # 返回标量 loss
    T = temperature

    # teacher probability: (B,C)；detach 切断梯度，teacher 只提供信号不被更新
    teacher_prob = F.softmax(
        teacher_logits.detach() / T,
        dim=-1,                   # 沿类别维 C 归一化
    )

    # student log-probability: (B,C)；log 域是 kl_div 对 input 的接口要求
    student_log_prob = F.log_softmax(
        student_logits / T,
        dim=-1,
    )

    # batchmean = sum KL / B（按样本数平均，而非按 B*C 个元素平均）
    soft_loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean",
    ) * (T ** 2)                  # 乘 T^2 补偿 1/T 对 gradient scale 的缩小

    hard_loss = F.cross_entropy(  # 用未软化的原始 logits 与真实标签算 CE
        student_logits,
        labels,
    )

    return (
        alpha * soft_loss         # 模仿 teacher 的软分布
        + (1.0 - alpha) * hard_loss   # 拟合 ground-truth 标签
    )
```

**（B）NumPy 版本（同一计算图）**

```python
import numpy as np

def log_softmax_numpy(x):
    # x: (...,C) -> 同 shape 的 log-probability
    x = x - x.max(                # 每行减最大值防 exp 溢出，不改变 softmax 结果
        axis=-1,
        keepdims=True,            # C 维保留为 1，便于 broadcast 回原 shape
    )
    return x - np.log(
        np.exp(x).sum(            # 沿类别维求和 -> (...,1)，即 logsumexp 的配分项
            axis=-1,
            keepdims=True,
        )
    )

def distillation_loss_numpy(
    student_logits,               # (B,C)
    teacher_logits,               # (B,C)
    labels,                       # (B,) 类别下标
    temperature=2.0,
    alpha=0.5,
):
    T = temperature

    t_logp = log_softmax_numpy(   # (B,C) teacher 软化后的 log-prob
        teacher_logits / T
    )
    s_logp = log_softmax_numpy(   # (B,C) student 软化后的 log-prob
        student_logits / T
    )

    t_prob = np.exp(t_logp)       # (B,C) KL 的权重项 p_T

    soft = np.sum(
        t_prob * (t_logp - s_logp),   # 逐类别的 KL 贡献
        axis=-1,                      # 沿 C 聚合掉类别维 -> (B,) 每样本一个 KL
    ).mean() * (T ** 2)               # 再对 batch 平均，等价 batchmean

    hard_logp = log_softmax_numpy(    # (B,C) 用 T=1 的原始 logits 算 CE
        student_logits
    )
    B = len(labels)
    hard = -hard_logp[
        np.arange(B),                 # 行索引 0..B-1
        labels,                       # 列索引取每行正确类 -> 花式索引得到 (B,)
    ].mean()

    return (
        alpha * soft
        + (1.0 - alpha) * hard
    )
```

**最小可运行测试**

```python
# Demo
student = torch.randn(4, 10)          # (B=4, C=10) 随机 student logits
teacher = torch.randn(4, 10)          # (B=4, C=10) 随机 teacher logits
labels = torch.tensor([1, 3, 2, 7])   # (B,) 每个样本的正确类别下标

print(
    "KD loss:",
    distillation_loss(
        student,
        teacher,
        labels,
        temperature=2.0,
        alpha=0.7,                    # soft loss 占 0.7，更偏重模仿 teacher
    ).item(),                         # 标量 tensor -> Python float
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def distillation_loss(
    student_logits: Tensor,       # (B,C)
    teacher_logits: Tensor,       # (B,C)
    labels: Tensor,               # (B,) 类别下标
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Tensor:                      # 返回标量
    pass                          # 待填：soft KL(T^2 缩放) + hard CE 的加权和
```

**Step 1：Temperature softening**

（i）公式

$$
p_T=\operatorname{softmax}(z_T/T)
$$

（ii）代码

```python
teacher_prob = F.softmax(         # (B,C) 概率，作为 KL 的 target
    teacher_logits.detach() / T,  # detach：teacher 不接收 student loss 的梯度
    dim=-1,                       # 沿类别维归一化
)

student_log_prob = F.log_softmax(   # (B,C) log 域，kl_div 的 input 要求
    student_logits / T,
    dim=-1,
)
```

（iii）$$T>1$$ 让分布更平，暴露 teacher 在非 top-1 类别上的相对信息；`detach()` 明确 teacher 只提供监督信号，不接收 student loss 的梯度。

**Step 2：Soft KL**

（i）公式

$$
L_{\mathrm{soft}}
=
T^2
\sum_c
p_T(c)
\left[
\log p_T(c)-\log p_S(c)
\right]
$$

（ii）代码

```python
soft_loss = F.kl_div(
    student_log_prob,             # input：log-probability
    teacher_prob,                 # target：probability
    reduction="batchmean",        # 逐元素 KL 求和后除以 batch size -> 标量
) * (T ** 2)                      # 抵消 1/T 对梯度尺度的缩小
```

（iii）PyTorch `kl_div(input, target)` 的 `input` 是 log-probability，target 是 probability。

**Step 3：Hard CE**

（i）公式

$$
L_{\mathrm{hard}}
=
-\log p_S(y)
$$

（ii）代码

```python
hard_loss = F.cross_entropy(
    student_logits,               # (B,C) 未软化的原始 logits，内部自带 log_softmax
    labels,                       # (B,) 类别下标（非 one-hot）
)
```

（iii）hard label 保留原任务监督。

**Step 4：Weighted sum**

（i）公式

$$
L
=
\alpha L_{\mathrm{soft}}
+
(1-\alpha)L_{\mathrm{hard}}
$$

（ii）代码

```python
return (
    alpha * soft_loss             # 模仿 teacher 分布
    + (1.0 - alpha) * hard_loss   # 拟合真实标签
)
```

（iii）$$\alpha$$ 控制 teacher soft targets 与 ground-truth labels 的权重。

### A−：理论很热，完整手写略少

> 本章分两部分：**第一部分**为主文档原有题目，**第二部分**为《补充文档-LLM面试手撕代码补充题单》新增题目。

#### ▍第一部分 · 主文档原有题目

#### 👀 DPO Loss（Direct Preference Optimization）

不训 reward model 也不做 PPO，直接用偏好对 $$(y_w, y_l)$$ 把策略往人类偏好方向推。核心对象：log-ratio 差 $$\Delta$$。

$$
L_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \,\Delta \right) \right]
$$

$$
\Delta = \left( \log \pi_\theta(y_w\mid x) - \log \pi_{\text{ref}}(y_w\mid x) \right)
       - \left( \log \pi_\theta(y_l\mid x) - \log \pi_{\text{ref}}(y_l\mid x) \right)
$$

$$\beta$$ 控制相对 reference 的偏离程度，常用起点约 0.1；更大的 $$\beta$$ 通常意味着更少偏离 reference。$$\sigma$$ 是 sigmoid。$$\Delta$$ 越正越好。

#### （A）PyTorch 版本

```python
import torch
import torch.nn.functional as F

def dpo_loss(pi_logp_chosen, pi_logp_rejected,
             ref_logp_chosen, ref_logp_rejected,
             beta: float = 0.1):
    # 全部形状 (B,) —— 每条样本一个对数概率（整条 response 的 token log-prob 之和）
    pi_diff  = pi_logp_chosen - pi_logp_rejected       # (B,) 当前策略偏好 chosen 的程度
    ref_diff = (ref_logp_chosen - ref_logp_rejected).detach()  # (B,) reference 冻结，作为偏好基线
    logits = beta * (pi_diff - ref_diff)               # (B,) 隐式 reward 差，beta 控制偏离 ref 的幅度
    return -F.logsigmoid(logits).mean()                # -log σ(z) 沿 batch 平均成标量
```

#### （B）NumPy 版本

```python
import numpy as np

def dpo_loss_numpy(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    # pi_c, pi_r, ref_c, ref_r: (B,) log-probs
    x = beta * ((pi_c - pi_r) - (ref_c - ref_r))   # (B,) DPO logits，越正越符合人类偏好
    # 稳定 logsigmoid: -log(1+exp(-x)) = -softplus(-x)
    return np.mean(np.logaddexp(0.0, -x))          # logaddexp 内部 max-shift，避免 exp 溢出

# Demo
import torch
B = 4                                           # batch 内 4 个偏好对
pi_c  = torch.tensor([-1.0, -2.0, -0.5, -1.5])  # (B,) 策略在 chosen 上的 log-prob
pi_r  = torch.tensor([-1.5, -1.8, -1.2, -1.0])  # (B,) 策略在 rejected 上的 log-prob
ref_c = torch.tensor([-1.2, -2.1, -0.7, -1.4])  # (B,) reference 在 chosen 上的 log-prob
ref_r = torch.tensor([-1.3, -1.9, -1.0, -1.1])  # (B,) reference 在 rejected 上的 log-prob
print('DPO loss:', dpo_loss(pi_c, pi_r, ref_c, ref_r).item())
```

#### Step-by-step

```python
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta): pass  # 四个入参均为 (B,) log-prob，返回标量 loss
```

（i）策略 log-ratio 差
（ii）`pi_diff = pi_c - pi_r`
（iii）当前策略在 chosen 上比 rejected 高多少。

（i）参考 log-ratio 差
（ii）`ref_diff = (ref_c - ref_r).detach()`
（iii）reference policy 固定，不应接收梯度；它提供相对偏好的基线。

（i）DPO logits $$z = \beta (\pi_{\Delta} - \text{ref}_{\Delta})$$
（ii）`z = beta * (pi_diff - ref_diff)`
（iii）$$\beta$$ 控制 KL 强度，越大越保守。

（i）目标 $$-\log \sigma(z)$$
（ii）`return -F.logsigmoid(z).mean()`
（iii）等价 $$\text{softplus}(-z)$$，比 `-log(sigmoid)` 数值更稳定。

#### 👀 PPO Loss（Proximal Policy Optimization）

带 clip 的重要性采样策略梯度。核心对象：策略比 $$r_t = \pi_\theta/\pi_{\text{old}}$$ 与优势 $$A_t$$。

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} = \exp\!\left(\log\pi_\theta - \log\pi_{\text{old}}\right)
$$

$$
L_{\text{PPO}} = -\mathbb{E}_t \left[ \min\!\big( r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t \big) \right]
$$

clip 防止单步策略改动过大（trust region 近似）。$$\epsilon$$ 常用 0.1~0.2。

#### （A）PyTorch 版本

```python
import torch

def ppo_loss(new_logp: torch.Tensor,
             old_logp: torch.Tensor,
             advantages: torch.Tensor,
             clip_ratio: float = 0.2) -> torch.Tensor:
    # 形状均为 (B,) 或 (B*T,)；梯度只流过 new_logp
    old_logp = old_logp.detach()                       # 采样用的行为策略，视为常量
    advantages = advantages.detach()                   # advantage 由 critic/reward 给出，不回传
    ratio = torch.exp(new_logp - old_logp)             # (N,) 重要性采样比 r_t，log 域相减更稳定
    s1 = ratio * advantages                            # (N,) 未剪枝项 r_t·A_t
    s2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages  # r_t 夹到 [1-ε,1+ε]
    return -torch.min(s1, s2).mean()                   # 逐元素取悲观下界后平均，取负变最小化目标
```

#### （B）NumPy 版本

```python
import numpy as np

def ppo_loss_numpy(new_logp, old_logp, advantages, clip_ratio=0.2):
    # 三个数组同为 (N,)：逐样本（或逐 token）的 log-prob 与 advantage
    ratio = np.exp(new_logp - old_logp)                # (N,) 策略比 r_t
    s1 = ratio * advantages
    s2 = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages   # 软 trust region
    return -np.mean(np.minimum(s1, s2))                # 逐元素 min -> 沿 N 平均 -> 取负

# Demo
new = np.array([0.0, -0.2, -0.4, -0.6])   # (N=4,) 新策略 log-prob
old = np.array([0.0, -0.1, -0.5, -0.5])   # (N,) 采样时旧策略 log-prob
adv = np.array([1.0, -1.0, 0.5, -0.5])    # (N,) 正=优于基线该加强，负=劣于基线该抑制
print('PPO loss:', ppo_loss_numpy(new, old, adv, 0.2))
```

#### Step-by-step

```python
def ppo_loss(new_logp, old_logp, adv, clip): pass  # 前三个入参 (N,)，clip 为 ε，返回标量
```

（i）策略比 $$r_t = e^{\log\pi_\theta - \log\pi_{\text{old}}}$$
（ii）`old_logp = old_logp.detach(); ratio = torch.exp(new_logp - old_logp)`
（iii）old policy 是采样行为策略，作为常量；log 域相减再 exp 比直接相除更稳定。

（i）未剪枝项 $$r_t A_t$$
（ii）`adv = adv.detach(); s1 = ratio * adv`
（iii）advantage 在 policy update 中作为常量，梯度只更新当前策略。

（i）剪枝项 $$\text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t$$
（ii）`s2 = torch.clamp(ratio, 1-clip, 1+clip) * adv`
（iii）限制 ratio 不偏离 1 太远，等价于一个软 trust region。

（i）pessimistic min + 取负
（ii）`return -torch.min(s1, s2).mean()`
（iii）取小者保证：当 $$A>0$$ 时不会一味放大 ratio；当 $$A<0$$ 时不会忽略大幅下推。

#### 👀 GRPO Loss（Group Relative Policy Optimization）

DeepSeek 提出：对同一 prompt 采样一组 responses，用组内相对奖励构造 advantage，从而省掉单独的 value/critic 网络。面试时最容易写错的一点是 **policy ratio 仍然按 completion token 计算**；一个 response 的标量 advantage $$A_i$$ 广播到该 response 的所有有效 token。

$$
A_i = \frac{R_i - \mu_R}{\sigma_R + \epsilon},\qquad
\mu_R = \frac{1}{G}\sum_{j=1}^{G} R_j
$$

对第 $$i$$ 条 response 的第 $$t$$ 个 completion token：

$$
r_{i,t}(\theta)
=
\exp\!\left(\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-
\log\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})\right)
$$

原始 GRPO 的 sample-level 聚合写成：

$$
L_{\mathrm{GRPO}}
=
-\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\left[
\min\!\left(
 r_{i,t}A_i,
 \operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
-
\beta D_{\mathrm{KL},i,t}
\right]
$$

相比 PPO：免 critic、advantage 来自组内对比；PPO-style clipping 仍作用于逐 token ratio。当前 Hugging Face TRL 还提供 DAPO / Dr. GRPO 等不同聚合方式；本节保留“原始 GRPO”定义，DAPO 在后文单独实现。

#### （A）PyTorch 版本

```python
import torch

def grpo_loss(new_logp: torch.Tensor,
              old_logp: torch.Tensor,
              ref_logp: torch.Tensor,
              rewards: torch.Tensor,
              completion_mask: torch.Tensor,
              clip_ratio: float = 0.2,
              beta: float = 0.04) -> torch.Tensor:
    # new/old/ref_logp: (G,T)，逐 completion-token log-prob
    # rewards: (G,)；completion_mask: (G,T)，有效 token=1，padding=0
    adv = (
        (rewards - rewards.mean())
        / (rewards.std(unbiased=False) + 1e-8)
    ).detach()                                           # (G,)，组内相对 advantage

    ratio = torch.exp(new_logp - old_logp.detach())      # (G,T)，逐 token importance ratio
    s1 = ratio * adv[:, None]                            # (G,T)，同一 response 共用 A_i
    s2 = torch.clamp(
        ratio, 1 - clip_ratio, 1 + clip_ratio
    ) * adv[:, None]

    # k3 KL estimate: d = log pi_ref - log pi_theta
    d = ref_logp.detach() - new_logp                     # (G,T)
    kl = torch.exp(d) - d - 1                            # (G,T)，逐点非负

    token_obj = torch.minimum(s1, s2) - beta * kl        # (G,T)，要最大化的逐 token objective
    mask = completion_mask.to(new_logp.dtype)            # (G,T)
    per_response = (
        (token_obj * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp_min(1.0)
    )                                                    # (G,)：原始 GRPO 先对每条 response 求均值
    return -per_response.mean()                          # 再沿 G 平均，取负作为 loss
```

#### （B）NumPy 版本

```python
import numpy as np

def grpo_loss_numpy(new_logp, old_logp, ref_logp, rewards,
                    completion_mask, clip_ratio=0.2, beta=0.04):
    # new/old/ref_logp: (G,T)；rewards:(G,)；mask:(G,T)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # (G,)

    ratio = np.exp(new_logp - old_logp)                  # (G,T)
    s1 = ratio * adv[:, None]
    s2 = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[:, None]

    d = ref_logp - new_logp                              # (G,T)
    kl = np.exp(d) - d - 1.0                            # k3 KL estimate
    token_obj = np.minimum(s1, s2) - beta * kl

    mask = completion_mask.astype(new_logp.dtype)
    per_response = (token_obj * mask).sum(-1) / np.maximum(mask.sum(-1), 1.0)
    return -per_response.mean()

# Demo：4 条 response，padding 长度不同
G, T = 4, 5
old = np.array([
    [-0.7, -0.8, -0.9, -1.0,  0.0],
    [-0.6, -0.7, -0.8,  0.0,  0.0],
    [-0.9, -0.8, -0.7, -0.6, -0.5],
    [-0.8, -0.9, -1.0, -1.1,  0.0],
])
new = old + np.array([
    [ 0.05, -0.02,  0.03,  0.01, 0.0],
    [-0.04,  0.03,  0.02,  0.00, 0.0],
    [ 0.02,  0.01, -0.03,  0.04, 0.01],
    [-0.01, -0.02,  0.03, -0.04, 0.0],
])
ref = old - 0.03
rew = np.array([1.0, 0.0, 0.5, -0.5])                  # (G,)
mask = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0],
])                                                       # (G,T)
print('GRPO loss:', grpo_loss_numpy(new, old, ref, rew, mask))
```

#### Step-by-step

```python
def grpo_loss(new_logp, old_logp, ref_logp, rewards, completion_mask):
    pass  # logp/mask:(G,T)，rewards:(G,)
```

（i）组内标准化得 response-level advantage  
（ii）`adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)`  
（iii）同一 prompt 的 $$G$$ 条 responses 做相对比较；策略更新时把 `adv` 视为常量。

（i）逐 token policy ratio  
（ii）`ratio = exp(new_logp - old_logp)`，shape `(G,T)`  
（iii）不要先把整条 response 的 log-prob 求和再做 PPO clip；原始 GRPO 的 clipping unit 是 completion token。

（i）clipped surrogate  
（ii）`min(ratio * A, clamp(ratio, 1-eps, 1+eps) * A)`  
（iii）`A_i` 用 `adv[:, None]` 广播到第 $$i$$ 条 response 的所有有效 token。

（i）KL 正则用 k3 估计：$$\widehat{D}_{\mathrm{KL}} = e^d-d-1$$，其中 $$d=\log\pi_{\mathrm{ref}}-\log\pi_\theta$$  
（ii）`d = ref_logp - new_logp; kl = d.exp() - d - 1`  
（iii）`ref_logp` detach；KL 与 clipped surrogate 都只在 completion mask 内聚合。

（i）原始 GRPO 聚合  
（ii）`per_response = (token_obj * mask).sum(-1) / mask.sum(-1)`，最后 `-per_response.mean()`  
（iii）即“每条 response 内先按有效 token 平均，再对组内 responses 平均”。截至 2026-09，TRL 的工程默认 `loss_type="dapo"` 使用不同的 token-level normalization，且默认 `beta=0`；面试若明确问 GRPO 原始公式，应把两者区分开。

---

#### ▍第二部分 · 补充题单新增题目

#### A−1. PagedAttention（Simplified Page-Table Version）

##### 1️⃣ Dense Opening

PagedAttention 解决 LLM serving 中动态增长 KV cache 的内存碎片与重复分配问题：逻辑上连续的 token KV 被切成固定大小 blocks，通过 block table 映射到任意物理 pages。核心对象是 `block_table[request, logical_block] -> physical_page`。真实 vLLM kernel 会直接按 page table 读取分散 KV；面试版先根据 page table 逻辑 gather，再做普通 attention，重点展示地址映射。[参考 16]

##### 2️⃣ Global Equations

设 block size 为 $$B_s$$，token 位置 $$t$$：

$$
b_{\mathrm{logical}}
=
\left\lfloor
\frac{t}{B_s}
\right\rfloor
$$

$$
o
=
t\bmod B_s
$$

物理 page：

$$
p
=
\mathrm{BlockTable}
[
r,
b_{\mathrm{logical}}
]
$$

因此：

$$
K_r[t]
=
K_{\mathrm{pages}}[p,o]
$$

$$
V_r[t]
=
V_{\mathrm{pages}}[p,o]
$$

attention 数学本身仍为：

$$
O
=
\operatorname{softmax}
\left(
QK^\top/\sqrt{d_h}
\right)V
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（面试简化实现）**

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def paged_attention_decode(
    q: Tensor,
    k_pages: Tensor,
    v_pages: Tensor,
    block_table: Tensor,
    seq_lens: Tensor,
    block_size: int,
) -> Tensor:
    # q: (B,H,D) 每个 request 当前 1 个 query token；k_pages/v_pages: (P,block_size,H,D) 物理页池
    # block_table: (B,max_blocks) logical block -> physical page id；seq_lens: (B,) 各自已有 KV 长度
    # 返回: (B,H,D)
    B, H, D = q.shape                      # B: request 数；H: head 数；D: head_dim
    outputs = []

    for b in range(B):
        L = int(seq_lens[b].item())        # 该 request 的有效 KV token 数
        n_blocks = (
            L + block_size - 1
        ) // block_size                    # ceil(L/block_size)：占用的 logical block 数

        # logical block ids -> physical page ids
        page_ids = block_table[
            b, :n_blocks
        ]                                  # (n_blocks,) 物理页号，可完全不连续

        # gather:
        # (n_blocks, block_size, H, D)
        # -> (n_blocks*block_size, H, D)
        k = k_pages[
            page_ids
        ].reshape(
            -1, H, D
        )[:L]                              # 切掉尾页多余的 slot -> (L,H,D)

        v = v_pages[
            page_ids
        ].reshape(
            -1, H, D
        )[:L]                              # (L,H,D)

        # token-major -> head-major
        k = k.transpose(0, 1)              # (H,L,D)
        v = v.transpose(0, 1)              # (H,L,D)

        # q[b]: (H,D)
        # einsum -> scores: (H,L)：每个 head 的 query 对 L 个 key 的点积
        scores = torch.einsum(
            "hd,hld->hl",
            q[b],
            k,
        ) / math.sqrt(D)                   # 除 sqrt(D) 防 softmax 饱和

        attn = F.softmax(
            scores,
            dim=-1,                        # 沿 L 归一化：每个 head 一套 key 权重
        )                                  # (H,L)

        out = torch.einsum(                # 用权重对 L 个 value 加权求和，聚合掉序列维
            "hl,hld->hd",
            attn,
            v,
        )                                  # (H,D)
        outputs.append(out)

    return torch.stack(                    # B 个 (H,D) 沿新建的 batch 维堆叠
        outputs,
        dim=0,
    )                                      # (B,H,D)
```

**（B）NumPy 版本（同一逻辑地址映射）**

```python
import numpy as np

def paged_attention_decode_numpy(
    q,
    k_pages,
    v_pages,
    block_table,
    seq_lens,
    block_size,
):
    # q: (B,H,D)；k_pages/v_pages: (P,block_size,H,D)；block_table: (B,max_blocks)；返回 (B,H,D)
    B, H, D = q.shape
    outputs = []

    for b in range(B):
        L = int(seq_lens[b])               # 该 request 的有效 KV 长度
        n_blocks = (
            L + block_size - 1
        ) // block_size                    # ceil 除法得 logical block 数

        page_ids = block_table[            # (n_blocks,) 逻辑块映射到的物理页号
            b, :n_blocks
        ]

        k = k_pages[                       # 花式索引 gather -> (n_blocks,block_size,H,D)
            page_ids
        ].reshape(
            -1, H, D                       # 展平前两维 -> (n_blocks*block_size,H,D)
        )[:L]                               # (L,H,D)

        v = v_pages[
            page_ids
        ].reshape(
            -1, H, D
        )[:L]                              # (L,H,D)

        k = k.transpose(1, 0, 2)           # (H,L,D)
        v = v.transpose(1, 0, 2)           # (H,L,D)

        scores = np.einsum(                # (H,D)·(H,L,D) 沿 D 收缩
            "hd,hld->hl",
            q[b],
            k,
        ) / np.sqrt(D)                     # (H,L)，缩放防 softmax 饱和

        scores -= scores.max(              # 每个 head 减去自身最大值，防 exp 溢出
            axis=-1,
            keepdims=True,                 # 保留 L 维为 1 以便 broadcast
        )
        a = np.exp(scores)
        a /= a.sum(                        # 沿 L 归一化成 attention 权重 (H,L)
            axis=-1,
            keepdims=True,
        )

        out = np.einsum(                   # 沿 L 加权求和，聚合掉序列维
            "hl,hld->hd",
            a,
            v,
        )                                  # (H,D)
        outputs.append(out)

    return np.stack(outputs, axis=0)       # B 个 (H,D) -> (B,H,D)
```

**最小可运行测试**

```python
# Demo
B, H, D = 2, 2, 4          # 2 个 request、2 个 head、head_dim=4
block_size = 2             # 每个物理 page 存 2 个 token 的 KV
P = 6                      # 物理页池共 6 页

q = torch.randn(B, H, D)   # (B,H,D) 当前 decode step 的 query
k_pages = torch.randn(     # (P,block_size,H,D) 全局 K page 池
    P, block_size, H, D
)
v_pages = torch.randn(     # (P,block_size,H,D) 全局 V page 池
    P, block_size, H, D
)

# request 0: logical pages -> [3,1]
# request 1: logical pages -> [4,0,2]
block_table = torch.tensor([   # (B,max_blocks=3)，-1 是未分配占位
    [3, 1, -1],
    [4, 0, 2],
])
seq_lens = torch.tensor([3, 5])   # (B,) 有效长度：3 需 2 页、5 需 3 页

out = paged_attention_decode(
    q,
    k_pages,
    v_pages,
    block_table,
    seq_lens,
    block_size,
)
print(out.shape)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def paged_attention_decode(
    q: Tensor,                # (B,H,D)
    k_pages: Tensor,          # (P,block_size,H,D)
    v_pages: Tensor,          # (P,block_size,H,D)
    block_table: Tensor,      # (B,max_blocks) logical block -> physical page id
    seq_lens: Tensor,         # (B,)
    block_size: int,
) -> Tensor:                  # 返回 (B,H,D)
    pass                      # 待填：查页表 -> gather KV -> 普通 attention
```

**Step 1：算需要多少 logical blocks**

（i）公式

$$
N_b
=
\left\lceil
\frac{L}{B_s}
\right\rceil
$$

（ii）代码

```python
n_blocks = (                  # 整数 ceil：ceil(L/block_size)
    L + block_size - 1
) // block_size               # 尾块可只用一部分 slot
```

（iii）整数 ceil division，最后一个 block 可以只用一部分。

**Step 2：Page-table lookup**

（i）公式

$$
p_j
=
\mathrm{BlockTable}[r,j]
$$

（ii）代码

```python
page_ids = block_table[       # 取第 b 行前 n_blocks 项 -> (n_blocks,)
    b, :n_blocks
]                             # 逻辑块连续，物理页号可任意分散
```

（iii）logical block 连续，但 physical page id 可以完全不连续。

**Step 3：按物理页 gather KV**

（i）shape

$$
(N_b,B_s,H,D)
\rightarrow
(N_bB_s,H,D)
\rightarrow
(L,H,D)
$$

（ii）代码

```python
k = k_pages[                  # gather 页 -> (n_blocks,block_size,H,D)
    page_ids
].reshape(
    -1, H, D                  # 合并页维与页内 slot 维 -> (n_blocks*block_size,H,D)
)[:L]                         # 截掉尾页 padding -> (L,H,D)
```

（iii）面试版显式 gather；真实 PagedAttention kernel 会直接按 block table 访问 page，避免重建完整 contiguous KV。

**Step 4：普通 attention**

（i）公式

$$
a
=
\operatorname{softmax}
(qK^\top/\sqrt D)
$$

（ii）代码

```python
scores = torch.einsum(        # (H,D) 与 (H,L,D) 沿 D 收缩 -> (H,L)
    "hd,hld->hl",
    q[b],
    k,
) / math.sqrt(D)              # 缩放防 softmax 饱和
```

（iii）PagedAttention 改的是 KV memory layout，不改 attention 的数学定义。

#### A−2. Continuous Batching Scheduler（Iteration-Level Scheduling）

##### 1️⃣ Dense Opening

Continuous batching 的核心是每个 decode iteration 都重新组成 active batch：完成的 request 立即离开，队列中的新 request 立即补入空 slot，不必等整个旧 batch 全部完成。核心对象是 active set $$A_t$$ 的逐步更新。下面用“每个 request 还剩多少 decode steps”模拟 scheduler；真实 serving 还会处理 prefill、KV blocks、优先级和 token budget。[参考 17]

##### 2️⃣ Global Equations

设 batch capacity 为 $$C$$。

每一步：

$$
A_t
=
\{r_1,\ldots,r_k\},
\qquad
k\le C
$$

执行一次 decode 后：

$$
\mathrm{remaining}_r
\leftarrow
\mathrm{remaining}_r-1
$$

完成集合：

$$
F_t
=
\{
r\in A_t:
\mathrm{remaining}_r=0
\}
$$

下一步 active set：

$$
A_{t+1}
=
(A_t\setminus F_t)
\cup
\mathrm{AdmitFromQueue}
$$

直到：

$$
|A_{t+1}|
=
\min(C,\text{available requests})
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（调度逻辑面试版）**

```python
import torch
from torch import Tensor

def continuous_batch_schedule(
    decode_steps: Tensor,
    max_batch_size: int,
):
    # decode_steps: (N,) 第 i 个 request 还需要生成多少 token；max_batch_size 即容量 C
    # 返回 list[list[int]]：第 t 项是第 t 个 decode iteration 的 active request IDs
    remaining = decode_steps.clone()   # (N,) 剩余步数，clone 避免原地修改调用方张量

    N = remaining.numel()              # request 总数
    next_request = 0                   # queue pointer：下一个尚未进入系统的 request
    active = []                        # 当前 active set A_t，存 request ID
    schedule = []

    # 先填满 batch
    while (
        len(active) < max_batch_size
        and next_request < N
    ):
        active.append(next_request)
        next_request += 1

    while active:
        # 当前 iteration 的 batch
        schedule.append(active.copy())   # copy：快照，后续对 active 的增删不影响它

        ids = torch.tensor(
            active,
            dtype=torch.long,
        )                                # list[int] -> (k,) long 索引张量，k=|A_t|
        remaining[ids] -= 1              # 按索引批量减 1：一次 iteration 只推进一个 token

        # 完成的 request 立即退出（remaining=0 者从 A_t 中过滤掉）
        active = [
            r
            for r in active
            if remaining[r] > 0
        ]

        # 空 slot 立即从等待队列补入
        while (
            len(active) < max_batch_size
            and next_request < N
        ):
            active.append(next_request)
            next_request += 1

    return schedule
```

**（B）NumPy 版本（同一 scheduler）**

```python
import numpy as np

def continuous_batch_schedule_numpy(
    decode_steps,
    max_batch_size,
):
    # decode_steps: 长度 N 的序列，第 i 个 request 剩余 decode 步数
    remaining = np.array(
        decode_steps,
        dtype=np.int64,
    ).copy()                          # -> (N,) int64，copy 保证不写回输入

    N = len(remaining)
    next_request = 0                  # queue pointer
    active = []                       # 当前 active set A_t
    schedule = []

    # 首个 iteration 前先把 batch 填到容量上限
    while (
        len(active) < max_batch_size
        and next_request < N
    ):
        active.append(next_request)
        next_request += 1

    while active:
        schedule.append(active.copy())   # 记录本 iteration 的 batch 快照

        ids = np.array(
            active,
            dtype=np.int64,
        )                                # list[int] -> (k,) 整数索引数组
        remaining[ids] -= 1              # fancy indexing 批量减 1

        # 过滤出仍未完成的 request：A_t \ F_t
        active = [
            r
            for r in active
            if remaining[r] > 0
        ]

        # 腾出的 slot 立即由等待队列补入
        while (
            len(active) < max_batch_size
            and next_request < N
        ):
            active.append(next_request)
            next_request += 1

    return schedule
```

**最小可运行测试**

```python
# Demo
steps = torch.tensor([2, 5, 1, 3])   # (4,) 4 个 request，各需 2/5/1/3 个 decode step

schedule = continuous_batch_schedule(
    steps,
    max_batch_size=2,                # 容量 C=2，同时最多 2 个 request 在 batch 内
)

for t, batch in enumerate(schedule):
    print(t, batch)                  # t: iteration 序号；batch: 该步的 request ID 列表
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def continuous_batch_schedule(
    decode_steps: Tensor,
    max_batch_size: int,
):
    # decode_steps: (N,) 每个 request 的剩余 decode 步数；返回每个 iteration 的 active ID 列表
    pass
```

**Step 1：维护 active set 与 queue pointer**

（i）状态

$$
A_t,\qquad
\text{next\_request}
$$

（ii）代码

```python
active = []        # active set A_t：当前占用 batch slot 的 request ID
next_request = 0   # queue pointer：等待队列中下一个待 admit 的 request
```

（iii）一个数组保存当前 batch request IDs，一个 pointer 指向尚未进入系统的下一个 request。

**Step 2：每个 iteration 只推进一个 token**

（i）公式

$$
R_r\leftarrow R_r-1
$$

（ii）代码

```python
ids = torch.tensor(
    active,
    dtype=torch.long,
)                      # list[int] -> (k,) long 索引张量，k 为本步 batch 大小
remaining[ids] -= 1    # 只对 active 位置减 1，其余 request 状态不动
```

（iii）这就是 iteration-level scheduling：调度粒度是 decode step，不是完整 request。

**Step 3：立即移除完成 request**

（i）公式

$$
A_t\setminus F_t
$$

（ii）代码

```python
# 保留 remaining>0 的 request：A_t \ F_t，列表长度只减不增
active = [
    r
    for r in active
    if remaining[r] > 0
]
```

（iii）短 request 不再被长 request 拖住。

**Step 4：立即补入新 request**

（i）约束

$$
|A_{t+1}|\le C
$$

（ii）代码

```python
# 两个终止条件：batch 填满（|A|=C）或等待队列已空
while (
    len(active) < max_batch_size
    and next_request < N
):
    active.append(next_request)   # admit 一个新 request 进入空 slot
    next_request += 1
```

（iii）这一步使 GPU batch slot 持续被利用。


#### A−3. DAPO Objective / Loss（Decoupled Clip and Dynamic Sampling Policy Optimization）

##### 1️⃣ Dense Opening

DAPO 可以理解为在 GRPO-style group advantage 上加入四个关键训练设计：Clip-Higher（上下界不对称）、Dynamic Sampling（过滤全对/全错、无有效梯度的 prompt groups）、Token-Level Policy Gradient Loss（按全部有效 response tokens 聚合）与 Overlong Reward Shaping。面试手写最值得落到代码的是前三者；overlong shaping 通常作为 reward preprocessing 说明即可。真实 Dynamic Sampling 会在 rollout/sampler 层持续补采直到凑够有效 groups，下面的 loss 只实现其中的 group filtering。[参考 18, 25]

##### 2️⃣ Global Equations

组内 advantage：

$$
A_i
=
\frac{
r_i-\mu_g
}{
\sigma_g+\epsilon
}
$$

token-level importance ratio：

$$
\rho_{i,t}
=
\exp
\left(
\log\pi_\theta(y_{i,t})
-
\log\pi_{\mathrm{old}}(y_{i,t})
\right)
$$

DAPO asymmetric clipping：

$$
\widetilde\rho_{i,t}
=
\operatorname{clip}
\left(
\rho_{i,t},
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}
\right)
$$

clipped surrogate：

$$
s_{i,t}
=
\min
\left(
\rho_{i,t}A_i,
\widetilde\rho_{i,t}A_i
\right)
$$

token-level policy loss：

$$
L_{\mathrm{DAPO}}
=
-
\frac{
\sum_{i,t}
m_{i,t}s_{i,t}
}{
\sum_{i,t}m_{i,t}
}
$$

Dynamic Sampling 对 binary correctness reward 的核心过滤条件可以写为：

$$
0
<
\frac{1}{G}\sum_{i\in g}r_i
<
1
$$

等价地，组内 reward 不能全部相同。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（核心 objective 面试版）**

```python
import torch
from torch import Tensor

def dapo_loss(
    new_logps: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    response_mask: Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    eps: float = 1e-8,
) -> Tensor:
    # new/old_logps: (B,T) 逐 token log 概率；rewards/group_ids: (B,)
    # response_mask: (B,T)，有效 response token=1（prompt/padding 为 0）
    # 返回: 标量 loss；eps_high>eps_low 即 DAPO 的 Clip-Higher
    B, T = new_logps.shape

    advantages = torch.zeros_like(rewards)   # (B,) 每条 response 一个 advantage
    valid_seq = torch.zeros(
        B,
        dtype=torch.bool,
        device=rewards.device,
    )                                        # (B,) 该 response 所属 group 是否通过 dynamic sampling

    # 逐 group 做 reward 标准化；reward 全相同（std≈0）的 group 直接判为无效
    for gid in group_ids.unique():           # unique: (B,) -> (G,) 去重后的 group ID
        idx = group_ids == gid               # (B,) bool mask，选中同一 prompt 的 G 条 response
        r = rewards[idx]                     # (B,) -> (n_g,) 组内 reward

        std = r.std(unbiased=False)          # 有偏 std，与公式 σ_g 一致（n 而非 n-1）

        if std > eps:
            advantages[idx] = (
                r - r.mean()
            ) / (std + eps)                  # 组内 z-score，写回 advantages 的对应位置
            valid_seq[idx] = True

    advantages = advantages.detach()         # advantage 视为常数，梯度只走 new_logps

    # token-level ratio: (B,T)
    ratio = torch.exp(
        new_logps - old_logps.detach()       # log 空间相减再 exp，等价于概率相除
    )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_low,                       # 下界仍取 0.2，抑制负向过冲
        1.0 + eps_high,                      # 上界放宽到 0.28，给低概率 token 留探索空间
    )

    # advantage: (B,) -> (B,1)，沿 T 维 broadcast 到每个 token
    adv = advantages[:, None]

    s1 = ratio * adv
    s2 = clipped_ratio * adv
    surrogate = torch.minimum(s1, s2)     # (B,T) 取悲观分支，构成 PPO 式下界

    # dynamic-sampling group filter
    # + response token mask
    token_mask = (
        response_mask.bool()
        & valid_seq[:, None]              # (B,T) & (B,1) -> broadcast 成 (B,T)
    ).to(surrogate.dtype)                 # (B,T) bool -> float，用于加权求和

    return -(
        (surrogate * token_mask).sum()        # 对 (B,T) 全部有效 token 求和（跨样本跨时间）
        / token_mask.sum().clamp_min(1.0)     # 除有效 token 总数；clamp 防全 mask 时除零
    )
```

**（B）NumPy 版本（同一 objective）**

```python
import numpy as np

def dapo_loss_numpy(
    new_logps,
    old_logps,
    rewards,
    group_ids,
    response_mask,
    eps_low=0.2,
    eps_high=0.28,
    eps=1e-8,
):
    # new/old_logps, response_mask: (B,T)；rewards, group_ids: (B,)
    B, T = new_logps.shape

    advantages = np.zeros_like(
        rewards,
        dtype=np.float64,
    )                                    # (B,)
    valid_seq = np.zeros(
        B,
        dtype=bool,
    )                                    # (B,) 是否属于通过 dynamic sampling 的 group

    for gid in np.unique(group_ids):     # (B,) -> (G,) 去重且升序的 group ID
        idx = group_ids == gid           # (B,) bool mask，定位同 prompt 的 response
        r = rewards[idx]                 # -> (n_g,)

        std = r.std()                    # numpy 默认 ddof=0，即有偏 std
        if std > eps:                    # std≈0 表示组内 reward 全同 -> 无有效梯度，跳过
            advantages[idx] = (
                r - r.mean()
            ) / (std + eps)              # 组内 z-score
            valid_seq[idx] = True

    ratio = np.exp(
        new_logps - old_logps            # (B,T) log-ratio -> 概率比
    )

    clipped_ratio = np.clip(
        ratio,
        1.0 - eps_low,
        1.0 + eps_high,                  # 非对称上界，Clip-Higher
    )

    adv = advantages[:, None]            # (B,) -> (B,1)，broadcast 到 T 维

    s1 = ratio * adv
    s2 = clipped_ratio * adv
    surrogate = np.minimum(s1, s2)       # (B,T) 逐 token 取悲观项

    token_mask = (
        response_mask.astype(bool)
        & valid_seq[:, None]             # (B,T) & (B,1) -> (B,T)：同时满足有效 token 与有效 group
    ).astype(np.float64)

    return -(
        (surrogate * token_mask).sum()   # 标量：所有有效 token 上求和
        / max(token_mask.sum(), 1.0)     # token-level 归一化，长 response 权重更大
    )
```

**最小可运行测试**

```python
# Demo：两个 prompt group，每组 3 个 response
B, T = 6, 4                                    # B=6 条 response（2 组×3 条），T=4 个 token

new = torch.randn(B, T) * 0.1                  # (B,T) 当前 policy 的 token log-prob
old = new.detach() + 0.02 * torch.randn(B, T)  # (B,T) behavior policy，与 new 略有偏差

rewards = torch.tensor([
    1.0, 0.0, 1.0,   # group 0：有效
    1.0, 1.0, 1.0,   # group 1：全对，动态过滤
])                                             # (B,) 每条 response 一个标量 reward
group_ids = torch.tensor([
    0, 0, 0,
    1, 1, 1,
])                                             # (B,) 把 6 条 response 划分为 2 个 prompt group
mask = torch.ones(B, T)                        # (B,T) 全 1：本例无 padding

print(
    "DAPO loss:",
    dapo_loss(
        new,
        old,
        rewards,
        group_ids,
        mask,
    ).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def dapo_loss(
    new_logps: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    response_mask: Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
    eps: float = 1e-8,
) -> Tensor:
    # logps/response_mask: (B,T)；rewards/group_ids: (B,)；返回标量 loss
    pass
```

**Step 1：Group-relative advantage**

（i）公式

$$
A_i
=
(r_i-\mu_g)/(\sigma_g+\epsilon)
$$

（ii）代码

```python
# idx: (B,) bool mask 选出同一 group 的 response；r=rewards[idx] 形如 (n_g,)
advantages[idx] = (
    r - r.mean()          # 减组内均值 μ_g，得到相对优劣
) / (std + eps)           # 除组内 std σ_g 做尺度归一，eps 防除零
```

（iii）同 prompt 的多个 response 相互比较，不需要 critic。

**Step 2：Dynamic Sampling**

（i）条件

$$
\sigma_g>0
$$

（ii）代码

```python
if std > eps:                # 组内 reward 有差异才有可用梯度信号
    valid_seq[idx] = True    # 把该 group 的所有 response 标记为参与 loss
```

（iii）全对或全错时 advantage 全为 0，这类 group 对 policy gradient 没有效信息。loss 内可以过滤；完整 DAPO sampler 还会继续补采，直到获得足够的有效 groups。

**Step 3：Token-level ratio**

（i）公式

$$
\rho_{it}
=
e^{
\log\pi_\theta
-
\log\pi_{\mathrm{old}}
}
$$

（ii）代码

```python
ratio = torch.exp(
    new_logps - old_logps.detach()   # (B,T) log-ratio；detach 冻结 behavior policy
)                                    # -> (B,T) 逐 token importance ratio ρ
```

（iii）DAPO 核心 loss 仍在 token level 处理 importance ratio；`old_logps` 来自固定 behavior policy，因此 PyTorch 版显式 `detach()`。

**Step 4：Clip-Higher**

（i）公式

$$
\operatorname{clip}
\left(
\rho,
1-\epsilon_{\mathrm{low}},
1+\epsilon_{\mathrm{high}}
\right)
$$

（ii）代码

```python
clipped_ratio = torch.clamp(
    ratio,
    1.0 - eps_low,     # 下界 1-0.2
    1.0 + eps_high,    # 上界 1+0.28 > 1+eps_low，即 Clip-Higher（shape 不变，仍 (B,T)）
)
```

（iii）上界大于传统对称 clip 的上界，给正向探索留更大空间。

**Step 5：Token-level aggregation**

（i）公式

$$
L
=
-
\frac{
\sum_{i,t}m_{it}
\min(
\rho_{it}A_i,
\widetilde\rho_{it}A_i
)
}{
\sum_{i,t}m_{it}
}
$$

（ii）代码

```python
return -(
    (surrogate * token_mask).sum()        # (B,T) 逐元素屏蔽后求和 -> 标量
    / token_mask.sum().clamp_min(1.0)     # 除以有效 token 总数，非逐序列平均
)
```

（iii）长 response 会贡献更多有效 token；这正是 DAPO 强调的 token-level policy gradient aggregation。Overlong Reward Shaping 应在 `rewards` 进入本函数前完成。

#### A−4. GSPO Objective / Loss（Group Sequence Policy Optimization）

##### 1️⃣ Dense Opening

GSPO 将 GRPO/DAPO 的 token-level importance ratio 改成 **sequence-level likelihood ratio**，再对整个 response 做 sequence-level clipping 与 optimization。核心对象是长度归一化后的 sequence ratio $$s_i(\theta)$$：先对每个 token 的 log-ratio 求平均，再 exponentiate。这样同一 response 内所有 token 共享同一个 ratio。论文实验采用极窄的非对称 clip：$$\epsilon_{\mathrm{low}}=3\times10^{-4}$$、$$\epsilon_{\mathrm{high}}=4\times10^{-4}$$，不要沿用 PPO/GRPO 常见的 `0.2`。[参考 19, 26]

##### 2️⃣ Global Equations

group-relative advantage：

$$
\widehat A_i
=
\frac{
r_i-\mu_g
}{
\sigma_g+\epsilon
}
$$

sequence-level importance ratio：

$$
s_i(\theta)
=
\left(
\frac{
\pi_\theta(y_i\mid x)
}{
\pi_{\theta_{\mathrm{old}}}(y_i\mid x)
}
\right)^{1/|y_i|}
$$

等价 log-space：

$$
s_i(\theta)
=
\exp
\left(
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
[
\log\pi_\theta(y_{i,t})
-
\log\pi_{\mathrm{old}}(y_{i,t})
]
\right)
$$

objective：

$$
J_{\mathrm{GSPO}}
=
\frac{1}{G}
\sum_i
\min
\left(
s_i\widehat A_i,
\operatorname{clip}
(s_i,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})
\widehat A_i
\right)
$$

训练 loss：

$$
L_{\mathrm{GSPO}}
=
-J_{\mathrm{GSPO}}
$$

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准核心实现）**

```python
import torch
from torch import Tensor

def gspo_loss(
    new_logps: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    response_mask: Tensor,
    eps_low: float = 3e-4,
    eps_high: float = 4e-4,
    eps: float = 1e-8,
) -> Tensor:
    # new/old_logps: (B,T) 逐 token log 概率；rewards/group_ids: (B,)
    # response_mask: (B,T)，有效 response token=1
    # 返回: 标量 loss；eps_low/eps_high 是 sequence 级 clip 半径（远小于 PPO 的 0.2）
    B, T = new_logps.shape

    advantages = torch.zeros_like(rewards)   # (B,) 每条 response 一个 advantage

    for gid in group_ids.unique():           # (B,) -> (G,) 去重的 group ID
        idx = group_ids == gid               # (B,) bool mask，选中同 prompt 的 response
        r = rewards[idx]                     # -> (n_g,) 组内 reward

        mean = r.mean()                      # μ_g
        std = r.std(unbiased=False)          # σ_g，有偏 std 与公式一致

        advantages[idx] = (
            r - mean
        ) / (std + eps)                      # 组内 z-score 写回对应位置

    advantages = advantages.detach()         # 当作常数，梯度只经 new_logps

    mask = response_mask.to(
        new_logps.dtype
    )                                      # (B,T) bool/int -> float，便于做加权求和

    lengths = mask.sum(
        dim=1
    ).clamp_min(1.0)                       # (B,) 沿 T 聚合出每条 response 的有效长度 |y_i|

    # token log-ratio -> sequence mean log-ratio
    seq_log_ratio = (
        (
            new_logps - old_logps.detach()   # (B,T) 逐 token log-ratio
        )
        * mask                               # 屏蔽 padding/prompt token 的贡献
    ).sum(dim=1) / lengths                 # (B,T) -> (B,)：长度归一化的平均 log-ratio

    # sequence-level ratio
    seq_ratio = torch.exp(
        seq_log_ratio
    )                                      # (B,)：同一 response 内所有 token 共享此 ratio

    clipped = torch.clamp(
        seq_ratio,
        1.0 - eps_low,
        1.0 + eps_high,                    # 非对称窄 clip，整条 response 一个 clip 决策
    )

    s1 = seq_ratio * advantages            # (B,) 未裁剪项
    s2 = clipped * advantages              # (B,) 裁剪项

    return -torch.minimum(
        s1,
        s2,
    ).mean()                               # 逐 sequence 取悲观项后对 B 条 response 平均
```

**（B）NumPy 版本（同一 sequence objective）**

```python
import numpy as np

def gspo_loss_numpy(
    new_logps,
    old_logps,
    rewards,
    group_ids,
    response_mask,
    eps_low=3e-4,
    eps_high=4e-4,
    eps=1e-8,
):
    # new/old_logps, response_mask: (B,T)；rewards, group_ids: (B,)
    B, T = new_logps.shape
    advantages = np.zeros_like(
        rewards,
        dtype=np.float64,
    )                                       # (B,)

    for gid in np.unique(group_ids):        # (B,) -> (G,) 去重 group ID
        idx = group_ids == gid              # (B,) bool mask，同 prompt 的 response
        r = rewards[idx]                    # -> (n_g,)

        advantages[idx] = (
            r - r.mean()
        ) / (r.std() + eps)                 # 组内 z-score（np.std 默认 ddof=0）

    mask = response_mask.astype(np.float64)  # (B,T) -> float，用作加权系数
    lengths = np.maximum(
        mask.sum(axis=1),                    # (B,T) -> (B,) 每条 response 的有效长度
        1.0,                                 # 下限 1，防空 response 除零
    )

    seq_log_ratio = (
        (
            new_logps - old_logps            # (B,T) 逐 token log-ratio
        )
        * mask                               # 屏蔽无效 token
    ).sum(axis=1) / lengths                  # (B,T) -> (B,) 平均 log-ratio

    seq_ratio = np.exp(seq_log_ratio)        # (B,) sequence-level ratio s_i

    clipped = np.clip(
        seq_ratio,
        1.0 - eps_low,
        1.0 + eps_high,                      # sequence 粒度的非对称窄 clip
    )

    s1 = seq_ratio * advantages
    s2 = clipped * advantages
    return -np.minimum(s1, s2).mean()        # (B,) 逐序列取小后平均 -> 标量 loss
```

**最小可运行测试**

```python
# Demo
B, T = 6, 5                                    # B=6 条 response（2 组×3 条），T=5 个 token 位
new = torch.randn(B, T) * 0.1                  # (B,T) 当前 policy 的 token log-prob
old = new.detach() + 0.02 * torch.randn(B, T)  # (B,T) behavior policy 的 log-prob

rewards = torch.tensor([
    1.0, 0.0, 0.5,
    0.2, 0.8, 0.4,
])                                             # (B,) 连续 reward，两组内部均有差异
group_ids = torch.tensor([
    0, 0, 0,
    1, 1, 1,
])                                             # (B,) 前 3 条属 group 0，后 3 条属 group 1

# (B,T) 有效 token mask：每行 1 的个数即该 response 的真实长度 |y_i|
mask = torch.tensor([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
])

print(
    "GSPO loss:",
    gspo_loss(
        new,
        old,
        rewards,
        group_ids,
        mask,
    ).item(),
)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
from torch import Tensor

def gspo_loss(
    new_logps: Tensor,
    old_logps: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    response_mask: Tensor,
    eps_low: float = 3e-4,
    eps_high: float = 4e-4,
    eps: float = 1e-8,
) -> Tensor:
    # logps/response_mask: (B,T)；rewards/group_ids: (B,)；返回标量 loss
    pass
```

**Step 1：Group advantage**

（i）公式

$$
\widehat A_i
=
(r_i-\mu_g)/(\sigma_g+\epsilon)
$$

（ii）代码

```python
# idx: (B,) bool mask 定位同一 group；r: (n_g,) 组内 reward
advantages[idx] = (
    r - r.mean()
) / (r.std(unbiased=False) + eps)   # 组内 z-score；unbiased=False 对应 σ_g
```

（iii）advantage 仍然来自同 prompt group 的相对 reward。

**Step 2：Sequence mean log-ratio**

（i）公式

$$
\ell_i
=
\frac{1}{|y_i|}
\sum_t
m_{it}
[
\log\pi_\theta
-
\log\pi_{\mathrm{old}}
]
$$

（ii）代码

```python
seq_log_ratio = (
    (
        new_logps - old_logps.detach()   # (B,T) 逐 token log-ratio
    )
    * mask                               # 只保留有效 response token
).sum(dim=1) / lengths                   # 沿 T 聚合 (B,T) -> (B,)，再按 |y_i| 归一
```

（iii）`(B,T) -> (B,)`；这是 GSPO 与 token-level GRPO/DAPO 的分界点。

**Step 3：Exponentiate 得 sequence ratio**

（i）公式

$$
s_i=e^{\ell_i}
$$

（ii）代码

```python
seq_ratio = torch.exp(
    seq_log_ratio      # (B,) 平均 log-ratio
)                      # -> (B,) 几何平均意义下的 sequence ratio s_i
```

（iii）长度归一化让不同 response length 的 ratio 数值尺度更可比。

**Step 4：Sequence-level clipping**

（i）公式

$$
\widetilde s_i
=
\operatorname{clip}
(s_i,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})
$$

（ii）代码

```python
clipped = torch.clamp(
    seq_ratio,         # (B,)，shape 不变
    1.0 - eps_low,     # 论文取 3e-4，远窄于 PPO 的 0.2
    1.0 + eps_high,    # 论文取 4e-4，上界略宽于下界
)
```

（iii）整个 response 使用同一个 clipping decision。论文实验的左右 clip 约为 `3e-4 / 4e-4`，比 PPO/GRPO 的 token-level clip 窄得多。

**Step 5：Sequence objective**

（i）公式

$$
L
=
-\frac{1}{B}
\sum_i
\min(
s_iA_i,
\widetilde s_iA_i
)
$$

（ii）代码

```python
return -torch.minimum(
    s1,                # (B,) s_i·A_i
    s2,                # (B,) clip(s_i)·A_i
).mean()               # 逐 sequence 取小 -> 对 B 条 response 平均 -> 标量
```

（iii）GSPO 直接在 sequence 粒度进行 importance weighting、clipping 与优化。

#### A−5. DDP Gradient All-Reduce（Simplified）

##### 1️⃣ Dense Opening

Distributed Data Parallel（DDP）在每个 worker 保留一份完整模型，各自处理不同 mini-batch；backward 得到 local gradients 后，对每个参数梯度做 all-reduce，再除以 world size 得到全局平均 gradient。核心对象是 $$g=\frac1N\sum_r g^{(r)}$$。面试不需要实现 ring all-reduce，只需正确说明“参数不每步广播，梯度同步后各 worker 执行相同 optimizer step”。[参考 20]

##### 2️⃣ Global Equations

worker $$r$$ 的 local gradient：

$$
g^{(r)}
=
\nabla_\theta
L^{(r)}
$$

all-reduce sum：

$$
g_{\mathrm{sum}}
=
\sum_{r=1}^{N}
g^{(r)}
$$

average：

$$
g
=
\frac{1}{N}
g_{\mathrm{sum}}
$$

各 rank 同步更新：

$$
\theta
\leftarrow
\operatorname{OptimizerStep}
(\theta,g)
$$

只要初始参数一致且 optimizer state 一致，更新后参数继续一致。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（真实 all_reduce 核心）**

```python
import torch
import torch.distributed as dist

def allreduce_gradients(model):
    # 前提：dist.init_process_group 已完成；原地修改 model 各参数的 .grad
    world_size = dist.get_world_size()   # 参与训练的 rank 数 N

    for p in model.parameters():
        if p.grad is None:
            continue                     # 该参数未参与本次 backward，无梯度可同步

        # p.grad: 每个 rank 上 shape 完全相同，因此可逐元素求和
        # 原地 SUM，collective 返回后所有 rank 的 p.grad 都是同一份 g_sum
        dist.all_reduce(
            p.grad,
            op=dist.ReduceOp.SUM,
        )

        # SUM -> average gradient，等价于把各 rank 的样本拼成大 batch 求平均
        p.grad /= world_size
```

**（B）NumPy 版本（模拟多 rank 梯度同步）**

```python
import numpy as np

def ddp_average_gradients_numpy(
    local_grads,
):
    # local_grads:
    #   (world_size, ...)；第 0 维模拟不同 ranks
    #   返回: (...)，即抹掉 rank 维后的平均梯度
    grad_sum = np.sum(
        local_grads,
        axis=0,
    )                                    # 沿 rank 维聚合 -> (...)，模拟 all-reduce SUM
    world_size = local_grads.shape[0]    # rank 数 N
    return grad_sum / world_size
```

**最小可运行测试**

```python
# Demo：单进程模拟 3 个 ranks
local_grads = torch.tensor([          # (3, 2)：3 个 rank，各持一份 2 维参数梯度
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

# 沿 rank 维求均值 (3,2) -> (2,)，等价于 all-reduce SUM 后 / world_size
global_grad = local_grads.mean(dim=0)
print("global grad:", global_grad)
```

##### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.distributed as dist

def allreduce_gradients(model):
    # 目标：把各 rank 的 p.grad 就地换成全局平均梯度，shape 不变
    pass
```

**Step 1：每个 rank 独立 backward**

（i）公式

$$
g^{(r)}
=
\nabla_\theta L^{(r)}
$$

（ii）语义

```python
loss.backward()   # 只用本 rank 的 mini-batch 求导，结果写入各参数的 .grad
```

（iii）此时 `p.grad` 只包含当前 rank 本地 mini-batch 的梯度。

**Step 2：All-Reduce SUM**

（i）公式

$$
g_{\mathrm{sum}}
=
\sum_r g^{(r)}
$$

（ii）代码

```python
dist.all_reduce(          # 原地把 p.grad 换成各 rank 之和，shape 不变
    p.grad,
    op=dist.ReduceOp.SUM,
)
```

（iii）collective 完成后，每个 rank 的 `p.grad` 都变成相同的梯度和。

**Step 3：除以 world size**

（i）公式

$$
g
=
g_{\mathrm{sum}}/N
$$

（ii）代码

```python
p.grad /= world_size   # g_sum -> 平均梯度 g，抵消 SUM 带来的 N 倍放大
```

（iii）与把所有 workers 的样本拼成一个大 batch 后取平均 loss 的梯度相对应。

**Step 4：所有 ranks 独立执行同一 optimizer step**

（i）公式

$$
\theta_r
\leftarrow
\theta_r-\eta g
$$

（ii）语义

```python
optimizer.step()   # 各 rank 用同一份 g 更新，参数保持一致，无需广播权重
```

（iii）因为初始参数和同步后的梯度一致，各 rank 更新后参数仍保持一致；PyTorch DDP 默认会把梯度分 bucket 并尽量让通信与 backward overlap。

### 其他（原文补充）

#### Groupwise InfoNCE

#### 1️⃣ Dense Opening

Groupwise InfoNCE 将每个 query 与自己组内的一个正文档、$$k$$ 个负文档做对比；第 $$i$$ 组只在 $$d_{i0},\ldots,d_{ik}$$ 内归一化，不使用其他 query 的文档。核心对象是温度缩放后的组内相似度 $$s_{ij}$$，训练目标等价于把正文档固定为第 0 类的多分类交叉熵（Cross-Entropy）。

#### 2️⃣ Global Equations

设 queries 为 $$Q\in\mathbb{R}^{G\times D}$$，documents 为 $$\mathcal{D}\in\mathbb{R}^{G\times(k+1)\times D}$$，其中 $$d_{i0}$$ 是第 $$i$$ 个 query 的正文档。点积相似度为：

$$
s_{ij}=\frac{q_i^\top d_{ij}}{\tau},
\qquad
S\in\mathbb{R}^{G\times(k+1)},
\qquad \tau>0
$$

组内 softmax 概率为：

$$
p_{ij}
=\frac{\exp(s_{ij})}
{\sum_{r=0}^{k}\exp(s_{ir})}
$$

单组损失与整体目标为：

$$
\mathcal{L}_i
=-\log p_{i0}
=-\log
\frac{\exp(s_{i0})}
{\sum_{j=0}^{k}\exp(s_{ij})}
$$

$$
\mathcal{L}
=\frac{1}{G}\sum_{i=1}^{G}\mathcal{L}_i
=\operatorname{CrossEntropy}
\left(S,\mathbf{0}_G\right)
$$

其中 $$\mathbf{0}_G$$ 表示长度为 $$G$$、元素全为 0 的类别标签，因为每组第 0 个 document 固定为 positive。对 score 的梯度为：

$$
\frac{\partial\mathcal{L}_i}{\partial s_{ij}}
=p_{ij}-\mathbb{1}[j=0]
$$

#### 3️⃣ Implementation Blocks

##### （A）PyTorch 版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor


def groupwise_infonce_loss(
    queries: Tensor,
    documents: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    # queries: (G, D)；documents: (G, K+1, D)，documents[:, 0] 是正文档
    # 返回: 标量 loss（G 组的均值）
    scores = torch.einsum(               # (G,D) 与 (G,K+1,D) 逐组点积 -> (G,K+1)
        "gd,gnd->gn",                    # g 为组维（不参与求和），d 为特征维（被求和掉）
        queries,
        documents,
    )
    scores = scores / temperature        # 温度越小组内 softmax 越尖锐

    # targets: (G,)，值全为 0，表示每组第 0 列（正文档）是正确类别
    targets = torch.zeros(
        queries.size(0),
        dtype=torch.long,                # cross_entropy 要求类别下标为整型
        device=queries.device,           # 与 scores 同设备，避免跨设备报错
    )
    return F.cross_entropy(scores, targets)   # 在 K+1 维上 log_softmax + NLL，再对 G 取均值
```

##### （B）NumPy 版本（苛刻面试版）

```python
import numpy as np


def groupwise_infonce_loss_numpy(
    queries,
    documents,
    temperature=0.07,
):
    # queries: (G, D)；documents: (G, K+1, D)，documents[:, 0] 是正文档
    # 返回: 标量 loss
    scores = np.einsum(                  # 逐组点积：(G,D) × (G,K+1,D) -> (G,K+1)
        "gd,gnd->gn",
        queries,
        documents,
    )
    scores = scores / temperature

    # 每行减最大值，不改变 softmax，防止 exp 上溢；keepdims 保 (G,1) 以便广播
    scores = scores - np.max(
        scores,
        axis=1,
        keepdims=True,
    )
    log_probs = scores - np.log(         # log-sum-exp 沿类别维 -> (G,1) 广播回 (G,K+1)
        np.exp(scores).sum(axis=1, keepdims=True)
    )
    return -log_probs[:, 0].mean()       # 取第 0 列（正文档）-> (G,)，再对组取均值
```

##### 最小可运行测试

```python
# Demo：G=2，每组 1 个正文档和 2 个负文档
queries = torch.tensor([            # (G=2, D=2)
    [1.0, 0.0],
    [0.0, 1.0],
])
documents = torch.tensor([          # (G=2, K+1=3, D=2)，每组首个向量与该组 query 同向即正例
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
])

torch_loss = groupwise_infonce_loss(
    queries,
    documents,
    temperature=0.5,
)
numpy_loss = groupwise_infonce_loss_numpy(
    queries.numpy(),
    documents.numpy(),
    temperature=0.5,
)
print("PyTorch loss:", torch_loss.item())
print("NumPy loss:", numpy_loss)
```

#### 4️⃣ Step-by-step Derivation

实现前置骨架：

```python
import torch
import torch.nn.functional as F
from torch import Tensor


def groupwise_infonce_loss(
    queries: Tensor,
    documents: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    # queries: (G, D)；documents: (G, K+1, D)；返回标量 loss
    pass
```

##### Step 1：计算组内相似度

（i）公式

$$
s_{ij}=\frac{q_i^\top d_{ij}}{\tau},
\qquad
(G,D)\times(G,k+1,D)\rightarrow(G,k+1)
$$

（ii）代码

```python
scores = torch.einsum(         # (G,D) × (G,K+1,D) -> (G,K+1)：只在组内做点积
    "gd,gnd->gn",
    queries,
    documents,
)
scores = scores / temperature  # 温度缩放，控制组内 softmax 的锐度
```

（iii）`einsum` 只让 $$q_i$$ 与同组的 $$d_{ij}$$ 做点积，不会把其他组的 document 当作负例。除以温度 $$\tau$$ 后，较小的温度会让组内 softmax 更尖锐。

##### Step 2：构造正类下标

（i）公式

$$
y_i=0,\qquad i=1,\ldots,G
$$

（ii）代码

```python
targets = torch.zeros(
    queries.size(0),           # G：组数，每组一个类别标签 -> (G,)
    dtype=torch.long,          # 类别下标须为整型
    device=queries.device,
)
```

（iii）`targets` 的 shape 是 `(G,)`。这里的 0 是文档维上的类别下标，表示每组 `documents[i, 0]` 是正例，并不是二分类标签 0。

##### Step 3：稳定计算 InfoNCE

（i）公式

$$
\mathcal{L}
=-\frac{1}{G}
\sum_{i=1}^{G}
\log
\frac{\exp(s_{i0})}
{\sum_{j=0}^{k}\exp(s_{ij})}
$$

（ii）代码

```python
return F.cross_entropy(scores, targets)   # (G,K+1) + (G,) -> 标量
```

（iii）`F.cross_entropy` 在 `scores` 的最后一维执行稳定的 `log_softmax + NLLLoss`，再对 $$G$$ 个组取均值。时间复杂度为 $$O(G(k+1)D)$$，额外 score 空间为 $$O(G(k+1))$$。

#### 👀 SGD（Stochastic Gradient Descent，随机梯度下降）

单样本估计梯度迭代逼近极小值。核心对象：参数 $$\theta$$，沿负梯度方向更新 $$\theta \leftarrow \theta - \eta \nabla_\theta L$$。

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t; x_i, y_i)
$$

线性回归常用最小二乘损失与解析梯度：
$$
L(\theta; x_i, y_i) = \tfrac{1}{2}(y_i - x_i^\top \theta)^2, \quad \nabla_\theta L = (x_i^\top \theta - y_i)\, x_i
$$

#### （A）PyTorch 版本（autograd 写法）

```python
import torch

class SGD:
    def __init__(self, lr=1e-2, max_iter=100, eps=1e-2):
        self.lr, self.max_iter, self.eps = lr, max_iter, eps   # lr 步长；max_iter 最大轮数；eps 早停阈值

    def fit(self, X, y):
        # X: (N, D) 样本特征；y: (N,) 标签；返回学好的 theta: (D,)
        N, D = X.shape
        self.theta = torch.randn(D, requires_grad=True)   # (D,) 参数向量，需追踪梯度
        for _ in range(self.max_iter):
            idx = torch.randperm(N)                  # 样本下标的随机置换 -> (N,)
            total = 0.0                              # 本轮 loss 累加，用于早停判断
            for i in idx:
                xi, yi = X[i], y[i]                  # 沿样本维取第 i 行：xi: (D,)  yi: ()
                pred = xi @ self.theta               # (D,)·(D,) -> ()，标量预测值
                loss = 0.5 * (pred - yi) ** 2
                loss.backward()                      # 梯度累加到 self.theta.grad: (D,)
                with torch.no_grad():                # 参数更新本身不入计算图
                    self.theta -= self.lr * self.theta.grad
                    self.theta.grad.zero_()          # 清零，否则下个样本的梯度会叠加
                total += loss.item()
            if total / N < self.eps: break           # 平均 loss 足够小则提前收敛
        return self.theta.detach()                   # 切断计算图，返回纯数值参数
```

#### （B）NumPy 版本（手写梯度，苛刻面试版）

```python
import numpy as np

class SGD:
    def __init__(self, lr=1e-2, max_iter=100, eps=1e-2):
        self.lr, self.max_iter, self.eps = lr, max_iter, eps   # lr 步长；max_iter 最大轮数；eps 早停阈值

    def fit(self, X, y, loss_fn, grad_fn):
        # X: (N, D)  y: (N,) 或 (N,1)；loss_fn/grad_fn 接收单样本，返回学好的 theta: (D,)
        N, D = X.shape
        self.theta = np.random.randn(D)              # (D,) 参数向量
        for _ in range(self.max_iter):
            idx = np.random.permutation(N)           # 打乱样本下标 -> (N,)
            Xs, ys = X[idx], y[idx].ravel()          # 按 idx 重排样本；ravel 压平成 (N,)
            total = 0.0
            for i in range(N):
                xi, yi = Xs[i], ys[i]                # xi: (D,)  yi: 标量
                total += loss_fn(xi, yi, self.theta) # 更新前 loss，与梯度同一参数点
                g = grad_fn(xi, yi, self.theta)        # 单样本梯度 (D,)
                self.theta -= self.lr * g            # 沿负梯度走一步
            if total / N < self.eps: break           # 平均 loss 达阈值则早停
        return self.theta

    def predict(self, X): return X @ self.theta      # (N,D)·(D,) -> (N,)

def linear_loss(x, y, theta): return 0.5 * (y - x @ theta) ** 2   # 单样本最小二乘
def linear_grad(x, y, theta): return (x @ theta - y) * x           # 标量残差广播回 (D,)

# Demo：单特征线性回归，真实斜率为 3
np.random.seed(42)
X = np.random.randn(100, 1); y = 3 * X + 0.1 * np.random.randn(100, 1)   # X:(100,1) y:(100,1)
print(SGD(lr=0.05, max_iter=200).fit(X, y, linear_loss, linear_grad))    # 期望 theta ≈ [3.0]
```

#### Step-by-step

```python
import numpy as np
def sgd_step(theta, xi, yi, lr): pass   # theta:(D,) xi:(D,) yi:标量 -> 更新后的 theta:(D,)
```

（i）单样本预测 $$\hat{y} = x_i^\top \theta$$
（ii）`pred = xi @ theta`
（iii）将参数和样本特征内积得到当前预测值。

（i）单样本损失 $$L = \tfrac{1}{2}(\hat{y} - y_i)^2$$
（ii）`loss = 0.5 * (pred - yi) ** 2`
（iii）取 0.5 让梯度形式 `(pred-y)*x` 更干净。

（i）解析梯度 $$\nabla_\theta L = (\hat{y} - y_i) x_i$$
（ii）`g = (pred - yi) * xi`
（iii）链式法则一步到位，无需 autograd。

（i）参数更新 $$\theta \leftarrow \theta - \eta g$$
（ii）`theta -= lr * g`
（iii）SGD 唯一一行核心更新。

<a id="probability-bce"></a>

#### 👀 KL Divergence（Kullback-Leibler 散度）

衡量两个离散分布 $$P$$、$$Q$$ 的不对称差异。核心对象：逐点 log-ratio $$\log(p_i/q_i)$$，由 $$p_i$$ 加权。

$$
D_{KL}(P\,\|\,Q)=\sum_i p_i\log\frac{p_i}{q_i}
$$

要求 $$P,Q$$ 是概率分布。数值实现中只在 `log` 的输入处做下界保护；不要先把整个 $$p$$ clamp 后再把被修改的 $$p$$ 当权重。

#### （A）PyTorch 版本

```python
import torch


def kl_divergence(p, q, eps=1e-10):
    # p,q: (...,C)，最后一维为类别；eps 只用于保护 log 输入，返回标量
    p = p / p.sum(-1, keepdim=True)              # 沿类别维归一化成概率；keepdim 保证广播
    q = q / q.sum(-1, keepdim=True)
    log_p = torch.log(p.clamp_min(eps))          # 下界截断避免 log(0) = -inf
    log_q = torch.log(q.clamp_min(eps))
    return (p * (log_p - log_q)).sum(-1).mean()  # 用未截断的 p 加权求和 -> (...)，再对样本取均值
```

#### （B）NumPy 版本

```python
import numpy as np


def kl_divergence(p, q, eps=1e-10):
    # p,q: (...,C)，最后一维为类别；返回标量
    p = np.asarray(p, dtype=np.float64)          # 统一转 float64，避免整型输入除法失真
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)        # 沿类别维归一化成概率分布
    q = q / q.sum(axis=-1, keepdims=True)
    log_p = np.log(np.clip(p, eps, None))        # 只在 log 输入处设下界，权重仍用原始 p
    log_q = np.log(np.clip(q, eps, None))
    return np.sum(p * (log_p - log_q), axis=-1).mean()   # 类别维聚合 -> (...)，再取样本均值

# Demo：单条 3 类分布
p = np.array([0.4, 0.4, 0.2])   # (C=3,)
q = np.array([0.5, 0.3, 0.2])   # (C=3,)
print('KL(p||q) =', kl_divergence(p, q))
```

#### Step-by-step

```python
def kl(p, q, eps=1e-10): pass   # p,q:(...,C) -> 标量 D_KL(P||Q)，不对称
```

（i）归一化：`p = p / p.sum(-1, keepdim=True)`，`q` 同理。

（ii）数值保护只放进 log：`log_p = log(p.clamp_min(eps))`、`log_q = log(q.clamp_min(eps))`。

（iii）逐点项：`p * (log_p - log_q)`；沿类别维求和得到每个样本的 $$D_{KL}(P\|Q)$$。

（iv）若直接用 `F.kl_div`，PyTorch 的 `input` 约定是 log-probability；手写面试题用上面的显式公式通常更不易混淆。

#### 👀 MQA（Multi-Query Attention）

所有 Q-head 共享同一组 K、V。核心动机：把 KV head 数降到 1，显著减小 KV cache 与 decode 内存带宽。

$$
Q\in\mathbb{R}^{B\times H\times T\times d_k},\qquad
K,V\in\mathbb{R}^{B\times1\times T\times d_k}
$$

K/V 只投影到 $$d_k$$ 维，通过 broadcasting 与所有 Q-head 计算 attention。

#### （A）PyTorch 版本

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MQA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0                  # 保证 D 能被 H 整除
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads             # d_k：每个 head 的维度
        self.Wq = nn.Linear(d_model, d_model)            # Q 保留 H 个 head
        self.Wk = nn.Linear(d_model, self.head_dim)     # 单 KV head：只投影到 d_k
        self.Wv = nn.Linear(d_model, self.head_dim)
        self.Wo = nn.Linear(d_model, d_model)            # output projection，合并 heads 后使用

    def forward(self, x):
        # x: (B,T,D)；返回 (B,T,D)
        B, T, _ = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B,T,D)->(B,T,H,d_k)->(B,H,T,d_k)
        K = self.Wk(x).unsqueeze(1)                    # (B,T,d_k) -> (B,1,T,d_k)，head 维占位
        V = self.Wv(x).unsqueeze(1)

        # Q:(B,H,T,d_k) @ K^T:(B,1,d_k,T) -> (B,H,T,T)；K 的 head 维为 1，matmul 时自动广播到 H
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)   # 除 sqrt(d_k) 防 softmax 饱和
        attn = F.softmax(scores, dim=-1)               # 沿 key 维归一化，每行和为 1
        out = torch.matmul(attn, V)                    # (B,H,T,T) @ (B,1,T,d_k) -> (B,H,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)   # (B,H,T,d_k)->(B,T,H,d_k)->(B,T,D) 合并 heads
        return self.Wo(out)                            # (B,T,D)
```

#### （B）NumPy 版本

```python
import numpy as np


def mqa_numpy(X, Wq, Wk, Wv, Wo, H):
    # X: (B,T,D)；Wq/Wo: (D,D)；Wk/Wv: (D,d_k)；返回 (B,T,D)
    B, T, D = X.shape
    d_k = D // H                                       # 每个 head 的维度
    Q = (X @ Wq).reshape(B, T, H, d_k).transpose(0, 2, 1, 3)   # (B,T,D)->(B,T,H,d_k)->(B,H,T,d_k)
    K = (X @ Wk)[:, None]                              # (B,T,d_k) -> (B,1,T,d_k)，None 插入 head 维
    V = (X @ Wv)[:, None]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)   # K 转置末两维后 (B,H,T,d_k)@(B,1,d_k,T) -> (B,H,T,T)
    a = scores - scores.max(-1, keepdims=True)         # 沿 key 维减最大值，防 exp 上溢
    attn = np.exp(a) / np.exp(a).sum(-1, keepdims=True)   # 手写 softmax，(B,H,T,T) 每行和为 1
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, T, D)   # (B,H,T,d_k)->(B,T,H,d_k)->(B,T,D)
    return out @ Wo                                    # (B,T,D)@(D,D) -> (B,T,D)

# Demo：B=2 batch，T=5 序列长度，D=16 模型维，H=4 个 Q-head（KV 只有 1 个）
B, T, D, H = 2, 5, 16, 4
X = np.random.randn(B, T, D)
Wq = np.random.randn(D, D) * 0.1
Wk = np.random.randn(D, D // H) * 0.1                  # KV 只投影到 d_k，故第 2 维是 D//H
Wv = np.random.randn(D, D // H) * 0.1
Wo = np.random.randn(D, D) * 0.1
print(mqa_numpy(X, Wq, Wk, Wv, Wo, H).shape)           # 期望 (2,5,16)
```

#### Step-by-step

```python
def mqa(x): pass   # x:(B,T,D) -> (B,T,D)；Q 多头、K/V 单头共享
```

（i）Q 多头、KV 单头：`Wq: D->D`，`Wk/Wv: D->d_k`。

（ii）`K = self.Wk(x).unsqueeze(1)` 把 KV 变成 `(B,1,T,d_k)`；head 维在 attention matmul 时广播到 H，无需显式 repeat。

（iii）`scores -> softmax -> @V` 与 MHA 相同。

（iv）合并 H 个 Query heads 后做 `Wo: D->D` output projection。公开 MQA 面试实现同样保留 `W_o`；省略它会漏掉标准 attention 的最终线性映射。

#### 👀 FFN（Feed-Forward Network）

Transformer block 的逐位置 MLP，两层 Linear 夹一个非线性。核心对象：升维投影 + 激活 + 降维。

$$
\text{FFN}(x) = W_2 \cdot \phi(W_1 x + b_1) + b_2
$$

经典 $$\phi = \text{ReLU}$$（Transformer 原始）；现代 LLM 多用 GELU 或 SwiGLU。隐藏维 $$d_{\text{ff}} \approx 4 d_{\text{model}}$$。

#### （A）PyTorch 版本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)      # 升维投影，d_ff 通常取 4*d_model
        self.fc2 = nn.Linear(d_ff, d_model)      # 降维回 d_model

    def forward(self, x):
        # x: (..., d_model) -> 同形状；逐位置作用，不跨 token 混合信息
        return self.fc2(F.relu(self.fc1(x)))     # d_model -> d_ff -> ReLU -> d_model
```

#### （B）NumPy 版本

```python
import numpy as np

def ffn_numpy(x, W1, b1, W2, b2):
    # x: (..., D)   W1: (D, H)   W2: (H, D)；返回 (..., D)
    h = np.maximum(0, x @ W1 + b1)                     # 升维 (...,D)->(...,H)，b1 广播后取 ReLU
    return h @ W2 + b2                                 # 降维 (...,H)->(...,D)

# Demo：B=2, T=4, d_model=8, d_ff=32
x = np.random.randn(2, 4, 8)
W1, b1 = np.random.randn(8, 32), np.zeros(32)
W2, b2 = np.random.randn(32, 8), np.zeros(8)
print(ffn_numpy(x, W1, b1, W2, b2).shape)              # 期望 (2,4,8)，与输入同形
```

#### 👀 SFT Loss（Supervised Fine-Tuning Loss）

SFT 的核心仍是 causal language modeling 的 next-token cross-entropy；与预训练相比，通常把 prompt / padding 的 label 设为 `ignore_index=-100`，只让 response token 贡献 loss。

$$
L_{\text{SFT}}
= -\frac{1}{|\mathcal M|}
\sum_{(b,t)\in\mathcal M}
\log \pi_\theta(x_{b,t}\mid x_{b,<t})
$$

因此 raw `logits: (B,T,V)` 与 `labels: (B,T)` 要先做 next-token shift：`logits[:, :-1]` 对齐 `labels[:, 1:]`。

#### （A）PyTorch 版本

```python
import torch
import torch.nn.functional as F


def sft_loss(logits, labels, ignore_index=-100):
    # logits: (B,T,V)；labels: (B,T)，prompt/pad 位置已置 -100；返回标量
    shift_logits = logits[:, :-1, :].contiguous()      # 丢最后一位：(B,T,V)->(B,T-1,V)，作为预测位
    shift_labels = labels[:, 1:].contiguous()          # 丢第一位：(B,T)->(B,T-1)，即 next token
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # 展平 batch 与时间维 -> (B*(T-1),V)
        shift_labels.view(-1),                         # (B*(T-1),)，与上面逐元素对齐
        ignore_index=ignore_index,                     # -100 位置不计入 loss 也不参与分母
    )
```

#### （B）NumPy 版本

```python
import numpy as np


def sft_loss_numpy(logits, labels, ignore_index=-100):
    # logits: (B,T,V)；labels: (B,T)；返回标量。next-token shift
    x = logits[:, :-1, :]                              # (B,T,V) -> (B,T-1,V)
    y = labels[:, 1:]                                  # (B,T) -> (B,T-1)
    V = x.shape[-1]                                    # 词表大小
    x = x.reshape(-1, V)                               # 展平成 (B*(T-1),V)
    y = y.reshape(-1)                                  # (B*(T-1),)

    mask = y != ignore_index                           # bool (N,)：True 表示该位置计入 loss
    x, y = x[mask], y[mask]                            # 布尔索引筛行 -> (M,V)/(M,)，只保留 response token
    x = x - x.max(axis=-1, keepdims=True)              # 沿词表维减最大值，防 exp 上溢
    log_probs = x - np.log(np.exp(x).sum(-1, keepdims=True))   # log_softmax -> (M,V)
    return -log_probs[np.arange(len(y)), y].mean()     # 按 (行号, 目标 token) 取值 -> (M,)，再取均值

# Demo：B=2, T=4, V=6；-100 表示 prompt/pad 位置
B, T, V = 2, 4, 6
logits = np.random.randn(B, T, V)
labels = np.array([                                    # (B,T)，只有后面几个 token 贡献 loss
    [-100, -100, 2, 3],
    [-100,    1, 4, 5],
])
print('SFT loss:', sft_loss_numpy(logits, labels))
```

#### Step-by-step

```python
def sft_loss(logits, labels): pass   # logits:(B,T,V) labels:(B,T) -> 标量，只算 response token
```

（i）next-token 对齐：`shift_logits = logits[:, :-1, :]`、`shift_labels = labels[:, 1:]`。shape 从 `(B,T,V)/(B,T)` 变为 `(B,T-1,V)/(B,T-1)`。

（ii）flatten：`(B,T-1,V) -> (B*(T-1),V)`，labels 同步变成 `(B*(T-1),)`。

（iii）`F.cross_entropy(..., ignore_index=-100)` 自动跳过 prompt / padding label。

（iv）如果上游明确已经提前做过 shift，则这里不要再 shift；本题默认函数接收 raw causal-LM logits 与同长度 labels。

#### 👀 GAE（Generalized Advantage Estimation）

GAE 用 $$\lambda$$ 对多步 TD error 做指数加权，控制 bias-variance trade-off。标准实现还要处理 episode boundary：terminal 后不能把下一 episode 的 value bootstrap 回来。

$$
\delta_t
= r_t + \gamma(1-d_t)V(s_{t+1}) - V(s_t)
$$

$$
A_t
= \delta_t + \gamma\lambda(1-d_t)A_{t+1}
$$

其中 $$d_t=1$$ 表示该 transition 在 $$t$$ 后真正终止。

#### （A）PyTorch 版本

```python
import torch


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                lam: float = 0.95) -> torch.Tensor:
    # rewards,dones: (T,)；values: (T+1,) 多出的一位是 V(s_T) 用于 bootstrap；返回 adv: (T,)
    T = rewards.numel()
    adv = torch.zeros_like(rewards)          # (T,) 输出缓冲
    gae = 0.0                                # running A_{t+1}，从轨迹末尾往前累积

    for t in reversed(range(T)):             # 必须倒序：A_t 依赖 A_{t+1}
        not_done = 1.0 - dones[t].float()    # bool/int -> float；terminal 处为 0，切断 bootstrap
        delta = (                            # TD error δ_t
            rewards[t]
            + gamma * values[t + 1] * not_done
            - values[t]
        )
        gae = delta + gamma * lam * not_done * gae   # 同一 mask 也阻断 advantage 跨 episode 传播
        adv[t] = gae
    return adv
```

#### （B）NumPy 版本

```python
import numpy as np


def compute_gae_numpy(rewards, values, dones,
                      gamma=0.99, lam=0.95):
    # rewards,dones: (T,)；values: (T+1,)；gamma 折扣，lam 控制 bias-variance；返回 (T,)
    T = len(rewards)
    adv = np.zeros_like(rewards, dtype=float)   # 强制 float，避免整型 rewards 截断结果
    gae = 0.0

    for t in reversed(range(T)):                # 倒序递推
        not_done = 1.0 - float(dones[t])        # terminal 处为 0
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]   # TD error
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    return adv

# Demo：t=2 与最后一步都结束 episode，递推不会跨边界传播
rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0])        # (T=5,)
values = np.array([0.5, 0.6, 0.4, 0.3, 0.2, 0.0])    # (T+1=6,)，末位 V(s_T)=0
dones = np.array([0, 0, 1, 0, 1])                    # (T=5,)，1 表示该步之后 episode 终止
print('GAE:', compute_gae_numpy(rewards, values, dones))
```

#### Step-by-step

```python
def gae(rewards, values, dones, gamma, lam): pass   # rewards/dones:(T,) values:(T+1,) -> adv:(T,)
```

（i）终止 mask：`not_done = 1.0 - dones[t].float()`。

（ii）TD error：`delta = r[t] + gamma * V[t+1] * not_done - V[t]`。terminal 时 `not_done=0`，不 bootstrap 下一状态。

（iii）反向递推：`gae = delta + gamma * lam * not_done * gae`。同一个 mask 同时阻断 advantage 跨 episode 传播。

（iv）若轨迹只是被固定 horizon / epoch cutoff 截断而非真正 terminal，可以保留 bootstrap value；工程代码通常会区分 terminated 与 truncated。

（v）$$\lambda=0$$ 接近单步 TD；$$\lambda\to1$$ 更接近长回报，方差通常更大。

附：所有损失函数的形状约定

| **函数**      | **输入形状**                            | **备注**                  |
| ------------- | --------------------------------------- | ------------------------- |
| `sft_loss`    | `logits` `(B, T, V)`, `labels` `(B, T)` | `ignore_index = -100`     |
| `dpo_loss`    | 四个 `(B,)`                             | log-probs                 |
| `ppo_loss`    | 三个 `(N,)`                             | N = B 或 N = B \times T   |
| `grpo_loss`   | `new/old/ref_logp, mask` `(G,T)`；`rewards` `(G,)` | 逐 completion-token ratio；原始 GRPO 先按 response 求均值 |
| `compute_gae` | `rewards/dones` `(T,)`, `values` `(T + 1,)` | terminal 用 `dones` 截断 bootstrap |

#### 网络实现核对来源（2026-09）

本轮只吸收“更正确或明显更适合面试手写”的部分；公开题库中更长的 framework / production 版本没有机械照搬。

**本轮实际改动很少：**
- RMSNorm：PyTorch 版的平方、均值与 `rsqrt` 提升到 `float32`，再 cast 回输入 dtype；与 Meta Llama reference 的数值处理一致。
- RoPE：PyTorch 版的 position / frequency / trigonometric angle 用 `float32` 计算，再 cast 回输入 dtype；保留原来的 offset 与 shape 写法。
- GRPO：将整句 `(G,)` 聚合 log-prob 的过度简化改为逐 completion-token `(G,T)` ratio + mask，并保留原始 GRPO 的 per-response normalization。
- MLA：补充说明示例返回 cache 内容但未读取历史 cache，避免把“cache shape 展示”误解成完整增量 decode。

- [ckd0817/LLM-Interview-Code](https://github.com/ckd0817/LLM-Interview-Code)：RMSNorm、GQA、SFT、DPO/PPO/GRPO、LoRA 等面试实现与 shape 约定。
- [cdhx/LLM-Code-Hot-100](https://github.com/cdhx/LLM-Code-Hot-100)：MHA/GQA/MQA、Causal Mask、Top-k/Top-p/Beam Search 等高频手写版本。
- [AIR-hl/llm-interview-code](https://github.com/AIR-hl/llm-interview-code)：MHA、KV Cache、RMSNorm、GQA、InfoNCE、DPO、sampling 等真实面试手撕整理。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：标准接口为 unnormalized logits + class-index targets。
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.LayerNorm.html)：LayerNorm 使用 biased variance（`correction=0`）。
- [PyTorch RMSNorm](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.RMSNorm.html)：标准 RMS 为 `sqrt(mean(x^2) + eps)`。
- [PyTorch KL divergence](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.kl_div.html)：官方 `kl_div` 的 `input` 约定为 log-probabilities；手写版显式实现定义。
- [Hugging Face: decoding methods](https://huggingface.co/blog/how-to-generate)：Top-p 保留累计概率至少达到阈值的最小 token 集合。
- [DPO reference implementation](https://github.com/eric-mitchell/direct-preference-optimization)：标准 `pi_logratios - ref_logratios -> -logsigmoid(beta * logits)`。
- [Hugging Face TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)：`beta` 控制相对 reference 的偏离，较大 `beta` 通常意味着更少偏离。
- [OpenAI Spinning Up PPO/GAE](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)：terminal 时 bootstrap value 为 0；非 terminal cutoff 可用 value bootstrap。
- [Meta Llama reference implementation](https://github.com/meta-llama/llama-models)：RMSNorm 在 `float32` 中做归一化；RoPE reference 同样避免直接用低精度 position/frequency 做角度计算。
- [Hugging Face TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)：GRPO ratio/clipping 是逐 completion-token 定义，并明确区分原始 `grpo`、`dapo`、`dr_grpo` 的聚合方式。
- [AIR-hl/llm-interview-code · GRPO](https://github.com/AIR-hl/llm-interview-code)：面试题库实现同样按 token log-prob + completion mask 展开，作为本轮面试可写性对照。

---

### 参考文献（补充题单）

[1] Loshchilov, I., Hutter, F. **Decoupled Weight Decay Regularization**. ICLR 2019.  
https://arxiv.org/abs/1711.05101

[2] Vaswani, A. et al. **Attention Is All You Need**. NeurIPS 2017.  
https://arxiv.org/abs/1706.03762

[3] Shazeer, N. **Fast Transformer Decoding: One Write-Head is All You Need**. 2019.  
https://arxiv.org/abs/1911.02150

[4] DeepSeek-AI. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**. 2024.  
https://arxiv.org/abs/2405.04434

[5] Su, J. et al. **RoFormer: Enhanced Transformer with Rotary Position Embedding**. 2021.  
https://arxiv.org/abs/2104.09864

[6] Schulman, J. et al. **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. 2015.  
https://arxiv.org/abs/1506.02438

[7] Ouyang, L. et al. **Training Language Models to Follow Instructions with Human Feedback**. NeurIPS 2022.  
https://arxiv.org/abs/2203.02155

[8] Touvron, H. et al. **LLaMA: Open and Efficient Foundation Language Models**. 2023.  
https://arxiv.org/abs/2302.13971

[9] van den Oord, A., Li, Y., Vinyals, O. **Representation Learning with Contrastive Predictive Coding**. 2018.  
https://arxiv.org/abs/1807.03748

[10] Xiong, R. et al. **On Layer Normalization in the Transformer Architecture**. ICML 2020.  
https://arxiv.org/abs/2002.04745

[11] Press, O., Smith, N. A., Lewis, M. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation**. ICLR 2022.  
https://arxiv.org/abs/2108.12409

[12] Dettmers, T. et al. **QLoRA: Efficient Finetuning of Quantized LLMs**. NeurIPS 2023.  
https://arxiv.org/abs/2305.14314

[13] Chen, T. et al. **Training Deep Nets with Sublinear Memory Cost**. 2016.  
https://arxiv.org/abs/1604.06174

[14] PyTorch Documentation. **Automatic Mixed Precision**.  
https://docs.pytorch.org/docs/stable/accelerator/amp.html

[15] Hinton, G., Vinyals, O., Dean, J. **Distilling the Knowledge in a Neural Network**. 2015.  
https://arxiv.org/abs/1503.02531

[16] Kwon, W. et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention**. SOSP 2023.  
https://arxiv.org/abs/2309.06180

[17] Yu, G.-I. et al. **Orca: A Distributed Serving System for Transformer-Based Generative Models**. OSDI 2022.  
https://www.usenix.org/conference/osdi22/presentation/yu

[18] Yu, Q. et al. **DAPO: An Open-Source LLM Reinforcement Learning System at Scale**. 2025.  
https://arxiv.org/abs/2503.14476

[19] Zheng, C. et al. **Group Sequence Policy Optimization**. 2025.  
https://arxiv.org/abs/2507.18071  
Qwen Team: https://qwenlm.github.io/blog/gspo/

[20] PyTorch Documentation. **DistributedDataParallel / torch.distributed**.  
https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html  
https://docs.pytorch.org/docs/stable/distributed.html

[21] ckd0817. **LLM-Interview-Code — SFT / Pretrain Loss implementations**.  
https://github.com/ckd0817/LLM-Interview-Code

[22] AIR-hl. **llm-interview-code — LLM 面试常见手撕合集**.  
https://github.com/AIR-hl/llm-interview-code

[23] cdhx. **LLM-Code-Hot-100 — 大模型面试手撕代码**.  
https://github.com/cdhx/LLM-Code-Hot-100

[24] PyTorch Tutorials. **Knowledge Distillation Tutorial**.  
https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html

[25] verl-project. **DAPO Recipe / Implementation Notes**.  
https://github.com/verl-project/verl-recipe/tree/main/dapo

[26] Hugging Face TRL. **GSPO paper implementation settings**.  
https://huggingface.co/docs/trl/paper_index


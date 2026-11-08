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

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


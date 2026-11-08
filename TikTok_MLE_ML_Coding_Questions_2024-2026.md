# TikTok / ByteDance MLE 手撕代码题库（2024–2026）

> **整理日期：2026-11-08。题库以题目为主，不提供答案；每题都有独立 Follow-up。**
> **重要：** `[原始追问]` 仅用于资料明确记录的追问；其余归类为模拟 Follow-up，用于你练习，不代表面试官原话。

## 阅读规则

- **A｜公开单场/候选人记录：** 来源对应具体面试或作者帖子；仍是个人回忆，未经企业官方证实。
- **B｜整理稿/题库/改编题：** 记录该公司相关题目，但题面可能被编辑或改写。
- **U｜你提供的线索：** 来自你在本对话提供的清单，尚未统一核实面试发生月份。
- **E｜扩展训练题：** 在真实考点基础上主动补充，并不声称在 TikTok 考过。

**时间口径：** `面试时间`、`公开面经日期`、`帖子发布日`、`文章更新日` 严格区分。能确认的年月用 `YYYY-MM`；不能确认的标明“年月未核实”，绝不把文章发布于 2026 误当作题在 2026 年考过。

**建议练习顺序：** P0 → P1 → P2；优先查看 2026 年的单场记录，然后看 2025 年与 2024 年的经典题。每题至少完成：独立实现 → 3 个边界测试 → 3 个 Follow-up。

## 数量与快速导航

- 共 **120** 道独立题目（P0 28 / P1 73 / P2 19）。
- 每一个**题目标题均为 Markdown 四级标题 `####`**；分类仅用二级标题 `##`。
- 同题在不同来源/岗位出现时优先合并，题意不完全相同时用备注区分。

- 01｜Transformer、Attention、核心算子（ML Coding）：16 题
- 02｜Loss、训练目标与采样（ML Coding）：21 题
- 03｜推荐系统指标、排序与召回：9 题
- 04｜NumPy、传统机器学习、数值编程：14 题
- 05｜TikTok OA：题目和编程填空：5 题
- 06｜LeetCode / DSA：明确出现的题与贴近变体：52 题
- 07｜AI-Assisted Coding / 工程现场题：3 题

## 2026 最新面经定位（优先读）

| 时间（性质） | 考察点 | 证据来源 |
|---|---|---|
| 2026-11（整理；覆盖 8–10 月） | Self-Attention + CE / 课程表拓扑排序 | [2026 Q4 统计](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4) |
| 2026-10（面试） | BCE、VAE Loss、Transformer 复杂度、三维箱子堆叠 | [牛客原帖](https://www.nowcoder.com/discuss/946798743163088896) |
| 2026-10（帖子） | 美国 RS Intern PyTorch MHA / Search MHA + Kth Largest | [RS 面经](https://www.1point3acres.com/interview/thread/1193792) / [Search 面经](https://www.1point3acres.com/interview/thread/1192875) |
| 2026-09（面试） | PPO/DPO/GRPO 公式、大数加法、双栈队列 | [牛客面经](https://www.nowcoder.com/discuss/940390216378888192) |
| 2026-09（整理） | OA：Decision Tree、GD 填空、字符统计 | [OA 题目](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/) |
| 2026-07～08（面试） | 岛屿 DFS/BFS、SFT Loss 提问 | [牛客面经](https://www.nowcoder.com/discuss/922309082860507136) |
| 2026-06～07（记录） | 单头 Attention、CE、LIS、堆、DP、Skyline | [MLE 记录集](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer) |
| 2026-04（发布） | Request GAUC / TikTok MHA + 平方根 | [GAUC](https://www.nowcoder.com/feed/main/detail/8aba7094ae844c3da7e31730f042f037) / [TikTok 搜推](https://www.nowcoder.com/users/214711400) |
| 2025-08（面经） | TikTok RecSys AUC | [Jointaro](https://www.jointaro.com/interviews/companies/tiktok/experiences/mlers-san-diego-california-august-1-2025-no-offer-neutral-1d743d60/) |
| 2025-02（面经） | CE 推导、Transformer Block、树题 | [Jointaro](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/) |
| 2024-09（发布） | 字节 DIN、InfoNCE、BCE、AUC、梯度下降 | [牛客](https://www.nowcoder.com/discuss/667864275997687808) |

---

## 01｜Transformer、Attention、核心算子（ML Coding）

#### Q001｜单头 Scaled Dot-Product Self-Attention

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06（面经标注月份）
- **题目：** 只使用 PyTorch 张量操作实现单头自注意力：线性投影得到 Q/K/V，缩放点积、Softmax，再加权 V。
- **输入输出 / 约束：** 输入 `x:[B,T,D]`、`d_head`；输出 `[B,T,d_head]` 或按要求映射回 `[B,T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. **[原始追问]** 说明为什么要除以 `sqrt(d_head)`；列出每一步的 tensor shape。
  2. 让它支持任意可广播的 Attention Mask，规定 True 是“允许”还是“禁止”。
  3. 如果输入 T 很长，时间与激活显存的复杂度分别是什么？
- **来源：** [Screna：TikTok Senior MLE 自注意力、交叉熵](https://www.screna.ai/experience/4498ae13-338f-4777-97dc-36b24b0b612a)

##### 1️⃣ Dense Opening

单头自注意力（Single-Head Self-Attention）让序列中每个 token 按相似度聚合其他 token 的信息，结构是“线性投影 Q/K/V → 缩放点积 → Softmax → 加权 V”，数学核心对象是注意力权重矩阵（attention weights）$A\in\mathbb{R}^{B\times T\times T}$。

##### 2️⃣ Global Equations

投影得到 Q/K/V：

$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v,\quad Q,K,V\in\mathbb{R}^{B\times T\times d_h}
$$

主目标，缩放点积注意力（Scaled Dot-Product Attention）：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

中间对象是分数矩阵 $S$ 与权重矩阵 $A$，其中 $A$ 的每一行和为 1：

$$
S=\frac{QK^\top}{\sqrt{d_h}},\quad A_{ij}=\frac{\exp(S_{ij})}{\sum_{l=1}^{T}\exp(S_{il})}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        # 缩放因子要用的单头维度
        self.d_head = d_head
        # Query 投影：D → d_head
        self.w_q = nn.Linear(d_model, d_head, bias=False)
        # Key 投影：D → d_head
        self.w_k = nn.Linear(d_model, d_head, bias=False)
        # Value 投影：D → d_head
        self.w_v = nn.Linear(d_model, d_head, bias=False)
        # 输出投影：d_head → D
        self.w_o = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # x: (B, T, D) → q: (B, T, d_head)
        q = self.w_q(x)
        # x: (B, T, D) → k: (B, T, d_head)
        k = self.w_k(x)
        # x: (B, T, D) → v: (B, T, d_head)
        v = self.w_v(x)
        # k^T: (B, d_head, T)，matmul → scores: (B, T, T)
        # scores[b, i, j]：第 i 个 query 对第 j 个 key 的相似度
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        # mask 可广播到 (B, T, T)，约定 True=允许，False=禁止
        if mask is not None:
            # 取反后填 -inf，禁止位置的权重变为 0
            scores = scores.masked_fill(~mask, float("-inf"))
        # 沿 key 维归一化，A: (B, T, T)，每行和为 1
        A = scores.softmax(dim=-1)
        # (B, T, T) @ (B, T, d_head) → (B, T, d_head)
        out = A @ v
        # (B, T, d_head) → (B, T, D)
        return self.w_o(out)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢，比值不变；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def self_attention_np(x, w_q, w_k, w_v, w_o, mask=None):
    # 权重形状：w_q/w_k/w_v: (D, d_head)，w_o: (d_head, D)
    d_head = w_q.shape[1]
    # x: (B, T, D) → q/k/v: (B, T, d_head)
    q, k, v = x @ w_q, x @ w_k, x @ w_v
    # swapaxes 得 k^T: (B, d_head, T)，matmul → (B, T, T)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d_head)
    # mask 可广播到 (B, T, T)，约定 True=允许，False=禁止
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    # 沿 key 维归一化，A: (B, T, T)
    A = softmax_np(scores, axis=-1)
    # (B, T, T) @ (B, T, d_head) @ (d_head, D) → (B, T, D)
    return A @ v @ w_o
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D, d_head = 2, 4, 8, 4
x = torch.randn(B, T, D)
attn = SelfAttention(D, d_head)
# Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
w = [m.weight.detach().T.numpy()
     for m in (attn.w_q, attn.w_k, attn.w_v, attn.w_o)]
# 下三角 causal mask：(T, T) 广播到 (B, T, T)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
out_pt = attn(x, mask)
out_np = self_attention_np(x.numpy(), *w, mask=mask.numpy())
print("shape:", out_pt.shape)
print("后端差异:", abs(out_pt.detach().numpy() - out_np).max())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        pass

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        pass
```

##### Step 1（i）公式：投影得到 Q/K/V

$$
Q=XW_q,\quad K=XW_k,\quad V=XW_v
$$

##### Step 1（ii）代码

```python
# x: (B, T, D) → q/k/v: (B, T, d_head)
q = self.w_q(x)
k = self.w_k(x)
v = self.w_v(x)
```

##### Step 1（iii）解释

同一个 $x$ 经三组独立权重得到三种角色：Q 负责查询，K 负责被匹配，V 负责被聚合。三者都来自 $x$，所以称为自注意力。

##### Step 2（i）公式：缩放点积

$$
S=\frac{QK^\top}{\sqrt{d_h}}\in\mathbb{R}^{B\times T\times T}
$$

##### Step 2（ii）代码

```python
# k^T: (B, d_head, T)，matmul → scores: (B, T, T)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
```

##### Step 2（iii）解释

`transpose(-2, -1)` 只交换最后两维，所以 Batch 维保持对齐。$S_{ij}$ 是第 $i$ 个 query 与第 $j$ 个 key 的点积相似度。

##### Step 3（i）公式：施加掩码（Mask）

$$
S_{ij}\leftarrow
\begin{cases}
S_{ij}, & M_{ij}=\mathrm{True}\\
-\infty, & M_{ij}=\mathrm{False}
\end{cases}
$$

##### Step 3（ii）代码

```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
```

##### Step 3（iii）解释

`masked_fill` 填充条件为 True 的位置，而此处约定 True 表示允许，所以要先取反。填入 $-\infty$ 后，这些位置经 Softmax 的权重严格为 0。

##### Step 4（i）公式：Softmax 得到权重

$$
A_{ij}=\frac{\exp(S_{ij})}{\sum_{l}\exp(S_{il})},\quad \sum_{j}A_{ij}=1
$$

##### Step 4（ii）代码

```python
A = scores.softmax(dim=-1)
```

##### Step 4（iii）解释

归一化维度必须是 `dim=-1`，也就是 key 维。这样才能保证每个 query 分配到所有 key 上的权重和为 1。

##### Step 5（i）公式：加权 V 并映射回 D

$$
O=AV,\quad \hat{X}=OW_o\in\mathbb{R}^{B\times T\times D}
$$

##### Step 5（ii）代码

```python
# (B, T, T) @ (B, T, d_head) → (B, T, d_head)
out = A @ v
# (B, T, d_head) → (B, T, D)
return self.w_o(out)
```

##### Step 5（iii）解释

$O_i=\sum_j A_{ij}V_j$，也就是第 $i$ 个 token 的输出是所有 V 的加权和。若题目要求输出 `[B,T,d_head]`，去掉最后的 `w_o` 即可。

##### 5️⃣ 复杂度

设 $d_h$ 与 $D$ 同量级，主项来自计算 $QK^\top$：

$$
\text{时间}=O(BT^2d_h+BTDd_h),\quad \text{激活显存}=O(BT^2+BTd_h)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么除以 $\sqrt{d_h}$，每步 shape 是什么（原始追问）

设 $q_l,k_l$ 独立、均值为 0、方差为 1，则点积方差随维度线性增长：

$$
\mathrm{Var}(q^\top k)=\sum_{l=1}^{d_h}\mathrm{Var}(q_lk_l)=d_h
$$

除以 $\sqrt{d_h}$ 后方差回到 1。不做缩放时，分数的绝对值随 $d_h$ 变大，Softmax 接近 one-hot，梯度接近 0。

```python
for d in (4, 64, 1024):
    q, k = torch.randn(10000, d), torch.randn(10000, d)
    dot = (q * k).sum(-1)
    # 未缩放方差约等于 d，缩放后约等于 1
    print(d, dot.var().item(), (dot / d**0.5).var().item())
```

shape 流为：`x:(B,T,D)` → `q,k,v:(B,T,d_head)` → `scores,A:(B,T,T)` → `A@v:(B,T,d_head)` → `w_o:(B,T,D)`。

##### Follow-up 2：支持任意可广播的 Attention Mask

主实现约定 True 为允许，只要 mask 能广播到 $(B,T,T)$ 就可以直接传入。Padding mask 的形状是 $(B,T)$，需要插入一维，让它作用在 key 维上。

```python
# 第 2 个样本的最后一个 token 是 padding
pad = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]).bool()
# (B, T) → (B, 1, T)，广播后每个 query 都看不到 pad key
pad_mask = pad.unsqueeze(1)
# (T, T) 与 (B, 1, T) 取交集 → (B, T, T)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
print(attn(x, causal & pad_mask).shape)
```

注意语义差异：`nn.MultiheadAttention` 的 bool mask 约定 True 为禁止，与这里相反。

##### Follow-up 3：T 很长时的时间与激活显存

$T$ 翻倍时，点积的计算量和权重矩阵都变为 4 倍。

$$
\text{乘法次数}\approx BT^2d_h,\quad \text{权重矩阵元素数}=BT^2
$$

```python
def attn_cost(B, T, d_head, bytes_per=4):
    # 乘法次数主项，以及 scores 占用的 GB
    macs = B * T * T * d_head
    return macs, B * T * T * bytes_per / 1024**3
print(attn_cost(1, 4096, 64))
print(attn_cost(1, 8192, 64))
```

#### Q002｜从零实现 Multi-Head Attention（MHA）

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 年月未公开
- **题目：** 现场写一个支持多头的 PyTorch MHA，不直接使用 `nn.MultiheadAttention`；线性变换、拆头、注意力、拼头、输出投影都要自己写。
- **输入输出 / 约束：** 输入 `x:[B,T,D]`，`H` 头且 `D%H==0`；输出 `[B,T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. **[原始追问]** 在 MHA 内加入遮蔽未来 token 的 Mask。
  2. `view/reshape/transpose` 的正确顺序？为什么不直接 `reshape(B,H,T,d)`？
  3. 处理 causal/padding mask 相加和全 masked 行，怎样避免 NaN？
  4. 如何在实现上验证和 PyTorch `nn.MultiheadAttention` 数值一致？
- **来源：** [Glassdoor：TikTok MLE Intern 单头/多头 Attention、Mask、KL](https://www.glassdoor.com/Interview/Implement-Single-head-and-Multi-head-attention-in-pytorch-A-follow-up-to-this-was-use-masking-to-mask-out-the-later-on-tok-QTN_8701206.htm)

##### 1️⃣ Dense Opening

多头注意力（Multi-Head Attention, MHA）把 $D$ 维表示切成 $H$ 个 $d=D/H$ 维子空间，各头独立做缩放点积后再融合，结构是“线性投影 → 拆头（split）→ 注意力 → 拼头（concat）→ 输出投影”，数学核心对象是逐头权重张量 $A\in\mathbb{R}^{B\times H\times T\times T}$。

##### 2️⃣ Global Equations

投影并拆头，第 $h$ 个头取投影结果的第 $h$ 段：

$$
Q^{(h)}=(XW_q)_{[\cdot,\,hd:(h+1)d]},\quad Q^{(h)},K^{(h)},V^{(h)}\in\mathbb{R}^{B\times T\times d}
$$

主目标，每个头独立做缩放点积后拼接并投影：

$$
\mathrm{MHA}(X)=\mathrm{Concat}\big(O^{(0)},\dots,O^{(H-1)}\big)W_o,\quad O^{(h)}=A^{(h)}V^{(h)}
$$

中间对象是逐头分数 $S^{(h)}$ 与权重 $A^{(h)}$，权重沿 key 维归一化：

$$
S^{(h)}=\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d}},\quad A^{(h)}_{ij}=\frac{\exp(S^{(h)}_{ij})}{\sum_{l=1}^{T}\exp(S^{(h)}_{il})}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # 要求 D 能被 H 整除
        assert d_model % n_heads == 0
        # 头数 H 与单头维度 d = D // H
        self.h, self.d = n_heads, d_model // n_heads
        # Query 投影：D → D = H * d，一次算完所有头
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Key 投影：D → D
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Value 投影：D → D
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影：融合拼接后的各头信息，D → D
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split(self, z: Tensor) -> Tensor:
        # 取出 Batch 和序列长度
        B, T, _ = z.shape
        # (B, T, D) → (B, T, H, d) → (B, H, T, d)
        return z.view(B, T, self.h, self.d).transpose(1, 2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # 拼头时恢复形状要用
        B, T, D = x.shape
        # x: (B, T, D) → q: (B, H, T, d)
        q = self.split(self.w_q(x))
        # x: (B, T, D) → k: (B, H, T, d)
        k = self.split(self.w_k(x))
        # x: (B, T, D) → v: (B, H, T, d)
        v = self.split(self.w_v(x))
        # k^T: (B, H, d, T)，matmul → scores: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        # mask 可广播到 (B, H, T, T)，约定 True=允许，False=禁止
        if mask is not None:
            # 取反后填 -inf，禁止位置的权重变为 0
            scores = scores.masked_fill(~mask, float("-inf"))
        # 沿 key 维归一化，A: (B, H, T, T)，每行和为 1
        A = scores.softmax(dim=-1)
        # (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
        out = A @ v
        # (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, D)
        # (B, T, D) → (B, T, D)
        return self.w_o(out)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢，比值不变；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    # 逐元素取指数，形状不变
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def split_np(z, h):
    # 取出 Batch、序列长度、总维度
    B, T, D = z.shape
    # (B, T, D) → (B, T, H, d) → (B, H, T, d)
    return z.reshape(B, T, h, D // h).transpose(0, 2, 1, 3)

def mha_np(x, w_q, w_k, w_v, w_o, h, mask=None):
    # 权重形状均为 (D, D)；单头维度 d = D // H
    d = x.shape[-1] // h
    # x: (B, T, D) → q: (B, H, T, d)
    q = split_np(x @ w_q, h)
    # x: (B, T, D) → k: (B, H, T, d)
    k = split_np(x @ w_k, h)
    # x: (B, T, D) → v: (B, H, T, d)
    v = split_np(x @ w_v, h)
    # swapaxes 得 k^T: (B, H, d, T)，matmul → (B, H, T, T)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)
    # mask 可广播到 (B, H, T, T)，约定 True=允许，False=禁止
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    # 沿 key 维归一化，A: (B, H, T, T)
    A = softmax_np(scores, axis=-1)
    # (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
    out = A @ v
    # (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
    out = out.transpose(0, 2, 1, 3).reshape(x.shape)
    # (B, T, D) @ (D, D) → (B, T, D)
    return out @ w_o
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D, H = 2, 5, 8, 2
x = torch.randn(B, T, D)
mha = MultiHeadAttention(D, H)
# Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
w = [m.weight.detach().T.numpy()
     for m in (mha.w_q, mha.w_k, mha.w_v, mha.w_o)]
# 下三角 causal mask：(T, T) 广播到 (B, H, T, T)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
out_pt = mha(x, mask)
out_np = mha_np(x.numpy(), *w, H, mask=mask.numpy())
print("shape:", out_pt.shape)
print("后端差异:", abs(out_pt.detach().numpy() - out_np).max())
# H=1 时 d=D，公式退化为单头，缩放因子变为 sqrt(D)
print("单头 d:", MultiHeadAttention(D, 1).d)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        pass

    def split(self, z: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        pass
```

##### Step 1（i）公式：确定头数与投影

$$
d=\frac{D}{H},\quad W_q,W_k,W_v,W_o\in\mathbb{R}^{D\times D}
$$

##### Step 1（ii）代码

```python
super().__init__()
assert d_model % n_heads == 0
self.h, self.d = n_heads, d_model // n_heads
self.w_q = nn.Linear(d_model, d_model, bias=False)
self.w_k = nn.Linear(d_model, d_model, bias=False)
self.w_v = nn.Linear(d_model, d_model, bias=False)
self.w_o = nn.Linear(d_model, d_model, bias=False)
```

##### Step 1（iii）解释

每个投影都输出 $D=Hd$ 维，所以一层线性就覆盖所有头，等价于 $H$ 个 $D\to d$ 的小投影。总参数量是 $4D^2$，与 $H$ 无关。

##### Step 2（i）公式：拆头

$$
\mathbb{R}^{B\times T\times D}\to\mathbb{R}^{B\times T\times H\times d}\to\mathbb{R}^{B\times H\times T\times d}
$$

##### Step 2（ii）代码

```python
B, T, _ = z.shape
# (B, T, D) → (B, T, H, d) → (B, H, T, d)
return z.view(B, T, self.h, self.d).transpose(1, 2)
```

##### Step 2（iii）解释

`view` 只切开每个 token 的最后一维，token 边界不变。`transpose(1, 2)` 把头维提到前面，后续 matmul 就把 $(B,H)$ 当作批维度，在 $(T,d)$ 上计算。

##### Step 3（i）公式：投影得到逐头 Q/K/V

$$
Q=\mathrm{split}(XW_q),\quad K=\mathrm{split}(XW_k),\quad V=\mathrm{split}(XW_v)
$$

##### Step 3（ii）代码

```python
B, T, D = x.shape
# x: (B, T, D) → q/k/v: (B, H, T, d)
q = self.split(self.w_q(x))
k = self.split(self.w_k(x))
v = self.split(self.w_v(x))
```

##### Step 3（iii）解释

自注意力的三种角色都来自同一个 $x$。投影在拆头之前做，每个头的输入都混合了 $D$ 维的全部信息。

##### Step 4（i）公式：逐头缩放点积

$$
S=\frac{QK^\top}{\sqrt{d}}\in\mathbb{R}^{B\times H\times T\times T}
$$

##### Step 4（ii）代码

```python
# k^T: (B, H, d, T)，matmul → scores: (B, H, T, T)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
```

##### Step 4（iii）解释

缩放因子用单头维度 $d$，因为每个头的点积只在 $d$ 维上求和。写成 $\sqrt{D}$ 是常见错误。

##### Step 5（i）公式：施加掩码（Mask）

$$
S_{hij}\leftarrow
\begin{cases}
S_{hij}, & M_{hij}=\mathrm{True}\\
-\infty, & M_{hij}=\mathrm{False}
\end{cases}
$$

##### Step 5（ii）代码

```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
```

##### Step 5（iii）解释

`masked_fill` 填充条件为 True 的位置，此处约定 True 表示允许，所以要先取反。mask 只需能广播到 $(B,H,T,T)$，所有头共用同一份。

##### Step 6（i）公式：Softmax 并加权 V

$$
A=\mathrm{softmax}(S),\quad O=AV\in\mathbb{R}^{B\times H\times T\times d}
$$

##### Step 6（ii）代码

```python
# 沿 key 维归一化，A: (B, H, T, T)
A = scores.softmax(dim=-1)
# (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
out = A @ v
```

##### Step 6（iii）解释

每个头有自己的权重矩阵，所以能关注不同的位置组合。$O^{(h)}_i=\sum_j A^{(h)}_{ij}V^{(h)}_j$。

##### Step 7（i）公式：拼头并输出投影

$$
\hat{X}=\mathrm{Concat}\big(O^{(0)},\dots,O^{(H-1)}\big)W_o\in\mathbb{R}^{B\times T\times D}
$$

##### Step 7（ii）代码

```python
# (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
out = out.transpose(1, 2).reshape(B, T, D)
# (B, T, D) → (B, T, D)
return self.w_o(out)
```

##### Step 7（iii）解释

拼头是拆头的逆操作，先把头维换回去再合并。`transpose` 之后内存不连续，所以这里用 `reshape`。$W_o$ 负责让各头的信息相互混合。

##### 5️⃣ 复杂度

由 $Hd=D$，主项来自投影和各头的 $QK^\top$：

$$
\text{时间}=O(BTD^2+BHT^2d)=O(BTD^2+BT^2D),\quad \text{激活显存}=O(BHT^2+BTD)
$$

参数量为 $4D^2$（无 bias）。时间主项不随 $H$ 变化，权重矩阵的显存随 $H$ 线性增长。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：在 MHA 内加入遮蔽未来 token 的 Mask（原始追问）

位置 $i$ 只能看 $j\le i$，下三角为 True：

$$
M_{ij}=\mathbb{1}[j\le i]
$$

主实现已经支持，形状 $(T,T)$ 会广播到 $(B,H,T,T)$。验证方法：改动最后一个 token，前面的输出应完全不变。

```python
# (T, T)：True 表示允许，对角线保留
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
y = x.clone()
# 只改最后一个 token：(B, T, D) 中 T-1 位置
y[:, -1] += 100.0
diff = (mha(x, causal) - mha(y, causal)).abs()
# 按 token 聚合：(B, T, D) → (T,)
per_t = diff.amax(dim=(0, 2))
print("逐位置最大变化:", per_t.tolist())
```

输出中只有最后一个位置非零。

##### Follow-up 2：`view/reshape/transpose` 的正确顺序，为什么不直接 `reshape(B,H,T,d)`

正确顺序是先 `view(B,T,H,d)`，再 `transpose(1,2)`。`reshape` 不交换元素，只按行优先顺序（row-major）重新划分内存。直接 `reshape(B,H,T,d)` 会跨 token 切分，第 0 个“头”实际拿到的是前 $T/H$ 个 token 的完整向量。

```python
# z: (1, 4, 6)，T=4，D=6，H=2，d=3
z = torch.arange(24.0).view(1, 4, 6)
# 正确：(1, 4, 6) → (1, 4, 2, 3) → (1, 2, 4, 3)
right = z.view(1, 4, 2, 3).transpose(1, 2)
# 错误：(1, 4, 6) → (1, 2, 4, 3)，形状相同但语义错误
wrong = z.reshape(1, 2, 4, 3)
# 第 0 头第 1 个 token 应为原 token 1 的前 3 维：6, 7, 8
print("right:", right[0, 0, 1].tolist())
print("wrong:", wrong[0, 0, 1].tolist())
# 拼头还原：(1, 2, 4, 3) → (1, 4, 2, 3) → (1, 4, 6)
print("还原成功:", torch.equal(right.transpose(1, 2).reshape(1, 4, 6), z))
try:
    # 新算出的 (B, H, T, d) 连续，transpose 后不连续，view 会报错
    torch.randn(1, 2, 4, 3).transpose(1, 2).view(1, 4, 6)
except RuntimeError:
    print("view 合并失败，需用 reshape 或 contiguous().view")
```

形状正确不代表语义正确，这个错误不会报错，只能靠上面这样的逐元素检查发现。

##### Follow-up 3：causal/padding mask 相加和全 masked 行，怎样不产生 NaN

组合规则分两种：bool mask（True=允许）用 `&`；additive mask（0 或 $-\infty$）用相加，$-\infty+(-\infty)=-\infty$ 不会出问题。不要把 $-\infty$ 与 0 相乘，$0\cdot(-\infty)$ 是 NaN。

全 masked 行的所有分数都是 $-\infty$，Softmax 为 NaN。典型场景是左 padding（left padding）：第 0 个 token 是 pad 时，causal 只允许看自己，而自己又被 padding 屏蔽。处理方法是先把这些行的分数置 0 算 Softmax，再把权重清零：

$$
A_{i\cdot}=\mathbf{0},\quad \text{若 }\sum_j M_{ij}=0
$$

```python
def safe_softmax(scores, allow):
    # allow: 可广播到 scores。dead: (..., T, 1)，True 表示全遮挡
    dead = ~allow.any(dim=-1, keepdim=True)
    # 禁止位置填 -inf
    s = scores.masked_fill(~allow, float("-inf"))
    # 全遮挡行先置 0 算 Softmax，再把权重清零
    return s.masked_fill(dead, 0.0).softmax(-1).masked_fill(dead, 0.0)

# 左 padding：第 0 个 token 是 pad。pad: (1, 3)，True=有效
pad = torch.tensor([[False, True, True]])
# (T, T) & (1, 1, T) → (1, 3, 3)
allow = torch.tril(torch.ones(3, 3, dtype=torch.bool)) & pad[:, None, :]
inf = float("-inf")
# additive 写法：(3, 3) 与 (1, 1, 3) 相加 → (1, 3, 3)
add = (torch.triu(torch.full((3, 3), inf), 1)
       + torch.zeros(1, 1, 3).masked_fill(~pad[:, None, :], inf))
print("两种写法一致:", torch.equal(add == 0, allow))
s = torch.randn(1, 3, 3, requires_grad=True)
bad = s.masked_fill(~allow, inf).softmax(-1)
print("直接 Softmax 第 0 行:", bad[0, 0].tolist())
A = safe_softmax(s, allow)
A.sum().backward()
print("safe:", A[0].tolist())
print("梯度含 NaN:", s.grad.isnan().any().item())
```

`safe_softmax` 让被遮挡的行输出全 0，梯度也保持有限。另一种常见做法是让每个 token 总能看到自己，也就是 `allow | torch.eye(T).bool()`，这样不会出现全遮挡行，代价是 pad 位置有非零权重，下游要忽略 pad 位置的输出。

##### Follow-up 4：如何验证和 `nn.MultiheadAttention` 数值一致

做法是把权重拷过去，再传入语义相反的 mask，比较输出。官方类把三个输入投影存在一个 $(3D,D)$ 的 `in_proj_weight` 中，所以按行拼接；它的 bool mask 约定 True=禁止，所以要取反；还要设 `batch_first=True`，并用 `bias=False` 与主实现对齐。

```python
ref = nn.MultiheadAttention(D, H, bias=False, batch_first=True)
with torch.no_grad():
    # 3 个 (D, D) 按行拼接 → (3D, D)
    ref.in_proj_weight.copy_(torch.cat(
        [mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0))
    ref.out_proj.weight.copy_(mha.w_o.weight)
# pad: (B, T)，第 2 个样本末尾 2 个 token 是 padding，True=有效
pad = torch.ones(B, T, dtype=torch.bool)
pad[1, -2:] = False
# 自写版：(T, T) & (B, 1, 1, T) → (B, 1, T, T)
ours = mha(x, causal & pad[:, None, None, :])
# 官方版：两个 mask 都是 True=禁止，所以都取反
theirs, _ = ref(x, x, x, attn_mask=~causal, key_padding_mask=~pad,
                need_weights=False)
print("与官方差异:", (ours - theirs).abs().max().item())
```

#### Q003｜MHA：正确构造 Causal Mask

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 年月未公开
- **题目：** 在上面的 MHA 中屏蔽未来 token，保证位置 `i` 不关注 `j>i`。
- **输入输出 / 约束：** 输入注意力分数 `[B,H,T,T]`；输出相同形状的 masked logits。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 上三角/下三角哪些位置应为不可见？你用的是 bool mask 还是 additive mask？
  2. 与 padding mask 同时存在时如何组合？不同库的 bool 语义会不会相反？
  3. 如果某行所有位置均被遮挡，Softmax 如何不产生 NaN？
- **来源：** [Glassdoor：TikTok MLE Intern 单头/多头 Attention、Mask、KL](https://www.glassdoor.com/Interview/Implement-Single-head-and-Multi-head-attention-in-pytorch-A-follow-up-to-this-was-use-masking-to-mask-out-the-later-on-tok-QTN_8701206.htm)

##### 1️⃣ Dense Opening

因果掩码（Causal Mask）保证位置 $i$ 只关注 $j\le i$，结构是“取序列长度 → 构造下三角 bool 矩阵 → 把上三角分数填 $-\infty$”，数学核心对象是允许矩阵 $M\in\{0,1\}^{T\times T}$。按题面，主实现的输入是注意力分数 `[B,H,T,T]`，输出相同形状的 masked logits，Softmax 留给调用方。

##### 2️⃣ Global Equations

允许矩阵是含对角线的下三角：

$$
M_{ij}=\mathbb{1}[j\le i]
$$

主目标是逐位置改写 logits，$M$ 广播到所有 Batch 和所有头：

$$
\tilde{S}_{bhij}=
\begin{cases}
S_{bhij}, & j\le i\\
-\infty, & j>i
\end{cases}
$$

后续 Softmax 后，未来位置的权重严格为 0，第 $i$ 行只在前 $i+1$ 个位置上归一化：

$$
A_{bhij}=\frac{\exp(\tilde{S}_{bhij})}{\sum_{l\le i}\exp(S_{bhil})}\,\mathbb{1}[j\le i]
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def apply_causal_mask(scores: Tensor) -> Tensor:
    # scores: (B, H, T, T)，取最后一维得序列长度 T
    T = scores.size(-1)
    # 全 True 的 (T, T)，放到 scores 所在设备
    full = torch.ones(T, T, dtype=torch.bool, device=scores.device)
    # 保留下三角含对角线：(T, T)，True=允许，False=未来位置
    allow = torch.tril(full)
    # (T, T) 广播到 (B, H, T, T)，未来位置填 -inf，形状不变
    return scores.masked_fill(~allow, float("-inf"))
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def apply_causal_mask_np(scores):
    # scores: (B, H, T, T)，取最后一维得序列长度 T
    T = scores.shape[-1]
    # 全 True 的 (T, T)
    full = np.ones((T, T), dtype=bool)
    # 保留下三角含对角线：(T, T)，True=允许，False=未来位置
    allow = np.tril(full)
    # (T, T) 广播到 (B, H, T, T)，未来位置填 -inf，形状不变
    return np.where(allow, scores, -np.inf)
```

##### 最小可运行测试（Demo）

```python
# 全 1 分数：(1, 1, 3, 3)，手算第 i 行权重为 1/(i+1)
s = torch.ones(1, 1, 3, 3)
print(apply_causal_mask(s).softmax(-1)[0, 0])
torch.manual_seed(0)
# 随机分数 (2, 4, 5, 5)，对比两个后端的 logits，inf 位置相等
s = torch.randn(2, 4, 5, 5)
m_pt = apply_causal_mask(s)
m_np = apply_causal_mask_np(s.numpy())
print("后端一致:", np.array_equal(m_pt.numpy(), m_np))
print("shape:", m_pt.shape)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def apply_causal_mask(scores: Tensor) -> Tensor:
    pass
```

##### Step 1（i）公式：确定掩码大小

$$
S\in\mathbb{R}^{B\times H\times T\times T}\ \Rightarrow\ M\in\{0,1\}^{T\times T}
$$

##### Step 1（ii）代码

```python
# scores: (B, H, T, T)，取最后一维得序列长度 T
T = scores.size(-1)
```

##### Step 1（iii）解释

第 2 维（从 0 计）是 query 位置 $i$，第 3 维是 key 位置 $j$。掩码只与这两维有关，所有 Batch 和头共用同一份。

##### Step 2（i）公式：构造下三角允许矩阵

$$
M_{ij}=\mathbb{1}[j\le i]
$$

##### Step 2（ii）代码

```python
full = torch.ones(T, T, dtype=torch.bool, device=scores.device)
# 保留下三角含对角线：(T, T)，True=允许，False=未来位置
allow = torch.tril(full)
```

##### Step 2（iii）解释

`torch.tril` 默认 `diagonal=0`，对角线保留，位置 $i$ 能看到自己。如果写成 `diagonal=-1`，第 0 行会全部被遮挡，Softmax 输出 NaN。

##### Step 3（i）公式：把未来位置的 logits 填为负无穷

$$
\tilde{S}_{bhij}=
\begin{cases}
S_{bhij}, & M_{ij}=1\\
-\infty, & M_{ij}=0
\end{cases}
$$

##### Step 3（ii）代码

```python
# (T, T) 广播到 (B, H, T, T)，未来位置填 -inf，形状不变
return scores.masked_fill(~allow, float("-inf"))
```

##### Step 3（iii）解释

`masked_fill` 填充条件为 True 的位置，所以要对 `allow` 取反。它返回新张量，不改动原始 `scores`。这一步要按位置选取，不能写成分数乘掩码再加 $-\infty$，因为 $0\cdot(-\infty)$ 是 NaN。

##### 5️⃣ 复杂度

构造掩码和填充分数都是逐元素操作：

$$
\text{时间}=O(T^2+BHT^2)=O(BHT^2),\quad \text{额外显存}=O(T^2+BHT^2)
$$

掩码只占 $T^2$ 字节。工程上常用 `register_buffer` 预存最大长度的下三角，再切片，这样每次前向不用重建。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：上三角/下三角哪些位置不可见，用 bool mask 还是 additive mask

严格上三角（$j>i$，不含对角线）不可见。两种写法等价：

$$
\text{bool：}\ M_{ij}=\mathbb{1}[j\le i],\qquad \text{additive：}\ E_{ij}=\begin{cases}0,& j\le i\\-\infty,& j>i\end{cases},\quad \tilde{S}=S+E
$$

主实现用 bool mask，它占内存少，也能直接做 `&`、`|` 组合。additive mask 可以直接与 logits 相加。

```python
T = 4
# 严格上三角为 -inf，其余为 0：(T, T)
add = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
s = torch.randn(2, 4, T, T)
# 两种写法的 masked logits 完全一致
print("等价:", torch.equal(s + add, apply_causal_mask(s)))
print(add)
```

##### Follow-up 2：与 padding mask 如何组合，不同库的 bool 语义是否相反

padding mask 形状是 $(B,T)$，作用在 key 维，需要扩成 $(B,1,1,T)$。统一到“True=允许”后取交集：

$$
M^{\text{final}}_{bhij}=\mathbb{1}[j\le i]\wedge P_{bj}
$$

语义确实会相反，接入前要逐个确认：

- 本文主实现：True=允许。
- `F.scaled_dot_product_attention` 的 `attn_mask`：True=参与注意力。
- `nn.MultiheadAttention` 的 `attn_mask` 和 `key_padding_mask`：True=禁止。

```python
B, H, T, D = 2, 1, 3, 4
# pad: (B, T)，True=有效，第 2 个样本末尾是 padding
pad = torch.tensor([[True, True, True], [True, True, False]])
allow = torch.tril(torch.ones(T, T, dtype=torch.bool))
# (T, T) & (B, 1, 1, T) → (B, 1, T, T)
final = allow & pad[:, None, None, :]
print("样本 2 的 mask:", final[1, 0].int().tolist())
x = torch.randn(B, T, D)
# 官方 SDPA 用 True=参与：(B, 1, T, D) 配 (B, 1, T, T)
q = x[:, None]
sdpa = F.scaled_dot_product_attention(q, q, q, attn_mask=final)
# 手写：分数 (B, 1, T, T)，填 -inf 后 Softmax，再乘 V
s = q @ q.transpose(-2, -1) / D**0.5
ours = s.masked_fill(~final, float("-inf")).softmax(-1) @ q
print("与 SDPA 差异:", (sdpa - ours).abs().max().item())
# nn.MultiheadAttention 用 True=禁止，传 ~allow 才得到因果权重
mha = torch.nn.MultiheadAttention(D, 1, batch_first=True)
_, w = mha(x, x, x, attn_mask=~allow)
print("MHA 第 0 行权重:", w[0, 0].tolist())
```

##### Follow-up 3：某行所有位置均被遮挡时，Softmax 如何不产生 NaN

只有 causal mask 时不会全遮挡，因为对角线总是保留。与 padding 组合后可能出现：左 padding 时第 0 行只能看第 0 个 token，而它是 pad。此时该行全是 $-\infty$，Softmax 为 NaN。处理方法是检测这些行，先把分数置 0 算 Softmax，再把权重清零：

$$
A_{i\cdot}=\mathbf{0},\quad \text{若 }\sum_j M_{ij}=0
$$

```python
def safe_attn_softmax(scores, allow):
    # allow 可广播到 scores。dead: (..., T, 1)，True 表示该行全遮挡
    dead = ~allow.any(dim=-1, keepdim=True)
    # 禁止位置填 -inf，形状不变
    s = scores.masked_fill(~allow, float("-inf"))
    # 全遮挡行先置 0 再 Softmax，最后清零权重
    return s.masked_fill(dead, 0.0).softmax(-1).masked_fill(dead, 0.0)

# 左 padding：(1, 3)，第 0 个 token 是 pad
pad = torch.tensor([[False, True, True]])
# (3, 3) & (1, 1, 1, 3) → (1, 1, 3, 3)
allow = torch.tril(torch.ones(3, 3).bool()) & pad[:, None, None, :]
s = torch.randn(1, 1, 3, 3, requires_grad=True)
print("直接算:", s.masked_fill(~allow, -torch.inf).softmax(-1)[0, 0, 0])
A = safe_attn_softmax(s, allow)
A.sum().backward()
print("safe:", A[0, 0])
print("梯度含 NaN:", s.grad.isnan().any().item())
# 另一种做法：允许每个位置看到自己，(1, 1, 3, 3) | (3, 3)
allow2 = allow | torch.eye(3).bool()
print("无全遮挡行:", allow2.any(-1).all().item())
```

第一种做法让 pad 查询的输出为 0；第二种做法结构更简单，但 pad 位置会有非零权重，下游要忽略该位置的输出。

#### Q004｜GQA：Grouped Query Attention

- **优先级：** P0 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 实现 `Hq` 个 Query 头共享 `Hkv` 个 KV 头的 GQA，要求 KV Heads 对应关系正确，并展示与 MHA 的参数/缓存差异。
- **输入输出 / 约束：** `q:[B,Hq,Tq,d]`、`k/v:[B,Hkv,Tk,d]`，且 `Hq%Hkv==0`；返回 `[B,Hq,Tq,d]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 怎么把 query 的第 h 个头映射到 KV 的哪一个头？
  2. `repeat_interleave` 与不显式复制 KV 的写法，内存差异是什么？
  3. 当 `Hkv=Hq`、`Hkv=1` 时分别退化成什么？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

##### 1️⃣ Dense Opening

分组查询注意力（Grouped Query Attention, GQA）让 $H_q$ 个 Query 头分成 $H_{kv}$ 组，每组共享一个 KV 头，结构是“头映射 → 对齐 KV → 缩放点积 → Softmax → 加权 V”，数学核心对象是头映射 $g(h)=\lfloor h/r\rfloor$，其中 $r=H_q/H_{kv}$。它的注意力计算量基本不变，KV 投影参数和 KV 缓存按 $H_{kv}/H_q$ 缩小。

##### 2️⃣ Global Equations

头映射，第 $h$ 个 Query 头读取第 $g(h)$ 个 KV 头：

$$
r=\frac{H_q}{H_{kv}},\quad g(h)=\left\lfloor\frac{h}{r}\right\rfloor,\quad h=0,\dots,H_q-1
$$

主目标，每个 Query 头在对应的 KV 头上做缩放点积：

$$
O^{(h)}=\mathrm{softmax}\!\left(\frac{Q^{(h)}K^{(g(h))\top}}{\sqrt{d}}\right)V^{(g(h))}\in\mathbb{R}^{B\times T_q\times d}
$$

与 MHA 的差异体现在参数量和缓存元素数（无 bias，$H_qd=D$）：

$$
P_{\text{GQA}}=2D^2+2DH_{kv}d,\quad C_{\text{GQA}}=2BH_{kv}T_kd
$$

$H_{kv}=H_q$ 时就是 MHA：$P=4D^2$，$C=2BH_qT_kd$。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

def gqa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    # q: (B, Hq, Tq, d)；k/v: (B, Hkv, Tk, d)；要求 Hq % Hkv == 0
    assert q.size(1) % k.size(1) == 0
    # 每个 KV 头被 r 个 Query 头共享
    r = q.size(1) // k.size(1)
    # (B, Hkv, Tk, d) → (B, Hq, Tk, d)，第 h 头对应 KV 头 h // r
    k = k.repeat_interleave(r, dim=1)
    # (B, Hkv, Tk, d) → (B, Hq, Tk, d)，与 k 用同一映射
    v = v.repeat_interleave(r, dim=1)
    # k^T: (B, Hq, d, Tk)，matmul → scores: (B, Hq, Tq, Tk)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    # mask 可广播到 (B, Hq, Tq, Tk)，约定 True=允许，False=禁止
    if mask is not None:
        # 取反后填 -inf，禁止位置的权重变为 0
        scores = scores.masked_fill(~mask, float("-inf"))
    # 沿 key 维归一化，A: (B, Hq, Tq, Tk)
    A = scores.softmax(dim=-1)
    # (B, Hq, Tq, Tk) @ (B, Hq, Tk, d) → (B, Hq, Tq, d)
    return A @ v

class GQASelfAttention(nn.Module):
    def __init__(self, D: int, Hq: int, Hkv: int):
        super().__init__()
        # 头数与单头维度 d = D // Hq
        self.Hq, self.Hkv, self.d = Hq, Hkv, D // Hq
        # Query 投影：D → Hq * d = D
        self.w_q = nn.Linear(D, Hq * self.d, bias=False)
        # Key 投影：D → Hkv * d，比 MHA 窄
        self.w_k = nn.Linear(D, Hkv * self.d, bias=False)
        # Value 投影：D → Hkv * d，比 MHA 窄
        self.w_v = nn.Linear(D, Hkv * self.d, bias=False)
        # 输出投影：D → D
        self.w_o = nn.Linear(D, D, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None):
        # 取出 Batch 和序列长度
        B, T, _ = x.shape
        # (B, T, Hq * d) → (B, T, Hq, d) → (B, Hq, T, d)
        q = self.w_q(x).view(B, T, self.Hq, self.d).transpose(1, 2)
        # (B, T, Hkv * d) → (B, T, Hkv, d) → (B, Hkv, T, d)
        k = self.w_k(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
        # (B, T, Hkv * d) → (B, T, Hkv, d) → (B, Hkv, T, d)
        v = self.w_v(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
        # (B, Hq, T, d) → (B, T, Hq, d) → 拼头 (B, T, D)
        out = gqa(q, k, v, mask).transpose(1, 2).reshape(B, T, -1)
        # 输出 (B, T, D)；缓存只需存未复制的 k/v: (B, Hkv, T, d)
        return self.w_o(out), (k, v)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢，比值不变；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    # 逐元素取指数，形状不变
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def gqa_np(q, k, v, mask=None):
    # 每个 KV 头被 r 个 Query 头共享
    r = q.shape[1] // k.shape[1]
    # (B, Hkv, Tk, d) → (B, Hq, Tk, d)，第 h 头对应 KV 头 h // r
    k = np.repeat(k, r, axis=1)
    # (B, Hkv, Tk, d) → (B, Hq, Tk, d)，与 k 用同一映射
    v = np.repeat(v, r, axis=1)
    # swapaxes 得 k^T，matmul → scores: (B, Hq, Tq, Tk)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(q.shape[-1])
    # mask 可广播到 (B, Hq, Tq, Tk)，约定 True=允许，False=禁止
    if mask is not None:
        # 禁止位置填 -inf，权重变为 0
        scores = np.where(mask, scores, -np.inf)
    # 沿 key 维归一化，A: (B, Hq, Tq, Tk)
    A = softmax_np(scores, axis=-1)
    # (B, Hq, Tq, Tk) @ (B, Hq, Tk, d) → (B, Hq, Tq, d)
    return A @ v
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# Hq=8, Hkv=2, d=8；Tq=3 与 Tk=5 可以不同
q = torch.randn(2, 8, 3, 8)
k, v = torch.randn(2, 2, 5, 8), torch.randn(2, 2, 5, 8)
out_pt = gqa(q, k, v)
out_np = gqa_np(q.numpy(), k.numpy(), v.numpy())
print("shape:", out_pt.shape)
print("后端差异:", abs(out_pt.numpy() - out_np).max())
x = torch.randn(1, 100, 64)
# 同样的 D=64, Hq=8，对比 GQA(Hkv=2) 与 MHA(Hkv=8)
for Hkv in (2, 8):
    m = GQASelfAttention(64, 8, Hkv)
    _, cache = m(x)
    # 参数量与缓存元素数，缓存是 k 和 v 两个张量
    print(Hkv, sum(p.numel() for p in m.parameters()),
          sum(c.numel() for c in cache))
```

输出显示：参数量由 16384 降到 10240（$2\cdot64^2+2\cdot64\cdot2\cdot8$），缓存元素数由 12800 降到 3200，正好是 $H_{kv}/H_q=1/4$。

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
def gqa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    pass

class GQASelfAttention(nn.Module):
    def __init__(self, D: int, Hq: int, Hkv: int):
        pass

    def forward(self, x: Tensor, mask: Tensor = None):
        pass
```

##### Step 1（i）公式：确定分组大小

$$
r=\frac{H_q}{H_{kv}}\in\mathbb{N}^{+}
$$

##### Step 1（ii）代码

```python
assert q.size(1) % k.size(1) == 0
# 每个 KV 头被 r 个 Query 头共享
r = q.size(1) // k.size(1)
```

##### Step 1（iii）解释

题面要求 $H_q$ 能被 $H_{kv}$ 整除，所以每组恰好有 $r$ 个 Query 头。

##### Step 2（i）公式：按映射对齐 KV 头

$$
\tilde{K}^{(h)}=K^{(\lfloor h/r\rfloor)},\quad \tilde{V}^{(h)}=V^{(\lfloor h/r\rfloor)}
$$

##### Step 2（ii）代码

```python
# (B, Hkv, Tk, d) → (B, Hq, Tk, d)，第 h 头对应 KV 头 h // r
k = k.repeat_interleave(r, dim=1)
v = v.repeat_interleave(r, dim=1)
```

##### Step 2（iii）解释

`repeat_interleave` 把每个 KV 头连续复制 $r$ 次，顺序是 $0,0,1,1,\dots$，正好实现 $\lfloor h/r\rfloor$。$K$ 和 $V$ 必须用同一映射。

##### Step 3（i）公式：逐头缩放点积

$$
S=\frac{Q\tilde{K}^\top}{\sqrt{d}}\in\mathbb{R}^{B\times H_q\times T_q\times T_k}
$$

##### Step 3（ii）代码

```python
# k^T: (B, Hq, d, Tk)，matmul → scores: (B, Hq, Tq, Tk)
scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
```

##### Step 3（iii）解释

对齐后的计算与 MHA 完全相同。分数有 $H_q$ 个头，所以注意力部分的计算量和权重显存都与 MHA 同量级。

##### Step 4（i）公式：施加掩码并归一化

$$
A=\mathrm{softmax}\big(S\ \text{（禁止位置置为}\ -\infty\text{）}\big)
$$

##### Step 4（ii）代码

```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
A = scores.softmax(dim=-1)
```

##### Step 4（iii）解释

约定 True=允许，所以填充前取反。归一化沿 key 维，也就是 `dim=-1`。

##### Step 5（i）公式：加权 V

$$
O=A\tilde{V}\in\mathbb{R}^{B\times H_q\times T_q\times d}
$$

##### Step 5（ii）代码

```python
# (B, Hq, Tq, Tk) @ (B, Hq, Tk, d) → (B, Hq, Tq, d)
return A @ v
```

##### Step 5（iii）解释

返回形状与题面的 `[B,Hq,Tq,d]` 一致。`gqa` 到这里结束，下面两步用完整模块展示参数和缓存差异。

##### Step 6（i）公式：KV 投影变窄

$$
W_q\in\mathbb{R}^{D\times H_qd},\quad W_k,W_v\in\mathbb{R}^{D\times H_{kv}d},\quad W_o\in\mathbb{R}^{D\times D}
$$

##### Step 6（ii）代码

```python
super().__init__()
self.Hq, self.Hkv, self.d = Hq, Hkv, D // Hq
self.w_q = nn.Linear(D, Hq * self.d, bias=False)
self.w_k = nn.Linear(D, Hkv * self.d, bias=False)
self.w_v = nn.Linear(D, Hkv * self.d, bias=False)
self.w_o = nn.Linear(D, D, bias=False)
```

##### Step 6（iii）解释

单头维度由 $H_q$ 决定，$d=D/H_q$。参数节省全部来自 $W_k$ 和 $W_v$，节省量是 $2Dd(H_q-H_{kv})$。

##### Step 7（i）公式：拆头、注意力、拼头、返回缓存

$$
\hat{X}=\mathrm{Concat}\big(O^{(0)},\dots,O^{(H_q-1)}\big)W_o,\quad \text{cache}=(K,V)
$$

##### Step 7（ii）代码

```python
B, T, _ = x.shape
# (B, T, Hq * d) → (B, T, Hq, d) → (B, Hq, T, d)
q = self.w_q(x).view(B, T, self.Hq, self.d).transpose(1, 2)
# (B, T, Hkv * d) → (B, T, Hkv, d) → (B, Hkv, T, d)
k = self.w_k(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
v = self.w_v(x).view(B, T, self.Hkv, self.d).transpose(1, 2)
# (B, Hq, T, d) → (B, T, Hq, d) → 拼头 (B, T, D)
out = gqa(q, k, v, mask).transpose(1, 2).reshape(B, T, -1)
# 输出 (B, T, D)；缓存只需存未复制的 k/v: (B, Hkv, T, d)
return self.w_o(out), (k, v)
```

##### Step 7（iii）解释

`gqa` 内部的复制不影响外层的 `k`、`v`，所以缓存存的是 $H_{kv}$ 个头。增量解码时只需缓存这两个张量，这就是 GQA 省缓存的来源。

##### 5️⃣ 复杂度

主项来自投影和 $H_q$ 个头的点积：

$$
\text{时间}=O(BH_qT_qT_kd+BTD^2)
$$

$$
\text{激活显存}=O(BH_qT_qT_k+BH_qT_kd),\quad \text{KV 缓存}=O(BH_{kv}T_kd)
$$

激活显存第二项来自 `repeat_interleave` 的显式复制，Follow-up 2 的广播写法可以去掉这部分。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：Query 的第 h 个头映射到 KV 的哪一个头

映射为 $g(h)=\lfloor h/r\rfloor$，每组是连续的 $r$ 个 Query 头。验证方法是逐头单独计算，再与批量结果对比。注意 `repeat` 产生的是 $h\bmod H_{kv}$，与 `repeat_interleave` 不同，这会让 Query 头配错组，加载已有权重时结果就错了。

```python
Hq, Hkv = 8, 2
r = Hq // Hkv
# (Hq,)：第 h 个 Query 头对应的 KV 头编号
print("映射:", (torch.arange(Hq) // r).tolist())
out = gqa(q, k, v)
for h in range(Hq):
    # 单头计算：q[:, [h]]: (B, 1, Tq, d)；k/v[:, [h//r]]: (B, 1, Tk, d)
    one = gqa(q[:, [h]], k[:, [h // r]], v[:, [h // r]])
    assert (one[:, 0] - out[:, h]).abs().max() < 1e-5
print("逐头映射正确")
# 错误写法：repeat 的顺序是 0,1,0,1,...，形状相同但配错组
k2, v2 = k.repeat(1, r, 1, 1), v.repeat(1, r, 1, 1)
print("repeat 写法差异:", (gqa(q, k2, v2) - out).abs().max().item())
```

##### Follow-up 2：`repeat_interleave` 与不显式复制 KV 的写法，内存差异是什么

`repeat_interleave` 每次前向临时分配 $r$ 倍的 KV，额外元素数为：

$$
2B(H_q-H_{kv})T_kd
$$

不复制的写法是把 Query 头按组 reshape，靠广播共享 KV。此时 $K$ 只多一个长度为 1 的维度，不额外分配内存。注意 `expand` 后再 `reshape` 仍然会复制。

```python
def gqa_no_copy(q, k, v):
    B, Hq, Tq, d = q.shape
    Hkv = k.size(1)
    # (B, Hq, Tq, d) → (B, Hkv, r, Tq, d)，连续 r 个头归为一组
    q5 = q.view(B, Hkv, Hq // Hkv, Tq, d)
    # (B, Hkv, Tk, d) → (B, Hkv, 1, Tk, d)，广播到每组的 r 个头
    k5, v5 = k.unsqueeze(2), v.unsqueeze(2)
    # scores: (B, Hkv, r, Tq, Tk)
    s = q5 @ k5.transpose(-2, -1) / math.sqrt(d)
    # (B, Hkv, r, Tq, Tk) @ (B, Hkv, 1, Tk, d) → (B, Hkv, r, Tq, d)
    # 再展平分组维：(B, Hkv, r, Tq, d) → (B, Hq, Tq, d)
    return (s.softmax(-1) @ v5).view(B, Hq, Tq, d)

print("广播写法差异:", (gqa_no_copy(q, k, v) - out).abs().max().item())
# 大序列下的额外内存：B=1, Hq=8, Hkv=2, Tk=4096, d=64，float32
big = torch.zeros(1, 2, 4096, 64)
copy = big.repeat_interleave(4, dim=1)
extra = 2 * (copy.numel() - big.numel()) * 4 / 1024**2
print("repeat_interleave 额外 MB:", extra)
# (B, Hkv, 1, Tk, d) → (B, Hkv, r, Tk, d)，只改视图
e = big.unsqueeze(2).expand(-1, -1, 4, -1, -1)
print("expand 共享内存:", e.data_ptr() == big.data_ptr())
# (B, Hkv, r, Tk, d) → (B, Hq, Tk, d)，这一步发生复制
print("再 reshape 共享内存:",
      e.reshape(1, 8, 4096, 64).data_ptr() == big.data_ptr())
```

差异只在临时显存，KV 缓存本身都是 $H_{kv}$ 个头，注意力权重也都是 $(B,H_q,T_q,T_k)$。

##### Follow-up 3：`Hkv=Hq`、`Hkv=1` 时分别退化成什么

- $H_{kv}=H_q$：$r=1$，$g(h)=h$，退化为标准 MHA。
- $H_{kv}=1$：$r=H_q$，$g(h)=0$，所有 Query 头共享一个 KV 头，退化为多查询注意力（Multi-Query Attention, MQA）。

```python
import torch.nn.functional as F
# Hkv=Hq：官方 SDPA 逐头对齐，结果应与 gqa 相同
k8, v8 = torch.randn(2, 8, 5, 8), torch.randn(2, 8, 5, 8)
sdpa = F.scaled_dot_product_attention(q, k8, v8)
print("Hkv=Hq 差异:", (gqa(q, k8, v8) - sdpa).abs().max().item())
# Hkv=1：SDPA 把 (B, 1, Tk, d) 广播到所有 Query 头
k1, v1 = torch.randn(2, 1, 5, 8), torch.randn(2, 1, 5, 8)
sdpa = F.scaled_dot_product_attention(q, k1, v1)
print("Hkv=1 差异:", (gqa(q, k1, v1) - sdpa).abs().max().item())
for Hkv in (8, 2, 1):
    m = GQASelfAttention(64, 8, Hkv)
    # 参数量随 Hkv 线性下降，下限是 Hkv=1 的 MQA
    print(Hkv, sum(p.numel() for p in m.parameters()))
```

#### Q005｜Cross-Attention（Q 与 KV 来自不同序列）

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-08（发布日）
- **题目：** 手写 Cross-Attention，Q 由目标序列产生，K/V 由上下文序列产生；允许不同长度。
- **输入输出 / 约束：** `query:[B,Tq,D]`、`context:[B,Tk,D]`，输出 `[B,Tq,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 score 是 `[B,H,Tq,Tk]` 而不是 `[B,H,Tq,Tq]`？
  2. Encoder-Decoder Attention 中如何组合 decoder causal mask 与 encoder padding mask？
  3. 如果 `Tq=1` 做增量解码，接口和缓存该如何修改？
- **来源：** [牛客：字节大模型算法一二三面 RMSNorm/SwiGLU/Cross-Attention](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)

##### 1️⃣ Dense Opening

交叉注意力（Cross-Attention）让目标序列的每个 token 从上下文序列中聚合信息，结构是“Q 由 query 投影、K/V 由 context 投影 → 拆头 → 缩放点积 → Softmax → 加权 V → 拼头投影”，数学核心对象是矩形权重张量 $A\in\mathbb{R}^{B\times H\times T_q\times T_k}$。输出长度由 $T_q$ 决定，$T_k$ 可以不同。

##### 2️⃣ Global Equations

Q 与 K/V 来自不同序列：

$$
Q=X_qW_q\in\mathbb{R}^{B\times T_q\times D},\quad K=X_cW_k,\ V=X_cW_v\in\mathbb{R}^{B\times T_k\times D}
$$

主目标，拆成 $H$ 个 $d=D/H$ 维的头后，逐头做缩放点积并拼接：

$$
\mathrm{CrossAttn}(X_q,X_c)=\mathrm{Concat}\big(A^{(0)}V^{(0)},\dots,A^{(H-1)}V^{(H-1)}\big)W_o
$$

中间对象是矩形的分数和权重，权重沿 key 维（长度 $T_k$）归一化：

$$
S^{(h)}=\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d}}\in\mathbb{R}^{B\times T_q\times T_k},\quad A^{(h)}_{ij}=\frac{\exp(S^{(h)}_{ij})}{\sum_{l=1}^{T_k}\exp(S^{(h)}_{il})}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # 头数 H 与单头维度 d = D // H
        self.h, self.d = n_heads, d_model // n_heads
        # Query 投影，作用在目标序列：D → D
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Key 投影，作用在上下文序列：D → D
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Value 投影，作用在上下文序列：D → D
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影：D → D
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split(self, z: Tensor) -> Tensor:
        # 长度由输入决定，同一函数可处理 Tq 和 Tk
        B, T, _ = z.shape
        # (B, T, D) → (B, T, H, d) → (B, H, T, d)
        return z.view(B, T, self.h, self.d).transpose(1, 2)

    def forward(self, query: Tensor, context: Tensor,
                mask: Tensor = None) -> Tensor:
        # 拼头时恢复形状要用，输出长度跟随 Tq
        B, Tq, D = query.shape
        # query: (B, Tq, D) → q: (B, H, Tq, d)
        q = self.split(self.w_q(query))
        # context: (B, Tk, D) → k: (B, H, Tk, d)
        k = self.split(self.w_k(context))
        # context: (B, Tk, D) → v: (B, H, Tk, d)
        v = self.split(self.w_v(context))
        # k^T: (B, H, d, Tk)，matmul → scores: (B, H, Tq, Tk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        # mask 可广播到 (B, H, Tq, Tk)，约定 True=允许，False=禁止
        if mask is not None:
            # 取反后填 -inf，禁止位置的权重变为 0
            scores = scores.masked_fill(~mask, float("-inf"))
        # 沿 key 维归一化，A: (B, H, Tq, Tk)
        A = scores.softmax(dim=-1)
        # (B, H, Tq, Tk) @ (B, H, Tk, d) → (B, H, Tq, d)
        out = A @ v
        # (B, H, Tq, d) → (B, Tq, H, d) → 拼头 (B, Tq, D)
        out = out.transpose(1, 2).reshape(B, Tq, D)
        # (B, Tq, D) → (B, Tq, D)
        return self.w_o(out)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢，比值不变；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    # 逐元素取指数，形状不变
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def split_np(z, h):
    # 长度由输入决定，同一函数可处理 Tq 和 Tk
    B, T, D = z.shape
    # (B, T, D) → (B, T, H, d) → (B, H, T, d)
    return z.reshape(B, T, h, D // h).transpose(0, 2, 1, 3)

def cross_attention_np(query, context, w_q, w_k, w_v, w_o, h, mask=None):
    # 权重形状均为 (D, D)；单头维度 d = D // H
    d = query.shape[-1] // h
    # query: (B, Tq, D) → q: (B, H, Tq, d)
    q = split_np(query @ w_q, h)
    # context: (B, Tk, D) → k: (B, H, Tk, d)
    k = split_np(context @ w_k, h)
    # context: (B, Tk, D) → v: (B, H, Tk, d)
    v = split_np(context @ w_v, h)
    # swapaxes 得 k^T，matmul → scores: (B, H, Tq, Tk)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)
    # mask 可广播到 (B, H, Tq, Tk)，约定 True=允许，False=禁止
    if mask is not None:
        # 禁止位置填 -inf，权重变为 0
        scores = np.where(mask, scores, -np.inf)
    # 沿 key 维归一化，A: (B, H, Tq, Tk)
    A = softmax_np(scores, axis=-1)
    # (B, H, Tq, Tk) @ (B, H, Tk, d) → (B, H, Tq, d)
    out = A @ v
    # (B, H, Tq, d) → (B, Tq, H, d) → 拼头 (B, Tq, D)
    out = out.transpose(0, 2, 1, 3).reshape(query.shape)
    # (B, Tq, D) @ (D, D) → (B, Tq, D)
    return out @ w_o
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, Tq, Tk, D, H = 2, 3, 5, 8, 2
query, context = torch.randn(B, Tq, D), torch.randn(B, Tk, D)
attn = CrossAttention(D, H)
# Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
w = [m.weight.detach().T.numpy()
     for m in (attn.w_q, attn.w_k, attn.w_v, attn.w_o)]
out_pt = attn(query, context)
out_np = cross_attention_np(query.numpy(), context.numpy(), *w, H)
print("shape:", out_pt.shape)
print("后端差异:", abs(out_pt.detach().numpy() - out_np).max())
# 打乱 context 顺序：K/V 同步重排，输出不变（集合语义）
perm = torch.randperm(Tk)
gap = attn(query, context[:, perm]) - out_pt
print("置换差异:", gap.abs().max().item())
```

没有位置编码时，交叉注意力对 context 的顺序不敏感，顺序信息只能来自位置编码。

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        pass

    def split(self, z: Tensor) -> Tensor:
        pass

    def forward(self, query: Tensor, context: Tensor,
                mask: Tensor = None) -> Tensor:
        pass
```

##### Step 1（i）公式：定义四个投影

$$
d=\frac{D}{H},\quad W_q,W_k,W_v,W_o\in\mathbb{R}^{D\times D}
$$

##### Step 1（ii）代码

```python
super().__init__()
self.h, self.d = n_heads, d_model // n_heads
self.w_q = nn.Linear(d_model, d_model, bias=False)
self.w_k = nn.Linear(d_model, d_model, bias=False)
self.w_v = nn.Linear(d_model, d_model, bias=False)
self.w_o = nn.Linear(d_model, d_model, bias=False)
```

##### Step 1（iii）解释

结构和自注意力相同，差别只在 forward 的输入来源。这里两条序列的维度都是 $D$，符合题面。

##### Step 2（i）公式：拆头

$$
\mathbb{R}^{B\times T\times D}\to\mathbb{R}^{B\times T\times H\times d}\to\mathbb{R}^{B\times H\times T\times d}
$$

##### Step 2（ii）代码

```python
B, T, _ = z.shape
# (B, T, D) → (B, T, H, d) → (B, H, T, d)
return z.view(B, T, self.h, self.d).transpose(1, 2)
```

##### Step 2（iii）解释

长度从输入读取，所以同一个函数既可以处理 $T_q$，也可以处理 $T_k$。

##### Step 3（i）公式：Q 来自 query，K/V 来自 context

$$
Q=\mathrm{split}(X_qW_q),\quad K=\mathrm{split}(X_cW_k),\quad V=\mathrm{split}(X_cW_v)
$$

##### Step 3（ii）代码

```python
B, Tq, D = query.shape
# query: (B, Tq, D) → q: (B, H, Tq, d)
q = self.split(self.w_q(query))
# context: (B, Tk, D) → k/v: (B, H, Tk, d)
k = self.split(self.w_k(context))
v = self.split(self.w_v(context))
```

##### Step 3（iii）解释

K 和 V 必须来自同一序列且长度相同，这样权重才能对应到 V 上。

##### Step 4（i）公式：矩形分数矩阵

$$
S=\frac{QK^\top}{\sqrt{d}}\in\mathbb{R}^{B\times H\times T_q\times T_k}
$$

##### Step 4（ii）代码

```python
# k^T: (B, H, d, Tk)，matmul → scores: (B, H, Tq, Tk)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
```

##### Step 4（iii）解释

$S_{hij}$ 是第 $i$ 个目标 token 对第 $j$ 个上下文 token 的相似度。矩阵一般不是方阵。

##### Step 5（i）公式：掩码与归一化

$$
A_{hij}=\frac{\exp(S_{hij})}{\sum_{l=1}^{T_k}\exp(S_{hil})}
$$

##### Step 5（ii）代码

```python
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
# 沿 key 维归一化，A: (B, H, Tq, Tk)
A = scores.softmax(dim=-1)
```

##### Step 5（iii）解释

归一化沿 $T_k$ 维，也就是每个目标 token 在所有上下文 token 上分配权重。mask 约定 True=允许，所以填充前取反。

##### Step 6（i）公式：加权 V、拼头、投影

$$
O=AV\in\mathbb{R}^{B\times H\times T_q\times d},\quad \hat{X}=\mathrm{Concat}(O)W_o\in\mathbb{R}^{B\times T_q\times D}
$$

##### Step 6（ii）代码

```python
# (B, H, Tq, Tk) @ (B, H, Tk, d) → (B, H, Tq, d)
out = A @ v
# (B, H, Tq, d) → (B, Tq, H, d) → 拼头 (B, Tq, D)
out = out.transpose(1, 2).reshape(B, Tq, D)
# (B, Tq, D) → (B, Tq, D)
return self.w_o(out)
```

##### Step 6（iii）解释

matmul 把 $T_k$ 维消掉，所以输出长度是 $T_q$。拼头时用 query 的形状恢复。

##### 5️⃣ 复杂度

主项来自投影和各头的 $QK^\top$：

$$
\text{时间}=O\big(B(T_q+T_k)D^2+BT_qT_kD\big),\quad \text{激活显存}=O(BHT_qT_k+B(T_q+T_k)D)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么 score 是 `[B,H,Tq,Tk]`，不是 `[B,H,Tq,Tq]`

分数的第 $i$ 行来自 Q，第 $j$ 列来自 K。K 由 context 产生，长度是 $T_k$：

$$
(B,H,T_q,d)\ @\ (B,H,d,T_k)\to(B,H,T_q,T_k)
$$

只有 K 的长度也是 $T_q$ 才会得到 $T_q\times T_q$，那就是 $T_q=T_k$ 的情况或者自注意力。

```python
# q: (B, H, Tq, d)；k: (B, H, Tk, d)
q = attn.split(attn.w_q(query))
k = attn.split(attn.w_k(context))
# 矩形分数：(B, H, Tq, Tk)
s = q @ k.transpose(-2, -1) / math.sqrt(attn.d)
print("scores:", s.shape)
# 沿 Tk 维归一化后，每个目标 token 的权重和为 1：(B, H, Tq)
print("每行权重和:", s.softmax(-1).sum(-1)[0, 0].tolist())
```

##### Follow-up 2：如何组合 decoder causal mask 与 encoder padding mask

两个 mask 作用在不同层，不需要合成一个张量。Decoder 自注意力的分数是 $T_q\times T_q$，使用 causal mask；Cross-Attention 的分数是 $T_q\times T_k$，使用 encoder padding mask。Cross-Attention 不需要 causal mask，因为 encoder 序列已知，第 $j$ 个 key 不是目标序列的未来 token。

$$
M^{\text{self}}_{bij}=\mathbb{1}[j\le i]\wedge P^{\text{tgt}}_{bj},\qquad M^{\text{cross}}_{bij}=P^{\text{src}}_{bj}
$$

```python
# src_pad: (B, Tk)，True=有效；第 2 个样本末尾 2 个 token 是 padding
src_pad = torch.ones(B, Tk, dtype=torch.bool)
src_pad[1, -2:] = False
# Cross mask：(B, Tk) → (B, 1, 1, Tk)，广播到 (B, H, Tq, Tk)
cross_mask = src_pad[:, None, None, :]
# Self mask：(Tq, Tq)，下三角，用于 decoder 自注意力层
self_mask = torch.tril(torch.ones(Tq, Tq, dtype=torch.bool))
print("cross/self:", cross_mask.shape, self_mask.shape)
# 改动 pad 位置的内容，输出应不变
ctx2 = context.clone()
ctx2[1, -2:] += 100.0
gap = attn(query, context, cross_mask) - attn(query, ctx2, cross_mask)
print("改 pad 的影响:", gap.abs().max().item())
```

##### Follow-up 3：`Tq=1` 增量解码时，接口和缓存如何修改

Context 在解码期间不变，所以 K/V 只算一次并缓存，之后每步只投影新 token 的 Q：

$$
\text{单步时间}=O(BD^2+BT_kD),\quad \text{缓存}=O(BT_kD)
$$

接口拆成两部分：`build_kv` 预计算，`cross_step` 接收 `(B,1,D)` 和缓存。Cross-Attention 的 K/V 不随步数增长，这一点与自注意力的 KV Cache 不同。

```python
def build_kv(attn, context):
    # context: (B, Tk, D) → k/v: (B, H, Tk, d)，解码前算一次
    return attn.split(attn.w_k(context)), attn.split(attn.w_v(context))

def cross_step(attn, q_t, k, v):
    # q_t: (B, 1, D) → q: (B, H, 1, d)
    q = attn.split(attn.w_q(q_t))
    # scores: (B, H, 1, Tk)
    s = q @ k.transpose(-2, -1) / math.sqrt(attn.d)
    # (B, H, 1, Tk) @ (B, H, Tk, d) → (B, H, 1, d)
    o = s.softmax(-1) @ v
    # (B, H, 1, d) → (B, 1, H, d) → (B, 1, D)
    return attn.w_o(o.transpose(1, 2).reshape(q_t.shape))

k_c, v_c = build_kv(attn, context)
# 逐步解码后按时间拼接：Tq 个 (B, 1, D) → (B, Tq, D)
steps = [cross_step(attn, query[:, [t]], k_c, v_c) for t in range(Tq)]
out_inc = torch.cat(steps, dim=1)
print("增量差异:", (out_inc - out_pt).abs().max().item())
```

#### Q006｜Transformer Decoder Block（完整从零实现）

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-02（公开面经日期）
- **题目：** 现场编码 Decoder Block：Causal MHA、Residual、Normalization、FFN，并给出可复用的 `forward`。
- **输入输出 / 约束：** 输入与输出 `x:[B,T,D]`；`D,H,D_ff,dropout` 可配置。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Pre-LN 与 Post-LN 的代码位置怎样不同？为何深层 LLM 偏好 Pre-LN？
  2. 训练与 eval 时 Dropout 有什么变化？
  3. 如果 FFN 替换为 SwiGLU，要增加哪些线性层，维度如何匹配？
- **来源：** [Jointaro：TikTok MLE CE 推导 + Transformer Block + 树题](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/)

##### 1️⃣ Dense Opening

Transformer Decoder Block 由“Causal MHA + 残差”和“FFN + 残差”两层子层组成，主实现采用 Pre-LN 结构，也就是先归一化，再进子层，再做 Dropout 和残差相加。数学核心对象是逐层残差更新 $x\leftarrow x+\mathrm{Drop}(f(\mathrm{LN}(x)))$。题面只要求 Causal MHA、Residual、Normalization、FFN，所以这里是不含 Cross-Attention 的 GPT 式 Decoder Block。

##### 2️⃣ Global Equations

主目标，两个子层依次做 Pre-LN 残差更新：

$$
x'=x+\mathrm{Drop}\big(\mathrm{MHA}(\mathrm{LN}_1(x),M)\big),\quad y=x'+\mathrm{Drop}\big(\mathrm{FFN}(\mathrm{LN}_2(x'))\big)
$$

Causal 掩码，只允许看当前及过去的位置：

$$
M_{ij}=\mathbb{1}[j\le i]
$$

FFN 逐 token 作用，$D\to D_{ff}\to D$：

$$
\mathrm{FFN}(z)=\mathrm{ReLU}(zW_1+b_1)W_2+b_2
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class CausalMHA(nn.Module):
    def __init__(self, D: int, H: int):
        super().__init__()
        # 头数 H 与单头维度 d = D // H
        self.h, self.d = H, D // H
        # Q/K/V 三个投影：D → D
        self.w_q = nn.Linear(D, D, bias=False)
        self.w_k = nn.Linear(D, D, bias=False)
        self.w_v = nn.Linear(D, D, bias=False)
        # 输出投影：D → D
        self.w_o = nn.Linear(D, D, bias=False)

    def split(self, z: Tensor) -> Tensor:
        # 取出 Batch 和序列长度
        B, T, _ = z.shape
        # (B, T, D) → (B, T, H, d) → (B, H, T, d)
        return z.view(B, T, self.h, self.d).transpose(1, 2)

    def forward(self, x: Tensor, pad: Tensor = None) -> Tensor:
        # 拼头时恢复形状要用
        B, T, D = x.shape
        # x: (B, T, D) → q: (B, H, T, d)
        q = self.split(self.w_q(x))
        # x: (B, T, D) → k: (B, H, T, d)
        k = self.split(self.w_k(x))
        # x: (B, T, D) → v: (B, H, T, d)
        v = self.split(self.w_v(x))
        # k^T: (B, H, d, T)，matmul → scores: (B, H, T, T)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        # 全 True 的 (T, T)，放到 x 所在设备
        one = torch.ones(T, T, dtype=torch.bool, device=x.device)
        # 下三角：(T, T)，True=允许，False=未来位置
        allow = torch.tril(one)
        # pad: (B, T)，True=有效；可选
        if pad is not None:
            # (T, T) & (B, 1, 1, T) → (B, 1, T, T)，取交集
            allow = allow & pad[:, None, None, :]
        # 广播到 (B, H, T, T)，禁止位置填 -inf
        scores = scores.masked_fill(~allow, float("-inf"))
        # 沿 key 维归一化，A: (B, H, T, T)
        A = scores.softmax(dim=-1)
        # (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
        out = A @ v
        # (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
        out = out.transpose(1, 2).reshape(B, T, D)
        # (B, T, D) → (B, T, D)
        return self.w_o(out)

class DecoderBlock(nn.Module):
    def __init__(self, D: int, H: int, D_ff: int, dropout: float = 0.1):
        super().__init__()
        # 第一个子层：Causal MHA 及其归一化
        self.attn = CausalMHA(D, H)
        self.ln1 = nn.LayerNorm(D)
        # 第二个子层：FFN，D → D_ff → D
        self.ffn = nn.Sequential(
            nn.Linear(D, D_ff), nn.ReLU(), nn.Linear(D_ff, D))
        self.ln2 = nn.LayerNorm(D)
        # 残差 Dropout，两个子层共用同一模块
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, pad: Tensor = None) -> Tensor:
        # Pre-LN：(B, T, D) 先归一化，再注意力，再残差相加
        x = x + self.drop(self.attn(self.ln1(x), pad))
        # Pre-LN：(B, T, D) 先归一化，再 FFN，再残差相加
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢，比值不变；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    # 逐元素取指数，形状不变
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm_np(x, gamma, beta, eps=1e-5):
    # 沿最后一维：(B, T, D) → (B, T, 1)
    mu = x.mean(-1, keepdims=True)
    # 沿最后一维的总体方差：(B, T, D) → (B, T, 1)
    var = x.var(-1, keepdims=True)
    # 归一化后施加仿射，gamma/beta: (D,) 广播
    return (x - mu) / np.sqrt(var + eps) * gamma + beta

def dropout_np(x, p, rng):
    # p=0 对应 eval 模式，直接返回
    if p == 0:
        return x
    # keep 与 x 同形，True 表示保留
    keep = rng.random(x.shape) >= p
    # 保留位置放大 1/(1-p)，期望不变
    return x * keep / (1 - p)

def causal_mha_np(x, w_q, w_k, w_v, w_o, H, pad=None):
    B, T, D = x.shape
    # 单头维度 d = D // H
    d = D // H
    # 拆头：(B, T, D) → (B, T, H, d) → (B, H, T, d)
    def split(z):
        return z.reshape(B, T, H, d).transpose(0, 2, 1, 3)
    # x: (B, T, D) → q/k/v: (B, H, T, d)
    q, k, v = split(x @ w_q), split(x @ w_k), split(x @ w_v)
    # swapaxes 得 k^T，matmul → scores: (B, H, T, T)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(d)
    # 下三角：(T, T)，True=允许，False=未来位置
    allow = np.tril(np.ones((T, T), dtype=bool))
    # pad: (B, T)，True=有效；可选
    if pad is not None:
        # (T, T) & (B, 1, 1, T) → (B, 1, T, T)，取交集
        allow = allow & pad[:, None, None, :]
    # 广播到 (B, H, T, T)，禁止位置填 -inf
    scores = np.where(allow, scores, -np.inf)
    # 沿 key 维归一化，再加权 V：(B, H, T, d)
    out = softmax_np(scores, axis=-1) @ v
    # (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    # (B, T, D) @ (D, D) → (B, T, D)
    return out @ w_o

def decoder_block_np(x, w, H, pad=None, p=0.0, rng=None):
    # Pre-LN 注意力子层，(B, T, D) → (B, T, D)
    a = causal_mha_np(layer_norm_np(x, *w["ln1"]), *w["attn"], H, pad)
    # 残差相加，p=0 时 Dropout 不起作用
    x = x + dropout_np(a, p, rng)
    # Pre-LN FFN 子层：(B, T, D) → (B, T, D_ff) → (B, T, D)
    h = layer_norm_np(x, *w["ln2"]) @ w["w1"] + w["b1"]
    # ReLU 后投影回 D
    f = np.maximum(h, 0.0) @ w["w2"] + w["b2"]
    return x + dropout_np(f, p, rng)
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D, H, D_ff = 2, 5, 8, 2, 32
x = torch.randn(B, T, D)
block = DecoderBlock(D, H, D_ff).eval()
# 辅助：Tensor 转 NumPy
n = lambda t: t.detach().numpy()
# Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
a, f = block.attn, block.ffn
w = {"ln1": (n(block.ln1.weight), n(block.ln1.bias)),
     "ln2": (n(block.ln2.weight), n(block.ln2.bias)),
     "attn": [n(m.weight).T for m in (a.w_q, a.w_k, a.w_v, a.w_o)],
     "w1": n(f[0].weight).T, "b1": n(f[0].bias),
     "w2": n(f[2].weight).T, "b2": n(f[2].bias)}
out_pt = block(x)
out_np = decoder_block_np(x.numpy(), w, H)
print("后端差异:", abs(n(out_pt) - out_np).max())
# 只改最后一个 token，前 T-1 个位置的输出不应改变
y = x.clone()
y[:, -1] += 100.0
print("因果差异:", (block(y) - out_pt)[:, :-1].abs().max().item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
class CausalMHA(nn.Module):
    def __init__(self, D: int, H: int):
        pass
    def split(self, z: Tensor) -> Tensor:
        pass
    def forward(self, x: Tensor, pad: Tensor = None) -> Tensor:
        pass

class DecoderBlock(nn.Module):
    def __init__(self, D: int, H: int, D_ff: int, dropout: float = 0.1):
        pass
    def forward(self, x: Tensor, pad: Tensor = None) -> Tensor:
        pass
```

##### Step 1（i）公式：定义注意力投影

$$
d=\frac{D}{H},\quad W_q,W_k,W_v,W_o\in\mathbb{R}^{D\times D}
$$

##### Step 1（ii）代码

```python
super().__init__()
self.h, self.d = H, D // H
self.w_q = nn.Linear(D, D, bias=False)
self.w_k = nn.Linear(D, D, bias=False)
self.w_v = nn.Linear(D, D, bias=False)
self.w_o = nn.Linear(D, D, bias=False)
```

##### Step 1（iii）解释

四个线性层都是 $D\to D$，所有头在一次投影中算完，拆头之后才形成 $d$ 维子空间。

##### Step 2（i）公式：拆头并计算分数

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times d},\quad S=\frac{QK^\top}{\sqrt{d}}\in\mathbb{R}^{B\times H\times T\times T}
$$

##### Step 2（ii）代码

```python
B, T, _ = z.shape
# (B, T, D) → (B, T, H, d) → (B, H, T, d)
return z.view(B, T, self.h, self.d).transpose(1, 2)
B, T, D = x.shape
# x: (B, T, D) → q/k/v: (B, H, T, d)
q = self.split(self.w_q(x))
k = self.split(self.w_k(x))
v = self.split(self.w_v(x))
# k^T: (B, H, d, T)，matmul → scores: (B, H, T, T)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
```

##### Step 2（iii）解释

前三行属于 `split`，后面属于 `forward`。先 `view` 再 `transpose`，头维才不会跨 token 切分。

##### Step 3（i）公式：构造并施加 Causal 掩码

$$
M_{bij}=\mathbb{1}[j\le i]\wedge P_{bj},\quad S_{bhij}\leftarrow-\infty\ \text{ 若 }M_{bij}=0
$$

##### Step 3（ii）代码

```python
one = torch.ones(T, T, dtype=torch.bool, device=x.device)
# 下三角：(T, T)，True=允许，False=未来位置
allow = torch.tril(one)
if pad is not None:
    # (T, T) & (B, 1, 1, T) → (B, 1, T, T)，取交集
    allow = allow & pad[:, None, None, :]
# 广播到 (B, H, T, T)，禁止位置填 -inf
scores = scores.masked_fill(~allow, float("-inf"))
```

##### Step 3（iii）解释

Causal 掩码写在 Block 内部，调用方不用每次构造，同一个 `forward` 可直接用于训练和推理。这里假设 padding 在右侧；左 padding 会产生全遮挡行，Softmax 会出现 NaN，需要额外处理。

##### Step 4（i）公式：Softmax、加权 V、拼头

$$
A=\mathrm{softmax}(S),\quad O=AV,\quad \mathrm{MHA}(x)=\mathrm{Concat}(O)W_o
$$

##### Step 4（ii）代码

```python
A = scores.softmax(dim=-1)
# (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
out = A @ v
# (B, H, T, d) → (B, T, H, d) → 拼头 (B, T, D)
out = out.transpose(1, 2).reshape(B, T, D)
# (B, T, D) → (B, T, D)
return self.w_o(out)
```

##### Step 4（iii）解释

拼头是拆头的逆操作，`transpose` 之后内存不连续，所以用 `reshape`。

##### Step 5（i）公式：定义 Block 的子模块

$$
\mathrm{FFN}:\ D\to D_{ff}\to D,\quad \mathrm{LN}_1,\mathrm{LN}_2:\ \gamma,\beta\in\mathbb{R}^{D}
$$

##### Step 5（ii）代码

```python
super().__init__()
self.attn = CausalMHA(D, H)
self.ln1 = nn.LayerNorm(D)
# FFN：D → D_ff → D
self.ffn = nn.Sequential(
    nn.Linear(D, D_ff), nn.ReLU(), nn.Linear(D_ff, D))
self.ln2 = nn.LayerNorm(D)
self.drop = nn.Dropout(dropout)
```

##### Step 5（iii）解释

`D,H,D_ff,dropout` 都可配置，满足题面。`self.ffn` 是独立属性，Follow-up 3 直接替换它就能换成 SwiGLU。

##### Step 6（i）公式：注意力子层的残差

$$
x'=x+\mathrm{Drop}\big(\mathrm{MHA}(\mathrm{LN}_1(x),M)\big)
$$

##### Step 6（ii）代码

```python
# Pre-LN：(B, T, D) 先归一化，再注意力，再残差相加
x = x + self.drop(self.attn(self.ln1(x), pad))
```

##### Step 6（iii）解释

残差路径上的 $x$ 不经过 LN，这是 Pre-LN 的关键。Dropout 作用在子层输出上，再与残差相加。

##### Step 7（i）公式：FFN 子层的残差

$$
y=x'+\mathrm{Drop}\big(\mathrm{FFN}(\mathrm{LN}_2(x'))\big)
$$

##### Step 7（ii）代码

```python
# Pre-LN：(B, T, D) 先归一化，再 FFN，再残差相加
x = x + self.drop(self.ffn(self.ln2(x)))
return x
```

##### Step 7（iii）解释

输入和输出都是 `[B,T,D]`，可以直接堆叠多层。Pre-LN 的最后一层输出未经归一化，整个模型末尾通常要再加一个 LN。

##### 5️⃣ 复杂度

主项来自注意力的 $QK^\top$ 和两个子层的线性投影：

$$
\text{时间}=O\big(BT^2D+BTD^2+BTDD_{ff}\big)
$$

$$
\text{激活显存}=O\big(BHT^2+BT(D+D_{ff})\big),\quad \text{参数量}=4D^2+2DD_{ff}+O(D_{ff}+D)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：Pre-LN 与 Post-LN 的代码位置有何不同，为何深层 LLM 偏好 Pre-LN

代码差别只在 LN 的位置：

$$
\text{Pre-LN：}\ x\leftarrow x+f(\mathrm{LN}(x)),\qquad \text{Post-LN：}\ x\leftarrow\mathrm{LN}(x+f(x))
$$

Pre-LN 的残差路径是恒等映射，每层 Jacobian 形如 $I+J$，信号和梯度可以不经过 LN 直接传到浅层。Post-LN 的主干每层都要穿过一次 LN。下面把子层输出清零来验证这条路径：Pre-LN 严格返回输入，Post-LN 仍会改变输入。

```python
def post_ln_forward(b, x):
    # Post-LN：先子层、再残差相加、最后归一化，形状 (B, T, D)
    x = b.ln1(x + b.drop(b.attn(x)))
    return b.ln2(x + b.drop(b.ffn(x)))

zero = DecoderBlock(D, H, D_ff).eval()
with torch.no_grad():
    # 注意力输出投影清零：子层 1 输出恒为 0
    zero.attn.w_o.weight.zero_()
    # FFN 最后一层清零：子层 2 输出恒为 0
    zero.ffn[2].weight.zero_()
    zero.ffn[2].bias.zero_()
print("Pre-LN 偏离输入:", (zero(x) - x).abs().max().item())
gap = post_ln_forward(zero, x) - x
print("Post-LN 偏离输入:", gap.abs().max().item())
```

Xiong et al.（2020，On Layer Normalization in the Transformer Architecture）分析并实验指出，Pre-LN 的梯度在初始化时更受控，训练可以不依赖学习率 warmup，这是深层模型偏好它的主要原因。代价是残差流的尺度随层数累积，所以模型末尾要再加一个 LN。

##### Follow-up 2：训练与 eval 时 Dropout 有什么变化

训练时，每个元素以概率 $p$ 置 0，保留的元素放大 $1/(1-p)$，期望不变：

$$
\tilde{z}_i=\frac{m_i}{1-p}z_i,\quad m_i\sim\mathrm{Bernoulli}(1-p),\quad \mathbb{E}[\tilde{z}_i]=z_i
$$

eval 时是恒等映射。所以 `model.eval()` 必须在推理前调用，否则输出是随机的。

```python
drop_block = DecoderBlock(D, H, D_ff, dropout=0.5)
# train 模式：两次前向的 mask 不同，输出不同
drop_block.train()
print("train 差异:", (drop_block(x) - drop_block(x)).abs().max().item())
# eval 模式：Dropout 为恒等，两次输出完全相同
drop_block.eval()
print("eval 差异:", (drop_block(x) - drop_block(x)).abs().max().item())
# 验证期望不变：全 1 输入的均值约等于 1
ones = torch.ones(100000)
print("Dropout 均值:", nn.functional.dropout(ones, 0.5, True).mean().item())
```

##### Follow-up 3：FFN 替换为 SwiGLU，要增加哪些线性层，维度如何匹配

原 FFN 两个线性层，SwiGLU 三个：gate 和 up 都是 $D\to D_{ff}$，down 是 $D_{ff}\to D$。

$$
\mathrm{SwiGLU}(z)=\big(\mathrm{SiLU}(zW_g)\odot zW_u\big)W_d
$$

参数量从 $2DD_{ff}$ 变为 $3DD_{ff}$。若要保持参数量相近，把 $D_{ff}$ 取为原来的 $2/3$。

```python
class SwiGLU(nn.Module):
    def __init__(self, D, D_ff):
        super().__init__()
        # 门控支路和数值支路：D → D_ff
        self.w_gate = nn.Linear(D, D_ff, bias=False)
        self.w_up = nn.Linear(D, D_ff, bias=False)
        # 输出：D_ff → D
        self.w_down = nn.Linear(D_ff, D, bias=False)

    def forward(self, z):
        # 逐元素门控：(B, T, D_ff) → (B, T, D_ff) → (B, T, D)
        gated = nn.functional.silu(self.w_gate(z)) * self.w_up(z)
        return self.w_down(gated)

swi_block = DecoderBlock(D, H, D_ff).eval()
# 只替换 ffn 属性，forward 不用修改
swi_block.ffn = SwiGLU(D, D_ff * 2 // 3)
print("输出 shape:", swi_block(x).shape)
count = lambda m: sum(p.numel() for p in m.parameters())
print("FFN/SwiGLU 参数:", count(block.ffn), count(swi_block.ffn))
```

这里原 FFN 带 bias，SwiGLU 不带，所以打印值是 $2DD_{ff}+D_{ff}+D=552$ 和 $3D\lfloor 2D_{ff}/3\rfloor=504$。

#### Q007｜从零实现 LayerNorm（NumPy）

- **优先级：** P0 ｜ **证据：** B ｜ **年月：** 2026-06（文章更新日，非面试日）
- **题目：** 不直接调用 LayerNorm API，用 NumPy 沿最后一维计算均值和方差，完成归一化并施加可学习仿射参数。
- **输入输出 / 约束：** 输入 `x:[B,T,D]`、`gamma,beta:[D]`、`eps`；输出 `[B,T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何通常用总体方差（`ddof=0`）而不是无偏方差？
  2. 归一化维度错成 Batch 维会发生什么？与 BatchNorm 对比。
  3. 常数输入、极小方差、FP16 计算时如何保证稳定？
- **来源：** [IGotAnOffer：TikTok MLE 编程题指南](https://igotanoffer.com/en/advice/tiktok-machine-learning-engineer-interview)

##### 1️⃣ Dense Opening

层归一化（Layer Normalization）对每个 token 的 $D$ 维特征独立做标准化，再施加可学习仿射，结构是“沿最后一维求均值和方差 → 标准化 → 乘 $\gamma$ 加 $\beta$”，数学核心对象是逐 token 的统计量 $\mu,\sigma^2\in\mathbb{R}^{B\times T\times 1}$。题面要求 NumPy，所以 Step 和骨架以 NumPy 版为准，PyTorch 版是同一计算图的另一后端。

##### 2️⃣ Global Equations

沿最后一维计算总体均值和总体方差（$\mathrm{ddof}=0$）：

$$
\mu_{bt}=\frac{1}{D}\sum_{l=1}^{D}x_{btl},\quad \sigma^2_{bt}=\frac{1}{D}\sum_{l=1}^{D}(x_{btl}-\mu_{bt})^2
$$

主目标，标准化后逐维施加仿射：

$$
\mathrm{LN}(x)_{btl}=\gamma_l\frac{x_{btl}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}+\beta_l
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn as nn
from torch import Tensor

def layer_norm_pt(x: Tensor, gamma: Tensor, beta: Tensor,
                  eps: float = 1e-5) -> Tensor:
    # 沿特征维求均值：(B, T, D) → (B, T, 1)
    mu = x.mean(dim=-1, keepdim=True)
    # 沿特征维求总体方差：(B, T, D) → (B, T, 1)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    # 广播 (B, T, 1) 到 (B, T, D)，得到标准化结果
    z = (x - mu) / torch.sqrt(var + eps)
    # gamma/beta: (D,) 广播到 (B, T, D)，逐维仿射
    return z * gamma + beta
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def layer_norm_np(x, gamma, beta, eps=1e-5):
    # 沿特征维求均值：(B, T, D) → (B, T, 1)
    mu = x.mean(axis=-1, keepdims=True)
    # 沿特征维求总体方差：(B, T, D) → (B, T, 1)
    var = x.var(axis=-1, keepdims=True)
    # 广播 (B, T, 1) 到 (B, T, D)，得到标准化结果
    z = (x - mu) / np.sqrt(var + eps)
    # gamma/beta: (D,) 广播到 (B, T, D)，逐维仿射
    return z * gamma + beta
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D = 2, 3, 4
x = torch.randn(B, T, D)
ln = nn.LayerNorm(D, eps=1e-5)
# 官方模块的 weight/bias 就是 gamma/beta，形状均为 (D,)
torch.nn.init.normal_(ln.weight)
torch.nn.init.normal_(ln.bias)
g, b = ln.weight.detach(), ln.bias.detach()
ref = ln(x).detach()
print("PT vs 官方:", (layer_norm_pt(x, g, b) - ref).abs().max().item())
out_np = layer_norm_np(x.numpy(), g.numpy(), b.numpy())
print("NP vs 官方:", abs(out_np - ref.numpy()).max())
# 手算：[1, 3] 的 mu=2、var=1，取 eps=0 时结果为 [-1, 1]
hand = np.array([[[1.0, 3.0]]])
print("手算:", layer_norm_np(hand, np.ones(2), np.zeros(2), eps=0.0))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def layer_norm_np(x, gamma, beta, eps=1e-5):
    pass
```

##### Step 1（i）公式：沿最后一维求均值

$$
\mu_{bt}=\frac{1}{D}\sum_{l=1}^{D}x_{btl}
$$

##### Step 1（ii）代码

```python
# 沿特征维求均值：(B, T, D) → (B, T, 1)
mu = x.mean(axis=-1, keepdims=True)
```

##### Step 1（iii）解释

`axis=-1` 是特征维，每个样本的每个 token 都有自己的均值。`keepdims=True` 保留长度为 1 的维度，后面才能与 `x` 广播。

##### Step 2（i）公式：求总体方差

$$
\sigma^2_{bt}=\frac{1}{D}\sum_{l=1}^{D}(x_{btl}-\mu_{bt})^2
$$

##### Step 2（ii）代码

```python
# 沿特征维求总体方差：(B, T, D) → (B, T, 1)
var = x.var(axis=-1, keepdims=True)
```

##### Step 2（iii）解释

`np.var` 默认 `ddof=0`，分母是 $D$。PyTorch 的 `var` 默认是无偏方差，所以 A 版要显式写 `unbiased=False`，这是两个后端最容易不一致的地方。

##### Step 3（i）公式：标准化

$$
z_{btl}=\frac{x_{btl}-\mu_{bt}}{\sqrt{\sigma^2_{bt}+\epsilon}}
$$

##### Step 3（ii）代码

```python
# 广播 (B, T, 1) 到 (B, T, D)，得到标准化结果
z = (x - mu) / np.sqrt(var + eps)
```

##### Step 3（iii）解释

$\epsilon$ 加在根号内，与 `nn.LayerNorm` 一致。方差为 0 时分母是 $\sqrt{\epsilon}$，不会出现除零。

##### Step 4（i）公式：施加可学习仿射

$$
\mathrm{LN}(x)_{btl}=\gamma_lz_{btl}+\beta_l
$$

##### Step 4（ii）代码

```python
# gamma/beta: (D,) 广播到 (B, T, D)，逐维仿射
return z * gamma + beta
```

##### Step 4（iii）解释

$\gamma$ 和 $\beta$ 是逐特征的参数，与序列长度无关，所以同一模块可以处理任意 $T$。初始化通常是 $\gamma=\mathbf{1},\beta=\mathbf{0}$。

##### 5️⃣ 复杂度

每个 token 上的统计量和仿射都是 $O(D)$：

$$
\text{时间}=O(BTD),\quad \text{额外内存}=O(BTD),\quad \text{参数量}=2D
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为何通常用总体方差（`ddof=0`），不用无偏方差

LayerNorm 的目的是把这 $D$ 个数标准化，这 $D$ 个数本身就是全部对象，不需要估计更大总体的方差。使用 $\mathrm{ddof}=0$ 时，标准化结果（$\epsilon\to0$）的方差恰好为 1。无偏方差会让结果缩小，比例是：

$$
\frac{\sigma_{\mathrm{ddof}=1}}{\sigma_{\mathrm{ddof}=0}}=\sqrt{\frac{D}{D-1}}
$$

此外，$D=1$ 时无偏方差的分母为 0。官方 `nn.LayerNorm` 也是 $\mathrm{ddof}=0$，下面的代码验证了这一点。

```python
xn = x.numpy()
# ddof=1 写法：(B, T, D) → (B, T, 1)
var1 = xn.var(axis=-1, ddof=1, keepdims=True)
z1 = (xn - xn.mean(-1, keepdims=True)) / np.sqrt(var1 + 1e-5)
# 官方 LN 取 gamma=1、beta=0，与 ddof=0 对齐
off = nn.LayerNorm(D)
print("ddof=0 差异:",
      abs(layer_norm_np(xn, np.ones(D), np.zeros(D))
          - off(x).detach().numpy()).max())
print("ddof=1 差异:", abs(z1 - off(x).detach().numpy()).max())
# D=1 时 ddof=1 的分母为 0，0/0 → NaN
d1 = np.array([2.0])
with np.errstate(all="ignore"):
    print("D=1, ddof=1:", ((d1 - d1.mean()) ** 2).sum() / np.float64(0))
```

##### Follow-up 2：归一化维度错成 Batch 维会发生什么，与 BatchNorm 对比

错用 `axis=0` 后，一个样本的输出会依赖同一 Batch 的其他样本，推理结果随 Batch 组成变化。当 $B=1$ 时方差恒为 0，输出被压成常数 $\beta$，信息全部丢失。

| 项目 | LayerNorm | BatchNorm |
|---|---|---|
| 统计维度 | 单 token 的特征维 | 跨 Batch 的同一特征 |
| 依赖其他样本 | 否 | 是 |
| train/eval 差异 | 无 | 有，eval 用滑动平均统计量 |
| 变长序列、小 Batch | 适用 | 统计量不稳定 |

```python
def wrong_ln(x, eps=1e-5):
    # 错误：沿 Batch 维求统计量，(B, T, D) → (1, T, D)
    mu, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

g1, b0 = np.ones(D), np.zeros(D)
# 只改第 2 个样本，观察第 1 个样本的输出
xn2 = xn.copy()
xn2[1] += 10.0
print("正确 LN 影响:", abs(layer_norm_np(xn, g1, b0)[0]
                           - layer_norm_np(xn2, g1, b0)[0]).max())
print("错误维度影响:", abs(wrong_ln(xn)[0] - wrong_ln(xn2)[0]).max())
# B=1：(1, T, D) 沿 Batch 维方差为 0，输出全 0
print("B=1 输出最大值:", abs(wrong_ln(xn[:1])).max())
```

##### Follow-up 3：常数输入、极小方差、FP16 计算时如何保证稳定

- 常数输入：$x-\mu=0$、$\sigma^2=0$，输出是 $\gamma\cdot0/\sqrt{\epsilon}+\beta=\beta$，前提是 $\epsilon>0$。
- 极小方差：$\sigma^2\ll\epsilon$ 时分母近似 $\sqrt{\epsilon}$，输出幅度被限制为 $|x-\mu|/\sqrt{\epsilon}$，不会爆炸。
- FP16：$\epsilon$ 过小时会下溢为 0，平方也可能上溢。$|x-\mu|=300$ 时平方为 90000，超过 FP16 上限 65504，方差变成 inf，标准化结果被压成 0。

处理方法是把统计量计算上转到 FP32，算完再转回 FP16，同时选用在目标精度下仍非零的 $\epsilon$。

```python
# 常数输入：输出等于 beta
const = np.full((1, 1, 4), 5.0)
print("常数输入:", layer_norm_np(const, g1, np.full(4, 0.5)))
# FP16 下 1e-8 直接下溢为 0，1e-5 仍然非零
print("FP16 eps:", np.float16(1e-8), np.float16(1e-5))
c16 = np.full((1, 1, 4), 5.0, dtype=np.float16)
zero16 = np.float16(1e-8)
with np.errstate(all="ignore"):
    # eps 下溢后 0/0 → NaN
    print("eps=1e-8:", layer_norm_np(c16, g1, b0, eps=zero16))
    # mu=0，(x-mu)^2 上溢：方差 inf，输出被压成 0。形状 (1, 1, 2)
    big = np.array([[[-300.0, 300.0]]], dtype=np.float16)
    print("FP16 计算:", layer_norm_np(big, np.ones(2), np.zeros(2),
                                      eps=np.float16(1e-5)))
# 上转 FP32 计算统计量，结果转回 FP16，手算应为 [-1, 1]
safe = layer_norm_np(big.astype(np.float32), np.ones(2), np.zeros(2))
print("上转 FP32:", safe.astype(np.float16))
```

#### Q008｜从零实现 RMSNorm

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-08（发布日）
- **题目：** 手写 RMSNorm 模块，保留 scale 参数，不对输入做去均值操作。
- **输入输出 / 约束：** 输入 `x:[...,D]`、`gamma:[D]`；输出同 shape。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. RMSNorm 与 LayerNorm 在数学上最关键的区别是什么？
  2. `eps` 放在平方均值开根号的什么位置？
  3. 在低精度训练中哪些累计计算建议上转 FP32？
- **来源：** [牛客：字节大模型算法一二三面 RMSNorm/SwiGLU/Cross-Attention](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)

##### 1️⃣ Dense Opening

均方根归一化（Root Mean Square Normalization, RMSNorm）用最后一维的均方根（RMS）缩放输入，不做去均值，也没有偏置项，结构是“平方 → 沿最后一维求均值 → 加 $\epsilon$ 开根号 → 相除 → 乘 $\gamma$”，数学核心对象是逐向量的尺度 $\mathrm{RMS}(x)\in\mathbb{R}^{\dots\times1}$。

##### 2️⃣ Global Equations

中间对象是沿最后一维计算的均方根，$\epsilon$ 放在根号内：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{D}\sum_{l=1}^{D}x_l^2+\epsilon}
$$

主目标，缩放后逐维乘可学习参数 $\gamma\in\mathbb{R}^{D}$：

$$
\mathrm{RMSNorm}(x)_l=\gamma_l\frac{x_l}{\mathrm{RMS}(x)}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, D: int, eps: float = 1e-8):
        super().__init__()
        # 根号内的稳定项
        self.eps = eps
        # 逐维缩放参数 gamma: (D,)，初始化为 1
        self.gamma = nn.Parameter(torch.ones(D))

    def forward(self, x: Tensor) -> Tensor:
        # 沿特征维求平方均值：(..., D) → (..., 1)，不去均值
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # 加 eps 后开根号：(..., 1) → (..., 1)
        rms = torch.sqrt(ms + self.eps)
        # 广播：(D,) * (..., D) / (..., 1) → (..., D)
        return self.gamma * x / rms
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def rms_norm_np(x, gamma, eps=1e-8):
    # 沿特征维求平方均值：(..., D) → (..., 1)，不去均值
    ms = (x ** 2).mean(axis=-1, keepdims=True)
    # 加 eps 后开根号：(..., 1) → (..., 1)
    rms = np.sqrt(ms + eps)
    # 广播：(D,) * (..., D) / (..., 1) → (..., D)
    return gamma * x / rms
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# 手算：[3, 4] 的 RMS = sqrt(12.5) = 3.5355，结果 [0.8485, 1.1314]
hand = torch.tensor([3.0, 4.0])
print("手算:", RMSNorm(2)(hand).tolist())
# 任意前缀维度都能工作：(2, 3, 5, 8)
x = torch.randn(2, 3, 5, 8)
norm = RMSNorm(8)
g = norm.gamma.detach().numpy()
gap = norm(x).detach().numpy() - rms_norm_np(x.numpy(), g)
print("后端差异:", abs(gap).max())
# 与官方 nn.RMSNorm 对齐，eps=1 放大差异，可确认都在根号内
off, ours = nn.RMSNorm(8, eps=1.0), RMSNorm(8, eps=1.0)
print("与官方差异:", (off(x) - ours(x)).abs().max().item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, D: int, eps: float = 1e-8):
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass
```

##### Step 1（i）公式：定义参数

$$
\gamma\in\mathbb{R}^{D},\quad \gamma^{\text{init}}=\mathbf{1}
$$

##### Step 1（ii）代码

```python
super().__init__()
self.eps = eps
# 逐维缩放参数 gamma: (D,)，初始化为 1
self.gamma = nn.Parameter(torch.ones(D))
```

##### Step 1（iii）解释

题面要求保留 scale 参数，所以只有 $\gamma$。没有偏置项 $\beta$，参数量是 $D$，LayerNorm 是 $2D$。

##### Step 2（i）公式：沿最后一维求平方均值

$$
m=\frac{1}{D}\sum_{l=1}^{D}x_l^2
$$

##### Step 2（ii）代码

```python
# 沿特征维求平方均值：(..., D) → (..., 1)，不去均值
ms = x.pow(2).mean(dim=-1, keepdim=True)
```

##### Step 2（iii）解释

这里直接对 $x$ 求平方，没有减均值，这就是“不去均值”。`dim=-1` 让输入可以是任意 `[...,D]`，`keepdim=True` 保证后面能广播。

##### Step 3（i）公式：计算 RMS

$$
\mathrm{RMS}(x)=\sqrt{m+\epsilon}
$$

##### Step 3（ii）代码

```python
# 加 eps 后开根号：(..., 1) → (..., 1)
rms = torch.sqrt(ms + self.eps)
```

##### Step 3（iii）解释

$\epsilon$ 放在根号内，分母有下限 $\sqrt{\epsilon}$，同时根号的导数也保持有限。Follow-up 2 用代码比较了两种位置。

##### Step 4（i）公式：缩放并乘 $\gamma$

$$
\mathrm{RMSNorm}(x)_l=\gamma_l\frac{x_l}{\mathrm{RMS}(x)}
$$

##### Step 4（ii）代码

```python
# 广播：(D,) * (..., D) / (..., 1) → (..., D)
return self.gamma * x / rms
```

##### Step 4（iii）解释

$\gamma$ 按最后一维广播，输出形状与输入相同。初始 $\gamma=\mathbf{1}$ 时，输出的平方均值约等于 1。

##### 5️⃣ 复杂度

每个向量上只有一次 $O(D)$ 的求和和一次逐元素相除，设 $N$ 为最后一维之外的元素总数：

$$
\text{时间}=O(ND),\quad \text{额外内存}=O(ND),\quad \text{参数量}=D
$$

相比 LayerNorm，RMSNorm 少一次减均值，也少一组 $\beta$。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：RMSNorm 与 LayerNorm 在数学上最关键的区别是什么

最关键的区别是 RMSNorm 不减均值，也就没有平移不变性。两者的尺度量满足：

$$
\mathrm{RMS}^2=\sigma^2+\mu^2\quad(\epsilon=0)
$$

因此 $\mu=0$ 时两者相同。LayerNorm 同时具有尺度不变性和平移不变性，RMSNorm 只有尺度不变性，并且不含 $\beta$。

```python
v = torch.randn(1000)
# 恒等式：平方均值 = 总体方差 + 均值平方
print("恒等式差:", (v.pow(2).mean() - v.var(unbiased=False)
                    - v.mean() ** 2).abs().item())
x4 = torch.randn(1, 4)
ln = nn.LayerNorm(4, elementwise_affine=False)
rn = RMSNorm(4)
# 尺度不变性：两者乘 5 后输出都基本不变
print("尺度 LN/RMS:", (ln(x4 * 5) - ln(x4)).abs().max().item(),
      (rn(x4 * 5) - rn(x4)).abs().max().item())
# 平移不变性：LN 基本不变，RMS 明显改变
print("平移 LN/RMS:", (ln(x4 + 5) - ln(x4)).abs().max().item(),
      (rn(x4 + 5) - rn(x4)).abs().max().item())
```

##### Follow-up 2：`eps` 放在平方均值开根号的什么位置

主实现放在根号内，这也是 Llama 的 RMSNorm 写法：

$$
\text{根号内：}\ \sqrt{m+\epsilon},\qquad \text{根号外：}\ \sqrt{m}+\epsilon
$$

全零输入时，两者的前向都不会除零，差别在反向传播。根号外写法在 $m=0$ 处的导数 $1/(2\sqrt{m})$ 为 inf，再乘以 $\partial m/\partial x=0$ 得到 NaN。根号内写法的分母下限是 $\sqrt{\epsilon}$，梯度有限。

```python
def rms_outside(x, eps=1e-8):
    # 根号外写法：(..., D) → (..., 1)，sqrt 在 0 处导数为 inf
    return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + eps)

for name, fn in (("根号内", RMSNorm(4)), ("根号外", rms_outside)):
    z = torch.zeros(1, 4, requires_grad=True)
    y = fn(z)
    y.sum().backward()
    print(name, "输出:", y.abs().max().item(), "梯度:", z.grad[0].tolist())
```

打印结果中，根号内的梯度约为 $1/\sqrt{\epsilon}=10^4$，根号外为 NaN。$\epsilon$ 也决定下限大小：$\epsilon=10^{-8}$ 在根号内对应分母下限 $10^{-4}$。

##### Follow-up 3：低精度训练中哪些累计计算建议上转 FP32

建议上转的是平方、求和（求均值内部的累加）和开根号这三步。FP16 的最大有限值是 65504，最小正数约 $6\times10^{-8}$，因此有三种失败方式：

- 单值上溢：$|x|\ge 256$ 时平方超过上限，变成 inf，归一化结果被压成 0。
- 累加上溢：单个平方不大，但 $D$ 很大时累加和超过上限。
- 下溢：$|x|\approx10^{-4}$ 时平方下溢为 0，$\epsilon=10^{-8}$ 也下溢为 0，分母为 0，输出变成 inf。

修复方法是 Llama 的写法：先 `x.float()` 算完整个归一化，再 `type_as(x)` 转回。

```python
def rms_fp16_safe(x, gamma, eps=1e-8):
    # 上转 FP32：(..., D) 算平方、求均值、开根号
    x32 = x.float()
    rms = torch.sqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    # 归一化结果转回输入精度，再乘 gamma，形状不变
    return (x32 / rms).type_as(x) * gamma

g4 = torch.ones(4).half()
# 单值上溢：x=300，平方为 inf，FP16 结果被压成 0
big = torch.full((4,), 300.0).half()
ms = big.pow(2).mean(-1, keepdim=True)
print("FP16 上溢:", (big / ms.sqrt()).tolist())
print("FP32 修复:", rms_fp16_safe(big, g4).tolist())
# 累加上溢：单个平方是 25，4096 个累加超过 65504
many = torch.full((4096,), 5.0).half()
print("平方和 FP16/FP32:", many.pow(2).sum().item(),
      many.float().pow(2).sum().item())
# 下溢：x=1e-4，平方和 eps 都变成 0，分母为 0，FP16 得到 inf
tiny = torch.full((4,), 1e-4).half()
ms = tiny.pow(2).mean(-1, keepdim=True)
eps16 = torch.tensor(1e-8).half()
print("FP16 下溢:", (tiny / (ms + eps16).sqrt()).tolist())
print("FP32 修复:", rms_fp16_safe(tiny, g4).tolist())
```

#### Q009｜手写 ReLU / SwiGLU

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-08（发布日）
- **题目：** 写出 ReLU 和 SwiGLU 的前向逻辑，不直接调用 `nn.SwiGLU`。
- **输入输出 / 约束：** ReLU 输入 `x`；SwiGLU 输入 `[B,T,D]`，经 gate/up/down 投影返回 `[B,T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. SwiGLU 的 gate 是 `silu(gate)*up` 还是 `silu(up)*gate`？
  2. SwiGLU 和传统两层 FFN 在参数量上怎样比较？
  3. 反向传播对负输入及接近零的输入分别有什么数值特性？
- **来源：** [牛客：字节大模型算法一二三面 RMSNorm/SwiGLU/Cross-Attention](https://www.nowcoder.com/feed/main/detail/ef14567a048e4e3fa30cbe28eb5c59e3)

##### 1️⃣ Dense Opening

ReLU 是逐元素截断 $\max(x,0)$，SwiGLU 是带门控的前馈层，结构是“gate/up 两路投影 → $\mathrm{SiLU}(\text{gate})\odot\text{up}$ → down 投影”，数学核心对象是逐元素门控信号 $\mathrm{SiLU}(zW_g)\in\mathbb{R}^{B\times T\times D_{ff}}$。PyTorch 没有 `nn.SwiGLU` 这个模块，这里按题面用 gate/up/down 三个线性层手写。

##### 2️⃣ Global Equations

ReLU 和 SiLU（也称 Swish，$\beta=1$）都是逐元素函数：

$$
\mathrm{ReLU}(x)=\max(x,0),\quad \mathrm{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$

主目标，门控结果经 down 投影回到 $D$ 维：

$$
\mathrm{SwiGLU}(z)=\big(\mathrm{SiLU}(zW_g)\odot zW_u\big)W_d,\quad W_g,W_u\in\mathbb{R}^{D\times D_{ff}},\ W_d\in\mathbb{R}^{D_{ff}\times D}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def relu(x: Tensor) -> Tensor:
    # 逐元素取 max(x, 0)，形状不变
    return torch.where(x > 0, x, torch.zeros_like(x))
def silu(x: Tensor) -> Tensor:
    # 逐元素 x * sigmoid(x)，形状不变
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, D: int, D_ff: int):
        super().__init__()
        # 门控投影：D → D_ff
        self.w_gate = nn.Linear(D, D_ff, bias=False)
        # 数值投影：D → D_ff
        self.w_up = nn.Linear(D, D_ff, bias=False)
        # 输出投影：D_ff → D
        self.w_down = nn.Linear(D_ff, D, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        # 门控信号：(B, T, D) → (B, T, D_ff)
        gate = silu(self.w_gate(z))
        # 数值信号：(B, T, D) → (B, T, D_ff)
        up = self.w_up(z)
        # 逐元素相乘后投影：(B, T, D_ff) → (B, T, D)
        return self.w_down(gate * up)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def relu_np(x):
    # 逐元素取 max(x, 0)，形状不变
    return np.where(x > 0, x, 0.0)

def silu_np(x):
    # 逐元素 x / (1 + e^{-x})，等于 x * sigmoid(x)
    return x / (1.0 + np.exp(-x))

def swiglu_np(z, w_gate, w_up, w_down):
    # 门控信号：(B, T, D) @ (D, D_ff) → (B, T, D_ff)
    gate = silu_np(z @ w_gate)
    # 数值信号：(B, T, D) @ (D, D_ff) → (B, T, D_ff)
    up = z @ w_up
    # 逐元素相乘后投影：(B, T, D_ff) @ (D_ff, D) → (B, T, D)
    return (gate * up) @ w_down
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
v = torch.tensor([-2.0, 0.0, 3.0])
# 手算 ReLU：[0, 0, 3]；手算 SiLU(1) = 1 / (1 + e^{-1}) = 0.7311
print("ReLU:", relu(v).tolist(), relu_np(v.numpy()).tolist())
print("SiLU(1):", silu(torch.tensor(1.0)).item())
# 与官方 F.relu、F.silu 对齐
print("官方差异:", (relu(v) - F.relu(v)).abs().max().item(),
      (silu(v) - F.silu(v)).abs().max().item())
B, T, D, D_ff = 2, 5, 8, 16
z = torch.randn(B, T, D)
swi = SwiGLU(D, D_ff)
# Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
w = [m.weight.detach().T.numpy()
     for m in (swi.w_gate, swi.w_up, swi.w_down)]
gap = swi(z).detach().numpy() - swiglu_np(z.numpy(), *w)
print("shape:", swi(z).shape)
print("后端差异:", abs(gap).max())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def relu(x: Tensor) -> Tensor:
    pass

def silu(x: Tensor) -> Tensor:
    pass

class SwiGLU(nn.Module):
    def __init__(self, D: int, D_ff: int):
        pass
    def forward(self, z: Tensor) -> Tensor:
        pass
```

##### Step 1（i）公式：ReLU

$$
\mathrm{ReLU}(x)=
\begin{cases}
x, & x>0\\
0, & x\le0
\end{cases}
$$

##### Step 1（ii）代码

```python
# 逐元素取 max(x, 0)，形状不变
return torch.where(x > 0, x, torch.zeros_like(x))
```

##### Step 1（iii）解释

`torch.where` 按条件逐元素选取，与上面的分段公式一一对应。这个写法在 $x=0$ 处的梯度为 0，与 `F.relu` 一致（Follow-up 3 中验证）。

##### Step 2（i）公式：SiLU

$$
\mathrm{SiLU}(x)=x\cdot\sigma(x)
$$

##### Step 2（ii）代码

```python
# 逐元素 x * sigmoid(x)，形状不变
return x * torch.sigmoid(x)
```

##### Step 2（iii）解释

SiLU 处处光滑。它在负半轴不恒为 0，最小值约为 $-0.2785$，在 $x\approx-1.278$ 处取到。

##### Step 3（i）公式：定义三个投影

$$
W_g,W_u\in\mathbb{R}^{D\times D_{ff}},\quad W_d\in\mathbb{R}^{D_{ff}\times D}
$$

##### Step 3（ii）代码

```python
super().__init__()
self.w_gate = nn.Linear(D, D_ff, bias=False)
self.w_up = nn.Linear(D, D_ff, bias=False)
self.w_down = nn.Linear(D_ff, D, bias=False)
```

##### Step 3（iii）解释

gate 和 up 必须同为 $D_{ff}$ 维才能逐元素相乘。down 把维度投回 $D$，所以输入输出都是 `[B,T,D]`。这里沿用 Llama 系列的做法，不带 bias。

##### Step 4（i）公式：计算门控信号和数值信号

$$
G=\mathrm{SiLU}(zW_g),\quad U=zW_u,\quad G,U\in\mathbb{R}^{B\times T\times D_{ff}}
$$

##### Step 4（ii）代码

```python
# 门控信号：(B, T, D) → (B, T, D_ff)
gate = silu(self.w_gate(z))
# 数值信号：(B, T, D) → (B, T, D_ff)
up = self.w_up(z)
```

##### Step 4（iii）解释

激活函数只作用在 gate 这一路，up 这一路保持线性。线性层只作用在最后一维，所以每个 token 独立计算。

##### Step 5（i）公式：门控相乘并投影

$$
\mathrm{SwiGLU}(z)=(G\odot U)W_d
$$

##### Step 5（ii）代码

```python
# 逐元素相乘后投影：(B, T, D_ff) → (B, T, D)
return self.w_down(gate * up)
```

##### Step 5（iii）解释

$\odot$ 是逐元素乘法，用 `*`，不能写成 `@`。这个乘积是两路投影之间的交互项，单路激活的 FFN 没有这一项。

##### 5️⃣ 复杂度

ReLU 和 SiLU 是逐元素操作，时间为 $O(N)$，$N$ 是元素总数。SwiGLU 的主项来自三次投影：

$$
\text{时间}=O(BTDD_{ff}),\quad \text{激活显存}=O(BTD_{ff}),\quad \text{参数量}=3DD_{ff}
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：SwiGLU 的 gate 是 `silu(gate)*up` 还是 `silu(up)*gate`

标准写法是 `silu(gate) * up`，SiLU 作用在 gate 这一路（Shazeer, 2020, GLU Variants Improve Transformer）。这两路只是命名不同，结构对称：交换两个投影的权重，`silu(up) * gate` 就与它完全相等。

$$
\mathrm{SiLU}(zW_g)\odot zW_u\ \big|_{W_g\leftrightarrow W_u}=\mathrm{SiLU}(zW_u)\odot zW_g
$$

所以函数族相同，写反后依然能训练。真正的风险在于加载已有权重，比如 Hugging Face Llama 的 `gate_proj`、`up_proj`，写反后结果就错了。

```python
def swiglu_swapped(m, z):
    # 错误写法：激活放在 up 路，(B, T, D) → (B, T, D)
    return m.w_down(silu(m.w_up(z)) * m.w_gate(z))

print("写反的差异:", (swi(z) - swiglu_swapped(swi, z)).abs().max().item())
swap = SwiGLU(D, D_ff)
with torch.no_grad():
    # 交换 gate 和 up 的权重，down 保持相同
    swap.w_gate.weight.copy_(swi.w_up.weight)
    swap.w_up.weight.copy_(swi.w_gate.weight)
    swap.w_down.weight.copy_(swi.w_down.weight)
print("交换权重后差异:",
      (swap(z) - swiglu_swapped(swi, z)).abs().max().item())
```

##### Follow-up 2：SwiGLU 和传统两层 FFN 在参数量上怎样比较

无 bias 时：

$$
P_{\text{FFN}}=2DD_{ff},\qquad P_{\text{SwiGLU}}=3DD_{ff}
$$

相同 $D_{ff}$ 下，SwiGLU 的参数是 FFN 的 1.5 倍。所以常把宽度缩到 $2/3$：传统 $D_{ff}=4D$ 对应 SwiGLU 的 $D_{ff}=8D/3$，两者都是 $8D^2$。

```python
D0 = 768
ffn = nn.Sequential(nn.Linear(D0, 4 * D0, bias=False), nn.ReLU(),
                    nn.Linear(4 * D0, D0, bias=False))
same = SwiGLU(D0, 4 * D0)
narrow = SwiGLU(D0, 8 * D0 // 3)
count = lambda m: sum(p.numel() for p in m.parameters())
# 依次是 8D^2、12D^2、8D^2
print("FFN/同宽/收窄:", count(ffn), count(same), count(narrow))
```

##### Follow-up 3：反向传播对负输入及接近零的输入分别有什么数值特性

ReLU 的导数是分段常数，SiLU 的导数处处连续：

$$
\mathrm{ReLU}'(x)=\mathbb{1}[x>0],\qquad \mathrm{SiLU}'(x)=\sigma(x)\big(1+x(1-\sigma(x))\big)
$$

- 负输入：ReLU 梯度恒为 0，单元可能“死亡”（dead ReLU）。SiLU 在中等负值处有非零梯度，$x\approx-1.278$ 处导数为 0，更左侧导数为负并趋近 0。
- 接近零：ReLU 在 0 处不可微，下面实测手写版、`F.relu` 和 `clamp(min=0)` 在 0 处的梯度都取 0，正数一侧立刻跳到 1。SiLU 的导数在 0 附近连续，约为 $0.5$，函数值为 0。

```python
pts = torch.tensor([-10.0, -1.278, -0.5, 0.0, 1e-6, 2.0])
def grad_of(fn):
    # 逐点求导：形状 (6,) → (6,)
    t = pts.clone().requires_grad_()
    fn(t).sum().backward()
    return [round(g, 5) for g in t.grad.tolist()]
print("手写 relu:", grad_of(relu))
print("F.relu:", grad_of(F.relu))
print("clamp:", grad_of(lambda t: t.clamp(min=0)))
print("手写 silu:", grad_of(silu))
print("F.silu:", grad_of(F.silu))
```

SwiGLU 对 up 路的梯度还要乘以门控值 $\mathrm{SiLU}(zW_g)$，当 gate 接近 0 时这一路的梯度也接近 0。

#### Q010｜Transformer FFN：两层线性网络

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06（面经标注月份）
- **题目：** 从零实现 Position-wise FFN，可选 GELU/ReLU，能在任意序列长度上逐 token 应用。
- **输入输出 / 约束：** 输入 `[B,T,D]`、隐藏宽度 `D_ff`；输出 `[B,T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 FFN 仍然需要，Attention 已经在混合 token 信息了？
  2. FFN 计算量如何与 Attention 的 `O(T²D)` 比较？
  3. 若用 SwiGLU 替换 GELU，参数与投影维度怎样变化？
- **来源：** [Screna：TikTok Senior MLE 自注意力、交叉熵](https://www.screna.ai/experience/4498ae13-338f-4777-97dc-36b24b0b612a)

##### 1️⃣ Dense Opening

逐位置前馈网络（Position-wise Feed-Forward Network, FFN）对每个 token 独立施加同一个两层网络，结构是“线性升维 $D\to D_{ff}$ → 激活（GELU/ReLU）→ 线性降维 $D_{ff}\to D$”，数学核心对象是作用在最后一维的映射 $f:\mathbb{R}^{D}\to\mathbb{R}^{D}$。权重在所有位置共享，所以它适用于任意序列长度。

##### 2️⃣ Global Equations

主目标，对每个位置 $t$ 独立应用同一个函数：

$$
\mathrm{FFN}(x_t)=\phi(x_tW_1+b_1)W_2+b_2,\quad W_1\in\mathbb{R}^{D\times D_{ff}},\ W_2\in\mathbb{R}^{D_{ff}\times D}
$$

激活函数 $\phi$ 可选，GELU 中的 $\Phi$ 是标准正态分布的累积分布函数：

$$
\mathrm{ReLU}(u)=\max(u,0),\quad \mathrm{GELU}(u)=u\,\Phi(u)=\frac{u}{2}\left(1+\mathrm{erf}\!\left(\frac{u}{\sqrt{2}}\right)\right)
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def gelu(u: Tensor) -> Tensor:
    # 逐元素 u * Phi(u)，形状不变
    return 0.5 * u * (1.0 + torch.erf(u / math.sqrt(2.0)))

class FFN(nn.Module):
    def __init__(self, D: int, D_ff: int, act: str = "gelu"):
        super().__init__()
        # 升维投影：D → D_ff
        self.w1 = nn.Linear(D, D_ff)
        # 降维投影：D_ff → D
        self.w2 = nn.Linear(D_ff, D)
        # 按名称选择激活函数，二者都是逐元素函数
        self.act = gelu if act == "gelu" else torch.relu

    def forward(self, x: Tensor) -> Tensor:
        # Linear 只作用在最后一维：(B, T, D) → (B, T, D_ff)
        h = self.w1(x)
        # 逐元素激活，形状不变：(B, T, D_ff)
        h = self.act(h)
        # (B, T, D_ff) → (B, T, D)
        return self.w2(h)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
# math.erf 只接受标量，向量化后逐元素计算
erf_np = np.vectorize(math.erf)
def gelu_np(u):
    # 逐元素 u * Phi(u)，形状不变
    return 0.5 * u * (1.0 + erf_np(u / np.sqrt(2.0)))

def ffn_np(x, w1, b1, w2, b2, act="gelu"):
    # (B, T, D) @ (D, D_ff) + (D_ff,) → (B, T, D_ff)
    h = x @ w1 + b1
    # 逐元素激活，形状不变：(B, T, D_ff)
    h = gelu_np(h) if act == "gelu" else np.maximum(h, 0.0)
    # (B, T, D_ff) @ (D_ff, D) + (D,) → (B, T, D)
    return h @ w2 + b2
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D, D_ff = 2, 5, 8, 32
x = torch.randn(B, T, D)
u = torch.linspace(-3, 3, 7)
print("gelu 与官方差异:", (gelu(u) - F.gelu(u)).abs().max().item())
for act in ("gelu", "relu"):
    ffn = FFN(D, D_ff, act)
    # Linear.weight 是 (out, in)，转置成 NumPy 版的 (in, out)
    p = (ffn.w1.weight.detach().T.numpy(), ffn.w1.bias.detach().numpy(),
         ffn.w2.weight.detach().T.numpy(), ffn.w2.bias.detach().numpy())
    gap = ffn(x).detach().numpy() - ffn_np(x.numpy(), *p, act)
    print(act, "后端差异:", abs(gap).max())
# 逐 token：单独跑第 2 个 token 与整段结果一致，形状 (B, 1, D)
one = ffn(x[:, [2]])
print("逐 token 差异:", (one[:, 0] - ffn(x)[:, 2]).abs().max().item())
# 位置置换等变：先打乱再 FFN，等于先 FFN 再打乱
perm = torch.randperm(T)
print("置换差异:", (ffn(x[:, perm]) - ffn(x)[:, perm]).abs().max().item())
# 任意长度：T=1 和 T=50 使用同一组权重
print(ffn(torch.randn(1, 1, D)).shape, ffn(torch.randn(1, 50, D)).shape)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
def gelu(u: Tensor) -> Tensor:
    pass

class FFN(nn.Module):
    def __init__(self, D: int, D_ff: int, act: str = "gelu"):
        pass
    def forward(self, x: Tensor) -> Tensor:
        pass
```

##### Step 1（i）公式：GELU 激活

$$
\mathrm{GELU}(u)=\frac{u}{2}\left(1+\mathrm{erf}\!\left(\frac{u}{\sqrt{2}}\right)\right)
$$

##### Step 1（ii）代码

```python
# 逐元素 u * Phi(u)，形状不变
return 0.5 * u * (1.0 + torch.erf(u / math.sqrt(2.0)))
```

##### Step 1（iii）解释

由 $\Phi(u)=\frac{1}{2}(1+\mathrm{erf}(u/\sqrt{2}))$ 直接得到。这是精确形式，与 `F.gelu` 的默认行为一致，Demo 中已对齐。

##### Step 2（i）公式：定义两个线性层和激活

$$
W_1\in\mathbb{R}^{D\times D_{ff}},\ b_1\in\mathbb{R}^{D_{ff}},\quad W_2\in\mathbb{R}^{D_{ff}\times D},\ b_2\in\mathbb{R}^{D}
$$

##### Step 2（ii）代码

```python
super().__init__()
self.w1 = nn.Linear(D, D_ff)
self.w2 = nn.Linear(D_ff, D)
# 按名称选择激活函数，二者都是逐元素函数
self.act = gelu if act == "gelu" else torch.relu
```

##### Step 2（iii）解释

参数量只与 $D$ 和 $D_{ff}$ 有关，与 $T$ 无关，这是任意长度都能运行的原因。通常取 $D_{ff}=4D$。

##### Step 3（i）公式：升维并激活

$$
h_t=\phi(x_tW_1+b_1)\in\mathbb{R}^{D_{ff}}
$$

##### Step 3（ii）代码

```python
# Linear 只作用在最后一维：(B, T, D) → (B, T, D_ff)
h = self.w1(x)
# 逐元素激活，形状不变：(B, T, D_ff)
h = self.act(h)
```

##### Step 3（iii）解释

`nn.Linear` 对三维输入只变换最后一维，等价于在每个 $t$ 上独立调用，不需要循环，token 之间也没有信息交换。

##### Step 4（i）公式：降维回到 $D$

$$
\mathrm{FFN}(x_t)=h_tW_2+b_2\in\mathbb{R}^{D}
$$

##### Step 4（ii）代码

```python
# (B, T, D_ff) → (B, T, D)
return self.w2(h)
```

##### Step 4（iii）解释

输出维度回到 $D$，所以能与残差相加。Demo 的置换测试验证了 FFN 对位置是等变的（permutation equivariant）。

##### 5️⃣ 复杂度

主项来自两次线性投影，乘法次数为 $2BTDD_{ff}$：

$$
\text{时间}=O(BTDD_{ff}),\quad \text{激活显存}=O(BTD_{ff}),\quad \text{参数量}=2DD_{ff}+D_{ff}+D
$$

时间对 $T$ 线性增长。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么 FFN 仍然需要，Attention 已经在混合 token 信息了

两者处理的方向不同。给定权重 $A$ 后，Attention 的输出是 V 的线性组合，只在 token 之间混合。$A$ 对 $x$ 是非线性的，但它只决定组合系数，输出仍落在 V 的行空间内：

$$
O_i=\sum_jA_{ij}V_j\ \Rightarrow\ \mathrm{rank}(O)\le\mathrm{rank}(V)
$$

FFN 负责在每个 token 内部做非线性特征变换，可以提升特征的表达能力。下面用秩来验证：$V$ 的秩为 1 时，Attention 输出的秩仍是 1，经 FFN 后秩上升。

```python
# 秩为 1 的 V：(T, 1) @ (1, D) → (T, D)
T1 = 6
V = torch.randn(T1, 1) @ torch.randn(1, D)
# 随机注意力权重：(T, T)，每行和为 1
A = torch.randn(T1, T1).softmax(-1)
# (T, T) @ (T, D) → (T, D)
O = A @ V
ffn = FFN(D, D_ff)
rank = torch.linalg.matrix_rank
print("rank V/O/FFN(O):", rank(V).item(), rank(O).item(),
      rank(ffn(O)).item())
```

##### Follow-up 2：FFN 计算量如何与 Attention 的 `O(T²D)` 比较

按乘法次数计，$D_{ff}=4D$ 时：

$$
\mathrm{MAC}_{\text{FFN}}=2BTDD_{ff}=8BTD^2,\quad \mathrm{MAC}_{\text{score}}=BT^2D
$$

两者相等的位置是 $T=8D$。$T<8D$ 时单层 FFN 的乘法次数更多；$T>8D$ 时注意力分数的 $T^2$ 项占主导。注意力还有 $4BTD^2$ 的投影开销（无 bias 的 Q/K/V/O），也是 $T$ 的线性项。

```python
def macs(T, D, B=1):
    # 返回 FFN、注意力分数、注意力投影的乘法次数
    return 8 * B * T * D * D, B * T * T * D, 4 * B * T * D * D

D0 = 768
# 交点 T = 8D = 6144，此时前两项应相等
for T0 in (512, 4096, 6144, 32768):
    print(T0, macs(T0, D0))
```

##### Follow-up 3：若用 SwiGLU 替换 GELU，参数与投影维度怎样变化

线性层从 2 个变为 3 个，激活从单路变成门控：

$$
\mathrm{SwiGLU}(x_t)=\big(\mathrm{SiLU}(x_tW_g)\odot x_tW_u\big)W_d
$$

无 bias 时参数量从 $2DD_{ff}$ 变为 $3DD_{ff}$。要保持参数量相近，把隐藏宽度取为 $D_{ff}'=\lfloor2D_{ff}/3\rfloor$。Llama 还会把宽度向上取整到 `multiple_of` 的倍数，所以实际参数量会略有上浮。

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, D, D_ff, multiple_of=1):
        super().__init__()
        # 宽度收窄到 2/3，再向上取整到 multiple_of 的倍数
        h = int(2 * D_ff / 3)
        h = multiple_of * ((h + multiple_of - 1) // multiple_of)
        # gate/up：D → h；down：h → D
        self.w_g = nn.Linear(D, h, bias=False)
        self.w_u = nn.Linear(D, h, bias=False)
        self.w_d = nn.Linear(h, D, bias=False)

    def forward(self, x):
        # 门控相乘：(B, T, h)，再投影 → (B, T, D)
        return self.w_d(F.silu(self.w_g(x)) * self.w_u(x))

D0, F0 = 512, 2048
count = lambda m: sum(p.numel() for p in m.parameters())
print("GELU FFN:", count(FFN(D0, F0)))
print("SwiGLU:", count(SwiGLUFFN(D0, F0)))
print("SwiGLU 取整:", count(SwiGLUFFN(D0, F0, 256)))
# 接口不变：输入输出都是 (B, T, D)
print(SwiGLUFFN(D, D_ff)(x).shape)
```

#### Q011｜Sinusoidal Position Encoding（正余弦位置编码）

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 从零生成 Transformer 绝对正余弦位置编码，并将它加到 Embedding 上。
- **输入输出 / 约束：** 输入最大长度 `T` 与维度 `D`，返回 `[T,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 奇偶维分别使用 sin/cos 的频率公式是什么？
  2. 当 `D` 是奇数时如何写成稳健实现？
  3. 为什么绝对编码不能直接等同于 RoPE 的旋转位置编码？
- **题面/来源说明：** 与字节算法岗的位置编码公式追问相邻；此处作为代码扩展，不是确认的独立现场编程题。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

正余弦位置编码（Sinusoidal Position Encoding）给每个绝对位置生成一个固定向量并加到 Embedding 上，结构是“位置索引 × 几何级数频率 → 偶数维 sin、奇数维 cos → 与 Embedding 相加”，数学核心对象是编码矩阵 $P\in\mathbb{R}^{T\times D}$。

##### 2️⃣ Global Equations

主目标是下面的编码矩阵，其中 $t$ 是位置，$i$ 是维度对编号：

$$
P_{t,2i}=\sin(t\,\omega_i),\quad P_{t,2i+1}=\cos(t\,\omega_i),\quad \omega_i=10000^{-2i/D}
$$

中间变量是相位矩阵 $\Phi$，最终与 Embedding $E\in\mathbb{R}^{B\times L\times D}$ 相加（$L\le T$）：

$$
\Phi_{t,i}=t\,\omega_i,\quad \tilde{E}_{b,t}=E_{b,t}+P_t
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
def sinusoidal_pe(T: int, D: int) -> Tensor:
    # 位置索引：(T,) → (T, 1)，留出频率维用于广播
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    # 偶数维下标 2i：(ceil(D/2),)
    idx = torch.arange(0, D, 2, dtype=torch.float32)
    # 频率 ω_i，log 空间计算：(ceil(D/2),)
    freq = torch.exp(-idx * math.log(10000.0) / D)
    # 广播相乘：(T, 1) * (ceil(D/2),) → (T, ceil(D/2))
    phase = pos * freq
    # 先分配全零编码矩阵：(T, D)
    pe = torch.zeros(T, D)
    # 写入偶数列：(T, ceil(D/2)) → pe[:, 0::2]
    pe[:, 0::2] = torch.sin(phase)
    # 截断到奇数列个数：(T, ceil(D/2)) → (T, floor(D/2))
    pe[:, 1::2] = torch.cos(phase)[:, :D // 2]
    return pe

class PositionalEncoding(nn.Module):
    def __init__(self, T: int, D: int):
        super().__init__()
        # 注册为 buffer：随设备移动，不参与训练
        self.register_buffer("pe", sinusoidal_pe(T, D))

    def forward(self, emb: Tensor) -> Tensor:
        # pe[:L]: (L, D) 广播到 emb: (B, L, D)，逐位置相加
        return emb + self.pe[:emb.size(1)]
```

##### （B）NumPy版本（苛刻面试版）

```python
import math
import numpy as np

def sinusoidal_pe_np(T, D):
    # 位置索引：(T,) → (T, 1)
    pos = np.arange(T, dtype=np.float32)[:, None]
    # 偶数维下标 2i：(ceil(D/2),)
    idx = np.arange(0, D, 2, dtype=np.float32)
    # 频率 ω_i：(ceil(D/2),)
    freq = np.exp(-idx * math.log(10000.0) / D)
    # 广播相乘：(T, 1) * (ceil(D/2),) → (T, ceil(D/2))
    phase = pos * freq
    # 先分配全零编码矩阵：(T, D)
    pe = np.zeros((T, D), dtype=np.float32)
    # 写入偶数列：(T, ceil(D/2))
    pe[:, 0::2] = np.sin(phase)
    # 截断到奇数列个数：(T, floor(D/2))
    pe[:, 1::2] = np.cos(phase)[:, :D // 2]
    return pe

def add_pe_np(emb, pe):
    # pe[:L]: (L, D) 广播到 emb: (B, L, D)
    return emb + pe[:emb.shape[1]]
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# 手算值：t=1 时前两列是 sin(1)=0.8415、cos(1)=0.5403
print(sinusoidal_pe(2, 4))
# 奇偶 D 都与 NumPy 版对齐
for D in (64, 65):
    p, n = sinusoidal_pe(128, D), sinusoidal_pe_np(128, D)
    print(D, p.shape, "后端差异:", abs(p.numpy() - n).max())
# 加到 Embedding：(2, 5, 8) 只用前 5 个位置
emb = torch.randn(2, 5, 8)
out_pt = PositionalEncoding(16, 8)(emb)
out_np = add_pe_np(emb.numpy(), sinusoidal_pe_np(16, 8))
print(out_pt.shape, "相加差异:", abs(out_pt.numpy() - out_np).max())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor
def sinusoidal_pe(T: int, D: int) -> Tensor:
    pass

class PositionalEncoding(nn.Module):
    def __init__(self, T: int, D: int):
        pass

    def forward(self, emb: Tensor) -> Tensor:
        pass
```

##### Step 1（i）公式：几何级数频率

$$
\omega_i=10000^{-2i/D}=\exp\!\left(-\frac{2i\,\ln 10000}{D}\right)
$$

##### Step 1（ii）代码

```python
# 偶数维下标 2i：(ceil(D/2),)
idx = torch.arange(0, D, 2, dtype=torch.float32)
# 频率 ω_i，log 空间计算：(ceil(D/2),)
freq = torch.exp(-idx * math.log(10000.0) / D)
```

##### Step 1（iii）解释

`idx` 直接取 $2i$，所以指数里不用再乘 2。频率从 $\omega_0=1$ 单调降到接近 $10^{-4}$。

##### Step 2（i）公式：相位矩阵

$$
\Phi=t\,\omega^\top\in\mathbb{R}^{T\times\lceil D/2\rceil}
$$

##### Step 2（ii）代码

```python
# 位置索引：(T,) → (T, 1)，留出频率维用于广播
pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
# 广播相乘：(T, 1) * (ceil(D/2),) → (T, ceil(D/2))
phase = pos * freq
```

##### Step 2（iii）解释

`unsqueeze(1)` 后位置和频率做外积，$\Phi$ 的行是位置，列是维度对编号。

##### Step 3（i）公式：奇偶维写入

$$
P_{t,2i}=\sin\Phi_{t,i},\quad P_{t,2i+1}=\cos\Phi_{t,i}
$$

##### Step 3（ii）代码

```python
# 先分配全零编码矩阵：(T, D)
pe = torch.zeros(T, D)
# 写入偶数列：(T, ceil(D/2)) → pe[:, 0::2]
pe[:, 0::2] = torch.sin(phase)
# 截断到奇数列个数：(T, ceil(D/2)) → (T, floor(D/2))
pe[:, 1::2] = torch.cos(phase)[:, :D // 2]
return pe
```

##### Step 3（iii）解释

偶数列有 $\lceil D/2\rceil$ 个，奇数列有 $\lfloor D/2\rfloor$ 个。D 为偶数时切片不起作用；D 为奇数时丢弃最后一个 cos 列。

##### Step 4（i）公式：加到 Embedding

$$
\tilde{E}_{b,t}=E_{b,t}+P_t,\quad t<L
$$

##### Step 4（ii）代码

```python
# 注册为 buffer：随设备移动，不参与训练
self.register_buffer("pe", sinusoidal_pe(T, D))
# pe[:L]: (L, D) 广播到 emb: (B, L, D)，逐位置相加
return emb + self.pe[:emb.size(1)]
```

##### Step 4（iii）解释

编码矩阵只生成一次，由所有 Batch 共享。buffer 会进入 `state_dict` 并跟随 `.to(device)`，但不会被优化器更新。

##### 5️⃣ 复杂度

生成一次编码矩阵，之后每个 Batch 只做加法：

$$
\text{生成时间}=O(TD),\quad \text{相加时间}=O(BLD),\quad \text{存储}=O(TD)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：奇偶维的 sin/cos 频率公式

同一对 $(2i,2i+1)$ 共用频率 $\omega_i$，只差 $\pi/2$ 相位，波长是 $2\pi/\omega_i$：

$$
\lambda_i=2\pi\cdot 10000^{2i/D}\in[2\pi,\ 2\pi\cdot 10000)
$$

```python
D = 512
freq = torch.exp(-torch.arange(0., D, 2) * math.log(10000.0) / D)
# D=512 时波长从约 6.28 到约 60611
print((2 * math.pi / freq)[[0, -1]])
pe = sinusoidal_pe(100, D)
# 同一维度对满足 sin^2 + cos^2 = 1
print((pe[:, 2]**2 + pe[:, 3]**2 - 1).abs().max())
```

##### Follow-up 2：D 为奇数时的稳健写法

D 为奇数时，cos 需要比 sin 少一列。不做截断时赋值会因 shape 不匹配而报错：

$$
\#\text{偶数列}=\lceil D/2\rceil,\quad \#\text{奇数列}=\lfloor D/2\rfloor
$$

```python
D = 5
idx = torch.arange(0., D, 2)
# phase: (T=4, 3)，奇数列只有 2 列
phase = torch.arange(4.).unsqueeze(1) * torch.exp(
    -idx * math.log(10000.0) / D)
pe_bad = torch.zeros(4, D)
try:
    # 未截断：把 (4, 3) 写入 (4, 2) 的切片
    pe_bad[:, 1::2] = torch.cos(phase)
except RuntimeError as e:
    print("未截断报错:", str(e)[:60])
# 主实现截断后正常，t=0 时为 [0, 1, 0, 1, 0]
print(sinusoidal_pe(4, D)[0])
```

##### Follow-up 3：绝对编码不能直接等同于 RoPE

绝对编码是加法，在 Embedding 上执行一次，经过投影后才进入注意力；RoPE 是乘法，在每层注意力的 Q、K 上旋转。忽略投影时，两种方式的注意力分数分别为：

$$
(q+P_m)^\top(k+P_n)=q^\top k+q^\top P_n+P_m^\top k+P_m^\top P_n
$$

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k
$$

前者的交叉项 $q^\top P_n$ 和 $P_m^\top k$ 依赖绝对位置。纯位置项 $P_m^\top P_n$ 在 D 为偶数时为 $\sum_i\cos((m-n)\omega_i)$，只依赖相对位置。后者的分数只依赖 $n-m$。下面用相同的相对距离 2 验证。

```python
torch.manual_seed(0)
D, T = 64, 32
P = sinusoidal_pe(T, D).double()
q, k = torch.randn(D).double(), torch.randn(D).double()
# 纯位置项：只依赖相对距离，两个值相同
print(P[1] @ P[3], P[5] @ P[7])
# 绝对编码加到 q、k 后：相对距离相同，分数不同
print((q + P[1]) @ (k + P[3]), (q + P[5]) @ (k + P[7]))

def rotate(x, t):
    # x: (D,) → 两半 (D/2,)，第 i 对是 (x1[i], x2[i])
    x1, x2 = x.chunk(2)
    # 每对旋转角 t·ω_i：(D/2,)
    a = t * torch.exp(-torch.arange(0., D, 2).double()
                      * math.log(10000.0) / D)
    # 二维旋转后拼回：(D,)
    return torch.cat((x1 * a.cos() - x2 * a.sin(),
                      x1 * a.sin() + x2 * a.cos()))
# RoPE：相对距离相同，分数相同
print(rotate(q, 1) @ rotate(k, 3), rotate(q, 5) @ rotate(k, 7))
```

#### Q012｜RoPE：旋转位置编码

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现对 Q/K 的 RoPE 旋转，要求明确半维配对方式和绝对位置索引。
- **输入输出 / 约束：** `q,k:[B,H,T,d]`，返回同形状，`d` 为偶数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 交错偶奇维和前后半维的两种布局怎样对应？
  2. 为什么 RoPE 不作用于 V？它怎样引入相对位置信息？
  3. 使用 KV cache 时 position_ids 为什么不能每步重置为 0？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

RoPE（Rotary Position Embedding）把每个 head 的 $d$ 维向量分成 $d/2$ 个二维平面，并按绝对位置旋转 Q 和 K，结构是“绝对位置 × 频率得角度 → 前后半维配对 → 二维旋转”，数学核心对象是分块旋转矩阵 $R_t$。它让 $Q$ 与 $K$ 的点积只依赖相对位置 $n-m$。

##### 2️⃣ Global Equations

位置 $t$ 的第 $i$ 个平面旋转角为：

$$
\alpha_{t,i}=t\,\theta_i,\quad \theta_i=\mathrm{base}^{-2i/d},\quad i=0,\dots,\tfrac{d}{2}-1
$$

本实现采用前后半维配对，把 $x_i$ 与 $x_{i+d/2}$ 组成一个平面：

$$
\begin{pmatrix}x'_i\\x'_{i+d/2}\end{pmatrix}
=\begin{pmatrix}\cos\alpha_{t,i}&-\sin\alpha_{t,i}\\ \sin\alpha_{t,i}&\cos\alpha_{t,i}\end{pmatrix}
\begin{pmatrix}x_i\\x_{i+d/2}\end{pmatrix}
$$

主目标是注意力分数只依赖相对位置：

$$
(R_m q)^\top(R_n k)=q^\top R_{n-m}k
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def rope_rotate(x: Tensor, pos: Tensor, base=10000.0) -> Tensor:
    # x: (B, H, T, d)，pos: (B, T) 为绝对位置
    d = x.size(-1)
    # θ_i = base^(-2i/d)：(d/2,)
    theta = base ** (-torch.arange(0, d, 2).to(x.dtype) / d)
    # (B, T, 1) * (d/2,) → 角度 (B, T, d/2)
    angle = pos.unsqueeze(-1).to(x.dtype) * theta
    # 两半共用角度：(B, T, d/2) → (B, T, d)
    angle = torch.cat((angle, angle), dim=-1)
    # 插入 head 维：(B, T, d) → (B, 1, T, d)，广播到 H
    cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
    # x: (B, H, T, d) → 前后两半 (B, H, T, d/2)
    x1, x2 = x.chunk(2, dim=-1)
    # 把 (x1, x2) 变成 (-x2, x1)，形状仍为 (B, H, T, d)
    x_rot = torch.cat((-x2, x1), dim=-1)
    # 逐元素旋转：x' = x cos + x_rot sin，(B, H, T, d)
    return x * cos + x_rot * sin

def apply_rope(q: Tensor, k: Tensor, pos: Tensor):
    # Q 和 K 用同一组位置时直接旋转，输出形状不变
    return rope_rotate(q, pos), rope_rotate(k, pos)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def rope_rotate_np(x, pos, base=10000.0):
    # x: (B, H, T, d)，pos: (B, T) 为绝对位置
    d = x.shape[-1]
    # θ_i = base^(-2i/d)：(d/2,)
    theta = base ** (-np.arange(0, d, 2).astype(x.dtype) / d)
    # (B, T, 1) * (d/2,) → 角度 (B, T, d/2)
    angle = pos[..., None].astype(x.dtype) * theta
    # 两半共用角度：(B, T, d/2) → (B, T, d)
    angle = np.concatenate((angle, angle), axis=-1)
    # 插入 head 维：(B, T, d) → (B, 1, T, d)
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    # x: (B, H, T, d) → 前后两半 (B, H, T, d/2)
    x1, x2 = np.split(x, 2, axis=-1)
    # 把 (x1, x2) 变成 (-x2, x1)，形状仍为 (B, H, T, d)
    x_rot = np.concatenate((-x2, x1), axis=-1)
    # 逐元素旋转：(B, H, T, d)
    return x * cos + x_rot * sin

def apply_rope_np(q, k, pos):
    # Q 和 K 用同一组位置时直接旋转，输出形状不变
    return rope_rotate_np(q, pos), rope_rotate_np(k, pos)
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, H, T, d = 2, 3, 5, 8
q = torch.randn(B, H, T, d, dtype=torch.float64)
k = torch.randn(B, H, T, d, dtype=torch.float64)
# 绝对位置：(T,) → (B, T)
pos = torch.arange(T).expand(B, T)
qr, kr = apply_rope(q, k, pos)
qn, kn = apply_rope_np(q.numpy(), k.numpy(), pos.numpy())
print("后端差异:", abs(qr.numpy() - qn).max(), abs(kr.numpy() - kn).max())
# 旋转保持范数，位置 0 不变
print("范数差:", (qr.norm(dim=-1) - q.norm(dim=-1)).abs().max())
print("位置 0 差:", (qr[:, :, 0] - q[:, :, 0]).abs().max())
# 手算：d=2、θ_0=1，(1, 0) 在 t=1 旋转到 (cos1, sin1)
x = torch.tensor([[[[1., 0.], [1., 0.]]]])
print(rope_rotate(x, torch.tensor([[0, 1]]))[0, 0])
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def rope_rotate(x: Tensor, pos: Tensor, base=10000.0) -> Tensor:
    pass

def apply_rope(q: Tensor, k: Tensor, pos: Tensor):
    pass
```

##### Step 1（i）公式：由绝对位置计算角度

$$
\theta_i=\mathrm{base}^{-2i/d},\quad \alpha_{t,i}=t\,\theta_i
$$

##### Step 1（ii）代码

```python
# x: (B, H, T, d)，pos: (B, T) 为绝对位置
d = x.size(-1)
# θ_i = base^(-2i/d)：(d/2,)
theta = base ** (-torch.arange(0, d, 2).to(x.dtype) / d)
# (B, T, 1) * (d/2,) → 角度 (B, T, d/2)
angle = pos.unsqueeze(-1).to(x.dtype) * theta
```

##### Step 1（iii）解释

`pos` 是每个 token 在序列中的绝对索引，形状带 Batch 维，所以 Batch 内每条样本可以有不同的位置。`arange(0, d, 2)` 直接给出 $2i$。

##### Step 2（i）公式：前后半维共用角度

$$
c=[\cos\alpha,\cos\alpha],\quad s=[\sin\alpha,\sin\alpha]
$$

##### Step 2（ii）代码

```python
# 两半共用角度：(B, T, d/2) → (B, T, d)
angle = torch.cat((angle, angle), dim=-1)
# 插入 head 维：(B, T, d) → (B, 1, T, d)，广播到 H
cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
```

##### Step 2（iii）解释

第 $i$ 维和第 $i+d/2$ 维属于同一平面，所以角度要重复一次。插入的 head 维长度为 1，表示所有 head 用同一组角度。

##### Step 3（i）公式：二维旋转

$$
x'=x\odot c+\tilde{x}\odot s,\quad \tilde{x}=[-x_2,x_1]
$$

##### Step 3（ii）代码

```python
# x: (B, H, T, d) → 前后两半 (B, H, T, d/2)
x1, x2 = x.chunk(2, dim=-1)
# 把 (x1, x2) 变成 (-x2, x1)，形状仍为 (B, H, T, d)
x_rot = torch.cat((-x2, x1), dim=-1)
# 逐元素旋转：x' = x cos + x_rot sin，(B, H, T, d)
return x * cos + x_rot * sin
```

##### Step 3（iii）解释

展开后前半维为 $x_1\cos\alpha-x_2\sin\alpha$，后半维为 $x_2\cos\alpha+x_1\sin\alpha$，正好是 2️⃣ 中的旋转矩阵。这种写法不需要构造矩阵，只做逐元素乘法。

##### Step 4（i）公式：作用于 Q 与 K

$$
Q'_{t}=R_t Q_{t},\quad K'_{t}=R_t K_{t}
$$

##### Step 4（ii）代码

```python
return rope_rotate(q, pos), rope_rotate(k, pos)
```

##### Step 4（iii）解释

Q 和 K 使用同一套旋转规则。题面的 `q,k` 长度相同，所以共用 `pos`。使用 KV cache 时两者长度不同，需要分别调用 `rope_rotate`，见 Follow-up 3。

##### 5️⃣ 复杂度

每个元素只做常数次乘加，额外内存是角度表和 Q/K 输出：

$$
\text{时间}=O(BHTd),\quad \text{额外内存}=O(BTd+BHTd)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：交错偶奇维与前后半维两种布局的对应

交错布局把 $(x_{2i},x_{2i+1})$ 配成一对，这也是复数写法的常见形式；前后半维布局把 $(x_i,x_{i+d/2})$ 配成一对。两者只差一个固定的维度置换 $\pi$：

$$
\pi=[0,\tfrac{d}{2},1,\tfrac{d}{2}+1,\dots],\quad
\mathrm{Rope}_{\mathrm{inter}}(x_\pi)=\mathrm{Rope}_{\mathrm{half}}(x)_\pi
$$

```python
def rope_complex(x, pos, base=10000.0):
    d = x.size(-1)
    theta = base ** (-torch.arange(0, d, 2).to(x.dtype) / d)
    # (B, 1, T, 1) * (d/2,) → 角度 (B, 1, T, d/2)
    angle = pos[:, None, :, None].to(x.dtype) * theta
    # (B, H, T, d) → (B, H, T, d/2, 2) → 复数 (B, H, T, d/2)
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    # 乘单位复数后还原：(B, H, T, d/2) → (B, H, T, d)
    return torch.view_as_real(z * torch.polar(
        torch.ones_like(angle), angle)).flatten(-2)
# 置换：[0, 4, 1, 5, 2, 6, 3, 7]，把前后半维布局变成交错布局
perm = torch.stack((torch.arange(d // 2),
                    torch.arange(d // 2) + d // 2), 1).flatten()
print(perm.tolist())
half = rope_rotate(q, pos)
inter = rope_complex(q[..., perm], pos)
# 置换后两种布局的结果一致
print("置换后差异:", (half[..., perm] - inter).abs().max())
# 不置换直接加载权重：布局错配，结果不同
print("不置换差异:", (half - rope_complex(q, pos)).abs().max())
```

两种布局本身没有优劣。Q/K 投影权重从一种实现转到另一种实现时，必须按 $\pi$ 做转换。

##### Follow-up 2：为什么 RoPE 不作用于 V，它怎样引入相对位置

旋转矩阵是正交矩阵，同一平面的两个旋转相乘后，角度只与差值有关：

$$
(R_m q)^\top(R_n k)=q^\top R_m^\top R_n k=q^\top R_{n-m}k
$$

相对位置信息在计算注意力分数时进入，并通过权重 $A$ 影响输出。V 只是被聚合的内容，不进入分数。若旋转 V，输出 $\sum_j A_{ij}R_jv_j$ 依赖绝对位置 $j$，序列整体平移后输出会变化。

```python
v = torch.randn(B, H, T, d, dtype=torch.float64)
def attn(pos, rotate_v):
    qr, kr = apply_rope(q, k, pos)
    # (B, H, T, d) @ (B, H, d, T) → 分数 (B, H, T, T)
    s = qr @ kr.transpose(-2, -1) / d**0.5
    vr = rope_rotate(v, pos) if rotate_v else v
    # (B, H, T, T) @ (B, H, T, d) → (B, H, T, d)
    return s, s.softmax(-1) @ vr
# 所有位置平移 100，相对距离不变
s0, o0 = attn(pos, False)
s1, o1 = attn(pos + 100, False)
print("分数差:", (s0 - s1).abs().max())
print("V 不旋转的输出差:", (o0 - o1).abs().max())
_, o2 = attn(pos, True)
_, o3 = attn(pos + 100, True)
print("V 旋转的输出差:", (o2 - o3).abs().max())
```

##### Follow-up 3：KV cache 下 position_ids 为什么不能每步重置为 0

缓存里的 K 已经按历史绝对位置旋转过。新 Q 必须使用当前绝对位置 $t$，分数才是正确的相对距离。若重置为 0，新 Q 与缓存 K 的相对距离从 $j-t$ 变成 $j-0$：

$$
(R_t q_t)^\top(R_j k_j)=q_t^\top R_{j-t}k_j
$$

```python
# 只取最后一个 token 做增量 query：(B, H, 1, d)
q_new = q[:, :, -1:]
# 缓存 K 全程按绝对位置旋转：(B, H, T, d)
k_cache = rope_rotate(k, pos)
def last_scores(pos_new):
    # 新 Q 旋转后与全部 K 计算分数：(B, H, 1, T)
    qr = rope_rotate(q_new, pos_new)
    return qr @ k_cache.transpose(-2, -1)
# 全量计算的最后一行：(B, H, T, T) → (B, H, 1, T)
full = (rope_rotate(q, pos)
        @ k_cache.transpose(-2, -1))[:, :, -1:]
# 位置续接：pos_new = T-1，与全量一致
right = last_scores(pos[:, -1:])
print("续接差异:", (right - full).abs().max())
# 位置重置为 0：分数错误
wrong = last_scores(torch.zeros(B, 1, dtype=torch.long))
print("重置差异:", (wrong - full).abs().max())
```

#### Q013｜KV Cache：增量注意力实现

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给定历史 `past_k/past_v` 与新 token 的 Q/K/V，仅计算新 Query 对历史加新 Key 的注意力。
- **输入输出 / 约束：** 输入 `past_k:[B,Hkv,Tpast,d]`、新 `k/v:[B,Hkv,Tnew,d]`；输出当前步 `[B,Hq,Tnew,d]` 和更新缓存。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 cache 减少重复计算但会占用随上下文增长的显存？
  2. GQA/MQA 怎样改变 KV cache 大小？
  3. Batch 内有不同上下文长度，如何正确设置位置和 Attention Mask？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

KV Cache 把历史 token 的 K/V 存下来，新 token 只计算自己的 Q/K/V，结构是“拼接缓存 → 按 GQA 组数扩展 head → 新 Q 对全部 K 做缩放点积 → Softmax → 加权 V”，数学核心对象是形状为 $B\times H_q\times T_{new}\times T_{all}$ 的矩形权重矩阵 $A$，其中 $T_{all}=T_{past}+T_{new}$。

##### 2️⃣ Global Equations

缓存沿时间维拼接：

$$
K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]\in\mathbb{R}^{B\times H_{kv}\times T_{all}\times d}
$$

设 $g=H_q/H_{kv}$，每组 KV 重复 $g$ 次，得到 $\tilde{K},\tilde{V}$。主目标是只计算新 Query 的注意力：

$$
O=\mathrm{softmax}\!\left(\frac{Q\tilde{K}^\top}{\sqrt{d}}\right)\tilde{V}\in\mathbb{R}^{B\times H_q\times T_{new}\times d}
$$

因果约束：第 $i$ 个新 token 可以看到全部历史，以及新 token 中不晚于自己的部分：

$$
M_{ij}=\mathrm{True}\iff j\le T_{past}+i
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
from torch import Tensor

def cached_attention(q: Tensor, k: Tensor, v: Tensor,
                     past_k: Tensor = None, past_v: Tensor = None,
                     mask: Tensor = None):
    # 历史长度，首步没有缓存时为 0
    t_past = 0 if past_k is None else past_k.size(2)
    # 沿时间维拼接 → (B, Hkv, Tpast+Tnew, d)
    k_all = k if past_k is None else torch.cat((past_k, k), dim=2)
    # V 与 K 同样拼接 → (B, Hkv, Tpast+Tnew, d)
    v_all = v if past_v is None else torch.cat((past_v, v), dim=2)
    # 每组 KV 被几个 Query head 共享
    g = q.size(1) // k.size(1)
    # (B, Hkv, Tall, d) → (B, Hq, Tall, d)，缓存仍只存 Hkv 份
    k_exp = k_all.repeat_interleave(g, dim=1)
    # V 同样扩展 → (B, Hq, Tall, d)
    v_exp = v_all.repeat_interleave(g, dim=1)
    # (B, Hq, Tnew, d) @ (B, Hq, d, Tall) → (B, Hq, Tnew, Tall)
    scores = q @ k_exp.transpose(-2, -1) / math.sqrt(q.size(-1))
    # 矩形因果掩码 (Tnew, Tall)：第 i 行可见 j <= Tpast + i
    causal = torch.ones(q.size(2), k_all.size(2), dtype=torch.bool)
    causal = causal.tril(diagonal=t_past)
    # 外部 mask 可广播到 (B, Hq, Tnew, Tall)，True=允许
    keep = causal if mask is None else causal & mask
    # False 位置填 -inf，Softmax 后权重为 0
    scores = scores.masked_fill(~keep, float("-inf"))
    # 沿 key 维归一化，A: (B, Hq, Tnew, Tall)
    A = scores.softmax(dim=-1)
    # (B, Hq, Tnew, Tall) @ (B, Hq, Tall, d) → (B, Hq, Tnew, d)
    out = A @ v_exp
    return out, (k_all, v_all)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def softmax_np(x, axis=-1):
    # 减最大值防止 exp 上溢；-inf 映射为权重 0
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def cached_attention_np(q, k, v, past_k=None, past_v=None, mask=None):
    # 历史长度，首步没有缓存时为 0
    t_past = 0 if past_k is None else past_k.shape[2]
    # 沿时间维拼接 → (B, Hkv, Tall, d)
    k_all = k if past_k is None else np.concatenate((past_k, k), 2)
    v_all = v if past_v is None else np.concatenate((past_v, v), 2)
    # 每组 KV 被几个 Query head 共享
    g = q.shape[1] // k.shape[1]
    # (B, Hkv, Tall, d) → (B, Hq, Tall, d)
    k_exp, v_exp = k_all.repeat(g, axis=1), v_all.repeat(g, axis=1)
    # (B, Hq, Tnew, d) @ (B, Hq, d, Tall) → (B, Hq, Tnew, Tall)
    scores = q @ k_exp.swapaxes(-2, -1) / np.sqrt(q.shape[-1])
    # 矩形因果掩码 (Tnew, Tall)：第 i 行可见 j <= Tpast + i
    causal = np.tril(np.ones((q.shape[2], k_all.shape[2]), bool), t_past)
    # 外部 mask 可广播到 (B, Hq, Tnew, Tall)，True=允许
    keep = causal if mask is None else causal & mask
    # False 位置填 -inf
    scores = np.where(keep, scores, -np.inf)
    # (B, Hq, Tnew, Tall) @ (B, Hq, Tall, d) → (B, Hq, Tnew, d)
    out = softmax_np(scores, axis=-1) @ v_exp
    return out, (k_all, v_all)
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, Hq, Hkv, T, d = 2, 4, 2, 7, 8
dt = torch.float64
q = torch.randn(B, Hq, T, d, dtype=dt)
k = torch.randn(B, Hkv, T, d, dtype=dt)
v = torch.randn(B, Hkv, T, d, dtype=dt)
# 全量结果：T 个 Query 一次算完 → (B, Hq, T, d)
full, _ = cached_attention(q, k, v)
# 增量：块长 1,2,1,3，首次无缓存；NumPy 版同步累积
pk = pv = npk = npv = None
outs, gap, s = [], 0.0, 0
for n in (1, 2, 1, 3):
    # 取当前时间片：(B, H, T, d) → (B, H, n, d)
    qn, kn, vn = q[:, :, s:s+n], k[:, :, s:s+n], v[:, :, s:s+n]
    o, (pk, pv) = cached_attention(qn, kn, vn, pk, pv)
    on, (npk, npv) = cached_attention_np(
        qn.numpy(), kn.numpy(), vn.numpy(), npk, npv)
    gap = max(gap, abs(o.numpy() - on).max())
    outs.append(o)
    s += n
# 沿时间维拼回：(B, Hq, T, d)
inc = torch.cat(outs, dim=2)
print("增量与全量差:", (inc - full).abs().max().item())
print("后端差异:", gap)
print("缓存 shape:", pk.shape)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
from torch import Tensor

def cached_attention(q: Tensor, k: Tensor, v: Tensor,
                     past_k: Tensor = None, past_v: Tensor = None,
                     mask: Tensor = None):
    pass
```

##### Step 1（i）公式：更新缓存

$$
K_{all}=[K_{past};K_{new}],\quad V_{all}=[V_{past};V_{new}]
$$

##### Step 1（ii）代码

```python
# 历史长度，首步没有缓存时为 0
t_past = 0 if past_k is None else past_k.size(2)
# 沿时间维拼接 → (B, Hkv, Tpast+Tnew, d)
k_all = k if past_k is None else torch.cat((past_k, k), dim=2)
# V 与 K 同样拼接 → (B, Hkv, Tpast+Tnew, d)
v_all = v if past_v is None else torch.cat((past_v, v), dim=2)
```

##### Step 1（iii）解释

时间维是 `dim=2`，Batch 与 head 维保持对齐。Q 不进缓存，因为后续步骤只有新 token 需要作为 Query。

##### Step 2（i）公式：扩展 KV head

$$
g=\frac{H_q}{H_{kv}},\quad \tilde{K}=\mathrm{repeat}(K_{all},g),\quad \tilde{V}=\mathrm{repeat}(V_{all},g)
$$

##### Step 2（ii）代码

```python
# 每组 KV 被几个 Query head 共享
g = q.size(1) // k.size(1)
# (B, Hkv, Tall, d) → (B, Hq, Tall, d)，缓存仍只存 Hkv 份
k_exp = k_all.repeat_interleave(g, dim=1)
# V 同样扩展 → (B, Hq, Tall, d)
v_exp = v_all.repeat_interleave(g, dim=1)
```

##### Step 2（iii）解释

`repeat_interleave` 让相邻的 $g$ 个 Query head 共用同一组 KV。$g=1$ 时就是标准 MHA。返回的缓存是扩展前的版本。

##### Step 3（i）公式：矩形分数矩阵

$$
S=\frac{Q\tilde{K}^\top}{\sqrt{d}}\in\mathbb{R}^{B\times H_q\times T_{new}\times T_{all}}
$$

##### Step 3（ii）代码

```python
# (B, Hq, Tnew, d) @ (B, Hq, d, Tall) → (B, Hq, Tnew, Tall)
scores = q @ k_exp.transpose(-2, -1) / math.sqrt(q.size(-1))
```

##### Step 3（iii）解释

分数矩阵的行是新 Query，列是全部 Key，所以每步只算 $T_{new}\times T_{all}$ 个元素。全量计算则是 $T_{all}\times T_{all}$ 个。

##### Step 4（i）公式：因果与外部掩码

$$
S_{ij}\leftarrow-\infty,\quad \text{若 } j>T_{past}+i \text{ 或外部 mask 为 False}
$$

##### Step 4（ii）代码

```python
# 矩形因果掩码 (Tnew, Tall)：第 i 行可见 j <= Tpast + i
causal = torch.ones(q.size(2), k_all.size(2), dtype=torch.bool)
causal = causal.tril(diagonal=t_past)
# 外部 mask 可广播到 (B, Hq, Tnew, Tall)，True=允许
keep = causal if mask is None else causal & mask
# False 位置填 -inf，Softmax 后权重为 0
scores = scores.masked_fill(~keep, float("-inf"))
```

##### Step 4（iii）解释

`diagonal=t_past` 把下三角向右移动 $T_{past}$ 列，这正是 2️⃣ 中的条件。$T_{new}=1$ 时掩码全为 True；$T_{new}>1$ 时才真正起作用。

##### Step 5（i）公式：加权并返回缓存

$$
A=\mathrm{softmax}(S),\quad O=A\tilde{V}
$$

##### Step 5（ii）代码

```python
# 沿 key 维归一化，A: (B, Hq, Tnew, Tall)
A = scores.softmax(dim=-1)
# (B, Hq, Tnew, Tall) @ (B, Hq, Tall, d) → (B, Hq, Tnew, d)
out = A @ v_exp
return out, (k_all, v_all)
```

##### Step 5（iii）解释

输出时间维是 $T_{new}$，与题面的 `[B,Hq,Tnew,d]` 一致。把多次增量输出沿时间维拼接，应当等于全量结果，Demo 用不等长的块验证了这一点。

##### 5️⃣ 复杂度

单步主项来自分数矩阵和加权 V，缓存随上下文线性增长：

$$
\text{单步时间}=O(BH_qT_{new}T_{all}d),\quad \text{缓存}=2BH_{kv}T_{all}d
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么 cache 减少计算但显存随上下文增长

生成 $N$ 个 token 时，无缓存每步都要重算 $t\times t$ 的分数矩阵，有缓存每步只算 $1\times t$。代价是必须保留全部历史 K/V：

$$
\sum_{t=1}^{N}t^2\approx\frac{N^3}{3},\quad \sum_{t=1}^{N}t\approx\frac{N^2}{2},\quad
\text{缓存字节数}=2LBH_{kv}Nd\cdot\text{bytes}
$$

```python
def cache_cost(N, B=1, L=12, Hkv=8, d=64, bytes_per=2):
    # 整个生成过程的分数元素总数：无缓存 / 有缓存
    no_cache = sum(t * t for t in range(1, N + 1))
    with_cache = sum(t for t in range(1, N + 1))
    # 全部层的 K/V 缓存，单位 MB
    mem = 2 * L * B * Hkv * N * d * bytes_per / 1024**2
    return no_cache, with_cache, mem
# N 翻倍：分数元素比值翻倍，缓存显存也翻倍
print(cache_cost(1024))
print(cache_cost(2048))
```

每层都要保存 K/V，所以表达式含层数 $L$。上述数值按 `float16` 的 2 字节估算。

##### Follow-up 2：GQA/MQA 怎样改变 KV cache 大小

设 $H_{kv}$ 组 KV 被 $H_q$ 个 Query head 共享，缓存大小只取决于 $H_{kv}$。MQA 是 $H_{kv}=1$ 的特例：

$$
\frac{\text{缓存}_{\mathrm{GQA}}}{\text{缓存}_{\mathrm{MHA}}}=\frac{H_{kv}}{H_q}
$$

```python
Hq, T = 8, 16
q8 = torch.randn(1, Hq, 1, 32)
for hkv in (8, 2, 1):
    k0 = torch.randn(1, hkv, T, 32)
    v0 = torch.randn(1, hkv, T, 32)
    o, (ck, cv) = cached_attention(
        q8, k0[:, :, -1:], v0[:, :, -1:], k0[:, :, :-1], v0[:, :, :-1])
    # 输出恒为 (1, 8, 1, 32)，缓存随 hkv 线性变化
    print(hkv, o.shape, ck.numel() + cv.numel())
```

Query 投影和分数矩阵大小不变，所以单步分数计算量基本不变。

##### Follow-up 3：Batch 内上下文长度不同时的位置与 Mask

采用左填充（left padding）时，所有样本的新 token 都在最右列对齐。位置由 mask 累加得到，增量步继续加 1。外部 mask 要屏蔽 pad 列：

$$
p_{b,t}=\sum_{s\le t}m_{b,s}-1
$$

下面按位置 $p$ 取出内容相关的 Q/K/V，比较左填充 Batch 与单条样本的结果。pad 位置的 Query 只允许看自身，这样它所在行不会全部被屏蔽，也就不会产生 NaN。

```python
torch.manual_seed(0)
d, Tmax = 8, 8
# 每个位置一组向量，代表“内容 + 位置”决定的 Q/K/V
tq, tk, tv = (torch.randn(Tmax, d, dtype=dt) for _ in range(3))
def qkv(pos):
    # 按位置查表：pos: (B, T) → (B, 1, T, d)
    return tuple(t[pos].unsqueeze(1) for t in (tq, tk, tv))
# 第 0 条长度 4，第 1 条长度 2 且左填充：(B, T)
m = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
# 累加减 1 得位置，pad 处裁到 0：(B, T)
pos = (m.cumsum(1) - 1).clamp(min=0)
print(pos.tolist())
# pad 列不可见，但 pad 行可见自身：(B, 1, 4, 4)
keep = m.bool()[:, None, None] | torch.eye(4, dtype=torch.bool)
_, cache = cached_attention(*qkv(pos), mask=keep)
# 增量一步：每条样本位置各加 1，(B, 1)
pos_new = pos[:, -1:] + 1
# mask 新增一列：(B, 4) → (B, 5)，再扩到 (B, 1, 1, 5)
m2 = torch.cat((m, torch.ones(2, 1, dtype=m.dtype)), 1)
out_b, _ = cached_attention(*qkv(pos_new), *cache,
                            mask=m2.bool()[:, None, None])
# 第 1 条单独计算（无 pad）：前 2 个位置缓存，第 3 个增量
_, c1 = cached_attention(*qkv(torch.tensor([[0, 1]])))
out_1, _ = cached_attention(*qkv(torch.tensor([[2]])), *c1)
print("Batch 与单条差:", (out_b[1] - out_1[0]).abs().max().item())
# 错误做法：每条都用同一个位置 4
out_w, _ = cached_attention(*qkv(torch.tensor([[4], [4]])), *cache,
                            mask=m2.bool()[:, None, None])
print("位置错误的差:", (out_w[1] - out_1[0]).abs().max().item())
```

#### Q014｜Online Softmax（FlashAttention 核心子问题）

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现分块处理 logits 的稳定 Softmax 累积，不需要先存储完整概率矩阵。
- **输入输出 / 约束：** 输入连续的 score blocks；维护 max、exp sum、加权值 accumulator。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何合并两个 block 的 running max、分母与加权和？
  2. 为什么不能分别对每块 Softmax 后直接取平均？
  3. 和普通 Attention 在中间矩阵显存上的差异是什么？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

Online Softmax 按块读取同一行 Query 的 logits，边读边聚合 Value，全程不保存完整概率矩阵，结构是“更新 running max → 重新缩放旧状态 → 累加当前块 → 最后相除”，数学核心对象是三元状态 $(m,\ell,o)$，分别是 running max、缩放后的 exp sum 和加权值 accumulator。这是 FlashAttention 分块计算注意力的核心子问题。

##### 2️⃣ Global Equations

主目标是下面的注意力输出，这里 $S_{i\cdot}$ 被切成块 $S^{(1)},\dots,S^{(J)}$：

$$
o_i=\sum_{l}\frac{\exp(S_{il})}{\sum_{r}\exp(S_{ir})}V_l
$$

分子分母同除 $e^{m}$，状态的不变量是：

$$
\ell^{(j)}=\sum_{l\in\text{前 }j\text{ 块}}e^{S_{il}-m^{(j)}},\quad
o^{(j)}=\sum_{l\in\text{前 }j\text{ 块}}e^{S_{il}-m^{(j)}}V_l
$$

逐块递推，最终输出 $o^{(J)}/\ell^{(J)}$：

$$
m^{(j)}=\max\bigl(m^{(j-1)},\max_l S^{(j)}_{il}\bigr),\quad
c=e^{m^{(j-1)}-m^{(j)}},\quad
\ell^{(j)}=c\,\ell^{(j-1)}+\sum_l P_{il},\quad
o^{(j)}=c\,o^{(j-1)}+P V^{(j)}
$$

其中 $P_{il}=e^{S^{(j)}_{il}-m^{(j)}}$。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def online_step(s: Tensor, v: Tensor, m: Tensor, l: Tensor, o: Tensor):
    # s: (B, N, Tj)，v: (B, Tj, d)，m/l: (B, N, 1)，o: (B, N, d)
    # 本块每行最大值：(B, N, Tj) → (B, N, 1)，再更新 running max
    m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
    # 旧状态的缩放系数，指数 <= 0 所以 c <= 1：(B, N, 1)
    c = torch.exp(m - m_new)
    # 当前块未归一化权重，指数 <= 0：(B, N, Tj)
    p = torch.exp(s - m_new)
    # 按行求和后累加分母：(B, N, Tj) → (B, N, 1)
    l_new = c * l + p.sum(dim=-1, keepdim=True)
    # (B, N, Tj) @ (B, Tj, d) → (B, N, d)，累加分子
    o_new = c * o + p @ v
    return m_new, l_new, o_new

def online_softmax(score_blocks: list, value_blocks: list) -> Tensor:
    # 由首块取 Batch 和 Query 数
    B, N, _ = score_blocks[0].shape
    # Value 维与数据类型
    d, dt = value_blocks[0].size(-1), score_blocks[0].dtype
    # 空集的 max 为 -inf：(B, N, 1)
    m = torch.full((B, N, 1), float("-inf"), dtype=dt)
    # 分母从 0 开始：(B, N, 1)
    l = torch.zeros(B, N, 1, dtype=dt)
    # 分子从 0 开始：(B, N, d)
    o = torch.zeros(B, N, d, dtype=dt)
    # 逐块更新状态；状态大小与块数无关
    for s, v in zip(score_blocks, value_blocks):
        m, l, o = online_step(s, v, m, l, o)
    # 最终归一化：(B, N, d) / (B, N, 1) → (B, N, d)
    return o / l
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def online_step_np(s, v, m, l, o):
    # s: (B, N, Tj)，v: (B, Tj, d)，m/l: (B, N, 1)，o: (B, N, d)
    # 本块每行最大值：(B, N, Tj) → (B, N, 1)，再更新 running max
    m_new = np.maximum(m, s.max(axis=-1, keepdims=True))
    # 旧状态缩放系数 c <= 1：(B, N, 1)
    c = np.exp(m - m_new)
    # 当前块未归一化权重：(B, N, Tj)
    p = np.exp(s - m_new)
    # 累加分母：(B, N, 1)
    l_new = c * l + p.sum(axis=-1, keepdims=True)
    # (B, N, Tj) @ (B, Tj, d) → (B, N, d)，累加分子
    o_new = c * o + p @ v
    return m_new, l_new, o_new

def online_softmax_np(score_blocks, value_blocks):
    # 由首块取 Batch 和 Query 数；d 取自 Value 维
    B, N, _ = score_blocks[0].shape
    d = value_blocks[0].shape[-1]
    # 初始状态：空集的 max 为 -inf，分母、分子为 0
    m, l = np.full((B, N, 1), -np.inf), np.zeros((B, N, 1))
    o = np.zeros((B, N, d))
    # 逐块更新状态
    for s, v in zip(score_blocks, value_blocks):
        m, l, o = online_step_np(s, v, m, l, o)
    # 最终归一化：(B, N, d)
    return o / l
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
dt = torch.float64
# logits 平移 1000：不减 max 时 exp 直接上溢
S = torch.randn(2, 3, 10, dtype=dt) * 5 + 1000
V = torch.randn(2, 10, 4, dtype=dt)
# 沿 key 维切成不等长块，每块 (B, N, Tj)，(B, Tj, d)
sizes = [1, 4, 2, 3]
ss, vs = S.split(sizes, dim=-1), V.split(sizes, dim=1)
# 一次性 Softmax 作为基准：(B, N, d)
ref = S.softmax(dim=-1) @ V
out_pt = online_softmax(ss, vs)
out_np = online_softmax_np([s.numpy() for s in ss],
                           [v.numpy() for v in vs])
print("分块与一次性差:", (out_pt - ref).abs().max().item())
print("后端差异:", abs(out_pt.numpy() - out_np).max())
# 不减 max 的分母：exp(1000) 已上溢为 inf
print("naive 分母:", torch.exp(S).sum(-1)[0, 0].item())
# causal 场景：后面块被 mask 为 -inf，结果仍与一次性对齐
tri = torch.ones(3, 10, dtype=torch.bool).tril(4)
S2 = S.masked_fill(~tri, float("-inf"))
out2 = online_softmax(S2.split(sizes, -1), vs)
print("mask 差:", (out2 - S2.softmax(-1) @ V).abs().max().item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def online_step(s: Tensor, v: Tensor, m: Tensor, l: Tensor, o: Tensor):
    pass

def online_softmax(score_blocks: list, value_blocks: list) -> Tensor:
    pass
```

##### Step 1（i）公式：初始化状态

$$
m^{(0)}=-\infty,\quad \ell^{(0)}=0,\quad o^{(0)}=\mathbf{0}
$$

##### Step 1（ii）代码

```python
# 由首块取 Batch 和 Query 数
B, N, _ = score_blocks[0].shape
# Value 维与数据类型
d, dt = value_blocks[0].size(-1), score_blocks[0].dtype
# 空集的 max 为 -inf：(B, N, 1)
m = torch.full((B, N, 1), float("-inf"), dtype=dt)
# 分母从 0 开始：(B, N, 1)
l = torch.zeros(B, N, 1, dtype=dt)
# 分子从 0 开始：(B, N, d)
o = torch.zeros(B, N, d, dtype=dt)
```

##### Step 1（iii）解释

首块时 $m^{(0)}-m^{(1)}=-\infty$，所以 $c=0$，空状态不影响结果。本实现要求首块的每行至少有一个未 mask 的 key。causal mask 下首块含第 0 个 key，满足这个条件；若首块整行都是 $-\infty$，指数中会出现 $-\infty-(-\infty)$ 并得到 NaN。

##### Step 2（i）公式：更新 running max

$$
m^{(j)}=\max\bigl(m^{(j-1)},\max_l S^{(j)}_{il}\bigr)
$$

##### Step 2（ii）代码

```python
# 本块每行最大值：(B, N, Tj) → (B, N, 1)，再更新 running max
m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
```

##### Step 2（iii）解释

max 按 Query 分别维护。保留 `keepdim` 后，可以直接广播到 `(B, N, Tj)`。

##### Step 3（i）公式：重新缩放旧状态

$$
c=e^{m^{(j-1)}-m^{(j)}}\le 1
$$

##### Step 3（ii）代码

```python
# 旧状态的缩放系数，指数 <= 0 所以 c <= 1：(B, N, 1)
c = torch.exp(m - m_new)
```

##### Step 3（iii）解释

旧状态的基准是 $e^{m^{(j-1)}}$，换成新基准 $e^{m^{(j)}}$ 需要乘 $c$。指数不大于 0，所以这一步不会上溢。

##### Step 4（i）公式：累加当前块

$$
P_{il}=e^{S^{(j)}_{il}-m^{(j)}},\quad
\ell^{(j)}=c\,\ell^{(j-1)}+\sum_l P_{il},\quad
o^{(j)}=c\,o^{(j-1)}+PV^{(j)}
$$

##### Step 4（ii）代码

```python
# 当前块未归一化权重，指数 <= 0：(B, N, Tj)
p = torch.exp(s - m_new)
# 按行求和后累加分母：(B, N, Tj) → (B, N, 1)
l_new = c * l + p.sum(dim=-1, keepdim=True)
# (B, N, Tj) @ (B, Tj, d) → (B, N, d)，累加分子
o_new = c * o + p @ v
return m_new, l_new, o_new
```

##### Step 4（iii）解释

更新后的 $\ell,o$ 仍满足 2️⃣ 的不变量，所以可以继续递推。`p` 用完即丢，内存里只留下三个状态。

##### Step 5（i）公式：循环结束后相除

$$
o_i=\frac{o^{(J)}_i}{\ell^{(J)}_i}
$$

##### Step 5（ii）代码

```python
# 逐块更新状态；状态大小与块数无关
for s, v in zip(score_blocks, value_blocks):
    m, l, o = online_step(s, v, m, l, o)
# 最终归一化：(B, N, d) / (B, N, 1) → (B, N, d)
return o / l
```

##### Step 5（iii）解释

分子分母都用 $e^{m^{(J)}}$ 缩放，相除后缩放系数抵消，结果等于一次性 Softmax 的加权和。最大元素对应的 $P=1$，所以 $\ell\ge 1$，除法安全。

##### 5️⃣ 复杂度

设 $N$ 个 Query、$M$ 个 Key、块长 $b$。总计算量与一次性相同，中间内存不再含 $NM$ 项：

$$
\text{时间}=O(BNMd),\quad \text{中间内存}=O(BNb+BNd)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：如何合并两个 block 的 max、分母与加权和

两个状态各自以自己的 max 为基准，合并时统一换到新基准。这个合并满足结合律，所以块可以先独立计算，再按任意分组合并：

$$
m=\max(m_a,m_b),\quad
\ell=e^{m_a-m}\ell_a+e^{m_b-m}\ell_b,\quad
o=e^{m_a-m}o_a+e^{m_b-m}o_b
$$

```python
def merge(a, b):
    # a、b 均为状态 (m, l, o)
    m = torch.maximum(a[0], b[0])
    # 各自换到新基准：(B, N, 1)
    ca, cb = torch.exp(a[0] - m), torch.exp(b[0] - m)
    return m, ca * a[1] + cb * b[1], ca * a[2] + cb * b[2]

def block_state(s, v):
    # 单块从空状态开始计算：m/l: (B, N, 1)，o: (B, N, d)
    B, N, _ = s.shape
    m0 = torch.full((B, N, 1), float("-inf"), dtype=dt)
    l0 = torch.zeros(B, N, 1, dtype=dt)
    o0 = torch.zeros(B, N, v.size(-1), dtype=dt)
    return online_step(s, v, m0, l0, o0)
st = [block_state(s, v) for s, v in zip(ss, vs)]
# 二叉合并：((0,1), (2,3))
m, l, o = merge(merge(st[0], st[1]), merge(st[2], st[3]))
print("合并与一次性差:", (o / l - ref).abs().max().item())
```

##### Follow-up 2：为什么不能分别对每块 Softmax 后直接取平均

每块 Softmax 只在块内归一化，丢掉了块与块之间的相对大小，也丢掉了块长信息。取平均相当于给每块相同的总质量，而正确权重是：

$$
\text{第 }j\text{ 块的总质量}=\frac{\sum_{l\in j}e^{S_{il}}}{\sum_r e^{S_{ir}}}
$$

手算例子：logits 为 `[10, 0]` 和 `[0, 0]`，正确概率的第一项接近 1；逐块平均后第一项约为 0.5。

```python
s = torch.tensor([[10., 0., 0., 0.]], dtype=dt)
a, b = s.split(2, dim=-1)
# 每块内部 Softmax 后取平均，再除以块数：(1, 4)
wrong = torch.cat((a.softmax(-1), b.softmax(-1)), -1) / 2
print("逐块平均:", wrong)
print("一次性:", s.softmax(-1))
```

##### Follow-up 3：和普通 Attention 在中间矩阵显存上的差异

普通注意力要保存完整分数矩阵；Online Softmax 只保存当前块分数和状态。下面按元素数估算：

$$
\text{普通}=BNM,\quad \text{Online}=BNb+2BN+BNd
$$

```python
def mid_elems(N, M, b, d, B=1):
    # 普通：完整 scores；Online：当前块 + m + l + o
    return B * N * M, B * N * b + 2 * B * N + B * N * d
# N=M=4096，d=64：块越小，中间显存越低
for b in (64, 256):
    plain, online = mid_elems(4096, 4096, b, 64)
    print(b, plain, online, round(plain / online, 1))
```

FlashAttention 还把 Query 也分块，每次计算的分数只有 $b_q\times b_k$，再把状态写回显存，所以实际占用还会更低。总运算量仍是 $O(BNMd)$，节省的是中间矩阵的存储。

#### Q015｜Attention 复杂度手推并用代码核对

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（面试日；11月发布）
- **题目：** 给定 `B,T,D,H,L,D_ff`，计算 MHA、FFN、Transformer Stack 的主要运算量与 Attention Matrix 显存，可写一个估算函数。
- **输入输出 / 约束：** 输入配置字典，返回各部分 FLOPs/内存估算（注明 MACs vs FLOPs）。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么在固定 D 下增加头数 H 不必然让乘法计算量成 H 倍？
  2. 当 `T<<D` 与 `T>>D` 时哪个项主导？
  3. 如果使用 GQA，哪些项变小，哪些项基本不变？
- **题面/来源说明：** 来源主要是现场推导，工程函数属于训练扩展。
- **来源：** [牛客：字节大模型算法三面 BCE/VAE/积木堆叠](https://www.nowcoder.com/discuss/946798743163088896)

##### 1️⃣ Dense Opening

这题把 Transformer 的运算量写成配置的闭式函数，结构是“按矩阵乘法计 MACs → 累加 MHA、FFN 和 L 层 Stack → 换算 FLOPs → 单独估算 Attention Matrix 显存”，数学核心对象是 MHA 的两类主项：投影项 $4BTD^2$ 与注意力项 $2BT^2D$。核对方式是用真实张量跑一遍同样的矩阵乘法并计数。

##### 2️⃣ Global Equations

口径：一次矩阵乘法 $A_{n\times m}B_{m\times k}$ 记 $nmk$ MACs；1 MAC 含一次乘法和一次加法，所以：

$$
\text{FLOPs}\approx 2\times\text{MACs}
$$

设 $d_h=D/H$。标准 MHA 中，$H_{kv}=H$；GQA 时 $H_{kv}<H$：

$$
\mathrm{MAC}_{\mathrm{proj}}=BTD\,(2D+2H_{kv}d_h),\quad
\mathrm{MAC}_{\mathrm{score}}=\mathrm{MAC}_{\mathrm{mix}}=BHT^2d_h=BT^2D
$$

FFN 与整个 Stack 的主目标是：

$$
\mathrm{MAC}_{\mathrm{FFN}}=2BTDD_{ff},\quad
\mathrm{MAC}_{\mathrm{stack}}=L\,(\mathrm{MAC}_{\mathrm{proj}}+2BT^2D+\mathrm{MAC}_{\mathrm{FFN}})
$$

每层 Attention Matrix 的显存，其中 $b$ 是每元素字节数：

$$
\mathrm{Mem}_{\mathrm{attn}}=BHT^2\cdot b
$$

Softmax、LayerNorm、残差加法只有 $O(BHT^2)$ 或 $O(BTD)$ 量级，不计入 MACs。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn

def estimate_cost(cfg: dict, bytes_per=4, flops_per_mac=2) -> dict:
    # 配置转 int64 张量，支持广播做参数扫描
    B, T, D, H, L, D_ff = (torch.as_tensor(cfg[n], dtype=torch.int64)
                           for n in ("B", "T", "D", "H", "L", "D_ff"))
    # KV 组数：缺省时等于 H，也就是标准 MHA
    Hkv = torch.as_tensor(cfg.get("Hkv", cfg["H"]), dtype=torch.int64)
    # 每个 head 的维度
    d_h = D // H
    # Q/输出投影 D→D，K/V 投影 D→Hkv*d_h
    proj = B * T * D * (2 * D + 2 * Hkv * d_h)
    # QK^T：H 个 head，每个 T*d_h*T
    score = B * H * T * T * d_h
    # A@V：H 个 head，每个 T*T*d_h
    mix = B * H * T * T * d_h
    # 单层 MHA 的 MACs
    mha = proj + score + mix
    # FFN 两层线性：D→D_ff→D
    ffn = 2 * B * T * D * D_ff
    # L 层累加
    stack = L * (mha + ffn)
    # 每层 Attention Matrix：(B, H, T, T) 的字节数
    attn_bytes = B * H * T * T * bytes_per
    # 无 bias 的总参数量
    params = L * (2 * D * D + 2 * D * Hkv * d_h + 2 * D * D_ff)
    return dict(proj=proj, score=score, mix=mix, mha=mha, ffn=ffn,
                stack_macs=stack, stack_flops=stack * flops_per_mac,
                attn_bytes=attn_bytes, params=params)

def measure_cost(cfg: dict) -> dict:
    # 取配置，Hkv 缺省为 H
    B, T, D, H, L = (cfg[n] for n in ("B", "T", "D", "H", "L"))
    # FFN 宽度、KV 组数、head 维度、共享倍数
    D_ff, Hkv = cfg["D_ff"], cfg.get("Hkv", H)
    d_h, g, macs = D // H, H // Hkv, 0
    def mm(a, b):
        # 记账：a 的元素数 * b 的最后一维，再做真实矩阵乘法
        nonlocal macs
        macs += a.numel() * b.size(-1)
        return a @ b
    # 随机权重：Q/O (D,D)，K/V (D,Hkv*d_h)，FFN (D,D_ff)、(D_ff,D)
    W = [torch.randn(*s) for s in ((D, D), (D, Hkv * d_h),
         (D, Hkv * d_h), (D, D), (D, D_ff), (D_ff, D))]
    # 输入：(B, T, D)
    x = torch.randn(B, T, D)
    for _ in range(L):
        # (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
        q = mm(x, W[0]).reshape(B, T, H, d_h).transpose(1, 2)
        # (B, T, Hkv*d_h) → (B, Hkv, T, d_h) → (B, H, T, d_h)
        k = mm(x, W[1]).reshape(B, T, Hkv, d_h).transpose(1, 2)
        k = k.repeat_interleave(g, dim=1)
        # V 与 K 相同：(B, H, T, d_h)
        v = mm(x, W[2]).reshape(B, T, Hkv, d_h).transpose(1, 2)
        v = v.repeat_interleave(g, dim=1)
        # 分数：(B, H, T, d_h) @ (B, H, d_h, T) → (B, H, T, T)
        s = mm(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        # 权重：(B, H, T, T)，每行和为 1
        A = s.softmax(dim=-1)
        # (B, H, T, d_h) → (B, T, H, d_h) → (B, T, D)
        o = mm(A, v).transpose(1, 2).reshape(B, T, D)
        # 输出投影与残差：(B, T, D)
        x = x + mm(o, W[3])
        # FFN：(B, T, D) → (B, T, D_ff) → (B, T, D)
        x = x + mm(torch.relu(mm(x, W[4])), W[5])
    # 真实 MACs、单层 A 的字节数、L 层参数量
    return dict(stack_macs=macs, attn_bytes=A.numel() * A.element_size(),
                params=L * sum(w.numel() for w in W))
```

##### （B）NumPy版本（苛刻面试版）

```python
import math
import numpy as np

def estimate_cost_np(cfg, bytes_per=4, flops_per_mac=2):
    # 配置转 int64 数组，支持广播做参数扫描
    B, T, D, H, L, D_ff = (np.asarray(cfg[n], dtype=np.int64)
                           for n in ("B", "T", "D", "H", "L", "D_ff"))
    # KV 组数：缺省时等于 H
    Hkv = np.asarray(cfg.get("Hkv", cfg["H"]), dtype=np.int64)
    # 每个 head 的维度
    d_h = D // H
    # Q/输出投影 D→D，K/V 投影 D→Hkv*d_h
    proj = B * T * D * (2 * D + 2 * Hkv * d_h)
    # QK^T：H 个 head，每个 T*d_h*T
    score = B * H * T * T * d_h
    # A@V：H 个 head，每个 T*T*d_h
    mix = B * H * T * T * d_h
    # 单层 MHA 的 MACs
    mha = proj + score + mix
    # FFN 两层线性：D→D_ff→D
    ffn = 2 * B * T * D * D_ff
    # L 层累加
    stack = L * (mha + ffn)
    # 每层 Attention Matrix：(B, H, T, T) 的字节数
    attn_bytes = B * H * T * T * bytes_per
    # 无 bias 的总参数量
    params = L * (2 * D * D + 2 * D * Hkv * d_h + 2 * D * D_ff)
    return dict(proj=proj, score=score, mix=mix, mha=mha, ffn=ffn,
                stack_macs=stack, stack_flops=stack * flops_per_mac,
                attn_bytes=attn_bytes, params=params)

def measure_cost_np(cfg, seed=0):
    # 取配置，Hkv 缺省为 H
    B, T, D, H, L = (cfg[n] for n in ("B", "T", "D", "H", "L"))
    # FFN 宽度、KV 组数
    D_ff, Hkv = cfg["D_ff"], cfg.get("Hkv", H)
    # head 维度、共享倍数、累计 MACs
    d_h, g, macs = D // H, H // Hkv, 0
    # 固定种子
    rng = np.random.default_rng(seed)
    # float32 随机张量生成器，与 PyTorch 默认精度一致
    rand = lambda *s: rng.standard_normal(s, dtype=np.float32)
    def mm(a, b):
        # 记账后做真实矩阵乘法
        nonlocal macs
        macs += a.size * b.shape[-1]
        return a @ b
    # 随机权重：与 PyTorch 版相同的 6 个矩阵
    W = [rand(*s) for s in ((D, D), (D, Hkv * d_h), (D, Hkv * d_h),
                            (D, D), (D, D_ff), (D_ff, D))]
    # 输入：(B, T, D)
    x = rand(B, T, D)
    for _ in range(L):
        # (B, T, D) → (B, H, T, d_h)
        q = mm(x, W[0]).reshape(B, T, H, d_h).swapaxes(1, 2)
        # (B, T, Hkv*d_h) → (B, H, T, d_h)
        k = mm(x, W[1]).reshape(B, T, Hkv, d_h).swapaxes(1, 2)
        # 每组 KV 重复 g 次：(B, Hkv, T, d_h) → (B, H, T, d_h)
        k = k.repeat(g, axis=1)
        # V 与 K 相同：(B, H, T, d_h)
        v = mm(x, W[2]).reshape(B, T, Hkv, d_h).swapaxes(1, 2)
        # 每组 KV 重复 g 次：(B, Hkv, T, d_h) → (B, H, T, d_h)
        v = v.repeat(g, axis=1)
        # 分数：(B, H, T, T)
        s = mm(q, k.swapaxes(-2, -1)) / math.sqrt(d_h)
        # 减最大值后 Softmax：(B, H, T, T)
        e = np.exp(s - s.max(-1, keepdims=True))
        # 按行归一化：(B, H, T, T)
        A = e / e.sum(-1, keepdims=True)
        # (B, H, T, d_h) → (B, T, D)
        o = mm(A, v).swapaxes(1, 2).reshape(B, T, D)
        # 输出投影与残差：(B, T, D)
        x = x + mm(o, W[3])
        # FFN：(B, T, D) → (B, T, D_ff) → (B, T, D)
        x = x + mm(np.maximum(mm(x, W[4]), 0), W[5])
    return dict(stack_macs=macs, attn_bytes=A.nbytes,
                params=L * sum(w.size for w in W))
```

##### 最小可运行测试（Demo）

```python
from torch.utils.flop_counter import FlopCounterMode
torch.manual_seed(0)
cfg = dict(B=2, T=16, D=32, H=4, L=3, D_ff=128)
est = estimate_cost(cfg)
est_np = estimate_cost_np(cfg)
# 闭式估算与真实张量计数逐项比较，两后端都应完全相等
for n in ("stack_macs", "attn_bytes", "params"):
    print(n, int(est[n]), int(est_np[n]),
          measure_cost(cfg)[n], measure_cost_np(cfg)[n])
# 官方计数器：这里实跑得到 FLOPs = 2 * MACs
with FlopCounterMode(display=False) as fc:
    measure_cost(cfg)
print("官方 FLOPs:", fc.get_total_flops(), int(est["stack_flops"]))
# 官方模块核对参数量和 (B, H, T, T) 权重形状
mha = nn.MultiheadAttention(32, 4, bias=False, batch_first=True)
ffn = [nn.Linear(32, 128, bias=False), nn.Linear(128, 32, bias=False)]
n_par = sum(p.numel() for m in [mha, *ffn] for p in m.parameters())
print("官方单层参数:", n_par, int(est["params"]) // 3)
x = torch.randn(2, 16, 32)
_, w = mha(x, x, x, average_attn_weights=False)
print(w.shape, w.numel() * w.element_size(), int(est["attn_bytes"]))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn

def estimate_cost(cfg: dict, bytes_per=4, flops_per_mac=2) -> dict:
    pass

def measure_cost(cfg: dict) -> dict:
    pass
```

##### Step 1（i）公式：MHA 投影项

$$
\mathrm{MAC}_{\mathrm{proj}}=BTD\,(2D+2H_{kv}d_h),\quad d_h=D/H
$$

##### Step 1（ii）代码

```python
# 配置转 int64 张量，支持广播做参数扫描
B, T, D, H, L, D_ff = (torch.as_tensor(cfg[n], dtype=torch.int64)
                       for n in ("B", "T", "D", "H", "L", "D_ff"))
# KV 组数：缺省时等于 H，也就是标准 MHA
Hkv = torch.as_tensor(cfg.get("Hkv", cfg["H"]), dtype=torch.int64)
# 每个 head 的维度
d_h = D // H
# Q/输出投影 D→D，K/V 投影 D→Hkv*d_h
proj = B * T * D * (2 * D + 2 * Hkv * d_h)
```

##### Step 1（iii）解释

Q 投影和输出投影都是 $BTD\cdot D$。K、V 投影各是 $BTD\cdot H_{kv}d_h$。标准 MHA 的 $H_{kv}d_h=D$，所以这一项是 $4BTD^2$，与 $H$ 无关。

##### Step 2（i）公式：注意力项

$$
\mathrm{MAC}_{\mathrm{score}}=\mathrm{MAC}_{\mathrm{mix}}=BHT^2d_h
$$

##### Step 2（ii）代码

```python
# QK^T：H 个 head，每个 T*d_h*T
score = B * H * T * T * d_h
# A@V：H 个 head，每个 T*T*d_h
mix = B * H * T * T * d_h
# 单层 MHA 的 MACs
mha = proj + score + mix
```

##### Step 2（iii）解释

矩阵乘法 $(T\times d_h)(d_h\times T)$ 需要 $T^2d_h$ MACs。注意 $Hd_h=D$，所以两项之和就是 $2BT^2D$。

##### Step 3（i）公式：FFN 与 Stack

$$
\mathrm{MAC}_{\mathrm{FFN}}=2BTDD_{ff},\quad
\mathrm{MAC}_{\mathrm{stack}}=L(\mathrm{MAC}_{\mathrm{MHA}}+\mathrm{MAC}_{\mathrm{FFN}})
$$

##### Step 3（ii）代码

```python
# FFN 两层线性：D→D_ff→D
ffn = 2 * B * T * D * D_ff
# L 层累加
stack = L * (mha + ffn)
```

##### Step 3（iii）解释

FFN 对每个 token 独立计算，只与 $T$ 成线性关系。这里按整个序列重新计算（无 KV cache），层间运算量相同。

##### Step 4（i）公式：FLOPs、显存与参数量

$$
\text{FLOPs}=2\,\text{MACs},\quad
\mathrm{Mem}_{\mathrm{attn}}=BHT^2b,\quad
\text{Params}=L(2D^2+2DH_{kv}d_h+2DD_{ff})
$$

##### Step 4（ii）代码

```python
# 每层 Attention Matrix：(B, H, T, T) 的字节数
attn_bytes = B * H * T * T * bytes_per
# 无 bias 的总参数量
params = L * (2 * D * D + 2 * D * Hkv * d_h + 2 * D * D_ff)
return dict(proj=proj, score=score, mix=mix, mha=mha, ffn=ffn,
            stack_macs=stack, stack_flops=stack * flops_per_mac,
            attn_bytes=attn_bytes, params=params)
```

##### Step 4（iii）解释

`flops_per_mac` 显式写成参数，便于和不同工具对齐。部分工具把 1 MAC 记作 1 FLOP，比较数字前要先确认口径。当前环境实跑 `FlopCounterMode` 的结果是 1 MAC 记 2 FLOPs。`attn_bytes` 按一层算；每层的 A 用完即释放时，峰值占用约为一层；若输出全部层的注意力，则要乘 $L$。

##### Step 5（i）公式：用真实矩阵乘法计数

$$
\mathrm{MAC}(a@b)=|a|\cdot b_{\mathrm{last}}
$$

##### Step 5（ii）代码

```python
# 取配置，Hkv 缺省为 H
B, T, D, H, L = (cfg[n] for n in ("B", "T", "D", "H", "L"))
# FFN 宽度、KV 组数、head 维度、共享倍数
D_ff, Hkv = cfg["D_ff"], cfg.get("Hkv", H)
d_h, g, macs = D // H, H // Hkv, 0
def mm(a, b):
    # 记账：a 的元素数 * b 的最后一维，再做真实矩阵乘法
    nonlocal macs
    macs += a.numel() * b.size(-1)
    return a @ b
# 随机权重：Q/O (D,D)，K/V (D,Hkv*d_h)，FFN (D,D_ff)、(D_ff,D)
W = [torch.randn(*s) for s in ((D, D), (D, Hkv * d_h),
     (D, Hkv * d_h), (D, D), (D, D_ff), (D_ff, D))]
# 输入：(B, T, D)
x = torch.randn(B, T, D)
for _ in range(L):
    # (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
    q = mm(x, W[0]).reshape(B, T, H, d_h).transpose(1, 2)
    # (B, T, Hkv*d_h) → (B, Hkv, T, d_h) → (B, H, T, d_h)
    k = mm(x, W[1]).reshape(B, T, Hkv, d_h).transpose(1, 2)
    k = k.repeat_interleave(g, dim=1)
    # V 与 K 相同：(B, H, T, d_h)
    v = mm(x, W[2]).reshape(B, T, Hkv, d_h).transpose(1, 2)
    v = v.repeat_interleave(g, dim=1)
    # 分数：(B, H, T, d_h) @ (B, H, d_h, T) → (B, H, T, T)
    s = mm(q, k.transpose(-2, -1)) / math.sqrt(d_h)
    # 权重：(B, H, T, T)，每行和为 1
    A = s.softmax(dim=-1)
    # (B, H, T, d_h) → (B, T, H, d_h) → (B, T, D)
    o = mm(A, v).transpose(1, 2).reshape(B, T, D)
    # 输出投影与残差：(B, T, D)
    x = x + mm(o, W[3])
    # FFN：(B, T, D) → (B, T, D_ff) → (B, T, D)
    x = x + mm(torch.relu(mm(x, W[4])), W[5])
# 真实 MACs、单层 A 的字节数、L 层参数量
return dict(stack_macs=macs, attn_bytes=A.numel() * A.element_size(),
            params=L * sum(w.numel() for w in W))
```

##### Step 5（iii）解释

`mm` 把闭式中的每一项都落实到一次真实矩阵乘法，并按实际 shape 累加。Demo 中闭式、两个后端计数和官方计数器的结果完全相等。参数量也与官方模块一致。

##### 5️⃣ 复杂度

估算函数本身是 $O(1)$ 算术。被估算模型的主项与显存为：

$$
\text{MACs}=L(4BTD^2+2BT^2D+2BTDD_{ff}),\quad
\text{Attention 显存}=BHT^2b
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：固定 D 时为什么增加 H 不会让乘法计算量成 H 倍

头数翻倍时，每个 head 的维度减半，注意力项中 $H$ 和 $d_h$ 抵消：

$$
BHT^2d_h=BT^2(Hd_h)=BT^2D
$$

增长的是 Attention Matrix 的元素数 $BHT^2$，它决定 Softmax 的计算量和权重显存。若固定 $d_h$ 并让 $D$ 随 $H$ 增长，注意力项才成 $H$ 倍。

```python
base = dict(B=2, T=16, D=32, L=1, D_ff=128)
for H in (1, 2, 4, 8):
    c = {**base, "H": H}
    # MACs 不变，attn_bytes 随 H 线性增长
    print(H, measure_cost(c)["stack_macs"], measure_cost(c)["attn_bytes"])
# 固定 d_h=8：D 随 H 增长时注意力项成 H 倍
for H in (1, 2, 4):
    c = {**base, "H": H, "D": 8 * H}
    print(H, int(estimate_cost(c)["score"]))
```

##### Follow-up 2：T 远小于 D 与 T 远大于 D 时哪项主导

比较注意力项和 MHA 投影项，再把 FFN 也算进来：

$$
\frac{2BT^2D}{4BTD^2}=\frac{T}{2D},\quad
\frac{2BT^2D}{4BTD^2+2BTDD_{ff}}=\frac{T}{2D+D_{ff}}
$$

所以 $T\ll D$ 时投影和 FFN 主导，运算量对 $T$ 线性；$T\gg D$ 时注意力项主导，运算量对 $T$ 平方增长。分界点是 $T=2D+D_{ff}$，当 $D_{ff}=4D$ 时为 $T=6D$。

```python
D, D_ff = 768, 3072
# 分界点 T=6D=4608：下面第 2 行的注意力占比应为 0.5
for T in (64, 2 * D + D_ff, 65536):
    e = estimate_cost(dict(B=1, T=T, D=D, H=12, L=1, D_ff=D_ff))
    attn = e["score"] + e["mix"]
    print(T, round((attn / (e["mha"] + e["ffn"])).item(), 4))
```

##### Follow-up 3：使用 GQA 时哪些项变小，哪些项基本不变

GQA 只减少 K/V 的组数，Q 的 head 数仍为 $H$：

$$
\mathrm{MAC}^{K,V}_{\mathrm{proj}}=2BTDH_{kv}d_h,\quad
\mathrm{Params}^{K,V}=2DH_{kv}d_h,\quad
\text{KV cache}=2BH_{kv}Td_hb
$$

变小的是 K/V 投影 MACs、K/V 参数量和 KV cache。基本不变的是 Q 投影、输出投影、注意力项、FFN 和 Attention Matrix，因为 K/V 要扩展回 $H$ 组参与计算。

```python
c = dict(B=2, T=16, D=32, H=8, L=1, D_ff=128)
keys = ("proj", "score", "mix", "ffn", "attn_bytes", "params")
for hkv in (8, 2, 1):
    e = estimate_cost({**c, "Hkv": hkv})
    m = measure_cost({**c, "Hkv": hkv})
    # 只有 proj 和 params 变小；闭式与实测差为 0
    print(hkv, [int(e[n]) for n in keys],
          "实测差:", m["stack_macs"] - int(e["stack_macs"]))
```

#### Q016｜单头与多头 Attention 结果对齐测试

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 编写一个数值单元测试，将自己的 MHA 与 PyTorch 官方实现复制权重后对齐。
- **输入输出 / 约束：** 随机输入与固定权重，输出 max abs error、梯度差。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Q/K/V 合并权重的排列顺序如何对齐？
  2. Dropout、dtype、mask polarity 如何保证一致？
  3. 为什么前向接近不等于 backward 也正确？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

这题写一个数值单元测试，把自己的 MHA（Multi-Head Attention）与 `nn.MultiheadAttention` 对齐，结构是“拆分官方合并权重 → 复制到自己的投影 → 同一输入前向 → 同一损失反向 → 输出 max abs error 与梯度差”，数学核心对象是官方合并权重 $W_{in}\in\mathbb{R}^{3D\times D}$ 和两个模型的梯度差。$H=1$ 时就是单头对齐，所以测试对单头和多头共用一套代码。

##### 2️⃣ Global Equations

官方合并权重按 Q、K、V 的顺序沿第 0 维切开，每块是 $D\times D$：

$$
W_{in}=\begin{bmatrix}W_q\\W_k\\W_v\end{bmatrix},\quad
b_{in}=\begin{bmatrix}b_q\\b_k\\b_v\end{bmatrix},\quad
Q=XW_q^\top+b_q
$$

两边实现同一条计算图，其中 $\bar{O}\in\mathbb{R}^{B\times T\times D}$ 是拼回的多头输出：

$$
A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right),\quad
O=\bar{O}W_o^\top+b_o,\quad \bar{O}=\mathrm{concat}(AV)
$$

主目标是下面两个误差都接近机器精度，损失取 $\mathcal{L}=\sum O^2$：

$$
\varepsilon_{\mathrm{out}}=\max|O^{my}-O^{off}|,\quad
\varepsilon_{\mathrm{grad}}=\max_{\theta}\max|\nabla_\theta\mathcal{L}^{my}-\nabla_\theta\mathcal{L}^{off}|
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class MyMHA(nn.Module):
    def __init__(self, D: int, H: int):
        super().__init__()
        # head 数和每个 head 的维度
        self.H, self.d_h = H, D // H
        # Q/K/V/输出投影，都是 D → D，默认带 bias
        self.w_q, self.w_k = nn.Linear(D, D), nn.Linear(D, D)
        # V 与输出投影：D → D
        self.w_v, self.w_o = nn.Linear(D, D), nn.Linear(D, D)

    def split(self, x: Tensor) -> Tensor:
        # (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
        return x.view(*x.shape[:2], self.H, self.d_h).transpose(1, 2)

    def forward(self, q_in, k_in, v_in, attn_mask=None,
                key_padding_mask=None):
        # Q：(B, T, D) → (B, H, T, d_h)
        q = self.split(self.w_q(q_in))
        # K：(B, S, D) → (B, H, S, d_h)
        k = self.split(self.w_k(k_in))
        # V：(B, S, D) → (B, H, S, d_h)
        v = self.split(self.w_v(v_in))
        # (B, H, T, d_h) @ (B, H, d_h, S) → (B, H, T, S)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)
        # 与官方相同，bool mask 的 True=禁止；(T, S) 广播
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        # (B, S) → (B, 1, 1, S)，True 的 key 列被屏蔽
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(pad, float("-inf"))
        # 沿 key 维归一化：(B, H, T, S)
        A = scores.softmax(dim=-1)
        # (B, H, T, d_h) → (B, T, H, d_h) → (B, T, D)
        out = (A @ v).transpose(1, 2).flatten(2)
        return self.w_o(out), A

def copy_official(my: MyMHA, off: nn.MultiheadAttention):
    with torch.no_grad():
        # (3D, D) → 3 个 (D, D)，顺序固定为 Q、K、V
        ws = off.in_proj_weight.chunk(3)
        # (3D,) → 3 个 (D,)，顺序相同
        bs = off.in_proj_bias.chunk(3)
        for m, w, b in zip((my.w_q, my.w_k, my.w_v), ws, bs):
            # 官方与 nn.Linear 都是 (out, in) 布局，不用转置
            m.weight.copy_(w)
            m.bias.copy_(b)
        # 输出投影直接对应 out_proj
        my.w_o.weight.copy_(off.out_proj.weight)
        my.w_o.bias.copy_(off.out_proj.bias)

def grad_or_zero(p: Tensor) -> Tensor:
    # 参数没收到梯度时按 0 比较
    return torch.zeros_like(p) if p.grad is None else p.grad

def align_report(my, off, x, **masks) -> dict:
    # 清空两边累积的梯度
    my.zero_grad()
    # 官方模型同样清空
    off.zero_grad()
    # 自己的 MHA：(B, T, D)，(B, H, T, S)
    o1, a1 = my(x, x, x, **masks)
    # False：返回 (B, H, T, S)，不按 head 平均
    o2, a2 = off(x, x, x, average_attn_weights=False, **masks)
    # 两边用同一个标量损失反向
    o1.pow(2).sum().backward()
    o2.pow(2).sum().backward()
    # 3 个 (D, D) 拼回 (3D, D)，与 in_proj_weight 对齐
    g_w = torch.cat([grad_or_zero(m.weight)
                     for m in (my.w_q, my.w_k, my.w_v)])
    # 3 个 (D,) 拼回 (3D,)，与 in_proj_bias 对齐
    g_b = torch.cat([grad_or_zero(m.bias)
                     for m in (my.w_q, my.w_k, my.w_v)])
    # 每对是（自己的梯度，官方梯度）
    pairs = [(g_w, off.in_proj_weight.grad),
             (g_b, off.in_proj_bias.grad),
             (grad_or_zero(my.w_o.weight), off.out_proj.weight.grad),
             (grad_or_zero(my.w_o.bias), off.out_proj.bias.grad)]
    return dict(out=(o1 - o2).abs().max().item(),
                attn=(a1 - a2).abs().max().item(),
                grad=max((a - b).abs().max().item() for a, b in pairs))
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def split_np(x, H):
    # (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
    B, T, D = x.shape
    return x.reshape(B, T, H, D // H).swapaxes(1, 2)

def mha_np(q_in, k_in, v_in, w_in, b_in, w_o, b_o, H,
           attn_mask=None, key_padding_mask=None):
    # 官方布局：(3D, D) → 3 个 (D, D)，(3D,) → 3 个 (D,)
    w_q, w_k, w_v = np.split(w_in, 3)
    # bias 同样按 Q、K、V 切开
    b_q, b_k, b_v = np.split(b_in, 3)
    # x @ W^T + b：(B, T, D)，再拆 head → (B, H, T, d_h)
    q = split_np(q_in @ w_q.T + b_q, H)
    # K：(B, S, D) → (B, H, S, d_h)
    k = split_np(k_in @ w_k.T + b_k, H)
    # V：(B, S, D) → (B, H, S, d_h)
    v = split_np(v_in @ w_v.T + b_v, H)
    # (B, H, T, d_h) @ (B, H, d_h, S) → (B, H, T, S)
    scores = q @ k.swapaxes(-2, -1) / np.sqrt(q.shape[-1])
    # True=禁止，(T, S) 广播
    if attn_mask is not None:
        scores = np.where(attn_mask, -np.inf, scores)
    # (B, S) → (B, 1, 1, S)，True 的 key 列被屏蔽
    if key_padding_mask is not None:
        pad = key_padding_mask[:, None, None, :]
        scores = np.where(pad, -np.inf, scores)
    # 减最大值后 Softmax：(B, H, T, S)
    e = np.exp(scores - scores.max(-1, keepdims=True))
    # 按行归一化：(B, H, T, S)
    A = e / e.sum(-1, keepdims=True)
    # (B, H, T, d_h) → (B, T, H, d_h) → (B, T, D)
    out = (A @ v).swapaxes(1, 2)
    # 合并最后两维：(B, T, H, d_h) → (B, T, D)
    out = out.reshape(*out.shape[:2], -1)
    return out @ w_o.T + b_o, A
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, D = 3, 5, 8
dt = torch.float64
x = torch.randn(B, T, D, dtype=dt)
# 官方 True=禁止：严格上三角禁止未来；每条样本最后一个 key 是 pad
am = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
kp = torch.zeros(B, T, dtype=torch.bool)
kp[:, -1] = True
for H in (1, 4):
    off = nn.MultiheadAttention(D, H, dropout=0.0, batch_first=True)
    off = off.double()
    my = MyMHA(D, H).double()
    copy_official(my, off)
    print(H, align_report(my, off, x, attn_mask=am,
                          key_padding_mask=kp))
    # NumPy 版直接吃官方权重
    W, b = off.in_proj_weight.detach().numpy().copy(), \
        off.in_proj_bias.detach().numpy().copy()
    wo = off.out_proj.weight.detach().numpy()
    bo = off.out_proj.bias.detach().numpy()
    run = lambda w: mha_np(*[x.numpy()] * 3, w, b, wo, bo, H,
                           am.numpy(), kp.numpy())
    o_off, a_off = off(x, x, x, attn_mask=am, key_padding_mask=kp,
                       average_attn_weights=False)
    print("  NumPy 前向差:",
          abs(run(W)[0] - o_off.detach().numpy()).max(),
          abs(run(W)[1] - a_off.detach().numpy()).max())
    # 梯度：NumPy 中心差分 vs 官方 autograd，取 K 块一个元素
    off.zero_grad()
    o_off.pow(2).sum().backward()
    i, j, h = D + 1, 2, 1e-6
    Wp, Wm = W.copy(), W.copy()
    Wp[i, j] += h
    Wm[i, j] -= h
    fd = ((run(Wp)[0]**2).sum() - (run(Wm)[0]**2).sum()) / (2 * h)
    print("  NumPy 差分梯度差:",
          abs(fd - off.in_proj_weight.grad[i, j].item()))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn as nn
from torch import Tensor

class MyMHA(nn.Module):
    def __init__(self, D: int, H: int):
        pass
    def split(self, x: Tensor) -> Tensor:
        pass
    def forward(self, q_in, k_in, v_in, attn_mask=None,
                key_padding_mask=None):
        pass

def copy_official(my: MyMHA, off: nn.MultiheadAttention):
    pass
def grad_or_zero(p: Tensor) -> Tensor:
    pass
def align_report(my, off, x, **masks) -> dict:
    pass
```

##### Step 1（i）公式：定义投影并拆 head

$$
Q=XW_q^\top+b_q\in\mathbb{R}^{B\times T\times D}\ \to\ \mathbb{R}^{B\times H\times T\times d_h}
$$

##### Step 1（ii）代码

```python
# head 数和每个 head 的维度
self.H, self.d_h = H, D // H
# Q/K/V/输出投影，都是 D → D，默认带 bias
self.w_q, self.w_k = nn.Linear(D, D), nn.Linear(D, D)
self.w_v, self.w_o = nn.Linear(D, D), nn.Linear(D, D)
# (B, T, D) → (B, T, H, d_h) → (B, H, T, d_h)
return x.view(*x.shape[:2], self.H, self.d_h).transpose(1, 2)
```

##### Step 1（iii）解释

官方默认带 bias，所以这里也带 bias，这样才能和 `in_proj_bias` 一一对应。

##### Step 2（i）公式：分数与 mask

$$
S=\frac{QK^\top}{\sqrt{d_h}},\quad S_{ij}\leftarrow-\infty\ \text{ 若 mask 为 True}
$$

##### Step 2（ii）代码

```python
# 三路投影后拆 head：(B, H, T 或 S, d_h)
q = self.split(self.w_q(q_in))
k = self.split(self.w_k(k_in))
v = self.split(self.w_v(v_in))
# (B, H, T, d_h) @ (B, H, d_h, S) → (B, H, T, S)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)
# 与官方相同，bool mask 的 True=禁止；(T, S) 广播
if attn_mask is not None:
    scores = scores.masked_fill(attn_mask, float("-inf"))
# (B, S) → (B, 1, 1, S)，True 的 key 列被屏蔽
if key_padding_mask is not None:
    pad = key_padding_mask[:, None, None, :]
    scores = scores.masked_fill(pad, float("-inf"))
```

##### Step 2（iii）解释

这里的 mask 语义采用官方约定：True 表示禁止，所以不需要取反。`attn_mask` 实现只支持 `(T, S)`；官方还支持 `(B*H, T, S)`，本题未使用这种形状。

##### Step 3（i）公式：Softmax、拼 head、输出投影

$$
A=\mathrm{softmax}(S),\quad O=\mathrm{concat}(AV)W_o^\top+b_o
$$

##### Step 3（ii）代码

```python
# 沿 key 维归一化：(B, H, T, S)
A = scores.softmax(dim=-1)
# (B, H, T, d_h) → (B, T, H, d_h) → (B, T, D)
out = (A @ v).transpose(1, 2).flatten(2)
return self.w_o(out), A
```

##### Step 3（iii）解释

返回未平均的 $A$，形状是 $(B,H,T,S)$。比较时官方要传 `average_attn_weights=False`，否则官方返回 $(B,T,S)$。

##### Step 4（i）公式：拆分官方权重

$$
(W_q,W_k,W_v)=\mathrm{chunk}(W_{in},3),\quad (b_q,b_k,b_v)=\mathrm{chunk}(b_{in},3)
$$

##### Step 4（ii）代码

```python
with torch.no_grad():
    # (3D, D) → 3 个 (D, D)，顺序固定为 Q、K、V
    ws = off.in_proj_weight.chunk(3)
    # (3D,) → 3 个 (D,)，顺序相同
    bs = off.in_proj_bias.chunk(3)
    for m, w, b in zip((my.w_q, my.w_k, my.w_v), ws, bs):
        # 官方与 nn.Linear 都是 (out, in) 布局，不用转置
        m.weight.copy_(w)
        m.bias.copy_(b)
    # 输出投影直接对应 out_proj
    my.w_o.weight.copy_(off.out_proj.weight)
    my.w_o.bias.copy_(off.out_proj.bias)
```

##### Step 4（iii）解释

`no_grad` 中的原地复制不进入计算图。官方的 `in_proj_weight` 只在 Q/K/V 维度都等于 D 时存在，这是本题的设定。

##### Step 5（i）公式：前向与梯度误差

$$
\varepsilon_{\mathrm{out}}=\max|O^{my}-O^{off}|,\quad
\varepsilon_{\mathrm{grad}}=\max_\theta\max|\nabla_\theta\mathcal{L}^{my}-\nabla_\theta\mathcal{L}^{off}|
$$

##### Step 5（ii）代码

```python
# 参数没收到梯度时按 0 比较
return torch.zeros_like(p) if p.grad is None else p.grad
# 清空两边累积的梯度
my.zero_grad()
off.zero_grad()
o1, a1 = my(x, x, x, **masks)
# False：返回 (B, H, T, S)，不按 head 平均
o2, a2 = off(x, x, x, average_attn_weights=False, **masks)
# 两边用同一个标量损失反向
o1.pow(2).sum().backward()
o2.pow(2).sum().backward()
# 3 个 (D, D) 拼回 (3D, D)，与 in_proj_weight 对齐
g_w = torch.cat([grad_or_zero(m.weight)
                 for m in (my.w_q, my.w_k, my.w_v)])
# 3 个 (D,) 拼回 (3D,)，与 in_proj_bias 对齐
g_b = torch.cat([grad_or_zero(m.bias)
                 for m in (my.w_q, my.w_k, my.w_v)])
# 每对是（自己的梯度，官方梯度）
pairs = [(g_w, off.in_proj_weight.grad),
         (g_b, off.in_proj_bias.grad),
         (grad_or_zero(my.w_o.weight), off.out_proj.weight.grad),
         (grad_or_zero(my.w_o.bias), off.out_proj.bias.grad)]
return dict(out=(o1 - o2).abs().max().item(),
            attn=(a1 - a2).abs().max().item(),
            grad=max((a - b).abs().max().item() for a, b in pairs))
```

##### Step 5（iii）解释

梯度按官方的合并布局拼回后逐参数比较。NumPy 版没有 autograd，Demo 用中心差分取一个元素，与官方梯度核对。

##### 5️⃣ 复杂度

测试的主项是一次前向和一次反向，$T=S$ 时都是：

$$
\text{时间}=O(BTD^2+BT^2D),\quad \text{激活显存}=O(BHT^2+BTD)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：Q/K/V 合并权重的排列顺序如何对齐

`in_proj_weight` 的前 D 行是 Q，中间 D 行是 K，最后 D 行是 V，它经 `F.linear` 计算，所以权重布局与 `nn.Linear.weight` 相同：

$$
W_{in}=[W_q;W_k;W_v],\quad \mathrm{linear}(x,W,b)=xW^\top+b
$$

把任意两块放错位置，前向就会偏离。这里故意交换 Q 和 K：

```python
off = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
off = off.double()
my = MyMHA(D, 4).double()
copy_official(my, off)
print("正确顺序:", align_report(my, off, x))
# 交换 Q、K 两块权重和 bias
with torch.no_grad():
    w_q, b_q = my.w_q.weight.clone(), my.w_q.bias.clone()
    my.w_q.weight.copy_(my.w_k.weight)
    my.w_q.bias.copy_(my.w_k.bias)
    my.w_k.weight.copy_(w_q)
    my.w_k.bias.copy_(b_q)
print("交换 Q、K:", align_report(my, off, x))
# kdim、vdim 不等于 D 时才拆成三个独立权重
sep = nn.MultiheadAttention(8, 2, kdim=6, vdim=6)
print([n for n, _ in sep.named_parameters()])
```

##### Follow-up 2：Dropout、dtype、mask polarity 如何保证一致

三个条件都要固定：

$$
p_{\mathrm{drop}}=0\ \text{或 eval},\quad \text{同一 dtype},\quad \text{同一 mask 语义}
$$

官方的 Dropout 作用在注意力权重上，训练模式下每次采样都不同，所以对齐前要令 $p=0$ 或切 eval。默认 float32 下前向误差约 $10^{-7}$，梯度差约 $10^{-6}$；换成 float64 后都降到 $10^{-15}$ 量级或更低。官方 bool mask 的 True 表示禁止，float mask 则是加到分数上，两者填 $-\infty$ 时结果一致。官方默认 `batch_first=False`，默认 `average_attn_weights=True`。下面逐项实跑。

```python
m = nn.MultiheadAttention(D, 4, dropout=0.5, batch_first=True).double()
# train 模式 Dropout 生效：两次前向不同
print("train 两次差:", (m(x, x, x)[0] - m(x, x, x)[0]).abs().max().item())
# eval 后 Dropout 关闭，两次前向相同
m.eval()
with torch.no_grad():
    print("eval 两次差:",
          (m(x, x, x)[0] - m(x, x, x)[0]).abs().max().item())
# dtype：float32 与 float64 的误差
off32 = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
my32 = MyMHA(D, 4)
copy_official(my32, off32)
print("float32:", align_report(my32, off32, x.float()))
off = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
off = off.double()
my = MyMHA(D, 4).double()
copy_official(my, off)
print("float64:", align_report(my, off, x))
# polarity：把 pad 取反后结果明显不同
print("极性取反差:", (off(x, x, x, key_padding_mask=kp)[0]
      - off(x, x, x, key_padding_mask=~kp)[0]).abs().max().item())
# bool mask 与 float(-inf) mask 等价
fm = torch.zeros(T, T, dtype=dt).masked_fill(am, float("-inf"))
print("bool/float 差:", (off(x, x, x, attn_mask=am)[0]
      - off(x, x, x, attn_mask=fm)[0]).abs().max().item())
# 默认按 head 平均 (B, T, S)；False 时 (B, H, T, S)
w_avg = off(x, x, x)[1]
w_all = off(x, x, x, average_attn_weights=False)[1]
print(w_avg.shape, w_all.shape,
      (w_all.mean(1) - w_avg).abs().max().item())
```

##### Follow-up 3：为什么前向接近不等于 backward 也正确

两个函数在一点上值相等，推不出导数相等：

$$
f(\theta_0)=g(\theta_0)\ \not\Rightarrow\ \nabla f(\theta_0)=\nabla g(\theta_0)
$$

典型 bug 是图被意外切断，例如某一路多写了 `detach()`。这时前向数值完全不变，只有梯度出错。下面用 forward hook 模拟 K 路被 detach，再用 `gradcheck` 与数值差分比较输入梯度，这个检查不依赖官方实现。

```python
off = nn.MultiheadAttention(D, 4, dropout=0.0, batch_first=True)
off = off.double()
my = MyMHA(D, 4).double()
copy_official(my, off)
print("正确:", align_report(my, off, x))
# hook 返回值替换 w_k 的输出：前向值不变，K 路断图
h = my.w_k.register_forward_hook(lambda m, i, o: o.detach())
print("K 路 detach:", align_report(my, off, x))
# 输入梯度也会错，gradcheck 与数值差分比较后返回 False
xs = torch.randn(1, 3, D, dtype=dt, requires_grad=True)
f = lambda z: my(z, z, z)[0]
print("断图 gradcheck:", torch.autograd.gradcheck(
    f, (xs,), raise_exception=False))
h.remove()
print("修复 gradcheck:", torch.autograd.gradcheck(
    f, (xs,), raise_exception=False))
```

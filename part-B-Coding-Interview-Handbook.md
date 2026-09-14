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

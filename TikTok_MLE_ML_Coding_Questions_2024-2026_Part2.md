## 02｜Loss、训练目标与采样（ML Coding）

#### Q017｜稳定 Softmax / LogSumExp

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-02（公开面经日期）
- **题目：** 从零实现数值稳定的 Softmax 和 LogSumExp，能够处理很大或很小的 logits。
- **输入输出 / 约束：** 输入 `logits:[B,C]`，输出与输入同 shape 的概率；LSE 输出 `[B]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么必须先减 `max`，又为什么这不改变概率？
  2. `axis` 设置为 0 与 -1 会有怎样的错误？
  3. 若某一行全为 `-inf`，如何定义或处理？
- **来源：** [Jointaro：TikTok MLE CE 推导 + Transformer Block + 树题](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/)

##### 1️⃣ Dense Opening

稳定 Softmax 和 LogSumExp（LSE）都先把每行 logits 减去最大值再做指数运算，结构是“按类别维取 max → 平移 → exp → 求和 → 归一化或取 log 后加回 max”，数学核心对象是平移后的 $\tilde{z}_{bc}=z_{bc}-m_b\le 0$，它让每个指数项落在 $(0,1]$，分母落在 $[1,C]$。

##### 2️⃣ Global Equations

每行最大值与平移后的 logits：

$$
m_b=\max_c z_{bc},\quad \tilde{z}_{bc}=z_{bc}-m_b
$$

主目标是下面两个量，形状分别是 $[B,C]$ 和 $[B]$：

$$
p_{bc}=\frac{e^{\tilde{z}_{bc}}}{\sum_{c'}e^{\tilde{z}_{bc'}}},\quad
\mathrm{LSE}(z_b)=\log\sum_c e^{z_{bc}}=m_b+\log\sum_c e^{\tilde{z}_{bc}}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    # 每行最大值：(B, C) → (B, 1)
    m = logits.amax(dim=dim, keepdim=True)
    # 广播平移后取 exp，每项 ∈ [0, 1]：(B, C)
    e = torch.exp(logits - m)
    # 按行归一化：(B, C) / (B, 1) → (B, C)
    return e / e.sum(dim=dim, keepdim=True)

def log_sum_exp(logits: Tensor, dim: int = -1) -> Tensor:
    # 每行最大值：(B, C) → (B, 1)
    m = logits.amax(dim=dim, keepdim=True)
    # 平移后求和再取 log：(B, C) → (B, 1)，sum >= 1
    s = torch.log(torch.exp(logits - m).sum(dim=dim, keepdim=True))
    # 加回最大值并压缩类别维：(B, 1) → (B,)
    return (m + s).squeeze(dim)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def stable_softmax_np(logits, axis=-1):
    # 每行最大值：(B, C) → (B, 1)
    m = logits.max(axis=axis, keepdims=True)
    # 广播平移后取 exp：(B, C)
    e = np.exp(logits - m)
    # 按行归一化：(B, C)
    return e / e.sum(axis=axis, keepdims=True)

def log_sum_exp_np(logits, axis=-1):
    # 每行最大值：(B, C) → (B, 1)
    m = logits.max(axis=axis, keepdims=True)
    # 平移后求和再取 log：(B, C) → (B, 1)
    s = np.log(np.exp(logits - m).sum(axis=axis, keepdims=True))
    # 加回最大值并压缩类别维：(B, 1) → (B,)
    return (m + s).squeeze(axis)
```

##### 最小可运行测试（Demo）

```python
z = torch.tensor([[1000., 1001., 1002.], [-1000., -1001., -1002.],
                  [1., 2., 3.], [1000., 0., -1000.]])
# 未减 max：exp(1000) 上溢成 inf，exp(-1000) 下溢成 0
e = torch.exp(z)
print("naive softmax:\n", e / e.sum(-1, keepdim=True))
print("naive LSE:", torch.log(e.sum(-1)))
p, lse = stable_softmax(z), log_sum_exp(z)
print("stable softmax:\n", p)
# 第 0、2 行都应为手算值 [0.0900, 0.2447, 0.6652]
print("LSE:", lse)
print("与官方差:", (p - z.softmax(-1)).abs().max().item(),
      (lse - z.logsumexp(-1)).abs().max().item())
print("后端差:", abs(p.numpy() - stable_softmax_np(z.numpy())).max(),
      abs(lse.numpy() - log_sum_exp_np(z.numpy())).max())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def stable_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    pass

def log_sum_exp(logits: Tensor, dim: int = -1) -> Tensor:
    pass
```

##### Step 1（i）公式：按行取最大值

$$
m_b=\max_c z_{bc}
$$

##### Step 1（ii）代码

```python
# 每行最大值：(B, C) → (B, 1)
m = logits.amax(dim=dim, keepdim=True)
```

##### Step 1（iii）解释

`keepdim=True` 保留类别维，这样 $m$ 才能广播回 $(B,C)$。Softmax 和 LSE 都从这一步开始。

##### Step 2（i）公式：平移后归一化

$$
p_{bc}=\frac{e^{z_{bc}-m_b}}{\sum_{c'}e^{z_{bc'}-m_b}}
$$

##### Step 2（ii）代码

```python
# 广播平移后取 exp，每项 ∈ [0, 1]：(B, C)
e = torch.exp(logits - m)
# 按行归一化：(B, C) / (B, 1) → (B, C)
return e / e.sum(dim=dim, keepdim=True)
```

##### Step 2（iii）解释

最大项平移后为 $e^0=1$，所以分母至少是 1，既不会上溢，也不会出现 0/0。差距很大的项可以下溢为 0，这时概率本身也可以忽略。

##### Step 3（i）公式：LogSumExp

$$
\mathrm{LSE}(z_b)=m_b+\log\sum_c e^{z_{bc}-m_b}
$$

##### Step 3（ii）代码

```python
# 平移后求和再取 log：(B, C) → (B, 1)，sum >= 1
s = torch.log(torch.exp(logits - m).sum(dim=dim, keepdim=True))
# 加回最大值并压缩类别维：(B, 1) → (B,)
return (m + s).squeeze(dim)
```

##### Step 3（iii）解释

提出公因子 $e^{m_b}$，再取 log，就得到 $m_b$ 这一项。$\log\sum$ 的范围是 $[0,\log C]$，所以结果满足 $m_b\le\mathrm{LSE}\le m_b+\log C$。

##### 5️⃣ 复杂度

每行做一次 max、一次 exp、一次求和：

$$
\text{时间}=O(BC),\quad \text{额外内存}=O(BC)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么必须先减 max，这为什么不改变概率

未减 max 时，float32 的 $e^{z}$ 在 $z\gtrsim 89$ 上溢，全部项很小时分子分母又都下溢成 0，结果是 NaN。分子分母同乘 $e^{-c}$ 时比值不变，所以任意平移都不改变概率：

$$
\frac{e^{z_c-a}}{\sum_{c'}e^{z_{c'}-a}}=\frac{e^{-a}e^{z_c}}{e^{-a}\sum_{c'}e^{z_{c'}}}=\frac{e^{z_c}}{\sum_{c'}e^{z_{c'}}}
$$

选择 $a=m$ 的原因是，这样最大指数恰好为 0，分母落在 $[1,C]$。

```python
x = torch.tensor([[0.5, -0.3, 2.0]], dtype=torch.float64)
# 平移不变性：加 700 后概率不变
print((stable_softmax(x) - stable_softmax(x + 700)).abs().max().item())
# float32 的上溢临界点：88 尚有限，89 已是 inf
print(torch.exp(torch.tensor([88., 89.])))
# 精度限制：float32 在 1e8 附近间距为 8，差值 1 被吃掉，概率变成均匀
print(stable_softmax(torch.tensor([[1e8, 1e8 + 1.0]])))
```

减 max 能解决溢出，无法恢复已经丢失的精度。最后一行说明，logits 大到相邻值无法区分时要换 float64。

##### Follow-up 2：axis 设置为 0 与 -1 会有怎样的错误

输入是 $[B,C]$ 时，`axis=0` 沿 Batch 归一化。Softmax 的 shape 不变，所以这个错误不会报错：每一列和为 1，每一行不再是概率分布，样本之间还会相互影响。

$$
p^{\mathrm{wrong}}_{bc}=\frac{e^{z_{bc}}}{\sum_{b'}e^{z_{b'c}}},\quad
\sum_b p^{\mathrm{wrong}}_{bc}=1
$$

```python
torch.manual_seed(0)
z2 = torch.randn(3, 4)
print("axis=-1 行和:", stable_softmax(z2, -1).sum(1))
print("axis=0 行和:", stable_softmax(z2, 0).sum(1))
# LSE 的 shape 错成 [C]，这个错误更容易被发现
print(log_sum_exp(z2, -1).shape, log_sum_exp(z2, 0).shape)
# B=1 时 axis=0 的结果全为 1，loss 也不再反映 logits
print(stable_softmax(z2[:1], 0))
```

##### Follow-up 3：若某一行全为 -inf，如何定义或处理

全 $-\infty$ 表示这一行的候选集为空。求和为空时 LSE 的自然定义是 $\log 0=-\infty$；Softmax 是 0/0，数学上没有定义。实现上有三种选择：保留 NaN 让错误暴露、返回全 0、返回均匀分布。主实现的 $m=-\infty$ 时会出现 $-\infty-(-\infty)$，所以这一行会得到 NaN，需要先把 $m$ 替换为 0：

$$
m'_b=\begin{cases}m_b,&m_b>-\infty\\0,&m_b=-\infty\end{cases}
$$

```python
def safe_softmax(z, empty=float("nan")):
    # (B, 1)：全 -inf 的行 m 换成 0，exp(-inf)=0
    m = z.amax(-1, keepdim=True)
    m = m.masked_fill(torch.isinf(m), 0.0)
    e = torch.exp(z - m)
    s = e.sum(-1, keepdim=True)
    # 空行整行填 empty：(B, C)
    return torch.where(s > 0, e / s, torch.full_like(e, empty))

def safe_lse(z):
    # (B, 1)：空行 sum=0，log 0=-inf，结果为 -inf
    m = z.amax(-1, keepdim=True)
    m = m.masked_fill(torch.isinf(m), 0.0)
    return (m + torch.exp(z - m).sum(-1, keepdim=True).log()).squeeze(-1)
inf = float("inf")
z3 = torch.tensor([[-inf, -inf, -inf], [0., -inf, 0.]])
print("主实现:", stable_softmax(z3)[0], log_sum_exp(z3)[0])
print("官方:", z3.softmax(-1)[0], z3.logsumexp(-1)[0])
print("保留 NaN:", safe_softmax(z3))
print("均匀分布:", safe_softmax(z3, 1 / 3))
print("安全 LSE:", safe_lse(z3))
```

本实现的建议是：LSE 返回 $-\infty$，Softmax 默认保留 NaN，这样 mask 全空的上游错误能被发现。业务上确实允许空候选集时，再显式选择 `empty`。

#### Q018｜多分类 Cross-Entropy：整数标签与 One-Hot

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06（面经标注月份）
- **题目：** 手写 batch 平均交叉熵，既支持 class-index labels，也支持 one-hot labels，不调用高阶 CE API。
- **输入输出 / 约束：** `logits:[B,C]`；`labels:[B]` 或 `[B,C]`；输出标量。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何通过 LogSumExp 直接求 CE，避免显式 Softmax 后取 log？
  2. 标签带 `ignore_index` 时，分母应如何处理？
  3. 若有 label smoothing，如何修改目标分布？
- **来源：** [Screna：TikTok Senior MLE 自注意力、交叉熵](https://www.screna.ai/experience/4498ae13-338f-4777-97dc-36b24b0b612a)

##### 1️⃣ Dense Opening

多分类交叉熵（Cross-Entropy）度量目标分布与模型分布之间的差异，结构是“logits → 用 LogSumExp 求 log 概率 → 按标签取对数概率或与 one-hot 求内积 → Batch 平均”，数学核心对象是对数概率矩阵 $\log p\in\mathbb{R}^{B\times C}$。整个过程只依赖 log 概率，所以不会经过易下溢的概率。

##### 2️⃣ Global Equations

由 logits 直接得到 log 概率：

$$
\log p_{bc}=z_{bc}-\mathrm{LSE}(z_b),\quad \mathrm{LSE}(z_b)=m_b+\log\sum_c e^{z_{bc}-m_b},\quad m_b=\max_c z_{bc}
$$

主目标是 Batch 平均交叉熵。one-hot 或软标签 $q_b$ 用内积，整数标签 $y_b$ 只取一个元素：

$$
\mathcal{L}=-\frac{1}{B}\sum_b\sum_c q_{bc}\log p_{bc},\quad
\mathcal{L}=-\frac{1}{B}\sum_b\log p_{b,y_b}
$$

当 $q_b$ 是 $y_b$ 的 one-hot 时，两式相等。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def log_softmax(logits: Tensor) -> Tensor:
    # 每行最大值：(B, C) → (B, 1)
    m = logits.amax(dim=-1, keepdim=True)
    # 平移后的 LSE：(B, C) → (B, 1)
    lse = m + torch.exp(logits - m).sum(-1, keepdim=True).log()
    # 广播相减得 log 概率：(B, C)
    return logits - lse

def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    # 对数概率：(B, C)
    log_p = log_softmax(logits)
    # labels 是 [B]：class-index 分支
    if labels.dim() == 1:
        # (B,) → (B, 1)，gather 后 (B, 1) → (B,)，取真实类 log 概率
        nll = -log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
    # labels 是 [B, C]：one-hot 或软标签分支
    else:
        # 逐元素相乘后沿类别维求和：(B, C) → (B,)
        nll = -(labels * log_p).sum(dim=-1)
    # Batch 平均：(B,) → 标量
    return nll.mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def log_softmax_np(logits):
    # 每行最大值：(B, C) → (B, 1)
    m = logits.max(axis=-1, keepdims=True)
    # 平移后的 LSE：(B, C) → (B, 1)
    lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
    # 广播相减得 log 概率：(B, C)
    return logits - lse

def cross_entropy_np(logits, labels):
    # 对数概率：(B, C)
    log_p = log_softmax_np(logits)
    # labels 是 [B]：class-index 分支
    if labels.ndim == 1:
        # 高级索引：行 (B,)、列 (B,) → (B,)
        nll = -log_p[np.arange(len(labels)), labels]
    # labels 是 [B, C]：one-hot 或软标签分支
    else:
        # 逐元素相乘后沿类别维求和：(B, C) → (B,)
        nll = -(labels * log_p).sum(axis=-1)
    # Batch 平均：(B,) → 标量
    return nll.mean()
```

##### 最小可运行测试（Demo）

```python
import torch.nn.functional as F
torch.manual_seed(0)
z = torch.randn(4, 5, dtype=torch.float64)
y = torch.tensor([0, 2, 4, 1])
# (B,) → (B, C)，转浮点才能和 log_p 相乘
y1 = F.one_hot(y, 5).double()
ce_idx, ce_hot = cross_entropy(z, y), cross_entropy(z, y1)
print("两种标签:", ce_idx.item(), ce_hot.item())
print("与官方差:", abs(ce_idx.item() - F.cross_entropy(z, y).item()))
print("后端差:", abs(ce_idx.item()
      - cross_entropy_np(z.numpy(), y.numpy())),
      abs(ce_hot.item() - cross_entropy_np(z.numpy(), y1.numpy())))
# 手算：logits 全 0 时 CE = log C = log 5 ≈ 1.6094
print(cross_entropy(torch.zeros(2, 5), torch.tensor([1, 3])).item())
# 大 logits：真实类落后 1000，CE 约等于 1000
big = torch.tensor([[1000., 0.], [0., 1000.]])
print(cross_entropy(big, torch.tensor([1, 0])).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def log_softmax(logits: Tensor) -> Tensor:
    pass

def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    pass
```

##### Step 1（i）公式：稳定的 log 概率

$$
\log p_{bc}=z_{bc}-\left(m_b+\log\sum_{c'}e^{z_{bc'}-m_b}\right)
$$

##### Step 1（ii）代码

```python
# 每行最大值：(B, C) → (B, 1)
m = logits.amax(dim=-1, keepdim=True)
# 平移后的 LSE：(B, C) → (B, 1)
lse = m + torch.exp(logits - m).sum(-1, keepdim=True).log()
# 广播相减得 log 概率：(B, C)
return logits - lse
```

##### Step 1（iii）解释

Softmax 的 log 可以化简成 logits 减去 LSE，全程不计算概率本身。求和项至少为 1，所以 `log` 是安全的。

##### Step 2（i）公式：class-index 标签

$$
\mathrm{nll}_b=-\log p_{b,y_b}
$$

##### Step 2（ii）代码

```python
# 对数概率：(B, C)
log_p = log_softmax(logits)
# labels 是 [B]：class-index 分支
if labels.dim() == 1:
    # (B,) → (B, 1)，gather 后 (B, 1) → (B,)，取真实类 log 概率
    nll = -log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
```

##### Step 2（iii）解释

`gather` 要求索引维数与输入相同，所以 labels 先插入一维，取完再压缩。这里 labels 必须是整数类型。

##### Step 3（i）公式：one-hot 标签

$$
\mathrm{nll}_b=-\sum_c q_{bc}\log p_{bc}
$$

##### Step 3（ii）代码

```python
# labels 是 [B, C]：one-hot 或软标签分支
else:
    # 逐元素相乘后沿类别维求和：(B, C) → (B,)
    nll = -(labels * log_p).sum(dim=-1)
```

##### Step 3（iii）解释

用 `labels.dim()` 判断分支，无需额外参数。这一分支对软标签同样成立，Follow-up 3 的 label smoothing 就复用它。

##### Step 4（i）公式：Batch 平均

$$
\mathcal{L}=\frac{1}{B}\sum_b\mathrm{nll}_b
$$

##### Step 4（ii）代码

```python
# Batch 平均：(B,) → 标量
return nll.mean()
```

##### Step 4（iii）解释

结果是标量，可以直接 `backward()`。logits 全为 0 时结果是 $\log C$，这个值常用作训练初始 loss 的检查基线。

##### 5️⃣ 复杂度

计算 log 概率是主项，one-hot 分支还需要一次内积：

$$
\text{时间}=O(BC),\quad \text{额外内存}=O(BC)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：通过 LogSumExp 直接求 CE，不经过显式 Softmax 再取 log

把 Softmax 代入 CE，并化简 log：

$$
\mathrm{CE}_b=-\log\frac{e^{z_{b,y_b}}}{\sum_c e^{z_{bc}}}=\mathrm{LSE}(z_b)-z_{b,y_b}
$$

Softmax 后再取 log 的问题在于：真实类概率下溢为 0 后，$-\log 0=\infty$，loss 变成 inf。LSE 形式只有 logits 相减，结果仍然有限。

```python
z4 = torch.tensor([[200., 0.]])
y4 = torch.tensor([1])
# 显式 Softmax：概率下溢为 0，-log 0 = inf
print("显式:", -torch.softmax(z4, -1)[0, 1].log().item())
# 直接公式：LSE - z_y
print("LSE:", (torch.logsumexp(z4, -1) - z4[0, 1]).item())
print("主实现:", cross_entropy(z4, y4).item())
print("官方:", F.cross_entropy(z4, y4).item())
```

##### Follow-up 2：标签带 ignore_index 时，分母应如何处理

分母必须是有效样本数，不能用 $B$。否则同样的有效样本在 pad 更多的 Batch 中 loss 会偏小：

$$
\mathcal{L}=\frac{\sum_b\mathbb{1}[y_b\ne\mathrm{ign}]\,\mathrm{nll}_b}{\sum_b\mathbb{1}[y_b\ne\mathrm{ign}]}
$$

ignore 标签本身不是合法类别，不能直接传给 `gather`，要先把它替换成任意合法下标，再用 mask 清零。

```python
def ce_ignore(logits, labels, ignore_index=-100):
    # 有效样本掩码：(B,)
    keep = labels != ignore_index
    # 把 ignore 换成 0，防止 gather 越界：(B,)
    safe = labels.masked_fill(~keep, 0)
    # (B, C) → (B,)，取真实类 log 概率
    nll = -log_softmax(logits).gather(1, safe[:, None]).squeeze(1)
    # 只累加有效项，再除以有效样本数
    return (nll * keep).sum() / keep.sum()
y5 = torch.tensor([0, -100, 4, -100])
print("手写:", ce_ignore(z, y5).item())
print("官方:", F.cross_entropy(z, y5, ignore_index=-100).item())
print("错误分母 B:", (ce_ignore(z, y5) * 2 / 4).item())
# 全部被忽略：0/0，手写与官方都得到 NaN
y6 = torch.full((4,), -100)
print(ce_ignore(z, y6).item(), F.cross_entropy(z, y6).item())
```

##### Follow-up 3：有 label smoothing 时，如何修改目标分布

把 one-hot 与均匀分布按 $\varepsilon$ 混合，然后直接走软标签分支。真实类最终概率是 $1-\varepsilon+\varepsilon/C$，其余类是 $\varepsilon/C$：

$$
q_{bc}=(1-\varepsilon)\mathbb{1}[c=y_b]+\frac{\varepsilon}{C}
$$

```python
def smooth_targets(labels, C, eps):
    # (B,) → (B, C)，再与均匀分布混合
    return (1 - eps) * F.one_hot(labels, C).double() + eps / C
q = smooth_targets(y, 5, 0.1)
print("目标行和:", q.sum(-1), "第 0 行:", q[0])
print("手写:", cross_entropy(z, q).item())
print("官方:", F.cross_entropy(z, y, label_smoothing=0.1).item())
```

#### Q019｜二分类 BCE / BCEWithLogits

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 手写二分类交叉熵，先写概率版，再写直接接收 logits 的稳定版本。
- **输入输出 / 约束：** `logits:[N]`、`y:[N]`；输出标量 loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么接近 0/1 的预测概率会导致 `log(0)`？
  2. 如何用 `softplus(z)-y*z` 实现稳定 BCE？
  3. 类别不平衡时 `pos_weight` 与简单 sample_weight 有何不同？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

##### 1️⃣ Dense Opening

二分类交叉熵（BCE）是伯努利分布的负对数似然，结构是“logit 经 Sigmoid 得概率 → 对两类取 log → 平均”，稳定版则把它化简成只含 logit 的闭式，数学核心对象是 Softplus 恒等式 $\ell(z,y)=\mathrm{softplus}(z)-yz$。它不经过概率，所以不存在 $\log 0$。

##### 2️⃣ Global Equations

概率版，这里 $\sigma$ 是 Sigmoid：

$$
p_i=\sigma(z_i),\quad
\mathcal{L}=-\frac{1}{N}\sum_i\bigl[y_i\log p_i+(1-y_i)\log(1-p_i)\bigr]
$$

由下面两个恒等式代入：

$$
-\log\sigma(z)=\mathrm{softplus}(z)-z,\quad
-\log(1-\sigma(z))=\mathrm{softplus}(z)
$$

得到主目标，也就是 logits 版：

$$
\mathcal{L}=\frac{1}{N}\sum_i\bigl[\mathrm{softplus}(z_i)-y_iz_i\bigr],\quad
\mathrm{softplus}(z)=\max(z,0)+\log(1+e^{-|z|})
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor
def bce_prob(p: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    # 裁剪到 [eps, 1-eps]，防止 log(0)：(N,)
    p = p.clamp(eps, 1 - eps)
    # 逐样本负对数似然，再平均：(N,) → 标量
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()

def softplus(z: Tensor) -> Tensor:
    # log(1+e^z)=max(z,0)+log(1+e^-|z|)，e^-|z| <= 1 不上溢
    return z.clamp(min=0) + torch.log1p(torch.exp(-z.abs()))

def bce_with_logits(z: Tensor, y: Tensor) -> Tensor:
    # softplus(z) - y*z，逐样本后平均：(N,) → 标量
    return (softplus(z) - y * z).mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def bce_prob_np(p, y, eps=1e-7):
    # 裁剪到 [eps, 1-eps]：(N,)
    p = np.clip(p, eps, 1 - eps)
    # 逐样本负对数似然，再平均：(N,) → 标量
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

def softplus_np(z):
    # max(z,0)+log(1+e^-|z|)，不上溢
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))

def bce_with_logits_np(z, y):
    # softplus(z) - y*z，逐样本后平均：(N,) → 标量
    return (softplus_np(z) - y * z).mean()
```

##### 最小可运行测试（Demo）

```python
import torch.nn.functional as F
torch.manual_seed(0)
dt = torch.float64
z = torch.randn(6, dtype=dt)
y = torch.tensor([1., 0., 1., 1., 0., 0.], dtype=dt)
l1 = bce_with_logits(z, y).item()
l2 = bce_prob(torch.sigmoid(z), y).item()
print("logits 版 / 概率版:", l1, l2)
print("与官方差:",
      abs(l1 - F.binary_cross_entropy_with_logits(z, y).item()),
      abs(l2 - F.binary_cross_entropy(torch.sigmoid(z), y).item()))
print("后端差:", abs(l1 - bce_with_logits_np(z.numpy(), y.numpy())),
      abs(l2 - bce_prob_np(torch.sigmoid(z).numpy(), y.numpy())))
# 手算：z=0 时 loss = log 2 ≈ 0.6931
print(bce_with_logits(torch.zeros(2), torch.tensor([0., 1.])).item())
# 极端错误预测：loss 应为 1000；概率版被 eps 截断在约 16
ze = torch.tensor([1000., -1000.], dtype=dt)
ye = torch.tensor([0., 1.], dtype=dt)
print(bce_with_logits(ze, ye).item(),
      bce_prob(torch.sigmoid(ze), ye).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor
def bce_prob(p: Tensor, y: Tensor, eps: float = 1e-7) -> Tensor:
    pass

def softplus(z: Tensor) -> Tensor:
    pass

def bce_with_logits(z: Tensor, y: Tensor) -> Tensor:
    pass
```

##### Step 1（i）公式：概率版

$$
\mathcal{L}=-\frac{1}{N}\sum_i\bigl[y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\bigr],\quad
\hat{p}_i=\mathrm{clip}(p_i,\epsilon,1-\epsilon)
$$

##### Step 1（ii）代码

```python
# 裁剪到 [eps, 1-eps]，防止 log(0)：(N,)
p = p.clamp(eps, 1 - eps)
# 逐样本负对数似然，再平均：(N,) → 标量
return -(y * p.log() + (1 - y) * (1 - p).log()).mean()
```

##### Step 1（iii）解释

裁剪可以防止 inf，但裁剪后的 loss 上限被固定在 $-\log\epsilon$，并且被裁剪样本的梯度为 0。`eps=1e-7` 在 float32 下仍可分辨 $1-\epsilon\ne 1$。

##### Step 2（i）公式：稳定 Softplus

$$
\mathrm{softplus}(z)=\log(1+e^z)=\max(z,0)+\log(1+e^{-|z|})
$$

##### Step 2（ii）代码

```python
# log(1+e^z)=max(z,0)+log(1+e^-|z|)，e^-|z| <= 1 不上溢
return z.clamp(min=0) + torch.log1p(torch.exp(-z.abs()))
```

##### Step 2（iii）解释

$z>0$ 时提出 $e^z$ 得到 $z+\log(1+e^{-z})$；$z\le 0$ 时原式不变。指数项始终不超过 1，所以不会上溢；`log1p` 在指数项很小时也能保留精度。

##### Step 3（i）公式：logits 版

$$
\ell_i=\mathrm{softplus}(z_i)-y_iz_i,\quad \mathcal{L}=\frac{1}{N}\sum_i\ell_i
$$

##### Step 3（ii）代码

```python
# softplus(z) - y*z，逐样本后平均：(N,) → 标量
return (softplus(z) - y * z).mean()
```

##### Step 3（iii）解释

代入 $y=1$ 得 $\mathrm{softplus}(z)-z=\mathrm{softplus}(-z)=-\log\sigma(z)$，代入 $y=0$ 得 $\mathrm{softplus}(z)=-\log(1-\sigma(z))$。两者都非负，且 $y$ 为软标签时公式同样成立。

##### 5️⃣ 复杂度

每个样本只做常数次运算：

$$
\text{时间}=O(N),\quad \text{额外内存}=O(N)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么接近 0/1 的预测概率会导致 log(0)

Sigmoid 在浮点数中会饱和。float32 下 $z$ 达到约 17 时 $\sigma(z)$ 就舍入成 1，$1-p=0$；$z$ 非常负时 $p$ 下溢成 0。此时错误预测样本的 loss 是：

$$
-\log(1-\sigma(z))\to-\log 0=\infty
$$

裁剪虽然消除了 inf，但也让梯度变成 0。logits 版的梯度是 $\sigma(z)-y$，错误预测时绝对值接近 1。

```python
zs = torch.tensor([10., 16., 17., 18.])
# float32 下 z 达到 17 后 1-p 舍入成 0
print("1-p:", 1 - torch.sigmoid(zs))
zb = torch.tensor([50.], requires_grad=True)
yb = torch.tensor([0.])
# 未裁剪：log(1-p) = -inf
print("未裁剪:", -(1 - torch.sigmoid(zb)).log().item())
# 裁剪后梯度为 0，这个样本学不动
bce_prob(torch.sigmoid(zb), yb).backward()
print("裁剪版梯度:", zb.grad.item())
zb.grad = None
# logits 版梯度 σ(z)-y = 1
bce_with_logits(zb, yb).backward()
print("logits 版梯度:", zb.grad.item())
```

##### Follow-up 2：如何用 softplus(z)-y*z 实现稳定 BCE

推导从 Sigmoid 的 log 开始：

$$
-\log\sigma(z)=\log(1+e^{-z})=\log(1+e^{z})-z=\mathrm{softplus}(z)-z
$$

$$
-\log(1-\sigma(z))=\log(1+e^{z})=\mathrm{softplus}(z)
$$

按 $y$ 加权：$y(\mathrm{softplus}(z)-z)+(1-y)\mathrm{softplus}(z)=\mathrm{softplus}(z)-yz$。直接写 $\log(1+e^z)$ 在大 $z$ 时仍会上溢，所以要用主实现的形式。

```python
zl = torch.tensor([-1000., -20., 0., 20., 1000.])
# 直接写 log(1+e^z)：z=1000 时 inf
print("直接写:", torch.log1p(torch.exp(zl)))
print("稳定版:", softplus(zl))
print("官方:", F.softplus(zl))
yl = torch.tensor([1., 1., 1., 0., 0.])
print(bce_with_logits(zl, yl).item(),
      F.binary_cross_entropy_with_logits(zl, yl).item())
```

##### Follow-up 3：类别不平衡时 pos_weight 与 sample_weight 有何不同

`pos_weight` 只放大正类项，按类别统一作用；`sample_weight` 乘在每个样本的整项 loss 上，可以按样本任意设定：

$$
\ell^{\mathrm{pos}}_i=-w_+y_i\log\sigma(z_i)-(1-y_i)\log(1-\sigma(z_i)),\quad
\ell^{\mathrm{samp}}_i=s_i\,\ell_i
$$

若令 $s_i=w_+$（$y_i=1$）且 $s_i=1$（$y_i=0$），两者在除以 $N$ 时相等；若按 $\sum_i s_i$ 归一化，则只差一个标量。`pos_weight` 还会改变最优预测：设正类比例为 $\pi$，常数预测的最优 logit 由 $\log\frac{\pi}{1-\pi}$ 平移 $\log w_+$：

$$
z^\ast=\log\frac{w_+\pi}{1-\pi}
$$

```python
def bce_pos_weight(z, y, w_pos):
    # 正类项乘 w_pos：softplus(z)-z 是 -log σ(z)，(N,) → 标量
    return (w_pos * y * (softplus(z) - z)
            + (1 - y) * softplus(z)).mean()
w_pos = 3.0
print("手写 pos_weight:", bce_pos_weight(z, y, w_pos).item())
print("官方 pos_weight:", F.binary_cross_entropy_with_logits(
    z, y, pos_weight=torch.tensor(w_pos, dtype=dt)).item())
# 等价的按类 sample_weight：(N,)
s = 1 + (w_pos - 1) * y
print("官方 weight:", F.binary_cross_entropy_with_logits(
    z, y, weight=s).item())
# 常数预测：网格搜索最优 logit，与闭式比较
pi, grid = 0.1, torch.linspace(-5, 5, 10001, dtype=dt)
loss = (w_pos * pi * (softplus(grid) - grid)
        + (1 - pi) * softplus(grid))
print("网格解:", grid[loss.argmin()].item(),
      "闭式:", torch.log(torch.tensor(w_pos * pi / (1 - pi))).item())
```

#### Q020｜Python 实现 KL Divergence

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 年月未公开
- **题目：** 根据两个离散概率分布 P、Q 计算 KL(P || Q)，明确概率为 0 时的规则。
- **输入输出 / 约束：** `p,q:[N,C]`，输出 batch mean KL 或每个样本 KL（接口需声明）。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. KL(P||Q) 与 KL(Q||P) 为什么通常不同？
  2. 当 `p_i=0` 和 `q_i=0,p_i>0` 时分别怎么处理？
  3. 输入是 logits、log-probs 还是概率，代码路径如何区分？
- **来源：** [Glassdoor：TikTok MLE Intern 单头/多头 Attention、Mask、KL](https://www.glassdoor.com/Interview/Implement-Single-head-and-Multi-head-attention-in-pytorch-A-follow-up-to-this-was-use-masking-to-mask-out-the-later-on-tok-QTN_8701206.htm)

##### 1️⃣ Dense Opening

KL 散度（Kullback-Leibler Divergence）衡量用分布 Q 近似分布 P 时多付出的期望对数代价，结构是“确定支撑集 → 计算逐类 $p\log(p/q)$ → 沿类别维求和 → 按接口聚合”，数学核心对象是每个样本的 KL 向量 $D\in\mathbb{R}^{N}$。本接口声明：`reduction="none"` 返回 `[N]`，`reduction="batchmean"` 返回标量。

##### 2️⃣ Global Equations

主目标，单个样本的 KL：

$$
D_{\mathrm{KL}}(P\,\|\,Q)=\sum_{c=1}^{C}p_c\log\frac{p_c}{q_c}
$$

概率为 0 时的规则：

$$
p_c\log\frac{p_c}{q_c}=
\begin{cases}
0, & p_c=0\\
+\infty, & p_c>0,\ q_c=0
\end{cases}
$$

Batch 平均：

$$
\mathrm{KL}_{\mathrm{mean}}=\frac{1}{N}\sum_{n=1}^{N}D_n
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def kl_div(p: Tensor, q: Tensor, reduction: str = "batchmean") -> Tensor:
    # p, q: (N, C) → support: (N, C)，True 表示 p>0
    support = p > 0
    # p=0 的位置保持 0，对应规则 0·log(0/q)=0
    log_ratio = torch.zeros_like(p)
    # 布尔索引取出 M 个元素 (M,)，只在支撑集上取 log；q=0 得 +inf
    log_ratio[support] = p[support].log() - q[support].log()
    # (N, C) 沿类别维求和 → per_sample: (N,)
    per_sample = (p * log_ratio).sum(dim=-1)
    # 不聚合：返回每个样本的 KL
    if reduction == "none":
        return per_sample
    # (N,) → 标量，除以样本数 N
    return per_sample.mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def kl_div_np(p, q, reduction="batchmean"):
    # p, q: (N, C) → support: (N, C)，True 表示 p>0
    support = p > 0
    # p=0 的位置保持 0，对应规则 0·log(0/q)=0
    log_ratio = np.zeros_like(p)
    # 关闭 log(0) 告警，q=0 时按定义得到 -inf
    with np.errstate(divide="ignore"):
        # 布尔索引取出 (M,)，只在支撑集上取 log
        log_ratio[support] = np.log(p[support]) - np.log(q[support])
    # (N, C) 沿类别维求和 → per_sample: (N,)
    per_sample = (p * log_ratio).sum(axis=-1)
    # 不聚合：返回每个样本的 KL
    if reduction == "none":
        return per_sample
    # (N,) → 标量，除以样本数 N
    return per_sample.mean()
```

##### 最小可运行测试（Demo）

```python
# 第 1 行手算：0.5·ln(0.5/0.9)+0.5·ln(0.5/0.1)≈0.5108
# 第 2 行含 p=0：只有前两类贡献，等于 ln2≈0.6931
p = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
q = torch.tensor([[0.9, 0.1, 0.0], [0.25, 0.25, 0.5]], dtype=torch.float64)
print("逐样本:", kl_div(p, q, "none").tolist())
print("NumPy :", kl_div_np(p.numpy(), q.numpy(), "none").tolist())
print("batchmean:", kl_div(p, q).item())
# 官方 API 约定：input 为 log Q，target 为 P；列 3 的 p=q=0 要先去掉
print("F.kl_div:", F.kl_div(
    q[0:1, :2].log(), p[0:1, :2], reduction="batchmean").item())
# p>0 且 q=0：KL 为 +inf
p2 = torch.tensor([[1.0, 0.0]])
q2 = torch.tensor([[0.0, 1.0]])
print("q=0:", kl_div(p2, q2).item(), kl_div_np(p2.numpy(), q2.numpy()))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def kl_div(p: Tensor, q: Tensor, reduction: str = "batchmean") -> Tensor:
    pass
```

##### Step 1（i）公式：确定支撑集

$$
\mathcal{S}=\{(n,c)\mid p_{n,c}>0\}
$$

##### Step 1（ii）代码

```python
# p, q: (N, C) → support: (N, C)，True 表示 p>0
support = p > 0
```

##### Step 1（iii）解释

KL 对 P 求期望，只有 $p_c>0$ 的类别才有贡献，所以先用布尔掩码圈出支撑集。

##### Step 2（i）公式：计算对数比值

$$
L_{n,c}=
\begin{cases}
\log p_{n,c}-\log q_{n,c}, & (n,c)\in\mathcal{S}\\
0, & (n,c)\notin\mathcal{S}
\end{cases}
$$

##### Step 2（ii）代码

```python
# p=0 的位置保持 0，对应规则 0·log(0/q)=0
log_ratio = torch.zeros_like(p)
# 布尔索引取出 M 个元素 (M,)，只在支撑集上取 log；q=0 得 +inf
log_ratio[support] = p[support].log() - q[support].log()
```

##### Step 2（iii）解释

若直接写 `p * (p.log() - q.log())`，$p=q=0$ 时得到 $0\times(-\infty+\infty)$，结果是 NaN。先按支撑集赋值后，$p=0$ 的位置恒为 0；$p>0,q=0$ 的位置保留 $+\infty$。

##### Step 3（i）公式：沿类别维求和

$$
D_n=\sum_{c=1}^{C}p_{n,c}L_{n,c}
$$

##### Step 3（ii）代码

```python
# (N, C) 沿类别维求和 → per_sample: (N,)
per_sample = (p * log_ratio).sum(dim=-1)
```

##### Step 3（iii）解释

求和维度是类别维，每个样本对应一个 KL 值。分布合法时 $D_n\ge 0$，当且仅当 $P=Q$ 时取 0。

##### Step 4（i）公式：按接口聚合

$$
\mathrm{KL}_{\mathrm{mean}}=\frac{1}{N}\sum_{n=1}^{N}D_n
$$

##### Step 4（ii）代码

```python
if reduction == "none":
    return per_sample
# (N,) → 标量，除以样本数 N
return per_sample.mean()
```

##### Step 4（iii）解释

这里的 mean 除以样本数 $N$，对应官方 `F.kl_div` 的 `batchmean`。官方的 `reduction="mean"` 除以元素总数 $NC$，数值不是 KL 的 batch 平均。

##### 5️⃣ 复杂度

逐元素运算一次：

$$
\text{时间}=O(NC),\quad \text{额外空间}=O(NC)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：KL(P||Q) 与 KL(Q||P) 为什么通常不同

两个方向的差别在于期望所用的分布和 log 比值的符号：

$$
D_{\mathrm{KL}}(P\|Q)=\mathbb{E}_{P}\!\left[\log\frac{P}{Q}\right],\quad
D_{\mathrm{KL}}(Q\|P)=\mathbb{E}_{Q}\!\left[\log\frac{Q}{P}\right]
$$

前者重点惩罚 P 大而 Q 小的类别，后者重点惩罚 Q 大而 P 小的类别。

```python
pa = torch.tensor([[0.9, 0.1]], dtype=torch.float64)
qa = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
# 手算：0.3681 与 0.5108，两个方向不相等
print(kl_div(pa, qa).item(), kl_div(qa, pa).item())
```

##### Follow-up 2：p_i=0 和 q_i=0, p_i>0 时分别怎么处理

$p_i=0$ 时，取极限 $\lim_{x\to 0^+}x\log x=0$，该项记 0，无论 $q_i$ 是否为 0。$p_i>0,q_i=0$ 时，KL 的数学结果是 $+\infty$。

```python
# 严格定义：返回 +inf，暴露 Q 的支撑集未覆盖 P
strict = kl_div(p2, q2).item()
# 训练时可平滑 Q，换来有限 loss；这时算的是修改后分布的 KL
q_s = (q2 + 1e-8) / (q2 + 1e-8).sum(-1, keepdim=True)
print(strict, kl_div(p2, q_s).item())
```

平滑属于工程处理，会改变目标函数。作为指标计算时，应保留 $+\infty$ 或单独报告。

##### Follow-up 3：输入是 logits、log-probs 还是概率，代码路径如何区分

三种输入都先转成 log-probs，再走同一条计算路径。logits 用 `log_softmax`，log-probs 直接使用，概率按主实现处理 0 的规则。

$$
\log q=\mathrm{log\_softmax}(z),\quad
D=\sum_c\exp(\log p_c)\,(\log p_c-\log q_c)
$$

```python
def kl_from_log(log_p, log_q):
    # log_p, log_q: (N, C) → term: (N, C)，逐类 KL 项
    term = log_p.exp() * (log_p - log_q)
    # log_p=-inf 时上式为 NaN，按 0·log0=0 替换；(N, C) → (N,)
    return torch.where(log_p.isneginf(), 0.0, term).sum(-1).mean()

torch.manual_seed(0)
# logits: (2, 4)，经 log_softmax 变成 log-probs，形状不变
z_p, z_q = torch.randn(2, 4), torch.randn(2, 4)
log_p, log_q = z_p.log_softmax(-1), z_q.log_softmax(-1)
print(kl_from_log(log_p, log_q).item())
# 概率路径
print(kl_div(log_p.exp(), log_q.exp()).item())
# 官方：input 恒为 log Q；log_target=True 表示 target 也是 log P
print(F.kl_div(log_q, log_p, reduction="batchmean",
               log_target=True).item())
```

从 logits 出发时不要先 softmax 再取 log，因为小概率在 softmax 后可能下溢成 0，`log_softmax` 没有这一步损失。`F.kl_div` 的方向也要注意：`input` 是 $\log Q$，`target` 是 P，写反就成了另一个方向的 KL。

#### Q021｜CPT / Causal LM Next-Token Loss

- **优先级：** P0 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 以 Decoder-only 的 Continued Pretraining / Causal LM 为前提，手写 Label Shift 后的 Token-Level CE；不能假定模型一定在外部 Shift。
- **输入输出 / 约束：** `logits:[B,T,V]`、`labels:[B,T]`，输出平均有效 token loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么是 `logits[:,:-1,:]` 对 `labels[:,1:]`？
  2. Causal Mask 和 Loss Mask 有什么不同？
  3. 如果 `model.forward(labels=...)` 已经在内部 Shift，外面再次 Shift 会怎样？
- **题面/来源说明：** “CPT Loss”在面经语境中可能指 Continual Pre-Training 的 next-token objective；用户原话保留，具体定义需按实际面试官说明。
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

##### 1️⃣ Dense Opening

Decoder-only 模型（Causal LM）在每个位置预测下一个 token，结构是“Label Shift → 逐位置 log-softmax → 取目标 token 的负对数似然 → 按有效 token 平均”，数学核心对象是形状为 $(B,T-1)$ 的逐 token 负对数似然矩阵 $\ell$。题面的 “CPT” 存在歧义，这里按题面说明处理为 Continued/Continual Pre-Training 的 next-token objective，若面试官另有定义，以其说明为准。

##### 2️⃣ Global Equations

Shift 后，位置 $t$ 的 logits 预测位置 $t+1$ 的 label：

$$
\ell_{b,t}=-\log\frac{\exp(z_{b,t,y_{b,t+1}})}{\sum_{v=1}^{V}\exp(z_{b,t,v})},\quad t=1,\dots,T-1
$$

有效掩码排除 `ignore_index`：

$$
m_{b,t}=\mathbb{1}\left[y_{b,t+1}\ne-100\right]
$$

主目标是有效 token 的平均 loss：

$$
\mathcal{L}=\frac{\sum_{b}\sum_{t=1}^{T-1}m_{b,t}\ell_{b,t}}{\sum_{b}\sum_{t=1}^{T-1}m_{b,t}}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def next_token_loss(logits: Tensor, labels: Tensor,
                    ignore_index: int = -100) -> Tensor:
    # (B, T, V) → (B, T-1, V)：丢弃最后一步，它没有下一个 token
    shift_logits = logits[:, :-1, :]
    # (B, T) → (B, T-1)：丢弃首 token，它没有上一步来预测
    shift_labels = labels[:, 1:]
    # 沿词表维归一化，形状不变：(B, T-1, V)
    log_probs = shift_logits.log_softmax(dim=-1)
    # (B, T-1) 的 bool，True 表示该目标 token 计入 loss
    valid = shift_labels != ignore_index
    # gather 的索引不能为负，把 -100 换成 0；(B, T-1)
    safe_labels = shift_labels.clamp(min=0)
    # 索引 (B, T-1, 1)，沿词表维取目标 log-prob → (B, T-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    # 掩码清零占位位置，再除以有效 token 数，得标量
    return (nll * valid).sum() / valid.sum()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def log_softmax_np(x, axis=-1):
    # 减最大值防上溢，形状不变
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))

def next_token_loss_np(logits, labels, ignore_index=-100):
    # (B, T, V) → (B, T-1, V)：丢弃最后一步
    shift_logits = logits[:, :-1, :]
    # (B, T) → (B, T-1)：丢弃首 token
    shift_labels = labels[:, 1:]
    # 沿词表维归一化：(B, T-1, V)
    log_probs = log_softmax_np(shift_logits, axis=-1)
    # (B, T-1) 的 bool，True 表示该目标 token 计入 loss
    valid = shift_labels != ignore_index
    # 把 -100 换成 0，索引才合法；(B, T-1)
    safe_labels = np.clip(shift_labels, 0, None)
    # 索引 (B, T-1, 1)，沿词表维取值 → (B, T-1)
    nll = -np.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1)[..., 0]
    # 掩码清零占位位置，再除以有效 token 数，得标量
    return (nll * valid).sum() / valid.sum()
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, T, V = 2, 5, 7
logits = torch.randn(B, T, V)
# 第 2 个样本尾部两个 token 是 padding，标记为 -100
labels = torch.tensor([[1, 2, 3, 4, 5], [6, 5, 4, -100, -100]])
ours = next_token_loss(logits, labels)
ours_np = next_token_loss_np(logits.numpy(), labels.numpy())
# 官方 API：展平为 (N, V) 与 (N,)，ignore_index 默认 -100
official = F.cross_entropy(
    logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
    ignore_index=-100)
print("PyTorch/NumPy/官方:", ours.item(), ours_np, official.item())
# 有效 token 数：第 1 行 4 个，第 2 行 2 个，共 6
print("有效 token:", (labels[:, 1:] != -100).sum().item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def next_token_loss(logits: Tensor, labels: Tensor,
                    ignore_index: int = -100) -> Tensor:
    pass
```

##### Step 1（i）公式：Label Shift

$$
\tilde z_{b,t}=z_{b,t},\quad \tilde y_{b,t}=y_{b,t+1},\quad t=1,\dots,T-1
$$

##### Step 1（ii）代码

```python
# (B, T, V) → (B, T-1, V)：丢弃最后一步，它没有下一个 token
shift_logits = logits[:, :-1, :]
# (B, T) → (B, T-1)：丢弃首 token，它没有上一步来预测
shift_labels = labels[:, 1:]
```

##### Step 1（iii）解释

Shift 写在 loss 函数内部，所以调用方传入与 `input_ids` 对齐的 `labels` 即可，不依赖模型或外部是否已经 Shift。

##### Step 2（i）公式：log-softmax

$$
\log\pi_{b,t,v}=z_{b,t,v}-\log\sum_{u=1}^{V}\exp(z_{b,t,u})
$$

##### Step 2（ii）代码

```python
log_probs = shift_logits.log_softmax(dim=-1)
```

##### Step 2（iii）解释

`dim=-1` 是词表维。这里没有先 softmax 再取 log，所以小概率不会因下溢变成 $\log 0$。

##### Step 3（i）公式：有效掩码

$$
m_{b,t}=\mathbb{1}\left[\tilde y_{b,t}\ne-100\right]
$$

##### Step 3（ii）代码

```python
valid = shift_labels != ignore_index
# gather 的索引不能为负，把 -100 换成 0；(B, T-1)
safe_labels = shift_labels.clamp(min=0)
```

##### Step 3（iii）解释

`gather` 遇到 -100 会报错，所以先换成合法索引 0。换入的 0 只是占位，下一步由掩码清零，不影响结果。

##### Step 4（i）公式：取目标 token 的 NLL

$$
\ell_{b,t}=-\log\pi_{b,t,\tilde y_{b,t}}
$$

##### Step 4（ii）代码

```python
# 索引 (B, T-1, 1)，沿词表维取目标 log-prob → (B, T-1)
nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
```

##### Step 4（iii）解释

`unsqueeze` 把索引补成三维，`gather` 在词表维只取 1 个值，`squeeze` 再去掉这一维。

##### Step 5（i）公式：按有效 token 平均

$$
\mathcal{L}=\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{\sum_{b,t}m_{b,t}}
$$

##### Step 5（ii）代码

```python
# 掩码清零占位位置，再除以有效 token 数，得标量
return (nll * valid).sum() / valid.sum()
```

##### Step 5（iii）解释

分母是有效 token 数，这与 `F.cross_entropy(..., ignore_index=-100)` 的 `mean` 语义一致，Demo 的数值也对齐。若整个 batch 没有有效标签，这里得到 NaN，官方 API 同样如此；Q022 处理这一边界。

##### 5️⃣ 复杂度

softmax 和 gather 的主项来自词表维：

$$
\text{时间}=O(BTV),\quad \text{额外空间}=O(BTV)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么是 logits[:,:-1,:] 对 labels[:,1:]

因果模型在位置 $t$ 只看到 $x_{1:t}$，输出分布是对下一个 token 的预测：

$$
z_{b,t}\ \longrightarrow\ p\left(x_{t+1}\mid x_{1:t}\right)
$$

最后一步的 logits 没有对应的下一个 token，首个 token 没有上一步来预测，因此两端各去掉一个，形状都变成 $T-1$。下面构造一个永远预测正确下一个 token 的 logits，只有对齐正确时 loss 才接近 0。

```python
ids = torch.tensor([[1, 2, 3, 4, 5]])
# 目标: (1, 4)，one_hot → (1, 4, V)，放大后作为完美预测
perfect = F.one_hot(ids[:, 1:], V).float() * 50.0
# 末尾补 1 步随意的 logits，(1, 4, V) → (1, 5, V)
perfect = torch.cat([perfect, torch.zeros(1, 1, V)], dim=1)
# 正确对齐，loss≈0
print(next_token_loss(perfect, ids).item())
# 不做 Shift，错位比较，loss 很大
print(F.cross_entropy(perfect.reshape(-1, V), ids.reshape(-1)).item())
```

##### Follow-up 2：Causal Mask 和 Loss Mask 有什么不同

Causal Mask 作用于注意力分数，控制信息流；Loss Mask 作用于逐 token loss，控制哪些位置进入目标函数。

$$
A_{ij}=0\ (j>i),\qquad
\mathcal{L}=\frac{\sum m\,\ell}{\sum m}
$$

| 项目 | Causal Mask | Loss Mask |
| --- | --- | --- |
| 形状 | $(T,T)$ | $(B,T-1)$ |
| 位置 | Attention 内部 | Loss 计算 |
| 缺失后果 | 模型看到未来，训练 loss 虚低 | padding 也被计入 loss |

```python
# Causal Mask：(T, T)，下三角 True，控制注意力可见范围
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
# Loss Mask：(B, T-1)，控制目标 token 是否计入
loss_mask = labels[:, 1:] != -100
print(causal.shape, loss_mask.shape, loss_mask.sum().item())
```

##### Follow-up 3：model.forward(labels=...) 已在内部 Shift，外面再次 Shift 会怎样

会 Shift 两次，代码不报错，但实际变成用 $x_{1:t}$ 预测 $x_{t+2}$，同时再少一个位置：

$$
\tilde y_{b,t}=y_{b,t+2},\quad t=1,\dots,T-2
$$

```python
# 外部先 Shift：(1, 5, V) → (1, 4, V)，(1, 5) → (1, 4)
outer_logits, outer_labels = perfect[:, :-1], ids[:, 1:]
# 内部再 Shift：完美预测器的 loss 也变得很大
print(next_token_loss(outer_logits, outer_labels).item())
```

Hugging Face 的 Causal LM 文档约定传入 `labels=input_ids`，由模型内部 Shift，具体行为以所用模型和版本的源码为准。判断方法是用上面的完美 logits 或小样本过拟合测试：对齐错误时，loss 无法降到接近 0。

#### Q022｜SFT Loss：Prompt Mask + Padding + Shift

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-07（原帖 Agent 实习问 SFT Loss；完整编码要求为扩展）
- **题目：** 针对对话 SFT，只在 assistant 的目标 token 上求 CE：支持 `ignore_index=-100`、padding、多轮对话。
- **输入输出 / 约束：** `logits:[B,T,V]`、`labels:[B,T]`；返回标量，并可返回有效 token 数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 是否把 assistant 起始 token 计入监督？解释标签生成与 Shift 的顺序。
  2. 如果一整个 batch 都没有有效标签，怎样避免 NaN？
  3. 按 token 平均与每样本平均会偏向短对话还是长对话？
- **题面/来源说明：** 字节同岗位有 SFT Loss 计算追问；完整 Prompt Mask 处理是练习扩展。
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

##### 1️⃣ Dense Opening

对话 SFT（Supervised Fine-Tuning）只监督 assistant 回复，结构是“生成与 `input_ids` 对齐的 labels（prompt 与 padding 置 -100）→ Shift → Token-Level CE → 按有效 token 平均”，数学核心对象是形状为 $(B,T-1)$ 的有效目标掩码 $m$。顺序必须是先生成 labels，再 Shift。

##### 2️⃣ Global Equations

标签生成，$a_{b,t}$ 表示 assistant 目标 token，$e_{b,t}$ 表示非 padding：

$$
y_{b,t}=
\begin{cases}
x_{b,t}, & a_{b,t}=1\ \text{且}\ e_{b,t}=1\\
-100, & \text{其他}
\end{cases}
$$

Shift 后的掩码与逐 token 负对数似然：

$$
m_{b,t}=\mathbb{1}\left[y_{b,t+1}\ne-100\right],\quad
\ell_{b,t}=-\log\pi_{b,t,y_{b,t+1}}
$$

主目标，分母下限取 1，全 -100 的 batch 得到 0：

$$
\mathcal{L}=\frac{\sum_{b}\sum_{t=1}^{T-1}m_{b,t}\ell_{b,t}}{\max\left(1,\ \sum_{b}\sum_{t=1}^{T-1}m_{b,t}\right)}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def build_labels(input_ids: Tensor, assistant_mask: Tensor,
                 attention_mask: Tensor) -> Tensor:
    # 三者都是 (B, T)；keep 为 True 时该 token 是有效监督目标
    keep = assistant_mask.bool() & attention_mask.bool()
    # (B, T) → (B, T)：prompt 与 padding 置 -100，尚未 Shift
    return input_ids.masked_fill(~keep, -100)

def sft_loss(logits: Tensor, labels: Tensor,
             ignore_index: int = -100) -> tuple:
    # (B, T, V) → (B, T-1, V)：丢弃最后一步
    shift_logits = logits[:, :-1, :]
    # (B, T) → (B, T-1)：位置 t 的目标是 t+1 的 label
    shift_labels = labels[:, 1:]
    # (B, T-1) 的 bool，True 表示该目标参与监督
    valid = shift_labels != ignore_index
    # 全 batch 的有效 token 数，标量
    n_valid = valid.sum()
    # 沿词表维归一化：(B, T-1, V)
    log_probs = shift_logits.log_softmax(dim=-1)
    # 把 -100 换成 0，gather 才合法；(B, T-1)
    safe_labels = shift_labels.clamp(min=0)
    # 索引 (B, T-1, 1)，沿词表维取值 → (B, T-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    # 分母下限为 1：全 -100 时 loss=0，且仍连接计算图
    loss = (nll * valid).sum() / n_valid.clamp(min=1)
    return loss, n_valid
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def log_softmax_np(x, axis=-1):
    # 减最大值防上溢，形状不变
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))

def build_labels_np(input_ids, assistant_mask, attention_mask):
    # 三者都是 (B, T)；keep 为 True 时该 token 是有效监督目标
    keep = assistant_mask.astype(bool) & attention_mask.astype(bool)
    # (B, T) → (B, T)：prompt 与 padding 置 -100，尚未 Shift
    return np.where(keep, input_ids, -100)

def sft_loss_np(logits, labels, ignore_index=-100):
    # (B, T, V) → (B, T-1, V)：丢弃最后一步
    shift_logits = logits[:, :-1, :]
    # (B, T) → (B, T-1)：位置 t 的目标是 t+1 的 label
    shift_labels = labels[:, 1:]
    # (B, T-1) 的 bool，True 表示该目标参与监督
    valid = shift_labels != ignore_index
    # 全 batch 的有效 token 数，标量
    n_valid = valid.sum()
    # 沿词表维归一化：(B, T-1, V)
    log_probs = log_softmax_np(shift_logits, axis=-1)
    # 把 -100 换成 0，索引才合法；(B, T-1)
    safe_labels = np.clip(shift_labels, 0, None)
    # 索引 (B, T-1, 1)，沿词表维取值 → (B, T-1)
    nll = -np.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1)[..., 0]
    # 分母下限为 1：全 -100 时 loss=0
    loss = (nll * valid).sum() / max(n_valid, 1)
    return loss, n_valid
```

##### 最小可运行测试（Demo）

```python
# token 约定：1 user，2 assistant 头，3~6 回复，7 EOS，0 PAD
# 样本 1 含两轮对话，样本 2 含一轮对话和 padding
ids = torch.tensor([[1, 2, 3, 4, 7, 1, 2, 5, 7],
                    [1, 1, 2, 6, 7, 0, 0, 0, 0]])
# 回复 token 与 EOS 为 1；user、assistant 头、PAD 为 0
asst = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0, 0, 0, 0]])
attn = (ids != 0).long()
labels = build_labels(ids, asst, attn)
print("labels:", labels.tolist())
torch.manual_seed(0)
logits = torch.randn(2, 9, 8, requires_grad=True)
loss, n = sft_loss(logits, labels)
loss_np, n_np = sft_loss_np(logits.detach().numpy(), labels.numpy())
# 官方 API 对齐：展平后 ignore_index 默认 -100
official = F.cross_entropy(
    logits[:, :-1].reshape(-1, 8), labels[:, 1:].reshape(-1))
print("ours/np/官方:", loss.item(), loss_np, official.item())
print("有效 token:", n.item(), n_np)
# 全 -100 batch：ours 与 NumPy 都为 0，官方 API 为 NaN
none = torch.full_like(labels, -100)
empty, n0 = sft_loss(logits, none)
empty_np = sft_loss_np(logits.detach().numpy(), none.numpy())[0]
official0 = F.cross_entropy(
    logits[:, :-1].reshape(-1, 8), none[:, 1:].reshape(-1))
print("全 -100:", empty.item(), empty_np, official0.item(), n0.item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def build_labels(input_ids: Tensor, assistant_mask: Tensor,
                 attention_mask: Tensor) -> Tensor:
    pass

def sft_loss(logits: Tensor, labels: Tensor,
             ignore_index: int = -100) -> tuple:
    pass
```

##### Step 1（i）公式：生成 Prompt Mask 后的 labels

$$
y_{b,t}=x_{b,t}\ \text{当}\ a_{b,t}e_{b,t}=1,\quad \text{否则}\ y_{b,t}=-100
$$

##### Step 1（ii）代码

```python
# 三者都是 (B, T)；keep 为 True 时该 token 是有效监督目标
keep = assistant_mask.bool() & attention_mask.bool()
# (B, T) → (B, T)：prompt 与 padding 置 -100，尚未 Shift
return input_ids.masked_fill(~keep, -100)
```

##### Step 1（iii）解释

此时 labels 与 `input_ids` 按位置对齐。多轮对话只需让 `assistant_mask` 覆盖每一段回复，无需改动其他代码。

##### Step 2（i）公式：Shift

$$
\tilde z_{b,t}=z_{b,t},\quad \tilde y_{b,t}=y_{b,t+1},\quad t=1,\dots,T-1
$$

##### Step 2（ii）代码

```python
# (B, T, V) → (B, T-1, V)：丢弃最后一步
shift_logits = logits[:, :-1, :]
# (B, T) → (B, T-1)：位置 t 的目标是 t+1 的 label
shift_labels = labels[:, 1:]
```

##### Step 2（iii）解释

Shift 放在标签生成之后。assistant 头的最后一个 token 在位置 $t$，它的 logits 负责预测第一个回复 token，这个对应关系由 Shift 自动得到。

##### Step 3（i）公式：统计有效目标

$$
m_{b,t}=\mathbb{1}\left[\tilde y_{b,t}\ne-100\right],\quad n=\sum_{b,t}m_{b,t}
$$

##### Step 3（ii）代码

```python
# (B, T-1) 的 bool，True 表示该目标参与监督
valid = shift_labels != ignore_index
# 全 batch 的有效 token 数，标量
n_valid = valid.sum()
```

##### Step 3（iii）解释

掩码必须基于 Shift 后的 labels 计算，这样它才与 `shift_logits` 按位置对齐。`n_valid` 同时用作分母和返回值。

##### Step 4（i）公式：逐 token 负对数似然

$$
\ell_{b,t}=-\log\pi_{b,t,\tilde y_{b,t}}
$$

##### Step 4（ii）代码

```python
log_probs = shift_logits.log_softmax(dim=-1)
# 把 -100 换成 0，gather 才合法；(B, T-1)
safe_labels = shift_labels.clamp(min=0)
# 索引 (B, T-1, 1)，沿词表维取值 → (B, T-1)
nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
```

##### Step 4（iii）解释

`clamp` 换入的 0 只是占位索引，这些位置在下一步乘以 $m=0$，不影响 loss 和梯度。

##### Step 5（i）公式：稳定平均

$$
\mathcal{L}=\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{\max(1,n)}
$$

##### Step 5（ii）代码

```python
# 分母下限为 1：全 -100 时 loss=0，且仍连接计算图
loss = (nll * valid).sum() / n_valid.clamp(min=1)
return loss, n_valid
```

##### Step 5（iii）解释

$n>0$ 时分母不变，结果与官方 `F.cross_entropy` 相同；$n=0$ 时得到 0。返回 `n_valid` 后，上层可以跨 batch 或跨卡按 token 数加权。

##### 5️⃣ 复杂度

主项来自词表维的 log-softmax：

$$
\text{时间}=O(BTV),\quad \text{额外空间}=O(BTV)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：是否把 assistant 起始 token 计入监督，标签生成与 Shift 的顺序

本实现把 assistant 头置 -100，把回复 token 和 EOS 计入监督。推理时 assistant 头由模板（generation prompt）给出，模型无需生成；EOS 决定回复何时停止，必须学习。若所用模板的推理流程要求模型自己生成某个起始 token，则该 token 要计入监督，训练与推理保持一致。

顺序是先生成对齐的 labels，再 Shift：

$$
y=\mathrm{mask}(x)\ \longrightarrow\ \tilde y_t=y_{t+1}
$$

下面对比错误顺序：先 Shift `ids`，再直接套用未 Shift 的 `asst`，掩码错位 1 格。

```python
def targets(shift_ids, shift_mask):
    # 两者 (T-1,)：取出第 1 个样本中被监督的目标 token
    return shift_ids[shift_mask.bool()].tolist()
# 正确：labels 已对齐，Shift 后非 -100 的就是监督目标
print("正确:", labels[0, 1:][labels[0, 1:] != -100].tolist())
# 错误：(T-1,) 的 ids[1:] 配 (T-1,) 的 asst[:-1]，错位 1 格
print("错位:", targets(ids[0, 1:], asst[0, :-1]))
```

正确结果是 `[3, 4, 7, 5, 7]`。错位结果是 `[4, 7, 1, 7]`：丢失了每段回复的第一个 token，还把 EOS 之后的 user token 当成监督目标。

##### Follow-up 2：整个 batch 都没有有效标签时如何防止 NaN

官方 `mean` 的分母是 0，结果为 $0/0$。把分母下限设为 1：

$$
\mathcal{L}=\frac{0}{\max(1,0)}=0
$$

```python
# 从 logits 构造，loss 为 0 但仍连接计算图
empty, _ = sft_loss(logits, torch.full_like(labels, -100))
empty.backward()
# backward 正常执行，梯度全为 0
print(empty.item(), logits.grad.abs().max().item())
```

这个 batch 的梯度全为 0。保留计算图后，DDP 下每张卡都能完成 `backward`，不会因个别卡跳过反向传播而卡住。

##### Follow-up 3：按 token 平均与每样本平均会偏向短对话还是长对话

按 token 平均时，每个 token 权重相同，所以长对话在 loss 中占比更大，偏向长对话。每样本平均时，每个对话总权重相同，所以短对话中单个 token 的权重更大，相对偏向短对话。

$$
\mathcal{L}_{\mathrm{token}}=\frac{\sum_b\sum_t m_{b,t}\ell_{b,t}}{\sum_b\sum_t m_{b,t}},\quad
\mathcal{L}_{\mathrm{sample}}=\frac{1}{B}\sum_b\frac{\sum_t m_{b,t}\ell_{b,t}}{\max\left(1,\sum_t m_{b,t}\right)}
$$

```python
# 长对话 100 个有效 token，loss 为 1；短对话 2 个，loss 为 3
nll_b = torch.zeros(2, 100)
nll_b[0], nll_b[1, :2] = 1.0, 3.0
m_b = torch.zeros(2, 100)
m_b[0], m_b[1, :2] = 1.0, 1.0
# 全部有效 token 求平均，标量
token_mean = (nll_b * m_b).sum() / m_b.sum().clamp(min=1)
# 先按样本维平均 (2, 100) → (2,)，再对样本平均
per_sample = (nll_b * m_b).sum(1) / m_b.sum(1).clamp(min=1)
# 手算：token 平均 106/102≈1.039，样本平均 (1+3)/2=2
print(token_mean.item(), per_sample.mean().item())
```

#### Q023｜CLIP 双向对比学习 Loss

- **优先级：** P0 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现图片向量和文本向量的归一化、两两相似度、温度缩放与 image-to-text / text-to-image 双向 CE。
- **输入输出 / 约束：** `img,txt:[B,D]`，输出标量；正样本默认为同一 batch 的对应行。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 标签为何是 `arange(B)`？若一张图对应多个描述怎么办？
  2. Temperature 变大/变小对梯度与模型区分度有什么影响？
  3. 分布式训练跨卡 Negative 时 logits 与标签如何构造？
- **题面/来源说明：** 先前调研存在 TikTok CLIP Loss 的二手说法，但本轮未独立核实到原帖，按扩展题收录。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

CLIP Loss 让同一 batch 中配对的图片和文本互相成为正样本，结构是“L2 归一化 → 两两余弦相似度 → 温度缩放 → image-to-text 与 text-to-image 双向 CE → 取平均”，数学核心对象是形状为 $(B,B)$ 的 logits 矩阵 $S$，其对角线是正样本。

##### 2️⃣ Global Equations

归一化与相似度矩阵：

$$
\hat I_i=\frac{I_i}{\|I_i\|_2},\quad \hat T_j=\frac{T_j}{\|T_j\|_2},\quad
S_{ij}=\frac{\hat I_i^\top\hat T_j}{\tau}
$$

两个方向的 CE：

$$
\mathcal{L}_{i2t}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ij})},\quad
\mathcal{L}_{t2i}=-\frac{1}{B}\sum_{j=1}^{B}\log\frac{\exp(S_{jj})}{\sum_{i=1}^{B}\exp(S_{ij})}
$$

主目标：

$$
\mathcal{L}=\frac{1}{2}\left(\mathcal{L}_{i2t}+\mathcal{L}_{t2i}\right)
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def clip_loss(img: Tensor, txt: Tensor,
              temperature: float = 0.07) -> Tensor:
    # (B, D) 沿特征维归一化，形状不变
    img = F.normalize(img, dim=-1)
    # (B, D) 沿特征维归一化，形状不变
    txt = F.normalize(txt, dim=-1)
    # (B, D) @ (D, B) → (B, B)；第 i 行是图 i 对所有文本的分数
    logits = img @ txt.T / temperature
    # (B,)：第 i 行的正样本在第 i 列
    labels = torch.arange(img.size(0))
    # 行方向分类：image → text，标量
    loss_i2t = F.cross_entropy(logits, labels)
    # (B, B) 转置后，第 j 行是文本 j 对所有图片的分数，标量
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def normalize_np(x):
    # (B, D) 沿特征维归一化，形状不变
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def cross_entropy_np(logits, labels):
    # 减每行最大值防上溢，(B, B) 形状不变
    x = logits - logits.max(axis=1, keepdims=True)
    # 沿行归一化成 log-prob：(B, B)
    log_p = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    # 花式索引取每行正样本，(B,) → 标量
    return -log_p[np.arange(len(labels)), labels].mean()

def clip_loss_np(img, txt, temperature=0.07):
    # (B, D) 沿特征维归一化，形状不变
    img, txt = normalize_np(img), normalize_np(txt)
    # (B, D) @ (D, B) → (B, B)
    logits = img @ txt.T / temperature
    # (B,)：第 i 行的正样本在第 i 列
    labels = np.arange(img.shape[0])
    # 两个方向分别做 CE，得到两个标量
    loss_i2t = cross_entropy_np(logits, labels)
    loss_t2i = cross_entropy_np(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, D = 4, 16
img, txt = torch.randn(B, D), torch.randn(B, D)
print("PT/NP:", clip_loss(img, txt).item(),
      clip_loss_np(img.numpy(), txt.numpy()))
# 交换 img 与 txt，两个方向互换，loss 不变
print("对称性:", clip_loss(txt, img).item())
# 完美配对：行向量正交，图文相同；对角线最大，loss 接近 0
eye = torch.eye(B)
print("完美配对:", clip_loss(eye, eye).item())
# tau 极大时 logits 接近全 0，loss 接近 log B=1.3863
print("tau 极大:", clip_loss(img, txt, 1e6).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def clip_loss(img: Tensor, txt: Tensor,
              temperature: float = 0.07) -> Tensor:
    pass
```

##### Step 1（i）公式：L2 归一化

$$
\hat I_i=\frac{I_i}{\|I_i\|_2},\quad \hat T_j=\frac{T_j}{\|T_j\|_2}
$$

##### Step 1（ii）代码

```python
# (B, D) 沿特征维归一化，形状不变
img = F.normalize(img, dim=-1)
# (B, D) 沿特征维归一化，形状不变
txt = F.normalize(txt, dim=-1)
```

##### Step 1（iii）解释

归一化后点积等于余弦相似度，取值范围是 $[-1,1]$，分数尺度只由温度控制。

##### Step 2（i）公式：相似度矩阵与温度

$$
S=\frac{\hat I\hat T^\top}{\tau}\in\mathbb{R}^{B\times B}
$$

##### Step 2（ii）代码

```python
# (B, D) @ (D, B) → (B, B)；第 i 行是图 i 对所有文本的分数
logits = img @ txt.T / temperature
```

##### Step 2（iii）解释

行对应图片，列对应文本。$S_{ii}$ 是配对正样本，同行或同列的其他元素是 batch 内负样本。

##### Step 3（i）公式：构造标签

$$
y_i=i,\quad i=0,\dots,B-1
$$

##### Step 3（ii）代码

```python
# (B,)：第 i 行的正样本在第 i 列
labels = torch.arange(img.size(0))
```

##### Step 3（iii）解释

正样本是对应行，所以类别下标就是行号，CE 在每一行做 $B$ 分类。

##### Step 4（i）公式：双向 CE 取平均

$$
\mathcal{L}=\frac{1}{2}\left[\mathrm{CE}(S,y)+\mathrm{CE}(S^\top,y)\right]
$$

##### Step 4（ii）代码

```python
# 行方向分类：image → text，标量
loss_i2t = F.cross_entropy(logits, labels)
# (B, B) 转置后，第 j 行是文本 j 对所有图片的分数，标量
loss_t2i = F.cross_entropy(logits.T, labels)
return (loss_i2t + loss_t2i) / 2
```

##### Step 4（iii）解释

转置后的 logits 让文本作为查询、图片作为候选。两个方向的分母不同，分别归一化每行和每列，所以需要同时计算。

##### 5️⃣ 复杂度

相似度矩阵是主项，CE 为 $O(B^2)$：

$$
\text{时间}=O(B^2D),\quad \text{激活显存}=O(B^2+BD)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：标签为何是 arange(B)，一张图对应多个描述怎么办

样本按行配对，第 $i$ 张图的正样本正好在第 $i$ 列，所以标签是 `arange(B)`。若一个 batch 中同一张图有多个描述，这些描述不再是负样本，把 one-hot 目标换成多正样本的软标签：

$$
\mathcal{L}_{i2t}=-\frac{1}{B}\sum_{i}\sum_{j}W_{ij}\log\pi_{ij},\quad
W_{ij}=\frac{\mathbb{1}[g_i=g_j]}{\sum_{l}\mathbb{1}[g_i=g_l]}
$$

```python
def clip_loss_multi(img, txt, group_ids, temperature=0.07):
    # 归一化后 (B, D) @ (D, B) → (B, B)
    logits = F.normalize(img, dim=-1) @ F.normalize(txt, dim=-1).T
    logits = logits / temperature
    # 广播 (B, 1) 与 (1, B) → (B, B)：同组为 1
    pos = (group_ids[:, None] == group_ids[None, :]).float()
    # 每行归一化成概率目标，(B, B)
    W = pos / pos.sum(1, keepdim=True)
    # F.cross_entropy 接受 (B, B) 的概率目标
    return (F.cross_entropy(logits, W) +
            F.cross_entropy(logits.T, W)) / 2

# 组号互不相同时，退化为标准 CLIP
print(clip_loss_multi(img, txt, torch.arange(B)).item())
# 第 0、1 条属于同一张图，它们之间不再互为负样本
print(clip_loss_multi(img, txt, torch.tensor([0, 0, 1, 2])).item())
```

##### Follow-up 2：Temperature 变大/变小对梯度与模型区分度的影响

对行 $i$ 的负样本 $j$，梯度权重是 softmax 概率：

$$
\frac{\partial\mathcal{L}_i}{\partial(\hat I_i^\top\hat T_j)}=\frac{\pi_{ij}}{\tau},\quad
\pi_{ij}=\frac{\exp(c_{ij}/\tau)}{\sum_l\exp(c_{il}/\tau)}
$$

$\tau$ 变小时，$\pi$ 集中到最难的负样本，区分度提高，但对标签噪声和假负样本更敏感。$\tau$ 变大时，负样本权重趋于均匀，区分度下降。总梯度权重 $\sum_j\pi_{ij}/\tau$ 不随 $\tau$ 单调变化：下例在 $\tau=1,0.07,0.01$ 时约为 0.58、2.76、0.0045，最后一项很小是因为正样本已占满概率。CLIP 原论文把温度设为可学习参数，初值等价于 0.07，并限制 logits 的缩放倍数不超过 100。

```python
# 第 0 行的余弦：正样本 0.9，两个负样本 0.8 和 0.1
c = torch.tensor([0.9, 0.8, 0.1])
for tau in (1.0, 0.07, 0.01):
    # (3,)：负样本的 softmax 权重
    w = (c / tau).softmax(-1)[1:]
    # 权重越集中，难负样本 0.8 的占比越高
    print(tau, (w / w.sum()).tolist(), (w / tau).sum().item())
```

##### Follow-up 3：分布式训练跨卡 Negative 时 logits 与标签如何构造

每张卡用本地样本作为查询，用 `all_gather` 得到的全局样本作为候选。设每卡 $B$ 个样本，共 $W$ 张卡，则 logits 形状是 $(B,WB)$，第 $r$ 张卡的标签要加上偏移：

$$
y_i=rB+i,\quad S^{(r)}=\frac{\hat I^{(r)}\hat T_{\mathrm{all}}^\top}{\tau}\in\mathbb{R}^{B\times WB}
$$

```python
def rank_loss(img_all, txt_all, rank, B_local, temperature=0.07):
    # 取本卡切片：(WB, D) → (B_local, D)
    start = rank * B_local
    idx = torch.arange(start, start + B_local)
    # 本地查询 × 全局候选：(B_local, D) @ (D, WB) → (B_local, WB)
    i2t = img_all[idx] @ txt_all.T / temperature
    t2i = txt_all[idx] @ img_all.T / temperature
    # 标签 idx 已含偏移 rank*B_local
    return (F.cross_entropy(i2t, idx) + F.cross_entropy(t2i, idx)) / 2

# 单进程模拟 2 张卡，全局 4 个样本，已归一化
img_n, txt_n = F.normalize(img, dim=-1), F.normalize(txt, dim=-1)
avg = sum(rank_loss(img_n, txt_n, r, 2) for r in range(2)) / 2
# 各卡等大时，各卡 loss 的平均等于全局 loss
print(avg.item(), clip_loss(img, txt).item())
```

`all_gather` 默认不传梯度。实践中有两种做法：只让本卡样本回传梯度，或用保留梯度的 gather。OpenCLIP 用 `local_loss` 和 `gather_with_grad` 两个选项控制这一行为。

#### Q024｜InfoNCE Loss：正负样本相似度矩阵

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 手写 InfoNCE：给定 Anchor、正样本及一组负样本，构造 temperature-scaled logits 并计算对比损失。
- **输入输出 / 约束：** `anchor:[B,D]`、`positive:[B,D]`、`negatives:[B,K,D]`；返回平均 loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如果每个 batch 其他样本都作为负例，标签和相似度矩阵如何改？
  2. 假负样本（false negative）会如何影响 loss？
  3. 需要 L2 normalize 吗？为什么温度会改变训练强度？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

##### 1️⃣ Dense Opening

InfoNCE 把“从 1 个正样本和 K 个负样本中认出正样本”写成 $(K+1)$ 分类，结构是“L2 归一化 → 计算正样本和负样本相似度 → 拼接并除以温度 → 第 0 类 CE”，数学核心对象是形状为 $(B,K+1)$ 的 logits 矩阵，第 0 列是正样本。

##### 2️⃣ Global Equations

归一化后的相似度：

$$
s^{+}_{b}=\hat a_b^\top\hat p_b,\quad s^{-}_{b,k}=\hat a_b^\top\hat n_{b,k}
$$

主目标：

$$
\mathcal{L}=-\frac{1}{B}\sum_{b=1}^{B}\log
\frac{\exp(s^{+}_{b}/\tau)}{\exp(s^{+}_{b}/\tau)+\sum_{k=1}^{K}\exp(s^{-}_{b,k}/\tau)}
$$

中间对象 logits，标签恒为 0：

$$
Z_b=\frac{1}{\tau}\left[s^{+}_{b},s^{-}_{b,1},\dots,s^{-}_{b,K}\right],\quad y_b=0
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def info_nce(anchor: Tensor, positive: Tensor, negatives: Tensor,
             temperature: float = 0.1) -> Tensor:
    # (B, D) 沿特征维归一化，形状不变
    a = F.normalize(anchor, dim=-1)
    # (B, D) 沿特征维归一化，形状不变
    p = F.normalize(positive, dim=-1)
    # (B, K, D) 沿特征维归一化，形状不变
    n = F.normalize(negatives, dim=-1)
    # 逐行点积 (B, D) → (B, 1)，第 b 行是 anchor 与正样本的相似度
    pos = (a * p).sum(-1, keepdim=True)
    # (B, 1, D) 广播乘 (B, K, D)，沿 D 求和 → (B, K)
    neg = (a.unsqueeze(1) * n).sum(-1)
    # 沿列拼接 (B, 1) 与 (B, K) → (B, 1+K)，再做温度缩放
    logits = torch.cat([pos, neg], dim=1) / temperature
    # (B,)：正样本固定在第 0 列
    labels = torch.zeros(a.size(0), dtype=torch.long)
    return F.cross_entropy(logits, labels)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def normalize_np(x):
    # 沿特征维归一化，形状不变
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def info_nce_np(anchor, positive, negatives, temperature=0.1):
    # (B, D)、(B, D)、(B, K, D) 沿特征维归一化
    a = normalize_np(anchor)
    p = normalize_np(positive)
    n = normalize_np(negatives)
    # 逐行点积 (B, D) → (B, 1)
    pos = (a * p).sum(-1, keepdims=True)
    # (B, 1, D) 广播乘 (B, K, D)，沿 D 求和 → (B, K)
    neg = (a[:, None, :] * n).sum(-1)
    # 沿列拼接 → (B, 1+K)，再做温度缩放
    logits = np.concatenate([pos, neg], axis=1) / temperature
    # 减最大值防上溢，(B, 1+K) 形状不变
    x = logits - logits.max(axis=1, keepdims=True)
    # 第 0 列的 log-prob：(B,)，再平均并取负号
    log_p0 = x[:, 0] - np.log(np.exp(x).sum(axis=1))
    return -log_p0.mean()
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, K, D = 4, 8, 16
anc, pos, negs = torch.randn(B, D), torch.randn(B, D), torch.randn(B, K, D)
print("PT/NP:", info_nce(anc, pos, negs).item(),
      info_nce_np(anc.numpy(), pos.numpy(), negs.numpy()))
# tau 极大时 logits 趋于 0，loss 趋于 log(K+1)=2.1972
print("tau 极大:", info_nce(anc, pos, negs, 1e6).item(), math.log(K + 1))
# 正样本等于 anchor，负样本取反，相似度为 ±1，loss 接近 0
print("完美分离:", info_nce(anc, anc, -anc[:, None].expand(B, K, D)).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def info_nce(anchor: Tensor, positive: Tensor, negatives: Tensor,
             temperature: float = 0.1) -> Tensor:
    pass
```

##### Step 1（i）公式：L2 归一化

$$
\hat a_b=\frac{a_b}{\|a_b\|_2},\quad
\hat p_b=\frac{p_b}{\|p_b\|_2},\quad
\hat n_{b,k}=\frac{n_{b,k}}{\|n_{b,k}\|_2}
$$

##### Step 1（ii）代码

```python
# (B, D) 沿特征维归一化，形状不变
a = F.normalize(anchor, dim=-1)
# (B, D) 沿特征维归一化，形状不变
p = F.normalize(positive, dim=-1)
# (B, K, D) 沿特征维归一化，形状不变
n = F.normalize(negatives, dim=-1)
```

##### Step 1（iii）解释

`negatives` 是三维输入，`dim=-1` 始终对应特征维 $D$，所以三个张量写法一致。

##### Step 2（i）公式：正负样本相似度

$$
s^{+}_{b}=\hat a_b^\top\hat p_b,\quad s^{-}_{b,k}=\hat a_b^\top\hat n_{b,k}
$$

##### Step 2（ii）代码

```python
# 逐行点积 (B, D) → (B, 1)，第 b 行是 anchor 与正样本的相似度
pos = (a * p).sum(-1, keepdim=True)
# (B, 1, D) 广播乘 (B, K, D)，沿 D 求和 → (B, K)
neg = (a.unsqueeze(1) * n).sum(-1)
```

##### Step 2（iii）解释

`pos` 保留第二维，才能和 `neg` 沿列拼接。`unsqueeze(1)` 让每个 anchor 复制到 $K$ 个负样本上。

##### Step 3（i）公式：拼接并做温度缩放

$$
Z_b=\frac{1}{\tau}\left[s^{+}_{b},s^{-}_{b,1},\dots,s^{-}_{b,K}\right]
$$

##### Step 3（ii）代码

```python
# 沿列拼接 (B, 1) 与 (B, K) → (B, 1+K)，再做温度缩放
logits = torch.cat([pos, neg], dim=1) / temperature
```

##### Step 3（iii）解释

正样本放在第 0 列，整个问题变成标准 $(K+1)$ 分类。

##### Step 4（i）公式：第 0 类 CE

$$
\mathcal{L}=-\frac{1}{B}\sum_{b}\log\frac{\exp(Z_{b,0})}{\sum_{l=0}^{K}\exp(Z_{b,l})}
$$

##### Step 4（ii）代码

```python
# (B,)：正样本固定在第 0 列
labels = torch.zeros(a.size(0), dtype=torch.long)
return F.cross_entropy(logits, labels)
```

##### Step 4（iii）解释

标签全为 0，因为每一行的正样本都在第 0 列。随机初始化且 $\tau$ 较大时，loss 接近 $\log(K+1)$，可用它快速检查实现。

##### 5️⃣ 复杂度

负样本相似度是主项：

$$
\text{时间}=O(BKD),\quad \text{激活显存}=O(BKD)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：batch 内其他样本作为负例时，标签和相似度矩阵如何改

相似度变成 $(B,B)$ 矩阵，对角线是正样本，标签从全 0 改为 `arange(B)`，每行的负样本数是 $B-1$：

$$
S_{ij}=\frac{\hat a_i^\top\hat p_j}{\tau},\quad y_i=i
$$

```python
def info_nce_in_batch(anchor, positive, temperature=0.1):
    # (B, D) @ (D, B) → (B, B)；S[i, j]：anchor i 对 positive j
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    S = a @ p.T / temperature
    # (B,)：第 i 行的正样本在第 i 列
    return F.cross_entropy(S, torch.arange(a.size(0)))

# 用其他行的 positive 显式构造 negatives，结果应与矩阵写法一致
neg_ib = torch.stack([
    # 第 i 行去掉自己：(B-1, D)，stack 后 (B, B-1, D)
    pos[torch.arange(B) != i] for i in range(B)])
print(info_nce_in_batch(anc, pos).item(),
      info_nce(anc, pos, neg_ib).item())
```

##### Follow-up 2：假负样本（false negative）会如何影响 loss

假负样本与 anchor 语义相同，却出现在分母中作为负例。设它与正样本的相似度相同，分母中有两个相等的大项，loss 产生下界：

$$
\mathcal{L}_b\ge-\log\frac{e^{s/\tau}}{e^{s/\tau}+e^{s/\tau}}=\log 2
$$

优化还会把语义相同的样本推开。温度越小，这类高相似度负样本的权重越大，影响越严重。常用处理是用标签或 ID 把已知假负样本从分母去掉。

```python
# 正样本是 anchor 本身，K 个负样本先全部取反：(B, K, D)
fn = -anc[:, None].repeat(1, K, 1)
# (B, K, D)：把第 0 个负样本替换成假负样本
fn[:, 0] = anc
# 其余负样本相似度为 -1，loss 仍停在 log2=0.6931
print(info_nce(anc, anc, fn).item(), math.log(2))
# 第 0 列是正样本，把假负样本的列置 -inf，从分母去掉
sim = torch.tensor([[1.0, 1.0, -1.0, -1.0]]) / 0.1
sim[0, 1] = float("-inf")
print(F.cross_entropy(sim, torch.zeros(1, dtype=torch.long)).item())
```

##### Follow-up 3：需要 L2 normalize 吗，为什么温度会改变训练强度

需要。不归一化时，模型可以通过放大向量模长来放大 logits，loss 下降不一定来自方向的改善；归一化后相似度范围固定为 $[-1,1]$，尺度由 $\tau$ 控制。

温度决定各负样本的梯度权重：

$$
\frac{\partial\mathcal{L}_b}{\partial s^{-}_{b,k}}=\frac{1}{\tau}\cdot
\frac{\exp(s^{-}_{b,k}/\tau)}{\exp(s^{+}_{b}/\tau)+\sum_{l}\exp(s^{-}_{b,l}/\tau)}
$$

$\tau$ 变小时，权重集中到相似度最高的负样本（hard negative），训练强度变大。$\tau$ 变大时，权重趋于均匀。

```python
# 放大向量模长：不归一化的点积随之放大，归一化后不变
raw = lambda a, p: (a * p).sum(-1)
print(raw(anc, pos)[0].item(), raw(10 * anc, pos)[0].item())
print(info_nce(10 * anc, pos, negs).item(), info_nce(anc, pos, negs).item())
# 三个负样本的相似度，查看最难负样本占的权重
s_neg = torch.tensor([0.8, 0.5, 0.1])
for tau in (1.0, 0.1, 0.02):
    print(tau, (s_neg / tau).softmax(-1)[0].item())
```

#### Q025｜NCE / Sampled Softmax 概念与代码片段

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 解释噪声对比估计的正负样本分类思路，并实现一个简化的负采样二分类 loss。
- **输入输出 / 约束：** 正样本 score、负样本 score，返回 loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 Negative 分布 `q(i)` 影响目标函数？
  2. NCE 与 InfoNCE、in-batch softmax 之间有何区别？
  3. 如何处理曝光分布不均导致的频繁热门物品？
- **题面/来源说明：** 公开面经记录为“介绍 NCE Loss”，未明确要求完整 NCE 编程；编码版为拓展。
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

##### 1️⃣ Dense Opening

噪声对比估计（Noise Contrastive Estimation，NCE）把“估计分布”转成“区分真实样本和噪声样本”的二分类，结构是“logQ 校正 → 正样本 logistic loss → K 个负样本 logistic loss → 求和平均”，数学核心对象是校正后的 logit $G(i)=s(i)-\log(Kq(i))$，其中 $q(i)$ 是负采样分布。题面要求的是简化版 loss，这里同时给出无校正（Negative Sampling）和有校正两条路径。

##### 2️⃣ Global Equations

每个正样本配 $K$ 个来自 $q$ 的负样本，后验概率与 logit 为：

$$
P(D=1\mid i)=\frac{p_d(i)}{p_d(i)+Kq(i)},\quad
\mathrm{logit}\,P(D=1\mid i)=\log p_d(i)-\log\left(Kq(i)\right)
$$

令模型分数 $s(i)$ 对应 $\log p_d(i)$，得到 logQ 校正：

$$
G(i)=s(i)-\log\left(Kq(i)\right)
$$

主目标：

$$
\mathcal{L}=\frac{1}{B}\sum_{b=1}^{B}\left[-\log\sigma\left(G(i_b^{+})\right)-\sum_{k=1}^{K}\log\sigma\left(-G(i_{b,k}^{-})\right)\right],\quad i_{b,k}^{-}\sim q
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def nce_loss(pos_score: Tensor, neg_score: Tensor,
             log_q_pos: Tensor = None,
             log_q_neg: Tensor = None) -> Tensor:
    # 输入：pos_score (B,)，neg_score (B, K)；K 为每行负样本数
    K = neg_score.size(1)
    # 传入 log q 时做校正 G=s-log(K q)，形状不变
    if log_q_pos is not None:
        # 正样本分数校正：(B,)
        pos_score = pos_score - (log_q_pos + math.log(K))
        # 负样本分数校正：(B, K)
        neg_score = neg_score - (log_q_neg + math.log(K))
    # -log σ(G) = softplus(-G)，形状 (B,)
    pos_loss = F.softplus(-pos_score)
    # -log σ(-G) = softplus(G)，沿负样本维求和 (B, K) → (B,)
    neg_loss = F.softplus(neg_score).sum(dim=1)
    # (B,) → 标量
    return (pos_loss + neg_loss).mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np
def softplus_np(x):
    # log(1+e^x)，logaddexp 防上溢，形状不变
    return np.logaddexp(0.0, x)

def nce_loss_np(pos_score, neg_score, log_q_pos=None, log_q_neg=None):
    # 输入：pos_score (B,)，neg_score (B, K)；K 为每行负样本数
    K = neg_score.shape[1]
    # 传入 log q 时做校正 G=s-log(K q)，形状不变
    if log_q_pos is not None:
        # 正样本分数校正：(B,)
        pos_score = pos_score - (log_q_pos + np.log(K))
        # 负样本分数校正：(B, K)
        neg_score = neg_score - (log_q_neg + np.log(K))
    # -log σ(G) = softplus(-G)，形状 (B,)
    pos_loss = softplus_np(-pos_score)
    # -log σ(-G) = softplus(G)，沿负样本维求和 (B, K) → (B,)
    neg_loss = softplus_np(neg_score).sum(axis=1)
    # (B,) → 标量
    return (pos_loss + neg_loss).mean()
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
B, K = 4, 3
pos, neg = torch.randn(B), torch.randn(B, K)
# 随机构造 log q：(B,) 与 (B, K)
lq_p, lq_n = -torch.rand(B) - 1, -torch.rand(B, K) - 1
print("无校正 PT/NP:", nce_loss(pos, neg).item(),
      nce_loss_np(pos.numpy(), neg.numpy()))
print("有校正 PT/NP:", nce_loss(pos, neg, lq_p, lq_n).item(),
      nce_loss_np(pos.numpy(), neg.numpy(), lq_p.numpy(), lq_n.numpy()))
# 官方 BCE 对齐：负样本求和，所以 sum 后再除以 B
bce = (F.binary_cross_entropy_with_logits(
    pos, torch.ones(B), reduction="sum") +
    F.binary_cross_entropy_with_logits(
    neg, torch.zeros(B, K), reduction="sum")) / B
print("官方 BCE:", bce.item())
# 手算：分数全 0 时每项 log2，共 (1+K)·log2=2.7726
print("全 0:", nce_loss(torch.zeros(B), torch.zeros(B, K)).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import math
import torch
import torch.nn.functional as F
from torch import Tensor

def nce_loss(pos_score: Tensor, neg_score: Tensor,
             log_q_pos: Tensor = None,
             log_q_neg: Tensor = None) -> Tensor:
    pass
```

##### Step 1（i）公式：logQ 校正

$$
G(i)=s(i)-\log q(i)-\log K
$$

##### Step 1（ii）代码

```python
K = neg_score.size(1)
# 传入 log q 时做校正 G=s-log(K q)，形状不变
if log_q_pos is not None:
    pos_score = pos_score - (log_q_pos + math.log(K))
    neg_score = neg_score - (log_q_neg + math.log(K))
```

##### Step 1（iii）解释

`log_q_pos` 和 `log_q_neg` 是候选 item 在负采样分布 $q$ 下的对数概率，正负样本都要校正，因为判别器比较的是 $p_d(i)$ 与 $Kq(i)$。不传时退化为 Negative Sampling，$s$ 直接作为 logit。

##### Step 2（i）公式：正样本项

$$
-\log\sigma(G^{+})=\log\left(1+e^{-G^{+}}\right)
$$

##### Step 2（ii）代码

```python
# -log σ(G) = softplus(-G)，形状 (B,)
pos_loss = F.softplus(-pos_score)
```

##### Step 2（iii）解释

用 `softplus` 计算，分数很大或很小时都稳定，不需要先算 `sigmoid` 再取 log。

##### Step 3（i）公式：负样本项

$$
-\sum_{k=1}^{K}\log\sigma\left(-G^{-}_{k}\right)=\sum_{k=1}^{K}\log\left(1+e^{G^{-}_{k}}\right)
$$

##### Step 3（ii）代码

```python
# -log σ(-G) = softplus(G)，沿负样本维求和 (B, K) → (B,)
neg_loss = F.softplus(neg_score).sum(dim=1)
```

##### Step 3（iii）解释

负样本沿 $K$ 维求和，这对应目标函数中的 $K\,\mathbb{E}_{q}$。如果改成平均，就等于换了正负权重，最优解也会变化。

##### Step 4（i）公式：Batch 平均

$$
\mathcal{L}=\frac{1}{B}\sum_{b=1}^{B}\left(\mathrm{pos\_loss}_b+\mathrm{neg\_loss}_b\right)
$$

##### Step 4（ii）代码

```python
# (B,) → 标量
return (pos_loss + neg_loss).mean()
```

##### Step 4（iii）解释

这是 $B$ 个正样本的平均值，分数全 0 时 loss 等于 $(1+K)\log 2$，可作为快速检查。

##### 5️⃣ 复杂度

每个正样本计算 1 个正分数和 $K$ 个负分数：

$$
\text{时间}=O(BK),\quad \text{额外空间}=O(BK)
$$

这里不计算分数本身的开销。全量 softmax 每步要算 $|\mathcal{I}|$ 个 item 的分数，负采样只算 $K+1$ 个，这是它的主要收益。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么 Negative 分布 q(i) 影响目标函数

对固定 item $i$，期望 loss 为：

$$
J(G)=p_d(i)\log\left(1+e^{-G}\right)+Kq(i)\log\left(1+e^{G}\right)
$$

令导数为 0，最优 logit 是：

$$
G^{*}(i)=\log\frac{p_d(i)}{Kq(i)}
$$

因此 $q$ 出现在最优解中。无校正时 $s^{*}=\log p_d-\log(Kq)$，模型学到的是相对 $q$ 的比值；有校正时 $s^{*}=\log p_d$。下面对小词表直接最小化期望 loss，核对两个结论。

```python
p_d = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
q_n = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
K_t = 4

def fit_table(use_log_q):
    # 每个 item 一个可学习分数：(C,)
    s = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([s], max_iter=200)
    # 有校正时从 s 减去 log(K q)，形状 (C,)
    shift = (K_t * q_n).log() if use_log_q else 0.0
    def closure():
        opt.zero_grad()
        G = s - shift
        # 按 p_d 和 K·q 加权的期望 loss，标量
        J = (p_d * F.softplus(-G) + K_t * q_n * F.softplus(G)).sum()
        J.backward()
        return J
    opt.step(closure)
    return s.detach()

# 有校正：s 恢复 log p_d，差值接近 0
print("校正:", (fit_table(True) - p_d.log()).abs().max().item())
# 无校正：s 恢复 log(p_d/(K q))
print("未校正:",
      (fit_table(False) - (p_d / (K_t * q_n)).log()).abs().max().item())
```

Negative Sampling（Mikolov et al., 2013）常用 $q\propto f^{3/4}$，并且不做这一校正，因此它学习的是这一比值。这对学习表示通常足够，但得到的不是 $\log p_d$。

##### Follow-up 2：NCE 与 InfoNCE、in-batch softmax 之间的区别

| 方法 | 判别方式 | 候选来源 | 校正后的最优 logit |
| --- | --- | --- | --- |
| NCE | 每个候选独立做二分类 | $K$ 个来自 $q$ | $G^{*}=\log\frac{p_d}{Kq}$ |
| InfoNCE | $K+1$ 选 1 的 softmax | $K$ 个来自 $q$ | $g^{*}=\log\frac{p_d}{q}+c$ |
| in-batch softmax | 同上，负样本来自 batch | batch 中其他正样本，$q$ 近似为流行度 | 同 InfoNCE |

InfoNCE 与 sampled softmax 的 logQ 校正没有 $\log K$，因为 softmax 中常数平移不影响结果：

$$
g(i)=s(i)-\log q(i),\quad
\mathcal{L}=-\log\frac{\exp(g(i^{+}))}{\exp(g(i^{+}))+\sum_{k=1}^{K}\exp(g(i_k^{-}))}
$$

```python
def sampled_softmax(pos_s, neg_s, lq_pos, lq_neg):
    # 校正后拼接 (B, 1) 与 (B, K) → (B, 1+K)
    logits = torch.cat([(pos_s - lq_pos)[:, None], neg_s - lq_neg], 1)
    # 正样本在第 0 列，(B,) 的全 0 标签
    return F.cross_entropy(logits, torch.zeros(len(pos_s)).long())

# q 均匀时 log q 是常数，校正不改变 softmax 结果
uni_p, uni_n = torch.full((B,), -2.0), torch.full((B, K), -2.0)
print(sampled_softmax(pos, neg, uni_p, uni_n).item(),
      F.cross_entropy(torch.cat([pos[:, None], neg], 1),
                      torch.zeros(B).long()).item())
```

##### Follow-up 3：如何处理曝光分布不均导致的频繁热门物品

in-batch 负样本的 $q(i)$ 近似等于 item 的曝光频率，所以热门 item 更常作为负样本。不校正时最优分数是 $\log(p(i\mid u)/q(i))$，热门 item 被过度压低。做 logQ 校正后，最优分数恢复 $\log p(i\mid u)$。下面用表格模型做随机模拟。

```python
p_true = torch.tensor([0.7, 0.15, 0.1, 0.05])
log_q = p_true.log()

def fit_softmax(alpha, steps=800, n=512, k=3):
    torch.manual_seed(0)
    # 每个 item 一个可学习分数：(C,)
    s = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([s], lr=0.05)
    for _ in range(steps):
        # 正样本 (n, 1)，负样本 (n, k)，都按流行度采样，模拟 in-batch
        cand = torch.multinomial(p_true, n * (1 + k), True).view(n, 1 + k)
        # 候选分数 (n, 1+k)；alpha=1 做 logQ 校正
        logits = s[cand] - alpha * log_q[cand]
        loss = F.cross_entropy(logits, torch.zeros(n).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 投影回全量词表的分布，(C,)
    return s.detach().softmax(-1)

print("真实:", p_true.tolist())
print("不校正:", fit_softmax(0.0).tolist())
print("校正:", fit_softmax(1.0).tolist())
```

不校正时学到的分布接近均匀，热门信号被抹去；校正后接近真实分布。工程上的处理方式如下。

- 流式估计频率：Yi et al.（RecSys 2019）用流式算法估计 $q(i)$，再校正 in-batch softmax。
- 混合负采样：Mixed Negative Sampling（Yang et al., 2020）混合 in-batch 负样本和按均匀分布采样的负样本，补充低曝光 item。
- 正样本降采样：对热门 item 的正样本按频率降采样，减少其在 loss 中的占比。
- Serving 控制：用超参数 $\alpha$ 调节 $s(i)-\alpha\log q(i)$，按线上指标决定保留多少流行度信号。

#### Q026｜MSE Loss 与线性回归梯度

- **优先级：** P0 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 写出均方误差并针对 `yhat=Xw+b` 推导、实现 `dw`、`db`。
- **输入输出 / 约束：** `X:[N,D]`、`w:[D]`、`y:[N]`；输出标量 loss 和梯度。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 `dw=(2/N)X.T@(yhat-y)` 而不是 `(yhat-y)@X`？
  2. 如果标签变为 `[N,1]`，怎样避免 NumPy 广播成 `[N,N]`？
  3. 把平均 MSE 改为总平方误差，梯度如何变化？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

##### 1️⃣ Dense Opening

线性回归的 MSE 梯度是一条“前向预测 → 残差 → 平方平均 → 残差投影回特征空间”的短计算图，数学核心对象是残差向量 $r=\hat y-y\in\mathbb{R}^{N}$。两个梯度都由 $r$ 得到：$dw$ 是 $r$ 在每个特征上的投影，$db$ 是残差均值的 2 倍。题面要求手写，所以 Step 以 NumPy 版为准。

##### 2️⃣ Global Equations

前向与残差：

$$
\hat y=Xw+b,\quad r=\hat y-y
$$

主目标：

$$
\mathcal{L}(w,b)=\frac{1}{N}\sum_{n=1}^{N}\left(x_n^\top w+b-y_n\right)^2=\frac{1}{N}r^\top r
$$

梯度：

$$
\nabla_w\mathcal{L}=\frac{2}{N}X^\top r,\quad
\frac{\partial\mathcal{L}}{\partial b}=\frac{2}{N}\mathbf{1}^\top r
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def mse_loss_grad(X: Tensor, w: Tensor, b: Tensor, y: Tensor):
    # X: (N, D)，w: (D,) → (N,)，再加标量 b 广播到 (N,)
    yhat = X @ w + b
    # (N,) - (N,) → (N,)，残差向量
    r = yhat - y
    # 样本数，梯度的归一化分母
    N = X.shape[0]
    # (N,) → 标量
    loss = (r ** 2).mean()
    # (D, N) @ (N,) → (D,)，把残差投影到每个特征
    dw = (2 / N) * (X.T @ r)
    # (N,) → 标量
    db = 2 * r.mean()
    return loss, dw, db
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def mse_loss_grad_np(X, w, b, y):
    # X: (N, D)，w: (D,) → (N,)，再加标量 b 广播到 (N,)
    yhat = X @ w + b
    # (N,) - (N,) → (N,)，残差向量
    r = yhat - y
    # 样本数，梯度的归一化分母
    N = X.shape[0]
    # (N,) → 标量
    loss = (r ** 2).mean()
    # (D, N) @ (N,) → (D,)，把残差投影到每个特征
    dw = (2 / N) * (X.T @ r)
    # (N,) → 标量
    db = 2 * r.mean()
    return loss, dw, db
```

##### 最小可运行测试（Demo）

```python
# 手算：r=[1, 2]，loss=2.5，dw=(2/2)(1·1+2·2)=5，db=2·1.5=3
X0 = np.array([[1.0], [2.0]])
print("手算:", mse_loss_grad_np(X0, np.array([1.0]), 0.0, np.zeros(2)))
torch.manual_seed(0)
X_t = torch.randn(5, 3, dtype=torch.float64)
y_t = torch.randn(5, dtype=torch.float64)
w_t = torch.randn(3, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
loss_t, dw_t, db_t = mse_loss_grad(X_t, w_t, b_t, y_t)
out_np = mse_loss_grad_np(X_t.numpy(), w_t.detach().numpy(),
                          0.5, y_t.numpy())
# 与 autograd 对齐：手写梯度应一致
loss_t.backward()
print("loss 差:", abs(loss_t.item() - out_np[0]))
print("dw 差:", abs(w_t.grad.numpy() - out_np[1]).max())
print("db 差:", abs(b_t.grad.item() - out_np[2]))
# 最小二乘解处梯度应接近 0：拼常数列 (N, D) → (N, D+1)
A0 = np.c_[X_t.numpy(), np.ones(5)]
sol = np.linalg.lstsq(A0, y_t.numpy(), rcond=None)[0]
print("最优解梯度:", mse_loss_grad_np(
    X_t.numpy(), sol[:3], sol[3], y_t.numpy())[1:])
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def mse_loss_grad_np(X, w, b, y):
    pass
```

##### Step 1（i）公式：前向预测

$$
\hat y=Xw+b\in\mathbb{R}^{N}
$$

##### Step 1（ii）代码

```python
# X: (N, D)，w: (D,) → (N,)，再加标量 b 广播到 (N,)
yhat = X @ w + b
```

##### Step 1（iii）解释

`X @ w` 对每行样本做点积。标量 $b$ 广播到全部样本，相当于 $X$ 多了一个全 1 的特征。

##### Step 2（i）公式：残差

$$
r=\hat y-y\in\mathbb{R}^{N}
$$

##### Step 2（ii）代码

```python
# (N,) - (N,) → (N,)，残差向量
r = yhat - y
# 样本数，梯度的归一化分母
N = X.shape[0]
```

##### Step 2（iii）解释

两者必须都是 `(N,)`。如果 `y` 是 `(N,1)`，这一行会静默广播成 `(N,N)`，见 Follow-up 2。

##### Step 3（i）公式：MSE

$$
\mathcal{L}=\frac{1}{N}r^\top r=\frac{1}{N}\sum_{n=1}^{N}r_n^2
$$

##### Step 3（ii）代码

```python
# (N,) → 标量
loss = (r ** 2).mean()
```

##### Step 3（iii）解释

`mean` 除以 $N$，所以 loss 尺度不随样本数变化。

##### Step 4（i）公式：对 w 求梯度

$$
\frac{\partial\mathcal{L}}{\partial w_d}=\frac{1}{N}\sum_{n}2r_n\frac{\partial r_n}{\partial w_d}=\frac{2}{N}\sum_{n}r_nX_{n,d}
\quad\Longrightarrow\quad
\nabla_w\mathcal{L}=\frac{2}{N}X^\top r
$$

##### Step 4（ii）代码

```python
# (D, N) @ (N,) → (D,)，把残差投影到每个特征
dw = (2 / N) * (X.T @ r)
```

##### Step 4（iii）解释

由 $\partial r_n/\partial w_d=X_{n,d}$ 和链式法则得到。结果与 $w$ 同为 `(D,)`，可以直接用于 `w -= lr * dw`。

##### Step 5（i）公式：对 b 求梯度

$$
\frac{\partial\mathcal{L}}{\partial b}=\frac{2}{N}\sum_{n}r_n
$$

##### Step 5（ii）代码

```python
# (N,) → 标量
db = 2 * r.mean()
return loss, dw, db
```

##### Step 5（iii）解释

$\partial r_n/\partial b=1$，因此 $db$ 是残差均值的 2 倍。Demo 中三项数值都与 autograd 对齐，最小二乘解处的梯度接近 0。

##### 5️⃣ 复杂度

前向和 $X^\top r$ 各做一次矩阵向量乘法：

$$
\text{时间}=O(ND),\quad \text{额外空间}=O(N+D)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么 dw=(2/N)X.T@(yhat-y) 而不是 (yhat-y)@X

题面这一问需要澄清：当 $r$ 是一维 `(N,)` 时，NumPy 把 `r @ X` 中的 $r$ 当作行向量，结果与 `X.T @ r` 数值相同，只差系数 $2/N$。写 `X.T @ r` 的理由如下。

$$
r^\top X=(X^\top r)^\top
$$

- 它与分母布局的矩阵公式 $\frac{2}{N}X^\top r$ 一一对应，结果形状就是 $w$ 的形状。
- 残差变成 `(N,1)` 时，`r @ X` 会因维度不匹配报错，而 `X.T @ r` 得到 `(D,1)`。

```python
X1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
r1 = np.array([1.0, -1.0, 2.0])
# 一维残差：两种写法数值相同，形状都是 (D,)
print(X1.T @ r1, r1 @ X1)
# (N, 1) 残差：X.T @ r → (D, 1)
r2 = r1[:, None]
print((X1.T @ r2).shape)
# (N, 1) @ (N, D) 内维不匹配，抛 ValueError
try:
    r2 @ X1
except ValueError as e:
    print("报错:", type(e).__name__)
```

##### Follow-up 2：标签为 [N,1] 时如何防止 NumPy 广播成 [N,N]

`yhat` 是 `(N,)`，`y` 是 `(N,1)` 时，广播规则先把前者看成 `(1,N)`，两者相减得到 `(N,N)`：

$$
(N,)\ -\ (N,1)\ \longrightarrow\ (1,N)\ -\ (N,1)\ \longrightarrow\ (N,N)
$$

此时代码不报错，但 loss 和梯度都是错的。修复方法是在入口把标签压成 `(N,)`。

```python
np.random.seed(0)
Xb = np.random.randn(4, 2)
wb, yb = np.array([1.0, -1.0]), np.random.randn(4)
# 错误：(4,) - (4, 1) → (4, 4)，每个预测与所有标签相减
bad = (Xb @ wb + 0.0) - yb[:, None]
print("错误残差:", bad.shape, "错误 loss:", (bad ** 2).mean())
# 错误传播：梯度从 (D,) 变成 (D, N)
print("错误 dw:", mse_loss_grad_np(Xb, wb, 0.0, yb[:, None])[1].shape)
# 修复：(4, 1) → (4,)，结果与一维标签相同
good = mse_loss_grad_np(Xb, wb, 0.0, yb[:, None].reshape(-1))
print("修复 loss:", good[0], "dw 形状:", good[1].shape)
print("基准 loss:", mse_loss_grad_np(Xb, wb, 0.0, yb)[0])
```

也可以在函数入口加 `assert yhat.shape == y.shape`，第一时间暴露这类错误。

##### Follow-up 3：把平均 MSE 改为总平方误差，梯度如何变化

去掉 $1/N$，loss 和两个梯度都变成原来的 $N$ 倍：

$$
\mathcal{L}_{\mathrm{sum}}=r^\top r=N\mathcal{L},\quad
\nabla_w\mathcal{L}_{\mathrm{sum}}=2X^\top r,\quad
\frac{\partial\mathcal{L}_{\mathrm{sum}}}{\partial b}=2\sum_{n}r_n
$$

最优解不变，但相同学习率下的更新步长变成 $N$ 倍。要保持同样的更新，学习率需要除以 $N$。

```python
def sse_grad(X, w, b, y):
    # (N,) 残差
    r = X @ w + b - y
    # 总平方误差与梯度：无 1/N
    return (r ** 2).sum(), 2 * (X.T @ r), 2 * r.sum()

m = mse_loss_grad_np(Xb, wb, 0.0, yb)
s = sse_grad(Xb, wb, 0.0, yb)
# 三项比值都等于 N=4
print(s[0] / m[0], s[1] / m[1], s[2] / m[2])
```

#### Q027｜PPO Clipped Policy Loss

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-09（面试时间）
- **题目：** 给定 old/new log_probs 与 advantage，写出并计算 PPO Clipped Objective（先固定 advantage）。
- **输入输出 / 约束：** `logp_new,logp_old,adv:[B,T]`、mask；返回 loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. `ratio=exp(new-old)` 如何避免极端 logits 的数值问题？
  2. clip 和 `min` 为什么必须按符号/目标方向正确处理？
  3. Value Loss、Entropy Bonus、KL 惩罚应否算在同一个标量里？
- **题面/来源说明：** 原始面经强调手写公式；完整可运行函数是模拟加练。
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

##### 1️⃣ Dense Opening

PPO 用裁剪后的重要性比值限制单次策略更新的幅度，结构是“log 差求 ratio → 计算两个 surrogate → 逐位置取 min → 按 mask 平均 → 取负号得到 loss”，数学核心对象是形状为 $(B,T)$ 的比值矩阵 $\rho=\pi_\theta/\pi_{\mathrm{old}}$。按题面固定 advantage，所以 `logp_old` 和 `adv` 都不回传梯度。

##### 2️⃣ Global Equations

比值由 log-prob 之差计算：

$$
\rho_{b,t}=\exp\left(\log\pi_\theta(a_{b,t}\mid s_{b,t})-\log\pi_{\mathrm{old}}(a_{b,t}\mid s_{b,t})\right)
$$

逐位置的 clipped objective：

$$
J_{b,t}=\min\left(\rho_{b,t}A_{b,t},\ \mathrm{clip}(\rho_{b,t},1-\epsilon,1+\epsilon)A_{b,t}\right)
$$

主目标：最大化 $J$，所以 loss 取负号：

$$
\mathcal{L}=-\frac{\sum_{b,t}m_{b,t}J_{b,t}}{\sum_{b,t}m_{b,t}}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def ppo_clip_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor,
                  mask: Tensor, clip_eps: float = 0.2) -> Tensor:
    # 固定 old 和 adv，切断梯度；三者形状都是 (B, T)
    logp_old, adv = logp_old.detach(), adv.detach()
    # log 空间做差再 exp，(B, T) → (B, T)
    ratio = (logp_new - logp_old).exp()
    # 未裁剪 surrogate，(B, T)
    surr1 = ratio * adv
    # 裁剪 ratio 后的 surrogate，(B, T)
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    # 逐位置取悲观下界，(B, T)
    obj = torch.min(surr1, surr2)
    # bool/int 掩码转浮点，(B, T)
    m = mask.to(obj.dtype)
    # 有效位置平均 → 标量；取负号，最大化变为最小化
    return -(obj * m).sum() / m.sum().clamp(min=1)
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def ppo_clip_loss_np(logp_new, logp_old, adv, mask, clip_eps=0.2):
    # log 空间做差再 exp，(B, T) → (B, T)
    ratio = np.exp(logp_new - logp_old)
    # 未裁剪 surrogate，(B, T)
    surr1 = ratio * adv
    # 裁剪 ratio 后的 surrogate，(B, T)
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    # 逐位置取悲观下界，(B, T)
    obj = np.minimum(surr1, surr2)
    # bool/int 掩码转浮点，(B, T)
    m = mask.astype(obj.dtype)
    # 有效位置平均 → 标量；取负号，最大化变为最小化
    return -(obj * m).sum() / max(m.sum(), 1)
```

##### 最小可运行测试（Demo）

```python
# ratio=[1.5, 1.5, 0.5, 0.5, 9.0]，第 5 个位置被 mask 掉
ratio0 = torch.tensor([[1.5, 1.5, 0.5, 0.5, 9.0]], dtype=torch.float64)
adv0 = torch.tensor([[1.0, -1.0, 1.0, -1.0, 7.0]], dtype=torch.float64)
mask0 = torch.tensor([[1, 1, 1, 1, 0]])
# old 取 0，则 new=log ratio；new 需要梯度
new0 = ratio0.log().requires_grad_()
old0 = torch.zeros_like(new0)
loss0 = ppo_clip_loss(new0, old0, adv0, mask0)
# 手算：逐位置 obj=[1.2, -1.5, 0.5, -0.8]，loss=-(-0.6)/4=0.15
print("loss:", loss0.item(), ppo_clip_loss_np(
    new0.detach().numpy(), old0.numpy(), adv0.numpy(), mask0.numpy()))
loss0.backward()
# 取到 clipped 项的位置梯度为 0；被 mask 的位置也为 0
print("grad:", new0.grad.tolist())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def ppo_clip_loss(logp_new: Tensor, logp_old: Tensor, adv: Tensor,
                  mask: Tensor, clip_eps: float = 0.2) -> Tensor:
    pass
```

##### Step 1（i）公式：计算比值

$$
\rho=\exp\left(\log\pi_\theta-\log\pi_{\mathrm{old}}\right)
$$

##### Step 1（ii）代码

```python
# 固定 old 和 adv，切断梯度；三者形状都是 (B, T)
logp_old, adv = logp_old.detach(), adv.detach()
# log 空间做差再 exp，(B, T) → (B, T)
ratio = (logp_new - logp_old).exp()
```

##### Step 1（iii）解释

`logp_old` 来自采样时的旧策略，`adv` 按题面先固定，都视为常数。梯度只经过 `logp_new`，且 $\partial\rho/\partial\log\pi_\theta=\rho$。

##### Step 2（i）公式：两个 surrogate

$$
S^{(1)}=\rho A,\quad S^{(2)}=\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)A
$$

##### Step 2（ii）代码

```python
# 未裁剪 surrogate，(B, T)
surr1 = ratio * adv
# 裁剪 ratio 后的 surrogate，(B, T)
surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
```

##### Step 2（iii）解释

裁剪对象是 ratio，不是 `ratio * adv`。$\rho$ 超出区间后，`clamp` 对它的导数为 0，第二项不再提供梯度。

##### Step 3（i）公式：取悲观下界

$$
J_{b,t}=\min\left(S^{(1)}_{b,t},S^{(2)}_{b,t}\right)
$$

##### Step 3（ii）代码

```python
# 逐位置取悲观下界，(B, T)
obj = torch.min(surr1, surr2)
```

##### Step 3（iii）解释

这里是两个张量逐元素取较小值。`torch.min` 传入两个张量时不聚合，形状保持 `(B, T)`。

##### Step 4（i）公式：Mask 平均并取负号

$$
\mathcal{L}=-\frac{\sum_{b,t}m_{b,t}J_{b,t}}{\max\left(1,\sum_{b,t}m_{b,t}\right)}
$$

##### Step 4（ii）代码

```python
# bool/int 掩码转浮点，(B, T)
m = mask.to(obj.dtype)
# 有效位置平均 → 标量；取负号，最大化变为最小化
return -(obj * m).sum() / m.sum().clamp(min=1)
```

##### Step 4（iii）解释

优化器做最小化，所以 PPO 目标要取负号。忘记负号时，策略会朝降低回报的方向更新，代码仍能运行。

##### 5️⃣ 复杂度

全部是逐元素运算：

$$
\text{时间}=O(BT),\quad \text{额外空间}=O(BT)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：ratio=exp(new-old) 如何处理极端 logits 的数值问题

分三层处理。第一，log-prob 用 `log_softmax` 计算，不走 `softmax` 再取 log。第二，比值在 log 空间做差，不计算两个概率再相除，否则两者都下溢时得到 $0/0$：

$$
\rho=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})\ne\frac{\exp(\log\pi_\theta)}{\exp(\log\pi_{\mathrm{old}})}\ \text{（浮点运算下）}
$$

第三，差值过大时 `exp` 仍然上溢，可在 `exp` 前限制 log 差。这是工程防护，不属于 PPO 原公式。

```python
# 极端 logits：softmax 下溢为 0，取 log 后为 -inf
z = torch.tensor([1000.0, -1000.0])
print(z.softmax(-1).log().tolist(), z.log_softmax(-1).tolist())
# 概率相除：两者都下溢，得到 NaN；log 差得到正常 ratio
ln, lo = torch.tensor(-800.0), torch.tensor(-800.1)
print((ln.exp() / lo.exp()).item(), (ln - lo).exp().item())
# float32 上 log 差为 100 时 exp 上溢，adv<0 得 -inf
d = torch.tensor(100.0)
r_big = d.exp()
print(r_big.item(), torch.min(-r_big, -r_big.clamp(0.8, 1.2)).item())
# 防护：先把 log 差限制到 ±20，再 exp
print(d.clamp(-20, 20).exp().item())
```

##### Follow-up 2：clip 和 min 为什么必须按符号/目标方向正确处理

clip 后的 ratio 还要乘以 $A$，$A$ 的符号决定哪一项更小。取 `min` 的结果是下界，所以策略无法靠大幅改变 ratio 继续提升目标：

| $A$ | $\rho$ | 取到的项 | 梯度 |
| --- | --- | --- | --- |
| $A>0$ | $\rho>1+\epsilon$ | clipped | 0，停止继续增大概率 |
| $A>0$ | $\rho<1-\epsilon$ | 未裁剪 | 非 0，继续增大概率 |
| $A<0$ | $\rho>1+\epsilon$ | 未裁剪 | 非 0，继续减小概率 |
| $A<0$ | $\rho<1-\epsilon$ | clipped | 0，停止继续减小概率 |

写成 loss 后，方向变为 `max`，因为 $-\min(a,b)=\max(-a,-b)$。如果对 loss 仍取 `min`，就变成乐观估计。

```python
r_t = torch.tensor([1.5, 1.5, 0.5, 0.5])
a_t = torch.tensor([1.0, -1.0, 1.0, -1.0])
# 两个 surrogate：(4,)
s1, s2 = r_t * a_t, r_t.clamp(0.8, 1.2) * a_t
# 正确：-min(s1, s2) 与 max(-s1, -s2) 相同
print("正确:", (-torch.min(s1, s2)).tolist(),
      torch.max(-s1, -s2).tolist())
# 错误：对 loss 取 min，A>0 且 ratio 变大时 loss 继续下降
print("错误:", torch.min(-s1, -s2).tolist())
# 错误：裁剪乘积，A=-1 时把 -1.5 截到 0.8，符号都变了
print("错误裁剪:", (r_t * a_t).clamp(0.8, 1.2).tolist())
```

##### Follow-up 3：Value Loss、Entropy Bonus、KL 惩罚应否算在同一个标量里

Value Loss 和 Entropy Bonus 通常放在一起。PPO 原论文给出的组合目标是最大化：

$$
J=J^{\mathrm{CLIP}}-c_1L^{\mathrm{VF}}+c_2S[\pi_\theta]
$$

对应 loss 取相反数。Actor 与 Critic 共享网络时通常合并成一个标量，由系数平衡共享部分的梯度；两者独立时可以合并，也可以分开优化，分开时学习率可以各自设置。

KL 惩罚是 PPO 论文中的另一种变体（Adaptive KL Penalty），与 Clip 版本是并列方案，所以常规 Clip 实现不把它加入 loss。工程上更常见的做法是记录近似 KL，用于监控或提前停止本轮更新。

```python
def ppo_total_loss(pg_loss, value, ret, entropy, m,
                   c1=0.5, c2=0.01):
    # 所有输入 (B, T)；分母是有效位置数
    n = m.sum().clamp(min=1)
    # Value Loss：有效位置的均方误差，标量
    vf = (((value - ret) ** 2) * m).sum() / n
    # Entropy Bonus：越大越好，所以在 loss 中减去
    ent = (entropy * m).sum() / n
    return pg_loss + c1 * vf - c2 * ent

torch.manual_seed(0)
ln2, lo2 = torch.randn(2, 3), torch.randn(2, 3)
adv2, m2 = torch.randn(2, 3), torch.ones(2, 3)
pg = ppo_clip_loss(ln2, lo2, adv2, m2)
total = ppo_total_loss(pg, torch.randn(2, 3), torch.randn(2, 3),
                       torch.rand(2, 3), m2)
# 近似 KL 只用于监控，no_grad 下计算，不进入 loss
with torch.no_grad():
    approx_kl = ((lo2 - ln2) * m2).sum() / m2.sum()
print(pg.item(), total.item(), approx_kl.item())
```

#### Q028｜GRPO Group Relative Advantage + Policy Loss

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-09（面试时间）
- **题目：** 给定同一 Prompt 的多个 Completion Reward，在组内计算归一化 advantage，再计算简化版 GRPO policy objective。
- **输入输出 / 约束：** `rewards:[B,G]`、token logprobs/ratio、token mask；输出 advantage 和 loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 组内 reward 全部相同，标准差为 0 怎么处理？
  2. Sequence-level reward 如何广播到 token，mask/归一化怎么选？
  3. GRPO 与 PPO 的 value model、KL 和 clip 分别有何差异？
- **题面/来源说明：** 原始面经强调公式；Coding 为扩展。
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

##### 1️⃣ Dense Opening

GRPO（Group Relative Policy Optimization，出自 DeepSeekMath，Shao et al., 2024）去掉 value model，用同一 Prompt 的多个 Completion 的组内 reward 统计量构造 advantage，结构是“组内归一化 → 广播到 token → PPO 式 clip surrogate → 减去到参考模型的 KL → 序列内平均、组内平均 → 取负号”，数学核心对象是形状为 $(B,G)$ 的组相对优势 $\hat A$。论文的 GRPO 目标同时包含 clip 和 KL 项，所以这里默认都实现；题面的简化版对应 $\beta=0$，只保留 policy 项。

##### 2️⃣ Global Equations

组内归一化，论文的结果监督（Outcome Supervision）把它赋给该输出的全部 token：

$$
\hat A_{b,i,t}=\tilde r_{b,i}=\frac{r_{b,i}-\mathrm{mean}(r_{b,1:G})}{\mathrm{std}(r_{b,1:G})}
$$

论文式 (3) 的目标，这里 $\rho_{i,t}=\pi_\theta(o_{i,t}\mid q,o_{i,<t})/\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})$：

$$
J=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\right)-\beta D_{\mathrm{KL}}\right]
$$

论文式 (4) 的 KL 估计量，这里 $x_{i,t}=\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})/\pi_\theta(o_{i,t}\mid q,o_{i,<t})$：

$$
D_{\mathrm{KL}}=x_{i,t}-\log x_{i,t}-1\ge 0
$$

实现要最小化 $\mathcal{L}=-J$ 对 batch 的平均。论文的公式没有写分母的小常数，也没有说明 std 用总体还是样本估计；这里采用总体 std（`unbiased=False`）并加 `std_eps`，这两点是实现选择。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
from torch import Tensor

def grpo_loss(rewards: Tensor, logp_new: Tensor, logp_old: Tensor,
              logp_ref: Tensor, mask: Tensor, clip_eps: float = 0.2,
              beta: float = 0.04, std_eps: float = 1e-4):
    # 组内均值：(B, G) → (B, 1)，dim=1 是组内维
    mean = rewards.mean(dim=1, keepdim=True)
    # 组内总体 std：(B, G) → (B, 1)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    # 广播归一化，adv: (B, G)，每个 completion 一个值
    adv = (rewards - mean) / (std + std_eps)
    # 序列级广播到 token：(B, G) → (B, G, 1) → (B, G, T)
    token_adv = adv.unsqueeze(-1).expand_as(logp_new)
    # old 和 ref 是冻结的；形状都是 (B, G, T)
    logp_old, logp_ref = logp_old.detach(), logp_ref.detach()
    # 重要性比值，(B, G, T)
    ratio = (logp_new - logp_old).exp()
    # 裁剪后的比值，(B, G, T)
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    # 逐 token 取下界，(B, G, T)
    surr = torch.min(ratio * token_adv, clipped * token_adv)
    # 论文式 (4) 的 log x，(B, G, T)
    log_x = logp_ref - logp_new
    # KL 估计量 x - log x - 1，(B, G, T)
    kl = log_x.exp() - log_x - 1
    # 逐 token 目标，(B, G, T)
    token_obj = surr - beta * kl
    # 掩码转浮点，(B, G, T)
    m = mask.to(token_obj.dtype)
    # 序列内平均 1/|o_i|：(B, G, T) → (B, G)
    seq_obj = (token_obj * m).sum(-1) / m.sum(-1).clamp(min=1)
    # 对 G 和 B 平均，取负号变 loss；同时返回 adv
    return adv, -seq_obj.mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def grpo_loss_np(rewards, logp_new, logp_old, logp_ref, mask,
                 clip_eps=0.2, beta=0.04, std_eps=1e-4):
    # 组内均值：(B, G) → (B, 1)
    mean = rewards.mean(axis=1, keepdims=True)
    # 组内总体 std：np.std 默认 ddof=0；(B, G) → (B, 1)
    std = rewards.std(axis=1, keepdims=True)
    # 广播归一化，adv: (B, G)
    adv = (rewards - mean) / (std + std_eps)
    # 序列级广播到 token：(B, G) → (B, G, 1)，乘法时再广播到 T
    token_adv = adv[..., None]
    # 重要性比值，(B, G, T)
    ratio = np.exp(logp_new - logp_old)
    # 裁剪后的比值，(B, G, T)
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    # 逐 token 取下界，(B, G, T)
    surr = np.minimum(ratio * token_adv, clipped * token_adv)
    # 论文式 (4) 的 log x，(B, G, T)
    log_x = logp_ref - logp_new
    # KL 估计量 x - log x - 1，(B, G, T)
    kl = np.exp(log_x) - log_x - 1
    # 逐 token 目标，(B, G, T)
    token_obj = surr - beta * kl
    # 掩码转浮点，(B, G, T)
    m = mask.astype(token_obj.dtype)
    # 序列内平均：(B, G, T) → (B, G)
    seq_obj = (token_obj * m).sum(-1) / np.maximum(m.sum(-1), 1)
    # 对 G 和 B 平均，取负号变 loss；同时返回 adv
    return adv, -seq_obj.mean()
```

##### 最小可运行测试（Demo）

```python
# 第 1 组：均值 2，总体 std≈0.8165，adv≈[-1.2247, 0, 1.2247]
# 第 2 组：reward 全相同，std=0，adv 全为 0
r = torch.tensor([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], dtype=torch.float64)
torch.manual_seed(0)
B, G, T = 2, 3, 4
lp = -torch.rand(B, G, T, dtype=torch.float64)
# 第 3 列之后按序列长度置 padding：(B, G, T)
tm = torch.ones(B, G, T)
tm[:, 1, 2:] = 0
# new=old=ref 时 ratio=1、KL=0，各序列目标等于 adv，组内和为 0
adv0, loss0 = grpo_loss(r, lp, lp, lp, tm)
print("adv:", adv0.tolist())
print("相同策略 loss:", loss0.item())
# 随机 new/old/ref：对比两个后端
ln = lp + 0.3 * torch.randn(B, G, T, dtype=torch.float64)
lr = lp + 0.3 * torch.randn(B, G, T, dtype=torch.float64)
pt = grpo_loss(r, ln, lp, lr, tm)[1].item()
npv = grpo_loss_np(r.numpy(), ln.numpy(), lp.numpy(), lr.numpy(),
                   tm.numpy())[1]
print("PT/NP:", pt, npv)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
from torch import Tensor

def grpo_loss(rewards: Tensor, logp_new: Tensor, logp_old: Tensor,
              logp_ref: Tensor, mask: Tensor, clip_eps: float = 0.2,
              beta: float = 0.04, std_eps: float = 1e-4):
    pass
```

##### Step 1（i）公式：组内归一化

$$
\hat A_{b,i}=\frac{r_{b,i}-\bar r_b}{\mathrm{std}(r_{b,1:G})+\delta}
$$

##### Step 1（ii）代码

```python
# 组内均值：(B, G) → (B, 1)，dim=1 是组内维
mean = rewards.mean(dim=1, keepdim=True)
# 组内总体 std：(B, G) → (B, 1)
std = rewards.std(dim=1, keepdim=True, unbiased=False)
# 广播归一化，adv: (B, G)，每个 completion 一个值
adv = (rewards - mean) / (std + std_eps)
```

##### Step 1（iii）解释

归一化只在每个 Prompt 的组内进行，组与组之间不混合。减去组均值后，它承担了 PPO 中 value model 提供的 baseline 作用，所以 GRPO 不需要 value model。

##### Step 2（i）公式：Reward 广播到 token

$$
\hat A_{b,i,t}=\hat A_{b,i},\quad t=1,\dots,|o_i|
$$

##### Step 2（ii）代码

```python
# 序列级广播到 token：(B, G) → (B, G, 1) → (B, G, T)
token_adv = adv.unsqueeze(-1).expand_as(logp_new)
```

##### Step 2（iii）解释

论文的结果监督把同一个归一化 reward 赋给该输出的全部 token，所以每个 token 的 advantage 相同。必须先补最后一维，才能按 token 维广播。

##### Step 3（i）公式：clip surrogate

$$
S_{i,t}=\min\left(\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\right)
$$

##### Step 3（ii）代码

```python
# old 和 ref 是冻结的；形状都是 (B, G, T)
logp_old, logp_ref = logp_old.detach(), logp_ref.detach()
# 重要性比值，(B, G, T)
ratio = (logp_new - logp_old).exp()
# 裁剪后的比值，(B, G, T)
clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
# 逐 token 取下界，(B, G, T)
surr = torch.min(ratio * token_adv, clipped * token_adv)
```

##### Step 3（iii）解释

这一项与 PPO 的形式相同，区别只在 advantage 的来源。论文实验中给出 $\beta=0.04$，我在实验设置段没有查到 $\epsilon$ 的取值，代码中的 0.2 只是常用默认值。

##### Step 4（i）公式：KL 估计量

$$
D_{\mathrm{KL}}=x_{i,t}-\log x_{i,t}-1,\quad \log x_{i,t}=\log\pi_{\mathrm{ref}}-\log\pi_\theta
$$

##### Step 4（ii）代码

```python
# 论文式 (4) 的 log x，(B, G, T)
log_x = logp_ref - logp_new
# KL 估计量 x - log x - 1，(B, G, T)
kl = log_x.exp() - log_x - 1
# 逐 token 目标，(B, G, T)
token_obj = surr - beta * kl
```

##### Step 4（iii）解释

由 $x-\log x-1\ge 0$ 可知该估计量非负，当且仅当 $x=1$ 时取 0。论文把 KL 直接放在 loss 中，没有放进 reward，所以 advantage 的计算不受它影响。

##### Step 5（i）公式：序列内和组内平均

$$
\mathcal{L}=-\frac{1}{BG}\sum_{b=1}^{B}\sum_{i=1}^{G}\frac{\sum_{t}m_{b,i,t}J_{b,i,t}}{\max\left(1,\sum_{t}m_{b,i,t}\right)}
$$

##### Step 5（ii）代码

```python
# 掩码转浮点，(B, G, T)
m = mask.to(token_obj.dtype)
# 序列内平均 1/|o_i|：(B, G, T) → (B, G)
seq_obj = (token_obj * m).sum(-1) / m.sum(-1).clamp(min=1)
# 对 G 和 B 平均，取负号变 loss；同时返回 adv
return adv, -seq_obj.mean()
```

##### Step 5（iii）解释

这与论文式 (3) 的 $\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t$ 对应：每个 Completion 先按自身长度平均，长短序列的总权重相同。Demo 中 new、old、ref 相同时，loss 为 0，因为 advantage 的组内和为 0。

##### 5️⃣ 复杂度

组内归一化是 $O(BG)$，主项来自 token 级运算：

$$
\text{时间}=O(BGT),\quad \text{额外空间}=O(BGT)
$$

这里不计算采样和前向的开销。论文实验中 $G=64$，主要开销在于每个问题要生成 $G$ 个输出。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：组内 reward 全部相同，标准差为 0 怎么处理

论文的公式写的是直接除以 std，没有说明这一边界。严格按公式计算是 $0/0$，实现时在分母加小常数：

$$
\hat A_{b,i}=\frac{r_{b,i}-\bar r_b}{\mathrm{std}_b+\delta}=\frac{0}{\delta}=0
$$

这一组的 policy 项梯度为 0，含义是组内没有相对优劣的信息。KL 项依然生效。还要注意两点：std 接近 0 时，$\delta$ 决定了噪声是否被放大；$G=1$ 时样本 std 为 NaN，总体 std 为 0。

```python
def group_adv(rw, eps):
    # (B, G) → (B, G)，总体 std 版本的组内归一化
    return (rw - rw.mean(1, keepdim=True)) / (
        rw.std(1, keepdim=True, unbiased=False) + eps)

same = torch.tensor([[1.0, 1.0, 1.0]])
# 无 eps：0/0=NaN；有 eps：0
print(group_adv(same, 0.0).tolist(), group_adv(same, 1e-4).tolist())
# 噪声级差异：无 eps 放大成 ±1，有 eps 只有约 ±0.005
tiny = torch.tensor([[1.0, 1.0 + 1e-6]], dtype=torch.float64)
print(group_adv(tiny, 0.0).tolist(), group_adv(tiny, 1e-4).tolist())
# G=1：样本 std 为 NaN（有告警）；总体 std 为 0
one = torch.tensor([[2.0]])
print(one.std(1, unbiased=False).item())
```

##### Follow-up 2：Sequence-level reward 如何广播到 token，mask/归一化怎么选

广播步骤已在主实现中，形状链为 $(B,G)\to(B,G,1)\to(B,G,T)$。如果漏掉 `unsqueeze`，当 $G=T$ 时可能静默广播到错误的维度。

```python
# 错误：(3, 3) 按尾部对齐到 (3, 3, 3)，组维被当成 token 维
a33 = torch.arange(9.0).view(3, 3)
wrong = a33 * torch.ones(3, 3, 3)
right = a33.unsqueeze(-1) * torch.ones(3, 3, 3)
# 第 0 组第 0 条序列：错误版每个 token 的值不同，正确版相同
print(wrong[0, 0].tolist(), right[0, 0].tolist())
```

归一化方式有两个选择，这里按论文式 (3) 取序列内平均。

$$
\mathcal{L}_{\mathrm{seq}}=-\frac{1}{BG}\sum_{b,i}\frac{\sum_t m\,J}{\sum_t m},\quad
\mathcal{L}_{\mathrm{token}}=-\frac{\sum_{b,i,t}m\,J}{\sum_{b,i,t}m}
$$

序列内平均让每条 Completion 权重相同。全局 token 平均让长序列权重更大：若组内好答案较短，错答案较长，长序列的负 advantage 项会主导，使 loss 偏向压低长回答。

```python
# 1 组 2 条：好答案 adv=+1 长度 2，差答案 adv=-1 长度 8
obj_t = torch.zeros(1, 2, 8)
obj_t[0, 0, :2], obj_t[0, 1] = 1.0, -1.0
m_t = torch.zeros(1, 2, 8)
m_t[0, 0, :2], m_t[0, 1] = 1.0, 1.0
# 序列内平均：(1, 2, 8) → (1, 2)，再求均值；结果 0
seq_avg = ((obj_t * m_t).sum(-1) / m_t.sum(-1)).mean()
# 全局 token 平均：(2-8)/10=-0.6，长序列主导
tok_avg = (obj_t * m_t).sum() / m_t.sum()
print(seq_avg.item(), tok_avg.item())
```

##### Follow-up 3：GRPO 与 PPO 的 value model、KL 和 clip 分别有何差异

以下按 DeepSeekMath 论文中的 PPO 式 (1)、(2) 和 GRPO 式 (3)、(4) 对比。

| 项目 | PPO（论文描述的 LLM RL 用法） | GRPO |
| --- | --- | --- |
| Value model | 需要，用 GAE 计算 $A_t$ | 不需要，用组内均值作为 baseline |
| Advantage | token 级 $A_t$ | 组内归一化 reward，结果监督下每个 token 相同 |
| KL | 每 token 的 KL 惩罚加入 reward | 加入 loss，使用式 (4) 的估计量 |
| Clip | 有，`min` 与 `clip` | 有，形式相同 |

需要说明的两点：

- PPO 原论文（Schulman et al., 2017）的 KL 是相对旧策略的自适应惩罚，与这里相对参考模型的 KL 不同，上表的 KL 对应论文式 (2) 引用的 RLHF 做法。
- 论文的 GRPO 目标包含 clip 和 KL。论文附录为简化分析，假设每次探索后只更新一次，因此 $\pi_{\mathrm{old}}=\pi_\theta$，推导时去掉 `min` 和 `clip`；这是分析上的简化。论文实验设置也写到每次探索后只更新一次。后续 DeepSeek-R1 等工作和各开源实现如何取舍这两项，我没有逐一核对，面试时按面试官使用的版本说明。

下面用 autograd 核对这一简化：$\rho=1$ 时，单 token 梯度系数等于论文式 (20) 的 $\hat A+\beta(x-1)$。

```python
# 1 个 prompt、2 条序列、每条 1 个 token：(1, 2, 1)
rw = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
lo = torch.log(torch.full((1, 2, 1), 0.5, dtype=torch.float64))
lf = torch.log(torch.full((1, 2, 1), 0.25, dtype=torch.float64))
ln1 = lo.clone().requires_grad_()
adv1, loss1 = grpo_loss(rw, ln1, lo, lf, torch.ones(1, 2, 1))
loss1.backward()
# 取负号抵消 loss 的负号，乘 B·G 抵消平均
coef = -ln1.grad * 2
# 论文系数：A + β(x-1)，x=0.25/0.5=0.5，β=0.04；(1,2) → (1,2,1)
expect = adv1.unsqueeze(-1) + 0.04 * (0.5 - 1)
print(coef.flatten().tolist(), expect.flatten().tolist())
```

#### Q029｜DPO Preference Loss

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-09（面试时间）
- **题目：** 给定 chosen/rejected 的策略与 reference 序列 logprob，实现 DPO 二元偏好优化损失。
- **输入输出 / 约束：** `pi_chosen,pi_rejected,ref_chosen,ref_rejected:[B]`、`beta`；返回标量。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么计算的是 chosen/rejected 的 log-ratio 差？
  2. `beta` 越大或越小会带来什么变化？
  3. 可否直接把 token 交叉熵当作 DPO？如何处理序列长度归一化？
- **题面/来源说明：** 原始面经强调手写公式；Coding 为扩展。
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

##### 1️⃣ Dense Opening

DPO（Direct Preference Optimization，Rafailov et al. 2023）用偏好对直接训练策略，跳过显式奖励模型和 RL 采样，结构是“序列 logprob → 相对 reference 的 log-ratio → 差值乘 $\beta$ → $-\log\sigma$”，数学核心对象是偏好 logit $h=\beta(r_w-r_l)$。输入 `pi_*` 和 `ref_*` 都是整条回复的 logprob，也就是各 token logprob 之和。

##### 2️⃣ Global Equations

主目标（原论文的 $-\log\sigma$ 形式）：

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]
$$

中间变量是两条回复的 log-ratio 和偏好 logit：

$$
r_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x),\quad r_l=\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)
$$

$$
h=\beta(r_w-r_l),\quad \ell=-\log\sigma(h)=\log(1+e^{-h})
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor,
             ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float = 0.1) -> Tensor:
    # chosen 的 log-ratio：(B,) - (B,) → (B,)
    r_chosen = pi_chosen - ref_chosen
    # rejected 的 log-ratio：(B,) - (B,) → (B,)
    r_rejected = pi_rejected - ref_rejected
    # 偏好 logit：(B,)
    h = beta * (r_chosen - r_rejected)
    # logsigmoid 逐样本求损失，mean 聚合：(B,) → 标量
    return -F.logsigmoid(h).mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def dpo_loss_np(pi_chosen, pi_rejected, ref_chosen, ref_rejected,
                beta=0.1):
    # chosen 的 log-ratio：(B,) - (B,) → (B,)
    r_chosen = pi_chosen - ref_chosen
    # rejected 的 log-ratio：(B,) - (B,) → (B,)
    r_rejected = pi_rejected - ref_rejected
    # 偏好 logit：(B,)
    h = beta * (r_chosen - r_rejected)
    # -log σ(h)=log(1+e^-h)，logaddexp 不溢出：(B,) → 标量
    return np.logaddexp(0.0, -h).mean()
```

##### 最小可运行测试（Demo）

```python
pi_w = torch.tensor([-10.0, -20.0], dtype=torch.float64)
pi_l = torch.tensor([-12.0, -20.0], dtype=torch.float64)
ref_w = torch.tensor([-11.0, -20.0], dtype=torch.float64)
ref_l = torch.tensor([-11.0, -20.0], dtype=torch.float64)
pt = dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=1.0).item()
# B 使用同一批输入，转成 ndarray
args = [t.numpy() for t in (pi_w, pi_l, ref_w, ref_l)]
print("后端差异:", abs(pt - dpo_loss_np(*args, beta=1.0)))
# 手算：h=[2, 0]，(log(1+e^-2)+log 2)/2
print("手算:", (np.log(1 + np.exp(-2)) + np.log(2)) / 2, "实现:", pt)
# 极端负 logit h=-200：sigmoid 下溢后手写 log 得到 inf
h = torch.tensor([-200.0])
print(-torch.log(torch.sigmoid(h)).item(), -F.logsigmoid(h).item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def dpo_loss(pi_chosen: Tensor, pi_rejected: Tensor,
             ref_chosen: Tensor, ref_rejected: Tensor,
             beta: float = 0.1) -> Tensor:
    pass
```

##### Step 1（i）公式：计算两条回复的 log-ratio

$$
r_w=\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x),\quad r_l=\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)
$$

##### Step 1（ii）代码

```python
# chosen 的 log-ratio：(B,) - (B,) → (B,)
r_chosen = pi_chosen - ref_chosen
# rejected 的 log-ratio：(B,) - (B,) → (B,)
r_rejected = pi_rejected - ref_rejected
```

##### Step 1（iii）解释

输入已经是 logprob，比值在 log 空间变成减法。`ref_*` 来自冻结的 reference 模型，不参与反向传播，梯度只流向 `pi_*`。

##### Step 2（i）公式：偏好 logit

$$
h=\beta(r_w-r_l)
$$

##### Step 2（ii）代码

```python
# 偏好 logit：(B,)
h = beta * (r_chosen - r_rejected)
```

##### Step 2（iii）解释

$h>0$ 表示相对 reference，策略提升 chosen 的幅度大于提升 rejected 的幅度。$\sigma(h)$ 就是模型给出的“chosen 优于 rejected”的概率。

##### Step 3（i）公式：$-\log\sigma$ 并取 batch 均值

$$
\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\log\sigma(h_i)
$$

##### Step 3（ii）代码

```python
# logsigmoid 逐样本求损失，mean 聚合：(B,) → 标量
return -F.logsigmoid(h).mean()
```

##### Step 3（iii）解释

这是标签恒为 1 的 BCE with logits。Demo 中 $h=-200$ 时，手写 `log(sigmoid(h))` 输出 `inf`，`logsigmoid` 输出正确值 200。

##### 5️⃣ 复杂度

损失本身只有逐元素运算；一次完整训练步还要算序列 logprob，设词表大小为 $V$、序列长度为 $T$：

$$
\text{损失时间}=O(B),\quad \text{logprob 时间}=O(BTV),\quad \text{损失额外显存}=O(B)
$$

一条样本要过 4 次前向：策略模型 2 次带梯度，reference 模型 2 次不带梯度。reference 的结果可以离线缓存。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为什么计算的是 chosen/rejected 的 log-ratio 差

KL 约束下的 RL 最优策略可以反解出奖励，其中配分函数 $Z(x)$ 依赖 prompt、难以计算：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

$$
r^*(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

代入 Bradley-Terry 偏好模型 $p(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$，同一 prompt 下两项 $\beta\log Z(x)$ 相减抵消，所以必须取差：

$$
p(y_w\succ y_l\mid x)=\sigma\left(\beta r_w-\beta r_l\right)
$$

下面用代码验证三点：给同一 prompt 的两条回复加同一常数，损失不变；$\pi_\theta=\pi_{\mathrm{ref}}$ 时损失为 $\log 2$；两条回复的梯度等大反向，大小为 $\beta\sigma(-h)$。

```python
pw = torch.tensor([-10.0], requires_grad=True)
pl = torch.tensor([-12.0], requires_grad=True)
rw, rl = torch.tensor([-11.0]), torch.tensor([-11.0])
loss = dpo_loss(pw, pl, rw, rl, beta=0.5)
loss.backward()
# 两条回复同加 7，模拟 Z(x) 这类公共项
shift = dpo_loss(pw + 7, pl + 7, rw, rl, beta=0.5)
print("平移前后:", loss.item(), shift.item())
print("π=ref:", dpo_loss(rw, rl, rw, rl).item(), "log2:", np.log(2))
# h=0.5*(1-(-1))=1，理论梯度 ∓0.5*σ(-1)
print(pw.grad.item(), pl.grad.item(),
      0.5 * torch.sigmoid(torch.tensor(-1.0)).item())
```

##### Follow-up 2：`beta` 越大或越小会带来什么变化

$\beta$ 同时出现在 logit 和梯度中：

$$
\frac{\partial\ell}{\partial\log\pi_\theta(y_w\mid x)}=-\beta\sigma(-h),\quad \Delta^*=\frac{\mathrm{logit}(p)}{\beta}
$$

若数据中 chosen 胜出的比例为 $p$，最优点满足 $\sigma(\beta\Delta^*)=p$，其中 $\Delta=r_w-r_l$。

- $\beta$ 大：较小的 $\Delta$ 就能达到目标偏好，策略贴近 reference，初始梯度 $\beta/2$ 也更大。
- $\beta$ 小：需要更大的 $\Delta$，策略可以偏离 reference 更远，初始梯度更小。

下面用两条回复的 softmax 策略和 $p=0.8$ 的数据验证 $\Delta^*=1.386/\beta$：

```python
for b in (0.5, 1.0, 5.0):
    theta = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.05)
    # 5 条偏好中 4 条选回复 0，1 条选回复 1，p=0.8
    idx_w = torch.tensor([0, 0, 0, 0, 1])
    idx_l = 1 - idx_w
    # reference 是均匀分布：(5,)
    ref = torch.full((5,), np.log(0.5))
    for _ in range(1500):
        # (2,) 按索引取值 → (5,)
        logp = theta.log_softmax(0)
        loss = dpo_loss(logp[idx_w], logp[idx_l], ref, ref, beta=b)
        opt.zero_grad(); loss.backward(); opt.step()
    # reference 均匀时 Δ=θ0-θ1
    print(b, (theta[0] - theta[1]).item(), np.log(4) / b)
```

##### Follow-up 3：可否直接把 token 交叉熵当作 DPO，如何处理序列长度归一化

不能。Token CE 只最大化 chosen 的 $\log\pi_\theta$，对应 SFT，其中没有 rejected 项和 reference 项。序列 CE 只是计算 logprob 的工具：

$$
\log\pi_\theta(y\mid x)=\sum_{t\in\text{response}}\log\pi_\theta(y_t\mid x,y_{<t})=-\mathrm{CE}_{\mathrm{sum}}
$$

长度处理：按 mask 对 response token 求和，不除以长度。这样得到的才是序列 logprob，配分函数才能按前面的推导抵消。长回复的 $\log\pi_\theta$ 绝对值更大，但减去 $\log\pi_{\mathrm{ref}}$ 后，初始 log-ratio 与长度无关，都是 0。

```python
def seq_logprob(logits, labels, mask):
    # logits:(B,T,V) → (B,T,V)，得到每个位置的对数概率
    logp = logits.log_softmax(-1)
    # labels:(B,T) → (B,T,1)，gather 取目标 token 后压回 (B,T)
    tok = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # mask 去掉 prompt 和 padding，按 T 求和：(B,T) → (B,)
    return (tok * mask).sum(-1)
torch.manual_seed(0)
logits, labels = torch.randn(2, 6, 9), torch.randint(0, 9, (2, 6))
# 样本 0 的 response 长 6，样本 1 的 response 长 2
mask = torch.tensor([[1.0] * 6, [1, 1, 0, 0, 0, 0]])
s = seq_logprob(logits, labels, mask)
# 取样本 0 对照官方 CE 求和：(6, 9) 与 (6,)
ce = F.cross_entropy(logits[0], labels[0], reduction="sum")
print("求和 logprob:", s.tolist(), "-CE:", -ce.item())
print("除以长度后:", (s / mask.sum(-1)).tolist())
```

#### Q030｜VAE：重构 Loss + KL 正则

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（面试日；11月发布）
- **题目：** 给定 VAE Decoder 输出、输入样本、潜变量分布参数，写出 ELBO 对应的重构项与 KL 项。
- **输入输出 / 约束：** `recon,x:[B,...]`、`mu,logvar:[B,Z]`；输出 reconstruction/KL/total。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 高斯潜变量相对 N(0,I) 的 KL 闭式解怎样推导？
  2. `logvar` 与 `std` 混淆会导致什么问题？
  3. `beta` 太大时为何可能出现 Posterior Collapse？
- **来源：** [牛客：字节大模型算法三面 BCE/VAE/积木堆叠](https://www.nowcoder.com/discuss/946798743163088896)

##### 1️⃣ Dense Opening

VAE（Variational Autoencoder）最大化 ELBO，损失由重构项和 KL 正则组成，结构是“逐样本重构误差 + 逐样本闭式 KL → 加权 → batch 平均”，数学核心对象是近似后验 $q_\phi(z\mid x)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))$ 相对先验 $\mathcal{N}(0,I)$ 的 KL。Encoder 输出 `logvar` 即 $\log\sigma^2$。

##### 2️⃣ Global Equations

ELBO 与训练目标（$\beta=1$ 时为标准 VAE）：

$$
\log p_\theta(x)\ge\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))
$$

$$
\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}\left(\mathcal{R}_i+\beta\mathcal{K}_i\right)
$$

重构项由似然决定。高斯似然这里取 $\sigma_x^2=\tfrac{1}{2}$，常数项省略，此时负对数似然恰好等于平方误差和。Bernoulli 似然对应 BCE，此时 `recon` 为 logits：

$$
\mathcal{R}_i^{\mathrm{Gauss}}=\|\hat{x}_i-x_i\|_2^2,\quad \mathcal{R}_i^{\mathrm{Bern}}=\sum_{j}\left[\log(1+e^{\hat{x}_{ij}})-x_{ij}\hat{x}_{ij}\right]
$$

KL 闭式解：

$$
\mathcal{K}_i=\frac{1}{2}\sum_{l=1}^{Z}\left(\mu_{il}^2+e^{\mathrm{logvar}_{il}}-\mathrm{logvar}_{il}-1\right)
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor,
             beta: float = 1.0, likelihood: str = "gauss"):
    # 高斯似然：逐元素平方误差，(B, ...) → (B, ...)
    if likelihood == "gauss":
        err = (recon - x) ** 2
    # Bernoulli 似然：recon 是 logits，(B, ...) → (B, ...)
    else:
        err = F.binary_cross_entropy_with_logits(
            recon, x, reduction="none")
    # 展平特征维后求和，(B, ...) → (B, N) → (B,)
    rec = err.reshape(x.shape[0], -1).sum(1)
    # 逐维 KL 沿潜变量维求和，(B, Z) → (B,)
    kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(-1)
    # 逐样本加权求和，(B,)
    total = rec + beta * kl
    # batch 平均，(B,) → 三个标量
    return rec.mean(), kl.mean(), total.mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def vae_loss_np(recon, x, mu, logvar, beta=1.0, likelihood="gauss"):
    # 按似然分支
    if likelihood == "gauss":
        # 逐元素平方误差，(B, ...) → (B, ...)
        err = (recon - x) ** 2
    else:
        # BCE with logits 的稳定形式，(B, ...) → (B, ...)
        err = np.logaddexp(0.0, recon) - x * recon
    # 展平特征维后求和，(B, ...) → (B, N) → (B,)
    rec = err.reshape(x.shape[0], -1).sum(1)
    # 逐维 KL 沿潜变量维求和，(B, Z) → (B,)
    kl = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).sum(-1)
    # 逐样本加权求和，(B,)
    total = rec + beta * kl
    # batch 平均，(B,) → 三个标量
    return rec.mean(), kl.mean(), total.mean()
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# 图像形状 x:(B, C, H, W)，潜变量 mu/logvar:(B, Z)
x, recon = torch.rand(2, 1, 4, 4), torch.randn(2, 1, 4, 4)
mu, logvar = torch.randn(2, 8), torch.randn(2, 8)
for mode in ("gauss", "bern"):
    pt = vae_loss(recon, x, mu, logvar, 0.5, mode)
    nu = vae_loss_np(recon.numpy(), x.numpy(), mu.numpy(),
                     logvar.numpy(), 0.5, mode)
    print(mode, [round(t.item(), 4) for t in pt],
          "后端差异:", max(abs(a.item() - b) for a, b in zip(pt, nu)))
# 手算：完美重构；mu=1, logvar=0 时每维 KL=0.5，8 维共 4
print(vae_loss(x, x, torch.ones(2, 8), torch.zeros(2, 8)))
# 后验等于先验时 KL 严格为 0
print(vae_loss(x, x, torch.zeros(2, 8), torch.zeros(2, 8))[1])
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F
from torch import Tensor

def vae_loss(recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor,
             beta: float = 1.0, likelihood: str = "gauss"):
    pass
```

##### Step 1（i）公式：逐元素重构误差

$$
\mathrm{err}^{\mathrm{Gauss}}_{ij}=(\hat{x}_{ij}-x_{ij})^2,\quad \mathrm{err}^{\mathrm{Bern}}_{ij}=-x_{ij}\log\sigma(\hat{x}_{ij})-(1-x_{ij})\log(1-\sigma(\hat{x}_{ij}))
$$

##### Step 1（ii）代码

```python
# 高斯似然：逐元素平方误差，(B, ...) → (B, ...)
if likelihood == "gauss":
    err = (recon - x) ** 2
# Bernoulli 似然：recon 是 logits，(B, ...) → (B, ...)
else:
    err = F.binary_cross_entropy_with_logits(
        recon, x, reduction="none")
```

##### Step 1（iii）解释

重构项是 $-\log p_\theta(x\mid z)$，所以损失形式由似然假设决定。BCE 分支传 logits，这样不必先过 sigmoid，数值更稳定。

##### Step 2（i）公式：沿特征维求和

$$
\mathcal{R}_i=\sum_j\mathrm{err}_{ij}
$$

##### Step 2（ii）代码

```python
# 展平特征维后求和，(B, ...) → (B, N) → (B,)
rec = err.reshape(x.shape[0], -1).sum(1)
```

##### Step 2（iii）解释

$\log p_\theta(x\mid z)$ 是所有像素的对数似然之和，所以这里求和。若用 `reduction="mean"`，重构项会缩小 $N$ 倍，等价于把 $\beta$ 放大 $N$ 倍。

##### Step 3（i）公式：KL 闭式解

$$
\mathcal{K}_i=\frac{1}{2}\sum_{l=1}^{Z}\left(\mu_{il}^2+e^{\mathrm{logvar}_{il}}-\mathrm{logvar}_{il}-1\right)
$$

##### Step 3（ii）代码

```python
# 逐维 KL 沿潜变量维求和，(B, Z) → (B,)
kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(-1)
```

##### Step 3（iii）解释

对角高斯的各维独立，所以 KL 可以按维相加。这里用闭式解，不需要采样，所以 KL 项没有蒙特卡洛噪声。

##### Step 4（i）公式：加权并取 batch 平均

$$
\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}\left(\mathcal{R}_i+\beta\mathcal{K}_i\right)
$$

##### Step 4（ii）代码

```python
# 逐样本加权求和，(B,)
total = rec + beta * kl
# batch 平均，(B,) → 三个标量
return rec.mean(), kl.mean(), total.mean()
```

##### Step 4（iii）解释

先逐样本求和，再对 batch 取平均，这样 $\beta$ 的含义不随 batch size 改变。三个返回值可以分别记录，用来监控 KL 是否塌到 0。

##### 5️⃣ 复杂度

设 $N$ 为单个样本的元素数，$Z$ 为潜变量维度：

$$
\text{时间}=O(B(N+Z)),\quad \text{额外显存}=O(BN+BZ)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：高斯潜变量相对 $\mathcal{N}(0,I)$ 的 KL 闭式解怎样推导

取单维 $q=\mathcal{N}(\mu,\sigma^2)$，$p=\mathcal{N}(0,1)$，按定义展开：

$$
D_{\mathrm{KL}}(q\|p)=\mathbb{E}_q[\log q(z)-\log p(z)]
$$

两个对数密度分别是：

$$
\log q(z)=-\frac{1}{2}\log(2\pi\sigma^2)-\frac{(z-\mu)^2}{2\sigma^2},\quad \log p(z)=-\frac{1}{2}\log(2\pi)-\frac{z^2}{2}
$$

代入后 $\log 2\pi$ 抵消：

$$
D_{\mathrm{KL}}(q\|p)=-\frac{1}{2}\log\sigma^2-\frac{\mathbb{E}_q[(z-\mu)^2]}{2\sigma^2}+\frac{\mathbb{E}_q[z^2]}{2}
$$

用 $\mathbb{E}_q[(z-\mu)^2]=\sigma^2$ 和 $\mathbb{E}_q[z^2]=\mu^2+\sigma^2$ 得到：

$$
D_{\mathrm{KL}}(q\|p)=-\frac{1}{2}\log\sigma^2-\frac{1}{2}+\frac{\mu^2+\sigma^2}{2}=\frac{1}{2}\left(\mu^2+\sigma^2-\log\sigma^2-1\right)
$$

对角协方差下各维独立，KL 按维相加。下面用官方 `kl_divergence` 和蒙特卡洛采样同时核对：

```python
from torch.distributions import Normal, kl_divergence
torch.manual_seed(0)
# 单个样本，Z=8：(1, 8)
m, lv = torch.randn(1, 8).double(), torch.randn(1, 8).double()
q, p = Normal(m, (0.5 * lv).exp()), Normal(0.0, 1.0)
# 闭式解：(1, 8) → (1,)
closed = vae_loss(m, m, m, lv)[1].item()
# 官方 API：逐维 KL (1, 8) → 标量
official = kl_divergence(q, p).sum().item()
# 采样：z:(S, 1, 8)，log q - log p 沿 Z 求和后对 S 取均值
z = q.sample((400000,))
mc = (q.log_prob(z) - p.log_prob(z)).sum(-1).mean().item()
print("闭式:", closed, "官方:", official, "采样:", mc)
```

##### Follow-up 2：`logvar` 与 `std` 混淆会导致什么问题

正确关系是下式。重参数化（Reparameterization）的采样方差要与 KL 中的 $e^{\mathrm{logvar}}$ 一致：

$$
\sigma=e^{\frac{1}{2}\mathrm{logvar}},\quad z=\mu+\sigma\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
$$

- `std = logvar.exp()`：实际方差变成 $e^{2\,\mathrm{logvar}}$，KL 惩罚的分布和采样的分布不一致。
- `z = mu + logvar * eps`：把可正可负的量当标准差，实际方差是 $\mathrm{logvar}^2$，且 `logvar=0` 处方差为 0。
- 把 `logvar` 当 $\sigma$ 代入 KL：`log(logvar**2)` 在 0 处发散，`log(logvar)` 在负值处得到 NaN。

```python
def reparameterize(mu, logvar):
    # std 与 eps 同形：(B, Z)
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)
torch.manual_seed(0)
# logvar=-2 时，真实方差应为 e^-2≈0.1353：(200000, 1)
lv = torch.full((200000, 1), -2.0)
eps = torch.randn_like(lv)
print("正确:", reparameterize(0 * lv, lv).var().item())
print("exp(logvar) 当 std:", (lv.exp() * eps).var().item())
print("logvar 当 std:", (lv * eps).var().item())
print("KL 中对负值取 log:", torch.log(torch.tensor(-2.0)).item())
```

##### Follow-up 3：`beta` 太大时为何可能出现 Posterior Collapse

后验塌缩（Posterior Collapse）指 $q_\phi(z\mid x)\approx p(z)$，此时 KL 约为 0，Decoder 忽略 $z$。KL 为 0 是一个总能达到的稳定点：当 $z$ 带来的重构收益小于 $\beta\mathcal{K}$ 的成本时，优化器就会选它。

线性高斯 VAE 可以直接算出阈值。本实现的重构项是平方误差和，所以 $\mathcal{L}/\beta$ 对应噪声方差 $\sigma_x^2=\beta/2$ 的 pPCA。设数据方差为 $s^2$，则 $s^2\le\sigma_x^2$ 时塌缩，即 $\beta\ge 2s^2$。$s^2=1$ 时：

$$
\mathcal{K}^*=\max\left(0,\;-\frac{1}{2}\log\frac{\beta}{2}\right)
$$

```python
for b in (0.5, 1.0, 1.5, 3.0, 8.0):
    torch.manual_seed(0)
    # 编码器 mu=a*x、方差 e^lv，解码器 recon=d*z，均为标量参数
    a = torch.tensor(0.5, requires_grad=True)
    lv = torch.tensor(-1.0, requires_grad=True)
    d = torch.tensor(0.5, requires_grad=True)
    opt = torch.optim.Adam([a, lv, d], lr=0.02)
    for _ in range(1500):
        # 数据方差为 1：x:(256, 1)
        x1 = torch.randn(256, 1)
        # lv 广播到 (256, 1)，每个样本同一方差
        lv_b = lv.expand_as(x1)
        z = reparameterize(a * x1, lv_b)
        loss = vae_loss(d * z, x1, a * x1, lv_b, b)[2]
        opt.zero_grad(); loss.backward(); opt.step()
    # E[(a x)^2]=a^2，直接写总体 KL
    kl = 0.5 * (a ** 2 + lv.exp() - lv - 1).item()
    print(b, "KL:", round(kl, 4), "理论:",
          round(max(0.0, -0.5 * np.log(b / 2)), 4),
          "|d|:", round(abs(d.item()), 4))
```

常见缓解手段有 KL 退火（KL annealing，$\beta$ 从 0 爬升）和 free bits。后者逐维保留 $\lambda$ 的 KL，低于该值时不再惩罚：

$$
\mathcal{K}^{\mathrm{free}}=\sum_{l}\max(\mathcal{K}_l,\lambda)
$$

#### Q031｜PPO / GRPO / DPO 公式推导对照

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（面试日期）
- **题目：** 在白板上写出三个目标函数中的采样来源、ratio 或 preference logit、KL/clip 项，并说明适用的数据结构。
- **输入输出 / 约束：** 输出不是程序，而是公式+参数含义+可运行伪代码。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 哪几种需要 old policy / reference policy / value model？
  2. On-policy 与 Off-policy 数据如何影响训练稳定性？
  3. 一条 rollout 的多个 token 到底如何汇总 loss？
- **题面/来源说明：** 面经直接记录的是手撕公式。
- **来源：** [牛客：字节大模型算法 MHA + DP、PPO/GRPO 公式](https://www.nowcoder.com/discuss/922308546966847488)

##### 1️⃣ Dense Opening

PPO、GRPO、DPO 都在限制新策略偏离某个基准策略，差别在信号来源和约束手段：PPO 用价值模型算优势并裁剪 ratio，GRPO 用同组回答的奖励归一化算优势并裁剪 ratio、外加 reference KL，DPO 用离线偏好对构造 logit。数学核心对象是 token 级 ratio $\rho_t$ 和序列级偏好 logit $h$。题面要求公式和可运行伪代码，这里按规范写成三个目标函数并列的统一实现，PPO 与 GRPO 共用一个裁剪函数。

##### 2️⃣ Global Equations

PPO（Schulman et al. 2017），最大化：

$$
J_{\mathrm{PPO}}=\mathbb{E}_t\left[\min\left(\rho_t\hat{A}_t,\;\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat{A}_t\right)-c_1(V_\phi(s_t)-V_t^{\mathrm{targ}})^2+c_2S[\pi_\theta](s_t)\right]
$$

GRPO（Shao et al. 2024，DeepSeekMath），同一问题采样 $G$ 个回答，最大化：

$$
J_{\mathrm{GRPO}}=\mathbb{E}_q\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left[\min\left(\rho_{i,t}\hat{A}_i,\;\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat{A}_i\right)-\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})\right]
$$

DPO（Rafailov et al. 2023），最小化：

$$
\mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]
$$

PPO 与 GRPO 共用的 ratio：

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
$$

白板对照表：

- 采样来源：PPO 是当前策略 rollout；GRPO 是当前策略每题采 $G$ 个回答；DPO 是离线偏好数据集。
- 核心量：PPO 和 GRPO 是 token ratio $\rho_t$；DPO 是 preference logit $h$。
- 约束项：PPO 是 clip；GRPO 是 clip 加 reference KL；DPO 是 $\beta$ 与 reference log-ratio。
- 优势信号：PPO 来自价值模型（GAE）；GRPO 来自组内奖励归一化；DPO 没有优势，用偏好标签。
- 数据结构：PPO 是 `(B,T)`；GRPO 是 `(Q,G,T)` 和奖励 `(Q,G)`；DPO 是序列 logprob `(B,)`。

$S$ 是策略熵，$\hat{A}_i=(r_i-\mathrm{mean}(r_{1:G}))/\mathrm{std}(r_{1:G})$。

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch
import torch.nn.functional as F

def masked_mean(x, mask):
    # 全部有效 token 的均值，(..., T) → 标量
    return (x * mask).sum() / mask.sum()
def clip_surrogate(logp, logp_old, adv, eps):
    # log 空间相减后取 exp，得到逐 token ratio：形状同 logp
    ratio = (logp - logp_old).exp()
    # ratio 裁剪到 [1-eps, 1+eps]：形状不变
    clipped = ratio.clamp(1 - eps, 1 + eps)
    # 逐元素取悲观下界：形状不变
    return torch.min(ratio * adv, clipped * adv)

def ppo_loss(logp, logp_old, adv, values, returns, ent, mask,
             eps=0.2, c1=0.5, c2=0.01):
    # 所有输入均为 (B, T)，逐 token 策略目标
    surr = clip_surrogate(logp, logp_old, adv, eps)
    # 价值回归误差：(B, T)
    v_err = (values - returns) ** 2
    # 合成最大化目标：(B, T)
    obj = surr - c1 * v_err + c2 * ent
    # 取负后最小化：(B, T) → 标量
    return -masked_mean(obj, mask)
def grpo_loss(logp, logp_old, logp_ref, rewards, mask,
              eps=0.2, beta=0.04):
    # 沿组维 G 算统计量：(Q, G) → (Q, 1)
    r_mean = rewards.mean(1, keepdim=True)
    r_std = rewards.std(1, keepdim=True, unbiased=False)
    # 组内归一化优势：(Q, G)，1e-4 处理全组奖励相同
    adv = (rewards - r_mean) / (r_std + 1e-4)
    # 补 token 维，广播到 logp 的 (Q, G, T)
    surr = clip_surrogate(logp, logp_old, adv[..., None], eps)
    # 对数差：(Q, G, T)
    delta = logp_ref - logp
    # 逐 token KL 估计，非负：(Q, G, T)
    kl = delta.exp() - delta - 1
    # 每条回答内按长度平均：(Q, G, T) → (Q, G)
    per_o = ((surr - beta * kl) * mask).sum(-1) / mask.sum(-1)
    # 对组和问题取均值并取负：(Q, G) → 标量
    return -per_o.mean()

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # 序列级 log-ratio 之差再乘 beta：(B,)
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    # (B,) → 标量
    return -F.logsigmoid(h).mean()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def masked_mean_np(x, mask):
    # 全部有效 token 的均值，(..., T) → 标量
    return (x * mask).sum() / mask.sum()
def clip_surrogate_np(logp, logp_old, adv, eps):
    # log 空间相减后取 exp，得到逐 token ratio：形状同 logp
    ratio = np.exp(logp - logp_old)
    # ratio 裁剪到 [1-eps, 1+eps]：形状不变
    clipped = np.clip(ratio, 1 - eps, 1 + eps)
    # 逐元素取悲观下界：形状不变
    return np.minimum(ratio * adv, clipped * adv)

def ppo_loss_np(logp, logp_old, adv, values, returns, ent, mask,
                eps=0.2, c1=0.5, c2=0.01):
    # 所有输入均为 (B, T)，逐 token 策略目标
    surr = clip_surrogate_np(logp, logp_old, adv, eps)
    # 价值回归误差：(B, T)
    v_err = (values - returns) ** 2
    # 合成最大化目标：(B, T)
    obj = surr - c1 * v_err + c2 * ent
    # 取负后最小化：(B, T) → 标量
    return -masked_mean_np(obj, mask)
def grpo_loss_np(logp, logp_old, logp_ref, rewards, mask,
                 eps=0.2, beta=0.04):
    # 沿组维 G 算统计量：(Q, G) → (Q, 1)
    r_mean = rewards.mean(1, keepdims=True)
    r_std = rewards.std(1, keepdims=True)
    # 组内归一化优势：(Q, G)，1e-4 处理全组奖励相同
    adv = (rewards - r_mean) / (r_std + 1e-4)
    # 补 token 维，广播到 logp 的 (Q, G, T)
    surr = clip_surrogate_np(logp, logp_old, adv[..., None], eps)
    # 对数差：(Q, G, T)
    delta = logp_ref - logp
    # 逐 token KL 估计，非负：(Q, G, T)
    kl = np.exp(delta) - delta - 1
    # 每条回答内按长度平均：(Q, G, T) → (Q, G)
    per_o = ((surr - beta * kl) * mask).sum(-1) / mask.sum(-1)
    # 对组和问题取均值并取负：(Q, G) → 标量
    return -per_o.mean()

def dpo_loss_np(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # 序列级 log-ratio 之差再乘 beta：(B,)
    h = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    # -log σ(h)=log(1+e^-h)：(B,) → 标量
    return np.logaddexp(0.0, -h).mean()
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# PPO 输入 (B, T)，GRPO 输入 (Q, G, T)，DPO 输入 (B,)
p = [torch.randn(2, 5).double() for _ in range(6)]
m1 = torch.ones(2, 5).double()
g = [torch.randn(2, 3, 5).double() for _ in range(3)]
r, m2 = torch.randn(2, 3).double(), torch.ones(2, 3, 5).double()
d = [torch.randn(4).double() for _ in range(4)]
# 三组目标函数分别对比两个后端
for f_pt, f_np, args in ((ppo_loss, ppo_loss_np, p + [m1]),
                         (grpo_loss, grpo_loss_np, g + [r, m2]),
                         (dpo_loss, dpo_loss_np, d)):
    a, b = f_pt(*args).item(), f_np(*[t.numpy() for t in args])
    print(f_pt.__name__, round(a, 6), "后端差异:", abs(a - b))
# 手算：ratio=2、eps=0.2。A=+1 取 1.2，A=-1 取 -2，裁剪只限制收益
lp = torch.log(torch.tensor([2.0, 2.0]))
adv = torch.tensor([1.0, -1.0])
print(clip_surrogate(lp, torch.zeros(2), adv, 0.2))
# 新旧策略与 reference 相同：GRPO 优势零和所以为 0，DPO 为 log 2
z = torch.zeros(2, 3, 5)
print(grpo_loss(z, z, z, torch.randn(2, 3), torch.ones(2, 3, 5)))
print(dpo_loss(*[torch.zeros(4)] * 4), np.log(2))
# 单点 KL 估计按 pi_theta 加权后等于真实 KL：(V,) → 标量
pt_, pr_ = torch.randn(9).softmax(0), torch.randn(9).softmax(0)
dl = pr_.log() - pt_.log()
print((pt_ * (dl.exp() - dl - 1)).sum().item(),
      (pt_ * (pt_.log() - pr_.log())).sum().item())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch
import torch.nn.functional as F

def masked_mean(x, mask):
    pass
def clip_surrogate(logp, logp_old, adv, eps):
    pass
def ppo_loss(logp, logp_old, adv, values, returns, ent, mask,
             eps=0.2, c1=0.5, c2=0.01):
    pass
def grpo_loss(logp, logp_old, logp_ref, rewards, mask,
              eps=0.2, beta=0.04):
    pass
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    pass
```

##### Step 1（i）公式：PPO 与 GRPO 共用的裁剪项

$$
\rho_t=e^{\log\pi_\theta-\log\pi_{\mathrm{old}}},\quad L_t^{\mathrm{clip}}=\min\left(\rho_t\hat{A}_t,\;\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat{A}_t\right)
$$

##### Step 1（ii）代码

```python
# log 空间相减后取 exp，得到逐 token ratio：形状同 logp
ratio = (logp - logp_old).exp()
# ratio 裁剪到 [1-eps, 1+eps]：形状不变
clipped = ratio.clamp(1 - eps, 1 + eps)
# 逐元素取悲观下界：形状不变
return torch.min(ratio * adv, clipped * adv)
```

##### Step 1（iii）解释

采样来自 $\pi_{\mathrm{old}}$，ratio 是重要性权重。`min` 让目标成为未裁剪目标的下界。Demo 中 ratio=2 时，$\hat{A}=+1$ 被限制为 1.2，$\hat{A}=-1$ 仍保留 -2。

##### Step 2（i）公式：PPO 完整目标

$$
J_{\mathrm{PPO}}=\frac{1}{\sum_{b,t}m_{b,t}}\sum_{b,t}m_{b,t}\left[L_{b,t}^{\mathrm{clip}}-c_1(V_{b,t}-V_{b,t}^{\mathrm{targ}})^2+c_2S_{b,t}\right]
$$

##### Step 2（ii）代码

```python
# 所有输入均为 (B, T)，逐 token 策略目标
surr = clip_surrogate(logp, logp_old, adv, eps)
# 价值回归误差：(B, T)
v_err = (values - returns) ** 2
# 合成最大化目标：(B, T)
obj = surr - c1 * v_err + c2 * ent
# 取负后最小化：(B, T) → 标量
return -masked_mean(obj, mask)
```

##### Step 2（iii）解释

`adv` 和 `returns` 来自价值模型与 GAE，这里作为常数传入，不反传。PPO 需要价值回归项，`ent` 是每个 token 位置的策略熵，由完整词表分布算出。

##### Step 3（i）公式：GRPO 组内优势

$$
\hat{A}_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})+10^{-4}}
$$

##### Step 3（ii）代码

```python
# 沿组维 G 算统计量：(Q, G) → (Q, 1)
r_mean = rewards.mean(1, keepdim=True)
r_std = rewards.std(1, keepdim=True, unbiased=False)
# 组内归一化优势：(Q, G)，1e-4 处理全组奖励相同
adv = (rewards - r_mean) / (r_std + 1e-4)
```

##### Step 3（iii）解释

论文公式的分母是 std，$10^{-4}$ 是工程上防止除零的补充。优势沿 $G$ 维归一化，同一条回答的所有 token 共享同一个 $\hat{A}_i$，所以 GRPO 不需要价值模型。这里用总体标准差，torch 须传 `unbiased=False` 才与 NumPy 默认一致。

##### Step 4（i）公式：GRPO 裁剪项与 KL 项

$$
\hat{D}_{i,t}=e^{\delta_{i,t}}-\delta_{i,t}-1,\quad \delta_{i,t}=\log\pi_{\mathrm{ref}}-\log\pi_\theta
$$

$$
\mathbb{E}_{\pi_\theta}[\hat{D}]=\mathbb{E}_{\pi_\theta}\left[\frac{\pi_{\mathrm{ref}}}{\pi_\theta}\right]+\mathbb{E}_{\pi_\theta}\left[\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right]-1=D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

##### Step 4（ii）代码

```python
# 补 token 维，广播到 logp 的 (Q, G, T)
surr = clip_surrogate(logp, logp_old, adv[..., None], eps)
# 对数差：(Q, G, T)
delta = logp_ref - logp
# 逐 token KL 估计，非负：(Q, G, T)
kl = delta.exp() - delta - 1
```

##### Step 4（iii）解释

论文写的是 KL，实现只有采样 token 的 logprob 时，用这个单点估计，TRL 的 GRPOTrainer 也用此式。第一项期望为 1，所以估计无偏；实际 token 来自 $\pi_{\mathrm{old}}$，因此只在两者接近时近似成立。Demo 用完整分布验证了这个恒等式。

##### Step 5（i）公式：GRPO 按回答和组聚合

$$
\mathcal{L}_{\mathrm{GRPO}}=-\frac{1}{QG}\sum_{q,i}\frac{1}{|o_i|}\sum_{t}m_{i,t}\left[L_{i,t}^{\mathrm{clip}}-\beta\hat{D}_{i,t}\right]
$$

##### Step 5（ii）代码

```python
# 每条回答内按长度平均：(Q, G, T) → (Q, G)
per_o = ((surr - beta * kl) * mask).sum(-1) / mask.sum(-1)
# 对组和问题取均值并取负：(Q, G) → 标量
return -per_o.mean()
```

##### Step 5（iii）解释

先除以每条回答自己的长度，再对回答取平均，所以每条回答的总权重相同。这与 PPO 对全部 token 统一平均不同。

##### Step 6（i）公式：DPO 偏好 logit

$$
h=\beta\left[(\log\pi_\theta(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_w\mid x))-(\log\pi_\theta(y_l\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x))\right],\quad \mathcal{L}=-\log\sigma(h)
$$

##### Step 6（ii）代码

```python
# 序列级 log-ratio 之差再乘 beta：(B,)
h = beta * ((pi_w - ref_w) - (pi_l - ref_l))
# (B,) → 标量
return -F.logsigmoid(h).mean()
```

##### Step 6（iii）解释

DPO 没有 ratio 裁剪，也不需要采样。输入是整条序列的 logprob，token 级求和已在外面完成。

##### 5️⃣ 复杂度

设 $T$ 为最大长度。损失部分都是逐元素运算；采样和前向的成本来自模型本身：

$$
\text{PPO}=O(BT),\quad \text{GRPO}=O(QGT),\quad \text{DPO}=O(B)
$$

每步前向次数：PPO 需要策略和价值模型；GRPO 每题生成 $G$ 条，还需要 reference 前向；DPO 每个偏好对需要策略和 reference 各 2 次前向。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：哪几种需要 old policy / reference policy / value model

- PPO：需要 old policy 和 value model，不需要 reference。
- GRPO：需要 old policy 和 reference，不需要 value model。
- DPO：只需要 reference，不需要 old policy 和 value model。

old 是生成当前批数据的策略，会随训练更新；reference 是固定基准。GRPO 取 $\beta=0$ 时 KL 项消失，也就用不到 reference：

```python
torch.manual_seed(0)
lp, lo = torch.randn(1, 4, 5), torch.randn(1, 4, 5)
r, mk = torch.randn(1, 4), torch.ones(1, 4, 5)
# 两个不同的 reference：(1, 4, 5)
ref_a, ref_b = torch.randn(1, 4, 5), torch.randn(1, 4, 5)
print("beta=0:", grpo_loss(lp, lo, ref_a, r, mk, beta=0.0).item(),
      grpo_loss(lp, lo, ref_b, r, mk, beta=0.0).item())
print("beta>0:", grpo_loss(lp, lo, ref_a, r, mk, beta=0.5).item(),
      grpo_loss(lp, lo, ref_b, r, mk, beta=0.5).item())
```

##### Follow-up 2：On-policy 与 Off-policy 数据如何影响训练稳定性

PPO 和 GRPO 的目标只在 $\pi_\theta$ 接近 $\pi_{\mathrm{old}}$ 时可靠：

$$
\mathbb{E}_{a\sim\pi_\theta}[A]=\mathbb{E}_{a\sim\pi_{\mathrm{old}}}[\rho A]
$$

数据只能复用几个 epoch。策略漂移后，$\hat{A}>0$ 的 token 进入裁剪区、梯度变 0，所以需要重新采样。稳定性风险来自采样方差和奖励噪声。

DPO 使用固定离线数据，没有采样方差，也不要求数据来自当前策略。它的限制是数据覆盖：训练中策略不会主动探索，数据集外的回答得不到反馈。

```python
# 漂移量 d：所有 token 的 log-ratio 都等于 d，A=+1
# PPO 打印目标（最大化）的梯度，DPO 打印损失（最小化）的梯度
for d in (0.0, 0.1, 0.3, 1.0):
    lp = torch.full((4,), d, requires_grad=True)
    clip_surrogate(lp, torch.zeros(4), torch.ones(4), 0.2).sum().backward()
    # DPO 的梯度不依赖 old，只随 h 平滑衰减
    pw = torch.full((4,), d, requires_grad=True)
    z4 = torch.zeros(4)
    (dpo_loss(pw, z4, z4, z4, beta=1.0) * 4).backward()
    print(d, "PPO 梯度:", round(lp.grad[0].item(), 4),
          "DPO 梯度:", round(pw.grad[0].item(), 4))
```

##### Follow-up 3：一条 rollout 的多个 token 到底如何汇总 loss

三者的权重不同：

$$
\text{PPO}:\frac{\sum_{b,t}m_{b,t}\ell_{b,t}}{\sum_{b,t}m_{b,t}},\quad \text{GRPO}:\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t\ell_{i,t},\quad \text{DPO}:\log\pi(y)=\sum_t\log\pi(y_t)
$$

- PPO：每个 token 是一个时间步，全局平均，长回答总权重更大。
- GRPO：先在回答内平均，每条回答总权重相同，同一回答的 token 共享优势。
- DPO：先把 token logprob 求和成序列 logprob，再进入一个 sigmoid，每个 token 的梯度都带同一个 $\beta\sigma(-h)$。

取长 4 和长 1 的两条回答，用 autograd 读出每个 token 的梯度权重：

```python
# 两条回答长度分别为 4 和 1：(2, 4)
m = torch.tensor([[1.0, 1, 1, 1], [1, 0, 0, 0]])
lp = torch.zeros(2, 4, requires_grad=True)
z24 = torch.zeros(2, 4)
# PPO：优势都为 1，只看策略项权重，每个 token 为 -1/5
adv_ppo = torch.ones(2, 4)
ppo = -masked_mean(clip_surrogate(lp, z24, adv_ppo, 0.2), m)
print("PPO:", torch.autograd.grad(ppo, lp)[0])
# GRPO：奖励 (1, 0)，优势约 (+1, -1)；(2, 4) 增加 Q 维到 (1, 2, 4)
lp3 = lp[None]
grpo = grpo_loss(lp3, lp3.detach(), lp3.detach(),
                 torch.tensor([[1.0, 0.0]]), m[None])
print("GRPO:", torch.autograd.grad(grpo, lp)[0])
# DPO：两条序列求和，h=0、beta=1 时每个 token 权重 ∓0.5
seq = (lp * m).sum(-1)
dpo = dpo_loss(seq[:1], seq[1:], torch.zeros(1), torch.zeros(1), 1.0)
print("DPO:", torch.autograd.grad(dpo, lp)[0])
```

#### Q032｜Top-p / Nucleus Sampling

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给定某一步 vocabulary logits 与概率阈值 p，先按概率排序，保留累计概率达到 p 的最小前缀，重归一化并抽样。
- **输入输出 / 约束：** `logits:[V]`、`0<p<=1`，返回采样 token id。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何必须包含“第一次使累计概率≥p”的 token？
  2. `p` 很小、ties、所有 logits 极小、temperature=0 时如何处理？
  3. 如何用 `torch.multinomial` 保证没有采样到被过滤 token？
- **题面/来源说明：** 第三方题库常见，未找到可独立核实的 TikTok 单场原帖。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

Top-p 采样（Nucleus Sampling）按概率从大到小保留一个最小前缀，使其累计概率首次达到 $p$，再重归一化并抽样，结构是“Softmax → 降序排序 → 累计概率定界 → 重归一化 → 映射回词表 → multinomial”，数学核心对象是截断位置 $K$。题面只给 `logits` 和 `p`；追问涉及 temperature，所以实现增加默认为 1 的 `temperature` 参数。

##### 2️⃣ Global Equations

设降序概率为 $q_1\ge q_2\ge\dots\ge q_V$，截断位置是：

$$
K=\min\left\{k:\sum_{i=1}^{k}q_i\ge p\right\}
$$

重归一化后的分布（$\mathrm{rank}(j)$ 是 token $j$ 的降序名次）：

$$
\tilde{\pi}_j=\frac{\pi_j\mathbb{1}[\mathrm{rank}(j)\le K]}{\sum_{i=1}^{K}q_i},\quad \pi=\mathrm{softmax}(\ell/\tau)
$$

代码中的等价判据是“当前 token 之前的累计概率小于 $p$”：

$$
\mathrm{rank}(j)\le K\iff\sum_{i<\mathrm{rank}(j)}q_i<p
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch

def nucleus_probs(logits, p, temperature=1.0):
    # 温度缩放后归一化：(V,) → (V,)
    probs = (logits / temperature).softmax(-1)
    # 降序排序：sorted_p:(V,)，idx[r] 是名次 r 的 token id
    sorted_p, idx = probs.sort(descending=True, stable=True)
    # 当前 token 之前的累计概率：(V,)，首位恒为 0
    before = sorted_p.cumsum(-1) - sorted_p
    # 布尔掩码：(V,)，包含首次使累计概率 ≥ p 的 token
    keep = before < p
    # 过滤并重归一化：(V,)，排序空间
    kept = sorted_p * keep / (sorted_p * keep).sum()
    # 从排序空间映射回词表空间：(V,) → (V,)
    return torch.zeros_like(probs).scatter(-1, idx, kept)

def top_p_sample(logits, p, temperature=1.0, gen=None):
    # temperature=0 无法做除法，退化为 greedy
    if temperature == 0:
        return logits.argmax(-1).item()
    # 得到过滤后的词表分布：(V,)
    probs = nucleus_probs(logits, p, temperature)
    # 按概率抽 1 个 id：(V,) → (1,) → int
    return torch.multinomial(probs, 1, generator=gen).item()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def nucleus_probs_np(logits, p, temperature=1.0):
    # 减最大值防止 exp 上溢：(V,)
    z = logits / temperature
    e = np.exp(z - z.max())
    # 归一化：(V,)
    probs = e / e.sum()
    # 降序排序：idx:(V,)，stable 保证并列时按 id 从小到大
    idx = np.argsort(-probs, kind="stable")
    # 按名次取概率：(V,) → (V,)
    sorted_p = probs[idx]
    # 当前 token 之前的累计概率：(V,)，首位恒为 0
    before = np.cumsum(sorted_p) - sorted_p
    # 布尔掩码：(V,)，包含首次使累计概率 ≥ p 的 token
    keep = before < p
    # 过滤并重归一化：(V,)，排序空间
    kept = sorted_p * keep / (sorted_p * keep).sum()
    # 从排序空间映射回词表空间：(V,) → (V,)
    out = np.zeros_like(probs)
    out[idx] = kept
    return out

def top_p_sample_np(logits, p, rng, temperature=1.0):
    # temperature=0 无法做除法，退化为 greedy
    if temperature == 0:
        return int(np.argmax(logits))
    # 得到过滤后的词表分布：(V,)
    probs = nucleus_probs_np(logits, p, temperature)
    # 按概率抽 1 个 id：(V,) → int
    return int(rng.choice(len(probs), p=probs))
```

##### 最小可运行测试（Demo）

```python
# 概率 [.15,.5,.05,.3] 的 logits，故意打乱顺序
base = np.array([0.15, 0.5, 0.05, 0.3])
lg_np = np.log(base)
lg = torch.from_numpy(lg_np)
a, b = nucleus_probs(lg, 0.6), nucleus_probs_np(lg_np, 0.6)
# 手算：保留 .5 和 .3，得到 id1=.625、id3=.375
print(a.tolist(), "后端差异:", abs(a.numpy() - b).max())
torch.manual_seed(0)
rng = np.random.default_rng(0)
# 两个后端各采样 5000 次，统计每个 id 的频率
s_pt = [top_p_sample(lg, 0.6) for _ in range(5000)]
s_np = [top_p_sample_np(lg_np, 0.6, rng) for _ in range(5000)]
print(np.bincount(s_pt, minlength=4) / 5000)
print(np.bincount(s_np, minlength=4) / 5000)
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch

def nucleus_probs(logits, p, temperature=1.0):
    pass

def top_p_sample(logits, p, temperature=1.0, gen=None):
    pass
```

##### Step 1（i）公式：温度缩放与归一化

$$
\pi=\mathrm{softmax}(\ell/\tau)
$$

##### Step 1（ii）代码

```python
# 温度缩放后归一化：(V,) → (V,)
probs = (logits / temperature).softmax(-1)
```

##### Step 1（iii）解释

温度要在排序和定界之前作用，因为它改变累计概率，从而改变截断位置。

##### Step 2（i）公式：降序排序

$$
q_r=\pi_{\mathrm{idx}_r},\quad q_1\ge q_2\ge\dots\ge q_V
$$

##### Step 2（ii）代码

```python
# 降序排序：sorted_p:(V,)，idx[r] 是名次 r 的 token id
sorted_p, idx = probs.sort(descending=True, stable=True)
```

##### Step 2（iii）解释

`idx` 记录名次到 token id 的映射，Step 5 要用它映射回去。`stable=True` 让并列概率按 id 排序，结果可复现。

##### Step 3（i）公式：确定截断位置

$$
\mathrm{before}_r=\sum_{i<r}q_i,\quad \mathrm{keep}_r=\mathbb{1}[\mathrm{before}_r<p]
$$

##### Step 3（ii）代码

```python
# 当前 token 之前的累计概率：(V,)，首位恒为 0
before = sorted_p.cumsum(-1) - sorted_p
# 布尔掩码：(V,)，包含首次使累计概率 ≥ p 的 token
keep = before < p
```

##### Step 3（iii）解释

用“之前的累计概率”比较，第一个使累计概率达到 $p$ 的 token 也会被保留。首位的 `before` 为 0，又有 $p>0$，所以至少保留一个 token。

##### Step 4（i）公式：重归一化

$$
\tilde{q}_r=\frac{q_r\mathrm{keep}_r}{\sum_{i}q_i\mathrm{keep}_i}
$$

##### Step 4（ii）代码

```python
# 过滤并重归一化：(V,)，排序空间
kept = sorted_p * keep / (sorted_p * keep).sum()
```

##### Step 4（iii）解释

被过滤位置的概率严格为 0。分母是保留 token 的总质量，至少为 $p$，所以不会除零。

##### Step 5（i）公式：映射回词表空间

$$
\tilde{\pi}_{\mathrm{idx}_r}=\tilde{q}_r
$$

##### Step 5（ii）代码

```python
# 从排序空间映射回词表空间：(V,) → (V,)
return torch.zeros_like(probs).scatter(-1, idx, kept)
```

##### Step 5（iii）解释

这是 Step 2 排序的逆映射。`scatter` 把第 $r$ 名的概率写回 `idx[r]` 位置，所以后面抽到的下标就是 token id。

##### Step 6（i）公式：抽样与温度为 0

$$
y\sim\mathrm{Categorical}(\tilde{\pi}),\quad \tau=0\Rightarrow y=\arg\max_j\ell_j
$$

##### Step 6（ii）代码

```python
# temperature=0 无法做除法，退化为 greedy
if temperature == 0:
    return logits.argmax(-1).item()
# 得到过滤后的词表分布：(V,)
probs = nucleus_probs(logits, p, temperature)
# 按概率抽 1 个 id：(V,) → (1,) → int
return torch.multinomial(probs, 1, generator=gen).item()
```

##### Step 6（iii）解释

Demo 手算的分布是 id1=0.625、id3=0.375，两个后端各采样 5000 次的频率都接近该值，被过滤的 id0 和 id2 频率为 0。

##### 5️⃣ 复杂度

主项来自排序：

$$
\text{时间}=O(V\log V),\quad \text{空间}=O(V)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：为何必须包含“第一次使累计概率≥p”的 token

定义要求保留质量至少为 $p$。若只保留 $\mathrm{cum}_r<p$ 的 token，保留质量严格小于 $p$，不满足定义；若 $q_1\ge p$，保留集合还会为空：

$$
\sum_{i=1}^{K-1}q_i<p\le\sum_{i=1}^{K}q_i
$$

```python
# 降序概率：(4,)
q = torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)
for pp in (0.3, 0.6, 0.9):
    # 错误写法：直接比较累计概率
    bad = q.cumsum(0) < pp
    # 正确写法：比较之前的累计概率
    good = q.cumsum(0) - q < pp
    print(pp, "错误:", bad.sum().item(), q[bad].sum().item(),
          "正确:", good.sum().item(), q[good].sum().item())
```

$p=0.3$ 时错误写法保留 0 个 token，随后 `multinomial` 会因权重全为 0 报错。

##### Follow-up 2：`p` 很小、ties、所有 logits 极小、temperature=0 时如何处理

- $p$ 很小：$p\le q_1$ 时只保留首位，退化为 greedy。
- ties：`stable=True` 让并列按 id 排序。若并列落在截断边界上，只保留 id 较小者，所以结果确定，但此时选择依赖 id 顺序。
- 所有 logits 极小：`softmax` 内部按差值计算，不受整体平移影响。手写 `exp` 必须先减最大值，否则全部下溢为 0，得到 NaN。
- temperature=0：直接取 `argmax`，不进入除法。

```python
lg2 = torch.tensor([1.0, 3.0, 2.0]).double()
# p 很小：只保留最大概率的 id1
print("小 p:", nucleus_probs(lg2, 1e-6).tolist())
# 四个并列概率 .25，p=.5 时保留 id 最小的两个
print("ties:", nucleus_probs(torch.zeros(4).double(), 0.5).tolist())
# logits 整体减 1e4，分布不变
lg3 = lg2 - 1e4
print("极小:", nucleus_probs_np(lg3.numpy(), 0.9))
print("极小:", nucleus_probs_np(lg2.numpy(), 0.9))
# 未减最大值时全部下溢：0/0 得到 NaN
with np.errstate(invalid="ignore"):
    print("手写 exp:", np.exp(lg3.numpy()) / np.exp(lg3.numpy()).sum())
print("T=0:", top_p_sample(lg2, 0.9, temperature=0))
```

##### Follow-up 3：如何用 `torch.multinomial` 保证没有采样到被过滤 token

被过滤位置的权重严格为 0，`multinomial` 不会抽到零权重位置，所以把过滤后的完整分布传入即可。也可以在排序空间抽样，但必须用 `idx` 映射回 token id：

$$
r\sim\mathrm{Categorical}(\tilde{q}),\quad y=\mathrm{idx}_r
$$

下面在高熵分布上大量采样验证，并对比漏掉映射的后果：

```python
torch.manual_seed(0)
# V=1000 的随机 logits：(V,)
lg4 = torch.randn(1000).double()
pf = nucleus_probs(lg4, 0.5)
# 一次抽 200000 个样本：(V,) → (200000,)
ids = torch.multinomial(pf, 200000, replacement=True)
print("保留数:", (pf > 0).sum().item(),
      "命中被过滤 token:", (pf[ids] == 0).sum().item())
# 排序空间抽样：名次 r:(1,)，idx[r] 才是 token id
q5, idx5 = lg2.softmax(-1).sort(descending=True, stable=True)
r = torch.multinomial(q5, 1, generator=torch.Generator().manual_seed(1))
print("名次:", r.item(), "token id:", idx5[r].item())
```

#### Q033｜Top-k + Temperature Sampling

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现 logits 温度缩放、Top-k 截断、Softmax 与抽样。
- **输入输出 / 约束：** `logits:[V]`、`k`、`temperature`；输出 token index。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. temperature 趋近 0 时如何退化为 greedy decode？
  2. Top-k 与 Top-p 组合的先后会影响结果吗？
  3. 为何要在截断后重新归一化？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

Top-k + Temperature 采样先保留 logits 最大的 $k$ 个 token，再用温度调节分布的集中程度，最后抽样，结构是“Top-k 截断 → 除以 $\tau$ → Softmax 重归一化 → 映射回词表 → multinomial”，数学核心对象是截断集合 $S_k$ 上的 Softmax 分布。$\tau$ 趋近 0 时走 greedy 分支。

##### 2️⃣ Global Equations

设 $S_k$ 为 logits 最大的 $k$ 个 token 的下标集合，主目标是：

$$
\tilde{\pi}_j=\frac{\mathbb{1}[j\in S_k]\exp(\ell_j/\tau)}{\sum_{i\in S_k}\exp(\ell_i/\tau)},\quad y\sim\mathrm{Categorical}(\tilde{\pi})
$$

$\tau>0$ 时除以常数不改变大小顺序，所以 $S_k$ 与 $\tau$ 无关。$\tau\to 0$ 时退化为 greedy：

$$
\lim_{\tau\to 0^+}\tilde{\pi}_j=\mathbb{1}\left[j=\arg\max_i\ell_i\right]\quad(\text{最大值唯一时})
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch

def top_k_probs(logits, k, temperature=1.0):
    # 词表大小为上限，k 超过 V 时等于不截断
    k = min(k, logits.shape[-1])
    # 取前 k 大：(V,) → vals:(k,)，idx:(k,)
    vals, idx = logits.topk(k)
    # 在保留集合上做温度 Softmax：(k,) → (k,)
    kept = (vals / temperature).softmax(-1)
    # 写回词表空间，其余位置为 0：(k,) → (V,)
    return torch.zeros_like(logits).scatter(-1, idx, kept)

def top_k_sample(logits, k, temperature=1.0, gen=None):
    # 温度趋近 0：直接 greedy，不做除法
    if temperature < 1e-6:
        return logits.argmax(-1).item()
    # 得到截断后的词表分布：(V,)
    probs = top_k_probs(logits, k, temperature)
    # 按概率抽 1 个 id：(V,) → (1,) → int
    return torch.multinomial(probs, 1, generator=gen).item()
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def top_k_probs_np(logits, k, temperature=1.0):
    # 词表大小为上限，k 超过 V 时等于不截断
    k = min(k, logits.shape[-1])
    # 降序取前 k 个下标：(V,) → idx:(k,)
    idx = np.argsort(-logits, kind="stable")[:k]
    # 取值并缩放：(k,)，减最大值防止 exp 上溢
    z = logits[idx] / temperature
    e = np.exp(z - z.max())
    # 归一化：(k,)
    kept = e / e.sum()
    # 写回词表空间，其余位置为 0：(k,) → (V,)
    out = np.zeros_like(logits)
    out[idx] = kept
    return out

def top_k_sample_np(logits, k, rng, temperature=1.0):
    # 温度趋近 0：直接 greedy，不做除法
    if temperature < 1e-6:
        return int(np.argmax(logits))
    # 得到截断后的词表分布：(V,)
    probs = top_k_probs_np(logits, k, temperature)
    # 按概率抽 1 个 id：(V,) → int
    return int(rng.choice(len(probs), p=probs))
```

##### 最小可运行测试（Demo）

```python
lg_np = np.array([1.0, 3.0, 2.0, 0.0])
lg = torch.from_numpy(lg_np)
# 手算：k=2、T=0.5 保留 3 和 2，差值变为 2，得 σ(2)≈.8808
a = top_k_probs(lg, 2, 0.5)
b = top_k_probs_np(lg_np, 2, 0.5)
print(a.tolist(), "后端差异:", abs(a.numpy() - b).max())
# k 超过词表：等于完整 Softmax，与官方 API 对齐
c = top_k_probs(lg, 99)
print("与 softmax 差异:", (c - lg.softmax(-1)).abs().max().item())
torch.manual_seed(0)
rng = np.random.default_rng(0)
# 两个后端各采样 5000 次，统计每个 id 的频率
s_pt = [top_k_sample(lg, 2, 0.5) for _ in range(5000)]
s_np = [top_k_sample_np(lg_np, 2, rng, 0.5) for _ in range(5000)]
print(np.bincount(s_pt, minlength=4) / 5000)
print(np.bincount(s_np, minlength=4) / 5000)
# 温度为 0 时两个后端都返回 argmax
print(top_k_sample(lg, 2, 0.0), top_k_sample_np(lg_np, 2, rng, 0.0))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import torch

def top_k_probs(logits, k, temperature=1.0):
    pass

def top_k_sample(logits, k, temperature=1.0, gen=None):
    pass
```

##### Step 1（i）公式：确定截断集合

$$
S_k=\arg\mathrm{top}_k(\ell),\quad k\leftarrow\min(k,V)
$$

##### Step 1（ii）代码

```python
# 词表大小为上限，k 超过 V 时等于不截断
k = min(k, logits.shape[-1])
# 取前 k 大：(V,) → vals:(k,)，idx:(k,)
vals, idx = logits.topk(k)
```

##### Step 1（iii）解释

`topk` 恰好返回 $k$ 个下标，这样并列时也不会多保留。$\tau>0$ 不改变大小顺序，所以先截断还是先缩放，得到的 $S_k$ 相同。

##### Step 2（i）公式：温度 Softmax

$$
\tilde{q}_r=\frac{\exp(v_r/\tau)}{\sum_{s=1}^{k}\exp(v_s/\tau)}
$$

##### Step 2（ii）代码

```python
# 在保留集合上做温度 Softmax：(k,) → (k,)
kept = (vals / temperature).softmax(-1)
```

##### Step 2（iii）解释

只在 $k$ 个值上算 Softmax，就完成了截断后的重归一化。$\tau<1$ 放大差距、分布更集中；$\tau>1$ 缩小差距、分布更均匀。Demo 中 $\tau=0.5$ 把差值 1 放大到 2，最大概率从 0.731 升到 0.881。

##### Step 3（i）公式：映射回词表空间

$$
\tilde{\pi}_{\mathrm{idx}_r}=\tilde{q}_r,\quad \tilde{\pi}_j=0\;(j\notin S_k)
$$

##### Step 3（ii）代码

```python
# 写回词表空间，其余位置为 0：(k,) → (V,)
return torch.zeros_like(logits).scatter(-1, idx, kept)
```

##### Step 3（iii）解释

`scatter` 把第 $r$ 大的概率写回 `idx[r]`。抽样时下标就是 token id，被截断 token 的权重严格为 0。

##### Step 4（i）公式：抽样与温度趋近 0

$$
y=
\begin{cases}
\arg\max_j\ell_j, & \tau<10^{-6}\\
y\sim\mathrm{Categorical}(\tilde{\pi}), & \text{否则}
\end{cases}
$$

##### Step 4（ii）代码

```python
# 温度趋近 0：直接 greedy，不做除法
if temperature < 1e-6:
    return logits.argmax(-1).item()
# 得到截断后的词表分布：(V,)
probs = top_k_probs(logits, k, temperature)
# 按概率抽 1 个 id：(V,) → (1,) → int
return torch.multinomial(probs, 1, generator=gen).item()
```

##### Step 4（iii）解释

$10^{-6}$ 是这里选定的数值阈值，用来同时处理 $\tau=0$ 的除零和 $\tau$ 极小时的溢出。低于阈值后，结果已经和 greedy 没有实际差别。

##### 5️⃣ 复杂度

`topk` 的主项是部分选择，Softmax 和写回部分是线性的：

$$
\text{时间}=O(V\log k+k),\quad \text{空间}=O(V)
$$

NumPy 版用全排序，时间为 $O(V\log V)$。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：temperature 趋近 0 时如何退化为 greedy decode

对任意非最大值 token，设 $\ell^*=\max_i\ell_i$，其与最大值 token 的概率比按指数衰减：

$$
\frac{\tilde{\pi}_j}{\tilde{\pi}_{j^*}}=\exp\left(\frac{\ell_j-\ell^*}{\tau}\right)\xrightarrow{\tau\to 0^+}0
$$

所以极限是 one-hot。若有并列最大值，极限是并列 token 上的均匀分布，而 `argmax` 只返回其中一个。

数值上直接除以极小的 $\tau$ 会出错：float32 下 `logits/τ` 上溢成 `inf`，Softmax 得到 NaN，所以要显式走 greedy 分支。

```python
lg32 = torch.tensor([1.0, 3.0, 2.0, 0.0])
# 最大概率随 T 减小趋近 1
for T in (1.0, 0.1, 0.01, 1e-3):
    print(T, top_k_probs(lg32, 3, T).max().item())
# T 极小：float32 除法上溢，Softmax 输出 NaN
print("T=1e-39:", top_k_probs(lg32, 3, 1e-39).tolist())
# 走 greedy 分支，结果稳定
print("分支:", top_k_sample(lg32, 3, 1e-39), lg32.argmax().item())
# 并列最大值：极限均匀，argmax 只返回一个
tie = torch.tensor([3.0, 3.0, 1.0])
print(top_k_probs(tie, 3, 1e-3).tolist(), tie.argmax().item())
```

##### Follow-up 2：Top-k 与 Top-p 组合的先后会影响结果吗

会。Top-k 的集合只看大小顺序，Top-p 的累计概率依赖当前分布。先 Top-k 重归一化后，概率变大，累计更快达到 $p$，Top-p 可能保留更少的 token：

$$
\text{top-p}(\text{top-k}(\pi))\ne\text{top-k}(\text{top-p}(\pi))\quad(\text{一般情形})
$$

取 $\pi=[0.4,0.3,0.2,0.1]$，$k=2$，$p=0.5$。先 Top-k 得 $[0.571,0.429]$，首位已超过 0.5，只留 1 个；先 Top-p 留前 2 个，再 Top-k 仍是 2 个。Hugging Face `generate` 的处理链依次是 temperature、top-k、top-p。

```python
def keep_k(q, k):
    # q 是降序概率 (V,)：保留前 k 个并重归一化
    out = torch.zeros_like(q)
    out[:k] = q[:k]
    return out / out.sum()
def keep_p(q, p):
    # q 是降序概率 (V,)：比较之前的累计概率并重归一化
    out = q * (q.cumsum(-1) - q < p)
    return out / out.sum()
q = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
print("先 k 后 p:", keep_p(keep_k(q, 2), 0.5).tolist())
print("先 p 后 k:", keep_k(keep_p(q, 0.5), 2).tolist())
```

##### Follow-up 3：为何要在截断后重新归一化

截断后概率和小于 1，已经不是合法分布。抽样、计算 entropy 和 logprob 都需要归一化后的分布。对截断后的 logits 做 Softmax，与截断概率再除以总和完全等价：

$$
\frac{\exp(\ell_j/\tau)}{\sum_{i\in S_k}\exp(\ell_i/\tau)}=\frac{\pi_j}{\sum_{i\in S_k}\pi_i},\quad \pi=\mathrm{softmax}(\ell/\tau)
$$

API 行为有差别：`torch.multinomial` 接受未归一化的非负权重，内部按比例抽样；`np.random.Generator.choice` 要求概率和为 1，否则报错。

```python
lg64 = torch.tensor([1.0, 3.0, 2.0, 0.0], dtype=torch.float64)
pi = (lg64 / 0.5).softmax(-1)
# 只留前 2 大，得到未归一化概率：(V,)
cut = torch.zeros_like(pi)
top_i = pi.topk(2).indices
cut[top_i] = pi[top_i]
print("概率和:", cut.sum().item())
print("等价差异:",
      (cut / cut.sum() - top_k_probs(lg64, 2, 0.5)).abs().max().item())
# NumPy：未归一化直接报错
try:
    np.random.default_rng(0).choice(4, p=cut.numpy())
except ValueError as e:
    print("NumPy:", e)
# torch：未归一化仍可抽样，并且不会抽到被截断的 id
torch.manual_seed(0)
print(torch.multinomial(cut, 10000, replacement=True).bincount() / 10000)
```

#### Q034｜Sigmoid 函数及导数

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-03（帖子发布时间）
- **题目：** 写一个数值稳定的 `sigmoid(x)` 并给出其导数，不能直接假定输入不会溢出。
- **输入输出 / 约束：** `x` 为 Python 标量或 NumPy 数组，输出同形状概率/导数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何在 `x<0` 时用另一种等价表达式避免 `exp(-x)` 溢出？
  2. Sigmoid + BCE 为何推荐合并为 logits 版本？
  3. 多标签分类和多类别互斥分类分别用 sigmoid 还是 softmax？
- **来源：** [牛客：字节推荐实习 MHA、AUC 变体](https://www.nowcoder.com/users/615039244)

##### 1️⃣ Dense Opening

Sigmoid 把实数映射到 $(0,1)$，难点是 $|x|$ 很大时 `exp` 上溢，以及 $x$ 很大时 $1-\sigma(x)$ 被舍入成 0。结构是“计算 $e=\exp(-|x|)$ → 按符号选等价表达式 → 用同一个 $e$ 算导数”，数学核心对象是恒在 $(0,1]$ 内的 $e$，因此没有任何指数运算会上溢。

##### 2️⃣ Global Equations

定义及两种等价形式：

$$
\sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^{x}}{1+e^{x}}
$$

令 $e=\exp(-|x|)$，分支写法是：

$$
\sigma(x)=
\begin{cases}
\frac{1}{1+e}, & x\ge 0\\
\frac{e}{1+e}, & x<0
\end{cases}
$$

导数关于 $x=0$ 对称，所以可以只用 $e$ 表示：

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))=\frac{e}{(1+e)^2}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import torch

def sigmoid_pt(x):
    # e∈(0,1]，指数非正所以不上溢：形状同 x
    e = torch.exp(-x.abs())
    # 按符号逐元素选分支：形状同 x
    return torch.where(x >= 0, 1 / (1 + e), e / (1 + e))

def sigmoid_grad_pt(x):
    # e∈(0,1]，指数非正所以不上溢：形状同 x
    e = torch.exp(-x.abs())
    # 对称形式：形状同 x
    return e / (1 + e) ** 2
```

##### （B）NumPy版本（苛刻面试版）

```python
import numpy as np

def sigmoid(x):
    # 标量或数组统一转成 float 数组：形状同 x
    x = np.asarray(x, dtype=float)
    # e∈(0,1]，指数非正所以不上溢：形状同 x
    e = np.exp(-np.abs(x))
    # 按符号逐元素选分支：形状同 x
    p = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
    # 0 维数组转回标量，其余形状不变
    return p[()]

def sigmoid_grad(x):
    # 标量或数组统一转成 float 数组：形状同 x
    x = np.asarray(x, dtype=float)
    # e∈(0,1]，指数非正所以不上溢：形状同 x
    e = np.exp(-np.abs(x))
    # 对称形式：形状同 x
    d = e / (1 + e) ** 2
    # 0 维数组转回标量，其余形状不变
    return d[()]
```

##### 最小可运行测试（Demo）

```python
xs = np.array([-1000.0, -40.0, -1.0, 0.0, 1.0, 40.0, 1000.0])
xt = torch.from_numpy(xs)
print("概率:", sigmoid(xs))
print("后端差异:", abs(sigmoid_pt(xt).numpy() - sigmoid(xs)).max(),
      abs(sigmoid_grad_pt(xt).numpy() - sigmoid_grad(xs)).max())
# 与官方 API 和 autograd 对齐
xg = xt.clone().requires_grad_()
torch.sigmoid(xg).sum().backward()
print("官方差异:", abs(torch.sigmoid(xt).numpy() - sigmoid(xs)).max())
print("autograd:", xg.grad.numpy())
print("闭式导数:", sigmoid_grad(xs))
# 标量返回标量，数组保持形状
print(type(sigmoid(0.0)).__name__, sigmoid(0.0), sigmoid_grad(0.0))
print(sigmoid(np.zeros((2, 3))).shape)
# 关键差异：p(1-p) 在 x=40 舍入成 0，真实值是 e^-40
p40 = sigmoid(40.0)
print(p40 * (1 - p40), sigmoid_grad(40.0), np.exp(-40.0))
# 把上溢告警转成异常，稳定版仍正常运行
with np.errstate(over="raise", under="ignore"):
    print(sigmoid(1e6), sigmoid(-1e6), sigmoid_grad(1e6))
# 对 sigmoid_pt 走 autograd：x=0 处 abs 的梯度为 0，真实导数是 .25
x0 = torch.tensor(0.0, requires_grad=True)
sigmoid_pt(x0).backward()
print("x=0 autograd:", x0.grad.item(), "闭式:", sigmoid_grad(0.0))
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def sigmoid(x):
    pass

def sigmoid_grad(x):
    pass
```

##### Step 1（i）公式：不上溢的指数项

$$
e=\exp(-|x|)\in(0,1]
$$

##### Step 1（ii）代码

```python
# 标量或数组统一转成 float 数组：形状同 x
x = np.asarray(x, dtype=float)
# e∈(0,1]，指数非正所以不上溢：形状同 x
e = np.exp(-np.abs(x))
```

##### Step 1（iii）解释

`np.exp` 在 float64 下输入约超过 709.78 就上溢。这里指数恒为非正，$|x|$ 很大时 $e$ 下溢到 0，这是正确极限。

##### Step 2（i）公式：按符号选等价表达式

$$
\sigma(x)=\frac{1}{1+e}\;(x\ge 0),\quad \sigma(x)=\frac{e}{1+e}\;(x<0)
$$

##### Step 2（ii）代码

```python
# 按符号逐元素选分支：形状同 x
p = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
# 0 维数组转回标量，其余形状不变
return p[()]
```

##### Step 2（iii）解释

$x<0$ 时 $|x|=-x$，所以 $e/(1+e)=e^{x}/(1+e^{x})$。这样负半轴的小概率保留相对精度（Demo 中 $x=-40$ 得 $4.25\times 10^{-18}$），且不产生 `inf`。`np.where` 两个分支都会计算，但都是安全的。

##### Step 3（i）公式：导数

$$
\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{e}{(1+e)^2}
$$

##### Step 3（ii）代码

```python
# 对称形式：形状同 x
d = e / (1 + e) ** 2
# 0 维数组转回标量，其余形状不变
return d[()]
```

##### Step 3（iii）解释

$\sigma(x)(1-\sigma(x))$ 在 $x=40$ 时，$\sigma(x)$ 舍入为 1，得到 0。对称形式给出正确值 $4.25\times 10^{-18}$。PyTorch 的 autograd 用 $p(1-p)$，Demo 中也在 $x=40$ 输出 0，在 $x=-40$ 输出正确值。另外，对 `sigmoid_pt` 直接做 autograd 时，`abs` 在 0 处的梯度为 0，输出 0，所以导数要用闭式函数。

##### 5️⃣ 复杂度

设输入元素数为 $N$，所有操作都逐元素进行：

$$
\text{时间}=O(N),\quad \text{空间}=O(N)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：如何在 `x<0` 时用另一种等价表达式防止 `exp(-x)` 溢出

$x<0$ 时 $-x>0$，直接算 $e^{-x}$ 可能上溢。分子分母同乘 $e^{x}$，指数变为负数：

$$
\frac{1}{1+e^{-x}}=\frac{e^{x}}{e^{x}+1}
$$

同样，$x>0$ 时不能用右式，否则 $e^{x}$ 上溢后变成 $\infty/\infty$，得到 NaN。

```python
big = np.array([-800.0, 800.0])
with np.errstate(over="ignore", invalid="ignore"):
    # 左式：x=-800 时 exp(800)=inf，发生上溢，数值碰巧得到 0
    print("左式:", 1 / (1 + np.exp(-big)))
    # 右式：x=800 时 inf/inf，得到 NaN
    print("右式:", np.exp(big) / (1 + np.exp(big)))
print("稳定版:", sigmoid(big))
# 打开上溢检查：左式抛异常，稳定版正常
with np.errstate(over="raise"):
    try:
        np.exp(-big)
    except FloatingPointError as err:
        print("捕获:", err)
    print("稳定版:", sigmoid(big))
```

##### Follow-up 2：Sigmoid + BCE 为何推荐合并为 logits 版本

先算概率再取 log 会丢精度：$x$ 很大时 $1-\sigma(x)=0$，$\log 0$ 发散。合并后不经过概率：

$$
\ell(x,y)=-y\log\sigma(x)-(1-y)\log(1-\sigma(x))=\log(1+e^{x})-yx
$$

$$
\ell(x,y)=\max(x,0)-xy+\log(1+e^{-|x|})
$$

第二行是 TensorFlow `sigmoid_cross_entropy_with_logits` 文档给出的稳定形式，每一项都不会上溢，下面的代码与 PyTorch 官方 API 对齐。梯度也更简单：$\partial\ell/\partial x=\sigma(x)-y$。

```python
import torch.nn.functional as F
def bce_logits(x, y):
    # log1p 对极小的 e 仍保持精度：形状同 x
    return np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
x2 = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
y2 = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
print("合并版:", bce_logits(x2, y2))
print("官方版:", F.binary_cross_entropy_with_logits(
    torch.from_numpy(x2), torch.from_numpy(y2),
    reduction="none").numpy())
# 分开写：概率恰好为 0 或 1 时 log 发散
p2 = sigmoid(x2)
with np.errstate(divide="ignore", invalid="ignore"):
    print("分开写:", -y2 * np.log(p2) - (1 - y2) * np.log(1 - p2))
```

分开写在错分的 $x=-800,y=1$ 和 $x=40,y=0$ 处都得到 `inf`，而这两个位置恰恰需要有效梯度；在 $x=800,y=1$ 处还会出现 $0\cdot\infty$，得到 NaN，真实损失是 0。

##### Follow-up 3：多标签分类和多类别互斥分类分别用 sigmoid 还是 softmax

- 多标签（Multi-label）：每个标签独立判断，逐类 sigmoid，各类概率之和不必为 1，配 BCE。
- 多类别互斥（Multi-class）：类别只能选一个，用 softmax，概率和为 1，配 CE。

$$
p_c^{\mathrm{multi\text{-}label}}=\sigma(z_c),\quad p_c^{\mathrm{multi\text{-}class}}=\frac{e^{z_c}}{\sum_k e^{z_k}}
$$

二分类时两者等价：两个类别的 softmax 等于 logit 差的 sigmoid。

```python
z = torch.tensor([2.0, 1.0, -1.0], dtype=torch.float64)
# 多标签：逐类独立，和可以超过 1
print("sigmoid:", sigmoid(z.numpy()), sigmoid(z.numpy()).sum())
# 互斥：竞争同一份概率，和为 1
print("softmax:", z.softmax(0).numpy(), z.softmax(0).sum().item())
# 二分类等价：softmax([z0, z1])[0] = sigmoid(z0-z1)
print(z[:2].softmax(0)[0].item(), sigmoid((z[0] - z[1]).item()))
```

#### Q035｜Logistic Regression：NumPy 完整训练

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-01（练习页面更新；非面试日期）
- **题目：** 从随机初始化开始写 Binary Logistic Regression，含 sigmoid、BCE、梯度与迭代更新。
- **输入输出 / 约束：** `X:[N,D]`、`y:[N]`、lr、steps，返回 `w,b` 和预测概率。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. L2 正则项是否应惩罚 bias？
  2. 类别比例极不均衡时，指标如何选择？
  3. 和直接 MSE 最小化相比，优化性质有什么不同？
- **来源：** [PracHub：TikTok AUC/Softmax/Logistic Regression 练习页](https://prachub.com/companies/tiktok/categories/machine-learning)

##### 1️⃣ Dense Opening

二分类逻辑回归（Binary Logistic Regression）用线性打分加 Sigmoid 建模 $P(y=1\mid x)$，用梯度下降最小化 BCE，结构是“随机初始化 → logit → logits 形式 BCE → 闭式梯度 → 更新”，数学核心对象是梯度中的残差 $p-y$。

##### 2️⃣ Global Equations

模型与主目标（$\lambda$ 为可选 L2 系数，默认 0）：

$$
z_i=x_i^\top w+b,\quad p_i=\sigma(z_i)
$$

$$
J(w,b)=\frac{1}{N}\sum_{i=1}^{N}\left[\log(1+e^{z_i})-y_iz_i\right]+\frac{\lambda}{2}\|w\|_2^2
$$

对 $z_i$ 求导得 $p_i-y_i$，再用链式法则得到梯度：

$$
\nabla_wJ=\frac{1}{N}X^\top(p-y)+\lambda w,\quad \frac{\partial J}{\partial b}=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import numpy as np
import torch
import torch.nn.functional as F

def fit_logreg_pt(X, y, lr=0.1, steps=500, l2=0.0, seed=0):
    # 用 NumPy 同种子初始化，与 B 版起点相同：w:(D,)
    w0 = np.random.default_rng(seed).normal(0, 0.01, X.shape[1])
    # ndarray 转叶子张量，dtype 不变：(D,)
    w = torch.from_numpy(w0).requires_grad_()
    # bias 从 0 开始：(1,)，z 中广播到 (N,)
    b = torch.zeros(1, dtype=X.dtype, requires_grad=True)
    # 全批量迭代
    for _ in range(steps):
        # (N, D) @ (D,) → (N,)，b 广播
        z = X @ w + b
        # logits 形式 BCE 加 L2：(N,) → 标量
        loss = F.binary_cross_entropy_with_logits(z, y)
        # 只惩罚 w，不惩罚 bias
        loss = loss + 0.5 * l2 * (w ** 2).sum()
        # autograd 计算梯度，与 B 版闭式梯度对应
        loss.backward()
        # 参数更新不进计算图
        with torch.no_grad():
            # 梯度下降更新 w
            w -= lr * w.grad
            # 梯度下降更新 b
            b -= lr * b.grad
            # 清空累积梯度
            w.grad.zero_(); b.grad.zero_()
    # 预测概率：(N, D) @ (D,) → (N,)
    probs = torch.sigmoid(X @ w + b).detach()
    return w.detach(), b.item(), probs
```

##### （B）NumPy版本（苛刻面试版）

```python
def sigmoid_np(z):
    # 指数非正，不上溢：形状同 z
    e = np.exp(-np.abs(z))
    # 按符号选等价表达式：形状同 z
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def bce_np(z, y):
    # log(1+e^z)-yz 的稳定形式：(N,) → 标量
    return (np.logaddexp(0.0, z) - y * z).mean()

def fit_logreg(X, y, lr=0.1, steps=500, l2=0.0, seed=0):
    # 读取样本数和特征数
    N, D = X.shape
    # 随机初始化：w:(D,)
    w = np.random.default_rng(seed).normal(0, 0.01, D)
    # bias 从 0 开始：标量
    b = 0.0
    # 全批量迭代
    for _ in range(steps):
        # 前向 logit：(N, D) @ (D,) → (N,)
        z = X @ w + b
        # 残差：(N,)
        r = sigmoid_np(z) - y
        # w 梯度：(D, N) @ (N,) → (D,)，再加 L2
        grad_w = X.T @ r / N + l2 * w
        # b 梯度：(N,) → 标量，不做 L2
        grad_b = r.mean()
        # 梯度下降更新 w
        w -= lr * grad_w
        # 梯度下降更新 b
        b -= lr * grad_b
    # 预测概率：(N,)
    probs = sigmoid_np(X @ w + b)
    return w, b, probs
```

##### 最小可运行测试（Demo）

```python
rng = np.random.default_rng(1)
# 按已知参数生成数据：X:(500, 3)，y:(500,)
X_np = rng.normal(size=(500, 3))
w_true, b_true = np.array([2.0, -1.0, 0.5]), -0.5
y_np = (rng.random(500) < sigmoid_np(X_np @ w_true + b_true)) * 1.0
w, b, p = fit_logreg(X_np, y_np, lr=0.5, steps=2000)
Xt, yt = torch.from_numpy(X_np), torch.from_numpy(y_np)
wp, bp, pp = fit_logreg_pt(Xt, yt, lr=0.5, steps=2000)
# autograd 与闭式梯度走完全部迭代后的差异
print("后端差异:", abs(wp.numpy() - w).max(), abs(bp - b))
print("w:", w, "b:", b)
# 收敛判据：驻点处残差的线性组合为 0
r_end = p - y_np
print("梯度范数:", np.linalg.norm(X_np.T @ r_end / 500), abs(r_end.mean()))
# 对照基线：全 0 logit 的 loss 恰好是 log 2
z0 = np.zeros(500)
print("loss:", bce_np(z0, y_np), "→", bce_np(X_np @ w + b, y_np))
print("acc:", ((p > 0.5) == y_np).mean())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def sigmoid_np(z):
    pass
def bce_np(z, y):
    pass

def fit_logreg(X, y, lr=0.1, steps=500, l2=0.0, seed=0):
    pass
```

##### Step 1（i）公式：稳定的 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-|z|}}\;(z\ge 0),\quad \sigma(z)=\frac{e^{-|z|}}{1+e^{-|z|}}\;(z<0)
$$

##### Step 1（ii）代码

```python
# 指数非正，不上溢：形状同 z
e = np.exp(-np.abs(z))
# 按符号选等价表达式：形状同 z
return np.where(z >= 0, 1 / (1 + e), e / (1 + e))
```

##### Step 1（iii）解释

训练中 $|z|$ 可能很大。指数恒为非正，所以不会上溢。

##### Step 2（i）公式：logits 形式的 BCE

$$
\ell(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z))=\log(1+e^{z})-yz
$$

##### Step 2（ii）代码

```python
# log(1+e^z)-yz 的稳定形式：(N,) → 标量
return (np.logaddexp(0.0, z) - y * z).mean()
```

##### Step 2（iii）解释

化简用到 $\log\sigma(z)=z-\log(1+e^{z})$ 和 $\log(1-\sigma(z))=-\log(1+e^{z})$。`logaddexp` 计算 $\log(e^0+e^z)$，不经过概率，所以没有 $\log 0$。

##### Step 3（i）公式：随机初始化

$$
w^{(0)}\sim\mathcal{N}(0,0.01^2I),\quad b^{(0)}=0
$$

##### Step 3（ii）代码

```python
# 读取样本数和特征数
N, D = X.shape
# 随机初始化：w:(D,)
w = np.random.default_rng(seed).normal(0, 0.01, D)
# bias 从 0 开始：标量
b = 0.0
```

##### Step 3（iii）解释

BCE 对 $(w,b)$ 是凸函数，全 0 初始化也能收敛。题面要求随机初始化，所以这里用小方差高斯。

##### Step 4（i）公式：前向与残差

$$
z=Xw+b,\quad r=\sigma(z)-y
$$

##### Step 4（ii）代码

```python
# 前向 logit：(N, D) @ (D,) → (N,)
z = X @ w + b
# 残差：(N,)
r = sigmoid_np(z) - y
```

##### Step 4（iii）解释

$\partial\ell/\partial z=\sigma(z)-y$。Sigmoid 的导数因子 $p(1-p)$ 被 BCE 的分母抵消，所以梯度里只剩残差。

##### Step 5（i）公式：闭式梯度

$$
\nabla_wJ=\frac{1}{N}X^\top r+\lambda w,\quad \frac{\partial J}{\partial b}=\frac{1}{N}\mathbf{1}^\top r
$$

##### Step 5（ii）代码

```python
# w 梯度：(D, N) @ (N,) → (D,)，再加 L2
grad_w = X.T @ r / N + l2 * w
# b 梯度：(N,) → 标量，不做 L2
grad_b = r.mean()
```

##### Step 5（iii）解释

`X.T @ r` 是残差对各特征的加权和，shape 为 $(D,)$，与 $w$ 对齐。Demo 中 A 版 autograd 和 B 版闭式梯度迭代 2000 步后结果一致，等于同时检验了这个公式。

##### Step 6（i）公式：梯度下降与预测

$$
w\leftarrow w-\eta\nabla_wJ,\quad b\leftarrow b-\eta\frac{\partial J}{\partial b},\quad \hat{p}=\sigma(Xw+b)
$$

##### Step 6（ii）代码

```python
# 梯度下降更新 w
w -= lr * grad_w
# 梯度下降更新 b
b -= lr * grad_b
# 预测概率：(N,)
probs = sigmoid_np(X @ w + b)
return w, b, probs
```

##### Step 6（iii）解释

全批量梯度下降每步都走完整数据。这里 `steps` 和 `lr` 都由题面输入决定，训练结束后用最终参数算一次概率。

##### 5️⃣ 复杂度

每步的主项是两次矩阵向量乘：

$$
\text{时间}=O(\text{steps}\cdot ND),\quad \text{额外空间}=O(N+D)
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：L2 正则项是否应惩罚 bias

不应该。Bias 只负责整体平移，不影响模型复杂度。不惩罚 $b$ 时，驻点条件给出“预测均值等于正例率”：

$$
\frac{\partial J}{\partial b}=0\Rightarrow\frac{1}{N}\sum_ip_i=\frac{1}{N}\sum_iy_i
$$

若惩罚 $b$，这一项会变为 $\bar{p}-\bar{y}+\lambda b=0$，预测均值产生偏移。类别不均衡时 $b$ 的绝对值大，偏移更明显。

```python
def fit_l2(X, y, l2, pen_b, lr=0.5, steps=5000):
    # 从 0 开始：w:(D,)，b 标量
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        # 残差：(N,)
        r = sigmoid_np(X @ w + b) - y
        # (D, N) @ (N,) → (D,)
        w -= lr * (X.T @ r / len(y) + l2 * w)
        # pen_b 控制 bias 是否受 L2 惩罚
        b -= lr * (r.mean() + l2 * b * pen_b)
    return sigmoid_np(X @ w + b).mean()
# 5% 正例：截距约为负值
Xi = rng.normal(size=(2000, 3))
yi = (Xi[:, 0] + rng.normal(size=2000) > 2.3) * 1.0
print("正例率:", yi.mean())
print("不罚 b:", fit_l2(Xi, yi, 0.1, 0), "罚 b:", fit_l2(Xi, yi, 0.1, 1))
```

##### Follow-up 2：类别比例极不均衡时，指标如何选择

Accuracy 会被多数类主导：正例率为 $\pi$ 时，全判负也有 $1-\pi$ 的准确率。推荐指标：

- 排序质量：PR-AUC（Average Precision），它的随机基线是 $\pi$；ROC-AUC 看全局排序，对负例数量不敏感。
- 业务阈值：固定召回下的精度，或 Recall@Precision、$F_\beta$。
- 概率质量：logloss 和校准曲线。下采样训练后要校正概率，校正时 logit 加上 $\log(\text{负例保留率})$。

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN},\quad \mathrm{AP}=\sum_k(R_k-R_{k-1})P_k
$$

```python
def average_precision(score, y):
    # 按分数降序重排标签：(N,) → (N,)
    ys = y[np.argsort(-score)]
    # 前缀 TP：(N,)
    tp = np.cumsum(ys)
    # 每个正例位置的精度取平均，(N,) → 标量
    return (tp / np.arange(1, len(ys) + 1))[ys == 1].mean()
ri = np.random.default_rng(2)
# 1% 正例，模型分数对正例略高
yb = (ri.random(20000) < 0.01) * 1.0
sb = ri.normal(size=20000) + 1.5 * yb
print("全判负 acc:", (yb == 0).mean())
print("随机 AP:", average_precision(ri.random(20000), yb))
print("模型 AP:", average_precision(sb, yb))
```

##### Follow-up 3：和直接 MSE 最小化相比，优化性质有什么不同

这里按“Sigmoid 输出接 MSE”比较。两者单样本的导数是：

$$
\frac{\partial\ell_{\mathrm{BCE}}}{\partial z}=p-y,\quad \frac{\partial\ell_{\mathrm{MSE}}}{\partial z}=2(p-y)p(1-p)
$$

- 饱和：严重错分时，比如 $z=-10,y=1$，MSE 梯度被 $p(1-p)$ 压到接近 0；BCE 梯度接近 -1。
- 凸性：BCE 的 Hessian 是 $\frac{1}{N}X^\top\mathrm{diag}(p(1-p))X\succeq 0$，全局凸；Sigmoid + MSE 对 $z$ 非凸，二阶导会变号。
- 统计含义：BCE 是 Bernoulli 似然的 MLE，MSE 不对应这个似然。

```python
for z_ in (-10.0, -2.0, 0.0, 2.0):
    p_ = sigmoid_np(np.array(z_))
    # y=1 时两种损失对 z 的导数
    print(z_, "BCE:", round(float(p_ - 1), 5),
          "MSE:", round(float(2 * (p_ - 1) * p_ * (1 - p_)), 5))
# 二阶差分判断凸性：y=1，z 网格 (401,)
zs = np.linspace(-10, 10, 401)
d2 = lambda f: np.diff(f, 2)
bce_c = np.logaddexp(0.0, zs) - zs
mse_c = (sigmoid_np(zs) - 1) ** 2
print("BCE 二阶差分最小:", d2(bce_c).min())
print("MSE 二阶差分最小:", d2(mse_c).min())
```

#### Q036｜SGD / Momentum / Adam：手写更新步骤

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 用 NumPy 实现一个最小的 SGD、Momentum 或 Adam Optimizer 的参数更新，不使用高级优化器。
- **输入输出 / 约束：** 梯度向量/字典、状态和 step；返回更新参数与状态。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Adam 的 bias correction 为什么依赖 step？
  2. Weight Decay 和 L2 正则在 Adam 中是否总等价？
  3. 参数不更新时如何定位梯度为 None、NaN 或学习率为 0？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

优化器把梯度变成参数更新量。SGD 直接用当前梯度，Momentum 累积梯度的指数和，Adam 再加上二阶矩缩放和偏差校正（bias correction），结构是“读取状态 → 更新矩 → 算步长 → 返回新参数和新状态”，数学核心对象是一阶矩 $m_t$ 和二阶矩 $s_t$。题面写的是“SGD、Momentum 或 Adam”，这里三种都实现，接口统一为参数字典、梯度字典和状态字典。

##### 2️⃣ Global Equations

SGD 与 Momentum（采用 PyTorch 约定，lr 乘在速度外面）：

$$
\theta_{t}=\theta_{t-1}-\eta g_t
$$

$$
v_t=\mu v_{t-1}+g_t,\quad \theta_t=\theta_{t-1}-\eta v_t
$$

Adam（Kingma & Ba 2015），$t$ 从 1 开始，$m_0=s_0=0$：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad \hat{s}_t=\frac{s_t}{1-\beta_2^t},\quad \theta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{s}_t}+\epsilon}
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import numpy as np
import torch

def sgd_pt(params, grads, state, lr=0.01):
    # 逐参数更新，形状不变；SGD 无状态，原样返回
    return {k: params[k] - lr * grads[k] for k in params}, state

def momentum_pt(params, grads, state, lr=0.01, mu=0.9):
    # 首次调用时速度为空字典
    v = state.get("v", {})
    # 新参数和新速度，不修改输入
    new_p, new_v = {}, {}
    # 逐参数处理
    for k in params:
        # 取上一步速度，首次为 0，形状同参数
        prev = v.get(k, torch.zeros_like(params[k]))
        # 累积速度
        new_v[k] = mu * prev + grads[k]
        # 沿速度方向更新
        new_p[k] = params[k] - lr * new_v[k]
    return new_p, {"v": new_v}

def adam_pt(params, grads, state, lr=0.001, b1=0.9, b2=0.999,
            eps=1e-8):
    # 步数加 1：第一次调用 t=1
    t = state.get("t", 0) + 1
    # 取历史矩，首次为空字典
    m, s = state.get("m", {}), state.get("s", {})
    # 新参数和新状态，不修改输入
    new_p, new_m, new_s = {}, {}, {}
    # 逐参数处理
    for k in params:
        # 当前梯度，形状同参数
        g = grads[k]
        # 一阶矩，首次为 0，形状同参数
        new_m[k] = b1 * m.get(k, torch.zeros_like(g)) + (1 - b1) * g
        # 二阶矩，逐元素平方，形状同参数
        new_s[k] = b2 * s.get(k, torch.zeros_like(g)) + (1 - b2) * g * g
        # 一阶矩偏差校正
        m_hat = new_m[k] / (1 - b1 ** t)
        # 二阶矩偏差校正
        s_hat = new_s[k] / (1 - b2 ** t)
        # 逐元素自适应步长
        new_p[k] = params[k] - lr * m_hat / (torch.sqrt(s_hat) + eps)
    return new_p, {"t": t, "m": new_m, "s": new_s}
```

##### （B）NumPy版本（苛刻面试版）

```python
def sgd_np(params, grads, state, lr=0.01):
    # 逐参数更新，形状不变；SGD 无状态，原样返回
    return {k: params[k] - lr * grads[k] for k in params}, state

def momentum_np(params, grads, state, lr=0.01, mu=0.9):
    # 首次调用时速度为空字典
    v = state.get("v", {})
    # 新参数和新速度，不修改输入
    new_p, new_v = {}, {}
    # 逐参数处理
    for k in params:
        # 取上一步速度，首次为 0，形状同参数
        prev = v.get(k, np.zeros_like(params[k]))
        # 累积速度
        new_v[k] = mu * prev + grads[k]
        # 沿速度方向更新
        new_p[k] = params[k] - lr * new_v[k]
    return new_p, {"v": new_v}

def adam_np(params, grads, state, lr=0.001, b1=0.9, b2=0.999,
            eps=1e-8):
    # 步数加 1：第一次调用 t=1
    t = state.get("t", 0) + 1
    # 取历史矩，首次为空字典
    m, s = state.get("m", {}), state.get("s", {})
    # 新参数和新状态，不修改输入
    new_p, new_m, new_s = {}, {}, {}
    # 逐参数处理
    for k in params:
        # 当前梯度，形状同参数
        g = grads[k]
        # 一阶矩，首次为 0，形状同参数
        new_m[k] = b1 * m.get(k, np.zeros_like(g)) + (1 - b1) * g
        # 二阶矩，逐元素平方，形状同参数
        new_s[k] = b2 * s.get(k, np.zeros_like(g)) + (1 - b2) * g * g
        # 一阶矩偏差校正
        m_hat = new_m[k] / (1 - b1 ** t)
        # 二阶矩偏差校正
        s_hat = new_s[k] / (1 - b2 ** t)
        # 逐元素自适应步长
        new_p[k] = params[k] - lr * m_hat / (np.sqrt(s_hat) + eps)
    return new_p, {"t": t, "m": new_m, "s": new_s}
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# 固定梯度序列：(20, 3)，三路读取相同的梯度
gs = torch.randn(20, 3, dtype=torch.float64)
cases = [
    ("sgd", sgd_np, sgd_pt, lambda w: torch.optim.SGD([w], lr=0.01)),
    ("mom", momentum_np, momentum_pt,
     lambda w: torch.optim.SGD([w], lr=0.01, momentum=0.9)),
    ("adam", adam_np, adam_pt, lambda w: torch.optim.Adam([w], lr=0.01))]
for name, f_np, f_pt, make in cases:
    # 三路使用相同的起点：(3,)
    w_off = torch.ones(3, dtype=torch.float64, requires_grad=True)
    p_np, st_np = {"w": np.ones(3)}, {}
    p_pt, st_pt = {"w": torch.ones(3, dtype=torch.float64)}, {}
    opt = make(w_off)
    for g in gs:
        p_np, st_np = f_np(p_np, {"w": g.numpy()}, st_np, lr=0.01)
        p_pt, st_pt = f_pt(p_pt, {"w": g}, st_pt, lr=0.01)
        # 官方优化器读取同一个梯度
        w_off.grad = g.clone()
        opt.step()
    print(name, "A/B 差异:", abs(p_np["w"] - p_pt["w"].numpy()).max(),
          "官方差异:", abs(p_np["w"] - w_off.detach().numpy()).max())
# 手算：Adam 第一步 m̂/√ŝ=sign(g)，步长约为 lr，与梯度大小无关
for gv in (1e-3, 1.0, 1e3):
    p1, _ = adam_np({"w": np.zeros(1)}, {"w": np.array([gv])}, {})
    print(gv, p1["w"])
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def sgd_np(params, grads, state, lr=0.01):
    pass
def momentum_np(params, grads, state, lr=0.01, mu=0.9):
    pass

def adam_np(params, grads, state, lr=0.001, b1=0.9, b2=0.999,
            eps=1e-8):
    pass
```

##### Step 1（i）公式：SGD

$$
\theta_t=\theta_{t-1}-\eta g_t
$$

##### Step 1（ii）代码

```python
# 逐参数更新，形状不变；SGD 无状态，原样返回
return {k: params[k] - lr * grads[k] for k in params}, state
```

##### Step 1（iii）解释

SGD 没有状态，统一返回 `(新参数, 状态)` 只是为了三种优化器接口一致。这里返回新字典，不修改输入，调用方可以直接比较更新前后的参数。

##### Step 2（i）公式：Momentum 速度

$$
v_t=\mu v_{t-1}+g_t,\quad v_0=0
$$

##### Step 2（ii）代码

```python
# 首次调用时速度为空字典
v = state.get("v", {})
# 新参数和新速度，不修改输入
new_p, new_v = {}, {}
# 逐参数处理
for k in params:
    # 取上一步速度，首次为 0，形状同参数
    prev = v.get(k, np.zeros_like(params[k]))
    # 累积速度
    new_v[k] = mu * prev + grads[k]
```

##### Step 2（iii）解释

展开得 $v_t=\sum_{i\le t}\mu^{t-i}g_i$，所以长期步长放大约 $1/(1-\mu)$ 倍，$\mu=0.9$ 时约 10 倍。方向一致的分量累积，来回振荡的分量相互抵消。

##### Step 3（i）公式：Momentum 更新

$$
\theta_t=\theta_{t-1}-\eta v_t
$$

##### Step 3（ii）代码

```python
# 沿速度方向更新
new_p[k] = params[k] - lr * new_v[k]
return new_p, {"v": new_v}
```

##### Step 3（iii）解释

另一种约定是 $v_t=\mu v_{t-1}-\eta g_t$、$\theta_t=\theta_{t-1}+v_t$。lr 不变时两者轨迹相同，lr 改变时历史速度的缩放不同。Demo 与 `torch.optim.SGD` 的差异在机器精度内，说明这里的约定与它一致。

##### Step 4（i）公式：Adam 更新一阶矩和二阶矩

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2
$$

##### Step 4（ii）代码

```python
# 步数加 1：第一次调用 t=1
t = state.get("t", 0) + 1
# 取历史矩，首次为空字典
m, s = state.get("m", {}), state.get("s", {})
# 新参数和新状态，不修改输入
new_p, new_m, new_s = {}, {}, {}
# 逐参数处理
for k in params:
    # 当前梯度，形状同参数
    g = grads[k]
    # 一阶矩，首次为 0，形状同参数
    new_m[k] = b1 * m.get(k, np.zeros_like(g)) + (1 - b1) * g
    # 二阶矩，逐元素平方，形状同参数
    new_s[k] = b2 * s.get(k, np.zeros_like(g)) + (1 - b2) * g * g
```

##### Step 4（iii）解释

`g * g` 是逐元素平方，每个参数维度有自己的二阶矩。`t` 是全局步数，所有参数共用同一个计数。

##### Step 5（i）公式：偏差校正

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad \hat{s}_t=\frac{s_t}{1-\beta_2^t}
$$

##### Step 5（ii）代码

```python
# 一阶矩偏差校正
m_hat = new_m[k] / (1 - b1 ** t)
# 二阶矩偏差校正
s_hat = new_s[k] / (1 - b2 ** t)
```

##### Step 5（iii）解释

矩从 0 开始累积，早期会偏小，所以除以累积权重之和。$t=1$ 时 $\hat{m}_1=g_1$，$\hat{s}_1=g_1^2$。

##### Step 6（i）公式：Adam 参数更新

$$
\theta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{s}_t}+\epsilon}
$$

##### Step 6（ii）代码

```python
# 逐元素自适应步长
new_p[k] = params[k] - lr * m_hat / (np.sqrt(s_hat) + eps)
return new_p, {"t": t, "m": new_m, "s": new_s}
```

##### Step 6（iii）解释

第一步的比值就是 $\mathrm{sign}(g)$，所以 Demo 中梯度从 $10^{-3}$ 变到 $10^3$，第一步位移都接近 lr。$\epsilon$ 防止除零，也让极小梯度的步长略小于 lr。

##### 5️⃣ 复杂度

设参数总元素数为 $P$：

$$
\text{单步时间}=O(P),\quad \text{状态空间}=\begin{cases}0, & \text{SGD}\\ P, & \text{Momentum}\\ 2P, & \text{Adam}\end{cases}
$$

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：Adam 的 bias correction 为什么依赖 step

把递推式展开，$m_t$ 是历史梯度的加权和。若梯度分布平稳，期望为：

$$
m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i,\quad \mathbb{E}[m_t]=\mathbb{E}[g](1-\beta_1^t)
$$

权重和 $1-\beta_1^t$ 随 $t$ 变化，所以校正因子必须用当前步数。$\hat{s}$ 尤其重要：$\beta_2=0.999$ 时 $1-\beta_2^t$ 增长缓慢。不做校正时，$t=1$ 的比值是：

$$
\frac{m_1}{\sqrt{s_1}}=\frac{(1-\beta_1)g}{\sqrt{1-\beta_2}\,|g|}=\frac{0.1}{\sqrt{0.001}}\mathrm{sign}(g)\approx 3.16\,\mathrm{sign}(g)
$$

第一步步长被放大约 3.16 倍。下面的输出显示，不校正时比值在 $t=10$ 附近达到约 6.5，$t=1000$ 时仍约为 1.26，校正后恒为 1。

```python
b1, b2, g0 = 0.9, 0.999, 2.0
for t_ in (1, 2, 10, 100, 1000):
    # 恒定梯度下的闭式解：m_t=(1-b1^t)g，s_t=(1-b2^t)g^2
    m_ = (1 - b1 ** t_) * g0
    s_ = (1 - b2 ** t_) * g0 ** 2
    # 不校正的比值
    raw = m_ / s_ ** 0.5
    # 校正后的比值，恒定梯度下恒为 sign(g)
    fix = (m_ / (1 - b1 ** t_)) / (s_ / (1 - b2 ** t_)) ** 0.5
    print(t_, "不校正:", round(raw, 4), "校正:", round(fix, 4))
```

##### Follow-up 2：Weight Decay 和 L2 正则在 Adam 中是否总等价

不等价。SGD 中两者一致（无 Momentum 时完全相同），因为步长是统一的 $\eta$。Adam 中 L2 的 $\lambda\theta$ 进入梯度后，会被 $\sqrt{\hat{s}}$ 缩放，梯度历史大的维度衰减更弱：

$$
\text{L2}:g_t\leftarrow g_t+\lambda\theta_{t-1}
$$

AdamW（Loshchilov & Hutter 2019）把衰减放到自适应缩放之外：

$$
\text{AdamW}:\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{s}_t}+\epsilon}
$$

下面同时验证 SGD 等价、Adam 不等价，并与 `torch.optim.Adam`（`weight_decay` 即 L2）和 `torch.optim.AdamW` 对齐：

```python
# 两个维度梯度尺度相差 1000 倍：(2,)
scale = np.array([1.0, 1000.0])
lam, lr_ = 0.1, 0.01
# 手写的两种写法，起点相同：(2,)
w_a, w_d = np.ones(2), np.ones(2)
st_a, st_d = {}, {}
# 官方的两种优化器，起点相同：(2,)
tw_a = torch.ones(2, dtype=torch.float64, requires_grad=True)
tw_d = torch.ones(2, dtype=torch.float64, requires_grad=True)
o_a = torch.optim.Adam([tw_a], lr=lr_, weight_decay=lam)
o_d = torch.optim.AdamW([tw_d], lr=lr_, weight_decay=lam)
# 损失梯度固定为 scale，只观察衰减项的差别
for _ in range(100):
    # L2：把 λw 加进梯度，再走 Adam
    pa, st_a = adam_np({"w": w_a}, {"w": scale + lam * w_a}, st_a, lr_)
    w_a = pa["w"]
    # AdamW：先衰减，再走不含 λw 的 Adam
    pd, st_d = adam_np({"w": w_d * (1 - lr_ * lam)}, {"w": scale},
                       st_d, lr_)
    w_d = pd["w"]
    # 官方优化器读取同一个固定梯度
    tw_a.grad = torch.from_numpy(scale.copy())
    tw_d.grad = torch.from_numpy(scale.copy())
    o_a.step()
    o_d.step()
print("L2   :", w_a, "官方差异:", abs(w_a - tw_a.detach().numpy()).max())
print("AdamW:", w_d, "官方差异:", abs(w_d - tw_d.detach().numpy()).max())
# SGD：两种写法在同一步内完全相同
w_s = np.array([1.0, 2.0])
sg = sgd_np({"w": w_s}, {"w": scale + lam * w_s}, {}, lr_)[0]["w"]
wd = sgd_np({"w": w_s * (1 - lr_ * lam)}, {"w": scale}, {}, lr_)[0]["w"]
print("SGD 差异:", abs(sg - wd).max())
```

恒定梯度下 Adam 每步移动约 lr，100 步后不含衰减的结果接近 0。输出中，L2 在大梯度维度上约为 $1.7\times 10^{-5}$，$\lambda w$ 几乎被 $\sqrt{\hat{s}}$ 压掉；小梯度维度约为 0.016，受影响更明显。AdamW 两个维度都约为 -0.047，衰减与梯度尺度无关。

##### Follow-up 3：参数不更新时如何定位梯度为 None、NaN 或学习率为 0

按“参数 → 梯度 → 优化器”的顺序查，每层给出明确判据：

$$
\theta_t-\theta_{t-1}=-\eta\,\mathrm{step}(g_t)=0\;\Rightarrow\;\eta=0\;\text{或}\;g_t\;\text{缺失或为零}
$$

- `requires_grad=False` 或没有传入优化器：参数被冻结或遗漏。
- `grad is None`：计算图断开（`detach`、`.item()`、`no_grad`），或者忘了 `backward`。`torch.optim` 会静默跳过这类参数，不报错。
- 梯度全 0：ReLU 死区、mask 全 0，或者 `zero_grad` 的位置错了。
- 梯度 NaN：此时参数通常变成 NaN，表现为 loss 异常。向前查 loss，用 `torch.autograd.set_detect_anomaly(True)` 定位算子。
- lr 为 0：打印 `param_groups`，warmup 第一步常见。

```python
def diagnose(model, opt):
    # 优化器管理的参数 id：用于发现遗漏
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    # 逐组打印学习率
    print("lr:", [g["lr"] for g in opt.param_groups])
    # 按“冻结 → 遗漏 → None → 非有限 → 全 0”依次判断
    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(name, "已冻结")
        elif id(p) not in in_opt:
            print(name, "未传入优化器")
        elif p.grad is None:
            print(name, "grad 为 None，图已断开或未反传")
        elif not p.grad.isfinite().all():
            print(name, "梯度含 NaN 或 Inf")
        elif p.grad.norm() == 0:
            print(name, "梯度全 0")
        else:
            print(name, "正常，范数", round(p.grad.norm().item(), 4))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 三层串联：a 与 b 是 (2→2)，c 是 (2→1)
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 2)
        self.c = torch.nn.Linear(2, 1)
        # d 没有接入 forward
        self.d = torch.nn.Linear(2, 1)
    def forward(self, x):
        # detach 切断 a 的回传；d 不参与输出
        return self.c(self.b(self.a(x).detach()))
torch.manual_seed(0)
net = Net()
# 冻结 b.weight
net.b.weight.requires_grad_(False)
# 优化器遗漏 c.bias，并把 lr 设为 0
opt = torch.optim.SGD(
    [net.a.weight, net.a.bias, net.b.bias, net.c.weight, net.d.weight],
    lr=0.0)
# 前向后反传，(1, 2) → (1, 1) → 标量
net(torch.ones(1, 2)).sum().backward()
# 手动注入 NaN，模拟数值爆炸
net.c.weight.grad[0, 0] = float("nan")
diagnose(net, opt)
```

#### Q037｜MLP 前向与完整反向传播

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 写一个两层 MLP 的 forward 和手工反向传播，至少支持 Linear + ReLU + MSE，最好用 NumPy。
- **输入输出 / 约束：** `X:[N,D]`，权重 `W1:[D,H]`、`W2:[H,C]`；返回 loss 和各梯度。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 矩阵求导中转置位置为何出现？用 shape 能否验证？
  2. ReLU 在 0 点的导数怎样约定？
  3. 和有限差分梯度检验比较时，需要多小的 `eps`？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

##### 1️⃣ Dense Opening

两层 MLP 的前向是“Linear → ReLU → Linear → MSE”，手工反向传播从输出层把误差信号逐层传回，结构是“缓存中间量 $Z,A,\hat{Y}$ → 残差梯度 $G$ → 输出层梯度 → ReLU 掩码 → 输入层梯度”，数学核心对象是链式法则中的 $G=\partial L/\partial\hat{Y}$。题面只给 `W1`、`W2`，所以这里不含 bias；MSE 对全部 $NC$ 个元素取平均，与 `F.mse_loss` 默认行为一致。

##### 2️⃣ Global Equations

前向（$X\in\mathbb{R}^{N\times D},W_1\in\mathbb{R}^{D\times H},W_2\in\mathbb{R}^{H\times C}$）：

$$
Z=XW_1,\quad A=\max(Z,0),\quad \hat{Y}=AW_2
$$

主目标：

$$
L=\frac{1}{NC}\|\hat{Y}-Y\|_F^2
$$

反向（$M=\mathbb{1}[Z>0]$，$\odot$ 为逐元素乘）：

$$
G=\frac{2}{NC}(\hat{Y}-Y),\quad \frac{\partial L}{\partial W_2}=A^\top G,\quad \frac{\partial L}{\partial W_1}=X^\top\left[(GW_2^\top)\odot M\right]
$$

##### 3️⃣ Implementation Blocks

##### （A）PyTorch版本（标准实现）

```python
import numpy as np
import torch
import torch.nn.functional as F

def mlp_grads_pt(X, Y, W1, W2):
    # (N, D) @ (D, H) → Z:(N, H)
    Z = X @ W1
    # 负值截断为 0：A:(N, H)
    A = Z.clamp(min=0)
    # (N, H) @ (H, C) → Y_hat:(N, C)
    Y_hat = A @ W2
    # 残差：(N, C)
    E = Y_hat - Y
    # 所有元素取平均：(N, C) → 标量
    loss = (E ** 2).mean()
    # dL/dY_hat：(N, C)
    G = 2 * E / E.numel()
    # (H, N) @ (N, C) → dL/dW2:(H, C)
    g2 = A.T @ G
    # (N, C) @ (C, H) → dL/dA:(N, H)
    dA = G @ W2.T
    # ReLU 掩码，0 点导数取 0：(N, H) → (N, H)
    dZ = dA * (Z > 0)
    # (D, N) @ (N, H) → dL/dW1:(D, H)
    g1 = X.T @ dZ
    return loss, g1, g2
```

##### （B）NumPy版本（苛刻面试版）

```python
def mlp_grads(X, Y, W1, W2):
    # (N, D) @ (D, H) → Z:(N, H)
    Z = X @ W1
    # 负值截断为 0：A:(N, H)
    A = np.maximum(Z, 0)
    # (N, H) @ (H, C) → Y_hat:(N, C)
    Y_hat = A @ W2
    # 残差：(N, C)
    E = Y_hat - Y
    # 所有元素取平均：(N, C) → 标量
    loss = (E ** 2).mean()
    # dL/dY_hat：(N, C)
    G = 2 * E / E.size
    # (H, N) @ (N, C) → dL/dW2:(H, C)
    g2 = A.T @ G
    # (N, C) @ (C, H) → dL/dA:(N, H)
    dA = G @ W2.T
    # ReLU 掩码，0 点导数取 0：(N, H) → (N, H)
    dZ = dA * (Z > 0)
    # (D, N) @ (N, H) → dL/dW1:(D, H)
    g1 = X.T @ dZ
    return loss, g1, g2
```

##### 最小可运行测试（Demo）

```python
torch.manual_seed(0)
# N, D, H, C 取不同值，shape 写错时会直接报错
N, D, H, C = 5, 4, 3, 2
X, Y = torch.randn(N, D).double(), torch.randn(N, C).double()
W1 = torch.randn(D, H).double().requires_grad_()
W2 = torch.randn(H, C).double().requires_grad_()
# autograd：官方 relu 与 mse_loss 组成同一计算图
loss_ag = F.mse_loss(torch.relu(X @ W1) @ W2, Y)
loss_ag.backward()
# 手写版本读取相同的数值
l_pt, g1_pt, g2_pt = mlp_grads_pt(X, Y, W1.detach(), W2.detach())
arr = [t.detach().numpy() for t in (X, Y, W1, W2)]
l_np, g1_np, g2_np = mlp_grads(*arr)
print("shape:", g1_np.shape, g2_np.shape)
print("A/B 差异:", abs(g1_pt.numpy() - g1_np).max(),
      abs(g2_pt.numpy() - g2_np).max())
print("autograd 差异:", abs(l_np - loss_ag.item()),
      abs(W1.grad.numpy() - g1_np).max(),
      abs(W2.grad.numpy() - g2_np).max())
```

##### 4️⃣ Step-by-step Derivation

##### 实现前置骨架

```python
import numpy as np

def mlp_grads(X, Y, W1, W2):
    pass
```

##### Step 1（i）公式：前向并缓存中间量

$$
Z=XW_1,\quad A=\max(Z,0),\quad \hat{Y}=AW_2
$$

##### Step 1（ii）代码

```python
# (N, D) @ (D, H) → Z:(N, H)
Z = X @ W1
# 负值截断为 0：A:(N, H)
A = np.maximum(Z, 0)
# (N, H) @ (H, C) → Y_hat:(N, C)
Y_hat = A @ W2
```

##### Step 1（iii）解释

反向要用到 $Z$（生成掩码）和 $A$（输出层梯度），所以都要保留。

##### Step 2（i）公式：损失与残差梯度

$$
L=\frac{1}{NC}\|E\|_F^2,\quad E=\hat{Y}-Y,\quad G=\frac{\partial L}{\partial\hat{Y}}=\frac{2}{NC}E
$$

##### Step 2（ii）代码

```python
# 残差：(N, C)
E = Y_hat - Y
# 所有元素取平均：(N, C) → 标量
loss = (E ** 2).mean()
# dL/dY_hat：(N, C)
G = 2 * E / E.size
```

##### Step 2（iii）解释

`E.size` 就是 $NC$。漏掉这个分母时，梯度与损失相差一个常数倍，与 autograd 的对比会立刻暴露。

##### Step 3（i）公式：输出层梯度

$$
dL=\mathrm{tr}(G^\top A\,dW_2)=\mathrm{tr}((A^\top G)^\top dW_2)\Rightarrow\frac{\partial L}{\partial W_2}=A^\top G
$$

##### Step 3（ii）代码

```python
# (H, N) @ (N, C) → dL/dW2:(H, C)
g2 = A.T @ G
```

##### Step 3（iii）解释

结果 shape 为 $(H,C)$，与 $W_2$ 一致。$A$ 中为 0 的隐藏单元对应的梯度分量也为 0。

##### Step 4（i）公式：误差传到 Z

$$
\frac{\partial L}{\partial A}=GW_2^\top,\quad \frac{\partial L}{\partial Z}=\frac{\partial L}{\partial A}\odot M,\quad M=\mathbb{1}[Z>0]
$$

##### Step 4（ii）代码

```python
# (N, C) @ (C, H) → dL/dA:(N, H)
dA = G @ W2.T
# ReLU 掩码，0 点导数取 0：(N, H) → (N, H)
dZ = dA * (Z > 0)
```

##### Step 4（iii）解释

$\hat{Y}=AW_2$ 关于 $A$ 的导数是右乘 $W_2^\top$。ReLU 是逐元素函数，所以它的雅可比是对角矩阵，实现上就是逐元素乘掩码。

##### Step 5（i）公式：输入层梯度

$$
\frac{\partial L}{\partial W_1}=X^\top\frac{\partial L}{\partial Z}
$$

##### Step 5（ii）代码

```python
# (D, N) @ (N, H) → dL/dW1:(D, H)
g1 = X.T @ dZ
return loss, g1, g2
```

##### Step 5（iii）解释

结果 shape 为 $(D,H)$，与 $W_1$ 一致。Demo 中，两个梯度与 autograd 的差异都在机器精度内。

##### 5️⃣ 复杂度

前向和反向都各有两次矩阵乘法：

$$
\text{时间}=O(NDH+NHC),\quad \text{额外空间}=O(NH+NC)
$$

全量有限差分检验要做 $2P$ 次前向，其中 $P=DH+HC$。

##### 6️⃣ Follow-up 讲解

##### Follow-up 1：矩阵求导中转置位置为何出现，用 shape 能否验证

转置来自 Frobenius 内积的迹形式。若 $dL=\mathrm{tr}(P^\top dW)$，则 $\partial L/\partial W=P$。用迹的循环性把 $dW$ 移到最右：

$$
dL=\mathrm{tr}(G^\top d\hat{Y}),\quad d\hat{Y}=A\,dW_2+dA\,W_2
$$

$$
\mathrm{tr}(G^\top dA\,W_2)=\mathrm{tr}(W_2G^\top dA)=\mathrm{tr}((GW_2^\top)^\top dA)
$$

$dA=M\odot(X\,dW_1)$。掩码可以从内积的一侧移到另一侧，再次用循环性得到 $X^\top$：

$$
\mathrm{tr}((GW_2^\top)^\top(M\odot X\,dW_1))=\mathrm{tr}\left(\left[X^\top\left((GW_2^\top)\odot M\right)\right]^\top dW_1\right)
$$

Shape 是必要条件：四个维度取不同值时，写错转置大多会直接报错。但它不是充分条件，比如 $H=D$ 时 `W1` 与 `W1.T` 的 shape 相同。所以还要用方向导数验证数值：

$$
\frac{L(W+\epsilon V)-L(W-\epsilon V)}{2\epsilon}\approx\mathrm{tr}\left(\left(\frac{\partial L}{\partial W}\right)^\top V\right)
$$

```python
Xn, Yn, W1n, W2n = arr
# 错误顺序：(N, C) @ (H, C) 维度不匹配，直接报错
try:
    G_bad = 2 * (np.maximum(Xn @ W1n, 0) @ W2n - Yn) / Yn.size
    G_bad @ W2n
except ValueError as err:
    print("shape 报错:", str(err)[:50])
_, g1_n, g2_n = mlp_grads(Xn, Yn, W1n, W2n)
# 随机方向 V1:(D, H)，只取前向的 loss
rv = np.random.default_rng(0)
V1 = rv.normal(size=W1n.shape)
e = 1e-6
lp = mlp_grads(Xn, Yn, W1n + e * V1, W2n)[0]
lm = mlp_grads(Xn, Yn, W1n - e * V1, W2n)[0]
# 迹内积 tr(g1^T V1)，即逐元素乘后求和
print((lp - lm) / (2 * e), (g1_n * V1).sum())
```

##### Follow-up 2：ReLU 在 0 点的导数怎样约定

ReLU 在 0 点不可导，次梯度（Subgradient）集合是 $[0,1]$。主流框架取 0，这里 `Z > 0` 与其一致：

$$
\partial\max(z,0)\big|_{z=0}=[0,1],\quad \text{约定 }\mathrm{ReLU}'(0):=0
$$

0 点约定只影响恰好落在折点上的样本，随机初始化时这种情况很少。一个例外是 $W_1$ 全零：所有 $Z=0$，所以 $A=0$、两个梯度都是 0，训练无法启动。

在折点处，中心差分取左右斜率的平均，与约定值不同，这不代表实现有错。取 $X=W_2=Y=1,W_1=0$，有 $L(W_1)=(\max(W_1,0)-1)^2$，右导数 -2，左导数 0，中心差分约 -1：

```python
one, zero = np.ones((1, 1)), np.zeros((1, 1))
# 手写约定：掩码为 0，g1=0
print("手写:", mlp_grads(one, one, zero, one)[1])
# 官方 relu 在 0 点的 autograd 同样为 0
w0 = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
F.mse_loss(torch.relu(w0), torch.ones(1, 1).double()).backward()
print("autograd:", w0.grad.item())
e = 1e-6
lf = lambda w: mlp_grads(one, one, w * one, one)[0]
print("中心差分:", (lf(e) - lf(-e)) / (2 * e))
print("右导数:", (lf(e) - lf(0.0)) / e, "左导数:", (lf(0.0) - lf(-e)) / e)
# W1 全零：两个梯度都是 0
print(mlp_grads(Xn, Yn, np.zeros_like(W1n), W2n)[1:])
```

##### Follow-up 3：和有限差分梯度检验比较时，需要多小的 `eps`

中心差分有两种误差：截断误差随 $\epsilon$ 减小而减小，舍入误差随 $\epsilon$ 减小而增大。其中 $u$ 是机器精度，$f'''$ 是三阶导数：

$$
\mathrm{err}(\epsilon)\approx\frac{\epsilon^2}{6}|f'''|+\frac{u|f|}{\epsilon},\quad \epsilon^*\sim u^{1/3}
$$

float64 下 $u\approx 2.2\times 10^{-16}$，$\epsilon^*\sim 6\times 10^{-6}$，所以常取 $10^{-6}$ 到 $10^{-5}$。float32 下 $u\approx 1.2\times 10^{-7}$，最佳精度很差，所以梯度检验要用 float64。本模型在激活模式固定的区域内对单个权重是二次函数，$f'''=0$，大 $\epsilon$ 的误差主要来自跨过 ReLU 折点。

```python
def numeric_grad(f, W, eps):
    # 与 W 同形的数值梯度：(D, H) 或 (H, C)
    g = np.zeros_like(W)
    # 逐元素扰动，共 2*W.size 次前向
    for i in np.ndindex(W.shape):
        old = W[i]
        W[i] = old + eps
        fp = f()
        W[i] = old - eps
        fm = f()
        # 恢复原值，再写中心差分
        W[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g
def rel_err(a, b):
    # 相对误差：不受梯度整体尺度影响
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))
for dt in (np.float64, np.float32):
    # 转换精度后重新计算解析梯度
    Xd, Yd = Xn.astype(dt), Yn.astype(dt)
    W1d, W2d = W1n.astype(dt).copy(), W2n.astype(dt).copy()
    _, g1_d, g2_d = mlp_grads(Xd, Yd, W1d, W2d)
    f = lambda: mlp_grads(Xd, Yd, W1d, W2d)[0]
    for eps in (1e-1, 1e-3, 1e-5, 1e-6, 1e-8, 1e-11):
        n1 = numeric_grad(f, W1d, eps)
        n2 = numeric_grad(f, W2d, eps)
        print(dt.__name__, eps, "W1:", f"{rel_err(n1, g1_d):.1e}",
              "W2:", f"{rel_err(n2, g2_d):.1e}")
```

输出符合上述分析：float64 下 $\epsilon=10^{-3}$ 到 $10^{-6}$ 的相对误差都在 $10^{-9}$ 以内，$\epsilon=10^{-1}$ 时 $W_1$ 因跨过折点升到 $10^{-2}$ 量级，$W_2$ 不经过 ReLU，仍在机器精度。$\epsilon=10^{-11}$ 时舍入误差升到 $10^{-5}$ 量级。float32 最好只到约 $10^{-4}$，$\epsilon=10^{-8}$ 时相对误差已接近 1。

判据建议：float64 下取 $\epsilon=10^{-6}$，相对误差小于 $10^{-7}$ 视为通过。若个别元素误差很大，先检查对应样本的 $Z$ 是否接近 0。

## 03｜推荐系统指标、排序与召回

#### Q038｜手写 AUC：排序法与成对比较

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-08（公开面经日期）
- **题目：** 实现二分类 ROC-AUC，不直接调用 sklearn；先写正负样本成对比较法，再写按 score 排序/秩次的高效版。
- **输入输出 / 约束：** `y_true:[N]` 取 0/1，`score:[N]`，输出浮点 AUC。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 预测分数相等如何记半个正确 pair？平均秩如何算？
  2. 若全正或全负，AUC 是否有定义？接口约定如何声明？
  3. `O(N²)` 怎样优化到 `O(NlogN)`？
- **来源：** [Jointaro：TikTok San Diego RecSys MLE/RS 手撕 AUC](https://www.jointaro.com/interviews/companies/tiktok/experiences/mlers-san-diego-california-august-1-2025-no-offer-neutral-1d743d60/)

#### Q039｜Request / User 维度 GAUC

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（发布日）
- **题目：** 按 Request 或 User 进行分组，分别计算 AUC，然后按给定权重加权汇总。
- **输入输出 / 约束：** `group_id,y_true,score:[N]`，权重策略作为参数；返回 GAUC。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 某个 group 只有正例或负例，跳过还是另设规则？
  2. 按曝光量、正负 pair 数、用户等权分别代表什么？
  3. 为什么全局 AUC 与 GAUC 可能朝不同方向变化？
- **来源：** [牛客：字节抖音推荐算法二面 Request 维度 GAUC](https://www.nowcoder.com/feed/main/detail/8aba7094ae844c3da7e31730f042f037)

#### Q040｜AUC 的多分类扩展（OvR / OvO）

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-03（帖子发布时间）
- **题目：** 给定多分类概率，按 One-vs-Rest 或 One-vs-One 构造多分类 AUC，并解释宏/微平均。
- **输入输出 / 约束：** `y:[N]`、`proba:[N,C]`；返回宏/微 AUC（约定方式）。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 各类别样本极不平衡时 macro 与 weighted 如何区别？
  2. 多分类为什么不能简单把类别 id 当成二分类 score？
  3. 当某一类在测试集没有正例时应如何处理？
- **来源：** [牛客：字节推荐实习 MHA、AUC 变体](https://www.nowcoder.com/users/615039244)

#### Q041｜NDCG@K：按 Request 计算

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给定每个 Request 的商品相关性标签和模型排序分数，手动实现 DCG/IDCG/NDCG@K。
- **输入输出 / 约束：** 每组 `(relevance,score)` 与 `K`；返回每组及均值。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. gain 用 `rel` 还是 `2^rel-1`？
  2. ties 与没有相关商品时 NDCG 怎么定义？
  3. 为什么按 Request 平均与把所有样本混排不等价？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q042｜Precision / Recall / F1 / Confusion Matrix

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 实现二分类 TP、FP、TN、FN、Precision、Recall、F1 指标；允许指定预测阈值。
- **输入输出 / 约束：** `y:[N]`、`score:[N]`、threshold；输出字典。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 正例极少时 Accuracy 为什么误导？
  2. Precision=0/0 时如何约定 `zero_division`？
  3. 改变阈值为何会使 Precision 和 Recall 权衡？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q043｜高频 Top-K / 召回候选排序

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06～07（分场次）
- **题目：** 输入 item id 与出现次数或预测分数，输出 Top-K，规定 ties 的优先级。
- **输入输出 / 约束：** `items` / `scores`、`k`；返回有序列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 堆、排序、桶计数分别适用于什么约束？
  2. 数据量大且流式输入时如何更新结果？
  3. 同分多个 item 的确定性顺序如何确保？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q044｜DIN Target Attention 前向

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 写出 DIN 的核心 Target-Aware Attention：将目标 item 和历史行为 Embedding 拼接交互后，生成每个历史行为的权重，再聚合。
- **输入输出 / 约束：** `target:[B,D]`、`hist:[B,T,D]`、`mask:[B,T]`；输出 `[B,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么经典 DIN 的 attention 权重不一定做 softmax 归一化？
  2. `[target,hist,target-hist,target*hist]` 拼接后特征维数是多少？
  3. 长度为 0、Padding 全满和长序列如何处理？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

#### Q045｜双塔召回：In-Batch Negatives 训练步

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 写一个最小双塔模型：用户 Embedding 与 Item Embedding 的矩阵相似度，使用对角正样本训练。
- **输入输出 / 约束：** `user_emb,item_emb:[B,D]`，返回 `[B,B]` logits 和 CE loss。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 热门 item 在 batch 重复，会出现什么 false negatives？
  2. 怎样用样本权重或采样校正减轻分布偏差？
  3. 线上 ANN 召回如何与训练时的相似度定义保持一致？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q046｜流式 AUC：ROC 点与阈值扫描

- **优先级：** P2 ｜ **证据：** B ｜ **年月：** 2026-01（练习页面更新；非面试日期）
- **题目：** 在 AUC 实现基础上输出 ROC 曲线点 `(FPR,TPR)`，阈值按 score 从高到低变化。
- **输入输出 / 约束：** `y_true,score`；返回一系列 `(FPR,TPR,threshold)`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 同一分数的样本为什么应一起跨越阈值？
  2. 如何由梯形面积得到 AUC？
  3. 流式数据内存不足时可以用什么近似策略？
- **来源：** [PracHub：TikTok AUC/Softmax/Logistic Regression 练习页](https://prachub.com/companies/tiktok/categories/machine-learning)

## 04｜NumPy、传统机器学习、数值编程

#### Q047｜NumPy 梯度下降训练 Linear Regression

- **优先级：** P0 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 不能调用 sklearn，自己实现 `fit(X,y)`：预测、均方误差、梯度、更新，训练到给定步数。
- **输入输出 / 约束：** 输入 `X:[N,D]`、`y:[N]`；返回 `w:[D]`、`b`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 `X.T@error` 的矩阵乘法维度正确？
  2. 如果没有除以 N，学习率需要怎样调整？
  3. 如何判断 loss 不降是梯度错误、特征尺度还是学习率过大？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

#### Q048｜K-Means（欧氏距离版本）

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现 K-Means 迭代：初始化 K 个质心、计算距离、分簇、更新质心，直到收敛/迭代上限。
- **输入输出 / 约束：** `X:[N,D]`、`K`；输出 `labels:[N]`、`centroids:[K,D]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 空簇出现时保留旧质心还是重置？
  2. 怎样广播得到 `[N,K]` 距离矩阵，避免 Python 双循环？
  3. 随机初始化与 KMeans++ 的收敛差异是什么？
- **题面/来源说明：** 过去调研出现 TikTok OA 二手整理版本；本轮将准确表述为训练拓展，未核实单场原卷。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q049｜Cosine K-Means

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 改造 K-Means：按余弦相似度为每个样本选择质心，并明确质心更新后的归一化策略。
- **输入输出 / 约束：** `X:[N,D]`、K；输出 normalized centroids 与 labels。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么原始向量最好先进行 L2 normalization？
  2. 质心更新后为什么需要重新归一化？
  3. 零向量、空簇与同分数如何处理？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q050｜Bagging Classifier：Bootstrap + 多数投票

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 年月未核实
- **题目：** 已提供若干基分类器，按随机种子执行 Bootstrap 有放回抽样、逐一拟合，并对测试样本多数投票。
- **输入输出 / 约束：** `xTrain,yTrain,xTest,nEstimators,seed`；返回每条测试样本标签。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 样本与 label 要怎样索引才能始终成对？
  2. 多个 label 平票时按题面选择哪一个？
  3. Bootstrap 为什么有放回，Bagging 如何减少模型方差？
- **题面/来源说明：** 该练习页明确自述约 85% 对应可见源题；细节为可复现而改写。
- **来源：** [FastPrep：TikTok Bagging OA 改编题（其自述约 85% 匹配）](https://www.fastprep.io/problems/tiktok-bagging-classifier)

#### Q051｜Entropy / Information Gain 计算

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 不给高级 ML API，计算分类标签熵，并对某个连续特征的阈值分裂计算信息增益。
- **输入输出 / 约束：** 输入 `y` 与左右划分索引，返回 entropy 和 gain。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 空集合/纯节点的熵如何处理？
  2. 为什么左右熵要按样本数加权？
  3. 能否用 Gini 替代 Entropy，结果一定相同吗？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q052｜决策树 Best Split / Predict

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 实现深度有限的 CART 树关键流程：遍历特征与阈值，选最优分裂，再用节点结构对新样本预测。
- **输入输出 / 约束：** `X:[N,D]`、`y:[N]`，返回 tree nodes 与预测标签。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 候选阈值为什么可选相邻不同值的中点？
  2. 如何避免相同值/无信息增益导致无限分裂？
  3. 预测时的缺失值和树深度限制如何处理？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q053｜局部极值（Local Maxima）识别

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 在一维数组里找到满足局部极大值条件的所有位置；题目开始必须声明严格大于还是允许等号。
- **输入输出 / 约束：** 输入数组 `nums:[N]`，返回索引列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 相同高度构成 plateau 时是否算局部极大值？
  2. 数组首尾是否参与比较？
  3. 若只要求找到一个峰值，如何用 O(logN) 二分？
- **题面/来源说明：** 完整“所有峰值”作为经典练习；用户另提供 O(logN) 局部最小值题。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q054｜O(logN) 寻找一个局部最小值

- **优先级：** P1 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 对相邻元素不相等的数组，找到一个严格小于其有效相邻元素的局部最小值，争取对数复杂度。
- **输入输出 / 约束：** `nums:[N]`，返回一个合法 index；按题面明确边界。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 若相邻元素允许相等，二分正确性如何变化？
  2. 为什么根据 `nums[mid]` 和 `nums[mid+1]` 的趋势就能丢弃半侧？
  3. 长度为 1/2 时边界条件如何规定？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

#### Q055｜从零实现线性/多类预测的完整 Batch 训练

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 把模型训练拆成 `forward`、`loss`、`backward`、`step`，不使用自动求导。
- **输入输出 / 约束：** 小数据 `X:[N,D]`、`y`；返回 train history 与参数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 怎样用数值差分检查 backward？
  2. Broadcasting 导致梯度静默错误如何发现？
  3. 如何用 mini-batch 与 shuffle 扩展？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q056｜复合概率函数（题面未完整公开）

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 公开记录只提“Composite Probability Function”或概率组合函数。将其作为开放式模拟：给定独立事件概率，计算并/交/条件概率；面试时先确认具体定义。
- **输入输出 / 约束：** 输入多个事件的概率；输出题面指定的组合概率。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 事件是否独立？为何不能默认为独立？
  2. 概率为 0/1 的边界如何处理？
  3. 多事件时怎样避免直接把概率相乘导致下溢？
- **题面/来源说明：** 最初回合 Glassdoor 的题名不足以还原唯一函数规范；本题不是声称逐字原题。 先前整理曾提及此题型，但未找到可核验的单场原始题面。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q057｜求立方根：二分 / 牛顿迭代

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 实现数值 `cuberoot(x)`，不依赖内建幂函数直接得到答案；满足面试官指定的绝对/相对误差。
- **输入输出 / 约束：** 输入任意正/负实数 `x`，返回浮点立方根。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. `x<0` 怎样处理？`|x|<1` 的二分搜索区间选什么？
  2. Newton 更新公式如何推导？
  3. 绝对误差与相对误差在接近 0 时有何区别？
- **题面/来源说明：**  先前整理曾提及此题型，但未找到可核验的单场原始题面。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q058｜算术平方根，保留两位小数

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（帖子发布时间）
- **题目：** 对非负数求平方根并按题意保留两位小数，不直接使用 `math.sqrt`。
- **输入输出 / 约束：** 输入 `x>=0`；返回两位小数的数值/字符串（按题面）。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 二分终止误差应如何选，才能避免最终舍入错一位？
  2. 如果 x 在 `(0,1)`，初始右边界怎样设置？
  3. 如何区分四舍五入与截断？
- **来源：** [牛客：27 搜推算法 TikTok 一面，MHA + 算术平方根](https://www.nowcoder.com/users/214711400)

#### Q059｜随机均匀采样器 / Randomized Choice

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 设计一个对 N 个元素等概率采样的函数，并做统计测试；可拓展到 Reservoir Sampling。
- **输入输出 / 约束：** 输入 `items`，输出一个采样元素；流式版输入 iterator。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何证明每个元素概率均为 `1/N`？
  2. 未知长度且不能保存全部元素时如何做 Reservoir Sampling？
  3. 如何用随机种子和卡方/频数统计验证正确性？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q060｜实现二分类评估与阈值选择

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给出预测概率与标签，扫描阈值并找到 F1 最大的阈值，输出 Precision/Recall 曲线摘要。
- **输入输出 / 约束：** `score,y`，返回最佳阈值及指标。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 测试集选择阈值是否构成数据泄漏？
  2. PR-AUC 和 ROC-AUC 在长尾场景有什么区别？
  3. 若业务约束 `Recall>=0.95`，如何修改选阈目标？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

## 05｜TikTok OA：题目和编程填空

#### Q061｜OA：梯度下降的线性回归训练步骤填空

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 把已有模板的预测、MSE、`dw/db`、学习率更新补齐，并确保形状正确。
- **输入输出 / 约束：** 输入 Python/NumPy 模板 + `X,y`，提交完整可运行函数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 模板里如果把 bias 包含进 `w`，矩阵该如何扩维？
  2. 少写 `1/N` 会对优化表现有何影响？
  3. 数值测试怎样验证一次 step 的参数变化？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q062｜OA：最长连续相同字符段

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 遍历字符串，找最长相同字符连续段，若长度相同按指定的“最靠右”规则打破平局。
- **输入输出 / 约束：** 字符串 `s`，返回字符、长度或区间（按题面）。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 空串与全相同字符如何处理？
  2. 更新最佳区间的比较符号是 `>` 还是 `>=`？
  3. 输入包含 Unicode、多字节字符时以什么为单位？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q063｜OA：F1 手算及实现

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-09（报道发布时间；面试日未核实）
- **题目：** 给定混淆矩阵 TP=90, FP=10, FN=30, TN=70，计算 Precision、Recall、F1，并写函数泛化。
- **输入输出 / 约束：** 输入四个计数，返回指标，注明零分母约定。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 F1 是调和平均而不是算术平均？
  2. `F_beta` 怎样调整 Recall 权重？
  3. 用于多分类时 micro/macro 分别怎么做？
- **来源：** [InterviewShow：TikTok MLE OA 7 选择 + 3 实现](https://interviewshow.com/tiktok-mle-oa-ml-implementation-questions/)

#### Q064｜OA：复杂状态模拟

- **优先级：** P2 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-12（公开报道日期）
- **题目：** 给定系统操作序列，按状态机规则执行并产生结果（原帖仅公开“复杂模拟和状态维护”，无完整题面）。
- **输入输出 / 约束：** 根据模拟练习题自设指令表，输入 operations，输出状态与结果。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何把状态转移与输入解析分离？
  2. 如何对边界状态写参数化单元测试？
  3. 规模大时怎样避免重复扫描历史记录？
- **题面/来源说明：** 原始报告没有具体操作规则，因此保留为“题型”，不得当作完整原题。
- **来源：** [TikTok Research Scientist Intern OA 4 题 CodeSignal](https://interviewexperiences.in/experience/tiktok/tiktok-oa-report-research-assistant-internship-dec-2025)

#### Q065｜OA：字符串计数 / 排列组合

- **优先级：** P2 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-12（公开报道日期）
- **题目：** 给定有限长度字符串，编写计数或排列组合函数（原帖仅给类别，没有唯一题面）。
- **输入输出 / 约束：** 输入字符串/字符频次与长度上限；输出组合数量。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 有重复字符时为何不能简单 `n!`？
  2. 答案需要取模时如何处理阶乘和逆元？
  3. 如何用短字符串暴力枚举验证公式？
- **题面/来源说明：** 原始报告没有具体完整题面；标注“题型”。
- **来源：** [TikTok Research Scientist Intern OA 4 题 CodeSignal](https://interviewexperiences.in/experience/tiktok/tiktok-oa-report-research-assistant-internship-dec-2025)

## 06｜LeetCode / DSA：明确出现的题与贴近变体

#### Q066｜LC 207 / 210 Course Schedule：拓扑排序

- **优先级：** P0 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** 给定课程依赖有向图，判断能否完成所有课程；扩展为返回一种合法课程顺序。
- **输入输出 / 约束：** `numCourses`、`prerequisites:[[a,b]]`，返回 bool 或 topological order。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Kahn BFS 与 DFS 三色标记如何检测环？
  2. 有多个合法顺序时是否必须固定输出顺序？
  3. 为什么 2026-08～10 的趋势汇总特别强调拓扑排序？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q067｜Alien Dictionary：外星文字典序

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** 根据已按未知字母顺序排序的字符串列表构造先后关系，输出合法字母顺序。
- **输入输出 / 约束：** 输入字符串列表，返回字符顺序或空串。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 前缀冲突例如 `abc` 在 `ab` 之前为什么无解？
  2. 重复边与入度计数如何去重？
  3. 循环依赖时如何判断？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q068｜LC 124 Binary Tree Maximum Path Sum

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-02（公开面经日期）
- **题目：** 二叉树每条路径可在任意节点开始结束，不一定过根；求最大节点值之和。
- **输入输出 / 约束：** 输入 TreeNode，输出整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 DFS 对父节点只能返回一条单向向下增益？
  2. 全负数时初始化为 0 会出什么问题？
  3. 若要求输出具体路径，该保存什么信息？
- **来源：** [Jointaro：TikTok MLE CE 推导 + Transformer Block + 树题](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/)

#### Q069｜LC 297 二叉树序列化/反序列化

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-02（公开面经日期）
- **题目：** 设计 `serialize(root)` 与 `deserialize(data)` 使任意二叉树可往返恢复。
- **输入输出 / 约束：** 输入树，输出字符串；输入字符串，输出树。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何表示空节点，为什么不能省略所有空节点？
  2. DFS 前序和 BFS 层序哪个更便于解析？
  3. 树只有一个节点或完全偏斜时如何处理？
- **来源：** [Jointaro：TikTok MLE CE 推导 + Transformer Block + 树题](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/)

#### Q070｜LC 33 搜索旋转排序数组

- **优先级：** P0 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-02（公开面经日期）
- **题目：** 在无重复元素的旋转有序数组中用 O(logN) 搜索目标值。
- **输入输出 / 约束：** `nums:[N]`、target；返回 index 或 -1。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 每轮如何判断哪一半单调有序？
  2. 为什么等于 `nums[mid]` 必须先处理？
  3. 如果允许重复元素，最坏复杂度会发生什么变化？
- **来源：** [Jointaro：TikTok MLE CE 推导 + Transformer Block + 树题](https://www.jointaro.com/interviews/companies/tiktok/experiences/machine-learning-engineer-united-states-february-1-2025-no-offer-neutral-d88842c0/)

#### Q071｜LC 146 LRU Cache

- **优先级：** P0 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** 实现 `get/put` 均摊 O(1) 的最近最少使用缓存，超容量时淘汰最久未访问项。
- **输入输出 / 约束：** 类接口 `LRUCache(capacity)`、`get(key)`、`put(key,value)`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 哈希表 + 双向链表中，get 命中后如何更新最近使用顺序？
  2. 容量为 0/1、重复 put、删除头尾等边界怎么处理？
  3. 扩展为 TTL 或 LFU，哪些逻辑必须改变？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q072｜LRU + TTL：带过期时间缓存

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** LRU Cache 增加每条记录过期时间（TTL），保证不会返回已过期值。
- **输入输出 / 约束：** `put(key,value,ttl)`、`get(key,now)`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 过期是惰性删除还是主动清扫？
  2. 过期项占用容量时应该优先淘汰谁？
  3. 时钟回拨、毫秒/秒混淆和并发访问如何测试？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q073｜LC 347 前 K 个高频元素

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** 返回数组中出现次数最高的 K 个值，按题目要求处理同频。
- **输入输出 / 约束：** 整数列表、`k`；返回 K 元素列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 堆与桶排序的时间复杂度怎样比较？
  2. 频率相同时输出顺序是否要求固定？
  3. 输入为数据流时如何近似统计 Top-K？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q074｜LC 155 Min Stack

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 设计支持 `push/pop/top/getMin` 都为 O(1) 的栈。
- **输入输出 / 约束：** 类接口，空栈操作按题面定义。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 辅助栈如何处理重复最小值？
  2. 只保存前缀最小值与保存差值的方案比较？
  3. 如果还要 `popMin`，数据结构会怎么变？
- **来源：** [Screna：TikTok MLE Stack/Median/Tree](https://www.screna.ai/experience/a72893f0-f35c-4696-99b5-d9b030bef8f7)

#### Q075｜LC 716 Max Stack

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 支持 `push/pop/top/peekMax/popMax`，重复最大值时删除最靠近顶部者。
- **输入输出 / 约束：** 类接口，重点实现 `popMax`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 仅用两个栈是否能做到全部操作 O(1)？
  2. 如何结合双向链表与有序结构处理 `popMax`？
  3. 相同值多个节点时怎样保留正确的 LIFO 决策？
- **来源：** [Screna：TikTok MLE Stack/Median/Tree](https://www.screna.ai/experience/a72893f0-f35c-4696-99b5-d9b030bef8f7)

#### Q076｜LC 295 Median Finder

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 动态插入数字后能实时返回当前所有数字的中位数。
- **输入输出 / 约束：** `addNum(x)`、`findMedian()`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 大根堆与小根堆维护哪些不变量？
  2. 插入负数、重复数、极大值时是否正确？
  3. 若需要删除旧数据做滑动窗口中位数，双堆怎么扩展？
- **来源：** [Screna：TikTok MLE Stack/Median/Tree](https://www.screna.ai/experience/a72893f0-f35c-4696-99b5-d9b030bef8f7)

#### Q077｜LC 300 最长递增子序列 LIS + 路径恢复

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 返回严格递增子序列的最大长度，并额外返回一个合法的原数组下标序列；要求 O(nlogn)。
- **输入输出 / 约束：** 整数数组，返回 `(length, indices)`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. **[原始追问]** 若允许非递减而非严格递增，二分逻辑如何变化？
  2. 如何保存 predecessor 重建一个实际 LIS？
  3. 有重复数字时如何保证结果仍严格递增？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q078｜LC 673 最长递增子序列数量

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 计算数组中最长严格递增子序列的个数。
- **输入输出 / 约束：** `nums`，返回计数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. `length[i]` 与 `count[i]` 如何同步 DP？
  2. 相同的数出现在不同索引上能否形成不同子序列？
  3. 有没有 Fenwick Tree / Segment Tree 优化方案？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q079｜LC 560 连续子数组和为 K

- **优先级：** P0 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-07（面经标注月份）
- **题目：** 对可能含负数的数组，返回和为 K 的非空连续子数组数量。
- **输入输出 / 约束：** `nums,k`，返回整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么单纯双指针不适合有负数的情况？
  2. **[原始追问]** 假设所有数字均非负，如何用 O(n) 时间、O(1) 额外空间返回一个合法区间？
  3. 若数组全非负，只要求返回一段索引，怎样用 O(1) 额外空间？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q080｜LC 218 Skyline 天际线问题

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06（面经标注月份）
- **题目：** 给定多个矩形建筑，输出合并后的建筑轮廓关键点，最后需要落到高度 0。
- **输入输出 / 约束：** `buildings:[[L,R,H]]`，返回 `[[x,height]]`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. **[原始追问]** 当多个建筑边界共享 x 时，进入和退出事件怎样排序？
  2. 为什么需要堆的延迟删除，或平衡树维护高度？
  3. 相邻点高度相同为何应去重？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q081｜LC 139 Word Break：第二小拆分成本

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06（面经标注月份）
- **题目：** 给定字符串和词典，成本为拆分使用的单词数，返回所有合法拆分方案中“第二小”的成本（要求先确认是否按方案区分还是按不同成本区分）。
- **输入输出 / 约束：** `s,wordDict`，返回第二小成本或 -1。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何只保存每个前缀的最小成本不够？
  2. 若多个不同方案成本都相同，“第二小”到底是第二个方案还是第二个不同值？
  3. 词典特别大时可否用 Trie 优化前缀枚举？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q082｜LC 54 螺旋矩阵

- **优先级：** P1 ｜ **证据：** A（结构化候选人记录） ｜ **年月：** 2026-06～07（分场次）
- **题目：** 顺时针打印一个二维矩阵中的所有元素。
- **输入输出 / 约束：** `matrix:[M,N]`，返回一维数组。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 上/下/左/右四个边界何时收缩？
  2. 单行、单列、空矩阵如何避免重复输出？
  3. 是否能用方向数组+visited 实现？
- **来源：** [Screna：TikTok MLE 面试记录](https://www.screna.ai/interview-questions/tiktok/machine-learning-engineer)

#### Q083｜LC 25 K 个一组反转链表

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（发布日）
- **题目：** 每连续 K 个节点作为一组反转，最后不足 K 个原样保留。
- **输入输出 / 约束：** 链表头、`k`，返回新头。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何先检测剩余是否有 K 个节点？
  2. 原地反转如何保证前后片段正确连接？
  3. 空间 O(1) 的迭代法怎么写？
- **来源：** [牛客：字节内容消费团队二面 K 组翻转 / 抛硬币](https://www.nowcoder.com/discuss/878370699188916224)

#### Q084｜LC 1249 移除无效括号

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 从包含字母与括号的字符串里删除最少括号，使结果有效。
- **输入输出 / 约束：** 字符串，返回一种合法输出。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 两遍扫描与栈标记，哪一种更容易保证最少删除？
  2. 多种答案是否均可接受？
  3. 如果要求返回所有最少删除结果，应如何改为 BFS/回溯？
- **题面/来源说明：** 字节面经相关题型；未核实该题对应的精确场次月份。
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q085｜LC 15 三数之和

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-06（文章更新日，非面试日）
- **题目：** 寻找数组中所有和为 0 的不重复三元组。
- **输入输出 / 约束：** 整数数组，返回三元组列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 排序后怎样跳过重复 i、left、right？
  2. 为何 O(n²) 是标准解法？
  3. 如果求最接近 target 的三数之和怎么变？
- **来源：** [IGotAnOffer：TikTok MLE 编程题指南](https://igotanoffer.com/en/advice/tiktok-machine-learning-engineer-interview)

#### Q086｜LC 72 编辑距离

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 计算将字符串 A 变为 B 的最少插入、删除、替换次数。
- **输入输出 / 约束：** 两个字符串，返回最小操作数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. DP 状态如何定义；三种转移分别对应什么操作？
  2. 如何把 O(mn) 空间压缩为 O(min(m,n))？
  3. 如果要求输出操作序列，为什么不能只保留滚动数组？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q087｜LC 63 Unique Paths II：带障碍物路径数

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 网格某些位置不能通过，只能向右/向下移动，求从左上到右下不同路径数。
- **输入输出 / 约束：** `obstacleGrid`，返回整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 初始格为障碍时如何处理？
  2. 一维 DP 更新顺序为什么必须从左往右？
  3. 答案极大且要求取模时要注意什么？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q088｜LC 232 双栈模拟队列

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-09（面试时间）
- **题目：** 只用两个 Stack 完成 Queue 的入队和出队。
- **输入输出 / 约束：** `appendTail/deleteHead` 或 `push/pop/peek`。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么必须只在出栈为空时搬运入栈元素？
  2. 均摊 O(1) 如何用摊还分析解释？
  3. 空队列弹出与顺序测试怎样写？
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

#### Q089｜LC 407 二维接雨水

- **优先级：** P2 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给定二维高度图，计算雨水能积多少单位；利用边界最小堆向内推进。
- **输入输出 / 约束：** `heightMap:[M,N]`，返回体积。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何不能直接把一维左右最大值方法推广到二维？
  2. 访问标记应该在入堆还是出堆时设置？
  3. 边界高度如何保证单调“水位”约束？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q090｜LC 32 最长有效括号

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-07（面试时间）
- **题目：** 给定只含圆括号的字符串，输出最长连续合法括号子串长度。
- **输入输出 / 约束：** 字符串，返回整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 栈里为何初始化一个 `-1` 哨兵？
  2. 如何用两次计数扫描得到 O(1) 空间？
  3. 前缀合法不等于整个子串合法，如何避免误判？
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

#### Q091｜LC 200 岛屿数量（DFS / BFS 两种写法）

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-08（面试日期）
- **题目：** 给定 0/1 网格，计算四方向连通的岛屿数量，分别实现 DFS 与 BFS。
- **输入输出 / 约束：** `grid:[M,N]`，返回数量。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 何时标记 visited 才能避免重复入队？
  2. 若不允许修改 grid，需要怎样处理？
  3. 递归 DFS 在极大网格时有何风险？
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

#### Q092｜LC 287 寻找重复数

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 长度 N+1 数组，数字范围 `[1,N]`，只存在一个重复值，要求不修改数组并使用 O(1) 额外空间。
- **输入输出 / 约束：** 数组，返回重复元素。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Floyd 快慢指针怎样把数组解释成链表？
  2. 二分数值范围的判定条件是什么？
  3. 如果多种数字均重复，原题条件失效后如何处理？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q093｜LC 42 一维接雨水

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 给定柱状高度图，计算雨水积水总量。
- **输入输出 / 约束：** `height:[N]`，返回整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 双指针为何只需左右最大值？
  2. 单调栈解法中凹槽宽和高如何求？
  3. 空数组、单调数组、平台的结果如何？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q094｜逆序对 / 反转对统计

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 统计数组中满足前索引元素大于后索引元素的逆序对数量（若是 LC493 需改为 `nums[i]>2*nums[j]`）。
- **输入输出 / 约束：** 整数数组，返回计数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么归并排序能在 O(nlogn) 完成？
  2. 有负数时 reverse pairs 的比较条件是否仍成立？
  3. 相等元素算不算逆序对？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q095｜LC 40 Combination Sum II

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 每个候选数字最多使用一次，返回所有和为 target 的不重复组合。
- **输入输出 / 约束：** `candidates,target`，返回组合列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 排序后如何在同一递归层去重而不丢失合法解？
  2. 为什么递归时使用 `i+1`？
  3. 当数字中包含 0 或负数时原有剪枝是否仍有效？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q096｜三维箱子旋转堆叠最大高度

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（面试日；11月发布）
- **题目：** 给定可旋转长方体箱子，要求每个上层箱子的底面长宽严格小于下层；求最大高度（先确认同一物理箱子可否重复使用）。
- **输入输出 / 约束：** `boxes:[[a,b,c]]`，输出最大堆叠高度。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 一个箱子的多个旋转状态是否可同时出现在同一个答案中？
  2. 经典箱子堆叠 DP 的状态怎么定义？
  3. 同底面尺寸如何处理严格不等式与排序 tie？
- **题面/来源说明：** 部分公开解法允许一个物理箱子使用多个旋转状态；若题面禁止重复使用，需额外约束，不能直接照抄经典 DP。
- **来源：** [牛客：字节大模型算法三面 BCE/VAE/积木堆叠](https://www.nowcoder.com/discuss/946798743163088896)

#### Q097｜把数组分为两段，最小化方差和

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 在每个合法切分点切成左右两段，求两段各自方差之和最小的切分点。
- **输入输出 / 约束：** 实数数组，输出最优位置与目标值。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 用前缀和与平方和如何 O(1) 算出一段方差？
  2. 是总体方差还是样本方差，是否允许长度 1？
  3. 数值很大时 `E[x²]-E[x]²` 可能有何精度问题？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q098｜大数相加

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-09（面试时间）
- **题目：** 不把整段十进制字符串转成整数，自己实现两个超长非负整数的加法。
- **输入输出 / 约束：** `a,b` 为数字字符串，返回数字字符串。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何处理末尾进位、前导 0 和一个输入为空？
  2. 如果允许正负号，如何扩展为大数加减法？
  3. 输入长度百万位时如何降低字符串拼接开销？
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

#### Q099｜最短子数组长度满足和 ≥ K

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 在数组中找和至少为 K 的最短非空连续子数组，先明确所有数字是否为正。
- **输入输出 / 约束：** `nums,K`，返回最短长度或 -1。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 全正数组为何可用滑动窗口？
  2. 若允许负数，为什么应改用前缀和 + 单调队列？
  3. 若 target=0 / 数组含 0，非空约束怎样处理？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q100｜LC 1219 黄金矿工

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-05（帖内记录）
- **题目：** 网格中每一格有黄金数，只能四方向移动且不能重复走格，找最大可采总量。
- **输入输出 / 约束：** `grid:[M,N]`，输出最大值。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么必须从每一个非零格出发 DFS？
  2. 回溯时要恢复现场，如何避免修改输入？
  3. 理论最坏复杂度为什么可能指数增长？
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

#### Q101｜LC 380 Insert Delete GetRandom O(1)

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 设计 `insert/remove/getRandom` 均摊 O(1) 的集合。
- **输入输出 / 约束：** 类接口，取随机元素应均匀分布。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 删除元素时为何用数组末元素覆盖空洞？
  2. 如何同步 hash map 中被移动元素的新索引？
  3. 如果要求可重复元素（LC381）怎么改？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q102｜LC 5 最长回文子串

- **优先级：** P1 ｜ **证据：** E ｜ **年月：** 不适用（扩展练习）
- **题目：** 返回字符串中最长连续回文子串；先实现中心扩展，再讨论线性时间优化。
- **输入输出 / 约束：** 字符串，返回子串。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 奇数与偶数长度如何分别扩展中心？
  2. Manacher 的镜像半径和右边界如何维护？
  3. 当最长长度相同时选最左还是最右？
- **来源：** 根据已出现题型设计的补充练习；不代表确认考过

#### Q103｜LC 199 二叉树左视图

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-11（发布日）
- **题目：** 输入二叉树，返回从左侧看到的每层第一个节点值；如按 ACM 模式需从序列建立树。
- **输入输出 / 约束：** 树或层序数组，输出左视图。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. BFS 每层第一个节点如何确定？
  2. DFS 用深度计数时先走左子树的重要性是什么？
  3. 如何从带空标记的层序输入正确建树？
- **来源：** [牛客：字节推荐算法 AUC + 二叉树左视图 + LC331](https://www.nowcoder.com/feed/main/detail/9b2f4d320a4940aa9cbb18c544f8c263)

#### Q104｜LC 331 验证二叉树前序序列化

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2025-11（发布日）
- **题目：** 输入一个使用 `#` 表示空节点的前序序列字符串，判断其能否对应一棵合法二叉树。
- **输入输出 / 约束：** 字符串如 `9,3,4,#,#,1,#,#,2,#,6,#,#`，输出 bool。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 槽位数为什么从 1 开始，非空节点净增 1、空节点减 1？
  2. 为何中途槽位变 0 之后不能还有 token？
  3. 能否 O(1) 额外空间扫描？
- **来源：** [牛客：字节推荐算法 AUC + 二叉树左视图 + LC331](https://www.nowcoder.com/feed/main/detail/9b2f4d320a4940aa9cbb18c544f8c263)

#### Q105｜LC 239 滑动窗口最大值

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 对数组每个长度为 K 的滑动窗口输出最大值，要求 O(N)。
- **输入输出 / 约束：** `nums,k`，返回长度 N-K+1 的列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 单调双端队列存索引还是存值？
  2. 同值元素入队时用 `<` 还是 `<=` 的区别？
  3. 窗口左端越界时何时弹队首？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

#### Q106｜合并 K 个有序数组：手写堆

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 给出 K 个分别升序排列的数组，不直接用 heapq，自己实现最小堆进行 K 路归并。
- **输入输出 / 约束：** `arrays:List[List[int]]`，返回升序数组。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 堆节点需要保存值、源数组 id、数组内位置中的哪些信息？
  2. 总体复杂度为何是 O(NlogK)？
  3. 各数组为空或元素相同如何处理？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

#### Q107｜LC 53 最大子数组和

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2024-09（发布日）
- **题目：** 求数组中和最大的非空连续子数组的和。
- **输入输出 / 约束：** `nums`，输出整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Kadane 的状态为何取 `max(x,prev+x)`？
  2. 全负数时为什么不能把答案初始化为 0？
  3. 如果要同时返回区间起止下标怎么做？
- **来源：** [牛客：字节 TT/商业化 DIN、AUC、BCE、InfoNCE、梯度下降](https://www.nowcoder.com/discuss/667864275997687808)

#### Q108｜LC 136 / 260：一个/两个只出现一次的数

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（面试日；11月发布）
- **题目：** 在其余数字都恰好出现两次的条件下，先找单个唯一数字，再扩展为找到两个唯一数字。
- **输入输出 / 约束：** 整数数组，返回唯一数或二元组。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么 XOR 可以消除所有成对数字？
  2. 两个唯一数如何用 XOR 的最低位 1 分组？
  3. 若其他数出现三次而非两次，如何修改？
- **来源：** [牛客：字节大模型算法三面 BCE/VAE/积木堆叠](https://www.nowcoder.com/discuss/946798743163088896)

#### Q109｜括号生成 LC 22

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（面试时间）
- **题目：** 生成 N 对圆括号的所有合法组合。
- **输入输出 / 约束：** 整数 n，返回字符串列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何任何前缀都必须满足 `right<=left`？
  2. 如何用回溯剪枝而不是生成后过滤？
  3. 如何估算返回结果的数量与复杂度（Catalan）？
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

#### Q110｜LC 8 atoi 字符串转整数

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（面试时间）
- **题目：** 模拟 `atoi`：忽略前置空格、解析可选符号、消费数字、处理 32 位溢出。
- **输入输出 / 约束：** 字符串，输出 int。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何在乘 10 加当前数字前检查溢出？
  2. `+/-` 后面没有数字如何返回？
  3. 字符串以字符开始、前导零、多重符号怎么处理？
- **来源：** [牛客：字节算法岗岛屿数量等题](https://www.nowcoder.com/discuss/922309082860507136)

#### Q111｜LC 215 第 K 大元素

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（帖子发布时间）
- **题目：** 给定无序整数数组，找排序后第 K 大的元素（不是第 K 个互异元素）。
- **输入输出 / 约束：** `nums,k`，返回整数。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Quickselect 平均和最坏复杂度是多少？
  2. 重复元素多时使用三路划分有何优势？
  3. 和大小为 K 的最小堆方案相比哪个适合流式数据？
- **来源：** [1Point3Acres：TikTok Search 手写 MHA / Kth Largest](https://www.1point3acres.com/interview/thread/1192875)

#### Q112｜LC 206 / 92 链表反转及区间反转

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-06（文章更新日，非面试日）
- **题目：** 先反转整个单链表，再实现只反转第 L 至 R 个节点的子链表。
- **输入输出 / 约束：** `head,left,right`，返回新头。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. Dummy Head 为什么能简化 L=1？
  2. 迭代时怎样防止丢失未处理链表？
  3. 递归反转的空间复杂度为何不是 O(1)？
- **来源：** [IGotAnOffer：TikTok MLE 编程题指南](https://igotanoffer.com/en/advice/tiktok-machine-learning-engineer-interview)

#### Q113｜埃氏筛 Prime Sieve

- **优先级：** P2 ｜ **证据：** B ｜ **年月：** 2026-06（文章更新日，非面试日）
- **题目：** 返回不超过 N 的所有质数或质数数量，使用埃拉托斯特尼筛法。
- **输入输出 / 约束：** 整数 N，返回列表/数量。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何标记从 `p*p` 开始？
  2. 时间复杂度为什么是 O(NloglogN)？
  3. N 很大内存不足时如何使用分段筛？
- **来源：** [IGotAnOffer：TikTok MLE 编程题指南](https://igotanoffer.com/en/advice/tiktok-machine-learning-engineer-interview)

#### Q114｜Merge Intervals 合并区间

- **优先级：** P1 ｜ **证据：** B ｜ **年月：** 2026-11（整理发布；样本覆盖 2026-08 至 2026-11）
- **题目：** 合并所有重叠的区间，明确首尾相等是否视为重叠。
- **输入输出 / 约束：** `intervals:[[start,end]]`，返回合并结果。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为何要按起点排序？
  2. 闭区间与半开区间的连接规则有什么区别？
  3. 在线不断插入区间时如何避免每次全量排序？
- **来源：** [一亩三分地 2026 Q4 趋势汇总](https://www.1point3acres.com/interview/guides/tiktok-interview-trend-2026q4)

#### Q115｜两个升序数组合并成降序数组（O(n+m)）

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-10（面试日期）
- **题目：** 两个独立升序数组，不调用排序函数，线性时间输出两数组元素合并后的降序结果。
- **输入输出 / 约束：** 输入 `a:[N]`、`b:[M]`；输出长度 N+M 的降序列表。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么从两个数组尾部开始取较大值，能保持 O(N+M)？
  2. 如果要求 O(1) 额外空间原地修改，输入可变性约束要怎么改？
  3. 输入数组有重复值时，合并过程怎样保证重复保留？
- **题面/来源说明：** 来源同一篇面经中 2026-10-03 的另一场，和 2026-09 公式题不是同一面。
- **来源：** [牛客：字节大模型算法岗 PPO/DPO/GRPO、大数相加、双栈队列](https://www.nowcoder.com/discuss/940390216378888192)

#### Q116｜最长公共子数组 LC 718（滚动空间优化）

- **优先级：** P1 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-03（帖子发布时间）
- **题目：** 给两个整数数组，求最长公共连续子数组的长度，并把二维 DP 空间优化为 O(min(n,m))。
- **输入输出 / 约束：** `nums1,nums2`，返回最长连续重复片段长度。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 为什么二维 DP 的状态只依赖左上方？
  2. 用一维滚动数组时必须从右往左还是左往右更新？
  3. “公共子数组”和“不要求连续的最长公共子序列”有什么不同？
- **来源：** [牛客：字节推荐实习 MHA、AUC 变体](https://www.nowcoder.com/users/615039244)

#### Q117｜交替抛硬币：先获得正面者获胜的概率

- **优先级：** P2 ｜ **证据：** A（公开面经） ｜ **年月：** 2026-04（帖子发布时间）
- **题目：** A、B 轮流抛公平或带偏概率的硬币，A 先手；谁先抛出正面谁获胜，计算 A 的获胜概率。
- **输入输出 / 约束：** 输入 A 的正面率 p、B 的正面率 q，返回 A 赢的概率。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何把“双方一轮都失败后重新开始”的概率写成递归方程？
  2. 公平硬币时 A 的获胜概率为何不是 1/2？
  3. 若允许无限轮，何时可以用等比数列求和？
- **来源：** [牛客：字节内容消费团队二面 K 组翻转 / 抛硬币](https://www.nowcoder.com/discuss/878370699188916224)

## 07｜AI-Assisted Coding / 工程现场题

#### Q118｜用 AI 工具实现 LRU 并审查代码

- **优先级：** P1 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 允许使用 AI 助手生成 LRU 类，但要求你能审查数据结构、直接运行测试并解释每一段代码。
- **输入输出 / 约束：** 交付 `lru.py` + `test_lru.py` 或项目内同等文件。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. AI 生成了线性查找的“假 O(1)”实现，怎样识别？
  2. 测试里怎样覆盖重复 put、容量 1、频繁 get 和淘汰顺序？
  3. 若 AI 给出复杂锁方案，如何简化并讲清 tradeoff？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

#### Q119｜用 AI 工具制作自然语言转 SQL 小工具

- **优先级：** P1 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 给定 Schema 和用户问题，用 AI 生成一个只读 Text-to-SQL 工具，包含 Schema 注入、生成、校验和结果展示。
- **输入输出 / 约束：** 交付最小项目：CLI/Web 接口 + 示例 SQLite 数据库 + 测试。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何阻止 DROP/DELETE/UPDATE 及通过多语句绕过校验？
  2. 大模型引用了不存在的表/列，怎样结合 Schema 自动验证？
  3. 生成的 SQL 很慢时怎样限制执行时间、行数和扫描范围？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

#### Q120｜Claude Code 在现有仓库新增个人站点

- **优先级：** P1 ｜ **证据：** U ｜ **年月：** 年月未提供
- **题目：** 共享屏幕进入已有项目，让 Claude Code 帮忙新增个人站点页面和导航；要求增量修改并成功构建。
- **输入输出 / 约束：** 交付可运行页面及修改说明，不新造不必要的脚手架。
- **Follow-up（原帖明确项另行标注，其余为模拟）：**
  1. 如何先确认框架、路由、包管理器与现有设计系统？
  2. AI 改坏无关文件时如何回滚最小变更？
  3. 你如何独立验证 lint、build、关键页面与移动端布局？
- **来源：** 用户在本对话提供的题目线索（未核实原始来源）

---

## 附录｜标准模拟面试检查表

- **代码正确性：** 写清接口 / Tensor Shape；添加至少一个正常用例、一个极端边界、一个错误输入；说明所有异常情形的约定。
- **数值稳定：** Softmax 减最大值、BCE logits 稳定式、KL 的零概率、半精度 epsilon、求和归一化分母。
- **深度学习：** Mask polarity、Label Shift 是否发生在模型内部、batch 有效监督位置、梯度是否能反向传播。
- **复杂度：** 时间复杂度、额外空间复杂度、对序列长度、样本数、特征维度的缩放影响。
- **可测试性：** 固定随机种子、PyTorch / NumPy 参考实现对齐、`assert shape`、finite-difference gradient check。
- **AI 工具环节：** 先阅读仓库、限制修改范围、主动跑测试、审查生成代码并说明缺陷。

## 附录｜原始信息与可信度说明

1. `A` 只代表“有对应的公开面经/个人记录”，并不代表 TikTok 官方题库认可或题面 100% 原文。
2. `B` 的公司归属通常来自站点题库或整理文章，可能将近似题、其他公司题、模拟题混入。
3. `E` 为刻意添加的练习能力点，和公开真题严格区分。
4. `U` 为你自己提供的题目，按原始线索整理；缺少年月时绝不捏造。
5. 仅明确标为 `[原始追问]` 的条目可追溯到相应公开记录；没有该标记的 Follow-up 均为练习补充，不能当作面试官原话。

_结束_

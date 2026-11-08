阅读排序：
| 优先级                         | 题目                                                         | 我的建议                         |
| ------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| **S：必须闭卷写**              | Softmax、Cross-Entropy、Linear Layer、LayerNorm、Softmax Attention、MHA、Causal Attention、KV Cache、RoPE | 核心中的核心                     |
| **S−：现代 LLM 高频**          | RMSNorm、GQA、LoRA、Top-k/Top-p                              | 2026 面试价值很高                |

> **合并说明**：本文档已并入《补充文档-LLM面试手撕代码补充题单》。每个级别章节内部先列主文档原有题目（第一部分），再列补充题单新增题目（第二部分）；章节顺序与原主文档一致。

补充题单部分的编写原则：实现目标是“面试现场可手推、可手写、可解释”，优先保留核心计算图、shape、mask、索引和梯度语义；不追求 CUDA kernel、分布式容错、量化库兼容等工业细节。所有 NumPy 版本与 PyTorch 版本保持同一数学对象和计算逻辑。本版进一步与公开 LLM 手撕题库、PyTorch 官方实现说明及相关论文逐题交叉核对：只有在不损失数学语义与正确性的前提下，才吸收更短、更适合白板手写的写法。

补充题单新增题目一览：
| 级别 | 补充题单新增题目 |
| ---- | ---------------- |
| **S：必须闭卷写** | S1. AdamW Optimizer（Adam with Decoupled Weight Decay）、S2. Causal Language Modeling / Pretraining Loss（Next-Token Loss）、S3. Manual Backpropagation（Two-Layer MLP）、S4. Decoder-Only Transformer Block（Generic Pre-Norm） |
| **S−：现代 LLM 高频** | S−1. Multi-Query Attention（MQA）、S−2. Multi-Head Latent Attention（MLA，面试简化版）、S−3. RoPE with Position Offset / KV-Cache RoPE、S−4. Generalized Advantage Estimation（GAE）、S−5. SFT Masked Loss（Assistant-Only Loss）、S−6. Modern LLaMA-Style Block（RMSNorm + RoPE + GQA + SwiGLU） |

### S：必须闭卷写

> 本章分两部分：**第一部分**为主文档原有题目，**第二部分**为《补充文档-LLM面试手撕代码补充题单》新增题目。

#### 1️⃣ 经典 ML Coding 面试题：Implement K-Means from Scratch

**题目要求：** 实现 K-Means 聚类算法，不使用 `sklearn.cluster.KMeans`。

输入包括：

- `points`：数据点，形状 $(N,D)$。
- `k`：聚类数量。
- `initial_centroids`：给定的初始聚类中心，形状 $(K,D)$。
- `max_iterations`：最大迭代次数。

输出最终的 $K$ 个聚类中心，每个坐标保留四位小数。Interview Query 的同类变体则要求返回每个样本所属的聚类标签。\[[Deep-ML](https://www.deep-ml.com/problems/17?utm_source=chatgpt.com); [Interview Query \| k-Means from Scratch](https://www.interviewquery.com/questions/k-means-from-scratch?utm_source=chatgpt.com)\]

原题示例

输入点

最终聚类中心

样本分别聚集在 $x=1$ 与 $x=10$ 附近，最终中心为 **$1, 2)** 和 **\(10, 2)**。

2️⃣ Global Equations：核心数学公式

K-Means 的目标是最小化簇内平方误差（Within-Cluster Sum of Squares）：

$$
\boxed{
J=\sum_{i=1}^{N}\|x_i-\mu_{z_i}\|_2^2
}
$$

其中 $x_i$ 是第 $i$ 个样本，$\mu_j$ 是第 $j$ 个中心，$z_i$ 是样本所属的聚类编号。

Step 1：Assignment（分配样本）

计算样本 $x_i$ 到中心 $\mu_j$ 的平方欧氏距离：

$$
D_{ij}=\sum_{d=1}^{D}(x_{id}-\mu_{jd})^2
$$

然后选择距离最小的中心：

$$
\boxed{z_i=\arg\min_j D_{ij}}
$$

使用平方距离即可，无须计算平方根，因为不会改变最近中心的选择。

Step 2：Update（更新聚类中心）

计算第 $j$ 个簇中所有样本的均值：

$$
\boxed{
\mu_j^{\text{new}}
=
\frac{
\sum_{i=1}^{N}\mathbf{1}[z_i=j]x_i
}{
\sum_{i=1}^{N}\mathbf{1}[z_i=j]
}
}
$$

如果第 $j$ 个簇为空，分母为零，则保留原来的中心。

不断执行 Assignment → Update，直到中心不再明显变化，或者达到最大迭代次数。

3️⃣ Implementation Blocks

（A）PyTorch 版本（标准实现）

```python
import torch

def k_means_clustering(
    points, k, initial_centroids, max_iterations
):
    # X: (N, D)，N 个样本，每个样本 D 维
    X = torch.tensor(points, dtype=torch.float64)
    # centroids: (K, D)，当前 K 个聚类中心
    centroids = torch.tensor(
        initial_centroids, dtype=torch.float64
    )
    # 最多执行 max_iterations 次聚类更新
    for _ in range(max_iterations):
        # 广播 (N,1,D)-(1,K,D) -> (N,K,D)
        # 沿特征维求和，得到距离矩阵 (N,K)
        distances = (
            (X[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(dim=-1)
        # labels: (N,)，每个点最近的中心编号
        labels = distances.argmin(dim=1)
        # 复制中心，空簇可以保留原来的位置
        new_centroids = centroids.clone()
        # 逐个更新 K 个聚类中心
        for j in range(k):
            # mask: (N,)，属于第 j 簇的样本位置
            mask = labels == j
            # 只有非空簇才更新中心
            if mask.any():
                # X[mask]: (N_j,D)，均值为 (D,)
                new_centroids[j] = X[mask].mean(dim=0)
        # 各坐标变化均不超过 1e-6 时停止
        if torch.allclose(
            centroids, new_centroids,
            rtol=0, atol=1e-6
        ):
            centroids = new_centroids
            break
        # 更新中心，进入下一轮迭代
        centroids = new_centroids
    # 转成坐标元组，保留四位小数
    return [
        tuple(round(v, 4) for v in row)
        for row in centroids.tolist()
    ]
```

（B）NumPy 版本（苛刻面试版，推荐背诵）

与 PyTorch 使用完全相同的计算图，不依赖聚类库。

```python
import numpy as np

def k_means_clustering_numpy(
    points, k, initial_centroids, max_iterations
):
    # X: (N,D)，所有输入数据点
    X = np.asarray(points, dtype=float)
    # centroids: (K,D)，当前聚类中心
    centroids = np.asarray(
        initial_centroids, dtype=float
    ).copy()
    # 最多执行 max_iterations 次迭代
    for _ in range(max_iterations):
        # 广播 (N,1,D)-(1,K,D) -> (N,K,D)
        # 平方求和得到距离矩阵 (N,K)
        distances = (
            (X[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(axis=-1)
        # labels: (N,)，选择最近中心的编号
        labels = distances.argmin(axis=1)
        # 先复制旧中心，避免空簇变成 NaN
        new_centroids = centroids.copy()
        # 依次计算每个簇的新中心
        for j in range(k):
            # mask: (N,)，选择属于第 j 簇的样本
            mask = labels == j
            # 空簇直接保留之前的中心
            if mask.any():
                # X[mask]: (N_j,D)，均值为 (D,)
                new_centroids[j] = X[mask].mean(axis=0)
        # 检查中心是否已经收敛
        if np.allclose(
            centroids, new_centroids,
            rtol=0, atol=1e-6
        ):
            centroids = new_centroids
            break
        # 用新中心开始下一轮聚类
        centroids = new_centroids
    # 转成元组列表，每个坐标保留四位小数
    return [
        tuple(round(v, 4) for v in row)
        for row in centroids.tolist()
    ]
```

最小可运行测试

```python
# Demo 1：原题样例
points = [
    (1, 2), (1, 4), (1, 0),
    (10, 2), (10, 4), (10, 0)
]
initial_centroids = [(1, 1), (10, 1)]
print(k_means_clustering(
    points, 2, initial_centroids, 10
))
print(k_means_clustering_numpy(
    points, 2, initial_centroids, 10
))
# 两者均输出 [(1.0, 2.0), (10.0, 2.0)]

# Demo 2：重复初始中心，测试空簇
points = [(0, 0), (0, 0), (10, 10)]
initial_centroids = [(0, 0), (0, 0), (10, 10)]
print(k_means_clustering_numpy(
    points, 3, initial_centroids, 10
))
# 输出 [(0.0, 0.0), (0.0, 0.0), (10.0, 10.0)]
```

上述两个测试都已在 PyTorch 和 NumPy 实现中验证。

4️⃣ Step-by-step Derivation：代码逐步推导

实现前置骨架

```python
import torch

def k_means_clustering(
    points, k, initial_centroids, max_iterations
):
    pass
```

Step 1：初始化中心

$$
X\in\mathbb{R}^{N\times D},\qquad
C^{(0)}\in\mathbb{R}^{K\times D}
$$

```python
X = torch.tensor(points, dtype=torch.float64)
centroids = torch.tensor(
    initial_centroids, dtype=torch.float64
)
```

`X` 包含所有数据点；`centroids` 保存当前中心。题目已经给定初始中心，因此不需要随机初始化。

Step 2：向量化距离计算（最重要）

$$
D_{ij}=\sum_{d=1}^{D}(X_{id}-C_{jd})^2
$$

```python
distances = (
    (X[:, None, :] - centroids[None, :, :]) ** 2
).sum(dim=-1)
labels = distances.argmin(dim=1)
```

这里考察 Broadcasting：

每个样本与每个中心按特征维计算平方距离，最后得到 N×K 距离矩阵。

具体来说：

- `X[:, None, :]`：$(N,D)\rightarrow(N,1,D)$。
- `centroids[None, :, :]`：$(K,D)\rightarrow(1,K,D)$。
- 相减后广播得到 $(N,K,D)$。
- `sum(dim=-1)`：沿最后一维求和，得到 $(N,K)$。
- `argmin(dim=1)`：每个样本取距离最小的中心索引，得到 $(N,)$。

注意 `argmin` 默认在距离相同时选最小索引。

Step 3：更新中心

$$
\mu_j^{new}=
\frac{1}{N_j}\sum_{i:z_i=j}x_i
$$

```python
new_centroids = centroids.clone()
for j in range(k):
    mask = labels == j
    if mask.any():
        new_centroids[j] = X[mask].mean(dim=0)
```

`mask` 是形状 $(N,)$ 的布尔数组。`X[mask]` 只保留第 $j$ 簇的样本，形状为 $(N_j,D)$，沿 `dim=0` 求均值得到 $(D,)$，即新的中心。

**为什么要 `if mask.any()`？** 如果某个簇没有样本，直接计算均值会产生 `NaN`。因此只更新非空簇，空簇保留旧中心。

Step 4：收敛判断

$$
\boxed{
\max_{j,d}
|C_{jd}^{new}-C_{jd}^{old}|
\leq 10^{-6}
}
$$

```python
if torch.allclose(
    centroids, new_centroids,
    rtol=0, atol=1e-6
):
    centroids = new_centroids
    break
centroids = new_centroids
```

所有中心坐标变化都足够小时提前停止。若没有提前收敛，最多执行 `max_iterations` 次。

Step 5：返回结果

$$
C_{\text{output}}
=\operatorname{round}(C_{\text{final}},4)
$$

```python
return [
    tuple(round(v, 4) for v in row)
    for row in centroids.tolist()
]
```

`tolist()` 将 $(K,D)$ Tensor 转成 Python 嵌套列表；逐行转换成元组后，输出就是题目要求的 `list[tuple]`。

5️⃣ 复杂度与面试高频追问

设 $N$ 为样本数，$K$ 为聚类数量，$D$ 为特征维度，$T$ 为实际迭代次数。

- **时间复杂度：** $O(TNKD)$。每次计算距离和更新中心都需要处理数据。
- **空间复杂度：** $O(NKD)$。当前实现通过 Broadcasting 创建了 $(N,K,D)$ 中间张量。

常见的进阶追问：

| 面试追问                       | 核心回答                                                     |
| ------------------------------ | ------------------------------------------------------------ |
| 为什么不用 `sqrt`？            | 平方距离与欧氏距离有相同的最小值位置                         |
| 如何处理 Empty Cluster？       | 保留旧中心，或者重新初始化                                   |
| K-Means 一定收敛到全局最优吗？ | 不一定，通常只能保证到局部最优                               |
| 怎么改进初始化？               | K-Means++                                                    |
| 数据特别大怎么办？             | Mini-Batch K-Means，或者分批计算距离                         |
| 如何优化 Broadcasting 内存？   | 利用 $\\|x-c\\|^2=\\|x\\|^2+\\|c\\|^2-2x^\top c$，将中间内存降至 $O(NK)$ |

**面试最重要的三行 NumPy 代码：**

```python
# 1. 计算所有样本到所有中心的平方距离
distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
# 2. 分配每个样本的聚类标签
labels = distances.argmin(axis=1)
# 3. 更新第 j 个非空簇的中心
centroids[j] = X[labels == j].mean(axis=0)
```

掌握这三行，再补充迭代、收敛和空簇判断，就覆盖了经典 K-Means ML Coding 的核心实现。



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
    x = x - x.max(dim=dim, keepdim=True).values # ⚠️ 减 max 防 exp 上溢  ‼️ values‼️
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


灵光一闪，突然觉得MHA 最容易忘的这一串操作：
view/reshape → transpose → attention → transpose → view/reshape
	
刚好可以压缩成：
VTATV
	
V = View / reshape
T = Transpose
A = Attention
	
重新断了个句那就是：
V | TAT | V
	
TAT，不就是一个哭脸吗！😭
	
两边两个 V 再换成两只手：
✋😭🤚
真的很像一个被 MHA 逼到哭着举手投降的人（正是我自己
	
对应的 shape flow就是：
	
(batch_size, seq_len, d_model)
	
→ view/reshape 把d_model展开
	
(batch_size, seq_len, num_heads, head_dim)
	
→ transpose 把num_heads取出来
	
(batch_size, num_heads, seq_len, head_dim)
	
→ attention
	
(batch_size, num_heads, seq_len, head_dim)
	
→ transpose 把num_heads放回去
	
(batch_size, seq_len, num_heads, head_dim)
	
→ view/reshape 把d_model还原，和input一致
(batch_size, seq_len, d_model)
	
而Attention 本身是好记的：
QKᵀ / √head_dim → mask → softmax over keys → attention @ V
	
于是现在，我脑子里的 MHA 已经压缩成：
QKV → ✋😭🤚 → out_proj
	
下次面试再让我手写 MHA，希望脑子空白的那一秒里，先出现这个投降哭泣小人：

```py
import math
import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """
    普通通用 MHA：
    forward(q_source, k_source, v_source, attn_mask=None)
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
            # ‼️ masked_fill() 是 PyTorch 中的条件填充操作 ；True 的位置被置为 -inf，softmax 后概率接近 0
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
      
      
output = attention(
    q_source=x,
    k_source=x,
    v_source=x
)
```
#### 👀 DecoderSelfAttentionWithKVCache

核心差别是：**谁当 Q/K/V，以及 cache 会不会变长。**

`DecoderSelfAttentionWithKVCache` 是 decoder **自注意力**：Q、K、V 都来自 decoder 自己的 token。每 decode 一步，只对**新 token** 算 K/V，再 `cat` 到历史 cache 上。cache 是**增长型**，长度从 `S_past` 变成 `S_past + S_new`。因为 Q 和 K 是同一条序列，还要做 **causal mask**（一次喂多个新 token 时，绝对位置靠 `S_past` 偏移，避免看到未来 token）。

`CrossAttentionWithKVCache` 是 encoder-decoder **交叉注意力**：Q 来自 decoder，K/V 来自 **encoder_output**。encoder 在整个 decoding 过程中不变，所以 encoder 的 K/V **算一次就原样复用**。cache 是**固定型**，长度始终是 `S_enc`，不会 `cat`。mask 通常是 **encoder padding mask**，没有 causal——decoder 可以看到全部有效 encoder token。

**在标准 encoder–decoder 模型做自回归生成时，真正长期保存的 KV cache 通常有两套，而且都服务于 decoder。**

````
Encoder
输入 source tokens
   ↓
Encoder self-attention
   ↓
encoder_output
   ↓
生成固定的 cross-attention K/V
   ↓
─────────────────────────────
              Decoder
                 │
        ┌────────┴────────┐
        ↓                 ↓
decoder self KV      encoder cross KV
不断增长              固定不变
````



|          | Decoder Self + KV Cache        | Cross + KV Cache                  |
| -------- | ------------------------------ | --------------------------------- |
| Q 来源   | decoder 新 token `x_new`       | decoder `decoder_x`（每步都重算） |
| K/V 来源 | 同一条 decoder 序列            | encoder 序列                      |
| cache    | `cat(past, new)`，变长         | 首次投影后原样复用，定长          |
| mask     | causal                         | encoder padding                   |
| 典型场景 | GPT / decoder-only 增量 decode | 翻译、encoder-decoder 模型        |

文档里的 demo 也把这件事写死了：self-attn 的 cache K 从 4 长到 5；cross-attn 的 cache K 一直是 `(B, H, S_enc, d_head)`。

核心逻辑

```py
q = W_q(x_new)
k = W_k(x_new)
v = W_v(x_new)

if cache:
    k = cat(past_k, k)
    v = cat(past_v, v)

scores = q @ k.T
scores = causal_mask(scores)
weights = softmax(scores)
out = weights @ v

return out, (k, v)
```



```py
class DecoderSelfAttentionWithKVCache(nn.Module):
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
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        B, H, S, d_head = x.shape

        return x.transpose(1, 2).contiguous().view(
            B, S, H * d_head
        )

    def forward(
        self,
        x_new: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, S_new, _ = x_new.shape

        q_new = self._split_heads(self.W_q(x_new))
        k_new = self._split_heads(self.W_k(x_new))
        v_new = self._split_heads(self.W_v(x_new))

        if cache is None:
            k_total = k_new
            v_total = v_new
        else:
            past_k, past_v = cache
            k_total = torch.cat([past_k, k_new], dim=2)
            v_total = torch.cat([past_v, v_new], dim=2)

        new_cache = (k_total, v_total)

        S_total = k_total.shape[2]
        S_past = S_total - S_new

        scores = torch.matmul(
            q_new,
            k_total.transpose(-2, -1),
        ) / math.sqrt(self.d_head)

        if S_new > 1:
            q_pos = torch.arange(
                S_past,
                S_total,
                device=x_new.device,
            ).unsqueeze(1)

            k_pos = torch.arange(
                S_total,
                device=x_new.device,
            ).unsqueeze(0)

            mask = k_pos > q_pos

            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v_total)

        out = self._merge_heads(attn)

        return self.W_o(out), new_cache
```



```py
class DecoderSelfAttentionWithKVCache(nn.Module):
    """
    Decoder Self-Attention + KV cache。

    【相对上一段 MultiHeadAttention 的核心差别】
    1. API：
       - 普通 MHA: forward(q_source, k_source, v_source, attn_mask)
         每次传入完整序列，Q/K/V 都整段重算。
       - 本类:     forward(x_new, cache=None)
         只传入「本步新 token」；历史 K/V 从 cache 读，不再重算。
    2. 返回值：
       - 普通 MHA 只返回 out
       - 本类返回 (out, new_cache)，供下一步 decode 复用
    3. 注意力时间维：
       - 普通 MHA: scores 是 (B,H,S_q,S_k)，S_q 与 S_k 都来自当次输入
       - 本类:     scores 是 (B,H,S_new,S_total)
         Q 只有新 token；K/V 是 past + new 拼起来的完整历史
    4. Mask：
       - 普通 MHA: 外部传入 attn_mask（可选）
       - 本类:     内建 causal mask，且要按 S_past 做绝对位置偏移
                   （decode 单 token 时 S_new==1，可省略）

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
        # 与普通 MHA 完全相同
        """
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, H, S, d_head)
        return: (B, S, D)
        # 与普通 MHA 完全相同
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
        - 行为接近「对整段 x 做一遍 self-attn」，但返回值多了 cache

        增量 decode:
        - S_new 通常等于 1
        - cache 保存过去所有 token 的 K/V
        - 【相对普通 MHA】不必再把历史 token 整段喂进来重算 K/V
        """

        B, S_new, _ = x_new.shape

        # 【相对普通 MHA】
        # 普通 MHA: q/k/v 分别由 q_source / k_source / v_source 投影，
        #           且通常对「整段序列」做投影。
        # 本类:     self-attn，Q/K/V 同源；但只对「本步新 token」投影。
        # q_new: (B, S_new, D) -> (B, H, S_new, d_head)
        q_new = self._split_heads(self.W_q(x_new))

        # k_new / v_new 也只算新 token；历史部分后面从 cache 拼回来
        # k_new: (B, S_new, D) -> (B, H, S_new, d_head)
        k_new = self._split_heads(self.W_k(x_new))

        # v_new: (B, S_new, D) -> (B, H, S_new, d_head)
        v_new = self._split_heads(self.W_v(x_new))

        # ========== 相对普通 MHA 的独特步骤：拼 / 建 KV cache ==========
        if cache is None:
            # prefill / 第一步：没有历史，total 就是当前 new
            k_total = k_new
            v_total = v_new
        else:
            past_k, past_v = cache

            # past_k: (B, H, S_past, d_head)
            # k_new:  (B, H, S_new,  d_head)
            # cat dim=2 = sequence 维
            # k_total: (B, H, S_past + S_new, d_head)
            k_total = torch.cat([past_k, k_new], dim=2)
            v_total = torch.cat([past_v, v_new], dim=2)

        # 更新 cache：下一步 decode 继续往后拼
        new_cache = (k_total, v_total)

        S_total = k_total.shape[2]
        S_past = S_total - S_new
        
        scores = torch.matmul(
            q_new, k_total.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if S_new > 1:
            # 一次输入多个新 token 时，需要 causal mask，
            # 防止当前 chunk 中前面的 token 看到后面的未来 token。
            #
            # 例子：
            # past = [A, B, C]
            # new  = [D, E]
            # D 的绝对位置是 3，只能看 A B C D，不能看 E
            # E 的绝对位置是 4，可以看 A B C D E
            #
            # 因此：
            # - query 的位置范围是 [S_past, ..., S_total - 1]
            # - key 的位置范围是   [0, ..., S_total - 1]
            # - 如果 key_pos > query_pos，说明这个 key 在未来，需要屏蔽
            #
            # 最终 mask shape 为 (S_new, S_total)
            # True  表示该位置禁止 attention
            # False 表示该位置可以 att ention
            q_pos = torch.arange(
                S_past, S_total, device=x_new.device
            ).unsqueeze(1) # (S_new, 1)
             """
             shape: (2, 1)
              [[3], 
              [4]]
             """
            k_pos = torch.arange(
                S_total, device=x_new.device
            ).unsqueeze(0) # (1, S_total) shape: (1, 5) [[0, 1, 2, 3, 4]]
             """
              第一行： [0,1,2,3,4] > 3
              第二行： [0,1,2,3,4] > 4
             """
            mask = k_pos > q_pos # (S_new, S_total)

            # (S_new, S_total) -> (1,1,S_new,S_total) 以匹配 scores
            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        # ⚠️ S_new == 1 时，不需要 causal mask。
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v_total)
        # out: (B, S_new, D) 
        out = self._merge_heads(attn)

        # 【相对普通 MHA】多返回 new_cache
        return self.W_o(out), new_cache
```
#### 👀 CrossAttentionWithKVCache

这个 `CrossAttentionWithKVCache` 的核心逻辑可以压缩成下面几行：

```python
q = W_q(decoder_x)

if cache is None:
    k = W_k(encoder_output)
    v = W_v(encoder_output)
else:
    k, v = cache

scores = q @ k.T

if encoder_mask is not None:
    scores = mask(scores, encoder_mask)

weights = softmax(scores)
out = weights @ v
return out, (k, v)
```

这就是它的全部灵魂。

```py
class CrossAttentionWithKVCache(nn.Module):
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
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
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
        B, S_q, _ = decoder_x.shape

        q = self._split_heads(self.W_q(decoder_x))

        if cache is None:
            enc_k = self._split_heads(self.W_k(encoder_output))
            enc_v = self._split_heads(self.W_v(encoder_output))
        else:
            enc_k, enc_v = cache

        new_cache = (enc_k, enc_v)

        scores = torch.matmul(
            q, enc_k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if encoder_mask is not None:
            scores = scores.masked_fill(
                encoder_mask, float("-inf")
            )

        weights = torch.softmax(scores, dim=-1)

        attn = torch.matmul(weights, enc_v)

        out = self._merge_heads(attn)
        return self.W_o(out), new_cache


if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, H = 2, 32, 4
    S_q, S_k, S_new, S_enc = 3, 5, 1, 6

    mha = MultiHeadAttention(D, H)
    q_src = torch.randn(B, S_q, D)
    k_src = torch.randn(B, S_k, D)
    v_src = torch.randn(B, S_k, D)
    out = mha(q_src, k_src, v_src)
    print("MHA:", out.shape)

    self_attn = DecoderSelfAttentionWithKVCache(D, H)
    x1 = torch.randn(B, 4, D)
    x2 = torch.randn(B, S_new, D)

    out1, cache = self_attn(x1, cache=None)
    out2, cache = self_attn(x2, cache=cache)

    print("Self out1:", out1.shape)
    print("Self out2:", out2.shape)
    print("Self cache K:", cache[0].shape)

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

    print("Cross out:", cross_out1.shape)
    print("Cross cache K:", cross_cache[0].shape)
```

```py

class CrossAttentionWithKVCache(nn.Module):
    """
    Cross-Attention + KV cache。

    【相对上一段 MultiHeadAttention 的核心差别】
    1. API：
       - 普通 MHA: forward(q_source, k_source, v_source, attn_mask)
         Q/K/V 三个源每次都传入并整段投影。
       - 本类:     forward(decoder_x, encoder_output, cache=None, encoder_mask=None)
         语义写死为 cross-attn：Q←decoder，K/V←encoder。
    2. Cache 语义（与 DecoderSelfAttentionWithKVCache 也不同！）：
       - Decoder Self + cache: K/V 随 decode 不断 cat 变长（增长型 cache）
       - Cross + cache:        encoder K/V 算一次后原样复用（固定型 cache）
         因为 encoder_output 在整个 decoding 过程中不变，不需要逐步追加。
    3. Mask：
       - 普通 MHA: 通用 attn_mask
       - 本类:     通常是 encoder padding mask；没有 causal mask
                   （decoder 可以看全部 encoder token）
    4. 返回值：同样是 (out, new_cache)，但 cache 长度始终 = S_enc，不会变长。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # 与普通 MHA / DecoderSelf+KVCache 相同的四套线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, S, D)
        return: (B, H, S, d_head)
        # 与普通 MHA 完全相同
        """
        B, S, D = x.shape

        return x.view(
            B, S, self.num_heads, self.d_head
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, H, S, d_head)
        return: (B, S, D)
        # 与普通 MHA 完全相同
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

        # 【相对普通 MHA】
        # 普通 MHA: q = split(W_q(q_source))，q_source 由调用方决定。
        # 本类:     Q 固定来自 decoder 当前 hidden；每一步都要重算 Q
        #           （decoder 在变；encoder 不变）。
        # q: (B, S_q, D) -> (B, H, S_q, d_head)
        q = self._split_heads(self.W_q(decoder_x))

        # ========== 相对普通 MHA 的独特步骤：缓存 encoder K/V ==========
        if cache is None:
            # 第一次：从 encoder_output 投影得到 K/V 并写入 cache。
            # 【相对普通 MHA】普通 MHA 每次都会 W_k(k_source)、W_v(v_source)。
            # 【相对 DecoderSelf+KVCache】这里不算「新 token 再 cat」，
            #   而是一次性算完整 encoder 序列的 K/V。
            # enc_k: (B, H, S_enc, d_head)
            # enc_v: (B, H, S_enc, d_head)
            enc_k = self._split_heads(self.W_k(encoder_output))
            enc_v = self._split_heads(self.W_v(encoder_output))
        else:
            # 后续 decode：直接复用，不再对 encoder_output 做投影。
            # encoder_output 不随 decoder step 改变，故 cache 长度恒为 S_enc。
            # 【注意】即使调用方仍传入 encoder_output，有 cache 时也不用它算 K/V。
            enc_k, enc_v = cache

        # 【相对 DecoderSelf+KVCache】
        # Self+cache: new_cache = cat(past, new) —— 变长
        # Cross+cache: new_cache = 原样 (enc_k, enc_v) —— 定长
        new_cache = (enc_k, enc_v)

        # 【相对普通 MHA】
        # 普通 MHA: scores = q @ k^T，k 来自当次 k_source
        # 本类:     scores = q @ enc_k^T，enc_k 来自 cache（或首次投影）
        # shape: (B, H, S_q, S_enc)
        scores = torch.matmul(
            q, enc_k.transpose(-2, -1)
        ) / math.sqrt(self.d_head)

        if encoder_mask is not None:
            # 【相对普通 MHA / DecoderSelf+KVCache】
            # 这里不是 causal mask，而是 encoder padding mask：
            # decoder 可以看全部有效 encoder token，只屏蔽 pad。
            scores = scores.masked_fill(
                encoder_mask, float("-inf")
            )

        # weights: (B, H, S_q, S_enc)
        weights = torch.softmax(scores, dim=-1)

        # attn: (B, H, S_q, d_head)
        attn = torch.matmul(weights, enc_v)

        # out: (B, S_q, D)
        out = self._merge_heads(attn)

        # 【相对普通 MHA】多返回 new_cache（定长 encoder K/V）
        return self.W_o(out), new_cache


# Demo：对照三种 forward 用法
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D, H = 2, 32, 4
    S_q, S_k, S_new, S_enc = 3, 5, 1, 6

    # 1. 普通 MHA：一次传入完整 Q/K/V 源，只返回 out，无 cache
    mha = MultiHeadAttention(D, H)
    q_src = torch.randn(B, S_q, D)
    k_src = torch.randn(B, S_k, D)
    v_src = torch.randn(B, S_k, D)
    out = mha(q_src, k_src, v_src)
    print("MHA:", out.shape)  # (B, S_q, D)

    # 2. Decoder Self + KV cache：
    #    先 prefill 整段，再只喂 1 个新 token；cache K 从 4 变长到 5
    self_attn = DecoderSelfAttentionWithKVCache(D, H)
    x1 = torch.randn(B, 4, D)
    x2 = torch.randn(B, S_new, D)

    out1, cache = self_attn(x1, cache=None)      # 增长型 cache 起步
    out2, cache = self_attn(x2, cache=cache)     # cat 新 K/V，不必重算历史

    print("Self out1:", out1.shape)          # (B, 4, D)
    print("Self out2:", out2.shape)          # (B, 1, D)
    print("Self cache K:", cache[0].shape)   # (B, H, 5, d_head) ← 变长

    # 3. Cross + KV cache：
    #    第一次算 encoder K/V；第二次复用，cache 长度始终 = S_enc（定长）
    cross_attn = CrossAttentionWithKVCache(D, H)
    decoder_x = torch.randn(B, S_new, D)
    encoder_output = torch.randn(B, S_enc, D)

    cross_out1, cross_cache = cross_attn(
        decoder_x,
        encoder_output,
        cache=None,          # 首次：投影 encoder → K/V
    )

    cross_out2, cross_cache = cross_attn(
        decoder_x,
        encoder_output,
        cache=cross_cache,   # 复用：不再投影 encoder
    )

    print("Cross out:", cross_out1.shape)         # (B, 1, D)
    print("Cross cache K:", cross_cache[0].shape) # (B, H, S_enc, d_head) ← 定长
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

#### 👀 ‼️MHA + KV Cache

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

<span style="color:red">...</span>

<span style="color:red">普通 Causal Mask：`diagonal=1`。有 KV Cache：在 1 上加历史长度。</span>

<span style="color:red">**`x.shape()` ❌ → `x.shape` ✅，因为 `shape` 是属性，不是函数。**</span>

<span style="color:red">**_merge 中 transpose() 后需要 .contiguous()，才能安全地使用 .view()**</span>



MHA + KV Cache 面试错误汇总表

| 分类           | 之前的错误 / 疑问                      | 正确写法 / 理解                                          | 原因                                  |
| -------------- | -------------------------------------- | -------------------------------------------------------- | ------------------------------------- |
| 类定义         | `class attentionwithcache:`            | `class attentionwithcache(nn.Module):`                   | PyTorch 模型需要继承 `nn.Module`      |
| 初始化         | 忘记 `super().__init__()`              | `super().__init__()`                                     | 正确初始化模块                        |
| 初始化参数     | `__init__(self, x, num_head, d_model)` | `__init__(self, num_head, d_model)`                      | `x` 是 forward 的输入                 |
| Head 数量      | 没检查能否整除                         | `assert d_model % num_head == 0`                         | 保证每个 Head 维度为整数              |
| 实例属性       | `num_head`, `d_head`                   | `self.num_head`, `self.d_head`                           | 要访问 `__init__` 中保存的属性        |
| Shape API      | `x.shape()`                            | `x.shape`                                                | `shape` 是属性，不是函数              |
| Shape API      | `q.shape(2)`                           | `q.size(2)` 或 `q.shape[2]`                              | `shape` 不能像函数一样调用            |
|                |                                        |                                                          |                                       |
| Linear         | 以为必须写 `Wq * x`                    | `self.Wq(x)`                                             | `nn.Linear` 执行 $xW^T+b$             |
| Linear         | 以为 Linear 会转置输入                 | `nn.Linear(512,256)`                                     | 只将最后一维从 512 映射到 256         |
| Split          | `view(B,H,S,d).transpose(1,2)`         | `view(B,S,H,d).transpose(1,2)`                           | 必须先拆特征维，再交换 Head 和序列维  |
| Merge          | `transpose(1,2).view(...)`             | `transpose(1,2).contiguous().view(...)`                  | 转置后通常需要连续内存布局            |
| KV 判断        | `kv_cache["k"] not None`               | `kv_cache["k"] is not None`                              | 判断 None 必须使用 `is`               |
| KV 判断        | 直接访问 `kv_cache["k"]`               | `kv_cache is not None and kv_cache.get("k") is not None` | 避免空缓存或不存在的 Key              |
|                |                                        | import torch.nn.functional as F                          |                                       |
| KV 拼接        | 忘了API格式`cat(...,dim=2)`            | `torch.cat([old_k,k],dim=2)`                             | `(B,H,T,d)` 的第 2 维是时间维         |
| Cache 保存     | 把 `new_cache` 放进 `if`               | 放在 `if` 外                                             | 第一次调用也必须生成 Cache            |
| Cache 保存     | `kv_cache["k"]=k`                      | `new_cache={"k":k,"v":v}`                                | `kv_cache=None` 时无法赋值            |
| 长度变量       | 把 `q.size(2)` 当作历史长度            | `T_new=q.size(2)`                                        | Q 只包含新增 Token                    |
| 长度变量       | 在拼接 K 之前计算 `T_total`            | 拼接后 `T_total=k.size(2)`                               | K 拼接后才包含历史 Token              |
| 长度变量       | 把 `T_total-T_new` 当作新长度          | `past_len=T_total-T_new`                                 | 差值是历史 Token 数                   |
| Matmul         | `k.transpose()`                        | `k.transpose(-2,-1)`                                     | 交换最后两个维度                      |
| Matmul         | 混淆 `*`、`@`、`matmul`                | `@` 和 `matmul` 是矩阵乘法                               | `*` 是逐元素乘法                      |
| Score 缩放     | `//sqrt(d_head)`                       | `/ math.sqrt(self.d_head)`                               | `//` 是向下取整除法                   |
| Mask 创建      | `dtype=bool`                           | `dtype=torch.bool`                                       | 参数拼写错误                          |
| Mask 形状      |                                        | `ones(T_new,T_total)`                                    | 形状和attention最后两层一样（广播）   |
| Mask 设备      | `torch.ones(...)` 未指定设备           | `device=x.device`                                        | 防止 Mask 在 CPU、Scores 在 GPU       |
| Mask 规则      | 不理解 `diagonal=1`                    | 从主对角线上方第一条对角线开始保留                       | `triu` 的偏移规则                     |
| Mask 规则      | `diagonal=T_new+1`                     | `diagonal=past_len+1`                                    | 历史长度决定偏移，`+1` 允许看到自己   |
| Mask 含义      | 混淆 0 和 1                            | `True/1` 为遮挡                                          | 与 `masked_fill(mask,-inf)` 配合      |
| Mask API       | `torch.mask_fill(...)`                 | `score.masked_fill(mask,-inf)`                           | 正确的方法名是 `masked_fill`          |
| Mask 值        | `float("inf")`                         | `float("-inf")`                                          | Softmax 后被屏蔽位置权重为 0          |
| Softmax        | 直接 `score @ v`                       | `att=F.softmax(score,dim=-1)`                            | 需要先把 Score 归一化为权重           |
| Attention 输出 | `att @ v.transpose(-2,-1)`             | `att @ v`                                                | V 不需要转置                          |
| 输出投影       | `return out`                           | `return self.Wo(out),new_cache`                          | 完整 MHA 包含输出投影，并需返回 Cache |

最容易混淆的 5 个公式

① Head 维度

$$
d_{\text{head}}=\frac{d_{\text{model}}}{H}
$$

---

② 历史、新增、总长度

$$
T_{\text{total}}=past\_len+T_{\text{new}}
$$

---

③ Attention Score

$$
Scores=\frac{QK^T}{\sqrt{d_{\text{head}}}}
$$

---

④ KV Cache 的因果 Mask 偏移

$$
diagonal=past\_len+1
$$

---

⑤ Attention 输出

$$
Output=\operatorname{Softmax}(MaskedScores)V
$$



```python
# 1. QKV Projection + Split
q = self._split(self.Wq(x))
k = self._split(self.Wk(x))
v = self._split(self.Wv(x))

# 2. KV Cache
T_new = q.size(2)

if kv_cache is not None and kv_cache.get("k") is not None:
    k = torch.cat([kv_cache["k"], k], dim=2)
    v = torch.cat([kv_cache["v"], v], dim=2)

new_cache = {"k": k, "v": v}

# 3. Length
T_total = k.size(2)
past_len = T_total - T_new

# 4. Attention Scores
scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

# 5. Causal Mask
mask = torch.triu(
    torch.ones(T_new, T_total, device=x.device, dtype=torch.bool),
    diagonal=past_len + 1
)

scores = scores.masked_fill(mask, float("-inf"))

# 6. Softmax + Value
attn = F.softmax(scores, dim=-1)
out = attn @ v

# 7. Merge + Output Projection
out = self.Wo(self._merge(out))

return out, new_cache
```



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
        new_cache = {"k": K, "v": V}                   # 应该放在 if 外面。  原因是：无论之前有没有 KV Cache，本次都需要生成新的 Cache

        # Q @ K^T: (B,H,T_new,d_k) @ (B,H,d_k,T_total)
        # -> scores: (B,H,T_new,T_total)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 注意 T_total = past_len + T_new 
        T_new, T_total = Q.size(2), K.size(2)          # 本步 query 数 / 全部可见 key 数
        #  True = 需屏蔽的未来位置 1（True）= 遮挡，不能关注该 Token # ‼️ triu = Triangular Upper（上三角矩阵）。T_new, T_total Mask 的形状必须和 Attention Scores 的最后两个维度一致 ‼️ device：PyTorch 要求参与 masked_fill() 运算的 scores 和 mask 位于同一个设备（Device）上。因为 torch.ones() 是凭空创建一个新 Tensor，而其他操作大多是基于已有 Tensor 计算，所以不需要重新指定 Device。
         mask = torch.triu( 
            torch.ones(T_new, T_total, device=x.device, dtype=torch.bool), 
            diagonal=past_len + 1,                     # ‼️ 偏移 past_len 使 query i 只能看到 key <= past_len+i
        )
        scores = scores.masked_fill(mask, float("-inf"))  # (T_new,T_total) 广播到 (B,H,T_new,T_total)

        attn = F.softmax(scores, dim=-1)               # 沿 T_total 归一化
        out = self._merge(torch.matmul(attn, V))       # (B,H,T_new,d_k) -> (B,T_new,D)
        return self.Wo(out), new_cache
```



Q.shape = (2, 8, 3, 64)(新增)
K.shape = (2, 8, 7, 64)(total)

torch.ones(T_new, T_total,...)

#### B）NumPy 版本

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

**变量含义**（公式符号 ↔ 代码名）：

| 符号 | 代码 | 含义 | shape |
| ---- | ---- | ---- | ----- |
| $$\theta$$ / $$\theta_{t-1}$$ | `param` | 待更新参数 | 任意，如 `(D,)` 或 `(M, N)` |
| $$g_t$$ | `grad` | 当前步纯梯度 $$\partial L/\partial\theta$$，**不含** weight decay | 与 `param` 相同 |
| $$m_t$$ | `m` | 一阶矩：梯度的 EWMA，给出更新方向 | 与 `param` 相同 |
| $$v_t$$ | `v` | 二阶矩：梯度平方的 EWMA，给出每个坐标的尺度 | 与 `param` 相同 |
| $$\hat m_t$$ | `m_hat` | bias-corrected 一阶矩 | 与 `param` 相同 |
| $$\hat v_t$$ | `v_hat` | bias-corrected 二阶矩 | 与 `param` 相同 |
| $$t$$ | `step` | 迭代计数，从 **1** 开始（不是 0） | 标量 |
| $$\eta$$ | `lr` | learning rate | 标量，默认 `1e-3` |
| $$\beta_1$$ | `beta1` | 一阶矩衰减率 | 标量，默认 `0.9` |
| $$\beta_2$$ | `beta2` | 二阶矩衰减率 | 标量，默认 `0.999` |
| $$\epsilon$$ | `eps` | 分母防除零 | 标量，默认 `1e-8` |
| $$\lambda$$ | `weight_decay` | 解耦的 weight decay 系数 | 标量，默认 `0.01` |

全程都是**逐元素**运算：`m`、`v`、`grad` 与 `param` 同 shape，不会在参数维上做 `sum` / `norm`。`m`、`v` 是优化器状态，下一步还要原样传回去。

**Adam+L2 vs AdamW**：若把 $$\lambda\theta$$ 加进梯度，再拿这个污染过的 $$g_t$$ 去更新 $$m,v$$，衰减会被自适应分母 $$\sqrt{\hat v}$$ 扭曲，不再是真正的 weight decay。AdamW 的 $$\lambda\theta$$ 不进 $$g_t$$，只在参数上乘 $$(1-\eta\lambda)$$。

##### 3️⃣ Implementation Blocks

**（A）PyTorch 版本（标准面试实现）**

```python
import torch
from torch import Tensor

def adamw_step(
    param: Tensor,                   # θ：待更新参数，任意 shape，如 (D,) / (M, N)
    grad: Tensor,                    # g_t：∂L/∂θ，与 param 同 shape，不含 λθ
    m: Tensor,                       # m_{t-1}：一阶矩状态，与 param 同 shape
    v: Tensor,                       # v_{t-1}：二阶矩状态，与 param 同 shape
    step: int,                       # t：从 1 开始的迭代计数，仅用于 bias correction
    lr: float = 1e-3,                # η
    beta1: float = 0.9,              # β1：一阶矩衰减率
    beta2: float = 0.999,            # β2：二阶矩衰减率
    eps: float = 1e-8,               # ε：分母防除零
    weight_decay: float = 0.01,      # λ
):
    # 全程逐元素，shape 不变。m/v 是优化器状态，不是梯度本身
    m = beta1 * m + (1.0 - beta1) * grad           # m_t：梯度的 EWMA（动量方向）
    v = beta2 * v + (1.0 - beta2) * grad.square()  # v_t：梯度平方的 EWMA（逐坐标尺度）

    # m_0=v_0=0，早期矩会被 (1-β^t) 缩小；step 必须从 1 起，不能是 0
    m_hat = m / (1.0 - beta1 ** step)              # m̂_t
    v_hat = v / (1.0 - beta2 ** step)              # v̂_t

    # decoupled weight decay：θ ← (1-ηλ)θ，不进 m/v
    param = param * (1.0 - lr * weight_decay)

    # Adam 自适应更新：每个坐标除以自己的 RMS 尺度
    param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return param, m, v                             # 新 θ 与新 m_t, v_t，供下一步传入
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
    # param=θ, grad=g_t, m=m_{t-1}, v=v_{t-1} 同 shape；step=t 从 1 开始
    m = beta1 * m + (1.0 - beta1) * grad               # 一阶矩 m_t：梯度 EWMA
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)        # 二阶矩 v_t：逐元素平方，不可用 norm 聚合

    m_hat = m / (1.0 - beta1 ** step)                  # m̂_t，抵消 m_0=0 的早期偏差
    v_hat = v / (1.0 - beta2 ** step)                  # v̂_t

    param = param * (1.0 - lr * weight_decay)          # θ ← (1-ηλ)θ，λ 不进 g_t
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)  # θ ← θ - η m̂ / (√v̂+ε)
    return param, m, v                                 # 新 θ, m_t, v_t
```

**最小可运行测试**

```python
# Demo
p = torch.tensor([1.0, -2.0])      # θ，待优化参数 (D=2,)
g = torch.tensor([0.1, -0.2])      # g_1，该步梯度，与 p 同 shape
m = torch.zeros_like(p)            # m_0 = 0
v = torch.zeros_like(p)            # v_0 = 0

p, m, v = adamw_step(
    p, g, m, v,
    step=1,                        # t=1；bias correction 分母为 1-β^1
    lr=1e-2,                       # η
    weight_decay=0.1,              # λ
)
print("param:", p)                 # 新 θ；同时得到新的 m_1, v_1
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
    pass   # param=θ, grad=g_t, m=m_{t-1}, v=v_{t-1} 同 shape；返回 (新 θ, 新 m_t, 新 v_t)
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

（iii）$$m_t$$（代码 `m`）是梯度 $$g_t$$ 的指数滑动平均（EWMA）。$$\beta_1$$ 越大，历史方向保留越多；$$1-\beta_1$$ 是当前梯度的权重。shape 与 `param` 完全一致，每个参数坐标各自一条动量。

**Step 2：更新二阶矩**

（i）公式

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

（ii）代码

```python
v = beta2 * v + (1.0 - beta2) * grad.square()   # square 是逐元素平方，不做任何维度聚合
```

（iii）$$v_t$$（代码 `v`）记录每个坐标梯度有多大。$$g_t^2$$ 必须逐元素平方；不能写成 `grad.norm() ** 2`，否则所有参数维会被聚成一个标量，Adam 的 per-coordinate 自适应就没了。

**Step 3：Bias correction**

（i）公式

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

（ii）代码

```python
m_hat = m / (1.0 - beta1 ** step)   # step=t 为标量，除数广播；shape 不变
v_hat = v / (1.0 - beta2 ** step)
```

（iii）$$m_0=v_0=0$$，所以 $$m_t$$ 的期望只有 $$(1-\beta_1^t)\mathbb{E}[g]$$，早期会系统性偏小。除以 $$1-\beta_1^t$$ 把这个因子除掉。`step` 必须从 1 起：若传入 0，分母变成 0。

**Step 4：Decoupled weight decay**

（i）公式

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

（ii）代码

```python
param = param * (1.0 - lr * weight_decay)   # 只缩放参数本身，不污染 m/v 的梯度统计
```

（iii）$$\lambda$$（代码 `weight_decay`）直接乘在 $$\theta$$ 上，等价于每步把参数朝 0 拉一点。它**不**加进 `grad`，因此 $$m,v$$ 只看纯梯度。这是 AdamW 和 “Adam + L2（$$g\leftarrow g+\lambda\theta$$）” 的核心差别。

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

（iii）$$\hat m_t$$ 给方向，$$\sqrt{\hat v_t}$$ 给该坐标的 RMS 尺度：梯度长期很大的坐标步长被缩小，很小的被放大。$$\epsilon$$ 防止 $$\hat v_t\approx 0$$ 时除零。这一步之后返回的 `m,v` 是 $$m_t,v_t$$，不是 hat 后的版本——下一步继续用未校正的矩做 EWMA。

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

![Screenshot 2026-11-02 at 20.26.39](./assets/Screenshot%202026-11-02%20at%2020.26.39.png)



![Screenshot 2026-11-02 at 20.34.18](./assets/Screenshot%202026-11-02%20at%2020.34.18.png)

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

```py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        assert d_model % num_heads == 0

        # ⭐【GQA 不同 1】要求 Q head 数能被 KV head 数整除
        # MHA: H = G，不需要单独分组
        # GQA: G < H，多个 Q head 共享一个 KV head
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads

        # ⭐【GQA 不同 2】新增 num_kv_heads
        # MHA: num_kv_heads = num_heads
        # GQA: num_kv_heads < num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = d_model // num_heads

        # ⭐【GQA 不同 3】新增 repeat，表示一个 KV head 服务多少个 Q head
        # 例如 H=8, G=2 -> repeat=4
        self.repeat = num_heads // num_kv_heads

        # 与 MHA 相同：Q 仍然有 H 个 head
        self.Wq = nn.Linear(d_model, d_model)

        # ⭐【GQA 不同 4】K/V 投影维度缩小
        # MHA: Linear(D, D) -> H * d_k
        # GQA: Linear(D, G * d_k)
        self.Wk = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.Wv = nn.Linear(d_model, num_kv_heads * self.head_dim)

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape

        # 与 MHA 相同：Q -> (B,H,T,d_k)
        Q = self.Wq(x).view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # ⭐【GQA 不同 5】K/V 只有 G 个 head，而不是 H 个
        # MHA: (B,H,T,d_k)
        # GQA: (B,G,T,d_k)
        K = self.Wk(x).view(
            B, T, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        V = self.Wv(x).view(
            B, T, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # ⭐【GQA 不同 6】核心区别：复制 K/V head
        # MHA: 不需要这两行
        # GQA: 每个 KV head 复制 repeat 次
        # (B,G,T,d_k) -> (B,H,T,d_k)
        K = K.repeat_interleave(self.repeat, dim=1)
        V = V.repeat_interleave(self.repeat, dim=1)

        # 以下与普通 MHA 完全相同
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)  # (B,H,T,T)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B,H,T,d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.Wo(out)  # (B,T,D)

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

```py
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, weight, rank=4, alpha=8):
        super().__init__()
        # weight: (d_in, d_out)，预训练权重
        d_in, d_out = weight.shape
        # 冻结原始权重 W0
        self.W0 = nn.Parameter(
            weight.clone(), requires_grad=False
        )
        # A: (d_in, r)，随机小值初始化
        self.A = nn.Parameter(
            torch.randn(d_in, rank) * 0.01
        )
        # B: (r, d_out)，初始化为零
        self.B = nn.Parameter(
            torch.zeros(rank, d_out)
        )
        # LoRA 缩放系数
        self.scale = alpha / rank

    def forward(self, x):
        # x: (N, d_in)
        # h: (N, r)，降维投影
        h = x @ self.A
        # delta: (N, d_out)，低秩增量
        delta = h @ self.B
        # 合并原始输出与 LoRA 输出
        return x @ self.W0 + self.scale * delta
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



#### **模拟题目：Implement Linear Regression Using NumPy and Gradient Descent**

给定训练数据 $X$ 和目标值 $y$，只使用 NumPy，从零实现一个线性回归模型，并使用 Batch Gradient Descent 训练参数。

要求支持多个输入特征、独立的权重 $w$ 与偏置 $b$、MSE Loss、指定学习率和迭代次数，最后返回训练后的模型参数及 Loss 历史，并证明 Loss 下降。

建议练习函数签名（自行设定，非官方接口）

`fit_linear_regression(X, y, lr, steps) -> (w, b, losses)`

设 $X$ 有 $N$ 个样本、$D$ 个特征：

预测：

$$
\hat y=Xw+b
$$

均方误差：

$$
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

对权重求梯度：

$$
\frac{\partial L}{\partial w}
=\frac{2}{N}X^\top(\hat y-y)
$$

对偏置求梯度：

$$
\frac{\partial L}{\partial b}
=\frac{2}{N}\sum_{i=1}^{N}(\hat y_i-y_i)
$$

然后更新：

$$
w\leftarrow w-\alpha\frac{\partial L}{\partial w}
$$

$$
b\leftarrow b-\alpha\frac{\partial L}{\partial b}
$$

其中 $\alpha$ 是学习率。



```python
import numpy as np

def train_step(X, y, w, b, lr):
    # X: (N, D)
    # y: (N,)
    # w: (D,)
    # b: scalar

    n = X.shape[0]

    # Step 1: Forward
    y_pred = X @ w + b           # (N,)

    # Step 2: MSE Loss
    error = y_pred - y           # (N,)
    loss = np.mean(error ** 2)   # scalar

    # Step 3: Gradients
    dw = (2.0 / n) * (X.T @ error)  # (D,)
    db = (2.0 / n) * np.sum(error)  # scalar

    # Step 4: Update
    w = w - lr * dw
    b = b - lr * db

    return w, b, float(loss)

def fit_linear_regression(X, y, lr=0.05, steps=2000):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if len(y) != X.shape[0]:
        raise ValueError("X and y must have same N")

    n, d = X.shape

    # Initialize
    w = np.zeros(d)
    b = 0.0
    losses = []

    # Training loop
    for _ in range(steps):
        w, b, loss = train_step(X, y, w, b, lr)
        losses.append(loss)

    return w, b, losses

# Test case
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])

w, b, losses = fit_linear_regression(X, y)

print("w:", w)
print("b:", b)
print("Initial loss:", losses[0])
print("Final loss:", losses[-1])
```

这段代码已经用示例数据实际计算验证：



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

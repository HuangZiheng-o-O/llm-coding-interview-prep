# LLM Coding Interview Handbook

面向 MLE / LLM 岗位手撕代码面试的复习资料。目标是**面试现场可手推、可手写、可解释**：优先保留核心计算图、shape、mask、索引和梯度语义，不追求 CUDA kernel、分布式容错、量化库兼容等工业细节。每题尽量同时给出 NumPy 与 PyTorch 版本，且两者保持同一数学对象和计算逻辑。

主文档：[LLM-Coding-Interview-Handbook.md](./LLM-Coding-Interview-Handbook.md)

## 按优先级组织

| 优先级 | 代表题目 |
| --- | --- |
| **S：必须闭卷写** | Softmax、Cross-Entropy、Linear Layer、LayerNorm、Softmax Attention、MHA、Causal Attention、KV Cache、RoPE、AdamW、Next-Token Loss、手推反向传播 |
| **S−：现代 LLM 高频** | RMSNorm、GQA、MQA、MLA、LoRA、Top-k / Top-p、GAE、SFT Masked Loss、LLaMA-style Block |
| **A：应熟练** | ReLU、Dropout、Embedding、GELU、SwiGLU、Cross-Attention、GPT-2 Block、Adam、Beam Search、Reward Model Loss、KL Penalty、InfoNCE、ALiBi、QLoRA、Gradient Checkpointing、混合精度 |
| **A−：理论热、完整手写略少** | FlashAttention、MoE、DPO、PPO、GRPO、DAPO、GSPO、BPE、PagedAttention、Continuous Batching |
| **B：有时间练** | BatchNorm、Kaiming 初始化、Gradient Clip / Accumulation、Cosine LR、Sliding Window、Speculative Decoding、INT8 |
| **C：岗位相关** | Linear Attention、Conv2D、ViT Patch Embedding、Linear Regression |

文档中标有 👀 的小节是重点复习条目。

## License

内容仅供个人学习与面试准备参考。

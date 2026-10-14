# Transformer from Scratch in PyTorch

基于 PyTorch 手动实现的经典 Encoder–Decoder Transformer，用于学习和理解 Transformer 的核心结构。

## 已实现

- Token Embedding
- 正弦位置编码（Sinusoidal Positional Encoding）
- 多头自注意力（Multi-Head Attention）
- Padding Mask 与 Causal Mask
- 前馈神经网络（Feed-Forward Network）
- Layer Normalization
- Encoder / Decoder 堆叠
- Decoder 的 masked self-attention 与 encoder–decoder cross-attention
- 词表输出层（Generator）

## 项目特点

核心注意力计算、mask 处理、位置编码、归一化和 Encoder–Decoder 结构均由本项目手动组织实现。项目使用 PyTorch 提供张量运算、参数管理、线性层、Dropout 和自动求导能力，没有调用 `torch.nn.Transformer` 或 `torch.nn.MultiheadAttention`。

## 文件

```text
Transformer.py   # Transformer 模型实现
```

## 快速使用

```python
import torch
from Transformer import Transformer

model = Transformer(
    d_model=512,
    vocab_size=10000,
    num_head=8,
    d_ff=2048,
    num_Layers=6,
    max_len=512,
)

src = torch.randint(0, 10000, (2, 16))
tgt = torch.randint(0, 10000, (2, 12))

# mask 使用 1 表示保留，0 表示屏蔽
src_mask = torch.ones(2, 16, dtype=torch.bool)
tgt_mask = torch.tril(torch.ones(11, 11, dtype=torch.bool))

logits = model(src, tgt, src_mask, tgt_mask)
print(logits.shape)  # [2, 11, 10000]
```

## 说明

这是一个学习型实现，当前仓库主要展示模型结构本身，不包含完整数据集、训练脚本或预训练权重。`Transformer.forward` 会使用 `tgt[:, :-1]` 作为 decoder 输入，因此目标 mask 应与截断后的目标序列长度对应。

## 依赖

- Python 3.9+
- PyTorch

## License

代码用于学习和交流。若要用于生产环境，请根据实际需求补充测试、训练流程和许可证信息。

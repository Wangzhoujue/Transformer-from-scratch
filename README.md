# Transformer from Scratch in PyTorch

一个用于学习 Transformer 原理的 Encoder–Decoder 实现。项目使用 PyTorch 的张量、参数和基础网络层，手动组织注意力计算、mask 处理、位置编码、归一化以及编码器和解码器堆叠。

## 实现内容

- Token embedding
- 正弦位置编码（sinusoidal positional encoding）
- 缩放点积注意力（scaled dot-product attention）
- 多头注意力（multi-head attention）
- Padding mask 与 causal mask
- 两层前馈网络（feed-forward network）
- Layer normalization
- Encoder self-attention
- Decoder masked self-attention
- Encoder–decoder cross-attention
- 词表投影层（generator）

项目没有调用 `torch.nn.Transformer` 或 `torch.nn.MultiheadAttention`。PyTorch 仅用于张量运算、参数管理、线性层、Dropout 和自动求导。

## 项目结构

```text
Transformer.py   # 模型实现
README.md        # 项目说明
```

## 安装

```bash
pip install torch
```

需要 Python 3.9 或更高版本。

## 快速开始

```python
import torch
from Transformer import Transformer

vocab_size = 10_000
src_len = 16
tgt_len = 12

model = Transformer(
    d_model=512,
    vocab_size=vocab_size,
    num_head=8,
    d_ff=2048,
    num_Layers=6,
    max_len=512,
)

src = torch.randint(0, vocab_size, (2, src_len))
tgt = torch.randint(0, vocab_size, (2, tgt_len))

# mask 中 1/True 表示保留，0/False 表示屏蔽。
src_mask = torch.ones(2, src_len, dtype=torch.bool)
decoder_len = tgt_len - 1
tgt_mask = torch.tril(torch.ones(decoder_len, decoder_len, dtype=torch.bool))

logits = model(src, tgt, src_mask, tgt_mask)
print(logits.shape)  # torch.Size([2, 11, 10000])
```

`Transformer.forward` 会将 `tgt[:, :-1]` 作为 decoder 输入，因此 `tgt_mask` 的序列长度应为 `tgt.size(1) - 1`。模型输出的是未归一化 logits，形状为 `[batch_size, target_length - 1, vocab_size]`，训练时通常配合交叉熵损失使用。

## Mask 约定

`src_mask` 可以使用形状 `[batch_size, source_length]` 的 padding mask；`tgt_mask` 可以使用形状 `[target_length, target_length]` 的下三角 causal mask。mask 的值为 0 的位置不会参与注意力计算。

## 项目定位

这是一个面向学习和源码阅读的实现，重点是展示 Transformer 的核心结构。目前仓库不包含完整数据集、训练脚本、分词器或预训练权重，因此不直接提供翻译质量或论文结果复现。

## License

本项目用于学习和交流。正式使用前，请根据实际需求补充测试、训练流程和明确的开源许可证。

# Embedding学习笔记

这是一个Embedding模型学习笔记，实现了一个简易的基于BERT的Embedding模型，用于实现语义相似度相关任务

### 1.训练tokenizer

Embedding模型常与tokenizer搭配使用，输入字符串先由tokenizer转换为id列表，Embedding模型再将id列表转为词向量。

训练tokenizer的过程实质上是建立词汇表，即token和id的一一对应关系，一般流程如下：

- 建立初始词汇表，如字节值为0-255的字符，即ASCII码表
- 对训练语料用正则表达式进行分词，并将每个分块转为UTF-8编码的字节列表
- 遍历所有列表，找出出现最多的相邻对，将其合并作为新词汇，并将字节列表中的该相邻对用新词汇的id替代
- 重复上述过程直至词汇表数量到达预设值

以字符串“hi! This apple belongs to him”为例进行说明，将其分词得到

```
['hi', '!', ' This', ' apple', ' belongs', ' to', ' him']
```

将分词结果转为UTF-8编码的字节列表得到

```
[[104, 105], [33], [32, 84, 104, 105, 115], [32, 97, 112, 112, 108, 101], [32, 98, 101, 108, 111, 110, 103, 115], [32, 116, 111], [32, 104, 105, 109]]
```

列表中(104,105)，即hi出现最多，作为词汇表下一个新词汇

词汇id自增生成，假设其id为256，则更新字节列表，用256代替所有(104,105)得到

```
[[256], [33], [32, 84, 256, 115], [32, 97, 112, 112, 108, 101], [32, 98, 101, 108, 111, 110, 103, 115], [32, 116, 111], [32, 256, 109]]
```

### 2.训练embedding模型

embedding模型的作用是将一个字符串转为一个向量，一般为transformer encoder only，大部分Embedding模型都是基于BERT模型的改进版本，这里实现一个简易的基于BERT的Embedding模型

其中Tokenizer用了开源的bert-base-chinese，数据集使用ChineseCSTS

Tokenizer可在huggingface下载：

```shell
cd embedding
huggingface-cli download --resume-download google-bert/bert-base-chinese --local-dir ./google-bert/bert-base-chinese
```

模型定义如下：

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
BERT Embedding 层有两种属性：
    1. Token Embedding : 类似GPT中的输入样本文字部分
    2. Positional Embedding : 通过正余弦添加位置信息
    将这两个embedding做加法，得到最终给transformer encoder层的输入。
"""
class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, device, vocab_size, context_length):
        super().__init__()
        # 每个 token 的向量维度
        self.d_model = d_model
        self.vocab_size = vocab_size
        # 句子最大长度
        self.context_length = context_length
        self.device = device
        # 创建一个 vocab_size × d_model 的嵌入矩阵，每行表示一个token
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model).to(self.device)
        # 创建一个 2 × d_model 的矩阵（表示两种句子编号 0 和 1）
        # self.segment_embedding = nn.Embedding(2, self.d_model).to(self.device)
        self.position_embedding = nn.Parameter(self.create_position_encoding().to(self.device), requires_grad=False)
        # 对最后的 embedding 进行归一化，稳定训练
        self.layer_norm = nn.LayerNorm(self.d_model)

    def create_position_encoding(self):
        # context_length × d_model 的零矩阵，存储所有位置的编码
        position_encoding = torch.zeros(self.context_length, self.d_model)
        # 生成从 0 到 context_length-1 的索引
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        # 计算 Transformer 位置编码的缩放因子，用于调整不同维度上的正弦/余弦波形，使得低维编码变化缓慢，高维编码变化较快
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        # 偶数索引的维度使用 sin()，奇数索引的维度使用 cos()
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding
    '''
    输入batch_size * seq_len向量
    输出batch_size * seq_len * d_model向量
    '''
    def forward(self, idx):
        # idx是一个1 * seq_len矩阵，表示一个句子
        idx = idx.to(self.device)

        position_embedding = self.position_embedding[:idx.size(1), :]

        # idx为batch_size * seq_len维向量， self.token_embedding(idx)返回的是batch_size * seq_len * d_model向量，通过索引映射
        # position_embedding为self.position_embedding的前idx.size(1)行
        x = self.token_embedding(idx) + position_embedding
        # 归一化
        return self.layer_norm(x)
'''
定义前馈网络
对每个 token 的表示进行 扩展、非线性变换、还原，增强特征表达能力
'''
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        # 输入输出维度，即每个token的向量维度
        self.d_model = d_model
        # 用于防止过拟合的丢弃率
        self.dropout = dropout
        # 输入为x,x为batch_size * seq_len * d_model向量
        # 第一层将维度扩展为batch_size*seq_len*4d_model
        # ReLU作用于每个元素，维度不变
        # 形状变回batch_size*seq_len*d_model
        # 训练时，随机屏蔽dropout的单元
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(self.dropout)
        )

    '''
    输入batch_size * seq_len * d_model向量
    输出batch_size * seq_len * d_model向量
    '''
    def forward(self, x):
        return self.ffn(x)


'''
定义单头注意力
'''
class Attention(nn.Module):
    def __init__(self, d_model, head_size, context_length, dropout):
        super().__init__()
        # 输入数据的维度,即每个token的向量维度
        self.d_model = d_model
        # 单个注意力头（Attention Head）的维度
        self.head_size = head_size
        # 序列的最大长度，即句子长度
        self.context_length = context_length
        # 用于防止过拟合的丢弃率
        self.dropout = dropout
        # Q、K、V线性变换矩阵
        # 将输入x(batch_size * seq_len * d_model)映射为batch_size * seq_len * head_size
        self.Wq = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_size, bias=False)
        self.dropout = nn.Dropout(self.dropout)

    '''
    输入batch_size * seq_len * d_model向量
    输出batch_size * seq_len * head_size向量
    '''
    def forward(self, x):
        # q、k、v为batch_size * seq_len * head_size
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # 将k的最后两个维度交换与q点乘，weights的维度为batch_size * seq_len * seq_len
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # 对最后一个维度做归一化
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # output维度为batch_size * seq_len * head_size
        output = weights @ v

        return output

# Define Multi-head Attention ｜ 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.d_model = d_model
        # 多头注意力的头数
        self.num_heads = num_heads
        # 单个注意力头（Attention Head）的维度
        self.head_size = head_size
        # 序列的最大长度，即句子长度
        self.context_length = context_length
        self.dropout = dropout
        self.heads = nn.ModuleList(
            [Attention(self.d_model, self.head_size, self.context_length, self.dropout) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    '''
    输入batch_size * seq_len * d_model向量
    输出batch_size * seq_len * （num_heads * head_size）向量
    其中num_heads *  head_size = d_model
    '''
    def forward(self, x):
        # head_outputs为num_heads * batch_size * seq_len * head_size向量
        head_outputs = [head(x) for head in self.heads]
        # # head_outputs为batch_size * seq_len * （num_heads * head_size）向量
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection_layer(head_outputs))
        return out

# Define Transformer Block ｜ 定义Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        # 归一化层
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, context_length, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)

    '''
    输入batch_size * seq_len * d_model向量
    输出batch_size * seq_len * d_model向量
    '''
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, head_size, context_length, num_blocks, dropout, device):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length
        self.num_blocks = num_blocks
        self.device = device
        self.embedding = EmbeddingLayer(self.d_model, self.device, vocab_size, self.context_length)
        # 按顺序堆叠多个TransformerBlock和一个LayerNorm
        self.transformer_blocks = nn.Sequential(*(
            [TransformerBlock(self.d_model, self.num_heads, self.head_size, self.context_length, dropout) for _ in
             range(self.num_blocks)] +
            [nn.LayerNorm(self.d_model)]
        ))

    def forward(self, idx):
        x = self.embedding(idx)
        for block in self.transformer_blocks:
            x = block(x)
        return torch.mean(x, dim=1)

```


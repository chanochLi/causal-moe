from torch import nn
import torch
from torch.nn import functional as F
import math
from config.model_config import BasicModelConfig


class Attention(nn.Module):

    def __init__(self, config: BasicModelConfig):
        super().__init__()

        # 记录参数
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_head = config.num_head
        self.head_dim = config.head_dim
        self.dropout_rate = config.dropout_rate
        assert self.head_dim * self.num_head == self.hidden_dim, "head * head_dim does not equal to hidden_dim"

        # 投影矩阵
        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        # dropout
        self.attn_drop = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 记录输入纬度 [batch, seq_len, hidden_dim]
        batch, seq_len, _ = x.shape

        # 利用投影矩阵映射 [batch, seq_len, hidden_dim]
        q = self.Wq(x)
        k = self.Wq(x)
        v = self.Wq(x)

        # 转换为各个头 [batch, head, seq_len, head_dim]
        q = q.view(batch, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_head, self.head_dim).transpose(1, 2)

        # 获取注意力分数 [batch, head, seq_len, seq_len]
        attn_score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.tril()
            attn_score = torch.masked_fill(attn_score, mask==0, -torch.inf)
        else:
            mask = torch.ones_like(attn_score).tril()
            attn_score = torch.masked_fill(attn_score, mask==0, -torch.inf)

        # 获取注意力权重 [batch, head, seq_len, seq_len]
        attn_weight = F.softmax(attn_score, -1)
        attn_weight = self.attn_drop(attn_weight)

        # 获取attn值 [batch, head, seq_len, head_dim]
        output = attn_weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, -1)
        output = self.output_layer(output)

        return output

if __name__ == '__main__':
    attention_mask = (
        torch.tensor(
            [
                [0, 1],
                [0, 0],
                [1, 0],
            ]
        )
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(3, 8, 2, 2)
    )

    x = torch.rand(3, 2, 128)

    from config.model_config import AttentionConfig
    config = AttentionConfig(128, 8, 16, 0.1)
    net = Attention(config)
    print(net(x, attention_mask).shape)
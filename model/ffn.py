from config import BasicConfig
from torch import nn
import torch
from torch.nn import functional as F


class UpDownFFN(nn.Module):
    # the initial up scale and down scale FFN layer

    def __init__(self, config: BasicConfig):
        super().__init__()

        # 记录参数
        self.hidden_dim = config.hidden_dim
        if hasattr(config, 'scale'):
            self.scale = config.scale
        else:
            self.scale = None
        self.dropout_rate = config.dropout_rate

        # 创建FFN
        if self.scale is not None:
            self.down = nn.Linear(self.hidden_dim, self.hidden_dim * self.scale)
            self.up = nn.Linear(self.scale * self.hidden_dim, self.hidden_dim)
        else:
            self.down = None
            self.up = None
        if hasattr(config, 'act'):
            self.act = config.act()
        else:
            self.act = None
        self.drop = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim * scale]
        mid = self.act(self.down(x))

        # output: [batch, seq_len, hidden_dim]
        output = self.drop(self.up(mid))
        
        return output

class SwiGLUFFN(nn.Module):
    def __init__(self, config: BasicConfig):
        super().__init__()

        # 记录参数
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.mid_dim = self.hidden_dim * 8 // 3
        self.dropout_rate = config.dropout_rate

        # 网络结构
        self.up = nn.Linear(self.hidden_dim, self.mid_dim, bias=False)
        self.gate = nn.Linear(self.hidden_dim, self.mid_dim, bias=False)
        self.down = nn.Linear(self.mid_dim, self.hidden_dim, bias=False)
        self.drop = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, mid_dim]
        mid = F.silu(self.up(x)) * self.gate(x)

        # [batch, seq_len, hidden_dim]
        output = self.drop(self.down(mid))

        return output
        
from config.model_config import BasicModelConfig, BlockConfig, MOEBlockConfig
from torch import nn
import torch
from attention import Attention
from ffn import UpDownFFN, SwiGLUFFN
from moe import BasicMoE

class Block(nn.Module):

    def __init__(self, config: BasicModelConfig):
        super().__init__()

        # 记录参数
        self.config = config
        self.hidden_dim = config.hidden_dim

        # transformer层
        self.attn_ln = nn.LayerNorm(self.hidden_dim, eps=1e-4)
        self.attn = Attention(config)
        self.ffn_ln = nn.LayerNorm(self.hidden_dim, eps=1e-4)
        self.ffn = UpDownFFN(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 先计算attn
        attn_out = self.attn(x, mask)
        mid = self.attn_ln(x + attn_out)

        # 计算ffn
        ffn_out = self.ffn(mid)
        output = self.ffn_ln(ffn_out + mid)

        return output

class SwiGLUBlock(Block):

    def __init__(self, config):
        super().__init__(config)

        # 替换FFN层
        self.ffn = SwiGLUFFN(config)

class BasicMOEBlock(Block):

    def __init__(self, config):
        super().__init__(config)

        # 替换FFN层
        self.ffn = BasicMoE(config)


if __name__ == '__main__':
    x = torch.rand(3, 4, 64)
    config = MOEBlockConfig(hidden_dim=64, num_head=8, head_dim=8, dropout_rate=0.1, expert_num=4, top_k=2, share_expert_num=2)
    net = BasicMOEBlock(config)
    mask = (
        torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
        .unsqueeze(1)
        .unsqueeze(2)
        .repeat(1, 8, 4, 1)
    )
    print(net)
    print(net(x, mask).shape)

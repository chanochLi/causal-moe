import torch
from torch import nn
from torch.nn import functional as F
from config.model_config import BasicModelConfig
from .ffn import UpDownFFN, SwiGLUFFN


class BasicExpert(nn.Module):
    # 最简单的expert就是一个线性层
    def __init__(self, config: BasicModelConfig):
        super(BasicExpert, self).__init__()

        self.expert = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor):
        return self.expert(x)

class UpDownExpert(BasicExpert):

    def __init__(self, config):
        super().__init__(config)

        self.expert = UpDownFFN(config)
from dataclasses import dataclass
import torch

@dataclass
class BasicConfig:
    hidden_dim: int
    dropout_rate: float

@dataclass
class MOEConfig(BasicConfig):
    expert_num: int
    top_k: int
    share_expert_num: int

@dataclass
class AttentionConfig(BasicConfig):
    num_head: int
    head_dim: int

@dataclass
class UpDownFFNConfig(BasicConfig):
    scale: int
    act: torch.nn.Module

@dataclass
class BlockConfig(AttentionConfig, UpDownFFNConfig):
    pass

@dataclass
class BasicMOEBlockConfig(AttentionConfig, MOEConfig):
    pass

@dataclass
class UpDownMOEBlockConfig(AttentionConfig, MOEConfig, UpDownFFNConfig):
    pass

@dataclass
class TransformerConfig(BlockConfig):
    num_layer: int
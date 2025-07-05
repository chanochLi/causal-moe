from dataclasses import dataclass
import torch

@dataclass
class BasicModelConfig:
    hidden_dim: int
    dropout_rate: float

@dataclass
class MOEConfig(BasicModelConfig):
    expert_num: int
    top_k: int
    share_expert_num: int

@dataclass
class AttentionConfig(BasicModelConfig):
    num_head: int
    head_dim: int

@dataclass
class UpDownFFNConfig(BasicModelConfig):
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
    block_size: int
    num_layer: int
    voc_size: int
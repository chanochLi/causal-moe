import torch
from torch import nn
from config.model_config import BasicModelConfig
from .block import Block
from torch.nn import functional as F


class CausalTransformer(nn.Module):
    
    def __init__(self, config: BasicModelConfig):
        super().__init__()
        
        # 记录参数
        self.config = config
        
        # embedding + block + ln + head
        self.token_embedding_table = nn.Embedding(config.voc_size, config.hidden_dim)
        self.position_embedding_table = nn.Embedding(config.block_size, config.hidden_dim)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_layer)]
        )
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.voc_size, bias=False)
        
        # tie weight
        self.token_embedding_table.weight = self.lm_head.weight
        
    def _init_weights(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
                
        elif isinstance(nn.Embedding):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        # 输入是token idx
        # [batch, seq_len]
        batch, seq_len = idx.size()
        # [batch, seq_len, token_embedding]
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device=idx.device)
        )
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.ln_final(x)
        
        # 计算自回归loss
        if targets is None:
            loss = None
        else:
            batch, seq_len, voc_size = logits.size()
            logits = logits.view(batch * seq_len, voc_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_token: int):
        # idx: [batch, seq_len]
        pass
import torch
from torch import nn
from torch.nn import functional as F
from config import BasicConfig


class BasicGate(nn.Module):
    # 最简单的使用线性层作为gate
    def __init__(self, config: BasicConfig):
        super().__init__()

        self.gate = nn.Linear(config.hidden_dim, config.expert_num)

    def forward(self, x: torch.Tensor):
        return self.gate(x)


## 参考mistral MOE的代码
class MOERouter(nn.Module):

    def __init__(self, config: BasicConfig):
        super().__init__()

        self.top_k = config.top_k
        self.expert_num = config.expert_num

        # 专家选择机制
        self.gate = nn.Linear(config.hidden_dim, config.expert_num)

    def forward(self, x: torch.Tensor):
        # x[batch*seq_len, hidden_dim]
        # 计算每个专家选到的概率
        router_logits = self.gate(x)
        router_prob = F.softmax(router_logits, dim=1)

        # [batch*seq_len, top_k]
        # 计算top-k专家输出，torch.topk可以反向传播
        router_weight, select_expert_index = torch.topk(router_prob, self.top_k, dim=-1)

        # 重新归一化
        router_weight = router_weight / router_weight.sum(dim=-1, keepdim=True).to(
            x.dtype
        )

        # [batch*seq, top_k, num_expert]
        expert_mask = F.one_hot(
            select_expert_index, num_classes=self.expert_num
        ).permute(2, 1, 0)

        return router_logits, router_weight, select_expert_index, expert_mask

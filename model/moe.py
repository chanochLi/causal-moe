import torch
from torch import nn
from torch.nn import functional as F
from experts import BasicExpert, UpDownExpert
from config.model_config import BasicModelConfig, MOEConfig
from gate import MOERouter


class BasicMoE(nn.Module):
    # 线性专家组合为专家层
    def __init__(self, config: BasicModelConfig):
        super(BasicMoE, self).__init__()

        # 门控选择所有专家，可以有很复杂的实现
        self.gate = nn.Linear(config.hidden_dim, config.expert_num)

        # 基础的线性层作为专家
        self.experts = nn.ModuleList(
            BasicExpert(config) for _ in range(config.expert_num)
        )

    def forward(self, x: torch.Tensor):
        # x: [batch, seq_len, hidden_dim]
        batch, seq_len, hidden_dim = x.size()

        # [batch*seq_len, hidden_dim]
        hidden_state = x.view(-1, hidden_dim)

        # 获取专家数[batch*seq_len, 1, num_expert]
        expert_weight = F.softmax(self.gate(hidden_state), dim=1).unsqueeze(1)

        # 获取所有专家输出[batch*seq_len, num_expert, feature_out]
        expert_output = torch.concat(
            [expert(hidden_state).unsqueeze(1) for expert in self.experts], dim=1
        )

        # 加权求和，[batch*seq_len, 1, feature_out]
        output = expert_weight @ expert_output

        return output.squeeze(1).view(batch, seq_len, hidden_dim)


class SparseMOE(nn.Module):

    def __init__(self, config: BasicModelConfig):
        super().__init__()

        self.config = config
        self.top_k = self.config.top_k
        self.hidden_dim = self.config.hidden_dim
        self.expert_num = self.config.expert_num

        # 初始化模型
        self.experts = nn.ModuleList(
            BasicExpert(config)
            for _ in range(config.expert_num)
        )

        # router路由选择，选择top_k个专家
        self.router = MOERouter(config)

    def forward(self, x: torch.Tensor):
        # x: [batch, seq, hidden_dim]
        batch, seq_len, hidden_dim = x.size()

        # [batch* seq, hidden_dim]
        hidden_state = x.view(-1, hidden_dim)

        # 做专家的计算
        router_logits, router_weight, select_expert_index, expert_mask = self.router(
            hidden_state
        )

        # 初始化最终的hidden_state
        final_hidden_state = torch.zeros(
            (batch * seq_len, hidden_dim),
            dtype=hidden_state.dtype,
            device=hidden_state.device,
        )

        # 从expert的角度算，遍历专家计算
        for expert_index in range(self.expert_num):
            expert_layer = self.experts[expert_index]
            # [expert_num, top_k, batch*seq_len] -> [top_k, batch*seq_len]
            current_expert_mask = expert_mask[expert_index]

            # idx: topk中的第几个，top_x是batch*seq_len的索引
            router_weight_idx, top_x = torch.where(current_expert_mask)

            # [select_token_num = len(top_x), hidden_dim]
            current_state = hidden_state.unsqueeze(0)[:, top_x, :].reshape(
                -1, hidden_dim
            )
            current_state = expert_layer(current_state)

            # [selelcted_token_num, 1]
            current_token_router_weight = router_weight[
                top_x, router_weight_idx
            ].unsqueeze(-1)

            current_hidden_state = current_state * current_token_router_weight

            final_hidden_state.index_add_(
                0, top_x, current_hidden_state.to(hidden_state.dtype)
            )

        final_hidden_state = final_hidden_state.reshape(batch, seq_len, hidden_dim)

        return final_hidden_state, router_logits


if __name__ == "__main__":
    x = torch.randn(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    model = SparseMOE(config)
    print(model(x)[0].shape)

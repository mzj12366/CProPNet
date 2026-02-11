import torch
import torch.nn as nn
from torch import Tensor
from models.layers import DayPositionalEncoding, NutBlock, NutBlock_Attn, NutBlock_Diff, ScalesPool, NutBlock_B

class NutEncoder(nn.Module):
    """
    输入: [B, T=max_days, intake_dim=79]
      ↓ intake_layers
    嵌入: [B, T, intake_emb_dim=128]  
      ↓ pos_enc
    位置编码: [B, T, intake_emb_dim=128]
      ↓ NutBlock × 5 stages
    多阶段处理: [B, T, intake_emb_dim=128]
      ↓ ScalesPool
    池化: [B, stage_num=5, intake_emb_dim=128]
      ↓ intake_output
    输出: [B, stage_num=5, out_intake_dim=64]
    """
    def __init__(self, config):
        super(NutEncoder, self).__init__()

        self.intake_dim = config.intake_dim
        self.intake_emb_dim = config.intake_emb_dim
        self.stage_num = config.stage_num
        self.out_intake_dim = config.out_intake_dim

        self.num_heads = config.num_heads
        self.feedward_dim = config.feedward_dim
        self.dropout = config.dropout
        self.attn_num_layers = config.attn_num_layers
        self.group_days = config.group_days
        self.max_days = config.max_days

        # 基本校验
        assert self.stage_num > 0, "stage_num must be > 0"
        assert self.intake_dim > 0 and self.intake_emb_dim > 0, "intake dims must be > 0"

        self.intake_layers = nn.Linear(self.intake_dim, self.intake_emb_dim) 

        self.pos_enc = DayPositionalEncoding(self.intake_emb_dim, self.max_days)

        self.blocks = nn.ModuleList([
            NutBlock(self.intake_emb_dim, self.num_heads, self.feedward_dim, self.dropout, self.attn_num_layers, self.group_days)
            for _ in range(self.stage_num)
        ])

        self.pool = ScalesPool(self.intake_emb_dim)
        self.intake_output = nn.Linear(self.intake_emb_dim, self.out_intake_dim)

    def forward(self, intake: Tensor) -> Tensor:
        """编码营养摄入时序。

        参数:
            intake: [B, T=max_days, intake_dim]

        返回:
            output: [B, stage_num, out_intake_dim]
        """
        intake_emb = self.intake_layers(intake)
        stage_outputs_days = []  # type: list[Tensor]  # 每个元素形状 [B, new_days, F]
        h = self.pos_enc(intake_emb)

        for block in self.blocks:
            h = block(h)
            stage_outputs_days.append(h)

        # 对每个 stage 的 [B, new_days, F] 进行加权平均 -> [B, F]
        pooled_per_stage = [self.pool(y) for y in stage_outputs_days]  # 列表内每个 [B, F]

        # 拼接成 [B, stage_num, F]
        x = torch.stack(pooled_per_stage, dim=1)
        output = self.intake_output(x)
        
        return output


class NutEncoder_Attn(nn.Module):
    """
    输入: [B, T=max_days, intake_dim=79]
      ↓ intake_layers
    嵌入: [B, T, intake_emb_dim=128]  
      ↓ pos_enc
    位置编码: [B, T, intake_emb_dim=128]
      ↓ NutBlock × 5 stages
    多阶段处理: [B, T, intake_emb_dim=128]
      ↓ ScalesPool
    池化: [B, stage_num=5, intake_emb_dim=128]
      ↓ intake_output
    输出: [B, stage_num=5, out_intake_dim=64]
    """
    def __init__(self, config):
        super(NutEncoder_Attn, self).__init__()

        self.intake_dim = config.intake_dim
        self.intake_emb_dim = config.intake_emb_dim
        self.stage_num = config.stage_num
        self.out_intake_dim = config.out_intake_dim

        self.num_heads = config.num_heads
        self.feedward_dim = config.feedward_dim
        self.dropout = config.dropout
        self.attn_num_layers = config.attn_num_layers
        self.group_days = config.group_days
        self.max_days = config.max_days

        # 基本校验
        assert self.stage_num > 0, "stage_num must be > 0"
        assert self.intake_dim > 0 and self.intake_emb_dim > 0, "intake dims must be > 0"

        self.intake_layers = nn.Linear(self.intake_dim, self.intake_emb_dim) 

        self.pos_enc = DayPositionalEncoding(self.intake_emb_dim, self.max_days)

        self.blocks = nn.ModuleList([
            NutBlock_Attn(self.intake_emb_dim, self.num_heads, self.feedward_dim, self.dropout, self.attn_num_layers, self.group_days)
            for _ in range(self.stage_num)
        ])

        self.pool = ScalesPool(self.intake_emb_dim)
        self.intake_output = nn.Linear(self.intake_emb_dim, self.out_intake_dim)

    def forward(self, intake: Tensor) -> Tensor:
        """编码营养摄入时序。

        参数:
            intake: [B, T=max_days, intake_dim]

        返回:
            output: [B, stage_num, out_intake_dim]
        """
        intake_emb = self.intake_layers(intake)
        stage_outputs_days = []  # type: list[Tensor]  # 每个元素形状 [B, new_days, F]
        h = self.pos_enc(intake_emb)
        for block in self.blocks:
            h = block(h)
            stage_outputs_days.append(h)

        # 对每个 stage 的 [B, new_days, F] 进行加权平均 -> [B, F]
        pooled_per_stage = [self.pool(y) for y in stage_outputs_days]  # 列表内每个 [B, F]

        # 拼接成 [B, stage_num, F]
        x = torch.stack(pooled_per_stage, dim=1)
        output = self.intake_output(x)
        
        return output

class NutEncoder_Diff(nn.Module):
    """
    输入: [B, T=max_days, intake_dim=79]
      ↓ intake_layers
    嵌入: [B, T, intake_emb_dim=128]  
      ↓ pos_enc
    位置编码: [B, T, intake_emb_dim=128]
      ↓ NutBlock × 5 stages
    多阶段处理: [B, T, intake_emb_dim=128]
      ↓ ScalesPool
    池化: [B, stage_num=5, intake_emb_dim=128]
      ↓ intake_output
    输出: [B, stage_num=5, out_intake_dim=64]
    """
    def __init__(self, config):
        super(NutEncoder_Diff, self).__init__()

        self.intake_dim = config.intake_dim
        self.intake_emb_dim = config.intake_emb_dim
        self.stage_num = config.stage_num
        self.out_intake_dim = config.out_intake_dim

        self.num_heads = config.num_heads
        self.feedward_dim = config.feedward_dim
        self.dropout = config.dropout
        self.attn_num_layers = config.attn_num_layers
        self.group_days = config.group_days
        self.max_days = config.max_days

        # 基本校验
        assert self.stage_num > 0, "stage_num must be > 0"
        assert self.intake_dim > 0 and self.intake_emb_dim > 0, "intake dims must be > 0"

        self.intake_layers = nn.Linear(self.intake_dim, self.intake_emb_dim) 

        self.pos_enc = DayPositionalEncoding(self.intake_emb_dim, self.max_days)

        self.blocks = nn.ModuleList([
            NutBlock_Diff(self.intake_emb_dim, self.num_heads, self.feedward_dim, self.dropout, self.attn_num_layers, self.group_days)
            for _ in range(self.stage_num)
        ])

        self.pool = ScalesPool(self.intake_emb_dim)
        self.intake_output = nn.Linear(self.intake_emb_dim, self.out_intake_dim)

    def forward(self, intake: Tensor) -> Tensor:
        """编码营养摄入时序。

        参数:
            intake: [B, T=max_days, intake_dim]

        返回:
            output: [B, stage_num, out_intake_dim]
        """
        intake_emb = self.intake_layers(intake)
        stage_outputs_days = []  # type: list[Tensor]  # 每个元素形状 [B, new_days, F]
        h = self.pos_enc(intake_emb)
        for block in self.blocks:
            h = block(h)
            stage_outputs_days.append(h)

        # 对每个 stage 的 [B, new_days, F] 进行加权平均 -> [B, F]
        pooled_per_stage = [self.pool(y) for y in stage_outputs_days]  # 列表内每个 [B, F]

        # 拼接成 [B, stage_num, F]
        x = torch.stack(pooled_per_stage, dim=1)
        output = self.intake_output(x)
        
        return output
class NutEncoder_B(nn.Module):
    """
    输入: [B, T=max_days, intake_dim=79]
      ↓ intake_layers
    嵌入: [B, T, intake_emb_dim=128]  
      ↓ pos_enc
    位置编码: [B, T, intake_emb_dim=128]
      ↓ NutBlock × 5 stages
    多阶段处理: [B, T, intake_emb_dim=128]
      ↓ ScalesPool
    池化: [B, stage_num=5, intake_emb_dim=128]
      ↓ intake_output
    输出: [B, stage_num=5, out_intake_dim=64]
    """
    def __init__(self, config):
        super(NutEncoder_B, self).__init__()

        self.intake_dim = config.intake_dim
        self.intake_emb_dim = config.intake_emb_dim
        self.stage_num = config.stage_num
        self.out_intake_dim = config.out_intake_dim

        self.num_heads = config.num_heads
        self.feedward_dim = config.feedward_dim
        self.dropout = config.dropout
        self.attn_num_layers = config.attn_num_layers
        self.group_days = config.group_days
        self.max_days = config.max_days

        self.intake_layers = nn.Linear(self.intake_dim, self.intake_emb_dim) 

        self.pos_enc = DayPositionalEncoding(self.intake_emb_dim, self.max_days)

        self.blocks_attn = nn.ModuleList([
            NutBlock_Attn(self.intake_emb_dim, self.num_heads, self.feedward_dim, self.dropout, self.attn_num_layers, self.group_days)
            for _ in range(self.stage_num)
        ])
        self.blocks_diff = nn.ModuleList([
            NutBlock_Diff(self.intake_emb_dim, self.num_heads, self.feedward_dim, self.dropout, self.attn_num_layers, self.group_days)
            for _ in range(self.stage_num)
        ])
        self.pool = ScalesPool(self.intake_emb_dim)
        self.intake_output = nn.Linear(self.intake_emb_dim * 2, self.out_intake_dim)

    def forward(self, intake: Tensor) -> Tensor:
        """编码营养摄入时序。

        参数:
            intake: [B, T=max_days, intake_dim]

        返回:
            output: [B, stage_num, out_intake_dim]
        """
        intake_emb = self.intake_layers(intake)
        stage_outputs_diff = []  # type: list[Tensor]  # 每个元素形状 [B, new_days, F]
        stage_outputs_attn = []
        d = self.pos_enc(intake_emb)
        a = self.pos_enc(intake_emb)

        for block in self.blocks_attn :
            a = block(a)
            stage_outputs_attn.append(a)
        for block in self.blocks_diff:
            d = block(d)
            stage_outputs_diff.append(d)

        # 对每个 stage 的 [B, new_days, F] 进行加权平均 -> [B, F]
        pooled_per_attn = [self.pool(y) for y in stage_outputs_attn]  # 列表内每个 [B, F]
        pooled_per_diff = [self.pool(y) for y in stage_outputs_diff]  # 列表内每个 [B, F]

        # 拼接成 [B, stage_num, F]
        x = torch.stack(pooled_per_attn, dim=1)
        y = torch.stack(pooled_per_diff, dim=1)
        s = torch.cat([x, y], dim=-1)
        output = self.intake_output(s)
        
        return output
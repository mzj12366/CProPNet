import torch
import torch.nn as nn
import torch.nn.functional as F
import models.userencoders as userencoders
import models.nutencoders as nutencoders
from models.layers import AttentionEncoder

class NutModel(nn.Module):
    def __init__(self, config):
        super(NutModel, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)
        self.nut_encoder = nutencoders.NutEncoder(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]
        combined_features = torch.cat([user_projected, nut_representation], dim=1)  # [batch, 1+stage_num, out_intake_dim]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction

class NutModel_B(nn.Module):
    def __init__(self, config):
        super(NutModel_B, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)
        self.nut_encoder = nutencoders.NutEncoder_B(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim*2,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim*2, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]
        batch_size, stage_num, _ = nut_representation.size()

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation)  # [batch, 1, out_intake_dim]
        user_projected = user_projected.unsqueeze(1).repeat(1, stage_num, 1)
        combined_features = torch.cat([user_projected, nut_representation], dim=-1)  # [batch, stage_num, out_intake_dim*2]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction
        
class NutModel_Attn(nn.Module):
    def __init__(self, config):
        super(NutModel_Attn, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)
        self.nut_encoder = nutencoders.NutEncoder_Attn(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]
        combined_features = torch.cat([user_projected, nut_representation], dim=1)  # [batch, 1+stage_num, out_intake_dim]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction

        
class NutModel_Diff(nn.Module):
    def __init__(self, config):
        super(NutModel_Diff, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)
        self.nut_encoder = nutencoders.NutEncoder_Diff(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]
        combined_features = torch.cat([user_projected, nut_representation], dim=1)  # [batch, 1+stage_num, out_intake_dim]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction

class NutModel_Nolearn(nn.Module):
    def __init__(self, config):
        super(NutModel_Nolearn, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder_Nolearn(config)
        self.nut_encoder = nutencoders.NutEncoder(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]
        combined_features = torch.cat([user_projected, nut_representation], dim=1)  # [batch, 1+stage_num, out_intake_dim]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction

class NutModel_NoDiff(nn.Module):
    def __init__(self, config):
        super(NutModel_NoDiff, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)
        self.nut_encoder = nutencoders.NutEncoder_NoDiff(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]
        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]
        combined_features = torch.cat([user_projected, nut_representation], dim=1)  # [batch, 1+stage_num, out_intake_dim]

        former_output = self.former_enc(combined_features)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction
class OnlyNutModel(nn.Module):
    def __init__(self, config):
        super(OnlyNutModel, self).__init__()
        self.config = config

        # 子编码器
        self.nut_encoder = nutencoders.NutEncoder(config)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):

        nut_representation = self.nut_encoder(intake) # [batch, stage_num, out_intake_dim]

        former_output = self.former_enc(nut_representation)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction

class OnlyUserModel(nn.Module):
    def __init__(self, config):
        super(OnlyUserModel, self).__init__()
        self.config = config

        # 子编码器
        self.user_encoder = userencoders.UserEncoder(config)

        # 将用户表征映射到与营养表征相同的特征维度，以便拼接
        self.user_projector = nn.Linear(config.sym_dim * 2, config.out_intake_dim)

        # 融合编码器：对 [1 + stage_num, out_intake_dim] 做自注意力编码
        self.former_enc = AttentionEncoder(
            input_dim=config.out_intake_dim,
            num_heads=config.num_heads,
            feedward_dim=config.feedward_dim,
            dropout=config.dropout,
            num_layers=getattr(config, 'attn_num_layers', 2),
        )

        # 分类 MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.out_intake_dim, config.mlp_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim // 2, config.model_output_dim)
        )
    

    def forward(self, info, exam, intake):
        user_representation = self.user_encoder(info, exam) # [batch, sym_dim*2]

        # 对用户表征做线性映射并在序列维度上拼接
        user_projected = self.user_projector(user_representation).unsqueeze(1)  # [batch, 1, out_intake_dim]

        former_output = self.former_enc(user_projected)  # [batch, 1+stage_num, out_intake_dim]
        final_features = former_output.mean(dim=1)  # [batch, out_intake_dim]

        prediction = self.mlp(final_features)

        return prediction
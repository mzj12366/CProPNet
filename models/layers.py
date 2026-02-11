import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor


# 深层症状使用的注意力，无mask
class Attention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(Attention, self).__init__()
        self.affine1 = nn.Linear(feature_dim, attention_dim, bias=True)
        self.affine2 = nn.Linear(attention_dim, 1, bias=False)

    def initialize(self):
        nn.init.xavier_uniform_(self.affine1.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, feature, mask=None):
        attention = torch.tanh(self.affine1(feature))                                 # [batch_size, length, attention_dim]
        a = self.affine2(attention).squeeze(dim=2)                                    # [batch_size, length]
        if mask is not None:
            alpha = F.softmax(a.masked_fill(mask == 0, -1e9), dim=1).unsqueeze(dim=1) # [batch_size, 1, length]
        else:
            alpha = F.softmax(a, dim=1).unsqueeze(dim=1)                              # [batch_size, 1, length]
        out = torch.bmm(alpha, feature).squeeze(dim=1)                                # [batch_size, feature_dim]
        return out

# nut、和user使用的普通注意力，有mask
class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, feedward_dim, dropout, num_layers):
        super(AttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.feedward_dim = feedward_dim
        self.dropout = dropout
        self.num_layers = num_layers
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=feedward_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, input_emb):
        output = self.encoder(input_emb)
        return output



# nut使用的差分注意力，有mask
class DiffAttention(nn.Module):
    def __init__(self, input_dim, clamp_xi=0.2, eps: float = 1e-12):
        super(DiffAttention, self).__init__()

        self.input_dim = input_dim
        self.clamp_xi = float(clamp_xi)
        self.eps = float(eps)

        # 权重矩阵
        self.W_q = nn.Linear(input_dim, input_dim, bias=True)
        self.W_k = nn.Linear(input_dim, input_dim, bias=True)
        self.W_v = nn.Linear(input_dim, input_dim, bias=True)
        self.W_s = nn.Parameter(torch.randn(input_dim, 1))

    def clamp(self, x, tau):
        if tau == +1:
            mask = x >= self.clamp_xi
        elif tau == -1:
            mask = x <= -self.clamp_xi
        else:
            raise ValueError('tau 必须是 +1 或 -1')
        return torch.where(mask, x, torch.zeros_like(x))

    def forward(self, input_emb):
        batch_size,num_days, d_k = input_emb.size()

        q = self.W_q(input_emb)
        k = self.W_k(input_emb)
        v = self.W_v(input_emb)

        A_t = torch.tanh(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k))
        
        q_expanded = q.unsqueeze(2).expand(batch_size, num_days, num_days, d_k)
        k_expanded = k.unsqueeze(1).expand(batch_size, num_days, num_days, d_k)
        diff = torch.abs(q_expanded - k_expanded)
        A_s = torch.sigmoid(torch.matmul(diff, self.W_s).squeeze(-1))

        A_raw = A_t * A_s

        A_pos = self.clamp(A_raw, +1)
        A_neg = self.clamp(A_raw, -1)

        pos_norm = torch.sum(torch.abs(A_pos), dim=(1, 2), keepdim=True)
        neg_norm = torch.sum(torch.abs(A_neg), dim=(1, 2), keepdim=True)

        A_pos_normed = A_pos / (pos_norm + self.eps)
        A_neg_normed = A_neg / (neg_norm + self.eps)

        A = A_pos_normed + A_neg_normed

        output = torch.matmul(A, v)
        return output

class DiffAttentionEncoderLayer(nn.Module):
    """
    单层 DiffAttention Transformer Block：
    LN -> DiffAttention -> 残差
    LN -> FFN -> 残差
    """
    def __init__(self, input_dim, feedward_dim=256, dropout=0.1, clamp_xi=0.2, eps=1e-12):
        super(DiffAttentionEncoderLayer, self).__init__()
        self.input_dim = input_dim
        self.feedward_dim = feedward_dim

        # 差分注意力
        self.diff_attn = DiffAttention(input_dim=input_dim,
                                       clamp_xi=clamp_xi,
                                       eps=eps)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        # 前馈网络 FFN
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, feedward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedward_dim, input_dim),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        x: [batch, days, input_dim]
        """
        # 子层1：DiffAttention + 残差
        h = self.norm1(x)
        attn_out = self.diff_attn(h)                  # [B, days, input_dim]
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out                              # 残差

        # 子层2：FFN + 残差
        h2 = self.norm2(x)
        ffn_out = self.ffn(h2)                        # [B, days, input_dim]
        ffn_out = self.dropout_ffn(ffn_out)
        x = x + ffn_out                               # 残差

        return x


class DiffAttentionEncoder(nn.Module):
    """
    多层堆叠的 DiffAttention 编码器，类似 nn.TransformerEncoder
    """
    def __init__(self, input_dim, feedward_dim=256, dropout=0.1,
                 num_layers=1, clamp_xi=0.2, eps=1e-12):
        super(DiffAttentionEncoder, self).__init__()
        self.input_dim = input_dim
        self.feedward_dim = feedward_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            DiffAttentionEncoderLayer(
                input_dim=input_dim,
                feedward_dim=feedward_dim,
                dropout=dropout,
                clamp_xi=clamp_xi,
                eps=eps,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [batch, days, input_dim]
        返回: [batch, days, input_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x

# 时序聚合模块,实现尺度缩放
class DaysFusion(nn.Module):
    def __init__(self, group_days): #  group_days是融合的天数
        super(DaysFusion, self).__init__()
        assert group_days > 0
        self.group_days = group_days
    
    def forward(self, days_emb):
        assert days_emb.dim() == 3 # [batch, days, dim]
        batch_size, days, day_dim = days_emb.size()

        x = days_emb
        if days % self.group_days != 0: # 补足到 group_days 的倍数
            pad_len = self.group_days - (days % self.group_days)
            pad = x[:, -1:, :].expand(batch_size, pad_len, day_dim)
            x = torch.cat([x, pad], dim=1)

        new_days = x.size(1) // self.group_days
        x = x.view(batch_size, new_days, self.group_days, day_dim)
        output = x.mean(dim=2)

        return output

# 尺度聚合模块
class ScalesPool(nn.Module):
    def __init__(self, day_dim):
        super(ScalesPool, self).__init__()

        self.input_dim = day_dim 
        self.scorer = nn.Linear(self.input_dim, 1, bias = False)
    
    def forward(self, days_emb): # [batch,days,day_dim]
        scores = self.scorer(days_emb).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        output = torch.sum(days_emb * weights.unsqueeze(-1), dim=1)

        return output # [batch,day_dim]

# 封装后的block
class NutBlock(nn.Module):
    def __init__(self,input_dim, num_heads, feedward_dim, dropout, attn_num_layers, group_days):
        super(NutBlock,self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.feedward_dim = feedward_dim
        self.dropout = dropout
        self.num_layers = attn_num_layers
        self.group_days = group_days

        self.diffattn = DiffAttentionEncoder(self.input_dim)
        self.attn = AttentionEncoder(self.input_dim, self.num_heads, self.feedward_dim, self.dropout, self.num_layers)

        self.fusion = DaysFusion(self.group_days)

        self.weight_diff = nn.Linear(self.input_dim, self.input_dim)
        self.weight_attn = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, input_emb):
        diff = self.diffattn(input_emb)
        attn = self.attn(input_emb)

        out_diff = self.weight_diff(diff)
        out_attn = self.weight_attn(attn)

        x = out_diff + out_attn

        output = self.fusion(x)

        return output
# block_No_diffattn
class NutBlock_Attn(nn.Module):
    def __init__(self,input_dim, num_heads, feedward_dim, dropout, attn_num_layers, group_days):
        super(NutBlock_Attn,self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.feedward_dim = feedward_dim
        self.dropout = dropout
        self.num_layers = attn_num_layers
        self.group_days = group_days
        self.attn = AttentionEncoder(self.input_dim, self.num_heads, self.feedward_dim, self.dropout, self.num_layers)
        self.weight_attn = nn.Linear(self.input_dim, self.input_dim)
        self.fusion = DaysFusion(self.group_days)

    def forward(self, input_emb):
        
        attn = self.attn(input_emb)
        out_attn = self.weight_attn(attn)
        output = self.fusion(out_attn)
        return output

class NutBlock_Diff(nn.Module):
    def __init__(self,input_dim, num_heads, feedward_dim, dropout, attn_num_layers, group_days):
        super(NutBlock_Diff,self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.feedward_dim = feedward_dim
        self.dropout = dropout
        self.num_layers = attn_num_layers
        self.group_days = group_days
        self.diffattn = DiffAttentionEncoder(self.input_dim)
        self.weight_diff = nn.Linear(self.input_dim, self.input_dim)
        self.fusion = DaysFusion(self.group_days)

    def forward(self, input_emb):
        
        diff = self.diffattn(input_emb)
        out_diff = self.weight_diff(diff)
        output = self.fusion(out_diff)
        return output

class NutBlock_B(nn.Module):
    def __init__(self,input_dim, num_heads, feedward_dim, dropout, attn_num_layers, group_days):
        super(NutBlock_B,self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.feedward_dim = feedward_dim
        self.dropout = dropout
        self.num_layers = attn_num_layers
        self.group_days = group_days

        self.diffattn = DiffAttention(self.input_dim)
        self.attn = AttentionEncoder(self.input_dim, self.num_heads, self.feedward_dim, self.dropout, self.num_layers)

        self.fusion_diff = DaysFusion(self.group_days)
        self.fusion_attn = DaysFusion(self.group_days)

        self.weight_diff = nn.Linear(self.input_dim, self.input_dim)
        self.weight_attn = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, input_diff,input__attn):
        diff = self.diffattn(input_diff)
        attn = self.attn(input__attn)

        out_diff = self.weight_diff(diff)
        out_attn = self.weight_attn(attn)

        fin_diff = self.fusion_diff(out_diff)
        fin_attn = self.fusion_attn(out_attn)

        return fin_diff,fin_attn

class DayPositionalEncoding(nn.Module):
    """
    可学习的 days 维位置编码，使用 nn.Embedding
    输入/输出: [B, days, F]
    """

    def __init__(self, input_dim, max_days):
        """
        初始化DayPositionalEncoding模块
        
        Args:
            config: 配置对象，包含所有模型参数
        """
        super(DayPositionalEncoding, self).__init__()
        
        # ==================== 基本配置参数 ====================
        self.input_dim = input_dim
        self.max_days = int(max_days)
        
        self.emb = nn.Embedding(self.max_days, self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, days, F]
        assert x.dim() == 3 and x.size(-1) == self.input_dim, 'x 形状应为 [B, days, F] 且 F==input_dim'
        batch_size, days, feat = x.size()
        if days > self.max_days:
            raise ValueError(f"days 超过 max_days: {days} > {self.max_days}")
        device = x.device
        positions = torch.arange(days, device=device, dtype=torch.long)  # [days]
        pe = self.emb(positions)  # [days, F]
        return x + pe.unsqueeze(0)  # [B, days, F]




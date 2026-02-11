import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from compare_model.multi.model import StaticFusion, DaynamicFusion, BCEWithLogitsLossWithLabelSmoothing
import os
import numpy as np
"""
Refactored multimodal fusion modules (cleaned, torch>=1.10 compatible).
- 移除 Variable/cuda() 直接调用；自动使用输入张量 device/dtype。
- 统一使用 dim 参数的 softmax；替换 F.tanh/F.sigmoid 为 torch.tanh/torch.sigmoid（或nn模块）。
- 修复若干潜在 bug（如 multiplication 模型中融合被覆盖、Conv1d padding 非整数等）。
- 增加类型注解与更易读的结构/注释。
- 避免在 __init__ 中使用 .cuda()；参数初始化用 xavier_normal_。

NOTE: 这些模块保留了与原始代码基本一致的 API（大多返回 (y, weights或占位)）。
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Encoders
# -------------------------

class EncoderA(nn.Module):
    """MLP encoder used for audio/video (pre-fusion)."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size * 5)
        self.linear_2 = nn.Linear(hidden_size * 5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = self.drop(F.relu(self.linear_1(x)))
        x = self.drop(F.relu(self.linear_2(x)))
        x = torch.tanh(self.linear_3(x))
        return x


class Encoder5(nn.Module):
    """Deeper encoder variant with intermediate norms."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm_h = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(in_size * 10)
        self.norm3 = nn.BatchNorm1d(hidden_size * 10)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size * 10)
        self.linear_2 = nn.Linear(in_size * 10, hidden_size * 10)
        self.linear_3 = nn.Linear(hidden_size * 10, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = F.leaky_relu(self.norm2(self.drop(self.linear_1(x))))
        y2 = F.leaky_relu(self.norm3(self.drop(self.linear_2(y1))))
        y2 = F.leaky_relu(self.norm_h(self.drop(self.linear_3(y2))))
        y3 = torch.tanh(self.linear_4(y2))
        return y3


class EncoderV(EncoderA):
    """Alias of EncoderA for clarity (video)."""
    pass


class EncoderL3(nn.Module):
    """Text encoder: 1D conv-gating + MLP."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size * 5)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size * 5, hidden_size * 5)
        self.linear_2 = nn.Linear(hidden_size * 5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.gates = nn.Conv1d(1, 5, kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_size) -> (B, 1, in_size)
        x = x.unsqueeze(1)
        x = self.gates(x)               # (B, 5, in_size)
        x = x.view(x.shape[0], -1)      # (B, 5 * in_size)
        x = self.drop(self.norm(x))
        x = self.drop(F.relu(self.linear_1(x)))
        x = self.drop(F.relu(self.linear_2(x)))
        x = torch.tanh(self.linear_3(x))
        return x


class EncoderL(nn.Module):
    """LSTM-based text encoder."""

    def __init__(self, in_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_size)
        _, (h_n, _) = self.rnn(x)
        h = self.dropout(h_n[-1])  # (B, H)
        y = torch.tanh(self.linear_1(h))
        return y


# -------------------------
# Decoders / Discriminators / Classifiers
# -------------------------

class Decoder2(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img_flat = self.model(z)
        return img_flat.view(img_flat.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Classifier2(nn.Module):
    def __init__(self, in_size: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, output_dim * 10)
        self.linear_2 = nn.Linear(output_dim * 10, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = F.relu(self.linear_1(x))
        x = F.softmax(self.linear_2(x), dim=1)
        return x


class Classifier3(nn.Module):
    def __init__(self, in_size: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size)
        self.linear_2 = nn.Linear(in_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = self.drop(torch.tanh(self.linear_1(x)))
        x = F.softmax(self.linear_2(x), dim=1)
        return x


# -------------------------
# Fusion variants
# -------------------------

class Graph11New(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.drop = nn.Dropout(p=dropout)

        def fusion_block():
            return nn.Sequential(
                nn.Linear(in_size * 2, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, in_size),
                nn.Tanh(),
            )

        self.graph_fusion = fusion_block()
        self.graph_fusion2 = fusion_block()
        self.attention = nn.Linear(in_size, 1)

        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

        self.in_size = in_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        # Unimodal attention
        sa = torch.sigmoid(self.attention(a1))  # (B,1)
        sv = torch.sigmoid(self.attention(v1))
        sl = torch.sigmoid(self.attention(l1))
        total_weights = torch.cat([sa, sv, sl], dim=1)

        unimodal_a = sa.expand_as(a1)
        unimodal_v = sv.expand_as(v1)
        unimodal_l = sl.expand_as(l1)
        unimodal = (unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1) / 3

        # Bimodal norms
        a = F.softmax(a1, dim=1)
        v = F.softmax(v1, dim=1)
        l = F.softmax(l1, dim=1)

        sav = (1 / (torch.bmm(a.unsqueeze(1), v.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sa + sv)).squeeze(-1)
        sal = (1 / (torch.bmm(a.unsqueeze(1), l.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sa + sl)).squeeze(-1)
        svl = (1 / (torch.bmm(v.unsqueeze(1), l.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sl + sv)).squeeze(-1)

        normalize = torch.stack([sav, sal, svl], dim=1)  # (B,3)
        normalize = F.softmax(normalize, dim=1)
        total_weights = torch.cat([total_weights, normalize], dim=1)

        a_v = F.elu(normalize[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, v1], dim=1)))
        a_l = F.elu(normalize[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, l1], dim=1)))
        v_l = F.elu(normalize[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v1, l1], dim=1)))
        bimodal = a_v + a_l + v_l

        # Trimodal
        a_v2 = F.softmax(self.graph_fusion(torch.cat([a1, v1], dim=1)), dim=1)
        a_l2 = F.softmax(self.graph_fusion(torch.cat([a1, l1], dim=1)), dim=1)
        v_l2 = F.softmax(self.graph_fusion(torch.cat([v1, l1], dim=1)), dim=1)

        def sp(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a_.unsqueeze(1), b_.unsqueeze(2)).squeeze(-1).squeeze(-1)

        savvl = (1 / (sp(a_v2, v_l2) + 0.5) * (sav + svl))
        saavl = (1 / (sp(a_v2, a_l2) + 0.5) * (sav + sal))
        savll = (1 / (sp(a_l2, v_l2) + 0.5) * (sal + svl))
        savl = (1 / (sp(a_v2, l) + 0.5) * (sav + sl.squeeze(-1)))
        salv = (1 / (sp(a_l2, v) + 0.5) * (sal + sv.squeeze(-1)))
        svla = (1 / (sp(v_l2, a) + 0.5) * (sa.squeeze(-1) + svl))

        normalize2 = torch.stack([savvl, saavl, savll, savl, salv, svla], dim=1)
        normalize2 = F.softmax(normalize2, dim=1)
        total_weights = torch.cat([total_weights, normalize2], dim=1)

        avvl = F.elu(normalize2[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, v_l], dim=1)))
        aavl = F.elu(normalize2[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, a_l], dim=1)))
        avll = F.elu(normalize2[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([v_l, a_l], dim=1)))
        avl = F.elu(normalize2[:, 3].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, l1], dim=1)))
        alv = F.elu(normalize2[:, 4].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_l, v1], dim=1)))
        vla = F.elu(normalize2[:, 5].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([v_l, a1], dim=1)))

        trimodal = avvl + aavl + avll + avl + alv + vla

        fusion = torch.cat([unimodal, bimodal, trimodal], dim=1)
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, total_weights


class Concat(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        fusion = torch.cat([a1, v1, l1], dim=1)
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y  # 第二个返回项保留占位以兼容旧接口


class Multiplication(nn.Module):
    """Elementwise multiplicative fusion.
    原代码中 `fusion = a1*v1` 后又被 `fusion = v1*l1` 覆盖，这里修正为三者逐元素乘。
    """

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        fusion = a1 * v1 * l1  # 修正覆盖问题
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y


class TensorFusion(nn.Module):
    """Full outer-product (tensor) fusion."""

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        B, D = a1.size(0), a1.size(1)
        device, dtype = a1.device, a1.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a1], dim=1)
        _v = torch.cat([ones, v1], dim=1)
        _l = torch.cat([ones, l1], dim=1)

        fusion_tensor = torch.bmm(_a.unsqueeze(2), _v.unsqueeze(1))  # (B, D+1, D+1)
        fusion_tensor = fusion_tensor.view(-1, (D + 1) * (D + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _l.unsqueeze(1)).view(B, -1)

        x = self.post_fusion_dropout(fusion_tensor)
        x = F.relu(self.post_fusion_layer_1(x))
        x = F.relu(self.post_fusion_layer_2(x))
        y = F.softmax(self.post_fusion_layer_3(x), dim=1)
        return y, y


class LowRankFusion(nn.Module):
    """Low-rank outer-product fusion (LMF)."""

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5, rank: int = 4):
        super().__init__()
        self.in_size = in_size
        self.output_dim = output_dim
        self.rank = rank

        # factors: (R, D+1, C)
        self.audio_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.video_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.text_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))

        # init
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.video_factor)
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        B, D = a1.size(0), a1.size(1)
        device, dtype = a1.device, a1.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a1], dim=1)  # (B, D+1)
        _v = torch.cat([ones, v1], dim=1)
        _l = torch.cat([ones, l1], dim=1)

        # (B, D+1, C) after multiplying by factors (R, D+1, C) per rank -> broadcast via einsum
        fa = torch.einsum('bd,rdc->brc', _a, self.audio_factor)
        fv = torch.einsum('bd,rdc->brc', _v, self.video_factor)
        fl = torch.einsum('bd,rdc->brc', _l, self.text_factor)

        fzy = fa * fv * fl  # (B, R, C)
        out = torch.einsum('brc,1r->bc', fzy, self.fusion_weights) + self.fusion_bias  # (B, C)
        return out, out


class LateFusion(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        a1 = self.norm2(a1)
        v1 = self.norm2(v1)
        l1 = self.norm2(l1)

        a = torch.tanh(self.attention(a1))  # (B,1)
        v = torch.tanh(self.attention(v1))
        l = torch.tanh(self.attention(l1))

        w = torch.cat([a, v, l], dim=1)
        w = F.softmax(w, dim=1)  # (B,3)

        fusion = (
            w[:, 0].unsqueeze(1).expand_as(a1) * a1
            + w[:, 1].unsqueeze(1).expand_as(v1) * v1
            + w[:, 2].unsqueeze(1).expand_as(l1) * l1
        )

        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y


class Graph12(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.drop = nn.Dropout(p=dropout)
        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh(),
        )
        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.in_size = in_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        sa = torch.tanh(self.attention(a1))  # (B,1)
        sv = torch.tanh(self.attention(v1))
        sl = torch.tanh(self.attention(l1))

        w = torch.cat([sa, sv, sl], dim=1)
        w = F.softmax(w, dim=1)
        sa, sv, sl = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        total_weights = w

        unimodal = (
            sa.expand_as(a1) * a1 + sv.expand_as(v1) * v1 + sl.expand_as(l1) * l1
        ) / 3

        a = F.softmax(a1, dim=1).unsqueeze(1)
        v = F.softmax(v1, dim=1).unsqueeze(2)
        l = F.softmax(l1, dim=1).unsqueeze(2)

        sav = (1 / (torch.bmm(a, v).squeeze(-1).squeeze(-1) + 0.5) * (sa.squeeze(-1) + sv.squeeze(-1)))
        sal = (1 / (torch.bmm(a, l).squeeze(-1).squeeze(-1) + 0.5) * (sa.squeeze(-1) + sl.squeeze(-1)))
        svl = (1 / (torch.bmm(v.squeeze(2).unsqueeze(1), l).squeeze(-1).squeeze(-1) + 0.5) * (sl.squeeze(-1) + sv.squeeze(-1)))

        norm_bi = torch.stack([sav, sal, svl], dim=1)
        norm_bi = F.softmax(norm_bi, dim=1)
        total_weights = torch.cat([total_weights, norm_bi], dim=1)

        a_v = F.leaky_relu(norm_bi[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, v1], dim=1)))
        a_l = F.leaky_relu(norm_bi[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, l1], dim=1)))
        v_l = F.leaky_relu(norm_bi[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v1, l1], dim=1)))
        bimodal = (a_v + a_l + v_l) / 3

        a_v2 = F.softmax(a_v, dim=1).unsqueeze(1)
        a_l2 = F.softmax(a_l, dim=1).unsqueeze(2)
        v_l2 = F.softmax(v_l, dim=1).unsqueeze(2)

        def sp(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a_, b_).squeeze(-1).squeeze(-1)

        savvl = (1 / (sp(a_v2, v_l2) + 0.5) * (sav + svl))
        saavl = (1 / (sp(a_v2, a_l2) + 0.5) * (sav + sal))
        savll = (1 / (sp(a_l2.transpose(1, 2), v_l2) + 0.5) * (sal + svl))
        savl = (1 / (sp(a_v2, l) + 0.5) * (sav + sl.squeeze(-1)))
        salv = (1 / (sp(a_l2.transpose(1, 2), v) + 0.5) * (sal + sv.squeeze(-1)))
        svla = (1 / (sp(v_l2.transpose(1, 2), a) + 0.5) * (sa.squeeze(-1) + svl))

        norm_tri = torch.stack([savvl, saavl, savll, savl, salv, svla], dim=1)
        norm_tri = F.softmax(norm_tri, dim=1)
        total_weights = torch.cat([total_weights, norm_tri], dim=1)

        avvl = F.leaky_relu(norm_tri[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, v_l], dim=1)))
        aavl = F.leaky_relu(norm_tri[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, a_l], dim=1)))
        avll = F.leaky_relu(norm_tri[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v_l, a_l], dim=1)))
        avl = F.leaky_relu(norm_tri[:, 3].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, l1], dim=1)))
        alv = F.leaky_relu(norm_tri[:, 4].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_l, v1], dim=1)))
        vla = F.leaky_relu(norm_tri[:, 5].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v_l, a1], dim=1)))

        trimodal = (avvl + aavl + avll + avl + alv + vla) / 6

        fusion = torch.cat([unimodal, bimodal, trimodal], dim=1)
        fusion = self.drop(self.norm2(fusion))
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, total_weights


class OuterProduct(nn.Module):
    """Full outer-product fusion with post-MLP for multi-class classification.

    Returns raw logits suitable for nn.CrossEntropyLoss.
    """

    def __init__(self, in_size: int=32, output_dim: int = 32, hidden: int = 50, dropout: float = 0.5, use_softmax: bool = False):
        super().__init__()
        self.audio_in = in_size
        self.video_in = in_size
        self.text_in = in_size
        self.audio_hidden = hidden
        self.output_dim = output_dim
        self.use_softmax = use_softmax

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, a: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        t = t[:, 0, :]
        B, D = a.size(0), a.size(1)
        device, dtype = a.device, a.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a], dim=1)
        _v = torch.cat([ones, v], dim=1)
        _t = torch.cat([ones, t], dim=1)

        fusion = torch.bmm(_a.unsqueeze(2), _v.unsqueeze(1))
        fusion = fusion.view(-1, (D + 1) * (D + 1), 1)
        fusion = torch.bmm(fusion, _t.unsqueeze(1)).view(B, -1)

        y = self.post_fusion_dropout(fusion)
        y = F.relu(self.post_fusion_layer_1(y))
        y = F.relu(self.post_fusion_layer_2(y))
        # return latent features of size `output_dim` (e.g., 32); final classification done by self.fc outside
        y = self.post_fusion_layer_3(y)
        return y


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1

class ARGFModel(nn.Module):
    def __init__(self, config, hidden_size=32, hidden_size_d=32, num_layers=3, output_size=3):
        super(ARGFModel, self).__init__()
        # device
        use_gpu = bool(getattr(config, 'use_gpu', True)) and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.info_dim = config.info_dim
        self.exam_dim = config.exam_dim
        self.intake_dim = config.intake_dim
        self.hidden_size = hidden_size
        self.hidden_size_d = hidden_size_d
        self.num_layers = num_layers
        # Static
        static_path = 'compare_model/multi/adj_matrix2.pt'
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"Adjacency matrix not found: {static_path}")
        static_obj = torch.load(static_path, map_location='cpu')
        self.adj_matrix_s = self._move_to_device(static_obj, self.device)
        self.static_fusion = StaticFusion(self.info_dim, self.exam_dim, out_size=hidden_size_d)
        # Dynamic
        dynamic_path = 'compare_model/multi/adj_matrix.pt'
        if not os.path.exists(dynamic_path):
            raise FileNotFoundError(f"Adjacency matrix not found: {dynamic_path}")
        dynamic_obj = torch.load(dynamic_path, map_location='cpu')
        self.adj_matrix = self._move_to_device(dynamic_obj, self.device)
        self.dynamic_fusion = DaynamicFusion(self.intake_dim, hidden_size, hidden_size_d, num_layers)
        # Classification
        self.fc = nn.Linear(hidden_size_d, output_size)
        self.relu = nn.ReLU()
        self.lmf = OuterProduct()

    def _move_to_device(self, obj, device: torch.device):
        """Move adjacency data to device while preserving structure (dict/list/tensor)."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(x, device) for x in obj)
        if isinstance(obj, dict):
            moved = {}
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    moved[k] = v.to(device)
                elif isinstance(v, np.ndarray):
                    moved[k] = torch.from_numpy(v).to(device)
                elif isinstance(v, (list, tuple)):
                    moved[k] = type(v)(self._move_to_device(x, device) for x in v)
                else:
                    moved[k] = v
            return moved
        return obj

    def forward(self, info, exam, intake):
        info, exam = self.static_fusion(info, exam, self.adj_matrix_s)
        sta_f  = torch.stack([info, exam], dim=1)
        # dynamic
        ni_f = self.dynamic_fusion(intake, self.adj_matrix)
        ni_f = ni_f.unsqueeze(1)  # [b, 32]
        # fusion
        out = self.lmf(info.squeeze(), exam.squeeze(), ni_f)

        out = self.fc(out)
        # return 3-class logits
        return out
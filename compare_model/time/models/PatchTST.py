import torch
from torch import nn
from compare_model.time.layers.Transformer_EncDec import Encoder, EncoderLayer
from compare_model.time.layers.SelfAttention_Family import FullAttention, AttentionLayer
from compare_model.time.layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    """封装 transpose，contiguous=True 时返回内存连续的转置结果"""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    """简单分类头：Flatten + Linear + Dropout"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)      # 展平最后两维
        self.linear = nn.Linear(nf, target_window)   # 将特征映射到输出长度
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchTSTModel(nn.Module):
    """
    PatchTST 用于时间序列分类的实现。
    参考论文: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: 每个 patch 的长度
        stride: patch 之间的步幅
        """
        super(PatchTSTModel, self).__init__()
        self.seq_len = configs.seq_len
        padding = stride  # 与原实现保持一致的 padding

        # Patch 划分 + 线性嵌入，将时间序列切成重叠片段后映射到 d_model
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Transformer 编码器，由多层 EncoderLayer 组成
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            # 使用 BatchNorm1d 的前后转置封装，适配形状 [bs*vars, len, d_model]
            norm_layer=nn.Sequential(Transpose(1, 2),
                                     nn.BatchNorm1d(configs.d_model),
                                     Transpose(1, 2))
        )

        # 分类头
        # head_nf = d_model * patch_num；patch_num 由 (seq_len - patch_len) / stride + 2 估算
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        # 最终映射到类别数 model_output_dim
        self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.model_output_dim)

    def classification(self, x_enc):
        """
        对输入的时间序列做归一化、patch 化、Transformer 编码，再展平做线性分类。
        输入 x_enc: [bs, seq_len, enc_in]
        输出: [bs, num_classes]
        """
        # 非平稳性归一化（减均值、除方差）
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 重排为 [bs, nvars, seq_len]，便于按变量拆分 patch
        x_enc = x_enc.permute(0, 2, 1)
        # patch_embedding 输出 enc_out: [bs*nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # 经过多层 Transformer Encoder
        enc_out, attns = self.encoder(enc_out)
        # 还原为 [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 转为 [bs, nvars, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 展平 patch 与特征维度，做 dropout + 线性分类
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, pe_info, ph_exam, x_enc, padding_mask=None):
        # 只使用 x_enc 进行时间序列分类，返回 logits
        dec_out = self.classification(x_enc)
        return dec_out  # [B, num_classes]
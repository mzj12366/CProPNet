from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Attention

class UserEncoder(nn.Module):

    def __init__(self, config):
        super(UserEncoder, self).__init__()

        # 维度
        self.info_dim = config.info_dim
        self.exam_dim = config.exam_dim
        self.info_emb_dim = config.info_emb_dim
        self.exam_emb_dim = config.exam_emb_dim
        self.sym_dim = config.sym_dim
        self.sym_num = config.sym_num
        self.output_sym_dim = self.sym_dim * 2
        self.attention_dim = config.attention_dim

        # 编码器
        self.info_layers = nn.Linear(self.info_dim, self.info_emb_dim, bias = True)
        self.exam_layers = nn.Linear(self.exam_dim, self.exam_emb_dim, bias = True)

        self.one_invi_emb = nn.Parameter(torch.randn(1, self.info_emb_dim)) # 隐形表征

        # 症状生成层：输入应为拼接后的嵌入维度（info_emb_dim + exam_emb_dim），而不是原始特征维度
        self.sym_layers = nn.ModuleList([
            nn.Linear(self.info_emb_dim + self.exam_emb_dim, self.sym_dim, bias=True)
            for _ in range(self.sym_num)
        ])

        self.info_sym_attention = Attention(self.sym_dim, self.attention_dim)
        self.invi_sym_attention = Attention(self.sym_dim, self.attention_dim)
    
    def symptom_disentangle(self, sym_num, input_emb): # [batch,info_emb_dim + exam_emb_dim]->[batch,sym_num,sym_dim]

        sym_output = []
        for i in range(sym_num):
            sym_emb = F.relu(self.sym_layers[i](input_emb), inplace = True)
            sym_emb_exp = sym_emb.unsqueeze(1)
            sym_output.append(sym_emb_exp)

        sym_output = torch.cat(sym_output, dim = 1)
        return sym_output

    def similarity_compute(self, info, exam):
        """
        计算个人信息和体检信息的相似度
        
        输入参数:
            info: 个人信息嵌入 [batch_size, intent_embedding_dim]
            exam: 体检信息嵌入 [batch_size, intent_embedding_dim]
            
        输出:
            info_exam_similarity: 信息相似度 [batch_size]
        """
        cosine_similarity = F.cosine_similarity(info, exam, dim=1)             
        info_exam_similarity = (cosine_similarity + 1) / 2.0  # 将相似度映射到[0,1]范围
        return info_exam_similarity    
    
    def forward(self, info, exam):
        batch_size = info.size(0)
        info_emb = self.info_layers(info)
        exam_emb = self.exam_layers(exam)
        invi_emb = self.one_invi_emb.expand(batch_size, self.info_emb_dim)

        info_exam_emb = torch.cat([info_emb,exam_emb],dim=1)
        invi_exam_emb = torch.cat([invi_emb,exam_emb],dim=1)
        # 挖掘症状，参数相同
        k = self.sym_num
        info_k_sym_emb = self.symptom_disentangle(k,info_exam_emb)
        invi_k_sym_emb = self.symptom_disentangle(k,invi_exam_emb)
        # 症状注意力,聚合症状
        info_sym_emb = self.info_sym_attention(info_k_sym_emb)
        invi_sym_emb = self.invi_sym_attention(invi_k_sym_emb)
        # 相似度
        info_invi_similarity = self.similarity_compute(info_sym_emb,invi_sym_emb)

        # 症状融合
        sym_final_emb = torch.cat([info_sym_emb,invi_sym_emb * info_invi_similarity.unsqueeze(1)],dim=1)

        return sym_final_emb # [batch,output_sym_dim]

class UserEncoder_Nolearn(nn.Module):

    def __init__(self, config):
        super(UserEncoder_Nolearn, self).__init__()

        # 维度
        self.info_dim = config.info_dim
        self.exam_dim = config.exam_dim
        self.info_emb_dim = config.info_emb_dim
        self.exam_emb_dim = config.exam_emb_dim
        self.sym_dim = config.sym_dim
        self.sym_num = config.sym_num
        self.output_sym_dim = self.sym_dim * 2
        self.attention_dim = config.attention_dim

        # 编码器
        self.info_layers = nn.Linear(self.info_dim, self.info_emb_dim, bias = True)
        self.exam_layers = nn.Linear(self.exam_dim, self.exam_emb_dim, bias = True)

        # 移除可学习的隐形表征，改用 info 均分后的固定替代

        # 症状生成层：输入应为拼接后的嵌入维度（info_emb_dim + exam_emb_dim），而不是原始特征维度
        self.sym_layers = nn.ModuleList([
            nn.Linear(self.info_emb_dim + self.exam_emb_dim, self.sym_dim, bias=True)
            for _ in range(self.sym_num)
        ])

        self.info_sym_attention = Attention(self.sym_dim, self.attention_dim)
        self.invi_sym_attention = Attention(self.sym_dim, self.attention_dim)
    
    def symptom_disentangle(self, sym_num, input_emb): # [batch,info_emb_dim + exam_emb_dim]->[batch,sym_num,sym_dim]

        sym_output = []
        for i in range(sym_num):
            sym_emb = F.relu(self.sym_layers[i](input_emb), inplace = True)
            sym_emb_exp = sym_emb.unsqueeze(1)
            sym_output.append(sym_emb_exp)

        sym_output = torch.cat(sym_output, dim = 1)
        return sym_output

    def similarity_compute(self, info, exam):
        """
        计算个人信息和体检信息的相似度
        
        输入参数:
            info: 个人信息嵌入 [batch_size, intent_embedding_dim]
            exam: 体检信息嵌入 [batch_size, intent_embedding_dim]
            
        输出:
            info_exam_similarity: 信息相似度 [batch_size]
        """
        cosine_similarity = F.cosine_similarity(info, exam, dim=1)             
        info_exam_similarity = (cosine_similarity + 1) / 2.0  # 将相似度映射到[0,1]范围
        return info_exam_similarity    
    
    def forward(self, info, exam):
        batch_size = info.size(0)
        info_emb = self.info_layers(info)
        exam_emb = self.exam_layers(exam)
        # 将 info_emb 均分为两半：一半用于 info 分支，另一半替代原先可学习的 invi 分支
        half_dim = self.info_emb_dim // 2
        first_half, second_half = info_emb[:, :half_dim], info_emb[:, half_dim:half_dim*2]
        # 保持维度不变：各自用零填充另一半
        zeros = torch.zeros_like(first_half)
        info_emb_for_info = torch.cat([first_half, zeros], dim=1)
        invi_emb = torch.cat([zeros, second_half], dim=1)

        info_exam_emb = torch.cat([info_emb_for_info,exam_emb],dim=1)
        invi_exam_emb = torch.cat([invi_emb,exam_emb],dim=1)
        # 挖掘症状，参数相同
        k = self.sym_num
        info_k_sym_emb = self.symptom_disentangle(k,info_exam_emb)
        invi_k_sym_emb = self.symptom_disentangle(k,invi_exam_emb)
        # 症状注意力,聚合症状
        info_sym_emb = self.info_sym_attention(info_k_sym_emb)
        invi_sym_emb = self.invi_sym_attention(invi_k_sym_emb)
        # 相似度
        info_invi_similarity = self.similarity_compute(info_sym_emb,invi_sym_emb)

        # 症状融合
        sym_final_emb = torch.cat([info_sym_emb,invi_sym_emb * info_invi_similarity.unsqueeze(1)],dim=1)

        return sym_final_emb # [batch,output_sym_dim]
        

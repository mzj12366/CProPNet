from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from compare_model.multi.model import StaticFusion, DaynamicFusion, BCEWithLogitsLossWithLabelSmoothing
from compare_model.multi.my_model.Attention import FeatureFusionMultiHeadAttention
import os
import numpy as np
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

    
class SOLORModel(nn.Module):
    def __init__(self, config, hidden_size=32, hidden_size_d=32, num_layers=3, output_size=3):
        super(SOLORModel, self).__init__()
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
        self.lmf = FeatureFusionMultiHeadAttention()

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
        sta_f  = torch.cat([info, exam], dim=1)
        # dynamic
        ni_f = self.dynamic_fusion(intake, self.adj_matrix)
        ni_f = ni_f.unsqueeze(1)  # [b, 32]
        # fusion
        out = self.lmf(sta_f, ni_f)

        out = self.fc(out)
        # return 3-class logits
        return out

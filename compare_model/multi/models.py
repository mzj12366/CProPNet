from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from compare_model.multi.model import StaticFusion, DaynamicFusion, BCEWithLogitsLossWithLabelSmoothing
from compare_model.multi.LMF.LMF import LMF
import os
import numpy as np

class LMFModel(nn.Module):
    def __init__(self, config, hidden_size=32, hidden_size_d=32, num_layers=3, output_size=3):
        super(LMFModel, self).__init__()
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
        self.lmf = LMF()

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

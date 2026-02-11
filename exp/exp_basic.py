import os
import torch
from models.models import NutModel

class Exp_Basic(object):
    def __init__(self, config):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device_id)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def _get_data(self):
        pass

    def _train_model(self):
        pass

    def _test_model(self):
        pass

    def _vali_model(self):
        pass
    

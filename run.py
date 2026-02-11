from config import Config
from exp.exp_nut import Exp_Nut
import os
import random
import numpy as np
import torch


if __name__ =="__main__":
	config = Config().parse_argument()

	# 固定随机性与确定性设置
	seed = int(getattr(config, 'seed', 42))
	print(f"[Seed] Using random seed: {seed}")
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# 可选：严格确定性（可能降低速度，某些算子不支持）
	# torch.use_deterministic_algorithms(True)
	# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 或 ":4096:2"

	# 构建实验并训练
	exp = Exp_Nut(config)
	exp._train_model()

import torch
from config import Config
from data.dataset import Dataset_Smote, Dataset_Nut
from torch.utils.data import DataLoader
import numpy as np

def build_dataloader(config, flag_model="train", flag_set="Nut"):
	"""构建DataLoader
	
	参数:
		config: 配置对象
		flag_model: 模型阶段 ("train", "test")
		flag_set: 数据集类型 ("Nut", "Smote")
	
	返回:
		(data_set, data_loader): 原始数据集实例与对应的 DataLoader
	"""
	# 根据flag_model设置参数
	if flag_model == "test":
		shuffle = False
		drop_last = True
	else:
		shuffle = True
		drop_last = False
	
	# 根据flag_set选择数据集（将 flag_model 传入以使用内部划分）
	if flag_set == "Smote":
		data_set = Dataset_Smote(config=config, flag_model=flag_model)
	else:
		data_set = Dataset_Nut(config=config, flag_model=flag_model)
	
	# 固定 DataLoader 随机性
	g = torch.Generator()
	g.manual_seed(int(getattr(config, 'seed', 42)))
	data_loader = DataLoader(
		data_set,
		batch_size=config.batch_size,
		shuffle=shuffle,
		drop_last=drop_last,
		num_workers=config.num_workers,
		pin_memory=config.pin_memory,
		generator=g,
		worker_init_fn=lambda worker_id: np.random.seed(int(getattr(config, 'seed', 42)) + worker_id),
	)

	return data_set, data_loader
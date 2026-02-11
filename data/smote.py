import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE

# 保证以包形式导入项目
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from config import Config
from data.dataset import Origin_Smote


def main():
	# 解析配置（兼容 Jupyter 的未知参数）
	config = Config().parse_argument()

	# 构建扁平化时序的数据集（nu_inta 已展平为 [max_days*feat]）
	dataset = Origin_Smote(config)
	loader = DataLoader(dataset, batch_size=1, shuffle=False)

	# 聚合特征与标签：特征顺序 = 个人信息 + 体检 + 时序扁平
	X_list = []
	y_list = []
	for uid, pe_info, ph_exam, nu_inta_flat, label in loader:
		features = torch.cat([
			pe_info.squeeze(0),
			ph_exam.squeeze(0),
			nu_inta_flat.squeeze(0),
		], dim=-1)
		X_list.append(features.detach().cpu().numpy())
		y_list.append(int(label.item()))

	X = np.asarray(X_list, dtype=np.float32)
	y = np.asarray(y_list, dtype=np.int64)

	# 过采样
	smote = SMOTE(sampling_strategy={0: 100, 1: 100, 2: 100}, random_state=42)
	X_resampled, y_resampled = smote.fit_resample(X, y)

	print("Original dataset shape:", X.shape)
	print("Resampled dataset shape:", X_resampled.shape)

	# 确保保存目录存在（保存到 data_smote 下）
	base_dir = os.path.join(ROOT, 'data', 'data_smote')
	os.makedirs(base_dir, exist_ok=True)

	# 保存结果
	n = len(y_resampled)
	resampled_df = pd.DataFrame({'id': range(n), 'label': y_resampled})
	resampled_df.to_csv(os.path.join(base_dir, 'label_smote.csv'), index=False)
	np.save(os.path.join(base_dir, 'X_resampled.npy'), X_resampled)


if __name__ == "__main__":
	main()

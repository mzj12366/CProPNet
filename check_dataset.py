import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from config import Config
from data.data_loader import build_dataloader


def main():
	config = Config().parse_argument()

	# 固定顺序，便于一致性校验
	dl = build_dataloader(config, shuffle=False)
	dataset = dl.dataset

	assert len(dataset) > 0, "Dataset 为空"
	print(f"Total samples: {len(dataset)}")

	# 基本形状检查（取第一批）
	batch = next(iter(dl))
	uid_b, pe_info_b, ph_exam_b, nu_inta_b, labels_b = batch
	print("Batch shapes:")
	print(" uid:", tuple(uid_b.shape))
	print(" pe_info:", tuple(pe_info_b.shape))
	print(" ph_exam:", tuple(ph_exam_b.shape))
	print(" nu_inta:", tuple(nu_inta_b.shape))
	print(" labels:", tuple(labels_b.shape))

	# 单样本一致性检查（前 K 个）
	K = min(5, len(dataset))
	for i in range(K):
		uid, pe_info, ph_exam, nu_inta, label = dataset[i]
		assert uid == dataset.user_ids[i], f"uid 不一致: {uid} vs {dataset.user_ids[i]}"
		assert nu_inta.shape[0] == int(getattr(config, "max_days", 1)), "时序长度与 max_days 不一致"

		# 与底层 DataFrame 映射值一致（允许浮点误差）
		pi_vals = dataset.pi_by_user.loc[uid].to_numpy(dtype=np.float32)
		pe_vals = dataset.pe_by_user.loc[uid].to_numpy(dtype=np.float32)
		assert np.allclose(pe_info.numpy(), pi_vals, atol=1e-5), "pe_info 与源数据不一致"
		assert np.allclose(ph_exam.numpy(), pe_vals, atol=1e-5), "ph_exam 与源数据不一致"

		# 时序维度对齐（前 T 行应与原始序列一致，后面为 0 填充）
		if uid in dataset.nu_by_user.groups:
			nu_df = dataset.nu_by_user.get_group(uid).sort_values("Day")
			seq = nu_df[dataset.nu_features].to_numpy(dtype=np.float32)
			T = min(seq.shape[0], dataset.max_days)
			assert np.allclose(nu_inta[:T].numpy(), seq[:T], atol=1e-5), "nu_inta 前 T 步与源数据不一致"
			if T < dataset.max_days:
				assert np.allclose(nu_inta[T:].numpy(), 0.0, atol=1e-8), "nu_inta 填充应为 0"

		# 标签一致性
		label_src = int(dataset.labels_by_user.loc[uid])
		assert int(label.item()) == label_src, "label 与源数据不一致"

	print(f"前 {K} 个样本一致性检查通过 ✅")
	print("All checks passed ✅")


if __name__ == "__main__":
	main()



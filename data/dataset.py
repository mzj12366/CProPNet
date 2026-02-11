import os    
import numpy as np 
import pandas as pd 
import torch  
from torch.utils.data import Dataset
import argparse
from config import Config

# 本文件实现用于多模态分类任务的数据集类：
# - 四个 CSV 文件分别提供：标签、个人信息、体检指标、营养摄入时序
# - 通过 UserID 交集对齐样本，并对营养摄入按 Day 排序与定长填充
# - __getitem__ 返回 (id, pe_info, ph_exam, nu_inta, label)

class Dataset_Nut(Dataset):
	"""训练/评估阶段的数据集

	职责：
	- 读取四个 CSV（路径来源于 config）
	- 校验并对齐 UserID，确保样本四模态齐全
	- 将营养摄入时序按 Day 排序，并截断/填充至固定长度 max_days
	- 预先张量化，加速 __getitem__ 访问

	返回：
	- id: Python int（UserID）
	- pe_info: Tensor[personal_info_features]
	- ph_exam: Tensor[physical_exam_features]
	- nu_inta: Tensor[max_days, nutrient_features]
	- label: LongTensor[]
	"""
	def __init__(self, config, flag_model: str = 'train'):
		# 接受 argparse.Namespace 或任意具备属性访问的对象
		self.config = config
		self.flag_model = (flag_model or 'train').lower()
		self._load_dataframes()
		self._prepare_index_and_features()
		self._materialize_samples()
		self._build_split_indices()

	def _read_csv(self, path: str) -> pd.DataFrame:
		"""读取 CSV 并清理列名空白"""
		# Read CSV and strip whitespace from headers
		df = pd.read_csv(path)
		df.columns = [str(c).strip() for c in df.columns]
		return df

	def _load_dataframes(self) -> None:
		"""加载四个数据表并完成基础清洗/类型转换"""
		self.df_labels = self._read_csv(self.config.blood_glucose_change_csv)
		self.df_pi = self._read_csv(self.config.personal_info_csv)
		self.df_pe = self._read_csv(self.config.physical_exam_csv)
		self.df_nu = self._read_csv(self.config.nutrient_intake_csv)

		# Standardize key column names
		# 要求四表包含 UserID，且营养摄入包含 Day
		for df in (self.df_labels, self.df_pi, self.df_pe, self.df_nu):
			if "UserID" not in df.columns:
				raise ValueError("Expected 'UserID' column in all CSVs")

		if "Day" not in self.df_nu.columns:
			raise ValueError("Expected 'Day' column in nutrient intake CSV")

		# Ensure numeric types where applicable
		# 将数值列转为数值类型，不可解析置 NaN，随后填 0
		for df in (self.df_pi, self.df_pe):
			for c in df.columns:
				if c == "UserID":
					continue
				# Coerce errors to NaN then fill with 0
				df[c] = pd.to_numeric(df[c], errors="coerce")
			df.fillna(0, inplace=True)

		for c in self.df_nu.columns:
			if c in ("UserID", "Day"):
				continue
			self.df_nu[c] = pd.to_numeric(self.df_nu[c], errors="coerce")
		self.df_nu.fillna(0, inplace=True)

		self.df_labels["Blood_Glucose_Change"] = pd.to_numeric(
			self.df_labels["Blood_Glucose_Change"], errors="coerce"
		).fillna(0).astype(int)

	def _prepare_index_and_features(self) -> None:
		"""对齐 UserID 并缓存特征列、索引映射"""
		# Determine intersection of users available across all modalities
		ids_labels = set(self.df_labels["UserID"].astype(int).tolist())
		ids_pi = set(self.df_pi["UserID"].astype(int).tolist())
		ids_pe = set(self.df_pe["UserID"].astype(int).tolist())
		ids_nu = set(self.df_nu["UserID"].astype(int).tolist())
		common_ids = sorted(list(ids_labels & ids_pi & ids_pe & ids_nu))
		if len(common_ids) == 0:
			raise ValueError("No common UserID across all CSVs.")
		self.user_ids: list[int] = common_ids

		# Cache feature columns
		# 记录各模态的特征列，方便后续提取
		self.pi_features = [c for c in self.df_pi.columns if c != "UserID"]
		self.pe_features = [c for c in self.df_pe.columns if c != "UserID"]
		self.nu_features = [c for c in self.df_nu.columns if c not in ("UserID", "Day")]
		self.max_days: int = int(getattr(self.config, "max_days", 8))

		# Build fast lookup tables
		# 建立快速检索结构，按 UserID 定位每个用户的数据
		self.pi_by_user = self.df_pi.set_index("UserID")[self.pi_features]
		self.pe_by_user = self.df_pe.set_index("UserID")[self.pe_features]
		# Nutrient: group by user
		self.nu_by_user = self.df_nu.sort_values(["UserID", "Day"]).groupby("UserID")
		self.labels_by_user = self.df_labels.set_index("UserID")["Blood_Glucose_Change"]

	def _materialize_samples(self) -> None:
		"""预先构建四个模态张量与标签，便于高效索引"""
		# Precompute tensors for faster __getitem__
		pi_list: list[torch.Tensor] = []
		pe_list: list[torch.Tensor] = []
		nu_list: list[torch.Tensor] = []
		label_list: list[torch.Tensor] = []

		for uid in self.user_ids:
			# Personal info and physical exam
			# 取个人信息与体检为定长向量
			pi_vals = self.pi_by_user.loc[uid].to_numpy(dtype=np.float32)
			pe_vals = self.pe_by_user.loc[uid].to_numpy(dtype=np.float32)

			# Nutrient intake sequence
			# 取营养摄入为时序矩阵，按 Day 排序
			if uid in self.nu_by_user.groups:
				nu_df = self.nu_by_user.get_group(uid).sort_values("Day")
				seq = nu_df[self.nu_features].to_numpy(dtype=np.float32)
			else:
				seq = np.zeros((0, len(self.nu_features)), dtype=np.float32)

			# Truncate or pad to max_days
			# 截断/尾部 0 填充到固定步长 max_days
			T = min(seq.shape[0], self.max_days)
			seq_trunc = seq[:T]
			if T < self.max_days:
				pad_rows = self.max_days - T
				pad = np.zeros((pad_rows, seq.shape[1] if seq.shape[0] > 0 else len(self.nu_features)), dtype=np.float32)
				seq_padded = np.vstack([seq_trunc, pad])
			else:
				seq_padded = seq_trunc

			label = int(self.labels_by_user.loc[uid])

			pi_list.append(torch.from_numpy(pi_vals))
			pe_list.append(torch.from_numpy(pe_vals))
			nu_list.append(torch.from_numpy(seq_padded))
			label_list.append(torch.tensor(label, dtype=torch.long))

		self.pi_tensor = torch.stack(pi_list, dim=0)
		self.pe_tensor = torch.stack(pe_list, dim=0)
		self.nu_tensor = torch.stack(nu_list, dim=0)  # [N, T, F]
		self.label_tensor = torch.stack(label_list, dim=0)

	def _build_split_indices(self) -> None:
		"""按照 config.seed 与 train_ratio 进行可复现划分，并选择对应子集索引"""
		num_samples = len(self.user_ids)
		all_indices = np.arange(num_samples)
		seed = getattr(self.config, 'seed', None)
		seed = int(seed) if seed is not None else 42
		rng = np.random.default_rng(seed)
		rng.shuffle(all_indices)
		train_ratio = float(getattr(self.config, 'train_ratio', 0.8))
		split_idx = int(num_samples * train_ratio)
		train_indices = all_indices[:split_idx]
		test_indices = all_indices[split_idx:]

		is_test = self.flag_model.startswith('test') or self.flag_model.startswith('text')
		self.sel_indices = test_indices if is_test else train_indices

	def __len__(self) -> int:
		"""子集样本总数"""
		return int(len(self.sel_indices))

	def __getitem__(self, idx: int):
		"""返回第 idx 个样本的五元组 (id, pe_info, ph_exam, nu_inta, label)（划分后子集）"""
		true_idx = int(self.sel_indices[idx])
		uid = self.user_ids[true_idx]
		pe_info = self.pi_tensor[true_idx]
		ph_exam = self.pe_tensor[true_idx]
		nu_inta = self.nu_tensor[true_idx]
		label = self.label_tensor[true_idx]
		return uid, pe_info, ph_exam, nu_inta, label

class Dataset_Smote(Dataset):
	"""加载SMOTE处理后的数据，分割三模态特征
	
	输入：
	- label_smote.csv: 包含ID索引和分类标签
	- X_resampled.npy: 拼接特征 [N, personal_info_dim + physical_exam_dim + nutrient_feat_dim * nutrient_days]
	
	输出：
	- uid: 用户ID (int)
	- pe_info: 个人信息 [personal_info_dim]
	- ph_exam: 体检信息 [physical_exam_dim] 
	- nu_inta: 营养摄入时序 [nutrient_days, nutrient_feat_dim]
	- label: 分类标签 (int)
	"""
	
	def __init__(self, config, flag_model: str = 'train'):
		"""SMOTE 数据集，支持基于随机种子的 train/test 划分。

		flag_model: 'train' 或 'test'（兼容 'text' 写法）
		"""
		self.config = config
		self.flag_model = (flag_model or 'train').lower()
		self._load_smote_data()
		self._split_features()
		self._build_split_indices()
	
	def _load_smote_data(self) -> None:
		"""加载SMOTE处理后的标签和特征数据"""
		import pandas as pd
		import numpy as np
		
		# 加载标签
		self.labels_df = pd.read_csv(self.config.smote_label_csv)
		self.labels_df.columns = [str(c).strip() for c in self.labels_df.columns]
		
		# 加载特征
		self.features = np.load(self.config.smote_data_npy).astype(np.float32)
		
		# 验证数据一致性
		assert len(self.labels_df) == len(self.features), f"标签数量 {len(self.labels_df)} 与特征数量 {len(self.features)} 不匹配"
		
		# 获取维度参数
		self.personal_info_dim = self.config.personal_info_dim
		self.physical_exam_dim = self.config.physical_exam_dim
		self.nutrient_feat_dim = self.config.nutrient_feat_dim
		self.nutrient_days = self.config.nutrient_days
		
		# 验证特征总维度
		expected_total_dim = self.personal_info_dim + self.physical_exam_dim + (self.nutrient_feat_dim * self.nutrient_days)
		assert self.features.shape[1] == expected_total_dim, f"特征维度 {self.features.shape[1]} 与期望维度 {expected_total_dim} 不匹配"
	
	def _split_features(self) -> None:
		"""将拼接特征分割为三个模态"""
		# 计算分割点
		pi_end = self.personal_info_dim
		pe_end = pi_end + self.physical_exam_dim
		nu_end = pe_end + (self.nutrient_feat_dim * self.nutrient_days)
		
		# 分割特征
		self.pe_info_features = self.features[:, :pi_end]  # [N, personal_info_dim]
		self.ph_exam_features = self.features[:, pi_end:pe_end]  # [N, physical_exam_dim]
		self.nu_flat_features = self.features[:, pe_end:nu_end]  # [N, nutrient_feat_dim * nutrient_days]
		
		# 重塑营养摄入为时序格式（按行优先顺序：先第1天的所有特征，再第2天的所有特征...）
		self.nu_inta_features = self.nu_flat_features.reshape(-1, self.nutrient_days, self.nutrient_feat_dim)  # [N, nutrient_days, nutrient_feat_dim]
		
		# 转换为张量
		self.pe_info_tensor = torch.from_numpy(self.pe_info_features)
		self.ph_exam_tensor = torch.from_numpy(self.ph_exam_features)
		self.nu_inta_tensor = torch.from_numpy(self.nu_inta_features)
		self.label_tensor = torch.from_numpy(self.labels_df['label'].values.astype(np.int64))
		self.uid_tensor = torch.from_numpy(self.labels_df['id'].values.astype(np.int64))

	def _build_split_indices(self) -> None:
		"""按照 config.seed 与 train_ratio 进行可复现划分，并选择对应子集索引"""
		num_samples = len(self.labels_df)
		all_indices = np.arange(num_samples)
		seed = getattr(self.config, 'seed', None)
		seed = int(seed) if seed is not None else 42
		rng = np.random.default_rng(seed)
		rng.shuffle(all_indices)
		train_ratio = float(getattr(self.config, 'train_ratio', 0.8))
		split_idx = int(num_samples * train_ratio)
		train_indices = all_indices[:split_idx]
		test_indices = all_indices[split_idx:]

		is_test = self.flag_model.startswith('test') or self.flag_model.startswith('text')
		self.sel_indices = test_indices if is_test else train_indices
	
	def __len__(self) -> int:
		"""子集样本总数"""
		return int(len(self.sel_indices))
	
	def __getitem__(self, idx: int):
		"""返回第 idx 个样本的 (uid, pe_info, ph_exam, nu_inta, label)（基于划分后的子集）"""
		true_idx = int(self.sel_indices[idx])
		uid = int(self.uid_tensor[true_idx].item())
		pe_info = self.pe_info_tensor[true_idx]
		ph_exam = self.ph_exam_tensor[true_idx]
		nu_inta = self.nu_inta_tensor[true_idx]
		label = int(self.label_tensor[true_idx].item())
		return uid, pe_info, ph_exam, nu_inta, label


class Origin_Smote(Dataset_Nut):
	"""与 Dataset_Nut 输入输出一致，但将 nu_inta 改为二维扁平化：
	- nu_inta: Tensor[max_days * nutrient_features]
	- 其他保持不变：id, pe_info, ph_exam, label
	"""

	def __init__(self, config):
		# 复用父类的数据加载与索引准备逻辑
		super().__init__(config)

	def _materialize_samples(self) -> None:
		"""预先构建扁平化后的时序张量 nu_inta_flat: [N, max_days * R]"""
		pi_list: list[torch.Tensor] = []
		pe_list: list[torch.Tensor] = []
		nu_list_flat: list[torch.Tensor] = []
		label_list: list[torch.Tensor] = []

		feature_dim = len(self.nu_features)
		flat_dim = int(self.max_days) * int(feature_dim)

		for uid in self.user_ids:
			# 静态模态向量
			pi_vals = self.pi_by_user.loc[uid].to_numpy(dtype=np.float32)
			pe_vals = self.pe_by_user.loc[uid].to_numpy(dtype=np.float32)

			# 时序取值并定长（与父类一致）
			if uid in self.nu_by_user.groups:
				nu_df = self.nu_by_user.get_group(uid).sort_values("Day")
				seq = nu_df[self.nu_features].to_numpy(dtype=np.float32)
			else:
				seq = np.zeros((0, feature_dim), dtype=np.float32)

			T = min(seq.shape[0], self.max_days)
			seq_trunc = seq[:T]
			if T < self.max_days:
				pad_rows = self.max_days - T
				pad = np.zeros((pad_rows, feature_dim), dtype=np.float32)
				seq_padded = np.vstack([seq_trunc, pad])
			else:
				seq_padded = seq_trunc

			# 扁平化为一维，长度 = max_days * feature_dim
			nu_flat = seq_padded.reshape(-1).astype(np.float32)

			label = int(self.labels_by_user.loc[uid])

			pi_list.append(torch.from_numpy(pi_vals))
			pe_list.append(torch.from_numpy(pe_vals))
			nu_list_flat.append(torch.from_numpy(nu_flat))
			label_list.append(torch.tensor(label, dtype=torch.long))

		self.pi_tensor = torch.stack(pi_list, dim=0)
		self.pe_tensor = torch.stack(pe_list, dim=0)
		self.nu_tensor = torch.stack(nu_list_flat, dim=0)  # [N, max_days*R]
		self.label_tensor = torch.stack(label_list, dim=0)

	def __getitem__(self, idx: int):
		"""返回第 idx 个样本的 (id, pe_info, ph_exam, nu_inta_flat, label)"""
		uid = self.user_ids[idx]
		pe_info = self.pi_tensor[idx]
		ph_exam = self.pe_tensor[idx]
		nu_inta_flat = self.nu_tensor[idx]
		label = self.label_tensor[idx]
		return uid, pe_info, ph_exam, nu_inta_flat, label

import argparse
import os

class Config:

    def parse_argument(self):
        """解析命令行参数"""
        parser = argparse.ArgumentParser(description='Nut_Model')

        # 通用配置
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='运行模式')
        parser.add_argument('--device_id', type=int, default=0, help='GPU设备ID')
        parser.add_argument('--use_gpu', action='store_true', default=True, help='是否使用GPU')
        parser.add_argument('--seed', type=int, default=42, help='随机数种子（不指定则自动生成）')
        parser.add_argument('--auto_seed', action='store_true', default=True, help='自动生成随机种子（默认启用）')
        parser.add_argument('--config_file', type=str, default='', help='配置文件路径')
        parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集划分比例(0-1)')

        # 数据集参数
        # 原数据信息
        parser.add_argument('--blood_glucose_change_csv', type=str,default='data/data_origin/blood_glucose_change.csv',help='标签CSV路径')
        parser.add_argument('--personal_info_csv', type=str,default='data/data_origin/personal_info.csv',help='个人信息CSV路径')
        parser.add_argument('--physical_exam_csv', type=str,default='data/data_origin/physical_exam.csv',help='体检指标CSV路径')
        parser.add_argument('--nutrient_intake_csv', type=str,default='data/data_origin/nutrient_intake.csv',help='营养摄入CSV路径')
        parser.add_argument('--max_days', type=int, default=33, help='营养摄入时序长度（按 Day 截断/填充）')
        parser.add_argument('--normalize_features', action='store_true', default=False, help='是否对特征做归一化/标准化')
        parser.add_argument('--batch_size', type=int, default=32, help='训练/评估批大小')
        parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 工作进程数')
        parser.add_argument('--pin_memory', action='store_true', default=False, help='是否启用 pin_memory')
        # Smote后数据信息
        parser.add_argument('--smote_label_csv', type=str, default='data/data_smote/label_smote.csv', help='SMOTE标签CSV路径')
        parser.add_argument('--smote_data_npy', type=str, default='data/data_smote/X_resampled.npy', help='SMOTE特征NPY路径')
        parser.add_argument('--personal_info_dim', type=int, default=16, help='个人信息特征维度')
        parser.add_argument('--physical_exam_dim', type=int, default=16, help='体检信息特征维度')
        parser.add_argument('--nutrient_feat_dim', type=int, default=79, help='营养摄入单天特征维度')
        parser.add_argument('--nutrient_days', type=int, default=33, help='营养摄入天数')

        # 模型框架参数
        # UserEncoder
        parser.add_argument('--info_dim', type=int, default=16, help='个人信息原始特征维度')
        parser.add_argument('--exam_dim', type=int, default=16, help='体检信息原始特征维度')
        parser.add_argument('--info_emb_dim', type=int, default=128, help='个人信息嵌入维度')
        parser.add_argument('--exam_emb_dim', type=int, default=128, help='体检信息嵌入维度')
        parser.add_argument('--sym_dim', type=int, default=32, help='症状表征维度')
        parser.add_argument('--sym_num', type=int, default=6, help='症状生成模块数量')
        parser.add_argument('--attention_dim', type=int, default=32, help='症状注意力内部维度')
        # NutEncoder 超参数
        parser.add_argument('--intake_dim', type=int, default=79, help='营养摄入单天特征维度')
        parser.add_argument('--intake_emb_dim', type=int, default=128, help='营养摄入嵌入维度')
        parser.add_argument('--stage_num', type=int, default=5, help='NutEncoder阶段数量')
        parser.add_argument('--num_heads', type=int, default=2, help='多头注意力头数')
        parser.add_argument('--feedward_dim', type=int, default=256, help='前馈网络维度')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout比例')
        parser.add_argument('--attn_num_layers', type=int, default=1, help='注意力层数')
        parser.add_argument('--group_days', type=int, default=3, help='分组天数')
        parser.add_argument('--out_intake_dim', type=int, default=64, help='营养摄入输出维度')

        parser.add_argument('--mlp_dim', type=int, default=128, help='MLP维度')
        parser.add_argument('--model_output_dim', type=int, default=3, help='模型输出维度')

        # 实验/训练参数
        parser.add_argument('--optimizer', type=str, default='adam', choices=['adam','sgd'], help='优化器类型')
        parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
        parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
        parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
        # 早停
        parser.add_argument('--early_stop', action='store_true', default=False, help='是否启用早停')
        parser.add_argument('--patience', type=int, default=20, help='早停耐心值（无提升轮数上限）')
        parser.add_argument('--min_delta', type=float, default=0.0, help='认为有提升所需的最小改变量（针对val_loss减少）')

        # 对比试验
        parser.add_argument("--enc_in", type=int, default=79, help="通道数")
        parser.add_argument('--seq_len', type=int, default=33, help="输入序列长度")
        parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
        parser.add_argument("--d_ff", type=int, default=128, help="dimension of fcn")
        parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
        parser.add_argument("--e_layers", type=int, default=1, help="num of encoder layers")
        parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
        parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")
        parser.add_argument("--activation", type=str, default="gelu", help="activation")
        parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")
        parser.add_argument('--resolution_list', type=str, default="2,4,6,8")
        parser.add_argument('--nodedim', type=int, default=10)
        parser.add_argument("--patch_len_list", type=str, default="2,2,2,4,4,4,16,16,16,16,32,32,32,32,32", help="a list of patch len used in Medformer")
        parser.add_argument("--single_channel", action="store_true", default=False, help="whether to use single channel patching for Medformer")
        parser.add_argument("--augmentations", type=str, default="none,drop0.35",
                        help="a comma-seperated list of augmentation types (none, jitter or scale). Append numbers to specify the strength of the augmentation, e.g., jitter0.1")
        parser.add_argument("--no_inter_attn", action="store_true", default=False, help="disable inter-attention in Medformer")
        parser.add_argument("--freq", type=str, default="d",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h")
        parser.add_argument("--factor", type=int, default=3, help="iTransformer prob attention factor")
         # TimeNet超参数
        parser.add_argument("--top_k", type=int, default=2, help="top k frequency components")
        parser.add_argument("--num_kernels", type=int, default=2, help="number of kernels")
        parser.add_argument("--pred_len", type=int, default=0, help="prediction length")
        # TimeMixer超参数
        parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
        parser.add_argument("--down_sampling_window", type=int, default=1, help="down sampling window")
        parser.add_argument("--down_sampling_layers", type=int, default=1, help="down sampling layers")
        parser.add_argument("--down_sampling_method", type=str, default="max", help="down sampling method")
        parser.add_argument("--moving_avg", type=int, default=25, help="moving average")
        parser.add_argument("--use_norm", type=int, default=0, help="use normalization")
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
        
        parser.add_argument('--seg_len', type=int, default=0,
                        help='the length of segment-wise iteration of SegRNN')
        parser.add_argument('--c_out', type=int, default=7, help='output size')



        # GPU参数（预留）

        args, _ = parser.parse_known_args()
        return args


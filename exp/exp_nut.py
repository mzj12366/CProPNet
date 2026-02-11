import random
import torch
import torch.nn as nn
import torch.optim as optim

from exp.exp_basic import Exp_Basic
from models.models import NutModel, OnlyNutModel, OnlyUserModel, NutModel_Attn, NutModel_Nolearn, NutModel_B, NutModel_Diff
from data.data_loader import build_dataloader
from compare_model.multi.models import LMFModel
from compare_model.multi.argf import ARGFModel
from compare_model.multi.glmnet import GLMModel
from compare_model.multi.solor import SOLORModel
from compare_model.time.models.MedGNN import MedGNNModel
from compare_model.time.models.Medformer import MedformerModel
from compare_model.time.models.iTransformer import iTransformerModel
from compare_model.time.models.TimesNet import TimesNetModel
from compare_model.time.models.TimeMixer import TimeMixerModel
from compare_model.time.models.PatchTST import PatchTSTModel
from compare_model.time.models.DLinear import DLinearModel
from compare_model.time.models.ETSformer import ETSformerModel
from compare_model.time.models.Reformer import ReformerModel
class Exp_Nut(Exp_Basic):
    def __init__(self, config):
        super(Exp_Nut, self).__init__(config)

        # 可选：SW A（若未配置则忽略）
        self.swa = getattr(config, 'swa', False)
        if self.swa:
            from torch.optim import swa_utils
            self.swa_model = swa_utils.AveragedModel(self.model)

    def _build_model(self):
        model = NutModel_Nolearn(self.config)
        if getattr(self.config, 'use_multi_gpu', False) and self.config.use_gpu:
            model = nn.DataParallel(model, device_ids=getattr(self.config, 'device_ids', [self.config.device_id]))
        return model
    
    def _get_data(self, flag_model, flag_set):
#        random.seed(self.config.seed)
        data_set, data_loader = build_dataloader(self.config, flag_model, flag_set)
        return data_set, data_loader

    def _select_optimizer(self):
        # 简化：统一从 config 读取，并支持优化器选择
        opt_name = str(self.config.optimizer).lower()
        lr = float(self.config.learning_rate)
        wd = float(self.config.weight_decay)
        opts = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd),
            'sgd':  lambda: optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9),
        }
        return opts.get(opt_name, opts['adam'])()
    
    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion
    
    def _train_model(self):
        # 数据
        _, train_loader = self._get_data('train', 'Smote')
        _, test_loader = self._get_data('test', 'Smote')

        # 优化器与损失
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        epochs = int(self.config.epochs)
        device = self.device

        # 始终跟踪最佳权重；是否提前停止由配置控制
        best_val = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # 训练
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for uid, pe_info, ph_exam, nu_inta, labels in train_loader:
                pe_info = pe_info.to(device)
                ph_exam = ph_exam.to(device)
                nu_inta = nu_inta.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self.model(pe_info, ph_exam, nu_inta)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

            train_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)

            # 验证/测试
            self.model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_correct = 0
                val_samples = 0
                for uid, pe_info, ph_exam, nu_inta, labels in test_loader:
                    pe_info = pe_info.to(device)
                    ph_exam = ph_exam.to(device)
                    nu_inta = nu_inta.to(device)
                    labels = labels.to(device)

                    logits = self.model(pe_info, ph_exam, nu_inta)
                    loss = criterion(logits, labels)

                    bs = labels.size(0)
                    val_loss_sum += loss.item() * bs
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_samples += bs

                val_loss = val_loss_sum / max(val_samples, 1)
                val_acc = val_correct / max(val_samples, 1)

            print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

            # 更新最佳模型；如启用早停，则按耐心值提前结束
            if best_val - val_loss > float(self.config.min_delta):
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if getattr(self.config, 'early_stop', False) and epochs_no_improve >= int(self.config.patience):
                    print(f"Early stopping triggered after {epoch} epochs (best val_loss={best_val:.4f}).")
                    break

        # 使用最佳模型评估并打印完整指标
        try:
            if best_state is not None:
                self.model.load_state_dict(best_state, strict=True)
        except Exception:
            pass

        self.model.eval()
        K = int(getattr(self.config, 'model_output_dim', 3))
        cm = torch.zeros((K, K), dtype=torch.long)
        test_loss_sum = 0.0
        test_samples = 0
        with torch.no_grad():
            _, test_loader = self._get_data('test', 'Smote')
            for uid, pe_info, ph_exam, nu_inta, labels in test_loader:
                pe_info = pe_info.to(device)
                ph_exam = ph_exam.to(device)
                nu_inta = nu_inta.to(device)
                labels = labels.to(device)
                logits = self.model(pe_info, ph_exam, nu_inta)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    if 0 <= int(t) < K and 0 <= int(p) < K:
                        cm[int(t), int(p)] += 1
                test_loss_sum += loss.item() * labels.size(0)
                test_samples += labels.size(0)

        test_loss = test_loss_sum / max(test_samples, 1)
        acc = cm.diag().sum().item() / max(cm.sum().item(), 1)
        per_class_prec = []
        per_class_rec = []
        per_class_f1 = []
        for c in range(K):
            tp = cm[c, c].item()
            fp = cm[:, c].sum().item() - tp
            fn = cm[c, :].sum().item() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class_prec.append(prec)
            per_class_rec.append(rec)
            per_class_f1.append(f1)
        macro_prec = sum(per_class_prec) / K if K > 0 else 0.0
        macro_rec = sum(per_class_rec) / K if K > 0 else 0.0
        macro_f1 = sum(per_class_f1) / K if K > 0 else 0.0

        # 打印最终结果时附带模型名称（兼容 DataParallel 包装）
        model_name = getattr(self.model, 'module', self.model).__class__.__name__
        print(f"Best model metrics on test set (model={model_name}):")
        print(f" loss={test_loss:.4f} acc={acc:.4f} macroP={macro_prec:.4f} macroR={macro_rec:.4f} macroF1={macro_f1:.4f}")
        for c in range(K):
            support = cm[c, :].sum().item()
            print(f"  class {c}: P={per_class_prec[c]:.4f} R={per_class_rec[c]:.4f} F1={per_class_f1[c]:.4f} support={support}")
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
import numpy as np
from backbone_loader import *
from const.path import CAUSAL_OUT_PATH

#
# class LinearClassifierHead(nn.Module):
#     def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
#         super().__init__()
#         # This part correctly defines the layers in a list
#         dims = [input_dim, hidden_dims, num_classes]
#         self.fcs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(len(dims) - 1):
#             self.fcs.append(nn.Linear(dims[i], dims[i + 1]))
#             self.bns.append(nn.BatchNorm1d(dims[i + 1]))
#
#         # We don't need a separate self.out
#         self.dropout = nn.Dropout(dropout)
#         self.act = nn.ReLU()
#
#     def forward(self, x):
#         # The loop already handles all layers up to the final classification output
#         for fc, bn in zip(self.fcs, self.bns):
#             # The final layer's output should not have an activation function
#             # So, we need to handle the last layer separately
#
#             # This is a bit tricky with the for loop. A better way to structure is to handle layers manually.
#             x = self.act(self.bns[0](self.fcs[0](x)))
#             x = self.dropout(x)
#
#             # This handles the final layer. No ReLU here.
#             logits = self.fcs[1](x)
#
#         return logits

class MLPEncoder(nn.Module):
    """Multi-layer perceptron encoder for feature transformation"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLPEncoder, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, J, D] -> [B*T*J, D]
        B, T, J, D = x.shape
        x = x.reshape(-1, D)
        out = self.encoder(x)
        # Reshape back: [B*T*J, out_dim] -> [B, T, J, out_dim]
        out = out.reshape(B, T, J, -1)
        return out

class OrdinalHead(nn.Module):
    '''
    target = torch.zeros(y.size(0), K-1).to(y.device)
    for j in range(1, K):
        target[:, j-1] = (y >= j).float()
    loss = F.binary_cross_entropy_with_logits(logits, target)
    '''
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, num_classes - 1)  # K-1 logits

    def forward(self, x):
        feats = self.fc(x)
        logits = self.out(feats)  # [B, K-1]
        return logits


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)     # 前向传播不变
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None      # 反向传播时，接收到来自预测器的梯度，然后取反并且乘上系数lambd

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def get_different_label_shuffled_idx(labels):
    """
    为每个样本找到一个标签不同的样本索引进行交换。
    如果找不到，则返回原始索引，不进行交换。
    """
    batch_size = labels.size(0)
    shuffled_idx = torch.zeros_like(labels).long()

    # 将标签相同的样本分组，确保字典的键是整数
    unique_labels = torch.unique(labels)
    groups = {label.item(): (labels == label).nonzero(as_tuple=True)[0] for label in unique_labels}

    # 为每个样本找到一个不同标签的样本索引
    for i in range(batch_size):
        current_label = labels[i].item()

        # 找到所有标签不同的组，确保列表中的元素是整数
        other_labels = [label.item() for label in unique_labels if label.item() != current_label]

        # 如果存在不同标签的样本
        if other_labels:
            # 随机选择一个不同标签的组
            target_label = other_labels[torch.randint(0, len(other_labels), (1,)).item()]

            # 从该组中随机选择一个样本索引，此时 target_label 已经是整数
            target_indices = groups[target_label]
            shuffled_idx[i] = target_indices[torch.randint(0, len(target_indices), (1,)).item()]
        else:
            # 如果批次内所有样本标签都相同，则不进行交换，保持原样
            shuffled_idx[i] = i

    return shuffled_idx


# ----------------------------------------------------
# 核心工具类：Zc 特征记忆库 (Memory Bank for z_c)
# ----------------------------------------------------
class MemoryBank_Zc:
    def __init__(self, total_samples, z_dim, momentum=0.999, device='cuda'):
        self.momentum = momentum
        self.device = device
        self.total_samples = total_samples
        self.z_dim = z_dim

        # M_Z: 存储 Zc 特征，初始化为随机归一化向量
        self.M_Z = F.normalize(torch.randn(total_samples, z_dim, device=device), dim=1)
        # M_Y: 存储 Y 标签
        self.M_Y = torch.zeros(total_samples, dtype=torch.long, device=device)
        self.M_ID_MAP = {}  # {video_idx: memory_bank_index}
        self.filled_count = 0  # 记录当前实际填充的样本数

    def update(self, zc_norm, y, video_idx):
        """
        使用动量更新当前批次的特征到记忆库中
        """
        with torch.no_grad():
            for i in range(zc_norm.size(0)):
                idx = video_idx[i].item()  # 获取当前样本在 memory bank 中的全局索引

                # 获取 memory bank 索引
                if idx not in self.M_ID_MAP:
                    # 如果 Dataloader 保证了 video_idx 是 0 到 N-1 且只出现一次，
                    # 那么这里 bank_idx 应该就是 idx，但我们使用 M_ID_MAP 来处理不连续 ID 或其他情况
                    bank_idx = self.filled_count
                    if bank_idx < self.total_samples:
                        self.M_ID_MAP[idx] = bank_idx
                        self.M_Y[bank_idx] = y[i].item()
                        self.filled_count += 1
                    else:
                        # 如果 ID 超过了容量，不再加入，但通常不应该发生
                        continue

                bank_idx = self.M_ID_MAP[idx]

                # 动量更新 (Momentum Update)
                self.M_Z[bank_idx] = (1 - self.momentum) * zc_norm[i] + self.momentum * self.M_Z[bank_idx]

    def hard_sample_mining(self, zc_norm_A, y_A, k_p=1, k_n=1):
        """
        在全局记忆库中挖掘硬正样本 P_hard 和特殊负样本 N_special
        """
        B = zc_norm_A.size(0)

        # 使用当前填充部分
        M_Z_filled = self.M_Z[:self.filled_count]
        M_Y_filled = self.M_Y[:self.filled_count]

        # 1. 计算锚点与所有 Memory Bank 样本的相似度 [B, filled_count]
        sim_matrix = torch.matmul(zc_norm_A, M_Z_filled.t())

        zc_P_hard_list = []
        zc_N_special_list = []

        for i in range(B):
            sim_A = sim_matrix[i]
            y_A_i = y_A[i]

            # --- 挖掘 P_hard (Y_P == Y_A, Sim 最小) ---
            P_mask = (M_Y_filled == y_A_i)
            # 排除当前 Batch 内的样本（非必须，但更严谨）

            sim_P = sim_A[P_mask]
            M_Z_P = M_Z_filled[P_mask]

            # 确保有足够的样本
            k_p_safe = min(k_p, sim_P.size(0))
            if k_p_safe > 0:
                # 找相似度最小的 K_P 个样本 (找 -Sim 最大的)
                _, P_indices = torch.topk(-sim_P, k=k_p_safe)
                zc_P_hard = M_Z_P[P_indices].mean(dim=0)
            else:
                # 如果没有同类样本，使用一个随机负样本 (避免 loss 崩溃)
                zc_P_hard = torch.zeros_like(zc_norm_A[i])  # 实际应使用更合理的 fallback

            # --- 挖掘 N_special (Y_N != Y_A, Sim 最大) ---
            N_mask = (M_Y_filled != y_A_i)
            sim_N = sim_A[N_mask]
            M_Z_N = M_Z_filled[N_mask]

            k_n_safe = min(k_n, sim_N.size(0))
            if k_n_safe > 0:
                # 找相似度最大的 K_N 个样本
                _, N_indices = torch.topk(sim_N, k=k_n_safe)
                zc_N_special = M_Z_N[N_indices].mean(dim=0)
            else:
                # 如果没有异类样本（极少发生），使用一个随机负样本
                zc_N_special = torch.zeros_like(zc_norm_A[i])

            zc_P_hard_list.append(zc_P_hard)
            zc_N_special_list.append(zc_N_special)

        return torch.stack(zc_P_hard_list, dim=0), torch.stack(zc_N_special_list, dim=0)

class CounterfactualCausalModeling(nn.Module):
    """
    Stage1: base model
    """

    def __init__(self, backbone,input_dim=512, hidden_dim=256, z_dim=128):
        super(CounterfactualCausalModeling, self).__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.regressor = OrdinalHead(       #事实回归头
            input_dim=input_dim,  # 而不是 z_dim*2
            hidden_dim=256,
            num_classes=3,
            dropout=0.2
        )

        self.metric_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, z_dim)
        )

        # 主要使用回归头 - 为GRL提供丰富梯度
        self.bmi_regressor = nn.Linear(input_dim, 1)
        self.age_regressor = nn.Linear(input_dim, 1)





    def forward(self, inputs, grl_lambda=0.0):
        # === backbone features ===
        features = self.backbone(inputs)  # [B, T, J, C]

        if len(features.shape)==2 :
            B = features.shape[0]  # batch size
            C = 256  # 特征维度
            T = 81  # 时间帧数
            V = 25  # 关节点数

            features = features.view(B, 1, 1, C)  # [B, 1, 1, C]
            features = features.expand(B, T, V, C)  # [B, T, V, C] = [B, T, J, C]




        feature_pooled = features.mean(dim=(1,2))
        logits= self.regressor(feature_pooled)

        metric_feats = self.metric_projector(feature_pooled)
        metric_feats = F.normalize(metric_feats, p=2, dim=1)

        # === GRL分支 ===
        # 应用梯度反转
        grl_features = grad_reverse(feature_pooled, grl_lambda)  # [B, C]

        # Non-ID预测
        bmi_pred = self.bmi_regressor(grl_features)  # [B, 1]
        age_pred = self.age_regressor(grl_features)  # [B, 1]






        out = {
            "logits": logits,       #z_g经过池化后输出的回归结果
            "features": metric_feats,  # [新增] 用于 Causal Metric Loss (已归一化)
            'bmi_pred': bmi_pred,           # BMI预测（GRL）
            'age_pred': age_pred,           # 年龄预测（GRL）
        }

        return out






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

        # === Disease-related and Confounding Encoders === #
        self.disease_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=z_dim,
            num_layers=3,
            dropout=0.1
        )

        self.confound_encoder = MLPEncoder(         # 需要加入正交损失或者对抗约束
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=z_dim,
            num_layers=3,
            dropout=0.1
        )



        # === Shared Regression Head === #
        # self.regressor = RegressionHead(
        #     input_dim=z_dim * 2,  # disease + confound features
        #     hidden_dim=256,
        #     dropout=0.2
        # )

        # self.regressor = OrdinalHead(
        #     input_dim=z_dim,  # 而不是 z_dim*2
        #     hidden_dims=256,
        #     num_classes=3,
        #     dropout=0.2
        # )

        # self.regressor = LinearClassifierHead(
        #     input_dim=z_dim,
        #     hidden_dims=256,
        #     num_classes=3,  # 直接输出 num_classes 个类别
        #     dropout=0.2
        # )

        self.regressor = OrdinalHead(       #事实回归头
            input_dim=z_dim,  # 而不是 z_dim*2
            hidden_dim=256,
            num_classes=3,
            dropout=0.2
        )

        # 重构
        self.decoder = nn.Sequential(
            nn.Linear(z_dim * 2 , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 重构回 backbone 的 feature dim
        )





    def forward(self, inputs, labels= None,metadata=None):
        # === backbone features ===
        features = self.backbone(inputs)  # [B, T, J, C]

        if len(features.shape)==2 :
            B = features.shape[0]  # batch size
            C = 256  # 特征维度
            T = 81  # 时间帧数
            V = 25  # 关节点数

            features = features.view(B, 1, 1, C)  # [B, 1, 1, C]
            features = features.expand(B, T, V, C)  # [B, T, V, C] = [B, T, J, C]


        # === encoded features ===
        # 病理特征和混淆特征都从同一个backbone出来
        z_g = self.disease_encoder(features)  # [B, z_dim]
        z_c = self.confound_encoder(features)   # [B, z_dim]

        # 现在获取了两个不同方面的特征，需要做的就是通过干预来分离病理因素和混淆因素
        # 以下是具体的实施方法

        # 1. 主任务，确保z_g包含了足够信息来进行正确分类
        # === ordinal prediction ===
        z_g_pooled = z_g.mean(dim=(1, 2))
        logits = self.regressor(z_g_pooled)  # [B, K-1]

        # 2. 让z_c无法预测病理标签，通过无监督的GRL
        # 但是这本质上只是让模型把“能预测疾病标签的信息”都塞到z_g里面，其余的都塞到z_c里面，并没有实现解耦
        # 假设有一个混淆变量，例如年龄，与疾病标签高度相关，但是它实际上通过X(疾病因素)->M(年龄)->Y(标签）的因果链来影响，在这种情况下，年龄依然会被塞入z_g里面，没有实现因果解耦
        # 因此在这种情况下，显式加入混淆因素信息作为监督信号是必要的，明确指定混淆因素（如年龄）作为z编码_c的任务;注意GRL依然需要保留，这样才是“不能预测病理但可以预测混淆变量”的双重保证
        # === GRL ===
        z_c_pooled = z_c.mean(dim=(1, 2))     # 进行时间和关节维度上的池化
        rev_zc=grad_reverse(z_c_pooled,lambd=1.0)


        confound_logits=self.regressor(rev_zc)  #用同样的回归头进行病理标签预测

        # 3. 重构损失，避免退化，确保编译后的z_g和z_c依然能够还原出原本的信息
        # === ReCon ===
        recon_in = torch.cat([z_g,z_c], dim = -1)       #在C维度上进行拼接
        recon_features = self.decoder(recon_in)     #进行decoder解码

        # # 4. 反事实损失，进行批次内特征交换
        # counterfactual_logits=None
        # shuffle_idx=None
        # if labels is not None:
        #     B = z_g_pooled.shape[0]
        #     shuffle_idx = torch.randperm(B).to(labels.device)
        #
        #     # 将交换后的疾病特征输入到回归头进行预测
        #     z_g_swapped = z_g_pooled[shuffle_idx]
        #     counterfactual_logits = self.regressor(z_g_swapped)






        out = {
            "logits": logits,       #z_g经过池化后输出的回归结果
            "confound_logits": confound_logits,  # 梯度反转之后的confound输出的回归结果

            # "counterfactual_logits": counterfactual_logits,  # 进行干预（z_g或者z_c交换）之后得到的回归结果

            "original_features": features,  # 原始特征，用于和重构特征进行比较
            "disease_features": z_g_pooled,    #病理特征z_g本身，没有经过池化
            "confound_features": z_c_pooled,  # 同上
            "recon_features": recon_features,   #重建得到的特征

            # "shuffle_idx": shuffle_idx,  # 将索引返回
        }

        return out






#
# # === Usage Example === #





#     # Model configuration
#     model = CounterfactualCausalModeling(
#         input_dim=512,
#         hidden_dim=256,
#         z_dim=128,
#         counterfactual_strategy='vae_sampling',
#         use_vae_generator=True,
#         use_disentanglement_loss=False
#     )
#
#     # Loss function
#     criterion = CounterfactualLoss(
#         lambda_consistency=0.5,
#         lambda_vae=0.1,
#         lambda_disentangle=0.2
#     )
#
#     # Dummy data
#     batch_size, seq_len, num_joints, feat_dim = 20, 243, 17, 512
#     inputs = torch.randn(batch_size, seq_len, num_joints, feat_dim)
#     labels = torch.randn(batch_size)  # Parkinson's gait scores
#     disease_labels = torch.randint(0, 2, (batch_size,))  # 0=Normal, 1=PD
#
#     # Training
#     model.train()
#     output = model(inputs, disease_labels)
#     losses = criterion(output, labels)
#
#     print("=== Training Results ===")
#     print("Losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})
#
#     # Counterfactual Inference
#     model.eval()
#     cf_disease = model.predict_counterfactual(inputs, intervention="vae_sampling")
#
#     print("\n=== Counterfactual Predictions ===")
#     print(f"Original scores: {output['factual_pred'].detach()}")
#     print(f"If VAE sampled: {cf_disease}")
# #



# counterfactual_logits = None
        # shuffle_idx=None
        #
        # if labels is not None:
        #     B = z_c_pooled.shape[0]
        #     shuffle_idx = torch.randperm(B).to(labels.device)
        #
        #     # 保持 z_g 不变，交换 confound 特征
        #     z_c_swapped = z_c_pooled[shuffle_idx]
        #     z_cf = torch.cat([z_g_pooled, z_c_swapped], dim=-1)
        #
        #     # 用新的 head（如果只想看 z_g，可以直接用 z_g；如果希望利用 z_c，最好用 concat）
        #     counterfactual_logits = self.regressor(z_g_pooled)  # 这里保持不变，关键是 loss 写法
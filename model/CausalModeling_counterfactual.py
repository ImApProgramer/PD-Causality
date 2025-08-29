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

        self.regressor = OrdinalHead(
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

    def forward(self, inputs, labels= None):
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
        z_c = self.confound_encoder(features)   # [B, z_dim]，准备通过GRL进行病理的预测，但是梯度反转

        # === ordinal prediction ===
        z_g_pooled = z_g.mean(dim=(1, 2))
        logits = self.regressor(z_g_pooled)  # [B, K-1]



        # === GRL ===
        # 让z_c无法预测病理标签
        z_c_pooled = z_c.mean(dim=(1, 2))     # 进行时间和关节维度上的池化
        rev_zc=grad_reverse(z_c_pooled,lambd=1.0)
        confound_logits=self.regressor(rev_zc)



        # === ReCon ===
        # 重构损失，避免退化
        recon_in = torch.cat([z_g,z_c], dim = -1)       #在C维度上进行拼接
        recon_features = self.decoder(recon_in)     #进行decoder解码


        # === counterfactual ===
        # 让模型知道标签必须随着z_g变化，而对z_c保持不变
        # ---- 构造交换样本 ----
        counterfactual_logits = None
        shuffle_idx=None
        if labels is not None:
            B = z_g_pooled.shape[0]
            shuffle_idx = torch.randperm(B).to(labels.device)

            # 将交换后的疾病特征输入到回归头进行预测
            z_g_swapped = z_g_pooled[shuffle_idx]
            counterfactual_logits = self.regressor(z_g_swapped)





        out = {
            "logits": logits,
            "disease_features": z_g,
            "confound_logits": confound_logits,
            "recon_features": recon_features,
            "original_features": features,
            "counterfactual_logits": counterfactual_logits,
            "shuffle_idx": shuffle_idx,  # 将索引返回
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



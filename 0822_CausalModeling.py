import os
import sys
import argparse
import datetime
from collections import defaultdict, Counter

import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.metrics import f1_score
from torch import nn
from torch import optim
from tqdm import tqdm
from test import process_reports,save_and_load_results
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from configs import generate_config_motionagformer
from data.augmentations import RandomNoise, RandomRotation, MirrorReflection, axis_mask
from data.dataloaders import PDReader, MotionAGFormerPreprocessor, PreserveKeysTransform, collate_fn,assert_backbone_is_supported,GCNPreprocessor,ProcessedDataset,dataset_factory

from const import path
from learning.utils import compute_class_weights, AverageMeter
from utility import utils
from utility.utils import set_random_seed
from test import update_params_with_best, setup_datasets,map_to_classifier_dim
import pkg_resources
from torchvision import transforms
import torch
from model.motionagformer.MotionAGFormer import MotionAGFormer
from  model.CausalModeling_counterfactual import *
from learning.losses import *


_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          #目前看来只有encoder-decoder中用到了它

_TOTAL_SCORES = 3
METADATA_MAP = {'gender': 0, 'age': 1, 'height': 2, 'weight': 3, 'bmi': 4}

_GCN_JOINTS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_path + "/../")

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class D3Selector:
    def __init__(self, model, loader, device, num_classes=3):
        self.model = model
        self.loader = loader
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def scan_dataset(self):
        """快速扫描整个数据集，获取每个样本的特征和不确定性"""
        self.model.eval()
        all_features = []
        all_entropies = []  #熵
        all_video_ids = []

        # 使用 tqdm 进度条，因为这步需要跑一遍前向
        for x, y, video_idx, metadata in tqdm(self.loader, desc="[D3] Scanning Data"):
            x = x.to(self.device)

            # 获取模型输出
            outputs = self.model(x, grl_lambda=0.0)

            # 1. 获取特征 (用于计算多样性 d1)
            # features: [B, D] 确保是展平的且归一化的
            feats = outputs['features'].detach().cpu()
            if len(feats.shape) > 2:  # 如果是时序特征，取平均
                feats = feats.mean(dim=(1, 2))
            all_features.append(feats)

            # 2. 获取熵 (用于计算难度 d2)
            # logits: [B, K-1] -> 转换为概率分布
            # 注意：你的模型是 Ordinal Regression，logits 是 K-1 个。
            # 这里简化处理：直接用 Sigmoid 的平均不确定性，或者转为分类概率
            logits = outputs['logits'].detach()
            probs = torch.sigmoid(logits)  # [B, 2]

            # 针对 Ordinal 任务的简化熵计算：
            # 如果 prob 接近 0.5，说明很不确定。
            # Entropy = -p*log(p) - (1-p)*log(1-p) 对所有二分类头求和
            entropy = -(probs * torch.log(probs + 1e-6) + (1 - probs) * torch.log(1 - probs + 1e-6))
            entropy = entropy.mean(dim=1).cpu()  # [B]
            all_entropies.append(entropy)

            all_video_ids.append(video_idx)

        return (torch.cat(all_features),
                torch.cat(all_entropies),
                torch.cat(all_video_ids))

    def select_coreset(self, budget_ratio=0.5):
        """执行 D3 选择算法"""
        features, entropies, video_ids = self.scan_dataset()
        num_samples = len(features)
        budget = int(num_samples * budget_ratio)

        selected_indices = []
        # mask 用于标记未被选中的样本 (True = 未选)
        remaining_mask = torch.ones(num_samples, dtype=torch.bool)

        # --- 初始化 ---
        # 选第一个样本：选熵最大的（最难的）作为种子
        first_idx = torch.argmax(entropies).item()
        selected_indices.append(first_idx)
        remaining_mask[first_idx] = False

        # 维护一个距离表：每个剩余样本到“当前已选集合”的最小距离
        # 初始化为到第一个样本的距离
        # dist: [N]
        current_dists = torch.cdist(features, features[first_idx].unsqueeze(0)).squeeze()

        print(f"[D3] Selecting {budget}/{num_samples} samples...")

        # --- 迭代选择 (Greedy Loop) ---
        # 这里的循环是核心：最大化 (Diversity * Difficulty)
        for _ in range(budget - 1):
            if not remaining_mask.any(): break

            # 1. Diversity (d1): 到已选集的最短距离
            # 在每一轮，只需要更新距离表（取 min）
            # 我们只关心剩余样本
            valid_dists = current_dists[remaining_mask]

            # 2. Difficulty (d2): 熵
            valid_entropies = entropies[remaining_mask]

            # 3. 综合打分: Score = d1 * d2 * d3(默认为1)
            # 归一化一下数值防止量级差异太大
            d1_norm = valid_dists / (valid_dists.max() + 1e-8)
            d2_norm = valid_entropies / (valid_entropies.max() + 1e-8)

            scores = d1_norm * d2_norm

            # 找到当前剩余样本中分数最高的索引（相对索引）
            best_rel_idx = torch.argmax(scores).item()

            # 映射回全局索引
            # 获取所有剩余样本的全局索引
            remaining_indices = torch.nonzero(remaining_mask).squeeze()
            if remaining_indices.dim() == 0: remaining_indices = remaining_indices.unsqueeze(0)
            best_global_idx = remaining_indices[best_rel_idx].item()

            # 加入集合
            selected_indices.append(best_global_idx)
            remaining_mask[best_global_idx] = False

            # --- 更新距离表 ---
            # 计算所有样本到新加入样本的距离
            new_dists = torch.cdist(features, features[best_global_idx].unsqueeze(0)).squeeze()
            # 更新最短距离：min(旧距离, 到新样本的距离)
            current_dists = torch.min(current_dists, new_dists)

        # 返回被选中样本的 Video IDs (用于在 Dataset 中索引)
        selected_video_ids = video_ids[selected_indices]
        return set(selected_video_ids.numpy().tolist())

def coral_loss(logits, labels, num_classes):
    """
    logits: [B, K-1]
    labels: [B] int64
    """
    # 构造 CORAL target
    target = torch.zeros(labels.size(0), num_classes - 1, device=labels.device)
    for j in range(1, num_classes):
        target[:, j-1] = (labels >= j).float()

    return F.binary_cross_entropy_with_logits(logits, target)


def final_test(model, test_loader, params):
    model.eval()
    video_logits = defaultdict(list)
    video_predclasses = defaultdict(list)
    video_labels = {}
    video_states = {}
    video_names = {}

    num_classes = params["num_classes"]

    loop = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device).long()

            # 使用您的模型前向传播
            outputs = model(x, grl_lambda=0.0)
            logits = outputs["logits"]  # [B, K-1]

            # CORAL预测逻辑
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).sum(dim=1)  # [B]

            # 收集每个视频的信息
            for i in range(x.size(0)):
                vid = video_idx[i].item()

                # 收集logits和预测
                video_logits[vid].append(logits[i].cpu().numpy())
                video_predclasses[vid].append(preds[i].item())

                # 只保存一次标签和元数据
                if vid not in video_labels:
                    video_labels[vid] = y[i].item()

                    # 获取视频名称和状态
                    video_name = test_loader.dataset.video_names[vid]
                    video_names[vid] = video_name

                    if 'on' in video_name.lower():
                        video_states[vid] = 'ON'
                    else:
                        video_states[vid] = 'OFF'

    # 对每个视频的预测进行聚合
    final_predictions = []
    final_labels = []
    final_logits = []
    final_states = []
    final_names = []

    for vid in video_logits.keys():
        # 多数投票决定最终预测
        class_counts = Counter(video_predclasses[vid])
        majority_class = class_counts.most_common(1)[0][0]

        # 对logits求平均
        avg_logits = np.mean(video_logits[vid], axis=0)

        final_predictions.append(majority_class)
        final_labels.append(video_labels[vid])
        final_logits.append(avg_logits)
        final_states.append(video_states[vid])
        final_names.append(video_names[vid])

    return final_predictions, final_labels, final_logits, final_states, final_names

def validate_model(model, val_loader, device, num_classes):
    """验证函数"""
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, video_idx, metadata in val_loader:
            x, y = x.to(device), y.to(device).long()

            outputs = model(x, grl_lambda=0.0)
            logits = outputs["logits"]

            # 计算损失
            loss = coral_loss(logits, y, num_classes)
            val_loss.update(loss.item(), x.size(0))

            # 预测
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).sum(dim=1)

            # 计算准确率
            batch_acc = (preds == y).float().mean().item()
            val_acc.update(batch_acc, x.size(0))

            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 计算F1分数
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'loss': val_loss.avg,
        'acc': val_acc.avg,
        'f1': val_f1,
        'all_preds': all_preds,
        'all_labels': all_labels
    }



def save_checkpoint(checkpoint_root_path, epoch, lr, optimizer, model, best_accuracy, fold, latest):
    checkpoint_path_fold = os.path.join(checkpoint_root_path, f"fold{fold}")
    if not os.path.exists(checkpoint_path_fold):
        os.makedirs(checkpoint_path_fold)
    checkpoint_path = os.path.join(checkpoint_path_fold,
                                   'latest_epoch.pth.tr' if latest else 'best_epoch.pth.tr')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'best_accuracy': best_accuracy
    }, checkpoint_path)

def orthogonal_loss(z1, z2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cos_sim = torch.sum(z1 * z2, dim=-1)  # batch-wise inner product
    return (cos_sim ** 2).mean()


def train_model(params, class_weights, train_loader, val_loader, model, fold, backbone_name, mode="RUN"):
    # ================= 1. 初始化与参数提取 =================
    # 初始化输出目录
    checkpoint_root_path = os.path.join(path.CAUSAL_OUT_PATH, params['model_prefix'], 'models')
    if not os.path.exists(checkpoint_root_path): os.makedirs(checkpoint_root_path)

    # 获取设备
    device = next(model.parameters()).device
    num_classes = params["num_classes"]

    # [参数] 提取权重
    lambda_rnc = params.get('lambda_rnc', 2.0)  # RNC 正则权重，原来为0.5
    lambda_grl = params.get('grl_loss_weight', 0.1)  # GRL 正则权重

    # [参数] GRL 调度参数
    max_grl_lambda = params.get('max_grl_lambda', 1.0)
    grl_warmup_epochs = params.get('grl_warmup_epochs', 10)

    # [组件] 初始化 Memory Bank 和 RNC Loss (修复：之前漏了定义)
    # 必须有这个才能算 loss_rnc
    memory_bank = OrdinalClassBalancedMemory(num_classes=3, feat_dim=128, device=device)
    ciml_criterion = MemoryCausalOrdinalLoss(temperature=2.0, memory_bank=memory_bank).to(device)

    # [优化器] 统一优化器 (负责全模型)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get('lr', 1e-4),
        weight_decay=params.get('weight_decay', 1e-3)   #原来为1e-4
    )
    scheduler = StepLR(optimizer, step_size=params.get('lr_step_size', 10), gamma=params.get('lr_decay', 0.5))

    # [循环控制]
    total_epochs = params.get("epochs", 50) #原为40
    patience = params.get("stopping_tolerance", 10)

    best_val_f1 = 0.0
    patience_counter = 0

    print(f"[INFO] Starting Joint Training (Regularization Mode) for {total_epochs} epochs")
    print(f"[INFO] Weights -> Main: 1.0 | RNC: {lambda_rnc} | GRL: {lambda_grl}")

    # [新增] D3 配置
    use_d3 = True  # 开关
    d3_start_epoch = 5  # Warm-up epoch 数量
    d3_interval = 5  # 每隔多少 epoch 重新选择一次
    d3_budget = 0.9  # 数据保留比例 (例如只练 60% 的数据)
    selected_video_ids = None  # 存储被选中的 ID

    # ================= 2. 训练循环 =================
    for epoch in range(total_epochs):

        # [新增] D3 选择阶段 (在 Epoch 开始前执行)
        if use_d3 and epoch >= d3_start_epoch and (epoch - d3_start_epoch) % d3_interval == 0:
            print(f"\n[D3] Triggering Data Selection at Epoch {epoch}...")
            selector = D3Selector(model, train_loader, device, num_classes)
            selected_video_ids = selector.select_coreset(budget_ratio=d3_budget)
            print(f"[D3] Selected {len(selected_video_ids)} samples for training.\n")

        model.train()
        model.requires_grad_(True)  # 确保全模型可训练

        train_loss_meter = AverageMeter()

        # 计算当前 GRL 强度
        if epoch < grl_warmup_epochs:
            current_grl_lambda = max_grl_lambda * (epoch / grl_warmup_epochs)
        else:
            current_grl_lambda = max_grl_lambda

        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}', unit="batch")

        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device).long()

            # [新增] D3 过滤逻辑
            # 如果当前开启了筛选，且样本不在选中列表里，则跳过
            # 注意：Batch 中可能部分在，部分不在。最简单的增量改法是基于Batch过滤
            # 但为了简单，如果 Batch 里包含未选中样本，我们通过 Mask 将其 Loss 置零

            x, y = x.to(device), y.to(device).long()

            if selected_video_ids is not None:
                # 找出当前 batch 中哪些样本是被选中的
                # video_idx 是 tensor，转成 list 判断
                batch_vid_list = video_idx.tolist()
                # 生成 mask: True 表示保留 (被选中), False 表示丢弃
                keep_mask = torch.tensor([vid in selected_video_ids for vid in batch_vid_list], device=device)

                if not keep_mask.any():
                    # 如果整个 Batch 都没被选中，直接跳过，省算力
                    continue

                # 仅保留被选中的样本进行训练 (这是 Sample-Efficient 的关键)
                x = x[keep_mask]
                y = y[keep_mask]
                if len(metadata) > 0:
                    metadata = metadata[keep_mask]

                # 重新计算 batch size (用于 logging)
                curr_bs = x.size(0)
            else:
                curr_bs = x.size(0)

            # 准备 GRL 标签
            if len(metadata) > 0:
                bmi_labels = metadata[:, METADATA_MAP['bmi']].to(device).float()
            else:
                bmi_labels = torch.zeros(x.size(0)).to(device)

            optimizer.zero_grad()

            # --- A. 前向传播 ---
            outputs = model(x, grl_lambda=current_grl_lambda)

            # --- B. 计算损失 (Joint Training) ---
            # 1. 主任务损失 (CORAL)
            loss_main = coral_loss(outputs["logits"], y, num_classes)

            # 2. 正则项: RNC (排序约束)
            loss_rnc = ciml_criterion(outputs['features'], y)

            # 3. 正则项: GRL (去偏约束)
            if current_grl_lambda > 0:
                loss_grl_val = F.mse_loss(outputs["bmi_pred"].squeeze(), bmi_labels)
            else:
                loss_grl_val = torch.tensor(0.0).to(device)

            # [核心] 加权求和
            total_loss = loss_main + (lambda_rnc * loss_rnc) + (lambda_grl * loss_grl_val)

            # --- C. 反向传播 ---
            total_loss.backward()
            optimizer.step()

            # 记录日志
            train_loss_meter.update(total_loss.item(), curr_bs)
            loop.set_postfix({
                'L_main': f'{loss_main.item():.3f}',
                'L_rnc': f'{loss_rnc.item():.3f}',
                'L_grl': f'{loss_grl_val.item():.3f}'
            })

        # 更新学习率
        scheduler.step()

        # ================= 3. 验证与保存 =================
        val_metrics = validate_model(model, val_loader, device, num_classes)
        val_f1_score = val_metrics['f1']

        # 获取当前学习率 (修复：变量名改为 optimizer)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1} | "
              f"Train Loss: {train_loss_meter.avg:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Val F1: {val_f1_score:.4f} | "
              f"LR: {current_lr:.1e}")

        # --- Checkpoint 保存逻辑 ---
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            patience_counter = 0

            save_checkpoint(
                checkpoint_root_path, epoch + 1, current_lr,
                optimizer,  # 修复：使用统一的 optimizer
                model,
                best_val_f1, fold, latest=False
            )
            print(f"[INFO] Best checkpoint saved! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{patience}")

        # 早停
        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Stop at epoch {epoch + 1}")
            break

    # 训练结束保存 Latest
    if mode == "RUN":
        save_checkpoint(checkpoint_root_path, total_epochs, 0.0, optimizer, model, best_val_f1, fold, latest=True)
        print(f'[INFO] Latest checkpoint saved at: {checkpoint_root_path}')



def initialize_wandb(params):
    wandb.init(name=params['wandb_name'], project='MotionEncoderEvaluator_PD', settings=wandb.Settings(start_method='fork'))
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update(params)
    wandb.config.update({'installed_packages': installed_packages})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( poseformer, ''motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only',
                        help='train mode( end2end, classifier_only )')
    parser.add_argument('--dataset', type=str, default='PD', help='**currently code only works for PD')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int,
                        help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--last_run_foldnum', default='7', type=str)
    parser.add_argument('--readstudyfrom', default=1, type=int)

    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str,
                        help="add metadata prob to the training 'gender,age,bmi,height,weight'")
    args = parser.parse_args()

    param = vars(args)

    backbone_name = param['backbone']

    if backbone_name == 'motionagformer':
        data_params = {
            'data_type': 'Kinect',  # options: "Kinect", "GastNet", "PCT", "ViTPose"
            'data_dim': 3,
            'in_data_dim': 2,
            'data_centered': True,
            'merge_last_dim': False,
            'use_validation': True,
            'simulate_confidence_score': True,
            'pretrained_dataset_name': 'h36m',
            'model_prefix': 'MotionAGFormer_',
            # options: mirror_reflection, random_rotation, random_translation
            # 'augmentation': [],
            'rotation_range': [-10, 10],
            'rotation_prob': 0.5,
            'mirror_prob': 0.5,
            'noise_prob': 0.5,
            'axis_mask_prob': 0.5,
            'translation_frac': 0.05,
            'data_norm': "rescaling",
            'select_middle': False,
            'exclude_non_rgb_sequences': False
        }
        model_params = {
            'source_seq_len': 27,
            'n_layers': 12,
            'dim_in': 3,
            'dim_feat': 64,
            'dim_rep': 512,
            'dim_out': 3,
            'mlp_ratio': 4,
            'attn_drop': 0.0,
            'drop': 0.0,
            "drop_path": 0.0,
            "use_layer_scale": True,
            "layer_scale_init_value": 0.00001,
            "use_adaptive_fusion": True,
            "num_heads": 8,
            "qkv_bias": False,
            "qkv_scale": None,
            "hierarchical": False,
            "use_temporal_similarity": True,
            "neighbour_num": 2,
            "temporal_connection_len": 1,
            "use_tcn": False,
            "graph_only": False,
            'classifier_dropout': 0.0,
            'merge_joints': True,
            'classifier_hidden_dims': [1024],
            'model_checkpoint_path': "/czl_ssd/motion_evaluator/Pretrained_checkpoints/motionagformer/motionagformer-xs-h36m.pth.tr"
        }
        learning_params = {
            'wandb_name': 'MotionAGFormer',
            'experiment_name': '',
            'batch_size': 32,
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'AdamW',
            'lr_backbone': 0.0001,
            'lr_head': 0.001,
            'weight_decay': 0.0,
            'lambda_l1': 0.0001,
            'scheduler': "StepLR",
            'lr_decay': 0.99,
            'epochs': 20,
            'stopping_tolerance': 10,
            'lr_step_size': 1
        }
    elif backbone_name == 'ctrgcn':
        data_params = {
            'data_type': 'PD',
            'in_channels': 3,
            'num_point': 25,
            'num_person': 1,
            'data_path': path.PD_PATH_POSES_forGCN,
            'labels_path': path.PD_PATH_LABELS,
            'data_centered': True,  # 用来考虑是否需要进行中心化
            'model_prefix': 'GCN_',
            'data_norm': "rescaling",  # 不确定用途，只是为了别报错
            'source_seq_len': 81,
            'use_validation': True,  # 是否启用验证
            'select_middle': False,
            'mirror_prob': 0.5,
            'rotation_range': [-10, 10],
            'rotation_prob': 0.5,
            'noise_prob': 0.5,
            'axis_mask_prob': 0.5
        }

        model_params = {
            'model': 'CTRGCN',
            'dim_rep': 256,
            'experiment_name': '',
            'classifier_dropout': 0.5,
            # 'classifier_hidden_dims': [1024],
            'model_args': {
                'in_channels': 3,
                'num_class': 3,  # 修改为你的动作类别数
                'graph_args': {
                    'layout': 'ntu_rgb_d',  # 或者 'coco'
                    'strategy': 'uniform'
                }
            },
            'weights': f"{path.CAUSAL_PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/ctrgcn/ctrgcn-weights.bin",
            'model_checkpoint_path': "/czl_ssd/motion_evaluator/Pretrained_checkpoints/ctrgcn/runs-58-57072.pt"
            # ntu120 csub joint
        }

        learning_params = {
            'batch_size': 64,  # 太大显存顶不住，其他模型压力比较小
            'epochs': 20,  # 建议60~80
            'lr_head': 0.001,
            'lr_backbone': 0.0001,
            'optimizer': 'SGD',
            'weight_decay': 0.0001,
            'momentum': 0.9,
            'nesterov': True,
            'lr_decay_step': [10, 15],
            'dropout_rate': 0.5,  # ✅ 添加此字段以匹配 update_params_with_best 中的 dropout_rate
            'use_weighted_loss': True,  # 0807调参
            'lambda_l1': 0.0,  # ✅ 默认值
            'wandb_name': 'CTRGCN',  # ✅ 必须手动提供才能被 update 函数处理
            'stopping_tolerance': 10,  # 用于早停，不知道是否用到，但是train中显式检查了
            'criterion': 'WCELoss',  # 和poseformer用的一样
            # 'scheduler': "StepLR",
            'lr_step_size': 1,
            'lr_decay': 0.99
        }

    params = {**param, **data_params, **model_params, **learning_params}

    backbone_name=params['backbone']
    if backbone_name == 'motionagformer':
        best_params = {
            "lr": 5e-05,
            "num_epochs": 30,
            "num_hidden_layers": 2,
            "layer_sizes": [256, 50, 16, 3],
            "optimizer": 'AdamW',
            "use_weighted_loss": True,
            "batch_size": 16,
            "dropout_rate": 0.4,
            'weight_decay': 0.0001,
            'momentum': 0.66
        }

    # if backbone_name == 'motionagformer':
    #     best_params = {
    #         "lr": 5e-05,  # 稍微加快收敛速度，但不过冲
    #         "num_epochs": 30,  # 多给点epoch，让增强样本有机会训练到
    #         "num_hidden_layers": 2,
    #         "layer_sizes": [128, 32, 8, 3],  # 降低模型容量，减少过拟合
    #         "optimizer": 'AdamW',  # 对小样本泛化稳定
    #         "use_weighted_loss": True,
    #         "batch_size": 16,  # 较小批量，增加梯度更新频率
    #         "dropout_rate": 0.4,  # 明显提高Dropout防过拟合
    #         "weight_decay": 0.001,  # L2正则更强
    #         "momentum": 0.9,  # 这里即使AdamW不用也可以留着给兼容
    #         "rotation_prob": 0.5,  # 增强概率提高
    #         "mirror_prob": 0.5,
    #         "noise_prob": 0.4,
    #         "axis_mask_prob": 0.3,
    #         "rotation_range": (-15, 15)  # 限制旋转幅度，防止过大扰动
    #     }
    elif backbone_name == 'ctrgcn':
        best_params = {  # ⚠️这些参数是否合理呢？
            "lr": 1e-05,  # 0807调参：似乎有点太大了，从原来的0.1调整到0.001
            "num_epochs": 20,
            "batch_size": 128,
            "optimizer": 'AdamW',
            "weight_decay": 0.00057,
            "momentum": 0.66,
            "dropout_rate": 0.1,  # 0807调参：保持和下面一致
            "use_weighted_loss": True  # 0807调参修改为True
        }

    params['classifier_dropout'] = best_params['dropout_rate']
    params['classifier_hidden_dims'] = map_to_classifier_dim(backbone_name, 'option1')
    params['optimizer'] = best_params['optimizer']
    params['lr_head'] = best_params['lr']
    if 'lambda_l1' in best_params:
        params['lambda_l1'] = best_params['lambda_l1']
    else:
        params['lambda_l1'] = 0.0  # 设置默认值

    params['epochs'] = best_params['num_epochs']
    params['criterion'] = 'WCELoss' if best_params['use_weighted_loss'] else 'CrossEntropyLoss'
    if params['optimizer'] in ['AdamW', 'Adam', 'RMSprop']:
        params['weight_decay'] = best_params['weight_decay']
    if params['optimizer'] == 'SGD':
        params['momentum'] = best_params['momentum']
    params['wandb_name'] = params['wandb_name'] + '_test' + str(params['last_run_foldnum'])

    # params['input_dim'] = 3*[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]   # 这个参数对于CTR-GCN来说无用，因为它不展平向量
    # params['pose_dim'] = 3*[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 同上，无用
    # params['num_joints'] = 17

    params['data_path']= path.PD_PATH_POSES_forGCN
    params['labels_path'] = path.PD_PATH_LABELS  # Data Path is the path to csv files by default

    if backbone_name == 'motionagformer':
        params['model_prefix'] = params['model_prefix'] + '1_xsmall'
    elif backbone_name == 'ctrgcn':
        params['model_prefix'] = params['model_prefix']


    initialize_wandb(params)
    splits = []


    #===数据集准备阶段===
    if param['dataset'] == 'PD':
        num_folds = 23
        params['num_classes'] = 3
    else:
        raise NotImplementedError(f"dataset '{param['dataset']}' is not supported.")

    all_folds = range(1, num_folds + 1)
    set_random_seed(param['seed'])

    for fold in all_folds:
        train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name,
                                                                                           fold)
        splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))

    total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
    device = _DEVICE

    #===数据集准备阶段===

    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits):
        start_time = datetime.now()
        params['input_dim'] = train_dataset_fn.dataset._pose_dim  # 这个参数对于CTR-GCN来说无用，因为它不展平向量
        params['pose_dim'] = train_dataset_fn.dataset._pose_dim  # 同上，无用
        params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS

        model_backbone = load_pretrained_backbone(params, backbone_name)


        model = CounterfactualCausalModeling(model_backbone,params['dim_rep'])


        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        if fold == 1:
            model_params = count_parameters(model)
            print(f"[INFO] Model has {model_params} parameters.")

        if torch.cuda.is_available():
            model = model.to(device)

        else:
            raise Exception("Cuda is not enabled")

        train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name)

        checkpoint_root_path = os.path.join(path.CAUSAL_OUT_PATH, params['model_prefix'], 'models', f"fold{fold}")
        best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
        load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)['model'])
        model.cuda()

        outs, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
        total_outs_best.extend(outs)
        total_gts.extend(gts)
        total_states.extend(states)
        total_video_names.extend(video_names)
        print(f'fold # of test samples: {len(video_names)}')
        print(f'current sum # of test samples: {len(total_video_names)}')
        attributes = [total_outs_best, total_gts]
        names = ['predicted_classes', 'true_labels']
        res_dir = path.CAUSAL_OUT_PATH + os.path.join(params['model_prefix'], 'results')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        utils.save_json(os.path.join(res_dir, 'results_Best_fold{}.json'.format(fold)), attributes, names)

        total_logits.extend(logits)
        attributes = [total_logits, total_gts]

        logits_dir = path.CAUSAL_OUT_PATH + os.path.join(params['model_prefix'], 'logits')
        if not os.path.exists(logits_dir):
            os.makedirs(logits_dir)
        utils.save_json(os.path.join(logits_dir, 'logits_Best_fold{}.json'.format(fold)), attributes, names)

        last_ckpt_path = os.path.join(checkpoint_root_path, 'latest_epoch.pth.tr')
        load_pretrained_weights(model, checkpoint=torch.load(last_ckpt_path)['model'])
        model.cuda()
        outs_last, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
        total_outs_last.extend(outs_last)
        attributes = [total_outs_last, total_gts]
        utils.save_json(os.path.join(res_dir, 'results_last_fold{}.json'.format(fold)), attributes, names)

        res = pd.DataFrame(
            {'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last,
             'total_gts': total_gts, 'total_states': total_states})
        rep_out = path.CAUSAL_OUT_PATH + os.path.join(params['model_prefix'])
        with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
            pickle.dump(res, file)

        end_time = datetime.now()

        duration = end_time - start_time
        print(f"Fold {fold} run time:", duration)

    process_reports(total_outs_best, total_outs_last, total_gts, total_states, rep_out)
    save_and_load_results(total_video_names, total_outs_best, total_outs_last, total_gts, rep_out)



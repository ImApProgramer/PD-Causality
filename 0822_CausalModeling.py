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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 此时所有参数都需要梯度
        lr=params.get('lr', 1e-4),
        weight_decay=params.get('weight_decay', 1e-4)
    )
    scheduler = StepLR(optimizer, step_size=params['lr_step_size'], gamma=params['lr_decay'])
    checkpoint_root_path = os.path.join(path.CAUSAL_OUT_PATH, params['model_prefix'],'models')
    if not os.path.exists(checkpoint_root_path): os.makedirs(checkpoint_root_path)      #原本的mkdir只能创建单级目录


    num_classes = params["num_classes"]
    epochs = params.get("epochs", 20)
    patience = params.get("stopping_tolerance", 10)

    loop = tqdm(range(epochs), desc=f'Training (fold{fold})', unit="epoch")
    best_val_f1 = 0.0
    patience_counter = 0  # 记录连续多少次没有提升

    # [新增] 定义两个阶段的 Epoch
    total_epochs = params.get("epochs", 20)
    stage1_epochs = int(total_epochs * 0.7)  # 例如前 70% 轮次练特征
    stage2_epochs = total_epochs - stage1_epochs  # 后 30% 轮次练回归头


    # --- 关键：获取训练集总样本数 ---
    N_train_samples = len(train_loader.dataset)
    print(f"[INFO] Total training samples detected: {N_train_samples}")

    # --- 获取设备信息 ---
    device = next(model.parameters()).device


    # -------------------
    # Stage 1: 基础模型训练
    # -------------------
    print(f"\n--- Starting Stage 1: Training base model for {stage1_epochs} epochs ---")

    # GRL相关参数
    max_grl_lambda = params.get('max_grl_lambda', 1.0)  # GRL最大强度
    grl_warmup_epochs = params.get('grl_warmup_epochs', 10)  # GRL热身轮数
    grl_loss_weight = params.get('grl_loss_weight', 0.1)  # GRL损失权重

    # 1. 初始化 (在 Loop 外)
    memory_bank = OrdinalClassBalancedMemory(num_classes=3, feat_dim=128, device=device)
    # Loss 包含 memory_bank 引用
    ciml_criterion = MemoryCausalOrdinalLoss(
        margin_base=0.1,
        alpha=0.1,
        topk=5,
        temperature=2.0,
        memory_bank=memory_bank
    ).to(device)

    lambda_ciml = 0.1  # 开始设小一点

    for epoch in range(stage1_epochs):
        model.train()
        train_loss = AverageMeter()

        # [修改] 确保全模型解冻
        model.requires_grad_(True)

        # 计算当前GRL参数
        if epoch < grl_warmup_epochs:
            current_grl_lambda = max_grl_lambda * (epoch / grl_warmup_epochs)  # 线性增加
        else:
            current_grl_lambda = max_grl_lambda  # 达到最大后稳定

        current_grl_weight = grl_loss_weight * current_grl_lambda  # 综合权重


        loop = tqdm(train_loader, desc=f'Stage1 Epoch {epoch + 1}/{stage1_epochs}', unit="batch")

        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device).long()

            if len(metadata) > 0:
                # metadata 的形状应该是 [batch_size, 5] 对应5种元数据
                bmi_labels = metadata[:, METADATA_MAP['bmi']].to(device).float()  # [B]
                age_labels = metadata[:, METADATA_MAP['age']].to(device).float()  # [B]
            else:
                bmi_labels = torch.zeros(x.size(0)).to(device)
                age_labels = torch.zeros(x.size(0)).to(device)

            optimizer.zero_grad()

            outputs = model(x,grl_lambda=current_grl_lambda)

            # --- 1. RNC Loss (建立序属性) ---
            loss_rnc = ciml_criterion(outputs['features'], y)


            #main_loss = coral_loss(outputs["logits"], y, num_classes) #阶段1不学习main_loss，防止回归头干扰特征排序，让特征纯粹地学习排序和去偏

            # ===监控训练准确率 ===
            with torch.no_grad():
                probs = torch.sigmoid(outputs["logits"])
                train_preds = (probs > 0.5).sum(dim=1)
                train_acc = (train_preds == y).float().mean()

            # 只有当GRL lambda > 0时才计算GRL损失
            if current_grl_lambda > 0:
                bmi_loss = F.mse_loss(outputs["bmi_pred"].squeeze(), bmi_labels)
                # age_loss = F.mse_loss(outputs["age_pred"].squeeze(), age_labels)
                # grl_loss = bmi_loss + age_loss
                grl_loss = bmi_loss
            else:
                grl_loss = torch.tensor(0.0).to(device)


            # ciml_loss=(lambda_ciml * loss_metric)
            total_loss = loss_rnc+grl_loss_weight*grl_loss
            total_loss.backward()


            optimizer.step()
            train_loss.update(total_loss.item(), x.size(0))

            loop.set_postfix({
                'total_loss': f'{train_loss.avg:.4f}',
                # 'main_loss': f'{main_loss.item():.4f}',
                # 'ciml_loss': f'{ciml_loss.item():.4f}',
                'ciml_loss': f'{loss_rnc.item():.4f}',
                'grl_loss': f'{grl_loss.item():.4f}' if current_grl_lambda > 0 else '0.0000',
            })

        scheduler.step()


    # ==========================================
    # [新增] Stage 2: 纯预测训练 (全部冻结)
    # ==========================================
    print("--- Stage 2: Frozen Encoder Predictor Training ---")

    # 1. 彻底冻结编码器和 GRL 分支
    model.requires_grad_(False)

    # 2. 只开启回归头梯度
    for param in model.regressor.parameters():
        param.requires_grad = True

    # 3. 重新定义只针对 regressor 的优化器
    optimizer_s2 = torch.optim.AdamW(model.regressor.parameters(), lr=1e-3)

    # [新增] 定义 scheduler_s2，否则后面调用 step() 会报错
    scheduler_s2 = StepLR(optimizer_s2, step_size=5, gamma=0.5)
    for epoch in range(stage2_epochs):
        current_epoch = stage1_epochs + epoch + 1  # 累计 Epoch 计数

        # 注意：这里虽然 model.train()，但由于 requires_grad=False，Backbone 参数不会变
        # BN 层会继续更新统计量(Running Stats)，这通常是期望的。
        # 如果想严格定死 BN，可以设为 model.eval() 但只让 regressor 训练，不过通常 model.train() 没问题。
        model.train()
        train_loss = AverageMeter()
        train_acc_meter = AverageMeter()

        loop = tqdm(train_loader, desc=f'S2 Ep {epoch + 1}/{stage2_epochs}', unit="batch")

        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device).long()
            optimizer_s2.zero_grad()

            # [调用] 此时 grl_lambda 设为 0，因为特征已定型，不需要 GRL 对抗了
            outputs = model(x, grl_lambda=0.0)

            # --- 核心：只训练预测结果 ---
            loss_main = coral_loss(outputs["logits"], y, num_classes)

            loss_main.backward()
            optimizer_s2.step()

            # 记录指标
            train_loss.update(loss_main.item(), x.size(0))

            # 计算训练集准确率用于监控
            with torch.no_grad():
                probs = torch.sigmoid(outputs["logits"])
                preds = (probs > 0.5).sum(dim=1)
                acc = (preds == y).float().mean()
                train_acc_meter.update(acc.item(), x.size(0))

            loop.set_postfix({'reg_loss': f'{train_loss.avg:.4f}', 'tr_acc': f'{train_acc_meter.avg:.3f}'})

        scheduler_s2.step()

        # =============================================
        # 验证与保存逻辑 (保留原逻辑的核心部分)
        # =============================================
        val_metrics = validate_model(model, val_loader, device, num_classes)

        current_lr = optimizer_s2.param_groups[0]['lr']
        print(f"Total Epoch {current_epoch} | "
              f"S2 Loss: {train_loss.avg:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"LR: {current_lr:.1e}")

        # WandB 日志 (如果启用了)
        if wandb.run is not None:
            wandb.log({
                "train_loss": train_loss.avg,
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['acc'],
                "val_f1": val_metrics['f1'],
                "epoch": current_epoch
            })

        # --- Checkpoint 保存逻辑 ---
        val_f1_score = val_metrics['f1']

        # 1. 保存最佳模型 (Best F1)
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            patience_counter = 0  # 重置早停计数

            save_checkpoint(
                checkpoint_root_path, current_epoch, current_lr,
                optimizer_s2,  # 注意保存的是当前的优化器
                model,
                best_val_f1, fold, latest=False
            )
            print(f"[INFO] Best checkpoint saved! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience: {patience_counter}/{patience}")

        # 2. 早停逻辑
        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Stop at S2 epoch {epoch + 1} (Total {current_epoch})")
            break

        # 训练结束，保存 Latest 模型
    if mode == "RUN":
        # 注意：这里保存的 latest 是 Stage 2 结束时的状态
        save_checkpoint(checkpoint_root_path, total_epochs, 0.0, optimizer_s2, model, best_val_f1, fold, latest=True)
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



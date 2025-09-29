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
            outputs = model(x)
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

            outputs = model(x)
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
        filter(lambda p: p.requires_grad, model.parameters()),
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

    stage1_epochs = 5  # 阶段一：只训练主分类任务

    lambd1=0.01
    lambd2=0.03
    lambd3=0.05

    # -------------------
    # Stage 1: 基础模型训练
    # -------------------
    print(f"\n--- Starting Stage 1: Training base model for {stage1_epochs} epochs ---")
    for epoch in range(stage1_epochs):
        model.train()
        train_loss = AverageMeter()

        loop = tqdm(train_loader, desc=f'Stage1 Epoch {epoch + 1}/{stage1_epochs}', unit="batch")

        for x, y, video_idx, metadata in loop:
            x, y = x.to(device), y.to(device).long()
            optimizer.zero_grad()

            outputs = model(x)
            loss = coral_loss(outputs["logits"], y, num_classes)

            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), x.size(0))
            loop.set_postfix(loss=train_loss.avg)

        scheduler.step()

        val_metrics = validate_model(model, val_loader, device, num_classes)
        print(f"Stage 1 - Epoch {epoch + 1} | "
              f"Train Loss: {train_loss.avg:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        if epoch == stage1_epochs - 1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            save_checkpoint(checkpoint_root_path, epoch + 1, optimizer.param_groups[0]['lr'], optimizer, model,
                            best_val_f1, fold, latest=True)
            print("[INFO] Stage 1 finished. Saved model for Stage 2.")

    # -------------------
    # Stage 2: 因果解耦训练
    # -------------------
    print(f"\n--- Starting Stage 2: Causal disentanglement training ---")
    stage2_epochs = epochs - stage1_epochs
    loop = tqdm(range(stage2_epochs), desc=f'Stage2 Training (fold{fold})', unit="epoch")

    for epoch in loop:
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        all_preds = []
        all_labels = []

        batch_loop = tqdm(train_loader, desc=f'Stage2 Epoch {epoch + 1}/{stage2_epochs}', leave=False)

        for x, y, video_idx, metadata in batch_loop:
            x, y = x.to(device), y.to(device).long()
            metadata = metadata.to(device)

            optimizer.zero_grad()

            # 传递标签给 forward 方法，用于反事实损失计算
            outputs = model(x, labels=y,metadata=metadata)
            logits = outputs["logits"]

            # 计算所有损失项
            # 2.GRL对应的confound损失
            # confound_loss = lambd1 * coral_loss(outputs["confound_logits"], y, num_classes)

            confound_losses = []
            # Age prediction loss (MSE)

            age_data = metadata[:, 0]
            gender_data = metadata[:, 1]
            bmi_data = metadata[:, 2]
            height_data = metadata[:, 3]
            weight_data = metadata[:, 4]

            age_loss = F.mse_loss(outputs["age_preds"].squeeze(), age_data.float())
            confound_losses.append(age_loss)
            # Gender prediction loss (CrossEntropy)
            gender_loss = F.cross_entropy(outputs["gender_preds"], gender_data.long())
            confound_losses.append(gender_loss)
            # BMI prediction loss (MSE)
            bmi_loss = F.mse_loss(outputs["bmi_preds"].squeeze(), bmi_data.float())
            confound_losses.append(bmi_loss)
            height_loss = F.mse_loss(outputs["height_preds"].squeeze(),height_data.float())
            confound_losses.append(height_loss)
            weight_loss = F.mse_loss(outputs["weight_preds"].squeeze(),weight_data.float())
            confound_losses.append(weight_loss)

            # 将所有混淆损失加权求和
            total_confound_loss = sum(confound_losses) * lambd1

            # 3.重构对应的重构损失
            recon_loss = lambd2 * F.mse_loss(
                outputs["recon_features"].mean(dim=(1, 2)),
                outputs["original_features"].mean(dim=(1, 2))
            )
            # 4.反事实干预对应的损失
            # counterfactual_loss = 0
            # if outputs["counterfactual_logits"] is not None:
            #     shuffle_idx = outputs["shuffle_idx"]    # the shuffled indexes
            #     y_swapped = y[shuffle_idx]              # the label of it
            #     counterfactual_loss = lambd3 * F.mse_loss(outputs["counterfactual_logits"], y_swapped)

            counterfactual_loss = 0.0

            if outputs["counterfactual_logits"] is not None:
                shuffle_idx = outputs["shuffle_idx"]
                y_swapped = y[shuffle_idx]
                counterfactual_loss = lambd3 * coral_loss(outputs["counterfactual_logits"], y_swapped, num_classes)

            loss = coral_loss(logits, y, num_classes) + total_confound_loss + recon_loss + counterfactual_loss

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), x.size(0))

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
                batch_acc = (preds == y).float().mean().item()
                train_acc.update(batch_acc, x.size(0))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            batch_loop.set_postfix(loss=train_loss.avg, acc=train_acc.avg)

        scheduler.step()

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        train_f1_score = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        val_metrics = validate_model(model, val_loader, device, num_classes)

        print(f"Epoch {epoch + 1 + stage1_epochs} | "
              f"Train Loss: {train_loss.avg:.4f} | "
              f"Train Acc: {train_acc.avg:.4f} | "
              f"Train F1: {train_f1_score:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        val_f1_score = val_metrics['f1']
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            patience_counter = 0
            save_checkpoint(checkpoint_root_path, epoch + 1 + stage1_epochs, optimizer.param_groups[0]['lr'],
                            optimizer, model,
                            best_val_f1, fold, latest=False)
            print(
                f"[INFO] Best checkpoint saved at epoch {epoch + 1 + stage1_epochs} with val_f1_score={val_f1_score:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. patience_counter = {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(
                f"[EARLY STOPPING] Stop training at epoch {epoch + 1 + stage1_epochs} | best val_f1={best_val_f1:.4f}")
            break


    lr_backbone = optimizer.param_groups[0]['lr']
    if mode == "RUN":
        save_checkpoint(checkpoint_root_path, epoch, lr_backbone, optimizer, model, None, fold, latest=True)
        print(f'[INFO] Latest checkpoint saved at: {checkpoint_root_path}')



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
            "lr": 1e-05,
            "num_epochs": 20,
            "num_hidden_layers": 2,
            "layer_sizes": [256, 50, 16, 3],
            "optimizer": 'RMSprop',
            "use_weighted_loss": True,
            "batch_size": 32,
            "dropout_rate": 0.1,
            'weight_decay': 0.00057,
            'momentum': 0.66
        }
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



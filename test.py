import os
import datetime
import pickle
import pandas as pd
import torch
from torch import nn
import wandb

import pkg_resources
from sklearn.metrics import classification_report, confusion_matrix

from data.dataloaders import dataset_factory
from model.motion_encoder import MotionEncoder
from model.backbone_loader import load_pretrained_backbone, count_parameters, load_pretrained_weights
from train import train_model, final_test
from utility import utils
from const import path
from eval_encoder import log_results
from stat_analysis.get_stats import get_stats


def setup_experiment_path(params):
    exp_path = path.OUT_PATH + os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    params['model_prefix'] = os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    rep_out = path.OUT_PATH + os.path.join(params['model_prefix'])
    return params, rep_out


def initialize_wandb(params):
    wandb.init(name=params['wandb_name'], project='MotionEncoderEvaluator_PD', settings=wandb.Settings(start_method='fork'))
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update(params)
    wandb.config.update({'installed_packages': installed_packages})


def map_to_classifier_dim(backbone_name, option):
    classifier_dims = {
        'poseformer': {'option1': []},
        'motionbert': {'option1': []},
        'poseformerv2': {'option1': []},
        'mixste': {'option1': []},
        'motionagformer': {'option1': []},
        'ctrgcn': {'option1': []}
    }
    return classifier_dims[backbone_name][option]


def configure_params_for_best_model(params, backbone_name):             #TODO:[GCN]这个函数到底有没有用上？
    if backbone_name == 'ctrgcn':
        best_params = {     # ⚠️这些参数是否合理呢？
            "lr": 1e-05,    #0807调参：似乎有点太大了，从原来的0.1调整到0.001
            "num_epochs": 20,
            "batch_size": 128,
            "optimizer": 'AdamW',
            "weight_decay": 0.00057,
            "momentum": 0.66,
            "dropout_rate": 0.1,  # 0807调参：保持和下面一致
            "use_weighted_loss": True  # 0807调参修改为True
        }

    elif backbone_name== 'poseformerv2':
        best_params = {
            "lr": 1e-05,            #这个其实是分类头的学习率
            "num_epochs": 20,
            "num_hidden_layers": 2,
            "layer_sizes": [256, 50, 16, 3],
            "optimizer": 'AdamW',           #从原本的RMSprop修改而来，涨了0.04
            "use_weighted_loss": True,
            "batch_size": 128,
            "dropout_rate": 0.1,
            'weight_decay': 0.00057,
            'momentum': 0.66
        }
    elif backbone_name == 'motionbert':     #跑不起来，炸显存严重
        best_params = {
            "lr": 1e-05,
            "num_epochs": 20,
            "num_hidden_layers": 2,
            "layer_sizes": [256, 50, 16, 3],
            "optimizer": 'RMSprop',
            "use_weighted_loss": True,
            "batch_size": 4,
            "dropout_rate": 0.1,
            'weight_decay': 0.00057,
            'momentum': 0.66
        }
    # elif backbone_name == 'motionagformer':
    #     best_params = {
    #         "lr": 1e-05,
    #         "num_epochs": 20,
    #         "num_hidden_layers": 2,
    #         "layer_sizes": [256, 50, 16, 3],
    #         "optimizer": 'RMSprop',
    #         "use_weighted_loss": True,
    #         "batch_size": 32,
    #         "dropout_rate": 0.1,
    #         'weight_decay': 0.00057,
    #         'momentum': 0.66
    #     }

    elif backbone_name == 'motionagformer':
        best_params = {
            "lr": 5e-05,  # 稍微加快收敛速度，但不过冲
            "num_epochs": 30,  # 多给点epoch，让增强样本有机会训练到
            "num_hidden_layers": 2,
            "layer_sizes": [128, 32, 8, 3],  # 降低模型容量，减少过拟合
            "optimizer": 'AdamW',  # 对小样本泛化稳定
            "use_weighted_loss": True,
            "batch_size": 16,  # 较小批量，增加梯度更新频率
            "dropout_rate": 0.4,  # 明显提高Dropout防过拟合
            "weight_decay": 0.001,  # L2正则更强
            "momentum": 0.9,  # 这里即使AdamW不用也可以留着给兼容
            "rotation_prob": 0.5,  # 增强概率提高
            "mirror_prob": 0.5,
            "noise_prob": 0.4,
            "axis_mask_prob": 0.3,
            "rotation_range": (-15, 15)  # 限制旋转幅度，防止过大扰动
        }
    #print_best_model_configuration(best_params, backbone_name) #KeyError: 'best_trial_number'
    update_params_with_best(params, best_params, backbone_name)
    return params


def print_best_model_configuration(best_params, backbone_name):
    print("====================================BEST MODEL====================================================")
    print(f"Trial {best_params['best_trial_number']}, lr: {best_params['lr']}, num_epochs: {best_params['num_epochs']}")
    print(f"classifier_hidden_dims: {map_to_classifier_dim(backbone_name, 'option1')}")
    print(f"optimizer_name: {best_params['optimizer']}, use_weighted_loss: {best_params['use_weighted_loss']}")
    print("========================================================================================")



def update_params_with_best(params, best_params, backbone_name):
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


def run_fold_tests(params, all_folds, backbone_name, device, rep_out):
    splits = setup_datasets(params, backbone_name, all_folds)
    return run_tests_for_each_fold(params, splits, backbone_name, device, rep_out)


def setup_datasets(params, backbone_name, all_folds):
    splits = []
    for fold in all_folds:
        dataset_result = dataset_factory(params, backbone_name, fold)
        if dataset_result is None:
            continue
        train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_result
        splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))
    return splits


def run_tests_for_each_fold(params, splits, backbone_name, device, rep_out):
    total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits):
        process_fold(fold, params, backbone_name, train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights, device, total_outs_best, total_gts, total_logits, total_states, total_video_names, total_outs_last, rep_out)
    return total_outs_best, total_gts, total_states, total_video_names, total_outs_last


def process_fold(fold, params, backbone_name, train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights, device, total_outs_best, total_gts, total_logits, total_states, total_video_names, total_outs_last, rep_out):
    start_time = datetime.datetime.now()
    params['input_dim'] = train_dataset_fn.dataset._pose_dim                #这个参数对于CTR-GCN来说无用，因为它不展平向量
    params['pose_dim'] = train_dataset_fn.dataset._pose_dim                 #同上，无用
    params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS

    model_backbone = load_pretrained_backbone(params, backbone_name)
    model = MotionEncoder(backbone=model_backbone,
                            params=params,
                            num_classes=params['num_classes'],
                            num_joints=params['num_joints'],
                            train_mode=params['train_mode'])
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if fold == 1:
        model_params = count_parameters(model)
        print(f"[INFO] Model has {model_params} parameters.")
    
    train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name)    #Layer2的入口
    
    checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'],'models', f"fold{fold}")
    best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
    load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)['model'])      #这里报错了，只在第一个fold上跑完了20个epoch，只保存了latest_epoch.pth.tr，所以加载不出来
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
    res_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'results')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    utils.save_json(os.path.join(res_dir, 'results_Best_fold{}.json'.format(fold)), attributes, names)

    total_logits.extend(logits)
    attributes = [total_logits, total_gts]

    logits_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'logits')
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
    
    res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last, 'total_gts':total_gts, 'total_states':total_states})
    with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
        pickle.dump(res, file)
    
    end_time = datetime.datetime.now()
    
    duration = end_time - start_time
    print(f"Fold {fold} run time:", duration)


def calculate_metrics(outputs, targets, states, phase, report_prefix, output_dir):      #分开ON和OFF进行统计
    # Filter outputs and targets based on the phase ('ON' or 'OFF')
    filtered_gts = [gt for gt, state in zip(targets, states) if state == phase]
    filtered_outs = [out for out, state in zip(outputs, states) if state == phase]

    if not filtered_gts:  # 没有该 phase 的样本
        print(f"[WARNING] Phase '{phase}' not found in dataset. Skipping...")
        print(f"[DEBUG] All unique states in current fold: {set(states)}")
        return

    report = classification_report(filtered_gts, filtered_outs)
    confusion = confusion_matrix(filtered_gts, filtered_outs)

    log_results(
        report, confusion, 
        f'{report_prefix}_allfolds_{phase}.txt', 
        f'{report_prefix}_confusion_matrix_allfolds_{phase}.png', 
        output_dir
    )

def process_reports(outputs_best, outputs_last, targets, states, output_dir):
    # Process reports for 'best' and 'last' data
    for prefix, outputs in [('best', outputs_best), ('last', outputs_last)]:
        print(f"=========={prefix.upper()} REPORTS============")
        # Full dataset metrics
        report_final = classification_report(targets, outputs)          #总的结果（不区分ON/OFF）
        confusion_final = confusion_matrix(targets, outputs)
        log_results(report_final, confusion_final, f'{prefix}_report_allfolds.txt', f'{prefix}_confusion_matrix_allfolds.png', output_dir)

        # 'ON' and 'OFF' group metrics
        for phase in ['ON', 'OFF']:
            calculate_metrics(outputs, targets, states, phase, prefix, output_dir)

def save_and_load_results(video_names, outputs_best, outputs_last, targets, output_dir):
    results = pd.DataFrame({
        'total_video_names': video_names,
        'total_outs_best': outputs_best,
        'total_outs_last': outputs_last,
        'total_gts': targets
    })
    results_path = os.path.join(output_dir, 'final_results.pkl')
    with open(results_path, 'wb') as file:
        pickle.dump(results, file)

    with open(results_path, 'rb') as file:
        loaded_results = pickle.load(file)
    
    total_video_names = loaded_results['total_video_names']
    total_outs_best = loaded_results['total_outs_best']
    total_outs_last = loaded_results['total_outs_last']
    
    get_stats(total_video_names, total_outs_best, output_dir, 'best')
    get_stats(total_video_names, total_outs_last, output_dir, 'last')


def test_and_report(params, new_params, all_folds, backbone_name, device):          #new_params没用到？怎么回事？
    params, rep_out = setup_experiment_path(params)
    params = configure_params_for_best_model(params, backbone_name)
    initialize_wandb(params)
    total_outs_best, total_gts, total_states, total_video_names, total_outs_last = run_fold_tests(params, all_folds, backbone_name, device, rep_out)
    process_reports(total_outs_best, total_outs_last, total_gts, total_states, rep_out)
    save_and_load_results(total_video_names, total_outs_best, total_outs_last, total_gts, rep_out)
    wandb.finish()

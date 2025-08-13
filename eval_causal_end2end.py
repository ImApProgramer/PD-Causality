import os
import sys
import argparse
import datetime
from collections import defaultdict

import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torch import nn
from torch import optim
from tqdm import tqdm

import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from configs import generate_config_motionagformer
from data.augmentations import RandomNoise, RandomRotation, MirrorReflection, axis_mask
from data.dataloaders import PDReader, MotionAGFormerPreprocessor, PreserveKeysTransform, collate_fn

from const import path
from learning.utils import compute_class_weights, AverageMeter
from utility.utils import set_random_seed
from test import update_params_with_best, setup_datasets
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


def log_results(rep, confusion, rep_name, conf_name, out_p):
    print(rep)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, ax=ax, cmap="Blues", fmt='g', annot_kws={"size": 26})
    ax.set_xlabel('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    ax.set_title('Confusion Matrix', fontsize=30)
    ax.xaxis.set_ticklabels(['class 0', 'class 1', 'class 2'], fontsize=22)  # Modify class names as needed
    ax.yaxis.set_ticklabels(['class 0', 'class 1', 'class 2'], fontsize=22)
    # Save the figure
    plt.savefig(os.path.join(out_p, conf_name))
    plt.close(fig)
    with open(os.path.join(out_p, rep_name), "w") as text_file:
        text_file.write(rep)

    artifact = wandb.Artifact(f'confusion_matrices', type='image-results')
    artifact.add_file(os.path.join(out_p, conf_name))
    wandb.log_artifact(artifact)

    artifact = wandb.Artifact('reports', type='txtfile-results')
    artifact.add_file(os.path.join(out_p, rep_name))
    wandb.log_artifact(artifact)


def setup_experiment_path(params):
    exp_path = path.CAUSAL_OUT_PATH + os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    params['model_prefix'] = os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    rep_out = path.CAUSAL_OUT_PATH + os.path.join(params['model_prefix'])
    return params, rep_out




def configure_params_for_best_model(params, backbone_name):

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
    #print_best_model_configuration(best_params, backbone_name) #KeyError: 'best_trial_number'
    update_params_with_best(params, best_params, backbone_name)
    return params


def initialize_wandb(params):
    wandb.init(name=params['wandb_name'], project='Causal_PD', settings=wandb.Settings(start_method='fork'))
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update(params)
    wandb.config.update({'installed_packages': installed_packages})


def assert_backbone_is_supported(backbone_data_location_mapper, backbone):
    if backbone not in backbone_data_location_mapper:
        raise NotImplementedError(f"Backbone '{backbone}' is not supported.")



class ProcessedDataset(data.Dataset):
    def __init__(self, data_dir, params=None, mode='train', fold=1, downstream='pd', transform=None):    #这里加了个gcn_mode来单独控制，力求保留原有兼容性
        super(ProcessedDataset, self).__init__()
        self._params = params
        self._mode = mode
        self.data_dir = data_dir
        self._task = downstream

        self.backbone = self._params['backbone']
        if self.backbone=='ctrgcn':
            self.gcn_mode=True
        else:
            self.gcn_mode=False

        # 这里要分开进行处理了
        if self.gcn_mode == True:
            self._NMAJOR_JOINTS = len(_GCN_JOINTS)
        else:
            self._NMAJOR_JOINTS = len(_MAJOR_JOINTS)


        if self._task == 'pd':
            self._updrs_str = ['normal', 'slight', 'moderate']  # , 'severe']
            self._TOTAL_SCORES = _TOTAL_SCORES

        self.fold = fold
        self.transform = transform

        self.poses, self.labels, self.video_names, self.metadata = self.load_data()
        self.video_name_to_index = {name: index for index, name in enumerate(self.video_names)}


        self._pose_dim = 3 * self._NMAJOR_JOINTS

    def load_data(self):
        dataset_name = self._params['dataset']

        if self._mode == 'train':
            train_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_train_{self.fold}.pkl"), "rb"))

        elif self._mode == 'test':
            test_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_test_{self.fold if self._mode != 'test_all' else 'all'}.pkl"), "rb"))

        elif self._mode == 'val':
            val_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_validation_{self.fold}.pkl"), "rb"))
        elif self._mode == 'train-eval':
            train_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_train_{self.fold}.pkl"), "rb"))
            val_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_validation_{self.fold}.pkl"), "rb"))
            train_data = {
                'pose': [*train_data['pose'], *val_data['pose'],],
                'label': [*train_data['label'], *val_data['label']],
                'video_name': [*train_data['video_name'], *val_data['video_name']]
            }
        elif self._mode == 'test_all':
            test_data = pickle.load(open(os.path.join(self.data_dir, f"{dataset_name}_all.pkl"), "rb"))


        if self._mode == 'train':
            poses, labels, video_names, metadatas = self.data_generator(train_data, mode='train', fold_number=self.fold)
        elif self._mode == 'test':
            poses, labels, video_names, metadatas = self.data_generator(test_data)
        elif self._mode == 'val':
            poses, labels, video_names, metadatas = self.data_generator(val_data)
        elif self._mode == 'train-eval':
            poses, labels, video_names, metadatas = self.data_generator(train_data, mode='train', fold_number=self.fold)
        else:
            poses, labels, video_names, metadatas = self.data_generator(test_data)

        return poses, labels,video_names, metadatas

    @staticmethod
    def data_generator(data, mode='test', fold_number=1):
        poses = []
        labels = []
        video_names = []
        metadatas = []

        # bootstrap_number = 3
        # num_samples = 39

        for i in range(len(data['pose'])):
            pose = np.copy(data['pose'][i])
            label = data['label'][i]
            poses.append(pose)
            labels.append(label)
            video_names.append(data['video_name'][i])
            metadata = data['metadata'][i]
            metadatas.append(metadata)
        # can't stack poses because not all have equal frames
        labels = np.stack(labels)
        video_names = np.stack(video_names)
        metadatas = np.stack(metadatas)     #这里metadata是有值的，为啥到了train.py里面就没有了？

        # For using a subset of the dataset (few-shot)
        # if mode == 'train':
        #   sampling_dir = 'PATH/TO/BOOTSTRAP_SAMPLING_DIR'
        #   all_clip_video_names = pickle.load(open(sampling_dir + "all_clip_video_names.pkl", "rb"))
        #   clip_video_names = all_clip_video_names[fold_number - 1]

        #   all_bootstrap_samples = pickle.load(open(sampling_dir + f'{num_samples}_samples/bootstrap_{bootstrap_number}_samples.pkl', 'rb'))
        #   bootstrap_samples = all_bootstrap_samples[fold_number - 1]

        #   mask_list = [1 if video_name in bootstrap_samples else 0 for video_name in clip_video_names]
        #   train_indices = [train_idx for train_idx, mask_value in enumerate(mask_list) if mask_value == 1]

        #   X_1 = [X_1[train_idx] for train_idx in train_indices]
        #   Y = Y[train_indices]

        return poses, labels, video_names, metadatas

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for the training mode."""
        x = self.poses[idx]         #如果是GCN，这里将是(3, T, V, 1)
        label = self.labels[idx]
        video_idx = self.video_name_to_index[self.video_names[idx]]


        if not self.gcn_mode:       #原有的非gcn处理逻辑
            if self._params['data_type'] != "GastNet":
                joints = self._get_joint_orders()
                x = x[:, joints, :]

            if self._params['in_data_dim'] == 2:
                if self._params['simulate_confidence_score']:
                    # TODO: Confidence score should be a function of depth (probably)
                    x[..., 2] = 1  # Consider 3rd dimension as confidence score and set to be 1.
                else:
                    x = x[..., :2]  # Make sure it's two-dimensional
            elif self._params['in_data_dim'] == 3:
                x = x[..., :3]  # Make sure it's 3-dimensional

            if self._params['merge_last_dim']:
                N = np.shape(x)[0]
                x = x.reshape(N, -1)  # N x 17 x 3 -> N x 51

            x = np.array(x, dtype=np.float32)

            if x.shape[0] > self._params['source_seq_len']:
                # If we're reading a preprocessed pickle file that has more frames
                # than the expected frame length, we throw away the last few ones.
                x = x[:self._params['source_seq_len']]
            elif x.shape[0] < self._params['source_seq_len']:
                raise ValueError("Number of frames in tensor x is shorter than expected one.")

        # 如果是 GCN 模式，直接裁剪后返回 (C, T, V, 1)，不做额外处理
        else:
            x = np.array(x, dtype=np.float32)  # 确保是 numpy array
            if x.shape[1] > self._params['source_seq_len']:  # 检查时间维度 T
                x = x[:, :self._params['source_seq_len'], :, :]  # 裁剪为 (3, T_crop, V, 1)
            elif x.shape[1] < self._params['source_seq_len']:
                raise ValueError(f"Sequence length {x.shape[1]} < required {self._params['source_seq_len']}")

        #处理metadata
        if len(self._params['metadata']) > 0:
            metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
            md = self.metadata[idx][0][metadata_idx].astype(np.float32)
        else:
            md = []

        sample = {
            'encoder_inputs': x,        # GCN 模式：(3, T, V, 1)；非 GCN 模式：(T, V*C) 或 (T, V, C)
            'label': label,
            'labels_str': self._updrs_str[label],
            'video_idx': video_idx,
            'metadata': md,
        }
        # if self.transform:
        #     sample = self.transform(sample)
        return sample

    def _get_joint_orders(self):
        joints = _MAJOR_JOINTS
        return joints

def dataset_factory(params, backbone, fold):

    root_dir = f'{path.CAUSAL_PREPROCESSED_DATA_ROOT_PATH}/{backbone}_processing'       #backbone应为motionagformer

    backbone_data_location_mapper = {  # backbone特定的数据处理目录
        'motionagformer': os.path.join(root_dir, params['experiment_name'],
                                       f"{params['dataset']}_center_{params['data_centered']}/")
    }

    backbone_preprocessor_mapper = {
        'motionagformer': MotionAGFormerPreprocessor
    }

    assert_backbone_is_supported(backbone_data_location_mapper, backbone)

    data_dir = backbone_data_location_mapper[backbone]

    if not os.path.exists(
            data_dir):
        if params['dataset'] == 'PD':
            raw_data = PDReader(params['data_path'], params['labels_path'])
            Preprocessor = backbone_preprocessor_mapper[backbone]
            Preprocessor(data_dir, raw_data, params)
        else:
            raise NotImplementedError(f"dataset '{params['dataset']}' is not supported.")

    use_validation = params['use_validation']



    train_transform = transforms.Compose([
        PreserveKeysTransform(
            transforms.RandomApply([MirrorReflection(format='ntu25', data_dim=3)], p=params['mirror_prob'])),
        # data_dim=3根本没用到，随便传的
        PreserveKeysTransform(
            transforms.RandomApply([RandomRotation(*params['rotation_range'], data_dim=3)],
                                   p=params['rotation_prob'])),
        PreserveKeysTransform(
            transforms.RandomApply([RandomNoise(data_dim=3)], p=params['noise_prob'])),
        PreserveKeysTransform(
            transforms.RandomApply([axis_mask(data_dim=3)], p=params['axis_mask_prob']))
    ])


    train_dataset = ProcessedDataset(data_dir, fold=fold, params=params,
                                     mode='train' if use_validation else 'train-eval',
                                     transform=train_transform)  # 这里传入的时候params根本没有metadata这一项，导致报错
    eval_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='val') if use_validation else None
    test_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='test')



    train_dataset_fn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    eval_dataset_fn = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    ) if use_validation else None

    test_dataset_fn = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    class_weights = compute_class_weights(train_dataset_fn)  # MotionAGFormer多次运行时，出错

    return train_dataset_fn, test_dataset_fn, eval_dataset_fn, class_weights  # ,train_dataset #用于可视化



def load_pretrained_weights(model, checkpoint, strict=True):

    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    model_first_key = next(iter(model_dict))
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if not 'module.' in model_first_key:
            if k.startswith('module.'):
                k = k[7:]
        if k in model_dict:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    if(strict):
        model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict)
    print(f'[INFO] (load_pretrained_weights) {len(matched_layers)} layers are loaded')
    print(f'[INFO] (load_pretrained_weights) {len(discarded_layers)} layers are discared')
    if len(matched_layers) == 0:
        print ("--------------------------model_dict------------------")
        print (model_dict.keys())
        print ("--------------------------discarded_layers------------------")
        print (discarded_layers)
        raise NotImplementedError(f"Loading problem!!!!!!")


def load_pretrained_backbone(params, backbone_name):
    if backbone_name == "motionagformer":
        model_backbone = MotionAGFormer(n_layers=params['n_layers'],
                               dim_in=params['dim_in'],
                               dim_feat=params['dim_feat'],
                               dim_rep=params['dim_rep'],
                               dim_out=params['dim_out'],
                               mlp_ratio=params['mlp_ratio'],
                               act_layer=nn.GELU,
                               attn_drop=params['attn_drop'],
                               drop=params['drop'],
                               drop_path=params['drop_path'],
                               use_layer_scale=params['use_layer_scale'],
                               layer_scale_init_value=params['layer_scale_init_value'],
                               use_adaptive_fusion=params['use_adaptive_fusion'],
                               num_heads=params['num_heads'],
                               qkv_bias=params['qkv_bias'],
                               qkv_scale=params['qkv_scale'],
                               hierarchical=params['hierarchical'],
                               num_joints=params['num_joints'],
                               use_temporal_similarity=params['use_temporal_similarity'],
                               temporal_connection_len=params['temporal_connection_len'],
                               use_tcn=params['use_tcn'],
                               graph_only=params['graph_only'],
                               neighbour_num=params['neighbour_num'],
                               n_frames=params['source_seq_len'])
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage)['model']

    else:
        raise Exception("Undefined backbone type.")


    load_pretrained_weights(model_backbone, checkpoint)
    return model_backbone


def count_parameters(model):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    return model_params


def choose_optimizer(model, params):
    optimizer_name = params['optimizer']
    try:
        backbone_params = set(model.module.backbone.parameters())
        head_params = set(model.module.head.parameters())
    except AttributeError:
        backbone_params = set(model.backbone.parameters())
        head_params = set(model.head.parameters())

    all_params = set(model.parameters())
    other_params = all_params - backbone_params - head_params

    param_groups = [
        {"params": filter(lambda p: p.requires_grad, backbone_params), "lr": params['lr_backbone']},
        {"params": filter(lambda p: p.requires_grad, head_params), "lr": params['lr_head']},
        {"params": filter(lambda p: p.requires_grad, other_params), "lr": params['lr_head']}
    ]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=params['weight_decay'])
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(param_groups, weight_decay=params['weight_decay'])
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(param_groups, momentum=params.get('momentum', 0.9))
    else:
        raise ModuleNotFoundError("Optimizer not found")

    return optimizer

def train_model(params, class_weights, train_loader, val_loader, model, fold, backbone_name, mode="RUN"):

    criterion = CounterfactualLoss(
        lambda_consistency=0.5,
        lambda_vae=0.1,
        lambda_disentangle=0.2
    )

    if torch.cuda.is_available():
        criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, causal_model.parameters()),
        lr=params.get('lr', 1e-4),
        weight_decay=params.get('weight_decay', 1e-4)
    )
    scheduler = StepLR(optimizer, step_size=params['lr_step_size'], gamma=params['lr_decay'])

    params['epochs']= 20

    loop = tqdm(range(params['epochs']), desc=f'Training (fold{fold})', unit="epoch")
    best_val_f1 = 0.0  # 最佳的f1分数

    for epoch in loop:
        # # Phase 1: 前20epoch只训练事实分支
        # if epoch < 20:
        #     model.use_vae_generator = False
        #     model.use_disentanglement_loss = False
        #     loss_weights = {'lambda_consistency': 0, 'lambda_vae': 0, 'lambda_disentangle': 0}
        # # Phase 2: 引入反事实训练
        # elif epoch < 40:
        #     model.use_vae_generator = True
        #     model.use_disentanglement_loss = False
        #     loss_weights = {'lambda_consistency': 0.3, 'lambda_vae': 0.1, 'lambda_disentangle': 0}
        # # Phase 3: 全目标训练
        # else:
        #     model.use_vae_generator = True
        #     model.use_disentanglement_loss = True
        #     loss_weights = {'lambda_consistency': 0.5, 'lambda_vae': 0.1, 'lambda_disentangle': 0.2}
        #
        # criterion = CounterfactualLoss(**loss_weights)

        # 训练步骤...
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        video_predictions = defaultdict(list)

        video_labels = {}

        for x, y, video_idx, metadata in train_loader:
            y = y.float().to(device)  # 回归任务需要float标签

            x=x.to(device)       #y是label，即分数
            metadata = metadata.to(device)
            batch_size = x.shape[0]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with causal modeling
            outputs = model(x, labels=y, intervention_type='vae_sampling')      #用因果Encoder Forward出一个结果来，是一个字典

            # Compute loss
            losses = criterion(outputs, y)
            total_loss = losses['total_loss']

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Update metrics
            train_loss.update(total_loss.item(), batch_size)

            # Track predictions per video for accuracy calculation
            for i, idx in enumerate(video_idx):
                video_predictions[idx.item()].append(outputs['factual_pred'][i].detach())
                video_labels[idx.item()] = y[i].item()


        scheduler.step()
        print("Losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})

    # optimizer = choose_optimizer(model, params)
    # scheduler = choose_scheduler(optimizer, params)
    #
    # checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'], 'models')
    # if not os.path.exists(checkpoint_root_path): os.makedirs(checkpoint_root_path)  # 原本的mkdir只能创建单级目录
    #
    # loop = tqdm(range(params['epochs']), desc=f'Training (fold{fold})', unit="epoch")
    #
    # best_val_f1 = 0.0  # 最佳的f1分数
    #
    # for epoch in loop:
    #     # print(f"[INFO] epoch {epoch}")
    #     train_acc = AverageMeter()
    #     train_loss = AverageMeter()
    #
    #     model.train()
    #
    #     video_predictions = defaultdict(list)
    #     video_labels = {}
    #
    #     epoch_start_time = time.time()
    #     for x, y, video_idx, metadata in train_loader:
    #         x, y = x.to(device), y.to(device)
    #         metadata = metadata.to(device)  # metadata是空的！已经修复
    #
    #         batch_size = x.shape[0]
    #         optimizer.zero_grad()
    #
    #         if params['medication']:
    #             vi = video_idx.tolist()
    #             vn = [train_loader.dataset.video_names[i] for i in vi]
    #             on_off = [1 if 'on' in name else 0 for name in vn]
    #             on_off = torch.tensor(on_off, dtype=torch.float32, device=device)
    #             out = model(x, metadata, on_off)
    #         else:
    #             out = model(x, metadata)
    #
    #         loss = criterion(out, y)
    #         train_loss.update(loss.item(), batch_size)
    #
    #         for i, idx in enumerate(video_idx):
    #             video_predictions[idx.item()].append(out[i].detach())
    #             video_labels[idx.item()] = y[i].item()
    #
    #         if params['lambda_l1'] > 0:
    #             learnable_params = torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])
    #             l1_regularization = torch.norm(learnable_params, p=1)
    #
    #             loss += params['lambda_l1'] * l1_regularization
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #     epoch_time = time.time() - epoch_start_time
    #     print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    #     # Compute accuracy per video
    #     total_correct = 0
    #     total_videos = 0
    #     for video_idx, predictions in video_predictions.items():
    #         video_prediction = torch.stack(predictions).mean(dim=0).unsqueeze(0)
    #         video_label = torch.tensor([video_labels[video_idx]], device=video_prediction.device)
    #
    #         acc, = accuracy(video_prediction, video_label)
    #         total_correct += acc
    #         total_videos += 1
    #
    #     video_accuracy = total_correct / total_videos
    #     train_acc.update(video_accuracy, total_videos)
    #
    #     val_loss, val_acc, val_f1_score = validate_model(model, val_loader, params, class_weights)
    #
    #     lr_backbone = optimizer.param_groups[0]['lr']
    #
    #     if scheduler:
    #         scheduler.step()
    #
    #     loop.set_postfix(train_loss=train_loss.avg, train_accuracy=train_acc.avg,
    #                      val_loss=val_loss, val_accuracy=val_acc, val_f1_score=val_f1_score)
    #
    #     log_wandb(epoch, fold, lr_backbone, train_acc, train_loss, 1, val_acc,
    #               val_loss, val_f1_score)
    #
    #     if val_f1_score > best_val_f1:  # best模型的保存逻辑
    #         best_val_f1 = val_f1_score
    #         save_checkpoint(checkpoint_root_path, epoch, lr_backbone, optimizer, model,
    #                         best_val_f1, fold, latest=False)
    #         print(f"[INFO] Best checkpoint saved at epoch {epoch} with val_f1_score={val_f1_score:.4f}")
    #
    # if mode == "RUN":
    #     save_checkpoint(checkpoint_root_path, epoch, lr_backbone, optimizer, model, None, fold, latest=True)
    #     print(f'[INFO] Latest checkpoint saved at: {checkpoint_root_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( poseformer, ''motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only',
                        help='train mode( end2end, classifier_only, causal )')
    parser.add_argument('--dataset', type=str, default='PD', help='**currently code only works for PD')
    parser.add_argument('--data_path', type=str, default=path.PD_PATH_POSES)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int,
                        help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--last_run_foldnum', default='7', type=str)
    parser.add_argument('--readstudyfrom', default=1, type=int)
    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str, help="add metadata prob to the training 'gender,age,bmi,height,weight'")

    # parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    # parser.add_argument('--metadata', default='', type=str,
    #                     help="add metadata prob to the training 'gender,age,bmi,height,weight'")

    args = parser.parse_args()

    param = vars(args)
    param['metadata'] = param['metadata'].split(',') if param['metadata'] else []   #这里显式建立了metadata项，但是在可视化中却没有



    torch.backends.cudnn.benchmark = False

    backbone_name = param['backbone']


    if backbone_name == 'motionagformer':
        conf_path = './configs/motionagformer'

    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")

    # 获取排序后的文件列表并只取第一个文件
    file_list = sorted(os.listdir(conf_path))
    if file_list:  # 确保列表不为空
        fi = file_list[0]  # 只取第一个文件
        params, new_params = generate_config_motionagformer.generate_config(param, fi)

        if param['dataset'] == 'PD':
            num_folds = 23
            params['num_classes'] = 3
        else:
            raise NotImplementedError(f"dataset '{param['dataset']}' is not supported.")

        all_folds = range(1, num_folds + 1)
        set_random_seed(param['seed'])

        # test_and_report(params, new_params, all_folds, backbone_name, _DEVICE)

        params, rep_out = setup_experiment_path(params)
        configure_params_for_best_model(params, backbone_name)
        initialize_wandb(params)

        splits = []

        for fold in all_folds:
            train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name,
                                                                                               fold)
            splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))

        total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
        device=_DEVICE

        for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits):
            start_time = datetime.datetime.now()
            params['input_dim'] = train_dataset_fn.dataset._pose_dim  # 这个参数对于CTR-GCN来说无用，因为它不展平向量
            params['pose_dim'] = train_dataset_fn.dataset._pose_dim  # 同上，无用
            params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS

            model_backbone = load_pretrained_backbone(params, backbone_name)        #motionagformer加载完毕

            causal_model = CounterfactualCausalModeling(
                backbone=model_backbone,
                input_dim=512,
                hidden_dim=256,
                z_dim=128,
                counterfactual_strategy='vae_sampling',
                use_vae_generator=True,
                use_disentanglement_loss=False
            )

            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs!")
                causal_model = nn.DataParallel(causal_model)
            if fold == 1:
                model_params = count_parameters(causal_model)
                print(f"[INFO] Model has {model_params} parameters.")

            if torch.cuda.is_available():
                causal_model = causal_model.to(device)

            else:
                raise Exception("Cuda is not enabled")

            train_model(params, class_weights, train_dataset_fn, val_dataset_fn, causal_model, fold, backbone_name)


            # model = MotionEncoder(backbone=model_backbone,
            #                       params=params,
            #                       num_classes=params['num_classes'],
            #                       num_joints=params['num_joints'],
            #                       train_mode=params['train_mode'])
            # model = model.to(device)
            # if torch.cuda.device_count() > 1:
            #     print("Using", torch.cuda.device_count(), "GPUs!")
            #     model = nn.DataParallel(model)
            # if fold == 1:
            #     model_params = count_parameters(model)
            #     print(f"[INFO] Model has {model_params} parameters.")
            #
            # train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold,
            #             backbone_name)  # Layer2的入口
            #
            # checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'], 'models', f"fold{fold}")
            # best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
            # load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)[
            #     'model'])  # 这里报错了，只在第一个fold上跑完了20个epoch，只保存了latest_epoch.pth.tr，所以加载不出来
            # model.cuda()
            # outs, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
            # total_outs_best.extend(outs)
            # total_gts.extend(gts)
            # total_states.extend(states)
            # total_video_names.extend(video_names)
            # print(f'fold # of test samples: {len(video_names)}')
            # print(f'current sum # of test samples: {len(total_video_names)}')
            # attributes = [total_outs_best, total_gts]
            # names = ['predicted_classes', 'true_labels']
            # res_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'results')
            # if not os.path.exists(res_dir):
            #     os.makedirs(res_dir)
            # utils.save_json(os.path.join(res_dir, 'results_Best_fold{}.json'.format(fold)), attributes, names)
            #
            # total_logits.extend(logits)
            # attributes = [total_logits, total_gts]
            #
            # logits_dir = path.OUT_PATH + os.path.join(params['model_prefix'], 'logits')
            # if not os.path.exists(logits_dir):
            #     os.makedirs(logits_dir)
            # utils.save_json(os.path.join(logits_dir, 'logits_Best_fold{}.json'.format(fold)), attributes, names)
            #
            # last_ckpt_path = os.path.join(checkpoint_root_path, 'latest_epoch.pth.tr')
            # load_pretrained_weights(model, checkpoint=torch.load(last_ckpt_path)['model'])
            # model.cuda()
            # outs_last, gts, logits, states, video_names = final_test(model, test_dataset_fn, params)
            # total_outs_last.extend(outs_last)
            # attributes = [total_outs_last, total_gts]
            # utils.save_json(os.path.join(res_dir, 'results_last_fold{}.json'.format(fold)), attributes, names)
            #
            # res = pd.DataFrame({'total_video_names': total_video_names, 'total_outs_best': total_outs_best,
            #                     'total_outs_last': total_outs_last, 'total_gts': total_gts,
            #                     'total_states': total_states})
            # with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
            #     pickle.dump(res, file)
            #
            # end_time = datetime.datetime.now()
            #
            # duration = end_time - start_time
            # print(f"Fold {fold} run time:", duration)
            #


    else:
        print(f"No files found in {conf_path}")
import copy
import os
import sys
import argparse
import datetime
from abc import ABC
from collections import defaultdict, Counter
import pandas as pd

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
from data.dataloaders import  PreserveKeysTransform, assert_backbone_is_supported

from const import path
from learning.utils import compute_class_weights, AverageMeter
from utility.utils import set_random_seed
from test import update_params_with_best, setup_datasets
import pkg_resources
from torchvision import transforms
import torch
from model.motionagformer.MotionAGFormer import MotionAGFormer
from  model.CausalModeling_counterfactual import *

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_TOTAL_SCORES = 3
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          #目前看来只有encoder-decoder中用到了它
#                1,   2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21
_GCN_JOINTS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_ROOT = 0
_MIN_STD = 1e-4

METADATA_MAP = {'gender': 0, 'age': 1, 'height': 2, 'weight': 3, 'bmi': 4}


class IdentityClassifier(nn.Module):
    """专为MotionAGFormer优化的身份分类头 (21类输出)"""

    def __init__(self, dim_rep=512, hidden_dim=1024, num_classes=21, dropout=0.1):
        super().__init__()
        # 特征聚合层 (适配MotionAGFormer的输出形状[B, T, J, C])
        self.pool = nn.Sequential(
            nn.LayerNorm(dim_rep),
            nn.AdaptiveAvgPool2d((1, 1))  # 输出 [B, 1, 1, C]
        )

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(dim_rep, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )


        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, J, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, T, J]
        x = x.flatten(2)  # [B, C, T*J]
        # x: [B, C, T*J] = [32, 512, 459]
        x = x.permute(0, 2, 1)  # [B, T*J, C] = [32, 459, 512]
        x = nn.LayerNorm(x.size(-1)).to(x.device)(x)  # 对 C 做 LN

        x = x.mean(1)  # 平均池化所有 T*J
        return self.head(x)

class MotionEncoder(nn.Module):
    """适配MotionAGFormer的身份编码器"""

    def __init__(self, backbone, num_classes=21, freeze_backbone=True, params=None, num_joints=None, train_mode=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = IdentityClassifier(
            num_classes=num_classes
        )

        if freeze_backbone:
            self._freeze_backbone()

        # 存储额外参数（如果需要）
        self.params = params
        self.num_joints = num_joints
        self.train_mode = train_mode

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Backbone weights frozen")

    def forward(self, x):
        features = self.backbone(x)  # [B, T, J, C]
        return self.classifier(features)


class PDReader():
    """
    Reads the data from the Parkinson's Disease dataset
    """

    ON_LABEL_COLUMN = 'ON - UPDRS-III - walking'
    OFF_LABEL_COLUMN = 'OFF - UPDRS-III - walking'
    DELIMITER = ';'

    def __init__(self, joints_path, labels_path,
                 med_status=None):  # med_status决定是否要只使用某个状态下的数据样本来处理，如果打开，那么--medication参数不应该加入
        self.joints_path = joints_path
        self.labels_path = labels_path
        self.med_status = med_status
        self.pose_dict, self.labels_dict, self.video_names, self.participant_ID, self.metadata_dict, self.identity_labels_dict = self.read_keypoints_and_labels()

    def read_sequence(self, path_file):
        """
        Reads skeletons from npy files
        """
        if os.path.exists(path_file):
            body = np.load(path_file)
            body = body / 1000  # convert mm to m
        else:
            body = None
        return body

    def read_label(self, file_name):
        subject_id, on_or_off = file_name.split("_")[:2]
        # df = pd.read_excel(self.labels_path)
        df = pd.read_csv(self.labels_path)  # 这里设置的deliminater有问题，导致报错
        # print(df.columns)
        df = df[['ID', self.ON_LABEL_COLUMN, self.OFF_LABEL_COLUMN]]
        subject_rows = df[df['ID'] == subject_id]
        if on_or_off == "on":
            label = subject_rows[self.ON_LABEL_COLUMN].values[0]
        else:
            label = subject_rows[self.OFF_LABEL_COLUMN].values[0]
        return int(label)

    def read_metadata(self, file_name):
        # If you change this function make sure to adjust the METADATA_MAP in the dataloaders.py accordingly
        subject_id = file_name.split("_")[0]
        # df = pd.read_excel(self.labels_path)
        df = pd.read_csv(self.labels_path)  # 这一行也是一样
        df = df[['ID', 'Gender', 'Age', 'Height (cm)', 'Weight (kg)', 'BMI (kg/m2)']]
        # print(df)
        df.rename(columns={
            "Gender": "gender",
            "Age": "age",
            "Height (cm)": "height",
            "Weight (kg)": "weight",
            "BMI (kg/m2)": "bmi"}, inplace=True)
        df.loc[:, 'gender'] = df['gender'].map({'M': 0, 'F': 1})

        # Using Min-Max normalization
        df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        df['height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
        df['weight'] = (df['weight'] - df['weight'].min()) / (df['weight'].max() - df['weight'].min())
        df['bmi'] = (df['bmi'] - df['bmi'].min()) / (df['bmi'].max() - df['bmi'].min())

        subject_rows = df[df['ID'] == subject_id]
        return subject_rows.values[:, 1:]

    def read_keypoints_and_labels(self):
        """
        Read npy files in given directory into arrays of pose keypoints.
        :return: dictionary with <key=video name, value=keypoints>
        """
        pose_dict = {}
        labels_dict = {}
        metadata_dict = {}
        video_names_list = []
        participant_ID = []
        identity_labels_dict = {}

        print('[INFO - PublicPDReader] Reading body keypoints from npy')

        print(self.joints_path)

        for file_name in tqdm(os.listdir(self.joints_path)):

            if file_name.startswith("SUB01"):       #我们加载fold0的预训练数据进行身份分类，就要排除掉SUB01的数据
                continue  # 排除 SUB01 的所有数据

            # 如果有med_status过滤要求
            if self.med_status is not None:
                lower_name = file_name.lower()
                if self.med_status == "ON" and "off" in lower_name:
                    continue
                elif self.med_status == "OFF" and "on" in lower_name:
                    continue

            path_file = os.path.join(self.joints_path, file_name)  # 注意这里的.npy依然是[T,V,C]，GCN后面需要进行额外处理
            joints = self.read_sequence(path_file)
            label = self.read_label(file_name)
            metadata = self.read_metadata(file_name)
            if joints is None:
                print(f"[WARN - PublicPDReader] Numpy file {file_name} does not exist")
                continue
            file_name = file_name.split(".")[0]
            pose_dict[file_name] = joints
            labels_dict[file_name] = label
            metadata_dict[file_name] = metadata
            video_names_list.append(file_name)
            participant_ID.append(file_name.split("_")[0])

        participant_ID = self.select_unique_entries(participant_ID)

        id2label = {pid: idx for idx, pid in enumerate(participant_ID)}
        for file_name in tqdm(os.listdir(self.joints_path)):

            if file_name.startswith("SUB01"):       #我们加载fold0的预训练数据进行身份分类，就要排除掉SUB01的数据
                continue  # 排除 SUB01 的所有数据

            file_name = file_name.split(".")[0]
            # 如果有med_status过滤要求
            if self.med_status is not None:
                lower_name = file_name.lower()
                if self.med_status == "ON" and "off" in lower_name:
                    continue
                elif self.med_status == "OFF" and "on" in lower_name:
                    continue
            identity_labels_dict[file_name] = id2label[file_name.split("_")[0]]

        return pose_dict, labels_dict, video_names_list, participant_ID, metadata_dict,identity_labels_dict

    @staticmethod
    def select_unique_entries(a_list):
        return sorted(list(set(a_list)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get item for the training mode."""

        # Based on index, get the video name
        video_name = self.video_names[idx]

        x = self.poses[video_name]
        label = self.labels[video_name]

        x = np.array(x, dtype=np.float32)

        sample = {
            'encoder_inputs': x,
            'label': label,

        }
        # if self.transform:
        #    sample = self.transform(sample)

        return sample


class DataPreprocessor(ABC):
    def __init__(self, raw_data, params=None):
        self.pose_dict = raw_data.pose_dict
        self.labels_dict = raw_data.labels_dict
        self.metadata_dict = raw_data.metadata_dict
        self.video_names = raw_data.video_names
        self.participant_ID = raw_data.participant_ID
        self.params = params

        self.data_dir = self.params['data_path']

    def __len__(self):
        return len(self.labels_dict)

    def center_poses(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            self.pose_dict[key] = joints3d - joints3d[:, _ROOT:_ROOT + 1, :]

    def normalize_poses(self):
        if self.params['data_norm'] == 'minmax':
            """
            Normalize each pose along each axis by video. Divide by the largest value in each direction
            and center around the origin.
            :param pose_dict: dictionary of poses
            :return: dictionary of normalized poses
            """
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(np.min(poses, axis=0), axis=0)
                maxes = np.max(np.max(poses, axis=0), axis=0)

                poses = (poses - mins) / (maxes - mins)

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'rescaling':
            normalized_pose_dict = {}
            for video_name in self.pose_dict:
                poses = self.pose_dict[video_name].copy()

                mins = np.min(poses, axis=(0, 1))
                maxes = np.max(poses, axis=(0, 1))

                poses = (2 * (poses - mins) / (maxes - mins)) - 1

                normalized_pose_dict[video_name] = poses
            self.pose_dict = normalized_pose_dict

        elif self.params['data_norm'] == 'zscore':
            norm_stats = self.compute_norm_stats()
            pose_dict_norm = self.pose_dict.copy()
            for k in self.pose_dict.keys():
                tmp_data = self.pose_dict[k].copy()
                tmp_data = tmp_data - norm_stats['mean']
                tmp_data = np.divide(tmp_data, norm_stats['std'])
                pose_dict_norm[k] = tmp_data
            self.pose_dict = pose_dict_norm

    def compute_norm_stats(self):
        all_data = []
        for k in self.pose_dict.keys():
            all_data.append(self.pose_dict[k])
        all_data = np.vstack(all_data)
        print('[INFO] ({}) Computing normalization stats!')
        norm_stats = {}
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        std[np.where(std < _MIN_STD)] = 1

        norm_stats['mean'] = mean  # .ravel()
        norm_stats['std'] = std  # .ravel()
        return norm_stats

    def generate_leave_one_out_folds(self, clip_dict, save_dir, labels_dict):
        """
        Generate folds for leave-one-out CV.
        :param clip_dict: dictionary of clips for each video
        :param save_dir: save directory for folds
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        video_names_list = list(clip_dict.keys())
        fold = 0

        dataset_name = self.params['dataset']
        val_folds_name = 'val_PD_SUBs_folds.pkl' if dataset_name == 'PD' else 'val_AMBIDs_folds.pkl'
        val_folds_path = os.path.join(save_dir, '..', val_folds_name)

        val_folds_exists = os.path.exists(val_folds_path)

        if not val_folds_exists:
            val_subs_folds = []
            print(f'[INFO] Previous selected {dataset_name} validation set does not exist.')
        else:
            val_subs_folds = pickle.load(open(val_folds_path, "rb"))

        for j in range(len(self.participant_ID)):
            train_list, val_list, test_list = [], [], []

            participant_ID_cloned = copy.deepcopy(self.participant_ID)
            subject_id = participant_ID_cloned.pop(j)

            class_participants = {}
            for participant in participant_ID_cloned:
                participant_labels = [labels_dict[key] for key in labels_dict if
                                      key.startswith(participant + "_on") or key.startswith(participant + "_off")]
                if participant_labels:
                    class_participants[participant] = participant_labels[0]  # Use the first label as the class

            if not val_folds_exists:
                val_subs = []
                for class_id in range(3):  # Assuming classes are 0, 1, 2
                    class_participants_for_class = [participant for participant, class_label in
                                                    class_participants.items() if class_label == class_id]
                    for _ in range(2):  # Select 2 participants from each class
                        val_idx = random.randint(0, len(class_participants_for_class) - 1)
                        val_subs.append(class_participants_for_class.pop(val_idx))
                val_subs_folds.append(val_subs)
            else:
                val_subs = val_subs_folds[j]

            for k in range(len(video_names_list)):
                video_name = video_names_list[k]
                # augmented = any(augmentation in video_name for augmentation in self.params['augmentation'])
                if dataset_name == 'PD':
                    if subject_id == video_name.split("_")[0]:
                        # if not augmented:
                        test_list.append(video_name)
                    elif video_name.split("_")[0] in val_subs:
                        val_list.append(video_name)
                    else:
                        train_list.append(video_name)
            print("Fold: ", fold)
            fold += 1
            train, validation, test = self.generate_pose_label_videoname(clip_dict, train_list, val_list, test_list)
            pickle.dump(train_list, open(os.path.join(save_dir, f"{dataset_name}_train_list_{fold}.pkl"), "wb"))
            pickle.dump(test_list, open(os.path.join(save_dir, f"{dataset_name}_test_list_{fold}.pkl"), "wb"))
            pickle.dump(val_list, open(os.path.join(save_dir, f"{dataset_name}_validation_list_{fold}.pkl"), "wb"))
            pickle.dump(train, open(os.path.join(save_dir, f"{dataset_name}_train_{fold}.pkl"), "wb"))
            pickle.dump(test, open(os.path.join(save_dir, f"{dataset_name}_test_{fold}.pkl"), "wb"))
            pickle.dump(validation, open(os.path.join(save_dir, f"{dataset_name}_validation_{fold}.pkl"), "wb"))
        pickle.dump(self.labels_dict, open(os.path.join(save_dir, f"{dataset_name}_labels.pkl"), "wb"))

        if not val_folds_exists:
            pickle.dump(val_subs_folds, open(val_folds_path, "wb"))




    def get_data_split(self, split_list, clip_dict):
        split = {'pose': [], 'label': [], 'video_name': [], 'metadata': []}
        for video_name in split_list:
            clips = clip_dict[video_name]
            for clip in clips:
                split['label'].append(self.labels_dict[video_name])
                split['pose'].append(clip)
                split['video_name'].append(video_name)
                split['metadata'].append(self.metadata_dict[video_name])
        return split

    def generate_pose_label_videoname(self, clip_dict, train_list, val_list, test_list):
        train = self.get_data_split(train_list, clip_dict)
        val = self.get_data_split(val_list, clip_dict)
        test = self.get_data_split(test_list, clip_dict)

        # print how many samples are in each split
        print(f"Train Length: {len(train['video_name'])}")
        print(f"Validation Length: {len(val['video_name'])}")
        print(f"Test Length: {len(test['video_name'])}")
        return train, val, test

    @staticmethod
    def resample(original_length, target_length):
        """
        Adapted from https://github.com/Walter0807/MotionBERT/blob/main/lib/utils/utils_data.py#L68

        Returns an array that has indices of frames. elements of array are in range (0, original_length -1) and
        we have target_len numbers (So it interpolates the frames)
        """
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        result = np.floor(even)
        result = np.clip(result, a_min=0, a_max=original_length - 1).astype(np.uint32)
        return result



class MotionAGFormerPreprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()
        else:
            self.place_depth_of_first_frame_to_zero()

        self.identity_labels_dict=raw_data.identity_labels_dict

        participant_number=params['participant_number']

        self.clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])        #某个视频名称如SUB01_walk_1_0，对应被切成的一个或多个片段

        video_names = list(self.identity_labels_dict.keys())

        self.splits = self.split_by_original_video(video_names, self.identity_labels_dict,
                                         train_ratio=0.7, val_ratio=0.15, seed=123)


    def split_by_original_video(self,video_names, identity_labels_dict,
                                train_ratio, val_ratio=0.15, seed=42):
        """
        根据原始视频划分 train/val/test，保证同一原始视频的 _0/_1/_2 不被分到不同集合
        :param video_names: list[str] 降采样后的 video_name 列表，例如 ["SUB01_on_walk_1_0", "SUB01_on_walk_1_1", ...]
        :param identity_labels_dict: dict {video_name: identity_label}
        :param train_ratio: float, 训练集比例
        :param val_ratio: float, 验证集比例
        :param seed: int, 随机种子
        :return: dict, 包含 train/val/test 三个集合的 video_name 列表和 label 字典
        """
        random.seed(seed)

        # 1. 映射回原始视频名
        video_groups = defaultdict(list)
        for vname in video_names:           #vname带有.npy
            base_name = "_".join(vname.split("_")[:-1])  # 去掉最后的 _0/_1/_2
            video_groups[base_name].append(vname)

        # 2. 打乱原始视频
        all_videos = list(video_groups.keys())
        random.shuffle(all_videos)

        # 3. 按比例切分 train/val/test
        n_total = len(all_videos)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_videos = all_videos[:n_train]
        val_videos = all_videos[n_train:n_train + n_val]
        test_videos = all_videos[n_train + n_val:]

        # 4. 展开到 video_name 级别
        def expand(videos):
            return [clip for v in videos for clip in video_groups[v]]       #video_groups:以原视频名（未经过降采样）为键，三个降采样的视频名称作为值（但是包含了.npy)

        train_names = expand(train_videos)
        val_names = expand(val_videos)
        test_names = expand(test_videos)

        # 5. 生成对应的 labels
        train_labels = {c: identity_labels_dict[c] for c in train_names}
        val_labels = {c: identity_labels_dict[c] for c in val_names}
        test_labels = {c: identity_labels_dict[c] for c in test_names}

        return {
            "train_videos": train_names,
            "val_videos": val_names,
            "test_videos": test_names,
            "train_labels": train_labels,
            "val_labels": val_labels,
            "test_labels": test_labels
        }


    def place_depth_of_first_frame_to_zero(self):
        for key in self.pose_dict.keys():
            joints3d = self.pose_dict[key]  # (n_frames, n_joints, 3)
            joints3d[..., 2] = joints3d[..., 2] - joints3d[0:1, _ROOT:_ROOT + 1, 2]

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():        #pose_dict：以文件名（1/3切分）为键，总长度的视频骨架为值，长度经过1/3切分，但是长度依然不一定
            clips = self.get_clips(self.pose_dict[video_name], clip_length)     #以给定长度进行切
            clip_dict[video_name] = clips
        return clip_dict                                #依然是视频名字为键，但是多了很多个小片段

    def get_clips(self, video_sequence, clip_length, data_stride=15):
        data_stride = clip_length
        clips = []
        video_length = video_sequence.shape[0]
        if video_length < clip_length:
            pass
            #new_indices = self.resample(video_length, clip_length)
            #clips.append(video_sequence[new_indices])
        else:
            if self.params['select_middle']:
                middle_frame = (video_length) // 2
                start_frame = middle_frame - (clip_length // 2)
                clip = video_sequence[start_frame: start_frame + clip_length]
                clips.append(clip)
            else:
                start_frame = 0
                while (video_length - start_frame) >= clip_length:
                    clips.append(video_sequence[start_frame:start_frame + clip_length])
                    start_frame += data_stride
                #new_indices = self.resample(video_length - start_frame, clip_length) + start_frame
                #clips.append(video_sequence[new_indices])
        return clips


class ProcessedDataset(data.Dataset):
    def __init__(self, poses_dict, splits, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.poses_dict = poses_dict  # 保存原始poses_dict引用

        # 根据 mode 选择 subset
        if mode == 'train':
            self.video_names = splits['train_videos']
            self.labels = [splits['train_labels'][vn] for vn in self.video_names]
        elif mode == 'val':
            self.video_names = splits['val_videos']
            self.labels = [splits['val_labels'][vn] for vn in self.video_names]
        elif mode == 'test':
            self.video_names = splits['test_videos']
            self.labels = [splits['test_labels'][vn] for vn in self.video_names]
        else:
            raise ValueError(f"Unknown mode {mode}")

        # 不再在这里预加载所有pose，改为在__getitem__中按需获取
        self._pose_dim = 3 * len(_GCN_JOINTS)
        self._NMAJOR_JOINTS = len(_MAJOR_JOINTS)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base_video_name = self.video_names[idx]

        # 找出所有以这个base_video_name开头的clip
        clip_keys = [k for k in self.poses_dict.keys() if k.startswith(base_video_name)]

        # 随机选择一个clip（训练时可以增加随机性，测试时可以固定）
        if len(clip_keys) == 0:
            raise ValueError(f"No clips found for video {base_video_name}")

        selected_clip_key = clip_keys[0]  # 默认取第一个，可以改为随机选择
        if self.mode == 'train' and len(clip_keys) > 1:
            selected_clip_key = random.choice(clip_keys)

        x = self.poses_dict[selected_clip_key]
        label = self.labels[idx]
        video_name = base_video_name  # 或者用 selected_clip_key 如果你想保留clip信息

        sample = {
            'encoder_inputs': x,        #给定一个idx，它会随机在所有降采样之后的视频里面选择一个，然后在它切割完之后的clip里面随机选择一个来作为输入
            'label': label,
            'video_name': video_name
        }
        return sample


def collate_fn(batch):
    """Collate function for identity classification task."""
    # 确保所有输入是float32类型
    encoder_inputs = torch.from_numpy(np.stack(
        [e['encoder_inputs'].astype(np.float32) for e in batch]
    ))

    # 确保标签是long类型(用于分类任务)
    labels = torch.from_numpy(np.stack(
        [e['label'] for e in batch]
    )).long()

    # 保留视频名称信息
    video_names = [e['video_name'] for e in batch]

    return {
        'encoder_inputs': encoder_inputs,
        'labels': labels,
        'video_names': video_names  # 用于调试或分析
    }


def compute_class_weights(data_loader):
    """Compute balanced class weights for identity classification."""
    class_counts = Counter()

    # 统计所有batch中的类别分布
    for batch in data_loader:
        labels = batch['labels'].numpy()
        class_counts.update(labels)

    if not class_counts:
        raise ValueError("No class counts collected - check your data loader")

    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    # 计算逆频率权重
    class_weights = {
        class_id: total_samples / (num_classes * count)
        for class_id, count in class_counts.items()
    }

    # 转换为按类别顺序排列的tensor
    weights_tensor = torch.tensor(
        [class_weights[i] for i in sorted(class_counts.keys())],
        dtype=torch.float32
    )

    # 可选: 归一化使权重总和等于类别数
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes

    return weights_tensor


def dataset_factory(params, backbone):

    root_dir = f'{path.IDCLS_PREPROCESSED_DATA_ROOT_PATH}/{backbone}_processing'       #backbone应为motionagformer

    backbone_data_location_mapper = {  # backbone特定的数据处理目录
        'motionagformer': os.path.join(root_dir, params['experiment_name'],
                                       f"{params['dataset']}_center_{params['data_centered']}/")
    }

    backbone_preprocessor_mapper = {
        'motionagformer': MotionAGFormerPreprocessor
    }

    assert_backbone_is_supported(backbone_data_location_mapper, backbone)

    data_dir = backbone_data_location_mapper[backbone]

    med_status = params['med_status']  # None或者ON/OFF

    if not os.path.exists(
            data_dir):
        if params['dataset'] == 'PD':
            raw_data = PDReader(params['data_path'], params['labels_path'],med_status)
            params['participant_number'] = len(raw_data.participant_ID)

            # Preprocessor = backbone_preprocessor_mapper[backbone]
            # Preprocessor(data_dir, raw_data, params)
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

    Preprocessor = backbone_preprocessor_mapper[backbone]
    preprocessor = Preprocessor(data_dir, raw_data, params)
    poses_dict = {}
    for vname, clips in preprocessor.clip_dict.items():
        for idx, clip in enumerate(clips):
            clip_name = f"{vname}_clip{idx}"
            poses_dict[clip_name] = clip


    #现在得到的poses_dict是：以降采样之后的多个clip作为拆分
    train_dataset = ProcessedDataset(poses_dict, preprocessor.splits, mode='train', transform=train_transform)
    eval_dataset = ProcessedDataset(poses_dict, preprocessor.splits, mode='val')
    test_dataset = ProcessedDataset(poses_dict, preprocessor.splits, mode='test')


    # 随机选择一个索引
    index_to_check = 42

    # 获取样本并打印其内容
    sample = train_dataset[index_to_check]

    # 打印 'label' 键对应的值，即 y
    print(f"Sample at index {index_to_check}:")
    print(f"Shape of poses (x): {sample['encoder_inputs'].shape}")
    print(f"Label (y): {sample['label']}")

    # 你还可以进一步打印更多信息
    # print(sample['labels_str'])
    # eval_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='val') if use_validation else None
    # test_dataset = ProcessedDataset(data_dir, fold=fold, params=params, mode='test')
    #
    #
    #
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
    #
    class_weights = compute_class_weights(train_dataset_fn)
    #
    return train_dataset_fn, test_dataset_fn, eval_dataset_fn, class_weights  # ,train_dataset #用于可视化


def count_parameters(model):
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    return model_params

def load_pretrained_weights(model, checkpoint, strict=False, skip_layers=None):
    """
    Load pretrained weights to model, skipping specified layers (e.g., classification head).
    Args:
        model (nn.Module): The model to load weights into.
        checkpoint (dict): Pretrained weights (can be state_dict or full checkpoint).
        strict (bool): If True, requires exact matching of all layer names and shapes.
        skip_layers (list[str]): List of layer prefixes to skip (e.g., ["head."]).
    """
    if skip_layers is None:
        skip_layers = ["head."]  # 默认跳过分类头

    # 1. 提取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 2. 去除 DataParallel 前缀（如果存在）
    if not any(k.startswith('module.') for k in model.state_dict().keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 3. 过滤掉需要跳过的层（如分类头）
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(skip) for skip in skip_layers)
    }

    # 4. 加载匹配的权重
    model_dict = model.state_dict()
    matched_keys = []
    for k, v in filtered_state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
            matched_keys.append(k)

    # 5. 严格模式检查（可选）
    if strict and len(matched_keys) != len(state_dict):
        missing = set(state_dict.keys()) - set(matched_keys)
        raise ValueError(f"Strict模式加载失败，缺失层: {missing}")

    # 6. 加载权重
    model.load_state_dict(model_dict, strict=False)

    # 7. 打印日志
    print(f"[INFO] 成功加载 {len(matched_keys)}/{len(state_dict)} 层权重")
    print(f"[INFO] 跳过的层: {[k for k in state_dict if any(k.startswith(skip) for skip in skip_layers)]}")

    if len(matched_keys) == 0:
        raise RuntimeError("未加载任何权重，请检查模型结构匹配性")


def load_pretrained_backbone(params, backbone_name):            #TODO:思考GCN应该怎么接入预训练模型

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


def initialize_wandb(params):
    wandb.init(name=params['wandb_name'], project='ID_CLS', settings=wandb.Settings(start_method='fork'))
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    wandb.config.update(params)
    wandb.config.update({'installed_packages': installed_packages})


def setup_experiment_path(params):
    exp_path = path.IDCLS_OUT_PATH + os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    params['model_prefix'] = os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    rep_out = path.IDCLS_OUT_PATH + os.path.join(params['model_prefix'])
    return params, rep_out


def choose_optimizer(model, params):
    optimizer_name = params['optimizer']
    try:
        backbone_params = set(model.module.backbone.parameters())
        head_params = set(model.module.classifier.parameters())  # ✅ 改这里
    except AttributeError:
        backbone_params = set(model.backbone.parameters())
        head_params = set(model.classifier.parameters())  # ✅ 改这里

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


def choose_criterion(key, params, class_weights):
    if key == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ModuleNotFoundError("Criterion does not exist")


def choose_scheduler(optimizer, params):
    scheduler_name = params.get('scheduler')
    if scheduler_name is None:
        print("[WARN] LR Scheduler is not used")
        return None

    if scheduler_name == "StepLR":
        scheduler = StepLR(optimizer, step_size=params['lr_step_size'], gamma=params['lr_decay'])
    else:
        raise ModuleNotFoundError("Scheduler is not defined")

    return scheduler


def train_model(params, class_weights, train_loader, val_loader, model, fold, backbone_name, mode="RUN"):

    params['criterion']='CrossEntropyLoss'
    criterion = choose_criterion(params['criterion'], params, class_weights)

    if torch.cuda.is_available():
        model = model.to(_DEVICE)
        criterion = criterion.to(_DEVICE)
    else:
        raise Exception("Cuda is not enabled")

    #
    optimizer = choose_optimizer(model, params)
    scheduler = choose_scheduler(optimizer, params)
    #
    checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'], 'models')
    if not os.path.exists(checkpoint_root_path): os.makedirs(checkpoint_root_path)  # 原本的mkdir只能创建单级目录
    #
    loop = tqdm(range(params['epochs']), desc=f'Training (fold{fold})', unit="epoch")
    #
    best_val_f1 = 0.0  # 最佳的f1分数
    #
    for epoch in loop:
        print(f"[INFO] epoch {epoch}")
        train_acc = AverageMeter()
        train_loss = AverageMeter()

    #
        model.train()

        all_preds = []
        all_labels = []

        epoch_start_time = datetime.time()

        for batch in train_loader:
            x = batch['encoder_inputs'].to(_DEVICE).float()  # 确保float
            y = batch['labels'].to(_DEVICE)


            optimizer.zero_grad()
            batch_size=x.shape[0]


            out = model(x)

            loss = criterion(out, y)
            train_loss.update(loss.item(), batch_size)


            # L1正则（可选）
            if params.get('lambda_l1', 0) > 0:
                learnable_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
                l1_reg = torch.norm(learnable_params, p=1)
                loss += params['lambda_l1'] * l1_reg

            loss.backward()
            optimizer.step()

            # 保存预测和标签，用于统计训练精度/指标
            all_preds.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        _, predicted = all_preds.max(1)
        train_acc = (predicted == all_labels).float().mean().item()
        print(f"Train accuracy: {train_acc:.4f}")
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

    # if mode == "RUN":
    #     save_checkpoint(checkpoint_root_path, epoch, lr_backbone, optimizer, model, None, fold, latest=True)
    #     print(f'[INFO] Latest checkpoint saved at: {checkpoint_root_path}')


def process_fold(fold, params, backbone_name, train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights, device,
                 total_outs_best, total_gts, total_logits, total_states, total_video_names, total_outs_last, rep_out):
    start_time = datetime.datetime.now()
    params['input_dim'] = train_dataset_fn.dataset._pose_dim  # 这个参数对于CTR-GCN来说无用，因为它不展平向量
    params['pose_dim'] = train_dataset_fn.dataset._pose_dim  # 同上，无用
    params['num_joints'] = train_dataset_fn.dataset._NMAJOR_JOINTS


    # params['model_checkpoint_path']='/czl_ssd/backup/motionagformer_fold0_without_SUB01_best_ckpt.pth.tr'

    model_backbone = load_pretrained_backbone(params, backbone_name)
    model = MotionEncoder(backbone=model_backbone,
                          params=params,
                          num_classes=params['num_classes'],
                          num_joints=params['num_joints'],
                          train_mode=params['train_mode'])
    best_ckpt_path = os.path.join('/czl_ssd/backup/motionagformer_fold0_without_SUB01_best_ckpt.pth.tr')
    load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)['model'])

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if fold == 1:
        model_params = count_parameters(model)
        print(f"[INFO] Model has {model_params} parameters.")
    #
    train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name)  # Layer2的入口
    #
    checkpoint_root_path = os.path.join(path.IDCLS_OUT_PATH, params['model_prefix'], 'models', f"fold{fold}")
    best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
    # load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)[
    #     'model'])  # 这里报错了，只在第一个fold上跑完了20个epoch，只保存了latest_epoch.pth.tr，所以加载不出来
    model.cuda()
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
    # res = pd.DataFrame(
    #     {'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last,
    #      'total_gts': total_gts, 'total_states': total_states})
    # with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
    #     pickle.dump(res, file)
    #
    # end_time = datetime.datetime.now()
    #
    # duration = end_time - start_time
    # print(f"Fold {fold} run time:", duration)

def run_tests_for_each_fold(params, splits, backbone_name, device, rep_out):
    total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits):
        process_fold(fold, params, backbone_name, train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights, device, total_outs_best, total_gts, total_logits, total_states, total_video_names, total_outs_last, rep_out)
    return total_outs_best, total_gts, total_states, total_video_names, total_outs_last


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='motionbert', help='model name ( poseformer, ''motionbert )')
    parser.add_argument('--train_mode', type=str, default='classifier_only',
                        help='train mode( end2end, classifier_only )')
    parser.add_argument('--dataset', type=str, default='PD', help='**currently code only works for PD')
    parser.add_argument('--data_path', type=str, default=path.PD_PATH_POSES)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--tune_fresh', default=1, type=int,
                        help='start a new tuning process or cont. on a previous study')
    parser.add_argument('--last_run_foldnum', default='7', type=str)
    parser.add_argument('--readstudyfrom', default=1, type=int)

    parser.add_argument('--medication', default=0, type=int, help='add medication prob to the training [0 or 1]')
    parser.add_argument('--metadata', default='', type=str,
                        help="add metadata prob to the training 'gender,age,bmi,height,weight'")
    parser.add_argument(
        '--med_status',         #最好打开这个，因为需要控制变量，只看某个情况下的患者特征，排除用药状态干扰
        choices=['ON', 'OFF'],
        default=None,
        help="only ON/OFF samples are used(including training, validating, and testing"
    )
    parser.add_argument('--participant_number', default=23, type=int,
                        help="all participants number")
    parser.add_argument('--eval_ID', default=0, type=int,
                        help="eval ID classifier of backbone feature")

    args = parser.parse_args()

    param = vars(args)
    param['metadata'] = param['metadata'].split(',') if param['metadata'] else []

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
            params['num_classes'] = 21      #比正常训练的22个人要少一个，并且我们需要弄清楚到底预训练的时候缺少了谁;如果你用的是fold0，那么就说明此时SUB01是没有被模型见过的，我们需要把它排除在外
        else:
            raise NotImplementedError(f"dataset '{param['dataset']}' is not supported.")

        set_random_seed(param['seed'])

        # test_and_report(params, new_params, all_folds, backbone_name, _DEVICE)

        params, rep_out = setup_experiment_path(params)
        # configure_params_for_best_model(params, backbone_name)        #不需要
        #initialize_wandb(params)

        splits = []
        train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name)
        splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))

        run_tests_for_each_fold(params, splits, backbone_name, _DEVICE, rep_out)






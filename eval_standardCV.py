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

from utility import utils
from train import train_model, final_test

import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from configs import generate_config_motionagformer,generate_config_poseformerv2
from data.augmentations import RandomNoise, RandomRotation, MirrorReflection, axis_mask
from data.dataloaders import  PreserveKeysTransform, assert_backbone_is_supported

from const import path
from learning.utils import compute_class_weights, AverageMeter
from model.utils import *
from utility.utils import set_random_seed
from test import update_params_with_best, setup_datasets,configure_params_for_best_model,initialize_wandb,process_reports,save_and_load_results
import pkg_resources
from torchvision import transforms
import torch
from model.motionagformer.MotionAGFormer import MotionAGFormer
from model.poseformerv2.model_poseformer import PoseTransformerV2
from  model.CausalModeling_counterfactual import *
from model.motion_encoder import MotionEncoder




_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_TOTAL_SCORES = 3
_MAJOR_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          #目前看来只有encoder-decoder中用到了它
#                1,   2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21
_GCN_JOINTS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_ROOT = 0
_MIN_STD = 1e-4

#
# def final_test(model, test_loader, params):
#     model.eval()
#     video_results = defaultdict(lambda: {'logits': [], 'preds': [], 'labels': None})
#
#     with torch.no_grad():
#         for x, y, video_idx, _ in test_loader:
#             x, y = x.to(_DEVICE), y.to(_DEVICE)
#             out = model(x)  # 简化示例，根据实际模型调整
#
#             # 对每个视频收集所有clip的结果
#             for i, idx in enumerate(video_idx):
#                 idx = idx.item()
#                 video_results[idx]['logits'].append(out[i].cpu())
#                 video_results[idx]['preds'].append(out[i].argmax().item())
#                 video_results[idx]['labels'] = y[i].item()
#
#     # 计算最终视频级预测
#     all_preds, all_labels = [], []
#     for idx, data in video_results.items():
#         # 方法1: 平均logits后取argmax
#         avg_logits = torch.stack(data['logits']).mean(0)
#         final_pred = avg_logits.argmax().item()
#
#         # 方法2: 多数投票
#         # final_pred = Counter(data['preds']).most_common(1)[0][0]
#
#         all_preds.append(final_pred)
#         all_labels.append(data['labels'])
#
#     return all_preds, all_labels

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

        # 在初始化时做一次划分
        self.train_list, self.val_list, self.test_list = self.split_train_val_test(
            self.video_names, self.labels_dict, train_ratio=0.75, val_ratio=0.15, test_ratio=0.15
        )
    #
    # def split_train_val_test(self, video_names, labels_dict, train_ratio=0.75, val_ratio=0.15, test_ratio=0.15):
    #     """
    #     按照原视频 (subXX_on/off_walk_Y) 分组，保证同组的 _0,_1,_2 不会跨集合。
    #     同时保证 train/val/test 内标签比例尽量接近。
    #     """
    #
    #     # step1: 按 "原视频" 分组
    #     grouped = defaultdict(list)  # key=原视频名, value=三个子片段
    #     for v in video_names:
    #         base = "_".join(v.split("_")[:-1])  # 去掉最后一个 0/1/2
    #         grouped[base].append(v)
    #
    #     groups = list(grouped.keys())
    #
    #     # step2: 给每个组打上标签（随便取一个子片段的label即可，三个片段一样）
    #     group_labels = {g: labels_dict[grouped[g][0]] for g in groups}
    #
    #     # step3: 按标签分层
    #     groups_by_label = defaultdict(list)
    #     for g, lab in group_labels.items():
    #         groups_by_label[lab].append(g)
    #
    #     train, val, test = [], [], []
    #
    #     # step4: 分层抽样
    #     for lab, g_list in groups_by_label.items():
    #         random.shuffle(g_list)
    #         n = len(g_list)
    #         n_train = int(n * train_ratio)
    #         n_val = int(n * val_ratio)
    #         n_test = n - n_train - n_val
    #
    #         train.extend(g_list[:n_train])
    #         val.extend(g_list[n_train:n_train + n_val])
    #         test.extend(g_list[n_train + n_val:])
    #
    #     # step5: 还原成 video_name 列表
    #     train_list = [v for g in train for v in grouped[g]]
    #     val_list = [v for g in val for v in grouped[g]]
    #     test_list = [v for g in test for v in grouped[g]]
    #
    #     # 打印一下比例
    #     print(f"[SPLIT] Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
    #
    #     def get_distribution(split):
    #         labs = [labels_dict[v] for v in split]
    #         return {c: labs.count(c) for c in sorted(set(labs))}
    #
    #     print("[SPLIT] Label distribution:")
    #     print("  Train:", get_distribution(train_list))
    #     print("  Val:  ", get_distribution(val_list))
    #     print("  Test: ", get_distribution(test_list))
    #
    #     return train_list, val_list, test_list

    def split_train_val_test(self, video_names, labels_dict, train_ratio=0.75, val_ratio=0.15, test_ratio=0.15):
        """
        按照原视频 (subXX_on/off_walk_Y) 分组，保证同组的 _0,_1,_2 不会跨集合。
        同时保证 train/val/test 内标签比例尽量接近。
        """
        import random
        from collections import defaultdict

        # step1: 按 "原视频" 分组 - 更鲁棒的分组方式
        grouped = defaultdict(list)

        for v in video_names:
            # 打印视频名称以便调试
            print(f"Processing video: {v}")

            # 假设格式是 SUB01_walk_1_0, SUB01_walk_1_1, SUB01_walk_1_2
            # 去掉最后的 _数字 来获取基础视频名
            parts = v.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                base = "_".join(parts[:-1])
            else:
                # 如果格式不符合预期，使用整个名称作为base
                base = v
                print(f"Warning: Unexpected video name format: {v}")

            grouped[base].append(v)

        # 打印分组结果以便调试
        print("\n=== 视频分组结果 ===")
        for base, videos in grouped.items():
            print(f"{base}: {videos}")

        groups = list(grouped.keys())

        # step2: 给每个组打上标签
        group_labels = {}
        for g in groups:
            # 确保标签存在
            first_video = grouped[g][0]
            if first_video in labels_dict:
                group_labels[g] = labels_dict[first_video]
            else:
                print(f"Warning: Label not found for video {first_video}")
                continue

        # 验证同一组内的标签是否一致
        for g, videos in grouped.items():
            if g not in group_labels:
                continue
            expected_label = group_labels[g]
            for v in videos:
                if v in labels_dict and labels_dict[v] != expected_label:
                    print(
                        f"Warning: Inconsistent labels in group {g}: {v} has label {labels_dict[v]}, expected {expected_label}")

        # step3: 按标签分层
        groups_by_label = defaultdict(list)
        for g, lab in group_labels.items():
            groups_by_label[lab].append(g)

        print(f"\n=== 按标签分组 ===")
        for lab, g_list in groups_by_label.items():
            print(f"Label {lab}: {len(g_list)} groups - {g_list}")

        train, val, test = [], [], []

        # step4: 分层抽样
        random.seed(10)  # 设置随机种子确保可重现
        for lab, g_list in groups_by_label.items():
            g_list_copy = g_list.copy()
            random.shuffle(g_list_copy)
            n = len(g_list_copy)

            if n == 0:
                continue

            n_train = max(1, int(n * train_ratio))  # 至少分配1个到train
            n_val = max(0, int(n * val_ratio))
            n_test = max(0, n - n_train - n_val)

            # 确保总数正确
            if n_train + n_val + n_test != n:
                n_test = n - n_train - n_val

            train.extend(g_list_copy[:n_train])
            val.extend(g_list_copy[n_train:n_train + n_val])
            test.extend(g_list_copy[n_train + n_val:])

        # step5: 还原成 video_name 列表
        train_list = []
        val_list = []
        test_list = []

        for g in train:
            train_list.extend(grouped[g])
        for g in val:
            val_list.extend(grouped[g])
        for g in test:
            test_list.extend(grouped[g])

        # 验证没有重复和遗漏
        all_videos_split = set(train_list + val_list + test_list)
        original_videos = set(video_names)

        if all_videos_split != original_videos:
            missing = original_videos - all_videos_split
            extra = all_videos_split - original_videos
            print(f"Warning: Video mismatch!")
            print(f"Missing: {missing}")
            print(f"Extra: {extra}")

        # 验证没有跨集合的同组视频
        print(f"\n=== 验证数据泄露 ===")
        for base, videos in grouped.items():
            in_train = sum(1 for v in videos if v in train_list)
            in_val = sum(1 for v in videos if v in val_list)
            in_test = sum(1 for v in videos if v in test_list)

            if (in_train > 0 and in_val > 0) or (in_train > 0 and in_test > 0) or (in_val > 0 and in_test > 0):
                print(f"DATA LEAKAGE DETECTED! Group {base} spans multiple splits:")
                print(f"  Train: {in_train}, Val: {in_val}, Test: {in_test}")
                print(f"  Videos: {videos}")

        # 打印分割结果
        print(f"\n[SPLIT] Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

        def get_distribution(split):
            labs = [labels_dict[v] for v in split if v in labels_dict]
            return {c: labs.count(c) for c in sorted(set(labs))}

        print("[SPLIT] Label distribution:")
        print("  Train:", get_distribution(train_list))
        print("  Val:  ", get_distribution(val_list))
        print("  Test: ", get_distribution(test_list))

        return train_list, val_list, test_list

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



class PoseformerV2Preprocessor(DataPreprocessor):
    def __init__(self, save_dir, raw_data, params):
        super().__init__(raw_data, params=params)

        if self.params['data_centered']:
            self.center_poses()

        self.remove_last_dim_of_pose()
        self.normalize_poses()
        self.clip_dict = self.partition_videos(clip_length=self.params['source_seq_len'])
        #self.generate_leave_one_out_folds(clip_dict, save_dir,raw_data.labels_dict)

        # 在初始化时做一次划分
        self.train_list, self.val_list, self.test_list = self.split_train_val_test(
            self.video_names, self.labels_dict, train_ratio=0.75, val_ratio=0.15, test_ratio=0.15
        )


    def split_train_val_test(self, video_names, labels_dict, train_ratio=0.75, val_ratio=0.15, test_ratio=0.15):
        """
        按照原视频 (subXX_on/off_walk_Y) 分组，保证同组的 _0,_1,_2 不会跨集合。
        同时保证 train/val/test 内标签比例尽量接近。
        """
        import random
        from collections import defaultdict

        # step1: 按 "原视频" 分组 - 更鲁棒的分组方式
        grouped = defaultdict(list)

        for v in video_names:
            # 打印视频名称以便调试
            print(f"Processing video: {v}")

            # 假设格式是 SUB01_walk_1_0, SUB01_walk_1_1, SUB01_walk_1_2
            # 去掉最后的 _数字 来获取基础视频名
            parts = v.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                base = "_".join(parts[:-1])
            else:
                # 如果格式不符合预期，使用整个名称作为base
                base = v
                print(f"Warning: Unexpected video name format: {v}")

            grouped[base].append(v)

        # 打印分组结果以便调试
        print("\n=== 视频分组结果 ===")
        for base, videos in grouped.items():
            print(f"{base}: {videos}")

        groups = list(grouped.keys())

        # step2: 给每个组打上标签
        group_labels = {}
        for g in groups:
            # 确保标签存在
            first_video = grouped[g][0]
            if first_video in labels_dict:
                group_labels[g] = labels_dict[first_video]
            else:
                print(f"Warning: Label not found for video {first_video}")
                continue

        # 验证同一组内的标签是否一致
        for g, videos in grouped.items():
            if g not in group_labels:
                continue
            expected_label = group_labels[g]
            for v in videos:
                if v in labels_dict and labels_dict[v] != expected_label:
                    print(
                        f"Warning: Inconsistent labels in group {g}: {v} has label {labels_dict[v]}, expected {expected_label}")

        # step3: 按标签分层
        groups_by_label = defaultdict(list)
        for g, lab in group_labels.items():
            groups_by_label[lab].append(g)

        print(f"\n=== 按标签分组 ===")
        for lab, g_list in groups_by_label.items():
            print(f"Label {lab}: {len(g_list)} groups - {g_list}")

        train, val, test = [], [], []

        # step4: 分层抽样
        random.seed(42)  # 设置随机种子确保可重现
        for lab, g_list in groups_by_label.items():
            g_list_copy = g_list.copy()
            random.shuffle(g_list_copy)
            n = len(g_list_copy)

            if n == 0:
                continue

            n_train = max(1, int(n * train_ratio))  # 至少分配1个到train
            n_val = max(0, int(n * val_ratio))
            n_test = max(0, n - n_train - n_val)

            # 确保总数正确
            if n_train + n_val + n_test != n:
                n_test = n - n_train - n_val

            train.extend(g_list_copy[:n_train])
            val.extend(g_list_copy[n_train:n_train + n_val])
            test.extend(g_list_copy[n_train + n_val:])

        # step5: 还原成 video_name 列表
        train_list = []
        val_list = []
        test_list = []

        for g in train:
            train_list.extend(grouped[g])
        for g in val:
            val_list.extend(grouped[g])
        for g in test:
            test_list.extend(grouped[g])

        # 验证没有重复和遗漏
        all_videos_split = set(train_list + val_list + test_list)
        original_videos = set(video_names)

        if all_videos_split != original_videos:
            missing = original_videos - all_videos_split
            extra = all_videos_split - original_videos
            print(f"Warning: Video mismatch!")
            print(f"Missing: {missing}")
            print(f"Extra: {extra}")

        # 验证没有跨集合的同组视频
        print(f"\n=== 验证数据泄露 ===")
        for base, videos in grouped.items():
            in_train = sum(1 for v in videos if v in train_list)
            in_val = sum(1 for v in videos if v in val_list)
            in_test = sum(1 for v in videos if v in test_list)

            if (in_train > 0 and in_val > 0) or (in_train > 0 and in_test > 0) or (in_val > 0 and in_test > 0):
                print(f"DATA LEAKAGE DETECTED! Group {base} spans multiple splits:")
                print(f"  Train: {in_train}, Val: {in_val}, Test: {in_test}")
                print(f"  Videos: {videos}")

        # 打印分割结果
        print(f"\n[SPLIT] Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

        def get_distribution(split):
            labs = [labels_dict[v] for v in split if v in labels_dict]
            return {c: labs.count(c) for c in sorted(set(labs))}

        print("[SPLIT] Label distribution:")
        print("  Train:", get_distribution(train_list))
        print("  Val:  ", get_distribution(val_list))
        print("  Test: ", get_distribution(test_list))

        return train_list, val_list, test_list


    def remove_last_dim_of_pose(self):
        for video_name in self.pose_dict:
            self.pose_dict[video_name] = self.pose_dict[video_name][..., :2]  # Ignoring confidence score

    def partition_videos(self, clip_length):
        """
        Partition poses from each video into clips.
        :return: dictionary of clips for each video
        """
        clip_dict = {}
        for video_name in self.pose_dict.keys():
            clips = self.get_clips(self.pose_dict[video_name], clip_length)
            clip_dict[video_name] = clips
        return clip_dict

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
    def __init__(self, preprocessor, params=None, mode='train', transform=None):
        super().__init__()
        self._params = params
        self._mode = mode
        self.transform = transform

        self.backbone = self._params['backbone']
        self.gcn_mode = (self.backbone == 'ctrgcn')
        self._NMAJOR_JOINTS = len(_GCN_JOINTS if self.gcn_mode else _MAJOR_JOINTS)

        # 定义 label 映射
        self._task = self._params.get("downstream", "pd")
        if self._task == 'pd':
            self._updrs_str = ['normal', 'slight', 'moderate']
            self._TOTAL_SCORES = _TOTAL_SCORES

        # 用 preprocessor 提供的划分结果
        if mode == 'train':
            video_list = preprocessor.train_list
        elif mode == 'val':
            video_list = preprocessor.val_list
        elif mode == 'test':
            video_list = preprocessor.test_list
        else:
            raise ValueError(f"Unknown mode {mode}")

        # 展开 clip_dict
        self.poses, self.labels, self.video_names, self.metadata = [], [], [], []
        for v in video_list:
            clips = preprocessor.clip_dict[v]   # list of (T, 17, 3)
            label = preprocessor.labels_dict[v]
            # meta = preprocessor.identity_labels_dict[v] if hasattr(preprocessor, "identity_labels_dict") else []

            for c in clips:
                self.poses.append(c)
                self.labels.append(label)
                self.video_names.append(v)
                # self.metadata.append(meta)

        self.video_name_to_index = {name: i for i, name in enumerate(self.video_names)}
        self._pose_dim = 3 * self._NMAJOR_JOINTS

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = np.array(self.poses[idx], dtype=np.float32)   # (T, V, C)
        label = self.labels[idx]
        video_idx = self.video_name_to_index[self.video_names[idx]]

        # --- 非 GCN ---
        if not self.gcn_mode:
            joints = _MAJOR_JOINTS
            x = x[:, joints, :]

            if self._params['in_data_dim'] == 2:
                if self._params['simulate_confidence_score']:
                    x[..., 2] = 1
                else:
                    x = x[..., :2]
            elif self._params['in_data_dim'] == 3:
                x = x[..., :3]

            if self._params['merge_last_dim']:
                N = x.shape[0]
                x = x.reshape(N, -1)

            if x.shape[0] > self._params['source_seq_len']:
                x = x[:self._params['source_seq_len']]
            elif x.shape[0] < self._params['source_seq_len']:
                raise ValueError("Clip shorter than expected length.")

        # --- GCN 模式 ---
        else:
            if x.shape[0] > self._params['source_seq_len']:
                x = x[:self._params['source_seq_len']]
            elif x.shape[0] < self._params['source_seq_len']:
                raise ValueError("Clip shorter than expected length.")
            # reshape 成 (C, T, V, 1)
            x = np.transpose(x, (2,0,1))[..., np.newaxis]

        # if len(self._params['metadata']) > 0:
        #     metadata_idx = [METADATA_MAP[element] for element in self._params['metadata']]
        #     md = self.metadata[idx][0][metadata_idx].astype(np.float32)
        # else:
        #     md = []

        sample = {
            'encoder_inputs': x,
            'label': label,
            'labels_str': self._updrs_str[label],
            'video_idx': video_idx,
            # 'metadata': md,
        }
        # if self.transform:
        #     sample = self.transform(sample)
        return sample

def compute_class_weights(data_loader):
    from collections import Counter

    class_counts = Counter()
    total_samples = 0

    for _, targets, _,_ in data_loader:   # 注意这里只解包 3 个
        class_counts.update(targets.tolist())
        total_samples += len(targets)

    num_classes = max(class_counts.keys()) + 1
    class_weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        weight = 0.0 if count == 0 else total_samples / (num_classes * count)
        class_weights.append(weight)

    total_weight = sum(class_weights)
    normalized_class_weights = [w / total_weight for w in class_weights]

    return normalized_class_weights


def collate_fn(batch):
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    labels = torch.from_numpy(np.stack([e['label'] for e in batch]))
    video_idxs = torch.from_numpy(np.stack([e['video_idx'] for e in batch]))
    metadata = torch.zeros((len(batch), 0), dtype=torch.float32)  # 占位
    return e_inp, labels, video_idxs, metadata

def dataset_factory(params, backbone):

    root_dir = f'{path.PREPROCESSED_DATA_ROOT_PATH}/{backbone}_processing'       #backbone应为motionagformer

    backbone_data_location_mapper = {  # backbone特定的数据处理目录
        'motionagformer': os.path.join(root_dir, params['experiment_name'],
                                       f"{params['dataset']}_center_{params['data_centered']}/"),
        'poseformerv2': os.path.join(root_dir, params['experiment_name'],
                                     f"{params['dataset']}_center_{params['data_centered']}/")
    }

    backbone_preprocessor_mapper = {
        'motionagformer': MotionAGFormerPreprocessor,
        'poseformerv2': PoseformerV2Preprocessor
    }

    assert_backbone_is_supported(backbone_data_location_mapper, backbone)

    data_dir = backbone_data_location_mapper[backbone]


    if not os.path.exists(
            data_dir):
        if params['dataset'] == 'PD':
            raw_data = PDReader(params['data_path'], params['labels_path'])
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
    # poses_dict = {}
    # for vname, clips in preprocessor.clip_dict.items():
    #     for idx, clip in enumerate(clips):
    #         clip_name = f"{vname}_clip{idx}"
    #         poses_dict[clip_name] = clip


    train_dataset = ProcessedDataset(preprocessor=preprocessor, params=params,mode='train', transform=train_transform)

    # sample0 = train_dataset[0]
    # print("第0个样本：")
    # for k, v in sample0.items():
    #     if hasattr(v, 'shape'):
    #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    #     else:
    #         print(f"  {k}: {v}")
    #
    # # 随机取几个样本
    # import random
    # for idx in random.sample(range(len(train_dataset)), 3):
    #     sample = train_dataset[idx]
    #     print(f"\n样本 {idx}:")
    #     for k, v in sample.items():
    #         if hasattr(v, 'shape'):
    #             print(f"  {k}: shape={v.shape}")
    #         else:
    #             print(f"  {k}: {v}")
    #
    # # 看看视频名字
    # print("\n视频名检查：")
    # print(sample0.get('video_name', '没有video_name字段'))

    eval_dataset = ProcessedDataset(preprocessor=preprocessor, params=params,mode='val', transform=None)
    test_dataset = ProcessedDataset(preprocessor=preprocessor, params=params,mode='test', transform=None)




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


def load_pretrained_weights(model, checkpoint, strict=True):
    """
    Load pretrained weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - checkpoint (dict): the checkpoint
    """
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

    if(strict):                                 #TODO:先写上，这里默认strict的就是其他模型，否则就是GCN;全部都加载（可是该checkpoint是双人版本训练的，是否会有问题？）
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


def load_pretrained_backbone(params, backbone_name):            #TODO:思考GCN应该怎么接入预训练模型
    if backbone_name == "poseformerv2":
        model_backbone = PoseTransformerV2(num_joints=params['num_joints'],
                                           embed_dim_ratio=params['embed_dim_ratio'],
                                           depth=params['depth'],
                                           number_of_kept_frames=params['number_of_kept_frames'],
                                           number_of_kept_coeffs=params['number_of_kept_coeffs'],
                                           in_chans=2,
                                           num_heads=8,
                                           mlp_ratio=2,
                                           qkv_bias=True,
                                           qk_scale=None,
                                           drop_path_rate=0,
                                           )
        checkpoint = torch.load(params['model_checkpoint_path'], map_location=lambda storage, loc: storage)['model_pos']
    elif backbone_name == "motionagformer":
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


    if backbone_name == "ctrgcn":
        load_pretrained_weights(model_backbone, checkpoint, strict=False)
    else:
        load_pretrained_weights(model_backbone, checkpoint)
    return model_backbone



    load_pretrained_weights(model_backbone, checkpoint)

    return model_backbone




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

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    if fold == 1:
        model_params = count_parameters(model)
        print(f"[INFO] Model has {model_params} parameters.")


    train_model(params, class_weights, train_dataset_fn, val_dataset_fn, model, fold, backbone_name)    #Layer2的入口

    checkpoint_root_path = os.path.join(path.OUT_PATH, params['model_prefix'], 'models', f"fold{fold}")
    best_ckpt_path = os.path.join(checkpoint_root_path, 'best_epoch.pth.tr')
    load_pretrained_weights(model, checkpoint=torch.load(best_ckpt_path)[
        'model'])  # 这里报错了，只在第一个fold上跑完了20个epoch，只保存了latest_epoch.pth.tr，所以加载不出来
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

    res = pd.DataFrame(
        {'total_video_names': total_video_names, 'total_outs_best': total_outs_best, 'total_outs_last': total_outs_last,
         'total_gts': total_gts, 'total_states': total_states})
    with open(os.path.join(rep_out, f'total_results_fold{fold}.pkl'), 'wb') as file:
        pickle.dump(res, file)

    end_time = datetime.datetime.now()

    duration = end_time - start_time
    print(f"Fold {fold} run time:", duration)


def run_tests_for_each_fold(params, splits, backbone_name, device, rep_out):
    total_outs_best, total_outs_last, total_gts, total_logits, total_states, total_video_names = [], [], [], [], [], []
    for fold, (train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights) in enumerate(splits):
        process_fold(fold, params, backbone_name, train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights, device, total_outs_best, total_gts, total_logits, total_states, total_video_names, total_outs_last, rep_out)
    return total_outs_best, total_gts, total_states, total_video_names, total_outs_last


def setup_experiment_path(params):
    exp_path = path.OUT_PATH + os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    params['model_prefix'] = os.path.join(params['model_prefix'], str(params['last_run_foldnum']))
    rep_out = path.OUT_PATH + os.path.join(params['model_prefix'])
    return params, rep_out



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
    elif backbone_name == 'motionagformer':
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


def print_best_model_configuration(best_params, backbone_name):
    print("====================================BEST MODEL====================================================")
    print(f"Trial {best_params['best_trial_number']}, lr: {best_params['lr']}, num_epochs: {best_params['num_epochs']}")
    print(f"classifier_hidden_dims: {map_to_classifier_dim(backbone_name, 'option1')}")
    print(f"optimizer_name: {best_params['optimizer']}, use_weighted_loss: {best_params['use_weighted_loss']}")
    print("========================================================================================")



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

    parser.add_argument('--participant_number', default=23, type=int,
                        help="all participants number")


    args = parser.parse_args()

    param = vars(args)
    param['metadata'] = param['metadata'].split(',') if param['metadata'] else []

    torch.backends.cudnn.benchmark = False

    backbone_name = param['backbone']

    if backbone_name == 'motionagformer':
        conf_path = './configs/motionagformer'
    elif backbone_name == 'poseformerv2':
        conf_path = './configs/poseformerv2'
    else:
        raise NotImplementedError(f"Backbone '{backbone_name}' is not supported")

    # 获取排序后的文件列表并只取第一个文件
    file_list = sorted(os.listdir(conf_path))
    if file_list:  # 确保列表不为空
        fi = file_list[0]  # 只取第一个文件

        if backbone_name == 'poseformerv2':
            params, new_params = generate_config_poseformerv2.generate_config(param, fi)

        elif backbone_name == 'motionagformer':
            params, new_params = generate_config_motionagformer.generate_config(param, fi)


        if param['dataset'] == 'PD':
            params['num_classes'] = 3      #
        else:
            raise NotImplementedError(f"dataset '{param['dataset']}' is not supported.")

        set_random_seed(param['seed'])

        # test_and_report(params, new_params, all_folds, backbone_name, _DEVICE)

        params, rep_out = setup_experiment_path(params)
        configure_params_for_best_model(params, backbone_name)        #不需要



        initialize_wandb(params)

        splits = []
        train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name)
        splits.append((train_dataset_fn, val_dataset_fn, test_dataset_fn, class_weights))


        # run_tests_for_each_fold(params, splits, backbone_name, _DEVICE, rep_out)
        total_outs_best, total_gts, total_states, total_video_names, total_outs_last=run_tests_for_each_fold(params, splits, backbone_name, _DEVICE, rep_out)
        process_reports(total_outs_best, total_outs_last, total_gts, total_states, rep_out)
        save_and_load_results(total_video_names, total_outs_best, total_outs_last, total_gts, rep_out)
        wandb.finish()
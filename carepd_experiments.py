import numpy as np
import pickle
import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torchvision import transforms  # 用来串联增强操作
from data.augmentations import *

from model.ctrgcn.ctrgcn import Model as CTRGCN

from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict, Counter


# ==============================================================================
# 总参数配置
# ==============================================================================
NPZ_PATH = 'care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz'
PKL_LABEL_PATH = 'care-pd-dataset/PD-GaM.pkl'
FOLD_PATH = 'care-pd-pdgam-folds/PD-GaM_6fold_participants.pkl'
OUTPUT_DIR = 'care-pd-dataset/ctrgcn_processing/PD_center_True/'

SOURCE_SEQ_LEN = 81
ROOT_JOINT = 0

USE_CAUSAL = False
USE_DATA_AUGMENTATION = False
USE_PRETRAINED = False #未实现
CENTER_POSE = True #未实现

# ==============================================================================
# -1. 核心算法模块
# ==============================================================================

def coral_loss(logits, y, num_classes=4):
    """
    CORAL (Consistent Rank Logits) Loss
    用于序数回归 (Ordinal Regression)，比如 0,1,2,3 分的预测。
    """
    target = torch.zeros_like(logits)
    for j in range(1, num_classes):
        target[:, j - 1] = (y >= j).float()
    return F.binary_cross_entropy_with_logits(logits, target)


class OrdinalHead(nn.Module):
    """序数回归预测头：输出 K-1 个 logits"""

    def __init__(self, input_dim, hidden_dim, num_classes=4, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 4分类只需要 3 个 logits 来划定边界
        self.out = nn.Linear(hidden_dim, num_classes - 1)

    def forward(self, x):
        return self.out(self.fc(x))




class OrdinalClassBalancedMemory(nn.Module):
    """类别均衡的显式记忆库 (0, 1, 2, 3 四个类别)"""

    def __init__(self, num_classes=4, feat_dim=128, memory_per_class=256, device='cuda'):
        super(OrdinalClassBalancedMemory, self).__init__()
        self.num_classes = num_classes
        self.memory_per_class = memory_per_class
        self.device = device

        for c in range(num_classes):
            init_feats = F.normalize(torch.randn(memory_per_class, feat_dim), dim=1)
            self.register_buffer(f'mem_feats_{c}', init_feats)
            self.register_buffer(f'ptr_{c}', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features, labels):
        features = features.detach()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0: continue

            feats_c = features[mask]
            num_to_add = feats_c.size(0)

            ptr = getattr(self, f'ptr_{c}')
            mem = getattr(self, f'mem_feats_{c}')

            current_ptr = ptr.item()
            if current_ptr + num_to_add <= self.memory_per_class:
                mem[current_ptr: current_ptr + num_to_add] = feats_c
                ptr[0] = (current_ptr + num_to_add) % self.memory_per_class
            else:
                first_part = self.memory_per_class - current_ptr
                mem[current_ptr:] = feats_c[:first_part]
                second_part = num_to_add - first_part
                mem[:second_part] = feats_c[first_part:]
                ptr[0] = second_part

    def get_memory(self):
        feats_list, labels_list = [], []
        for c in range(self.num_classes):
            feats_list.append(getattr(self, f'mem_feats_{c}'))
            labels_list.append(torch.full((self.memory_per_class,), c, dtype=torch.long, device=self.device))
        return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


class MemoryCausalOrdinalLoss(nn.Module):
    """基于记忆库的因果序数度量损失"""

    def __init__(self, margin_base=0.1, alpha=0.1, topk=5, memory_bank=None):
        super(MemoryCausalOrdinalLoss, self).__init__()
        self.margin_base = margin_base
        self.alpha = alpha
        self.topk = topk
        self.memory_bank = memory_bank

    def forward(self, batch_feats, batch_labels, epoch=0):
        mem_feats, mem_labels = self.memory_bank.get_memory()
        sim_mat = torch.matmul(batch_feats, mem_feats.T)
        B = batch_feats.size(0)

        label_diff_mat = torch.abs(batch_labels.unsqueeze(1) - mem_labels.unsqueeze(0)).float()
        mask_pos = (label_diff_mat == 0)
        mask_neg = (label_diff_mat > 0)

        sim_pos_safe = torch.where(mask_pos, sim_mat, torch.tensor(1.0).to(sim_mat.device))
        min_pos_sim, _ = sim_pos_safe.min(dim=1)
        mcd_threshold = min_pos_sim - self.margin_base

        if epoch >= 5:
            confusing_neg_mask = mask_neg & (sim_mat > mcd_threshold.unsqueeze(1))
        else:
            confusing_neg_mask = mask_neg

        loss = torch.tensor(0.0).to(sim_mat.device)
        valid_triplets = 0

        for i in range(B):
            if not mask_pos[i].any(): continue
            s_ap = min_pos_sim[i]
            neg_indices = confusing_neg_mask[i].nonzero(as_tuple=True)[0]
            if len(neg_indices) == 0: continue

            s_an_candidates = sim_mat[i, neg_indices]
            if len(neg_indices) > self.topk:
                _, indices = torch.topk(s_an_candidates, k=self.topk)
                final_neg_indices = neg_indices[indices]
            else:
                final_neg_indices = neg_indices

            for neg_idx in final_neg_indices:
                s_an = sim_mat[i, neg_idx]
                diff = label_diff_mat[i, neg_idx]
                dynamic_margin = self.margin_base + self.alpha * diff
                loss += F.relu(s_an - s_ap + dynamic_margin)
                valid_triplets += 1

        self.memory_bank.update(batch_feats, batch_labels)
        return loss / valid_triplets if valid_triplets > 0 else loss




class CounterfactualCausalModeling(nn.Module):
    """极致精简版：只有 Backbone + CORAL头 + 度量投影头"""

    def __init__(self, backbone, input_dim=256, hidden_dim=256, z_dim=128, num_classes=4):
        super(CounterfactualCausalModeling, self).__init__()
        self.backbone = backbone

        # 事实回归头 (CORAL)
        self.regressor = OrdinalHead(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)

        # 度量学习投影头
        self.metric_projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, z_dim)
        )

    def forward(self, inputs):
        # 1. 过 GCN 骨干网
        features = self.backbone(inputs)  # 预期输出: [B, C, T, V] 或类似形状

        # 如果出来是 4 维，全局平均池化掉时空维度
        if features.ndim == 4:
            feature_pooled = features.mean(dim=(2, 3))
        else:
            feature_pooled = features

        # 2. 分类预测
        logits = self.regressor(feature_pooled)

        # 3. 度量学习特征 (归一化)
        metric_feats = F.normalize(self.metric_projector(feature_pooled), p=2, dim=1)

        return {
            "logits": logits,
            "features": metric_feats
        }



# ==============================================================================
# 0. 预处理函数群 (已集成 25点映射 & 短序列循环填充)
# ==============================================================================

def center_poses(poses):
    """每一帧都减去根节点坐标，实现原地太空步"""
    # poses: (T, 17, 3)
    # H36M 的骨盆(Pelvis)索引是 0
    return poses - poses[:, 0:1, :]

def convert_h36m_to_ntu25(sequence):
    if sequence.ndim != 3 or sequence.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected (T, 17, 3), got {sequence.shape}"
        )

    output = np.zeros(
        (sequence.shape[0], 25, 3),
        dtype=sequence.dtype
    )

    # Torso
    output[:, 0] = sequence[:, 0]    # Spine Base / pelvis
    output[:, 1] = sequence[:, 7]    # Spine Mid
    output[:, 20] = sequence[:, 8]   # Spine Shoulder
    output[:, 2] = sequence[:, 9]    # Neck
    output[:, 3] = sequence[:, 10]   # Head

    # Left arm
    output[:, 4] = sequence[:, 11]
    output[:, 5] = sequence[:, 12]
    output[:, 6] = sequence[:, 13]
    output[:, 7] = sequence[:, 13]
    output[:, 21] = sequence[:, 13]
    output[:, 22] = sequence[:, 13]

    # Right arm
    output[:, 8] = sequence[:, 14]
    output[:, 9] = sequence[:, 15]
    output[:, 10] = sequence[:, 16]
    output[:, 11] = sequence[:, 16]
    output[:, 23] = sequence[:, 16]
    output[:, 24] = sequence[:, 16]

    # Left leg
    output[:, 12] = sequence[:, 4]
    output[:, 13] = sequence[:, 5]
    output[:, 14] = sequence[:, 6]
    output[:, 15] = sequence[:, 6]

    # Right leg
    output[:, 16] = sequence[:, 1]
    output[:, 17] = sequence[:, 2]
    output[:, 18] = sequence[:, 3]
    output[:, 19] = sequence[:, 3]

    return output


def get_gcn_clips(video_sequence, clip_length):
    """时序切片 (包含短片段循环填充)，产出 GCN 格式 (3, T, 25, 1)"""
    clips = []
    video_length = video_sequence.shape[0]

    # 【核心修正】：如果视频不够长，进行循环拼接 (Loop Padding)
    if video_length < clip_length:
        num_repeats = (clip_length // video_length) + 1
        padded_video = np.tile(video_sequence, (num_repeats, 1, 1))[:clip_length]

        # 核心几何变换：(T, 25, 3) -> (3, T, 25) -> (3, T, 25, 1)
        clip_expanded = np.transpose(padded_video, (2, 0, 1))[..., np.newaxis]
        clips.append(clip_expanded)
        return clips

    # 长片段正常滑窗
    start_frame = 0
    stride = clip_length
    while (video_length - start_frame) >= clip_length:
        clip = video_sequence[start_frame: start_frame + clip_length]  # (T, 25, 3)
        clip_expanded = np.transpose(clip, (2, 0, 1))[..., np.newaxis]
        clips.append(clip_expanded)
        start_frame += stride

    return clips




os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 1. 加载数据、fold、标签
# ==============================================================================
print("[1/3] 正在加载原始数据、Folds 划分以及真实标签...")
raw_data = np.load(NPZ_PATH, allow_pickle=True)
with open(FOLD_PATH, 'rb') as f:
    folds_config = pickle.load(f)
with open(PKL_LABEL_PATH, 'rb') as f:
    label_db = pickle.load(f)

label_mapping = {}
print("  [INFO] 正在建立 [Subject_ID][Walk_ID] -> UPDRS_GAIT 映射表...")
for sub_id, walks in label_db.items():
    for walk_id, attr_dict in walks.items():
        # 确保 label 是 int 类型
        label_mapping[f"{sub_id}__{walk_id}"] = int(attr_dict.get('UPDRS_GAIT', -1))

# ==============================================================================
# 2. 按 Fold 进行全量处理与分流打包
# ==============================================================================
print("[2/3] 开始遍历 Fold 分流处理...")
all_video_names = list(raw_data.keys())

for fold_idx in sorted(folds_config.keys()):
    print(f"--- 正在处理 Fold {fold_idx} ---")
    fold_info = folds_config[fold_idx]

    train_subjects = fold_info.get('train', [])
    test_subjects = fold_info.get('eval', fold_info.get('test', []))

    splits = {
        'train': {'pose': [], 'label': [], 'video_name': [], 'metadata': []},
        'test': {'pose': [], 'label': [], 'video_name': [], 'metadata': []}
    }

    for v_name in all_video_names:
        sub_id = v_name.split('__')[0]

        if sub_id in train_subjects:
            target_split = 'train'
        elif sub_id in test_subjects:
            target_split = 'test'
        else:
            continue

        label = label_mapping.get(v_name, -1)
        if label == -1:
            continue

        # ==========================================
        # 核心数据流转线 (严格按顺序)
        # ==========================================
        # 1. 读取 (T, 17, 3)
        poses = raw_data[v_name]

        # 2. 居中 (T, 17, 3)
        centered_poses = center_poses(poses)

        # 3. H36M 转 NTU25拓扑 (T, 25, 3)
        ntu25_poses = convert_h36m_to_ntu25(centered_poses)

        # 4. 时序切片与塑形 -> (3, 81, 25, 1)
        clips = get_gcn_clips(ntu25_poses, SOURCE_SEQ_LEN)
        # ==========================================

        fake_metadata = np.zeros((1, 5))

        for clip in clips:
            splits[target_split]['pose'].append(clip)
            splits[target_split]['label'].append(label)
            splits[target_split]['video_name'].append(v_name)
            splits[target_split]['metadata'].append(fake_metadata)

    print(f"  [SAVE] 正在写入 Fold {fold_idx} 的打包文件...")

    for mode in ['train', 'test']:
        out_path = os.path.join(OUTPUT_DIR, f"PD_{mode}_{fold_idx}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(splits[mode], f)
        print(f"    -> 已生成: {out_path} (有效样本数: {len(splits[mode]['video_name'])})")

    val_out_path = os.path.join(OUTPUT_DIR, f"PD_validation_{fold_idx}.pkl")
    with open(val_out_path, 'wb') as f:
        pickle.dump(splits['test'], f)
    print(f"    -> 已生成镜像验证集: {val_out_path} (等同于 test 数据)")


# ==============================================================================
# 3. 数据加载器编写
# ==============================================================================


class CarePDSkeletonDataset(Dataset):
    def __init__(self, pkl_path, is_train=False, transform=None):
        self.is_train = is_train
        self.transform = transform

        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        self.poses = self.data_dict['pose']
        self.labels = self.data_dict['label']
        self.names = self.data_dict['video_name']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 强制转为 float32 避免精度冲突
        pose_np = self.poses[index].astype(np.float32)
        label = self.labels[index]
        name = self.names[index]

        sample = {
            'encoder_inputs': pose_np,
            'label': label,
            'labels_str': name
        }

        # 只有训练集且设置了 transform 才增强
        if self.is_train and self.transform is not None:
            sample = self.transform(sample)

        data_tensor = sample['encoder_inputs']
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)

        # 【核心修改】：第三个返回值改为 name，适配下游评估函数
        return data_tensor, label_tensor, sample['labels_str']


# ==========================================
# 4. 组装数据增强 Pipeline (只在 Train 阶段开启)
# ==========================================
# ⚠️ 注意：这里强制指定了 format='ntu25'，因为我们之前已经映射过了
train_transforms = transforms.Compose([
    MirrorReflection(format='ntu25', data_dim=3),
    RandomRotation(min_rotate=-15, max_rotate=15, data_dim=3),
    RandomNoise(mean=0, std=0.01, data_dim=3),
    axis_mask(data_dim=3)
])

# 测试集不需要任何数据增强
test_transforms = None

# ==========================================
# 5. 实例化 Dataset
# ==========================================
train_dataset = CarePDSkeletonDataset(
    pkl_path='care-pd-dataset/ctrgcn_processing/PD_center_True/PD_train_1.pkl',
    is_train=True,
    transform=train_transforms
)

test_dataset = CarePDSkeletonDataset(
    pkl_path='care-pd-dataset/ctrgcn_processing/PD_center_True/PD_test_1.pkl',
    is_train=False,
    transform=test_transforms
)

# ==========================================
# 6. 创建 DataLoader (供模型按 Batch 抽取)
# ==========================================
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,      # 根据你的显存调整
    shuffle=True,       # 训练集必须打乱
    num_workers=4,      # 多线程加载，加速 GPU 喂数据
    pin_memory=True     # 加快数据转移到 GPU 的速度
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,      # 测试集不需要打乱
    num_workers=4,
    pin_memory=True
)

# 测试一下 DataLoader 是否跑通
for batch_data, batch_labels, batch_idx in train_loader:
    print("Batch Data Shape:", batch_data.shape)   # 期望输出: [32, 3, 81, 25, 1]
    print("Batch Labels Shape:", batch_labels.shape) # 期望输出: [32]
    break # 测试成功即退出




# ==========================================
# 7. 参数设置
# ==========================================
# 直接在代码最前面定义，一目了然
params = {
    # --- 数据参数 ---
    'dataset': 'PD',
    'num_classes': 4,  # ⚠️最大改动之一
    'in_channels': 3,
    'num_point': 25,
    'num_person': 1,
    'source_seq_len': 81,

    # --- 模型与训练参数 ---
    'dim_rep': 256,
    'epochs': 20,
    'batch_size': 32,  # 建议先从 32 开始跑，防显存爆
    'lr': 0.001,  # 合并 lr_head 和 lr_backbone，先用统一 LR 跑通
    'weight_decay': 0.0001,
    'lr_step_size': 10,
    'lr_decay': 0.1,
    'stopping_tolerance': 10,

    # --- GRL 与 度量学习 (CIML) ---
    'max_grl_lambda': 1.0,
    'grl_warmup_epochs': 10,
    'grl_loss_weight': 0.1,

    # 如果你有预训练权重可以留着，没有就设为 None 从头训
    'model_checkpoint_path': "Pretrained_checkpoints/ctrgcn/ctrgcn/runs-58-57072.pt"
}


# ==========================================
# 8. 设置运行
# ==========================================




def aggregate_by_walk(preds, labels, names):
    walk_preds = defaultdict(list)
    walk_labels = {}

    for pred, label, name in zip(preds, labels, names):
        walk_preds[name].append(int(pred))
        walk_labels[name] = int(label)

    final_preds = []
    final_labels = []
    final_names = []

    for name, clip_preds in walk_preds.items():
        majority_pred = Counter(clip_preds).most_common(1)[0][0]

        final_preds.append(majority_pred)
        final_labels.append(walk_labels[name])
        final_names.append(name)

    return final_preds, final_labels, final_names

def log_results_fallback(report, conf_matrix, txt_name, img_name, output_dir):
    with open(os.path.join(output_dir, txt_name), 'w') as f:
        f.write(report)
        f.write('\n\nConfusion Matrix:\n')
        f.write(str(conf_matrix))



def evaluate_model(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for x, y, names in loader:
            x = x.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            if USE_CAUSAL:
                # CORAL / ordinal model
                outputs = model(x)
                ordinal_logits = outputs["logits"]

                # sigmoid(logit) > 0.5 等价于 logit > 0
                preds = (ordinal_logits > 0).sum(dim=1)

            else:
                # CTR-GCN baseline
                logits = model(x)
                preds = torch.argmax(logits, dim=1)

            all_preds.extend(
                preds.detach().cpu().numpy().tolist()
            )

            all_labels.extend(
                y.detach().cpu().numpy().tolist()
            )

            all_names.extend(list(names))

    if len(all_labels) == 0:
        raise RuntimeError(
            "Evaluation loader is empty. No samples were evaluated."
        )

    all_preds_np = np.asarray(all_preds, dtype=np.int64)
    all_labels_np = np.asarray(all_labels, dtype=np.int64)

    acc = float(
        np.mean(all_preds_np == all_labels_np)
    )

    return acc, all_preds, all_labels, all_names


def run_6_fold_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 20
    NUM_FOLDS = 6
    OUTPUT_DIR = "care-pd-dataset/ctrgcn_processing/models_out"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 宏观收集容器
    total_outs_last = []
    total_gts = []
    total_video_names = []
    total_states = []

    print("=" * 60)
    print(f"🚀 开始 6-Fold 全自动训练与评测打榜 | 设备: {device}")

    for fold in range(1, NUM_FOLDS + 1):
        print("\n" + "=" * 40)
        print(f"🌟 正在执行 Fold {fold}/{NUM_FOLDS}")
        print("=" * 40)

        # 1. 挂载当前折数据
        train_pkl = f'care-pd-dataset/ctrgcn_processing/PD_center_True/PD_train_{fold}.pkl'
        test_pkl = f'care-pd-dataset/ctrgcn_processing/PD_center_True/PD_test_{fold}.pkl'

        current_train_transforms = train_transforms if USE_DATA_AUGMENTATION else None      #0720:开关数据增强的按钮

        print(
            f"[CONFIG] Data augmentation: "
            f"{'ON' if USE_DATA_AUGMENTATION else 'OFF'}"
        )

        train_loader = DataLoader(
            CarePDSkeletonDataset(
                train_pkl,
                is_train=True,
                transform=current_train_transforms
            ),
            batch_size=64,
            shuffle=True
        )

        test_loader = DataLoader(
            CarePDSkeletonDataset(
                test_pkl,
                is_train=False,
                transform=None
            ),
            batch_size=64,
            shuffle=False
        )

        # 2. 重新初始化模型（必须放在循环里，确保每折是全新权重）
        # model_backbone = CTRGCN(num_class=4, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', in_channels=3)
        # model = CounterfactualCausalModeling(model_backbone, input_dim=256, z_dim=128, num_classes=4).to(device)

        model_backbone = CTRGCN(num_class=4, num_point=25, num_person=1, graph='graph.ntu_rgb_d.Graph', in_channels=3)

        # 2. 根据开关选择模型和损失函数
        if USE_CAUSAL:
            # --- 复杂版本 (真模型) ---
            model = CounterfactualCausalModeling(model_backbone, input_dim=256, z_dim=128, num_classes=4).to(device)
            memory_bank = OrdinalClassBalancedMemory(num_classes=4, feat_dim=128, device=device)
            ciml_criterion = MemoryCausalOrdinalLoss(margin_base=0.1, alpha=0.1, topk=5, memory_bank=memory_bank).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        else:
            # --- 极简消融版本 (纯 CTR-GCN) ---
            # 直接把 backbone 接一个线性层
            model = nn.Sequential(
                model_backbone,
                nn.Linear(256, 4)
            ).to(device)
            # 使用最标准的交叉熵
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


        # 3. 开始该折训练
        for epoch in range(EPOCHS):
            model.train()

            running_loss = 0.0
            num_batches = 0

            loop = tqdm(
                train_loader,
                desc=f'Fold {fold} Epoch {epoch + 1}/{EPOCHS}'
            )

            for x, y, _ in loop:
                x = x.to(device)
                y = y.to(device).long()

                optimizer.zero_grad()

                if USE_CAUSAL:
                    outputs = model(x)

                    main_loss = coral_loss(
                        outputs["logits"],
                        y,
                        num_classes=4
                    )

                    ciml_loss = ciml_criterion(
                        outputs["features"],
                        y,
                        epoch=epoch
                    )

                    loss = main_loss + 0.1 * ciml_loss

                else:
                    # CTR-GCN baseline
                    logits = model(x)
                    loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

                # 显示当前 batch loss，而不是 epoch 平均值
                loop.set_postfix(
                    BatchLoss=f"{loss.item():.4f}"
                )

            scheduler.step()

            # 当前 epoch 的平均训练损失
            epoch_loss = running_loss / max(num_batches, 1)

            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"   -> Fold {fold} Epoch {epoch + 1}/{EPOCHS} "
                f"| Train Loss: {epoch_loss:.4f} "
                f"| LR: {current_lr:.6f}"
            )

        # ============================================================
        # 4. 当前 Fold 训练结束
        #    现在才第一次使用 test_loader
        # ============================================================
        print(
            f"🎉 Fold {fold} 训练完成，"
            f"开始在外层测试集上进行最终评价。"
        )

        # 保存固定 epoch 的最终模型
        checkpoint_path = os.path.join(
            OUTPUT_DIR,
            f"last_model_fold{fold}.pt"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        # 测试集只评价一次
        test_acc, test_preds, test_labels, test_names = evaluate_model(
            model,
            test_loader,
            device
        )

        print(
            f"   -> Fold {fold} Final Test Acc: "
            f"{test_acc * 100:.2f}%"
        )

        print(
            f"   -> Final checkpoint saved to: "
            f"{checkpoint_path}"
        )

        # 将当前 Fold 的最终预测加入六折汇总
        total_outs_last.extend(test_preds)
        total_gts.extend(test_labels)
        total_video_names.extend(test_names)

        # CARE-PD 当前没有使用 medication state
        total_states.extend(
            ['UNKNOWN'] * len(test_labels)
        )

    # ==============================================================================
    #  6 折全部跑完：生成终极报告
    # ==============================================================================


    print("\n" + "=" * 60)
    print("🏆 6 折交叉验证全部结束，开始生成最终报告")

    # ------------------------------------------------------------------------------
    # A. Clip-level 报告
    # ------------------------------------------------------------------------------
    print("\n========== CLIP-LEVEL REPORT ==========")

    clip_report = classification_report(
        total_gts,
        total_outs_last,
        labels=[0, 1, 2, 3],
        zero_division=0,
        digits=4
    )

    clip_confusion = confusion_matrix(
        total_gts,
        total_outs_last,
        labels=[0, 1, 2, 3]
    )

    print(clip_report)
    print("\nClip-level Confusion Matrix:")
    print(clip_confusion)

    log_results_fallback(
        clip_report,
        clip_confusion,
        "clip_level_report_allfolds.txt",
        None,
        OUTPUT_DIR
    )

    # ------------------------------------------------------------------------------
    # B. Walk-level 聚合
    # ------------------------------------------------------------------------------
    walk_preds, walk_labels, walk_names = aggregate_by_walk(
        total_outs_last,
        total_gts,
        total_video_names
    )

    print("\n========== WALK-LEVEL REPORT ==========")
    print(
        f"Clip count: {len(total_gts)} | "
        f"Walk count: {len(walk_labels)}"
    )

    walk_report = classification_report(
        walk_labels,
        walk_preds,
        labels=[0, 1, 2, 3],
        zero_division=0,
        digits=4
    )

    walk_confusion = confusion_matrix(
        walk_labels,
        walk_preds,
        labels=[0, 1, 2, 3]
    )

    print(walk_report)
    print("\nWalk-level Confusion Matrix:")
    print(walk_confusion)

    log_results_fallback(
        walk_report,
        walk_confusion,
        "walk_level_report_allfolds.txt",
        None,
        OUTPUT_DIR
    )


run_6_fold_experiment()


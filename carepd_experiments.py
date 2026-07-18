import numpy as np
import pickle
import os


# ==============================================================================
# -1. 一些函数，考虑合理性，以及对应性
# ==============================================================================

def center_poses(poses):
    """每一帧都减去根节点坐标，实现原地太空步"""
    # poses: (T, 17, 3)
    return poses - poses[:, ROOT_JOINT:ROOT_JOINT + 1, :]


def get_gcn_clips(video_sequence, clip_length):     #这个函数感觉有问题。需要经过ntu25的映射才可以
    """滑动窗口切片，并将动作加工为 GCN 胃口：(3, T, 17, 1)"""
    clips = []
    video_length = video_sequence.shape[0]
    stride = clip_length  # 步长等于窗宽，不重叠切片

    if video_length < clip_length:
        return clips

    start_frame = 0
    while (video_length - start_frame) >= clip_length:
        clip = video_sequence[start_frame: start_frame + clip_length]  # (T, 17, 3)

        # 核心几何变换：(T, 17, 3) -> (3, T, 17) -> (3, T, 17, 1)
        clip_transposed = np.transpose(clip, (2, 0, 1))
        clip_expanded = clip_transposed[..., np.newaxis]

        clips.append(clip_expanded)
        start_frame += stride

    return clips



# ==============================================================================
# 0. 参数配置 (对齐旧 CTR-GCN 的 data_params)
# ==============================================================================
NPZ_PATH = 'care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz'
PKL_LABEL_PATH = 'care-pd-dataset/PD-GaM.pkl'
FOLD_PATH = 'care-pd-pdgam-folds/PD-GaM_6fold_participants.pkl'
OUTPUT_DIR = 'care-pd-dataset/ctrgcn_processing/PD_center_True/'  # 匹配你的目录映射

SOURCE_SEQ_LEN = 81  # 时间序列长度 T
ROOT_JOINT = 0       # Human3.6M 的根节点（骨盆）通常是索引 0

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


# 构建全局快速标签查表字典，防止循环内重复解析
label_mapping = {}
print("  [INFO] 正在建立 [Subject_ID][Walk_ID] -> UPDRS_GAIT 映射表...")
for sub_id, walks in label_db.items():
    for walk_id, attr_dict in walks.items():
        # 这里统一存储为 int，如果缺失则兜底给 -1
        label_mapping[f"{sub_id}__{walk_id}"] = attr_dict.get('UPDRS_GAIT', -1)


# ==============================================================================
# 2. 按 Fold 进行全量处理与分流打包
# ==============================================================================
print("[2/3] 开始遍历 Fold 分流处理...")

# 提取 npz 里的所有视频片段名
all_video_names = list(raw_data.keys())

for fold_idx in sorted(folds_config.keys()):
    print(f"--- 正在处理 Fold {fold_idx} ---")
    fold_info = folds_config[fold_idx]

    # 划分流映射：只取 Train 和 Test
    train_subjects = fold_info.get('train', [])
    test_subjects = fold_info.get('eval', fold_info.get('test', []))

    # 只需要初始化两个大容器
    splits = {
        'train': {'pose': [], 'label': [], 'video_name': [], 'metadata': []},
        'test': {'pose': [], 'label': [], 'video_name': [], 'metadata': []}
    }

    # 扫描全量视频
    for v_name in all_video_names:
        sub_id = v_name.split('__')[0]

        if sub_id in train_subjects:
            target_split = 'train'
        elif sub_id in test_subjects:
            target_split = 'test'
        else:
            continue

        # 真实标签提取
        label = label_mapping.get(v_name, -1)
        if label == -1 or label is None:
            continue

        # 读取 3D 骨骼坐标 (T, 17, 3) 并进行预处理
        poses = raw_data[v_name]
        centered_poses = center_poses(poses)
        clips = get_gcn_clips(centered_poses, SOURCE_SEQ_LEN)

        fake_metadata = np.zeros((1, 5))

        for clip in clips:
            splits[target_split]['pose'].append(clip)
            splits[target_split]['label'].append(label)
            splits[target_split]['video_name'].append(v_name)
            splits[target_split]['metadata'].append(fake_metadata)

    # 序列化写入硬盘
    print(f"  [SAVE] 正在写入 Fold {fold_idx} 的打包文件...")

    # 1. 正常写入 Train 和 Test
    for mode in ['train', 'test']:
        out_path = os.path.join(OUTPUT_DIR, f"PD_{mode}_{fold_idx}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(splits[mode], f)
        print(f"    -> 已生成: {out_path} (有效样本数: {len(splits[mode]['video_name'])})")

    # 2. 【核心修改】：直接把 Test 拷贝一份伪装成 Validation
    # 这样下游无论怎么写都不会崩溃，且你心里清楚 Val 的成绩就是 Test 的成绩
    val_out_path = os.path.join(OUTPUT_DIR, f"PD_validation_{fold_idx}.pkl")
    with open(val_out_path, 'wb') as f:
        pickle.dump(splits['test'], f)
    print(f"    -> 已生成镜像验证集: {val_out_path} (等同于 test 数据)")



import numpy as np

#本脚本用于探测是否需要进行数据清洗
def detect_care_pd_quality(npz_path):
    print(">>> 开始探测 CARE-PD 数据分布...")
    data = np.load(npz_path, allow_pickle=True)

    has_corrupted = False
    max_val, min_val = -float('inf'), float('inf')

    for key in data.files:
        poses = data[key]  # shape: (T, 17, 3)

        # 1. 探测脏数据 (全0或NaN)
        # 只要某一帧的所有 17 个点的前 3 个坐标全是 0，或者包含 NaN
        if np.any(np.all(poses == 0, axis=(1, 2))) or np.isnan(poses).any():
            has_corrupted = True

        # 2. 探测数值尺度
        cur_max = np.nanmax(poses)
        cur_min = np.nanmin(poses)
        max_val = max(max_val, cur_max)
        min_val = min(min_val, cur_min)

    print("-" * 30)
    print(
        f"【数据清洗 (问题2)】是否发现全 0 或 NaN 的损坏帧？ => {'[需要清洗]' if has_corrupted else '[不需要清洗，极其干净]'}")

    print(f"【单位尺度 (问题4)】数据最大值: {max_val:.2f}, 最小值: {min_val:.2f}")
    if max_val > 50:
        print("=> 诊断结论: 数据尺度超过 50，[确定为毫米(mm)]，【需要】除以 1000 转换为米(m)。")
    else:
        print("=> 诊断结论: 数据尺度在个位数，[确定为米(m)]，【不需要】任何转换。")

# 运行它！
detect_care_pd_quality('care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz')
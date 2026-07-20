import numpy as np
import time

def main():
    # 把你想要测试的 npz 文件路径都写在这里
    file_paths = [
        'care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz',
        # 'care-pd-dataset/另外一个文件.npz',
        # 'care-pd-dataset/第三个文件.npz'
    ]

    for npz_path in file_paths:
        print(f"\n正在扫描文件: {npz_path}")
        start_time = time.time()

        try:
            # 加载数据
            data_dict = np.load(npz_path, allow_pickle=True)
            walk_ids = list(data_dict.keys())

            if len(walk_ids) == 0:
                print("  - 警告：这是一个空文件！")
                continue

            # 提取所有样本的帧数 (shape的第一个维度)
            lengths = [data_dict[w_id].shape[0] for w_id in walk_ids]

            # 计算统计信息
            min_len = min(lengths)
            max_len = max(lengths)
            mean_len = sum(lengths) / len(lengths)

            # 统计低于某个阈值（比如30帧）的极短样本数量
            short_clips = sum(1 for l in lengths if l < 30)

            print(f"  - 总样本数: {len(walk_ids)} 段")
            print(f"  - 最短帧数: {min_len} 帧")
            print(f"  - 最长帧数: {max_len} 帧")
            print(f"  - 平均帧数: {mean_len:.1f} 帧")
            print(f"  - 小于30帧的超短样本数: {short_clips} 段")
            print(f"  - 扫描耗时: {time.time() - start_time:.2f} 秒")

        except Exception as e:
            print(f"  - 读取失败，错误信息: {e}")


if __name__ == "__main__":
    main()

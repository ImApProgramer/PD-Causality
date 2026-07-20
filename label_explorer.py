import pickle
import numpy as np

pkl_path = 'care-pd-dataset/PD-GaM.pkl'  # 请确保路径与你本地一致

print("=" * 60)
print(f"[INFO] 开始深挖核心数据集文件: {pkl_path}")
print("=" * 60)

with open(pkl_path, 'rb') as f:
    db = pickle.load(f)

print(f"顶层数据类型: {type(db)}")

if isinstance(db, dict):
    subjects = list(db.keys())
    print(f"总共包含受试者数量: {len(subjects)}")
    print(f"前 5 个受试者 ID 样例: {subjects[:5]}")

    # 深入第一位受试者
    first_sub = subjects[0]
    walks_dict = db[first_sub]

    print(f"\n--- 深入检查受试者 '{first_sub}' 的内部结构 ---")
    print(f"数据类型: {type(walks_dict)}")

    if isinstance(walks_dict, dict):
        walk_ids = list(walks_dict.keys())
        print(f"该受试者拥有的 walk_id 数量: {len(walk_ids)}")
        print(f"walk_id 样例: {walk_ids[:3]}")

        # 深入第一段步行数据
        first_walk = walk_ids[0]
        data_content = walks_dict[first_walk]

        print(f"\n--- 深入检查单段步行数据 '{first_walk}' 的叶子节点 ---")
        print(f"数据类型: {type(data_content)}")

        if isinstance(data_content, dict):
            print("【大发现！】成功找到标准字典属性列表 (Keys):")
            for k, v in data_content.items():
                if isinstance(v, np.ndarray):
                    print(f"  -> '{k}': 类型=NumPy数组, 形状={v.shape}")
                else:
                    print(f"  -> '{k}': 类型={type(v)}, 值={v}")

            print("\n" + "=" * 60)
            print("【行动结论】")
            if 'UPDRS_GAIT' in data_content:
                print("搞定！标签确实就在这里。")
                print("请把终端打印出的完整 Keys 和属性贴给我，我们立刻把这个标签映射接入一键生成脚本！")
            else:
                print("这里有包含什么临床评分相关的字段吗？请把上面的 Keys 贴出来分析。")
            print("=" * 60)
else:
    print(f"文件内容不是预期字典，前200字符内容为:\n{str(db)[:200]}")
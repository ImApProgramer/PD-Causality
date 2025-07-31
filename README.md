# 0731

准备先完成数据处理。

-----
*现在依然没有搞懂为啥motionBert、poseFormer里面data_path和label_path是同一个目录，而其他的有些却不是*。


----
运行以下代码的时候报错：
```
python data/Visualize_reconst3d.py --data_path /czl_ssd/Public_PD/C3Dfiles_processed_new --backbone poseformer --dataset PD

```


```
Validation Length: 1257
Test Length: 135
Fold:  22
Train Length: 2969
Validation Length: 1022
Test Length: 72
Traceback (most recent call last):
  File "data/Visualize_reconst3d.py", line 194, in <module>
    train_dataset_fn, test_dataset_fn, val_dataset_fn, class_weights = dataset_factory(params, backbone_name, 1)
  File "/root/MotionEncoders_parkinsonism_benchmark-main/data/../data/dataloaders.py", line 783, in dataset_factory
    class_weights = compute_class_weights(train_dataset_fn)
  File "/root/MotionEncoders_parkinsonism_benchmark-main/data/../learning/utils.py", line 74, in compute_class_weights
    for _, targets, _, _ in data_loader:
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 681, in __next__
    data = self._next_data()
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1376, in _next_data
    return self._process_data(data)
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1402, in _process_data
    data.reraise()
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/_utils.py", line 461, in reraise
    raise exception
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/conda/envs/1128/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/root/MotionEncoders_parkinsonism_benchmark-main/data/../data/dataloaders.py", line 631, in __getitem__
    if len(self._params['metadata']) > 0:
KeyError: 'metadata'

```

初步判断应该是读取raw_data的时候，`read_metadata`函数**没有读到数据**。


添加了这个在ProcessedDataset运行之前：
```python

    if 'metadata' not in params:            #用于修复metadata错误，权宜之计，后面一定要回来改
        params['metadata'] = []
```

问题暂时解决.
但是现在，这个可视化需要预训练的权重文件，太麻烦，这里直接放弃，转向训练过程

## 训练

尝试运行:`python /root/.pycharm_helpers/pydev/pydevd.py --multiprocess --qt-support=auto --client localhost --port 43015 --file /root/MotionEncoders_parkinsonism_benchmark-main/eval_encoder.py --backbone poseformer --medication 1`

------
k-fold cross-validation 是一种评估模型泛化能力的方法。

它的基本思想是：
>把原始数据集划分成 k 份（folds），每次选择其中 1 份作为验证集，其余 k-1 份作为训练集，重复 k 次，最终将这 k 次的评估结果做平均，得出模型在“看不见数据”上的综合性能。
------



```css
generate_leave_one_out_folds()                  # ← 主控入口：生成 leave-one-out 的多个fold数据集
├── 检查/创建 save_dir                          # 创建用于保存fold数据的文件夹
├── 构造 video_names_list                       # 获取所有clip的名称列表
├── 准备 val_folds 信息                         # 从 val_xxx_folds.pkl 加载验证集分配（或初始化为空）
│   └── val_folds_exists → pickle.load(...)    # 如果存在就加载，否则准备生成新的
├── for j in range(len(participant_ID)):       # ← 遍历每个参与者，作为 test subject
│   ├── 构造 train/val/test 空列表
│   ├── subject_id = leave-one-out 当前被排除者
│   ├── 构建 class_participants 映射            # 把每个参与者映射到其标签（用于分层采样验证集）
│   ├── if not val_folds_exists:               # 若第一次生成验证集
│   │   ├── 按 class_id 逐类随机采样 2人       # 保证 val 集分布均匀（stratified）
│   │   └── 保存 val_subs_folds.append(...)
│   └── else:
│       └── 使用已有 val_subs = val_subs_folds[j]
│
│   ├── 遍历每个 clip_name → 分配到 train/val/test
│   │   ├── if subject_id → test
│   │   ├── elif in val_subs → val
│   │   └── else → train
│
│   ├── train, val, test = generate_pose_label_videoname(...)   # 构造对应的输入格式
│   └── 保存 train/test/val 到多个 pkl 文件
│
└── 保存 labels_dict, val_subs_folds → pkl    # 全部 fold 生成完毕后，统一存储标签和验证划分

```


**batch_size、batch和epoch的关系**：
✅ 它们之间的关系：

Batch	一小批训练数据，执行一次更新	20 张图像组成一个 batch，训练完就更新一次权重
Epoch	完整训练集的一轮迭代（被分成多个 batch）	1000 张图像，batch_size = 20 → 50 个 batch 构成 1 epoch
关系	一个 epoch 包含多个 batch	epoch = N × batch（直到遍历所有数据）
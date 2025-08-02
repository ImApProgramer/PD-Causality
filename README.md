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

尝试运行:`python /root/.pycharm_helpers/pydev/pydevd.py --multiprocess --qt-support=auto --client localhost --port 43015 --file /root/MotionEncoders_parkinsonism_benchmark-main/eval_encoder.py --backbone poseformerv2 --medication 1`

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

Batch	一小批训练数据，执行一次更新	20 张图像组成一个 batch，训练完就更新一次权重

Epoch	完整训练集的一轮迭代（被分成多个 batch）	1000 张图像，batch_size = 20 → 50 个 batch 构成 1 epoch

关系	一个 epoch 包含多个 batch	epoch = N × batch（直到遍历所有数据）


------
出现问题：train_model()没有提供保存best检查点的逻辑。
不过由于save_checkpoint提供了，并且在train过程也有完备的指标可供参考，这里选择使用**macro F1-Score**来进行评估选择best，手动更新一下。



复现成功，log如下
```
Training (fold22):   0%|                              | 0/20 [00:00<?, ?epoch/s]Epoch 0 completed in 0.45s
Training (fold22):   0%| | 0/20 [00:00<?, ?epoch/s, train_accuracy=28.9, train_l[INFO] Best checkpoint saved at epoch 0 with val_f1_score=0.2313
Training (fold22):   5%| | 1/20 [00:00<00:13,  1.36epoch/s, train_accuracy=28.9,Epoch 1 completed in 0.41s
Training (fold22):   5%| | 1/20 [00:01<00:13,  1.36epoch/s, train_accuracy=30.4,[INFO] Best checkpoint saved at epoch 1 with val_f1_score=0.2557
Training (fold22):  10%| | 2/20 [00:01<00:14,  1.26epoch/s, train_accuracy=30.4,Epoch 2 completed in 0.57s
Training (fold22):  10%| | 2/20 [00:02<00:14,  1.26epoch/s, train_accuracy=30.1,[INFO] Best checkpoint saved at epoch 2 with val_f1_score=0.2633
Training (fold22):  15%|▏| 3/20 [00:02<00:15,  1.09epoch/s, train_accuracy=30.1,Epoch 3 completed in 0.57s
Training (fold22):  15%|▏| 3/20 [00:03<00:15,  1.09epoch/s, train_accuracy=29.7,[INFO] Best checkpoint saved at epoch 3 with val_f1_score=0.2664
Training (fold22):  20%|▏| 4/20 [00:03<00:15,  1.01epoch/s, train_accuracy=29.7,Epoch 4 completed in 0.55s
Training (fold22):  20%|▏| 4/20 [00:04<00:15,  1.01epoch/s, train_accuracy=30.8,[INFO] Best checkpoint saved at epoch 4 with val_f1_score=0.2798
Training (fold22):  25%|▎| 5/20 [00:04<00:15,  1.03s/epoch, train_accuracy=30.8,Epoch 5 completed in 0.56s
Training (fold22):  25%|▎| 5/20 [00:05<00:15,  1.03s/epoch, train_accuracy=30.7,[INFO] Best checkpoint saved at epoch 5 with val_f1_score=0.3155
Training (fold22):  30%|▎| 6/20 [00:05<00:14,  1.04s/epoch, train_accuracy=30.7,Epoch 6 completed in 0.57s
Training (fold22):  30%|▎| 6/20 [00:06<00:14,  1.04s/epoch, train_accuracy=30.2,[INFO] Best checkpoint saved at epoch 6 with val_f1_score=0.3408
Training (fold22):  35%|▎| 7/20 [00:06<00:13,  1.05s/epoch, train_accuracy=30.2,Epoch 7 completed in 0.56s
Training (fold22):  35%|▎| 7/20 [00:07<00:13,  1.05s/epoch, train_accuracy=32.1,[INFO] Best checkpoint saved at epoch 7 with val_f1_score=0.3512
Training (fold22):  40%|▍| 8/20 [00:08<00:12,  1.06s/epoch, train_accuracy=32.1,Epoch 8 completed in 0.57s
Training (fold22):  40%|▍| 8/20 [00:09<00:12,  1.06s/epoch, train_accuracy=34.1,[INFO] Best checkpoint saved at epoch 8 with val_f1_score=0.3564
Training (fold22):  45%|▍| 9/20 [00:09<00:12,  1.14s/epoch, train_accuracy=34.1,Epoch 9 completed in 0.58s
Training (fold22):  45%|▍| 9/20 [00:10<00:12,  1.14s/epoch, train_accuracy=32.7,[INFO] Best checkpoint saved at epoch 9 with val_f1_score=0.3627
Training (fold22):  50%|▌| 10/20 [00:10<00:11,  1.15s/epoch, train_accuracy=32.7Epoch 10 completed in 0.51s
Training (fold22):  50%|▌| 10/20 [00:11<00:11,  1.15s/epoch, train_accuracy=33.8[INFO] Best checkpoint saved at epoch 10 with val_f1_score=0.3732
Training (fold22):  55%|▌| 11/20 [00:11<00:09,  1.09s/epoch, train_accuracy=33.8Epoch 11 completed in 0.57s
Training (fold22):  55%|▌| 11/20 [00:12<00:09,  1.09s/epoch, train_accuracy=34.2[INFO] Best checkpoint saved at epoch 11 with val_f1_score=0.3754
Training (fold22):  60%|▌| 12/20 [00:12<00:08,  1.09s/epoch, train_accuracy=34.2Epoch 12 completed in 0.55s
Training (fold22):  60%|▌| 12/20 [00:13<00:08,  1.09s/epoch, train_accuracy=33.7[INFO] Best checkpoint saved at epoch 12 with val_f1_score=0.3843
Training (fold22):  65%|▋| 13/20 [00:13<00:07,  1.10s/epoch, train_accuracy=33.7Epoch 13 completed in 0.59s
Training (fold22):  65%|▋| 13/20 [00:14<00:07,  1.10s/epoch, train_accuracy=36.6[INFO] Best checkpoint saved at epoch 13 with val_f1_score=0.4017
Training (fold22):  70%|▋| 14/20 [00:14<00:06,  1.11s/epoch, train_accuracy=36.6Epoch 14 completed in 0.58s
Training (fold22):  70%|▋| 14/20 [00:16<00:06,  1.11s/epoch, train_accuracy=34.7[INFO] Best checkpoint saved at epoch 14 with val_f1_score=0.4043
Training (fold22):  75%|▊| 15/20 [00:16<00:05,  1.18s/epoch, train_accuracy=34.7Epoch 15 completed in 0.58s
Training (fold22):  75%|▊| 15/20 [00:17<00:05,  1.18s/epoch, train_accuracy=37.3[INFO] Best checkpoint saved at epoch 15 with val_f1_score=0.4062
Training (fold22):  80%|▊| 16/20 [00:17<00:04,  1.15s/epoch, train_accuracy=37.3Epoch 16 completed in 0.55s
Training (fold22):  80%|▊| 16/20 [00:18<00:04,  1.15s/epoch, train_accuracy=37.6[INFO] Best checkpoint saved at epoch 16 with val_f1_score=0.4099
Training (fold22):  85%|▊| 17/20 [00:18<00:03,  1.13s/epoch, train_accuracy=37.6Epoch 17 completed in 0.57s
Training (fold22):  85%|▊| 17/20 [00:19<00:03,  1.13s/epoch, train_accuracy=36.9[INFO] Best checkpoint saved at epoch 17 with val_f1_score=0.4100
Training (fold22):  90%|▉| 18/20 [00:19<00:02,  1.12s/epoch, train_accuracy=36.9Epoch 18 completed in 0.57s
Training (fold22):  90%|▉| 18/20 [00:20<00:02,  1.12s/epoch, train_accuracy=37.2[INFO] Best checkpoint saved at epoch 18 with val_f1_score=0.4138
Training (fold22):  95%|▉| 19/20 [00:20<00:01,  1.12s/epoch, train_accuracy=37.2Epoch 19 completed in 0.57s
Training (fold22):  95%|▉| 19/20 [00:21<00:01,  1.12s/epoch, train_accuracy=37.7[INFO] Best checkpoint saved at epoch 19 with val_f1_score=0.4140
Training (fold22): 100%|█| 20/20 [00:21<00:00,  1.08s/epoch, train_accuracy=37.7
[INFO] Latest checkpoint saved at: /czl_ssd/log/motion_encoder/out/poseformerv2_test/7/models
[INFO] (load_pretrained_weights) 141 layers are loaded
[INFO] (load_pretrained_weights) 0 layers are discared
100%|██████████████████████████████████████████| 69/69 [00:00<00:00, 117.21it/s]
fold # of test samples: 60
current sum # of test samples: 2316
[INFO] (load_pretrained_weights) 141 layers are loaded
[INFO] (load_pretrained_weights) 0 layers are discared
100%|██████████████████████████████████████████| 69/69 [00:00<00:00, 113.17it/s]
Fold 22 run time: 0:00:23.282845
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.45      0.31      0.37      1005
           1       0.41      0.42      0.42       828
           2       0.31      0.49      0.38       483

    accuracy                           0.39      2316
   macro avg       0.39      0.41      0.39      2316
weighted avg       0.41      0.39      0.39      2316

              precision    recall  f1-score   support

           0       0.54      0.41      0.46       654
           1       0.17      0.22      0.19       306
           2       0.33      0.43      0.38       249

    accuracy                           0.36      1209
   macro avg       0.35      0.35      0.34      1209
weighted avg       0.40      0.36      0.38      1209

              precision    recall  f1-score   support

           0       0.23      0.14      0.17       351
           1       0.61      0.54      0.57       522
           2       0.30      0.55      0.39       234

    accuracy                           0.42      1107
   macro avg       0.38      0.41      0.38      1107
weighted avg       0.42      0.42      0.41      1107

==========LAST REPORTS============
              precision    recall  f1-score   support

           0       0.48      0.30      0.37      1005
           1       0.44      0.50      0.47       828
           2       0.31      0.49      0.38       483

    accuracy                           0.41      2316
   macro avg       0.41      0.43      0.41      2316
weighted avg       0.43      0.41      0.41      2316

              precision    recall  f1-score   support

           0       0.60      0.39      0.47       654
           1       0.26      0.39      0.31       306
           2       0.33      0.43      0.37       249

    accuracy                           0.40      1209
   macro avg       0.40      0.40      0.39      1209
weighted avg       0.46      0.40      0.41      1209

              precision    recall  f1-score   support

           0       0.24      0.14      0.17       351
           1       0.61      0.56      0.58       522
           2       0.30      0.55      0.39       234

    accuracy                           0.42      1107
   macro avg       0.38      0.42      0.38      1107
weighted avg       0.43      0.42      0.41      1107

/opt/conda/envs/1128/lib/python3.8/site-packages/scipy/stats/_morestats.py:3414: UserWarning: Exact p-value calculation does not work if there are zeros. Switching to normal approximation.
  warnings.warn("Exact p-value calculation does not work if there are "
/opt/conda/envs/1128/lib/python3.8/site-packages/scipy/stats/_morestats.py:3428: UserWarning: Sample size too small for normal approximation.
  warnings.warn("Sample size too small for normal approximation.")
/opt/conda/envs/1128/lib/python3.8/site-packages/scipy/stats/_morestats.py:3414: UserWarning: Exact p-value calculation does not work if there are zeros. Switching to normal approximation.
  warnings.warn("Exact p-value calculation does not work if there are "
/opt/conda/envs/1128/lib/python3.8/site-packages/scipy/stats/_morestats.py:3428: UserWarning: Sample size too small for normal approximation.
  warnings.warn("Sample size too small for normal approximation.")
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:                          epoch ▄▃▄▅▆▆▇▇▅▆▅▅▇▂▇▄█▄█▃▂▅▆▇▃▇▁▄▅█▁▇▂▆▆▂▃█▁▅
wandb:        eval_acc/fold0_accuracy ▁▁▁▂▂▃▅▅▆▆▇▇▇▇▇█████
wandb:       eval_acc/fold10_accuracy ▁▁▁▁▁▁▂▂▂▂▃▃▄▅▆▇▇▇██
wandb:       eval_acc/fold11_accuracy ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       eval_acc/fold12_accuracy ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▃▃▄▅▅█
wandb:       eval_acc/fold13_accuracy ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       eval_acc/fold14_accuracy ▁███████████████████
wandb:       eval_acc/fold15_accuracy ▁▃▅▅▅▅▅▆▅▅▆▆▆▆▆▇▇▇██
wandb:       eval_acc/fold16_accuracy █▇▅▄▄▃▃▃▃▃▃▃▃▃▃▃▃▃▂▁
wandb:       eval_acc/fold17_accuracy ▁▁▂▂▂▃▄▄▅▄▅▆▇▇▇█▇▇▇█
wandb:       eval_acc/fold18_accuracy ▁▁▂▂▃▄▅▆▇▇▇███▇▇▇▆▆▇
wandb:       eval_acc/fold19_accuracy ▁▁▁▁▁▂▂▂▃▃▄▅▆▇███▇▇█
wandb:        eval_acc/fold1_accuracy ████████████▇▇▅▅▃▃▂▁
wandb:       eval_acc/fold20_accuracy ▁▁▁▁▁▁▁▃▄▅▆▆▇███████
wandb:       eval_acc/fold21_accuracy ██▇▆▅▅▄▄▃▃▂▂▁▁▁▁▁▁▂▂
wandb:       eval_acc/fold22_accuracy ▁▂▂▂▃▃▄▅▆▆▆▆▇███████
wandb:        eval_acc/fold2_accuracy ▁▁▂▂▂▂▂▂▃▃▄▅▆▇▇█████
wandb:        eval_acc/fold3_accuracy ████████████▇▇▆▂▂▂▁▁
wandb:        eval_acc/fold4_accuracy █████▇▆▆▆▆▆▆▆▆▆▅▄▄▂▁
wandb:        eval_acc/fold5_accuracy ██▇▇▇▇▇▇▆▄▃▂▁▁▁▁▁▁▁▁
wandb:        eval_acc/fold6_accuracy ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:        eval_acc/fold7_accuracy ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▃▄▄▆▇█
wandb:        eval_acc/fold8_accuracy ███████████▆▆▆▅▄▄▃▂▁
wandb:        eval_acc/fold9_accuracy ▁▂▂▂▃▄▄▄▄▄▅▄▆▆▅▆▆▇██
wandb:               eval_f1/fold0_f1 ▁▁▁▁▂▃▄▅▅▆▆▇▇▇██████
wandb:              eval_f1/fold10_f1 ▁▁▁▁▁▂▂▃▃▃▃▃▄▅▆▇▇▇██
wandb:              eval_f1/fold11_f1 ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:              eval_f1/fold12_f1 ▁▁▁▁▁▁▂▃▃▃▃▃▄▅▅▅▆▇▇█
wandb:              eval_f1/fold13_f1 ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:              eval_f1/fold14_f1 ▁███████████████████
wandb:              eval_f1/fold15_f1 ▁▂▃▃▃▃▃▃▃▄▄▄▄▅▅▅▆▆██
wandb:              eval_f1/fold16_f1 █▇▄▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁
wandb:              eval_f1/fold17_f1 ▁▁▁▁▃▅▆▆▅▄▄▆████▇▇▇▇
wandb:              eval_f1/fold18_f1 ▁▁▁▁▂▃▄▆▆▆▇▇████████
wandb:              eval_f1/fold19_f1 ▁▁▁▁▃▅▅▅▆▇▇▇▇▇▇▇█▇▆▆
wandb:               eval_f1/fold1_f1 ████████████▇▇▆▅▃▃▂▁
wandb:              eval_f1/fold20_f1 ▁▁▂▅▅▅▅▆████████████
wandb:              eval_f1/fold21_f1 ▇█▇▆▅▅▄▄▄▃▂▂▂▁▁▁▁▂▂▃
wandb:              eval_f1/fold22_f1 ▁▂▂▂▃▄▅▆▆▆▆▇▇███████
wandb:               eval_f1/fold2_f1 ▁▁▂▂▂▃▃▃▃▃▅▅▅▇██████
wandb:               eval_f1/fold3_f1 ████████████▇▄▃▂▂▂▂▁
wandb:               eval_f1/fold4_f1 █████▇▆▆▆▅▅▅▄▄▄▄▃▃▂▁
wandb:               eval_f1/fold5_f1 ▁▃▇█▆▅▄▅▇▆▅▄▃▃▃▃▃▃▃▄
wandb:               eval_f1/fold6_f1 ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:               eval_f1/fold7_f1 ▁▁▁▁▁▁▁▁▁▁▁▁▂▂▃▄▄▆▇█
wandb:               eval_f1/fold8_f1 ███████████▆▆▆▅▄▄▃▁▁
wandb:               eval_f1/fold9_f1 ▁▂▂▂▄▅▆▆▆▆▆▆▆▆▆▆▆███
wandb:           eval_loss/fold0_loss █▇▇▆▆▅▅▄▄▄▄▃▃▃▂▂▂▂▁▁
wandb:          eval_loss/fold10_loss █▇▇▆▆▅▅▄▄▄▃▃▃▃▂▂▂▂▁▁
wandb:          eval_loss/fold11_loss ▁▁▁▂▂▃▃▄▄▅▆▆▆▆▇██▇▇█
wandb:          eval_loss/fold12_loss █▇▇▆▆▅▅▄▄▄▄▃▃▃▂▂▂▂▁▁
wandb:          eval_loss/fold13_loss █▇▆▆▅▅▅▄▄▄▃▃▃▂▂▂▂▂▁▁
wandb:          eval_loss/fold14_loss ▁▃▂▃▃▄▅▅▅▅▅▆▆▆▇▇████
wandb:          eval_loss/fold15_loss █▇▇▆▆▅▅▅▄▄▄▃▃▃▂▂▂▁▁▁
wandb:          eval_loss/fold16_loss █▇▆▆▅▅▄▃▃▂▂▂▂▂▂▂▂▂▁▁
wandb:          eval_loss/fold17_loss █▇▇▆▆▆▅▅▄▄▄▃▃▃▃▂▂▂▁▁
wandb:          eval_loss/fold18_loss █▇▇▆▆▅▅▅▄▄▄▃▃▃▂▂▂▁▁▁
wandb:          eval_loss/fold19_loss █▇▇▆▆▅▅▄▄▃▃▃▃▃▂▂▂▁▁▁
wandb:           eval_loss/fold1_loss █▇▇▆▆▅▅▅▄▄▄▃▃▃▂▂▂▁▁▁
wandb:          eval_loss/fold20_loss █▇▅▆▅▅▄▃▂▂▂▃▂▁▁▁▁▃▃▃
wandb:          eval_loss/fold21_loss ▁▁▂▃▃▄▄▄▅▆▆▆▅▆▇▇▇▇██
wandb:          eval_loss/fold22_loss ▁▂▃▃▄▄▄▅▅▅▅▆▆▆▇▇▇███
wandb:           eval_loss/fold2_loss ▁▂▂▃▃▄▄▅▅▅▅▆▆▆▇▇▇███
wandb:           eval_loss/fold3_loss ▁▂▁▂▃▄▆▅▅▆▆█▇▇▆▇█▇▇█
wandb:           eval_loss/fold4_loss █▇▇▆▆▆▅▅▄▄▄▃▃▃▂▂▂▁▁▁
wandb:           eval_loss/fold5_loss █▇▇▆▆▅▅▅▄▄▃▃▃▃▂▂▂▁▁▁
wandb:           eval_loss/fold6_loss ▁▁▂▃▄▄▄▅▆▆▆▇▇▇▇▇█▇██
wandb:           eval_loss/fold7_loss █▇▇▇▆▅▅▄▄▄▄▃▃▂▂▂▂▂▁▁
wandb:           eval_loss/fold8_loss ██▆▆▆▆▆▅▄▄▄▃▃▃▃▂▂▂▁▁
wandb:           eval_loss/fold9_loss █▇▇▆▆▅▅▄▄▄▃▃▃▃▂▂▂▁▁▁
wandb:                 train/fold0_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold10_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold11_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold12_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold13_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold14_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold15_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold16_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold17_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold18_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold19_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold1_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold20_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold21_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                train/fold22_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold2_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold3_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold4_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold5_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold6_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold7_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold8_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:                 train/fold9_lr ██▇▇▆▆▆▅▅▅▄▄▃▃▃▂▂▂▁▁
wandb:  train_accuracy/fold0_accuracy ▂▁▁▂▂▃▄▄▄▅▄▄▆▅▇▅▆▆▆█
wandb: train_accuracy/fold10_accuracy ▁▂▃▃▄▂▅▃▄▅▄▅▅▅▆█▇▆▆█
wandb: train_accuracy/fold11_accuracy ▄▁▄▅▄▄▄▄▅▆▆▆▇▇▇▇▅▇▆█
wandb: train_accuracy/fold12_accuracy ▆▆█▆▁▅▄▅█▃▁▄▇▆▁▂▃▄▇▅
wandb: train_accuracy/fold13_accuracy ▁▃▃▄▄▃▄▆▅▆▆▆▆▆▆▇▇▆▇█
wandb: train_accuracy/fold14_accuracy ▁▅▅▇▃▄▄▄▃▃▆▆▅▆▆█▅▆▄▆
wandb: train_accuracy/fold15_accuracy ▁▃▄▅▃▅▄▄▅▇▆▆▅▆▆▇█▇▇▇
wandb: train_accuracy/fold16_accuracy ▄▅▅▆▂▆█▅▅▇▆█▅▅▅▄▃▃▁▂
wandb: train_accuracy/fold17_accuracy ▂▁▂▂▂▃▃▄▃▄▄▆▅▅▆▆▇▇▇█
wandb: train_accuracy/fold18_accuracy ▂▁▁▁▂▄▅▄▃▅▆▅▆▅▅▆█▅█▇
wandb: train_accuracy/fold19_accuracy ▄▃▆▃▄▇▁▅▅▄▅▂▂▆▂▅▅▅▇█
wandb:  train_accuracy/fold1_accuracy ▁▃▃▄▅▄▆▄▆▆▆▇▆▇▆▇▆▇█▆
wandb: train_accuracy/fold20_accuracy ▁▁▃▃▆▃▅▃▃▁▂▁█▇▇▃▄▃▆▅
wandb: train_accuracy/fold21_accuracy ▁▁▂▄▃▄▄▃▅▄▆▆▅▆▇▄█▇▆█
wandb: train_accuracy/fold22_accuracy ▁▂▂▂▃▂▂▄▅▄▅▅▅▇▆██▇██
wandb:  train_accuracy/fold2_accuracy ▃▂▁▃▄▄▃▃▃▅▅▅▄▆▆▆▆▆█▇
wandb:  train_accuracy/fold3_accuracy ▄▃▆▁█▄▅▃▇▄▆▆▃▃▃▇▆▄▅█
wandb:  train_accuracy/fold4_accuracy ▁▁▃▅▃▄▆▅▇█▅▂▃▇█▅▆▅█▅
wandb:  train_accuracy/fold5_accuracy ▁▂▃▄▄▄▅▅▅▆▅▇▆▇▇▇██▇▇
wandb:  train_accuracy/fold6_accuracy ▁▅▁▅▃▆▃▄▃▃▆▄▃▅▅▆▄█▄▆
wandb:  train_accuracy/fold7_accuracy ▂▃▃▂▄▆▃▂▄▆▃▄▃▄▃▄▃▅█▁
wandb:  train_accuracy/fold8_accuracy ▁▃▃▃▄▅▅▆▆▇▅▅▅▆▆█▇██▇
wandb:  train_accuracy/fold9_accuracy ▄▂▆▄▇▇▆▃▅▄▃█▄▄▁▃▄█▇▇
wandb:          train_loss/fold0_loss █▇▇▆▆▅▅▅▅▄▄▃▃▃▃▃▂▂▂▁
wandb:         train_loss/fold10_loss █▇▇▅▆▇▅▆▄▅▄▃▃▂▂▂▂▂▁▂
wandb:         train_loss/fold11_loss ▇██▆▆▄▅▄▄▃▂▃▂▃▂▂▂▂▁▁
wandb:         train_loss/fold12_loss █▇▇▆▆▅▅▄▄▄▅▄▃▂▃▂▂▁▁▁
wandb:         train_loss/fold13_loss █▇▇▄▅▅▃▄▄▂▂▃▃▄▃▂▃▁▂▂
wandb:         train_loss/fold14_loss █▆▇▅▅▅▄▄▃▄▄▃▃▂▂▂▂▂▁▁
wandb:         train_loss/fold15_loss █▆▆▅▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁
wandb:         train_loss/fold16_loss █▇▅▆▆▅▅▄▄▃▃▃▃▂▃▁▂▁▂▂
wandb:         train_loss/fold17_loss ▇██▇▇▄▅▄▄▄▃▂▂▂▄▃▁▂▂▁
wandb:         train_loss/fold18_loss ██▇▇▆▅▅▅▅▅▃▄▄▃▂▃▁▁▂▂
wandb:         train_loss/fold19_loss ██▆▅▆▅▅▄▄▄▄▄▂▄▃▃▃▃▁▁
wandb:          train_loss/fold1_loss █▆▆▆▆▅▅▆▅▄▄▃▄▃▄▃▃▂▁▃
wandb:         train_loss/fold20_loss █▇▆▆▅▄▅▄▄▃▄▄▂▁▁▁▃▁▁▁
wandb:         train_loss/fold21_loss ██▆▆▆▅▅▄▄▄▃▄▃▃▂▃▂▁▁▂
wandb:         train_loss/fold22_loss █▇▅▇▆▆▄▄▄▄▃▃▄▂▂▂▂▂▁▁
wandb:          train_loss/fold2_loss █▇▇▆▅▅▄▄▃▂▃▃▃▃▂▂▁▂▁▁
wandb:          train_loss/fold3_loss █▇▆▆▅▅▅▅▄▃▃▂▂▂▂▁▁▂▁▁
wandb:          train_loss/fold4_loss █▇▆▆▆▅▄▅▄▃▄▃▅▂▂▁▃▁▂▁
wandb:          train_loss/fold5_loss ██▆▇▆▅▄▅▄▄▃▃▃▂▂▃▁▁▃▃
wandb:          train_loss/fold6_loss █▇▇▆▇▄▄▆▆▅▄▄▃▃▁▂▃▁▃▃
wandb:          train_loss/fold7_loss █▅▇▆▇▄▆▆▆▃▂▄▄▄▃▅▃▃▁▄
wandb:          train_loss/fold8_loss █▇▆▆▅▅▅▄▅▄▄▄▄▃▃▃▂▂▁▂
wandb:          train_loss/fold9_loss █▇▇▆▅▆▅▄▅▄▃▄▃▃▃▃▃▃▂▁
wandb: 
wandb: Run summary:
wandb:                          epoch 19
wandb:        eval_acc/fold0_accuracy 52.70936
wandb:       eval_acc/fold10_accuracy 44.1989
wandb:       eval_acc/fold11_accuracy 21.97802
wandb:       eval_acc/fold12_accuracy 21.83406
wandb:       eval_acc/fold13_accuracy 26.36364
wandb:       eval_acc/fold14_accuracy 34.80176
wandb:       eval_acc/fold15_accuracy 34.24908
wandb:       eval_acc/fold16_accuracy 34.59716
wandb:       eval_acc/fold17_accuracy 43.04933
wandb:       eval_acc/fold18_accuracy 68.09117
wandb:       eval_acc/fold19_accuracy 38.42795
wandb:        eval_acc/fold1_accuracy 38.24561
wandb:       eval_acc/fold20_accuracy 39.19598
wandb:       eval_acc/fold21_accuracy 25.73099
wandb:       eval_acc/fold22_accuracy 52.25225
wandb:        eval_acc/fold2_accuracy 61.92661
wandb:        eval_acc/fold3_accuracy 32.48945
wandb:        eval_acc/fold4_accuracy 21.6885
wandb:        eval_acc/fold5_accuracy 7.48899
wandb:        eval_acc/fold6_accuracy 41.11675
wandb:        eval_acc/fold7_accuracy 24.08293
wandb:        eval_acc/fold8_accuracy 26.56514
wandb:        eval_acc/fold9_accuracy 41.66667
wandb:               eval_f1/fold0_f1 0.38692
wandb:              eval_f1/fold10_f1 0.35991
wandb:              eval_f1/fold11_f1 0.0792
wandb:              eval_f1/fold12_f1 0.20351
wandb:              eval_f1/fold13_f1 0.15941
wandb:              eval_f1/fold14_f1 0.2704
wandb:              eval_f1/fold15_f1 0.27469
wandb:              eval_f1/fold16_f1 0.2694
wandb:              eval_f1/fold17_f1 0.32528
wandb:              eval_f1/fold18_f1 0.65372
wandb:              eval_f1/fold19_f1 0.2743
wandb:               eval_f1/fold1_f1 0.27238
wandb:              eval_f1/fold20_f1 0.32336
wandb:              eval_f1/fold21_f1 0.18801
wandb:              eval_f1/fold22_f1 0.414
wandb:               eval_f1/fold2_f1 0.56982
wandb:               eval_f1/fold3_f1 0.33072
wandb:               eval_f1/fold4_f1 0.13059
wandb:               eval_f1/fold5_f1 0.07707
wandb:               eval_f1/fold6_f1 0.30727
wandb:               eval_f1/fold7_f1 0.22261
wandb:               eval_f1/fold8_f1 0.22337
wandb:               eval_f1/fold9_f1 0.27845
wandb:           eval_loss/fold0_loss 1.0949
wandb:          eval_loss/fold10_loss 1.10462
wandb:          eval_loss/fold11_loss 1.10472
wandb:          eval_loss/fold12_loss 1.09728
wandb:          eval_loss/fold13_loss 1.10857
wandb:          eval_loss/fold14_loss 1.09283
wandb:          eval_loss/fold15_loss 1.09414
wandb:          eval_loss/fold16_loss 1.09932
wandb:          eval_loss/fold17_loss 1.0996
wandb:          eval_loss/fold18_loss 1.0913
wandb:          eval_loss/fold19_loss 1.09807
wandb:           eval_loss/fold1_loss 1.097
wandb:          eval_loss/fold20_loss 1.09436
wandb:          eval_loss/fold21_loss 1.12336
wandb:          eval_loss/fold22_loss 1.09444
wandb:           eval_loss/fold2_loss 1.09075
wandb:           eval_loss/fold3_loss 1.0971
wandb:           eval_loss/fold4_loss 1.10905
wandb:           eval_loss/fold5_loss 1.10279
wandb:           eval_loss/fold6_loss 1.0862
wandb:           eval_loss/fold7_loss 1.09642
wandb:           eval_loss/fold8_loss 1.09244
wandb:           eval_loss/fold9_loss 1.1012
wandb:                 train/fold0_lr 8e-05
wandb:                train/fold10_lr 8e-05
wandb:                train/fold11_lr 8e-05
wandb:                train/fold12_lr 8e-05
wandb:                train/fold13_lr 8e-05
wandb:                train/fold14_lr 8e-05
wandb:                train/fold15_lr 8e-05
wandb:                train/fold16_lr 8e-05
wandb:                train/fold17_lr 8e-05
wandb:                train/fold18_lr 8e-05
wandb:                train/fold19_lr 8e-05
wandb:                 train/fold1_lr 8e-05
wandb:                train/fold20_lr 8e-05
wandb:                train/fold21_lr 8e-05
wandb:                train/fold22_lr 8e-05
wandb:                 train/fold2_lr 8e-05
wandb:                 train/fold3_lr 8e-05
wandb:                 train/fold4_lr 8e-05
wandb:                 train/fold5_lr 8e-05
wandb:                 train/fold6_lr 8e-05
wandb:                 train/fold7_lr 8e-05
wandb:                 train/fold8_lr 8e-05
wandb:                 train/fold9_lr 8e-05
wandb:  train_accuracy/fold0_accuracy 28.78018
wandb: train_accuracy/fold10_accuracy 46.58537
wandb: train_accuracy/fold11_accuracy 47.48915
wandb: train_accuracy/fold12_accuracy 32.18319
wandb: train_accuracy/fold13_accuracy 45.99448
wandb: train_accuracy/fold14_accuracy 21.91235
wandb: train_accuracy/fold15_accuracy 29.85258
wandb: train_accuracy/fold16_accuracy 35.87097
wandb: train_accuracy/fold17_accuracy 30.62629
wandb: train_accuracy/fold18_accuracy 49.52767
wandb: train_accuracy/fold19_accuracy 52.51204
wandb:  train_accuracy/fold1_accuracy 48.10533
wandb: train_accuracy/fold20_accuracy 50.60013
wandb: train_accuracy/fold21_accuracy 48.71026
wandb: train_accuracy/fold22_accuracy 37.73087
wandb:  train_accuracy/fold2_accuracy 43.54293
wandb:  train_accuracy/fold3_accuracy 39.57392
wandb:  train_accuracy/fold4_accuracy 22.71505
wandb:  train_accuracy/fold5_accuracy 39.87646
wandb:  train_accuracy/fold6_accuracy 23.70607
wandb:  train_accuracy/fold7_accuracy 34.35419
wandb:  train_accuracy/fold8_accuracy 46.0892
wandb:  train_accuracy/fold9_accuracy 44.45111
wandb:          train_loss/fold0_loss 1.0991
wandb:         train_loss/fold10_loss 1.0973
wandb:         train_loss/fold11_loss 1.08821
wandb:         train_loss/fold12_loss 1.09753
wandb:         train_loss/fold13_loss 1.09708
wandb:         train_loss/fold14_loss 1.09706
wandb:         train_loss/fold15_loss 1.09759
wandb:         train_loss/fold16_loss 1.10071
wandb:         train_loss/fold17_loss 1.10036
wandb:         train_loss/fold18_loss 1.09415
wandb:         train_loss/fold19_loss 1.0937
wandb:          train_loss/fold1_loss 1.09441
wandb:         train_loss/fold20_loss 1.09403
wandb:         train_loss/fold21_loss 1.08675
wandb:         train_loss/fold22_loss 1.09805
wandb:          train_loss/fold2_loss 1.09369
wandb:          train_loss/fold3_loss 1.09291
wandb:          train_loss/fold4_loss 1.09716
wandb:          train_loss/fold5_loss 1.0985
wandb:          train_loss/fold6_loss 1.09312
wandb:          train_loss/fold7_loss 1.09526
wandb:          train_loss/fold8_loss 1.09077
wandb:          train_loss/fold9_loss 1.09559
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /root/MotionEncoders_parkinsonism_benchmark-main/wandb/offline-run-20250801_100615-5qdnqvsu
wandb: Find logs at: ./wandb/offline-run-20250801_100615-5qdnqvsu/logs
```





------
打算看看把GCN接入进去，

本文里的PoseFormer是一个跟GCN拼凑起来的模型，
并且它是加载了一个PoseFormer的预训练数据`pre-trained_NTU_ckpt_epoch_199_enc_80_dec_20.pt`，来微调的，这个恐怕没法自己获取，官方提供的是Human3.6M，怎么办呢？要不要自己跑一遍？



到底**要不要纯GCN**？

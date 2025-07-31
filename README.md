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



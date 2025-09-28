# baseline





# 序回归、单病理建模

**先只建模病理因素$z_g$**，目标是把$z_g$进行任务导向，让$z_g$提取出来能更好分类。



---

用一个结构简单（可能建模能力不太足）的序回归头：

```python

class OrdinalHead(nn.Module):
    '''
    target = torch.zeros(y.size(0), K-1).to(y.device)
    for j in range(1, K):
        target[:, j-1] = (y >= j).float()
    loss = F.binary_cross_entropy_with_logits(logits, target)
    '''
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, num_classes - 1)  # K-1 logits

    def forward(self, x):
        feats = self.fc(x)
        logits = self.out(feats)  # [B, K-1]
        return logits

```



```python

class CounterfactualCausalModeling(nn.Module):
    """
    Stage1: base model
    """

    def __init__(self, backbone,input_dim=512, hidden_dim=256, z_dim=128):
        super(CounterfactualCausalModeling, self).__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.z_dim = z_dim

        # === Disease-related and Confounding Encoders === #
        self.disease_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=z_dim,
            num_layers=3,
            dropout=0.1
        )
        
        
	 def forward(self, inputs, labels=None):
        # === backbone features ===
        features = self.backbone(inputs)  # [B, T, J, C]

        # === encoded features ===
        # 病理特征和混淆特征都从同一个backbone出来
        z_g = self.disease_encoder(features)  # [B, z_dim]

        # === ordinal prediction ===
        z_g = z_g.mean(dim=(1, 2))
        logits = self.regressor(z_g)  # [B, K-1]

        out = {
            "logits": logits,
            "disease_features": z_g
        }
        
    
```



损失计算：

```python

def coral_loss(logits, labels, num_classes):
    """
    logits: [B, K-1]
    labels: [B] int64
    """
    # 构造 CORAL target
    target = torch.zeros(labels.size(0), num_classes - 1, device=labels.device)
    for j in range(1, num_classes):
        target[:, j-1] = (labels >= j).float()

    return F.binary_cross_entropy_with_logits(logits, target)




x, y = x.to(device), y.to(device).long()

outputs = model(x)                        # forward
logits = outputs["logits"]                # [B, K-1]

loss = coral_loss(logits, y, num_classes)


```



结果：

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.60      0.65      0.63      1026
           1       0.45      0.50      0.48       828
           2       0.21      0.12      0.16       486

    accuracy                           0.49      2340
   macro avg       0.42      0.43      0.42      2340
weighted avg       0.46      0.49      0.47      2340
```



```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.67      0.72      0.69      1026
           1       0.51      0.44      0.47       828
           2       0.26      0.28      0.27       486

    accuracy                           0.53      2340
   macro avg       0.48      0.48      0.48      2340
weighted avg       0.53      0.53      0.53      2340
```









# 序回归结构增强

同上，只更改了序回归头的结构，使其表达能力和原本的ClassifierHead对齐：

```python

class OrdinalHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.2):
        super().__init__()
        dims = [input_dim, hidden_dims, num_classes]
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.fcs.append(nn.Linear(dims[i], dims[i+1]))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
        self.out = nn.Linear(dims[-1], num_classes - 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        for fc, bn in zip(self.fcs, self.bns):
            x = self.act(bn(fc(x)))
            x = self.dropout(x)
        logits = self.out(x)
        return logits

```



结果：差了非常多

```
           0       0.51      0.41      0.46      1026
           1       0.32      0.46      0.38       828
           2       0.25      0.16      0.20       486

    accuracy                           0.38      2340
   macro avg       0.36      0.34      0.34      2340
weighted avg       0.39      0.38      0.37      2340
```





GCN(--seed 40 )

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.71      0.55      0.62      1005
           1       0.42      0.43      0.43       828
           2       0.26      0.37      0.30       483

    accuracy                           0.47      2316
   macro avg       0.46      0.45      0.45      2316
weighted avg       0.51      0.47      0.49      2316
```

-

```
=========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.45      0.35      0.39      1005
           1       0.43      0.50      0.46       828
           2       0.22      0.27      0.24       483

    accuracy                           0.38      2316
   macro avg       0.37      0.37      0.36      2316
weighted avg       0.39      0.38      0.38      2316

```





# 启用GRL+Reconstruction+干预+多阶段训练策略

## 说明



### GRL

```python
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)     # 前向传播不变
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None      # 反向传播时，接收到来自预测器的梯度，然后取反并且乘上系数lambd

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

-----------------------------------------------------

self.confound_encoder = MLPEncoder(         # 需要加入正交损失或者对抗约束
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=z_dim,
    num_layers=3,
    dropout=0.1
)

z_c = self.confound_encoder(features) 

# === GRL ===
# 让z_c无法预测病理标签
z_c_flat = z_c.mean(dim=(1, 2))     # 进行时间和关节维度上的池化
rev_zc=grad_reverse(z_c_flat,lambd=1.0)
confound_logits=self.regressor(rev_zc)		#也就是说和z_g分支用的是同一个回归头
-----------------------------------------------------

loss = coral_loss(logits, y, num_classes) + F.cross_entropy(outputs["confound_logits"], y)
```



### reconstruction+干预

```python

self.decoder = nn.Sequential(
    nn.Linear(z_dim * 2 , hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, input_dim)  # 重构回 backbone 的 feature dim
)

# === ReCon ===
    # 重构损失，避免退化
    recon_in = torch.cat([z_g,z_c], dim = -1)       #在C维度上进行拼接
    recon_features = self.decoder(recon_in)     #进行decoder解码

        
 # === counterfactual ===
    # 让模型知道标签必须随着z_g变化，而对z_c保持不变
    # ---- 构造交换样本 ----
    counterfactual_logits = None
    shuffle_idx=None
    if labels is not None:
        B = z_g_pooled.shape[0]
        shuffle_idx = torch.randperm(B).to(labels.device)	#这里用的是随机替换

        # 将交换后的疾病特征输入到回归头进行预测
        z_g_swapped = z_g_pooled[shuffle_idx]#<--GRL进行解耦
        counterfactual_logits = self.regressor(z_g_swapped)#结果应该与给出病理特征的个体的标签对应

```



第二阶段：

```python

# 计算所有损失项
confound_loss = lambd1 * coral_loss(outputs["confound_logits"], y, num_classes)
recon_loss = lambd2 * F.mse_loss(
    outputs["recon_features"].mean(dim=(1, 2)),
    outputs["original_features"].mean(dim=(1, 2))
)

counterfactual_loss = 0
        if outputs["counterfactual_logits"] is not None:
            shuffle_idx = outputs["shuffle_idx"]
            y_swapped = y[shuffle_idx]
            counterfactual_loss = lambd3 * coral_loss(outputs["counterfactual_logits"], y_swapped, num_classes)
```



## 如何验证？

* adversary loss 曲线应该震荡在一个比较高的位置



## 结果（全部安排上）

          0       0.58      0.61      0.60      1026
           1       0.54      0.53      0.53       828
           2       0.37      0.34      0.36       486
    
    accuracy                           0.53      2340
       macro avg       0.50      0.49      0.50      2340
    weighted avg       0.52      0.53      0.52      2340


```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.65      0.73      0.69      1026
           1       0.54      0.44      0.49       828
           2       0.28      0.30      0.29       486

    accuracy                           0.54      2340
   macro avg       0.49      0.49      0.49      2340
weighted avg       0.54      0.54      0.54      2340
```

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.60      0.70      0.65      1026
           1       0.48      0.37      0.41       828
           2       0.36      0.39      0.37       486

    accuracy                           0.52      2340
   macro avg       0.48      0.48      0.48      2340
weighted avg       0.51      0.52      0.51      2340
```



# 0830：准备做的改动考虑

先权衡**为啥要做，有针对性**，然后考虑**是否会涨点**，再考虑**怎么做**的问题，以及**怎么实验看结果**的问题，减少黑箱，不要只是一味看总结果是否涨点，而且一次做最小的配套改动就好。



## 正交损失/HSIC？⇌ GRL

* 是否可以单独用正交损失？

  



## 加入metadata进行显式预测⇌ GRL让z_c无法预测病理标签

* GRL：实质上是*拔河失败*，没有明确的学习目标。
* 显式预测指标：更明确、可建模，逼迫分工



seed 1：（`loss = coral_loss(logits, y, num_classes) + total_confound_loss + recon_loss`）

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.60      0.52      0.56      1026
           1       0.45      0.46      0.45       828
           2       0.25      0.30      0.27       486

    accuracy                           0.45      2340
   macro avg       0.43      0.43      0.43      2340
weighted avg       0.47      0.45      0.46      2340
```



seed 2:

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.59      0.52      0.55      1026
           1       0.41      0.42      0.42       828
           2       0.33      0.41      0.37       486

    accuracy                           0.46      2340
   macro avg       0.45      0.45      0.45      2340
weighted avg       0.47      0.46      0.47      2340
```



加入反事实逻辑：seed 2

```
# === ReCon ===
        recon_in = torch.cat([z_g,z_c], dim = -1)       #在C维度上进行拼接
        recon_features = self.decoder(recon_in)     #进行decoder解码

        counterfactual_logits=None
        shuffle_idx=None
        if labels is not None:
            # 4. 反事实干预，进行特征交换，通过一致性损失进行约束
            shuffle_idx = get_different_label_shuffled_idx(labels)

            # 交换 z_g，保持 z_c 不变
            z_g_shuffled = z_g_pooled[shuffle_idx]

            # 用交换后的 z_g 进行预测
            counterfactual_logits = self.regressor(z_g_shuffled)
```



```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.59      0.52      0.55      1026
           1       0.40      0.41      0.40       828
           2       0.23      0.27      0.25       486

    accuracy                           0.43      2340
   macro avg       0.40      0.40      0.40      2340
weighted avg       0.44      0.43      0.44      2340
```



seed 1

```
==========BEST REPORTS============
              precision    recall  f1-score   support

           0       0.53      0.65      0.58      1026
           1       0.37      0.33      0.35       828
           2       0.24      0.16      0.19       486

    accuracy                           0.44      2340
   macro avg       0.38      0.38      0.37      2340
weighted avg       0.41      0.44      0.42      2340
```







## 动态学习率&lambd > 固定

* 👉Uncertainty weighting，每个损失配一个可学习的权重





## 分阶段决定开loss⇌一开始就全开，权重变化⇌交替小阶段

* 一开始就全开的方案：*sigmoid ramp*
* 交替小阶段：主任务轻解耦+解耦 轻主任务交替



### 拉高类别2正确率的解决方案： **class-2 的 F1** 可以作为早停信号

直接把优化目标指向关心的点，但是要注意是否会导致另外两个类别下降。



## 随机交换z_g，确保标签同给出者⇌随机交换z_c，确保标签与原来一致

* 这二者真的有优劣之分？

* 但是在代码实现容易/困难程度上，这样写无疑要更简单

* 相应地换用不变性约束：*为啥要停止z_g的梯度*？

  

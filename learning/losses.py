import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalClassBalancedMemory(nn.Module):
    """
    类别均衡的显式记忆库 (Class-Balanced Explicit Memory Bank)
    维护 0, 1, 2 每个类别固定数量的历史特征，用于挖掘负样本。
    """

    def __init__(self, num_classes=3, feat_dim=128, memory_per_class=256, device='cuda'):
        super(OrdinalClassBalancedMemory, self).__init__()
        self.num_classes = num_classes
        self.memory_per_class = memory_per_class
        self.feat_dim = feat_dim
        self.device = device

        # 为每个类别初始化一个队列
        for c in range(num_classes):
            # 初始化为归一化的随机向量
            init_feats = F.normalize(torch.randn(memory_per_class, feat_dim), dim=1)
            # register_buffer 会将 tensor 注册为模块状态，随模型保存/移动，但不更新梯度
            self.register_buffer(f'mem_feats_{c}', init_feats)
            self.register_buffer(f'ptr_{c}', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update(self, features, labels):
        """
        更新记忆库：将当前 batch 的特征写入对应类别的队列
        features: [B, D] (已归一化)
        labels: [B]
        """
        batch_size = features.size(0)
        features = features.detach()  # 断开梯度

        for c in range(self.num_classes):
            # 选出当前 batch 中属于类别 c 的样本
            mask = (labels == c)
            if mask.sum() == 0: continue

            feats_c = features[mask]
            num_to_add = feats_c.size(0)

            # 获取对应类别的 buffer 和 指针
            ptr = getattr(self, f'ptr_{c}')
            mem = getattr(self, f'mem_feats_{c}')

            # 循环写入 (Circular Buffer)
            current_ptr = ptr.item()
            if current_ptr + num_to_add <= self.memory_per_class:
                mem[current_ptr: current_ptr + num_to_add] = feats_c
                ptr[0] = (current_ptr + num_to_add) % self.memory_per_class
            else:
                # 如果超过尾部，分两段写入
                first_part = self.memory_per_class - current_ptr
                mem[current_ptr:] = feats_c[:first_part]
                second_part = num_to_add - first_part
                mem[:second_part] = feats_c[first_part:]
                ptr[0] = second_part

    def get_memory(self):
        """取出所有存储的特征和对应的标签"""
        feats_list = []
        labels_list = []
        for c in range(self.num_classes):
            feats = getattr(self, f'mem_feats_{c}')
            labels = torch.full((self.memory_per_class,), c, dtype=torch.long, device=self.device)
            feats_list.append(feats)
            labels_list.append(labels)

        return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)


class MemoryCausalOrdinalLoss(nn.Module):
    """
    基于记忆库的因果序数度量损失 (GaitC3I Logic Adapted for Ordinal Regression)
    逻辑:
    1. MCD: 计算同类样本中最远的距离，作为该类别的"协变量容忍阈值"。
    2. PGNS: 如果一个负样本比 MCD 还近，说明它是混淆样本 (Visual Spurious Correlation)，必须推开。
    3. Ordinal Margin: 0与2混淆的惩罚 > 0与1混淆的惩罚。
    """

    def __init__(self, margin_base=0.1, alpha=0.1, topk=5, memory_bank=None):
        super(MemoryCausalOrdinalLoss, self).__init__()
        self.margin_base = margin_base  # 基础边界
        self.alpha = alpha  # 序数惩罚系数
        self.topk = topk  # PGNS 挖掘数量
        self.memory_bank = memory_bank  # 引用 Memory Bank 实例

    def forward(self, batch_feats, batch_labels, epoch=0):
        # 1. 获取记忆库全量数据
        mem_feats, mem_labels = self.memory_bank.get_memory()

        # 2. 计算相似度矩阵: Batch (Anchor) vs Memory (Candidates)
        # [B, M_total]
        sim_mat = torch.matmul(batch_feats, mem_feats.T)

        B = batch_feats.size(0)

        # 3. 标签差异矩阵 [B, M_total]
        # label_diff_mat[i, j] = |y_i - y_j|
        label_diff_mat = torch.abs(batch_labels.unsqueeze(1) - mem_labels.unsqueeze(0)).float()

        mask_pos = (label_diff_mat == 0)  # 同类
        mask_neg = (label_diff_mat > 0)  # 异类

        # 4. 计算 MCD (Maximum Covariate Distance)
        # 对于每个 Anchor，找到 Memory 中最不相似的那个正样本
        # 填充非正样本为 1.0 (最大相似度)，防止 min 选中它们
        sim_pos_safe = torch.where(mask_pos, sim_mat, torch.tensor(1.0).to(sim_mat.device))
        min_pos_sim, _ = sim_pos_safe.min(dim=1)  # [B]

        # 阈值：比最远正样本还要松一点点
        mcd_threshold = min_pos_sim - self.margin_base

        # 5. PGNS 筛选策略 (Positive-Guided Negative Selection)
        if epoch >= 5:
            # 必须是负样本 AND 相似度竟然比正样本还高 (sim > threshold)
            confusing_neg_mask = mask_neg & (sim_mat > mcd_threshold.unsqueeze(1))
        else:
            # 前期尚未形成稳定的正样本聚类，挖掘所有负样本
            confusing_neg_mask = mask_neg

        loss = torch.tensor(0.0).to(sim_mat.device)
        valid_triplets = 0

        # 6. 计算 Triplet Loss
        # 优化目标: sim(A, P_hardest) > sim(A, N_confusing) + margin
        for i in range(B):
            # 确保 Memory 里有该类的正样本 (防止初始化阶段 crash)
            if not mask_pos[i].any(): continue

            s_ap = min_pos_sim[i]  # Hardest Positive

            # 取出该 Anchor 对应的混淆负样本索引
            neg_indices = confusing_neg_mask[i].nonzero(as_tuple=True)[0]

            if len(neg_indices) == 0: continue

            # Top-K Mining: 只选最难的前 K 个
            s_an_candidates = sim_mat[i, neg_indices]
            if len(neg_indices) > self.topk:
                values, indices = torch.topk(s_an_candidates, k=self.topk)
                final_neg_indices = neg_indices[indices]
            else:
                final_neg_indices = neg_indices

            # 累加 Loss
            for neg_idx in final_neg_indices:
                s_an = sim_mat[i, neg_idx]

                # 动态 Margin: 差异越大，推得越远
                # diff=1 -> margin = base + 0.1
                # diff=2 -> margin = base + 0.2
                diff = label_diff_mat[i, neg_idx]
                dynamic_margin = self.margin_base + self.alpha * diff

                # Relu(Neg - Pos + Margin)
                curr_loss = F.relu(s_an - s_ap + dynamic_margin)

                loss += curr_loss
                valid_triplets += 1

        # 7. 更新 Memory Bank (使用当前 Batch)
        self.memory_bank.update(batch_feats, batch_labels)

        if valid_triplets > 0:
            return loss / valid_triplets
        else:
            return loss
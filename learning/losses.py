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
        B = batch_feats.size(0)
        device = batch_feats.device

        # 1. 获取记忆库全量数据
        mem_feats, mem_labels = self.memory_bank.get_memory()

        # 拼接: [B + M, D]
        all_feats = torch.cat([batch_feats, mem_feats.detach()], dim=0)
        all_labels = torch.cat([batch_labels, mem_labels], dim=0)

        # 计算相似度矩阵: Batch (Anchor) vs Memory (Candidates)
        sim_mat = torch.matmul(batch_feats, mem_feats.T)

        # label_diff_mat[i, j] = |y_i - y_j|
        label_diff_mat = torch.abs(batch_labels.unsqueeze(1) - mem_labels.unsqueeze(0)).float()
        mask_pos = (label_diff_mat == 0)  # 同类
        mask_neg = (label_diff_mat > 0)  # 异类

        # 排除 Batch 内部的对角线 (自己不和自己比)
        identity_mask = torch.zeros_like(mask_pos).bool()
        identity_mask[:, :B] = torch.eye(B, device=device).bool()
        mask_pos = mask_pos & (~identity_mask)

        # =================================================================
        # 正向挖掘 (GaitC3I 逻辑)
        # =================================================================

        # 1. 定义正向阈值 (MCD): 同类中最不相似的距离
        sim_pos_safe = torch.where(mask_pos, sim_mat, torch.tensor(1.0).to(sim_mat.device))
        min_pos_sim, _ = sim_pos_safe.min(dim=1)  # [B]

        # 阈值：比最远正样本还要松一点点
        fwd_threshold = min_pos_sim - self.margin_base

        if epoch >= 5:
            mask_neg_mining = mask_neg & (sim_mat > fwd_threshold.unsqueeze(1))
        else:
            mask_neg_mining = mask_neg

        # =================================================================
        # 反向挖掘
        # =================================================================
        # 1. 定义反向阈值: 异类中最相似的距离
        sim_neg_safe = torch.where(mask_neg, sim_mat, torch.tensor(-1.0).to(device))
        max_neg_sim, _ = sim_neg_safe.max(dim=1)  # [B]

        # 阈值 = 最近 + Margin (安全区)
        rev_threshold = max_neg_sim + self.margin_base

        # 2. 筛选混淆正样本 (Confusing Positives)
        # 条件：是正样本 AND 距离竟然比反向阈值还远
        if epoch >= 5:
            mask_pos_mining = mask_pos & (sim_mat < rev_threshold.unsqueeze(1))
        else:
            mask_pos_mining = mask_pos

        # =================================================================
        #  计算总 Loss (Sum of Both Strategies)
        # =================================================================

        loss = torch.tensor(0.0).to(sim_mat.device)
        valid_triplets = 0

        # 6. 计算 Triplet Loss
        for i in range(B):
            # 确保 Memory 里有该类的正样本 (防止初始化阶段 crash)
            if not mask_pos[i].any() or not mask_neg[i].any():
                continue


            # --- Part A: 正向 Loss (优化混淆负样本) ---
            # 锚点: i, 正样本: 最难的正样本 (min_pos_sim[i]), 负样本: 挖掘出的 mask_neg_mining
            neg_indices = mask_neg_mining[i].nonzero(as_tuple=True)[0]
            if len(neg_indices) > 0:
                # Top-K Mining
                s_an_candidates = sim_mat[i, neg_indices]
                if len(neg_indices) > self.topk:
                    _, idxs = torch.topk(s_an_candidates, k=self.topk)
                    sel_neg_idx = neg_indices[idxs]
                else:
                    sel_neg_idx = neg_indices

                s_ap = min_pos_sim[i]  # Hardest Positive
                for neg_idx in sel_neg_idx:
                    s_an = sim_mat[i, neg_idx]
                    diff = label_diff_mat[i, neg_idx]
                    margin = self.margin_base + self.alpha * diff
                    # Loss = max(0, S_an - S_ap + m)
                    loss += F.relu(s_an - s_ap + margin)        #三元组损失 InfoNCE
                    valid_triplets += 1

            # --- Part B: 反向 Loss (优化混淆正样本) ---
            # 锚点: i, 负样本: 最难的负样本 (max_neg_sim[i]), 正样本: 挖掘出的 mask_pos_mining
            # 注意：这里的逻辑是拉近正样本，所以正样本是变量
            pos_indices = mask_pos_mining[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) > 0:
                # Top-K Mining (找最不相似的正样本，即最小的)
                s_ap_candidates = sim_mat[i, pos_indices]
                if len(pos_indices) > self.topk:
                    # largest=False 取最小
                    _, idxs = torch.topk(s_ap_candidates, k=self.topk, largest=False)
                    sel_pos_idx = pos_indices[idxs]
                else:
                    sel_pos_idx = pos_indices

                s_an = max_neg_sim[i]  # Hardest Negative
                # 这里为了简单，Margin 统一使用基础值，或者使用一个固定的大值
                # 因为我们要把正样本拉得比最近的负样本还要近
                margin = self.margin_base

                for pos_idx in sel_pos_idx:
                    s_ap_curr = sim_mat[i, pos_idx]
                    # Loss = max(0, S_an - S_ap_curr + m)
                    # 这一次，S_an 是固定的(相对而言)，我们在优化 S_ap_curr 让它变大
                    loss += F.relu(s_an - s_ap_curr + margin)
                    valid_triplets += 1

        # 7. 更新 Memory Bank (使用当前 Batch)
        self.memory_bank.update(batch_feats, batch_labels)

        if valid_triplets > 0:
            return loss / valid_triplets
        else:
            return loss
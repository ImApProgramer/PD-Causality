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
    [修改] 将原本的 Causal Ordinal Loss 修改为 Rank-N-Contrast (RNC) Loss
    """

    def __init__(self, margin_base=0.1, alpha=0.1, topk=5, temperature=2.0,memory_bank=None):
        super(MemoryCausalOrdinalLoss, self).__init__()
        # self.margin_base = margin_base  # 基础边界
        # self.alpha = alpha  # 序数惩罚系数
        # self.topk = topk  # PGNS 挖掘数量
        self.memory_bank = memory_bank  # 引用 Memory Bank 实例
        self.temperature = temperature #RNC推荐temperature=2.0

    def forward(self, batch_feats, batch_labels, epoch=0):
        # 1. 获取记忆库全量数据
        mem_feats, mem_labels = self.memory_bank.get_memory()

        # 原代码：使用点积相似度
        #sim_mat = torch.matmul(batch_feats, mem_feats.T)
        # 新代码:
        dist_mat = torch.cdist(batch_feats, mem_feats, p=2)  # 计算欧式距离
        sim_mat = -dist_mat  # 转为相似度（越小越相似，取负后越大越相似）

        B = batch_feats.size(0)

        # 3. 标签差异矩阵 [B, M_total] ，这个留着
        # label_diff_mat[i, j] = |y_i - y_j|
        label_diff_mat = torch.abs(batch_labels.unsqueeze(1) - mem_labels.unsqueeze(0)).float()

        # [删除] RNC 不需要显式的正负样本 Mask，因为它是基于排名的
        # mask_pos = (label_diff_mat == 0)  # 同类
        # mask_neg = (label_diff_mat > 0)  # 异类

        # 4. 计算 MCD (Maximum Covariate Distance)
        # 对于每个 Anchor，找到 Memory 中最不相似的那个正样本
        # 填充非正样本为 1.0 (最大相似度)，防止 min 选中它们
        # sim_pos_safe = torch.where(mask_pos, sim_mat, torch.tensor(1.0).to(sim_mat.device))
        # min_pos_sim, _ = sim_pos_safe.min(dim=1)  # [B]
        #
        # # 阈值：比最远正样本还要松一点点
        # mcd_threshold = min_pos_sim - self.margin_base
        #
        # # 5. PGNS 筛选策略 (Positive-Guided Negative Selection)
        # if epoch >= 5:
        #     # 必须是负样本 AND 相似度竟然比正样本还高 (sim > threshold)
        #     confusing_neg_mask = mask_neg & (sim_mat > mcd_threshold.unsqueeze(1))
        # else:
        #     # 前期尚未形成稳定的正样本聚类，挖掘所有负样本
        #     confusing_neg_mask = mask_neg
        #
        # loss = torch.tensor(0.0).to(sim_mat.device)
        # valid_triplets = 0

        total_loss = torch.tensor(0.0).to(batch_feats.device)

        # 6. 计算 Triplet Loss
        # 优化目标: sim(A, P_hardest) > sim(A, N_confusing) + margin


        for i in range(B):
            # # 确保 Memory 里有该类的正样本 (防止初始化阶段 crash)
            # if not mask_pos[i].any(): continue
            #
            # s_ap = min_pos_sim[i]  # Hardest Positive
            #
            # # 取出该 Anchor 对应的混淆负样本索引
            # neg_indices = confusing_neg_mask[i].nonzero(as_tuple=True)[0]
            #
            # if len(neg_indices) == 0: continue
            #
            # # Top-K Mining: 只选最难的前 K 个
            # s_an_candidates = sim_mat[i, neg_indices]
            # if len(neg_indices) > self.topk:
            #     values, indices = torch.topk(s_an_candidates, k=self.topk)
            #     final_neg_indices = neg_indices[indices]
            # else:
            #     final_neg_indices = neg_indices
            #
            # # 累加 Loss
            # for neg_idx in final_neg_indices:
            #     s_an = sim_mat[i, neg_idx]
            #
            #     # 动态 Margin: 差异越大，推得越远
            #     # diff=1 -> margin = base + 0.1
            #     # diff=2 -> margin = base + 0.2
            #     diff = label_diff_mat[i, neg_idx]
            #     dynamic_margin = self.margin_base + self.alpha * diff
            #
            #     # Relu(Neg - Pos + Margin)
            #     curr_loss = F.relu(s_an - s_ap + dynamic_margin)
            #
            #     loss += curr_loss
            #     valid_triplets += 1

            # 获取当前 Anchor 到所有 Memory 样本的信息
            sims_i = sim_mat[i]  # [M]
            diffs_i = label_diff_mat[i]  # [M]

            # 1. 对 Memory 中的样本按标签距离从小到大排序
            # 距离越近的样本，应该排在越前面 (相似度应该越高)
            sorted_diffs, sorted_indices = torch.sort(diffs_i)
            sorted_sims = sims_i[sorted_indices]

            # 2. 缩放 Logits
            scaled_logits = sorted_sims / self.temperature

            # 3. 计算分母 (动态竞争集合)
            # RNC逻辑: 对于排名第 j 的样本，它的分母是所有 k >= j 的样本
            # 使用后缀和 (Suffix Sum) 高效计算
            exp_logits = torch.exp(scaled_logits)
            # 从后往前累加: sum(exp[j:])
            denominator_sums = torch.flip(torch.cumsum(torch.flip(exp_logits, dims=[0]), dim=0), dims=[0])

            # 避免数值不稳定性
            denominator_sums = torch.clamp(denominator_sums, min=1e-8)

            # 4. 计算 Log-Likelihood
            # loss = - log( exp(s_ij) / sum_{k >= j} exp(s_ik) )
            #      = - ( s_ij - log(sum) )
            log_likelihoods = scaled_logits - torch.log(denominator_sums)

            # 对所有样本取平均
            loss_i = -log_likelihoods.mean()
            total_loss += loss_i

        # 7. 更新 Memory Bank (使用当前 Batch)
        self.memory_bank.update(batch_feats, batch_labels)

        return total_loss / B



class MemoryCLOCLoss(nn.Module):
    """
    [创新核心] Memory-Augmented Contrastive Learning for Ordinal Classification
    将 CLOC 的多边界累加机制 (Multi-Margin N-pair) 与 Class-Balanced Memory Bank 结合。
    """

    def __init__(self, n_classes=3, device='cuda', learnable_map=None, memory_bank=None):
        super().__init__()
        self.n_distances = n_classes - 1
        self.device = device
        self.memory_bank = memory_bank

        # --- 1. 创建可学习边界 (Learnable Margins) ---
        if learnable_map == None:
            learnable_map = [['learnable', None] for _ in range(self.n_distances)]

        self.distances_ori = torch.zeros(self.n_distances, device=self.device, dtype=torch.float64)
        learnable_indices = []

        for i, (isFixed, value) in enumerate(learnable_map):
            if isFixed == 'learnable':
                self.distances_ori[i] = self.__inverse_softplus(
                    0.5 + torch.rand(1) * 0.5) if value is None else self.__inverse_softplus(torch.tensor([value]))
                learnable_indices.append(i)
            elif isFixed == 'fixed':
                self.distances_ori[i] = self.__inverse_softplus(torch.tensor([value]))

        if len(learnable_indices) > 0:
            learnable_indices = torch.tensor(learnable_indices, device=self.device)
            self.learnables = nn.Parameter(self.distances_ori[learnable_indices])
            self.mask_learnables = torch.zeros_like(self.distances_ori, dtype=torch.bool)
            self.mask_learnables[learnable_indices] = True

    def __inverse_softplus(self, t):
        return torch.where(t > 20, t, torch.log(torch.exp(t) - 1))

    def forward(self, batch_feats, batch_labels):
        # --- 2. 获取记忆库中均衡的全量数据 ---
        mem_feats, mem_labels = self.memory_bank.get_memory()

        # 更新最新参数
        distances = self.distances_ori.clone()
        if hasattr(self, 'mask_learnables'):
            distances[self.mask_learnables] = self.learnables

        N = batch_feats.size(0)
        M = mem_feats.size(0)

        # --- 3. 计算 Batch 与 Memory 的相似度与标签差异 ---
        cos_sim = F.cosine_similarity(batch_feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)  # [N, M]
        label_diff = torch.abs(batch_labels.unsqueeze(1) - mem_labels.unsqueeze(0)).float()

        positives = label_diff <= 0
        negatives = ~positives

        pos_cossim, neg_cossim = cos_sim.clone(), cos_sim.clone()
        pos_cossim[~positives] = torch.inf
        neg_cossim[~negatives] = -torch.inf

        # --- 4. 生成全等级距离矩阵 (数轴坐标法) ---
        pos_distances = F.softplus(distances)
        class_positions = torch.cumsum(torch.cat([torch.tensor([0.0], device=self.device), pos_distances]), dim=0)
        distance_matrix = torch.abs(class_positions.unsqueeze(0) - class_positions.unsqueeze(1))

        # --- 5. 为当前 (N, M) 对分配专属护城河门槛 ---
        margins = distance_matrix[batch_labels.unsqueeze(1), mem_labels.unsqueeze(0)]
        margins[~negatives] = 0

        # --- 6. MMNP Loss 核心惩罚逻辑 ---
        mean_n_pair_loss = []
        loss_masks_2_list = []

        for j in range(M):  # 遍历 Memory 中的每一个正样本
            pos_col = pos_cossim[:, j]  # [N]
            n_pair_loss = -pos_col.unsqueeze(1) + neg_cossim + margins  # [N, 1] + [N, M] + [N, M] -> [N, M]

            loss_mask1 = ~torch.isinf(n_pair_loss)
            loss_mask2 = loss_mask1.sum(dim=1)

            n_pair_loss = F.relu(n_pair_loss)
            n_pair_loss = (n_pair_loss * loss_mask1).sum(dim=1) / loss_mask2.clamp(min=1)

            mean_n_pair_loss.append(n_pair_loss.unsqueeze(1))
            loss_masks_2_list.append(loss_mask2.unsqueeze(1))

        mean_n_pair_loss = torch.cat(mean_n_pair_loss, dim=1)  # [N, M]
        loss_masks_2 = torch.cat(loss_masks_2_list, dim=1)  # [N, M]
        final_loss = (mean_n_pair_loss * (loss_masks_2 > 0)).sum(dim=1) / (loss_masks_2 > 0).sum(dim=1).clamp(min=1)

        # --- 7. 更新记忆库 ---
        self.memory_bank.update(batch_feats, batch_labels)

        return final_loss.mean()
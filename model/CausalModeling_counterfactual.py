import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class MLPEncoder(nn.Module):
    """Multi-layer perceptron encoder for feature transformation"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLPEncoder, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, J, D] -> [B*T*J, D]
        B, T, J, D = x.shape
        x = x.reshape(-1, D)
        out = self.encoder(x)
        # Reshape back: [B*T*J, out_dim] -> [B, T, J, out_dim]
        out = out.reshape(B, T, J, -1)
        return out


class VAEConfoundGenerator(nn.Module):              # ❕得再看看能不能行，需要足够说明它的有用性
    """VAE to generate diverse counterfactual confounding factors"""

    def __init__(self, z_dim, hidden_dim=256, latent_dim=64):
        super(VAEConfoundGenerator, self).__init__()
        self.z_dim = z_dim
        self.latent_dim = latent_dim

        # Encoder: z_c -> latent distribution
        # z_c:真实观测到的混杂特征
        # 逐步降维
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)       # 均值头
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)   # 对数方差头

        # Decoder: latent -> z_cf
        # 逐步升维
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, z_dim),
            nn.Tanh()  # Bounded output👈用于激活
        )

    def encode(self, z_c):
        # z_c: [B, z_dim]

        h = self.encoder(z_c)

        # mu/logVar:隐空间的均值和方差，用于学习真实混杂特征的分布，从中采样生成合理的反事实版本
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)   # 计算标准差
        eps = torch.randn_like(std)     # 从标准正态分布采样噪声
        return mu + eps * std           # 缩放平移得到潜在变量

    def forward(self, z_c):                             # 训练流程
        mu, logvar = self.encode(z_c)
        z_latent = self.reparameterize(mu, logvar)      # 在隐空间中根据标准差、正态分布进行采样重组
        z_cf = self.decoder(z_latent)                   # 然后decode
        return z_cf, mu, logvar

    def sample_counterfactual(self, batch_size, device): # 这里是用于推理的时候
        """Sample counterfactual confounders from prior distribution"""
        z_latent = torch.randn(batch_size, self.latent_dim).to(device)  #在已经从真实数据中学到的分布用随机噪声进行采样👈如何确保有意义❓
        z_cf = self.decoder(z_latent)
        return z_cf


class RegressionHead(nn.Module):
    """Regression head for Parkinson's gait score prediction"""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(RegressionHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Pool over T and J dimensions
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single score output
        )

    def forward(self, x):
        # x: [B, T, J, D]
        B, T, J, D = x.shape
        # Global average pooling over temporal and joint dimensions
        x = x.mean(dim=[1, 2])  # [B, D]
        score = self.regressor(x)  # [B, 1]
        return score.squeeze(-1)  # [B]


class CounterfactualCausalModeling(nn.Module):
    """
    Counterfactual causal modeling for Parkinson's gait assessment

    Core idea: Answer "What would be the gait score if the confounding factors were different?"
    without explicitly enumerating all possible confounding combinations.

    Architecture:
    - Disease-related encoder: extracts gait patterns related to Parkinson's disease
    - Confounding encoder: captures nuisance factors (age, gender, camera angle, etc.)
    - VAE generator: learns the distribution of confounding factors for counterfactual sampling
    - Factual branch: real-world prediction
    - Counterfactual branch: "what-if" prediction under different confounding scenarios
    """

    def __init__(self, backbone,input_dim=512, hidden_dim=256, z_dim=128,
                 counterfactual_strategy='learned_prior',
                 use_vae_generator=True, use_disentanglement_loss=True):
        super(CounterfactualCausalModeling, self).__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.counterfactual_strategy = counterfactual_strategy
        self.use_vae_generator = use_vae_generator
        self.use_disentanglement_loss = use_disentanglement_loss

        # === Disease-related and Confounding Encoders === #
        self.disease_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=z_dim,
            num_layers=3,
            dropout=0.1
        )

        self.confound_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=z_dim,
            num_layers=3,
            dropout=0.1
        )

        # === Counterfactual Generation === #
        if self.use_vae_generator:
            self.confound_generator = VAEConfoundGenerator(
                z_dim=z_dim,
                hidden_dim=256,
                latent_dim=64
            )

        # === Shared Regression Head === #
        self.regressor = RegressionHead(
            input_dim=z_dim * 2,  # disease + confound features
            hidden_dim=256,
            dropout=0.2
        )

        # === Domain Classifier for Disentanglement === #
        if self.use_disentanglement_loss:
            self.domain_classifier = nn.Sequential(
                nn.Linear(z_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2)  # PD vs Normal
            )

    def generate_counterfactual_confounders(self, z_c_real, labels, strategy=None):     #可以在生成反事实混淆特征时手动指定生成方式，否则通过类默认进行
        """
        Generate counterfactual confounding factors

        Args:
            z_c_real: Real confounding features [B, T, J, z_dim]
            labels: Disease labels [B] (1=PD, 0=Normal)
            strategy: Override default strategy

        Returns:
            z_cf: Counterfactual confounding features [B, T, J, z_dim]
        """
        B, T, J, z_dim = z_c_real.shape
        device = z_c_real.device
        strategy = strategy or self.counterfactual_strategy

        z_c_pooled = z_c_real.mean(dim=[1, 2])  # 进行时间和关节维度上的池化


        if strategy == 'vae_sampling':
            # Use VAE to generate diverse counterfactuals
            if self.use_vae_generator:
                z_cf_pooled = self.confound_generator.sample_counterfactual(B, device)  # VAE编码学习，有意义❓
            else:
                # Fallback to random sampling
                z_cf_pooled = torch.randn_like(z_c_pooled) * 0.5                        # 随机sampling，不一定有意义


        else:
            raise ValueError(f"Unknown counterfactual strategy: {strategy}")

        # Reshape back to [B, T, J, z_dim]
        z_cf = z_cf_pooled.unsqueeze(1).unsqueeze(1).expand(-1, T, J, -1)               #恢复时间和关节维度
        return z_cf

    def compute_disentanglement_loss(self, z_g, z_c, labels):                           # ❓这个得看仔细
        """
        Encourage disentanglement: disease features should not predict confounders
        """
        # Pool features
        z_g_pooled = z_g.mean(dim=[1, 2])  # [B, z_dim]
        z_c_pooled = z_c.mean(dim=[1, 2])  # [B, z_dim]

        # Disease features shouldn't predict domain (should be confused)
        domain_pred = self.domain_classifier(z_g_pooled)

        # Create uniform labels to confuse the classifier
        uniform_labels = torch.randint_like(labels, 0, 2)
        disentangle_loss = F.cross_entropy(domain_pred, uniform_labels)

        # Orthogonality loss: encourage z_disease ⊥ z_confound
        z_disease_norm = F.normalize(z_g_pooled, p=2, dim=1)
        z_confound_norm = F.normalize(z_c_pooled, p=2, dim=1)
        orthogonal_loss = torch.mean((z_disease_norm * z_confound_norm).sum(dim=1) ** 2)

        return disentangle_loss + 0.1 * orthogonal_loss

    def forward(self, inputs, labels=None, intervention_type=None):
        """

        inputs:backbone outputs feature, [B, T, J=17, D=512] from MotionAGFormer
        labels:0,1,2
        intervention_type: currently only "vae_sample"

        Returns:
            Dictionary with factual/counterfactual predictions and features
        """
        inputs=self.backbone(inputs)        #先经过backbone跑一波


        # === 特征解耦 === #
        z_g = self.disease_encoder(inputs)      # 步态特征中与帕金森步态评分有关的特征，由疾病特征编码器输出
        z_c = self.confound_encoder(inputs)     # 步态特征中的混淆变量特征，由混淆特征编码器输出

        # === 事实分支 === #
        Z_f = torch.cat([z_g, z_c], dim=-1)  # 将两个编码器的特征输出的结果拼接送入回归头
        Y_f = self.regressor(Z_f)  # [B]

        # === 反事实分支 === #
        z_cf = (self.
        generate_counterfactual_confounders(        # 给定事实上的混淆特征z_c，进行反事实混淆特征的生成
            z_c, labels, strategy=intervention_type # 目前只支持vae采样策略
        ))
        Z_cf = torch.cat([z_g, z_cf], dim=-1)
        Y_cf = self.regressor(Z_cf)

        # === 干预效果评估 === #
        # 上面的干预起了多大作用？通过回归分数的差值来衡量
        individual_effect = Y_f - Y_cf

        # === 返回值字典 === #
        retval = {
            'factual_pred': Y_f,
            'counterfactual_pred': Y_cf,
            'individual_effect': individual_effect,

            'disease_features': z_g,
            'confound_features': z_c,
            'counterfactual_confound': z_cf

        }

        # === VAE损失  === #
        if self.use_vae_generator and self.training:
            z_confound_pooled = z_c.mean(dim=[1, 2])  # 事实混淆特征的池化结果
            z_cf_vae, mu, logvar = self.confound_generator(z_confound_pooled)       # VAE前馈输出的z_cf混淆变量

            # VAE losses                                                            #反事实干预要求：必须要对应现实中可能存在的状态，满足了某些变量的改变，而非无意义的生成;下面这两个损失尝试确保这一点
            recon_loss = F.mse_loss(z_cf_vae, z_confound_pooled)                    # 重构损失，确保z_cf与原始输入在特征空间接近？与真实样本同分布，防止生成不合理的样本👈如果不适用它，VAE容易生成安全但无意义的反事实，例如全零向量，但是那样没有意义
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]    # KL散度损失，约束潜在空间服从标准正态分布

            retval['vae_recon_loss'] = recon_loss
            retval['vae_kld_loss'] = kld_loss

        # === 解耦损失 === #
        if self.use_disentanglement_loss and self.training and labels is not None:
            disentangle_loss = self.compute_disentanglement_loss(z_g, z_c, labels)
            retval['disentanglement_loss'] = disentangle_loss

        return retval

    def predict_counterfactual(self, inputs, intervention="healthy"):               #评估流程，回答“如果它是正常人”，或者“如果它是帕金森患者”
        """
        Predict counterfactual outcomes for different interventions

        Args:
            inputs: Input features [B, T, J, D]
            intervention: Type of intervention ("healthy", "disease", "vae_sample")

        Returns:
            Counterfactual predictions
        """
        self.eval()
        with torch.no_grad():
            B = inputs.shape[0]
            device = inputs.device

            # Dummy labels for intervention
            if intervention == "healthy":
                labels = torch.ones(B).to(device)  # Assume all are PD, intervene to healthy
            else:
                labels = torch.zeros(B).to(device)  # Assume all are healthy, intervene to disease

            result = self.forward(inputs, labels, intervention_type=intervention)
            return result['counterfactual_pred']


# === Counterfactual Loss Function === #
class CounterfactualLoss(nn.Module):
    """Loss function for counterfactual causal modeling"""

    def __init__(self, lambda_consistency=0.5, lambda_vae=0.1, lambda_disentangle=0.2):
        super(CounterfactualLoss, self).__init__()
        self.lambda_consistency = lambda_consistency
        self.lambda_vae = lambda_vae
        self.lambda_disentangle = lambda_disentangle

        self.mse_loss = nn.MSELoss()

    def forward(self, model_output, labels):
        """
        Compute counterfactual causal loss

        Args:
            model_output: Output from CounterfactualCausalModeling
            labels: Ground truth gait scores [B]

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0

        # === Primary Regression Loss === #
        factual_pred = model_output['factual_pred']
        regression_loss = self.mse_loss(factual_pred, labels)
        losses['regression_loss'] = regression_loss
        total_loss += regression_loss

        # === Counterfactual Consistency Loss === #
        # Encourage meaningful but not too large differences between factual/counterfactual
        individual_effect = model_output['individual_effect']

        # Prevent trivial solutions (effect should not be zero)
        effect_magnitude = torch.mean(torch.abs(individual_effect))
        consistency_loss = torch.mean(individual_effect ** 2) - 0.1 * effect_magnitude

        losses['consistency_loss'] = consistency_loss
        total_loss += self.lambda_consistency * consistency_loss

        # === VAE Losses === #
        if 'vae_recon_loss' in model_output:
            vae_loss = model_output['vae_recon_loss'] + 0.1 * model_output['vae_kld_loss']
            losses['vae_loss'] = vae_loss
            total_loss += self.lambda_vae * vae_loss

        # === Disentanglement Loss === #
        if 'disentanglement_loss' in model_output:
            disentangle_loss = model_output['disentanglement_loss']
            losses['disentanglement_loss'] = disentangle_loss
            total_loss += self.lambda_disentangle * disentangle_loss

        losses['total_loss'] = total_loss
        return losses

#
# # === Usage Example === #
# if __name__ == "__main__":
#     # Model configuration
#     model = CounterfactualCausalModeling(
#         input_dim=512,
#         hidden_dim=256,
#         z_dim=128,
#         counterfactual_strategy='vae_sampling',
#         use_vae_generator=True,
#         use_disentanglement_loss=False
#     )
#
#     # Loss function
#     criterion = CounterfactualLoss(
#         lambda_consistency=0.5,
#         lambda_vae=0.1,
#         lambda_disentangle=0.2
#     )
#
#     # Dummy data
#     batch_size, seq_len, num_joints, feat_dim = 20, 243, 17, 512
#     inputs = torch.randn(batch_size, seq_len, num_joints, feat_dim)
#     labels = torch.randn(batch_size)  # Parkinson's gait scores
#     disease_labels = torch.randint(0, 2, (batch_size,))  # 0=Normal, 1=PD
#
#     # Training
#     model.train()
#     output = model(inputs, disease_labels)
#     losses = criterion(output, labels)
#
#     print("=== Training Results ===")
#     print("Losses:", {k: f"{v.item():.4f}" for k, v in losses.items()})
#
#     # Counterfactual Inference
#     model.eval()
#     cf_disease = model.predict_counterfactual(inputs, intervention="vae_sampling")
#
#     print("\n=== Counterfactual Predictions ===")
#     print(f"Original scores: {output['factual_pred'].detach()}")
#     print(f"If VAE sampled: {cf_disease}")
#



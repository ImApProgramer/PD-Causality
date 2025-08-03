import torch.nn as nn
import torch
from model.ctrgcn.ctrgcn import Model as CTRGCN
from model.ctrgcn import ntu_rgb_d as Graph



class ClassifierHead(nn.Module):
    def __init__(self, params, num_classes=3, num_joints=17):       #below says num_classes=4, how come now becomes 3?
        super(ClassifierHead, self).__init__()
        self.params = params
        input_dim = self._get_input_dim(num_joints)     #GCN:6406??
        if self.params['medication']:
            input_dim += 1
        if len(self.params['metadata']) > 0:
            input_dim += len(self.params['metadata'])
        self.dims = [input_dim, *self.params['classifier_hidden_dims'], num_classes]

        self.fc_layers = self._create_fc_layers()
        self.batch_norms = self._create_batch_norms()
        self.dropout = nn.Dropout(p=self.params['classifier_dropout'])
        self.activation = nn.ReLU()

    def _create_fc_layers(self):
        fc_layers = nn.ModuleList()
        mlp_size = len(self.dims)

        for i in range(mlp_size - 1):
            fc_layer = nn.Linear(in_features=self.dims[i],
                                 out_features=self.dims[i+1])
            fc_layers.append(fc_layer)
        
        return fc_layers
    
    def _create_batch_norms(self):
        batch_norms = nn.ModuleList()
        n_batchnorms = len(self.dims) - 2
        if n_batchnorms == 0:
            return batch_norms
        
        for i in range(n_batchnorms):
            batch_norm = nn.BatchNorm1d(self.dims[i+1], momentum=0.1)
            batch_norms.append(batch_norm)
        
        return batch_norms

    def _get_input_dim(self, num_joints):           #这个其实就是将backbone输出的x展平，喂入分类头，所以要搞清楚GCN输出的是啥样的
        backbone = self.params['backbone']
        if backbone == 'poseformer':
            if self.params['preclass_rem_T']:
                return self.params['model_dim']
            else:
                return self.params['model_dim'] * self.params['source_seq_len']
        elif backbone == "motionbert":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * num_joints
        elif backbone == 'poseformerv2':
            return self.params['embed_dim_ratio'] * num_joints * 2
        elif backbone == "mixste":
            if self.params['merge_joints']:
                return self.params['embed_dim_ratio']
            else:
                return self.params['embed_dim_ratio'] * num_joints
        elif backbone == "motionagformer":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * num_joints
        elif backbone == "ctrgcn":          #这里默认你已经做了时间和空间上的池化
            return self.params['dim_rep']


    def forward(self, feat):
        feat = self.dropout(feat)
        if self.params['backbone'] == 'motionbert':
            return self._forward_motionbert(feat)
        elif self.params['backbone'] == 'poseformer':
            return self._forward_poseforemer(feat)
        elif self.params['backbone'] == 'poseformerv2':
            return self._forward_poseformerv2(feat)
        elif self.params['backbone'] == "mixste":
            return self._forward_mixste(feat)
        elif self.params['backbone'] == "motionagformer":
            return self._forward_motionagformer(feat)
        elif self.params['backbone'] == "ctrgcn":
            return self._forward_ctrgcn(feat)

    def _forward_fc_layers(self, feat):
        mlp_size = len(self.dims)
        for i in range(mlp_size - 2):
            fc_layer = self.fc_layers[i]
            batch_norm = self.batch_norms[i]

            feat = self.activation(batch_norm(fc_layer(feat)))

        last_fc_layer = self.fc_layers[-1]      #倒数第一个层
        feat = last_fc_layer(feat)
        return feat
    
    def _forward_motionagformer(self, feat):
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat
    
    def _forward_mixste(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        """
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_poseformerv2(self, feat):
        """
        x: Tensor with shape (batch_size, 1, embed_dim_ratio * num_joints * 2)
        """
        B, _, C = feat.shape
        feat = feat.reshape(B, C)  # (B, 1, C) -> (B, C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_motionbert(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        """
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_poseforemer(self, feat):
        """
        x: Tensor with shape (batch_size, n_frames, dim_representation)
        """
        T, B, C = feat.shape
        if self.params['preclass_rem_T']:
            # Reshape the tensor to (B, 1, C, T)   J=1
            feat = feat.permute(1, 2, 0).unsqueeze(1)
            feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        else:
            feat = feat.permute(1, 0, 2)  # (B, T, C)

        feat = feat.reshape(B, -1)  # (B, J * C) or (B, T * C)
        feat  = self._forward_fc_layers(feat)
        return feat

    def _forward_ctrgcn(self, feat):            #首先要知道，各个模型之所以需要不同的forward逻辑，就是因为它们输入分类头的形状不同，这些分支用来把它们统一变成[B,C]输入

        return self._forward_fc_layers(feat)


class MotionEncoder(nn.Module):
    def __init__(self, backbone, params, num_classes=4, num_joints=17, train_mode='end2end'):
        super(MotionEncoder, self).__init__()
        assert train_mode in ['end2end', 'classifier_only'], "train_mode should be either end2end or classifier_only." \
                                                             f" Found {train_mode}"
        self.backbone = backbone
        if train_mode == 'classifier_only':
            self.freeze_backbone()
        self.head = ClassifierHead(params, num_classes=num_classes, num_joints=num_joints)
        self.num_classes = num_classes
        self.medprob = params['medication']
        self.metadata = params['metadata']

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO - MotionEncoder] Backbone parameters are frozen")

    def forward(self, x, metadata, med=None):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, C=3)
        """
        feat = self.backbone(x)
        if self.medprob and med is not None:
            med = med.to(feat.device)
            med = med.view(*[-1] + [1] * (feat.dim() - 1))
            s = list(feat.shape)
            s[-1] = 1  # Set the last dimension to 1
            med = med.expand(*s)
            feat = torch.cat((feat, med), dim=-1)
        if len(self.metadata) > 0:
            metadata = metadata.view(metadata.shape[0], *([1] * (feat.dim() - 2)), metadata.shape[-1])
            metadata = metadata.expand(*feat.shape[:-1], metadata.shape[-1])
            feat = torch.cat((feat, metadata), dim=-1)
        out = self.head(feat)
        return out


class CTRGCNEncoderWrapper(nn.Module):          #继承CTR-GCN Model来初始化,仅用于本文件测试，主模型不这样加载；主模型改动之后它就废了
    def __init__(self, ctrgcn_model: nn.Module):
        super().__init__()
        self.model = ctrgcn_model

    def forward(self, x):
        # x shape: (N, C, T, V, M)
        N, C, T, V, M = x.shape

        # ==== 与原Model.forward相同的预处理 ====
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.model.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # === 仅提取10层TCN-GCN的特征 ===
        x = self.model.l1(x)
        x = self.model.l2(x)
        x = self.model.l3(x)
        x = self.model.l4(x)
        x = self.model.l5(x)
        x = self.model.l6(x)
        x = self.model.l7(x)
        x = self.model.l8(x)
        x = self.model.l9(x)
        x = self.model.l10(x)

        # Output: (N*M, 256, T, V)
        # 如果你不想保留M个 skeleton，可以平均一下：
        x = x.view(N, M, x.size(1), x.size(2), x.size(3)).mean(1)  # => (N, 256, T, V)，这里要考虑和ClassifierHead的对接

        x = x.mean(dim=-1)  # pool over V => (N, 256, T)
        x = x.mean(dim=-1)  # pool over T => (N, 256)


        return x



def _test_classifier_head():
    params = {
        "backbone": "motionbert",
        "dim_rep": 512,
        "classifier_hidden_dims": [],
        'classifier_dropout': 0.5
    }
    head = ClassifierHead(params, num_classes=3, num_joints=17)

    B, T, J, C = 4, 243, 17, 512
    feat = torch.randn(B, T, J, C)
    out = head(feat)
    assert out.shape == (4, 3)



def _test_GCN_model():
    x = torch.randn(2, 3, 81, 25, 1)  # (B=2, C=3, T=81, V=25, M=1)，这就是到时需要的输入
    metadata = torch.randn(2, 5)  # 5个 metadata features
    med = torch.randint(0, 2, (2, 1)).float()  # binary medication flag


    params = {
        "backbone": "ctrgcn",
        "dim_rep": 256,
        "classifier_hidden_dims": [],
        'classifier_dropout': 0.5,
        'num_classes': 3,
        'source_seq_len': 81,
        'medication': 1,
        'metadata': ['gender', 'age', 'height', 'weight', 'bmi']
    }
    ctrgcn_backbone = CTRGCN(
        num_class=60,  # 这里的值无所谓，不用fc
        num_point=25,
        num_person=1,
        graph='graph.ntu_rgb_d.Graph',
        graph_args={},
        in_channels=3,
        adaptive=True
    )
    motion_backbone = CTRGCNEncoderWrapper(ctrgcn_backbone)
    motion_encoder=MotionEncoder(motion_backbone, params,params['num_classes'], num_joints=25, train_mode='end2end')


    out = motion_encoder(x, metadata, med)
    print(out.shape)  # 应为 (2, 3)



if __name__ == "__main__":
    #_test_classifier_head()
    _test_GCN_model()

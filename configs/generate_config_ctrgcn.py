import json
from const import path


#TODO:根据后面的模型、data细节再回来修改，现在给了一个极简版本；不要对着这个空改，要看后面的参数怎么用再说

def generate_config(param, f_name):
    data_params = {
        'data_type': 'PD',
        'in_channels': 3,
        'num_point': 25,
        'num_person': 1,
        'data_path': path.PD_PATH_POSES_forGCN,
        'labels_path': path.PD_PATH_LABELS,
        'data_centered': True,              #用来考虑是否需要进行中心化
        'model_prefix': 'GCN_',
        'data_norm': "rescaling",           #不确定用途，只是为了别报错
        'source_seq_len': 81,
        'use_validation': True,              #是否启用验证
        'select_middle': False,
        'mirror_prob': 0.5,
        'rotation_range': [-10, 10],
        'rotation_prob': 0.5,
        'noise_prob': 0.5,
        'axis_mask_prob': 0.5
    }

    model_params = {
        'model': 'CTRGCN',
        'dim_rep': 256,
        'experiment_name': '',
        'classifier_dropout': 0.5,
        #'classifier_hidden_dims': [1024],
        'model_args': {
            'in_channels': 3,
            'num_class': 3,  # 修改为你的动作类别数
            'graph_args': {
                'layout': 'ntu_rgb_d',  # 或者 'coco'
                'strategy': 'uniform'
            }
        },
        'weights': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/ctrgcn/ctrgcn-weights.bin",
        'model_checkpoint_path': f"{path.PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH}/ctrgcn/runs-58-57072.pt"      #ntu120 csub joint
    }

    learning_params = {
        'batch_size': 64,   #太大显存顶不住，其他模型压力比较小
        'epochs': 20,          #建议60~80
        'lr_head': 0.001,
        'lr_backbone': 0.0001,
        'optimizer': 'SGD',
        'weight_decay': 0.0001,
        'momentum' : 0.9,
        'nesterov': True,
        'lr_decay_step': [10, 15],
        'dropout_rate': 0.5,  # ✅ 添加此字段以匹配 update_params_with_best 中的 dropout_rate
        'use_weighted_loss': True,  # 0807调参
        'lambda_l1': 0.0,  # ✅ 默认值
        'wandb_name': 'CTRGCN',  # ✅ 必须手动提供才能被 update 函数处理
        'stopping_tolerance': 10,       #用于早停，不知道是否用到，但是train中显式检查了
        'criterion': 'WCELoss',          #和poseformer用的一样
        #'scheduler': "StepLR",
        'lr_step_size': 1,
        'lr_decay': 0.99
    }

    params = {**param, **data_params, **model_params, **learning_params}

    # f = open("./configs/ctrgcn/" + f_name, "rb")
    # new_param = json.load(f)
    # for p in new_param:
    #     if not p in params:
    #         raise ValueError("Unrecognized parameter in config: " + p)
    #     params[p] = new_param[p]

    return params, {}#, new_param

#注：/czl_hdd/Public_PD/是数据存放点，preprocess_pd.py的parse_args()中的"data_path"是它的起点
#这里的PD_PATH开头的必须与它对齐

NDRIVE_PROJECT_ROOT = '/czl_ssd/motion_evaluator'      #最终存放.npz和.pkl等预处理之后文件的地方
CAUSAL_NDRIVE_PROJECT_ROOT = '/czl_ssd/causal'

PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = f'{NDRIVE_PROJECT_ROOT}/Pretrained_checkpoints'
CAUSAL_PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = f'{CAUSAL_NDRIVE_PROJECT_ROOT}/Pretrained_checkpoints'

OUT_PATH = '/czl_ssd/log/motion_encoder/out'
CAUSAL_OUT_PATH='/czl_ssd/log/causal/out'

# KINECT
PREPROCESSED_DATA_ROOT_PATH = f'{NDRIVE_PROJECT_ROOT}/data'
CAUSAL_PREPROCESSED_DATA_ROOT_PATH = f'{CAUSAL_NDRIVE_PROJECT_ROOT}/data'

# PD
PD_PATH_POSES='/czl_ssd/Public_PD/C3Dfiles_processed_new'
PD_PATH_POSES_forGCN='/czl_ssd/Public_PD/C3Dfiles_processed_GCN'
PD_PATH_LABELS='/czl_ssd/Public_PD/PDGinfo.csv'

CHECKPOINT_ROOT_PATH =  '/caa/Homes01/iballester/log/motion_encoder/out/motionbert/finetune_6_pd,json/1/models'
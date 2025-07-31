NDRIVE_PROJECT_ROOT = '/czl_ssd/motion_evaluator'      #最终存放.npz和.pkl等预处理之后文件的地方

PRETRAINEDD_MODEL_CHECKPOINTS_ROOT_PATH = '/data/iballester/motion_evaluator/Pretrained_checkpoints'    #需要给出一个存放预训练文件的点，但是预训练文件去哪里找？

OUT_PATH = '/caa/Homes01/iballester/log/motion_encoder/out/'        #可视化阶段还用不到

# KINECT
PREPROCESSED_DATA_ROOT_PATH = f'{NDRIVE_PROJECT_ROOT}/data'

# PD
PD_PATH_POSES='/czl_ssd/Public_PD/C3Dfiles_processed_new'
PD_PATH_LABELS='/czl_ssd/Public_PD/PDGinfo.csv'

CHECKPOINT_ROOT_PATH =  '/caa/Homes01/iballester/log/motion_encoder/out/motionbert/finetune_6_pd,json/1/models'
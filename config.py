from pathlib import Path

import utils
# 公開できない情報をまとめたファイル
import secrets

class Config:
    # GPU param
    gpu_count = 1

    # Basic Parameter
    batch_size = 16
    data_path = secrets.data_path
    epochs = 1
    img_size = 128
    channel = 3
    max_queue_size = 10
    save_model_root = secrets.save_model_root
    save_model_root.mkdir(exist_ok=True)
    ## When you save model, you must not use mounted file path
    ## dirac経由でモデルを保存することはできない
    save_model_path = Path(save_model_root, 'model.h5')
    use_multiprocessing = False
    train_list, train_steps = utils.calc_steps(Path(data_path, 'train'))
    val_list, val_steps = utils.calc_steps(Path(data_path, 'validation'))
    verbose = 1
    weights_path = Path('')
    workers = 1 # when workers > 1, multi process is executed

    # DataAugmentation
    da_channel = 0
    da_height = 0.1
    da_hflip = True
    da_rotate = 90
    da_shear = 90
    da_std = 0
    da_vflip = True
    da_width = 0.1
    da_zoom = 0.1

    # Generator
    gen_class = 'categorical'
    gen_color = 'rgb'

    # For Classification
    clf_class = 3
    clf_lr = 0.0001
    clf_loss = 'categorical_crossentropy'
    clf_mdl_name = "inception"
    clf_opt = 'adam'
    clf_met = ['accuracy']
    clf_vis_path = Path(save_model_root, 'arch.png') # plot model's architecure

    # For R-CNN
    rcnn_stage = 'all'
    rcnn_lr = 0.0001
    ## For ReduceLROnPlateau
    rcnn_rlr_monitor='val_loss'
    rcnn_rlr_verbose=1
    rcnn_rlr_factor=0.7
    rcnn_rlr_patience=10
    rcnn_rlr_min=rcnn_lr/30

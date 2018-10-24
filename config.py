from pathlib import Path

import bp_utils
# 公開できない情報をまとめたファイル
import secrets

class Config:
    def __init__(self):
        # GPU param
        self.gpu_count = 1

        # execute stage
        ## 'train': training only
        ## 'eval': evaluate only
        ## 'all': execute all
        self.stage = 'eval'

        # Basic Parameter(all)
        ## When you save model, you must not use mounted file path
        ## dirac経由でモデルを保存することはできない
        self.save_model_root = secrets.save_model_root
        self.save_model_root.mkdir(exist_ok=True)
        self.save_model_path = Path(self.save_model_root, 'model.h5')

        # Basic Parameter(training)
        self.batch_size = 16
        self.data_path = secrets.data_path
        self.epochs = 1
        self.img_size = (128, 128)
        self.channel = 3
        self.max_queue_size = 10
        self.use_multiprocessing = False
        self.train_list = bp_utils.make_img_list(Path(self.data_path, 'train'))
        self.val_list = bp_utils.make_img_list(Path(self.data_path, 'validation'))
        self.train_steps, self.val_steps = len(self.train_list), len(self.val_list)
        self.train_verbose = 1
        self.weights_path = Path('')
        self.workers = 1 # when workers > 1, multi process is executed

        # Basic Parameter(evaluate)
        self.eval_data_path = secrets.eval_data_path
        self.eval_verbose = 1
        self.sample_weight = None

        # DataAugmentation
        self.da_channel = 0
        self.da_height = 0.1
        self.da_hflip = True
        self.da_rotate = 90
        self.da_shear = 90
        self.da_std = 0
        self.da_vflip = True
        self.da_width = 0.1
        self.da_zoom = 0.1

        # Generator
        self.gen_class = 'categorical'
        self.gen_color = 'rgb'

        # For Classification
        self.clf_class = 3
        self.clf_lr = 0.0001
        self.clf_loss = 'categorical_crossentropy'
        self.clf_mdl_name = "inception"
        self.clf_opt = 'adam'
        self.clf_met = ['accuracy']
        self.clf_vis_path = Path(self.save_model_root, 'arch.png') # plot model's architecure

        # For R-CNN
        self.rcnn_stage = 'all'
        self.rcnn_lr = 0.0001
        ## For ReduceLROnPlateau
        self.rcnn_rlr_monitor='val_loss'
        self.rcnn_rlr_verbose=1
        self.rcnn_rlr_factor=0.7
        self.rcnn_rlr_patience=10
        self.rcnn_rlr_min=self.rcnn_lr/30

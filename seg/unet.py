'''define unet model for a finder module'''
import csv
import cv2
from datetime import datetime
import fire
from glob import glob
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from scipy.misc import imresize, toimage
import shutil
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from skimage.io import imread, imsave
import sys
import tensorflow as tf
import time
from tqdm import tqdm
import yaml

from keras import backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, TensorBoard,
                             ModelCheckpoint, ReduceLROnPlateau)
from keras.layers import (BatchNormalization, concatenate, Conv2D,
                          Conv2DTranspose, MaxPooling2D, Activation,
                          Dropout, Flatten, Dense, Input)
from keras.models import load_model, Model, Sequential
from keras.utils import np_utils

sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[3]))

from module.segmentation import base_segment
from misc import loss_and_metric

# Define of Unet
class Unet(base_segment.BaseSegment):
    def _vgg_block(self, channels, x, act_func='relu'):
        '''the process of UNet convolution
        # Arguments
            channels:   feature channels
            x:          input
        # Returns
            x:          processed output by vgg_block
        '''
        x = Conv2D(channels, (3, 3), padding='same')(x)
        ### add BatchNormalization(this is not written in the paper of UNet, so it is original)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        x = Conv2D(channels, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)

        return x

    def dice_coef(y_true, y_pred):
        y_true_f = K.round(K.flatten(y_true))
        y_pred_f = K.round(K.flatten(y_pred))
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

    def _build_arch(self):
        '''Build model of UNet
        '''

        if input_dim[0] == 0:
            input_dim = (self.config.img_size[0], self.config.img_size[1], self.config.channel)
        if lr == 0:
            lr = self.config.seg_lr
        if not opt_name:
            opt_name = self.mdl_opt
        ### define the input (shape=(img_row, img_col, img_dim))
        x_input = Input(shape=(input_dim[0], input_dim[1], input_dim[2]))

        ## Contracting Path
        ### convolution block
        conv1 = self._vgg_block(base_num, x_input, act_func=act_func)
        ### pooling block
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self._vgg_block(base_num*2, pool1, act_func=act_func)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self._vgg_block(base_num*4, pool2, act_func=act_func)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self._vgg_block(base_num*8, pool3, act_func=act_func)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self._vgg_block(base_num*16, pool4, act_func=act_func)

        ## Expansive Path
        ### concatnating between the output from conv5 and conv4
        ### this is skip connection
        up6 = concatenate([Conv2DTranspose(base_num*8, (3, 3), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        ### add BatchNormalization(this is not written in the paper of UNet, so it is original)
        up6 = BatchNormalization()(up6)
        ### activation function
        up6 = Activation(act_func)(up6)
        ### convolution
        conv6 = self._vgg_block(base_num*8, up6, act_func=act_func)


        up7 = concatenate([Conv2DTranspose(base_num*4, (3, 3), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        up7 = BatchNormalization()(up7)
        up7 = Activation(act_func)(up7)
        conv7 = self._vgg_block(base_num*4, up7, act_func=act_func)

        up8 = concatenate([Conv2DTranspose(base_num*2, (3, 3), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        up8 = BatchNormalization()(up8)
        up8 = Activation(act_func)(up8)
        conv8 = self._vgg_block(base_num*2, up8, act_func=act_func)

        up9 = concatenate([Conv2DTranspose(base_num, (3, 3), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        up9 = BatchNormalization()(up9)
        up9 = Activation(act_func)(up9)
        conv9 = self._vgg_block(base_num, up9, act_func=act_func)

        ## output by sigmoid
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        ## define model from above code
        model = Model(inputs=[x_input], outputs=[conv10])

        ## Compile
        model.compile(loss=self.mdl_loss,
                      optimizer=getattr(sys.modules['keras.optimizers'],
                                        opt_name)(lr=lr),
                      metrics=[dice_coef])
                      # metrics=[getattr(sys.modules['misc.loss_and_metric'],
                      #                  self.mdl_mat)])

        return model.get_config(), model

if __name__=="__main__":
    fire.Fire(Unet)

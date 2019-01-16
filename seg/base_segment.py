'''define unet model for a finder module'''
import csv
import cv2
import fire
from datetime import datetime
from glob import glob
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from scipy.misc import imresize, toimage
import scipy.misc
import shutil
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from skimage.io import imread, imsave
import sys
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))
from misc import (augmentation, data_generator, loss_and_metric,
                  write_data, img_processing, load_data)

# Define of a finder module
class Baseseg(object):
    def __init__(self, config):
        self.config = config

    def _build_arch(self):
        model = Sequential()
        return model

    def build_model(self):
        with tf.device('/cpu:0'):
            model = self._build_arch()
        if self.config.gpu_count > 1:
            model = utils.multi_gpu_model(model, self.config.gpu_count)

        model.compile(loss=self.config.seg_loss,
                      optimizer=self.config.seg_opt,
                      metrics=self.config.seg_met)
        return model

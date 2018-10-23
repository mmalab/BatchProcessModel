from keras.models import Sequential
from keras import utils
import tensorflow as tf

class Baseclf:
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

        model.compile(loss=self.config.clf_loss,
                      optimizer=self.config.clf_opt,
                      metrics=self.config.clf_met)
        return model

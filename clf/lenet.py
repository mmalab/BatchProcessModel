from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Activation,
                          Dropout, Flatten, Dense, Input)
from pathlib import Path
import sys

sys.path.append(Path(__file__).parent.as_posix())
from base_clf import Baseclf

class Lenet(Baseclf):
    def _build_arch(self):
        model = Sequential()
        # First Layer
        model.add(Conv2D(32,
                         (5, 5),
                         input_shape=(self.config.img_size,
                                      self.config.img_size,
                                      self.config.channel)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Second Layer
        model.add(Conv2D(64, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Full Connected Layer
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.config.clf_class))
        model.add(Activation('sigmoid'))

        return model

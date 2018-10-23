import config
from logging import getLogger, StreamHandler, INFO
import tensorflow as tf
from keras import backend as K
from keras import utils
import keras.callbacks

from clf import lenet
import generator

# Set Logger
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False
logger.info("Executing {}".format(__name__))

# Set session
logger.info("Setting GPU")
sess_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=sess_conf)
K.set_session(session)

logger.info("Building Model")
# Fetch config
conf = config.Config()

# Main
arch = lenet.Lenet(conf)
model = arch.build_model()
if conf.clf_vis_path:
    print("Under implemented")
    # utils.plot_model(model,
    #                  conf.clf_vis_path.as_posix(),
    #                  True, True)

if conf.weights_path.is_file():
    model.load_weights(conf.weights_path, by_name=True)

callbacks = [keras.callbacks.TerminateOnNaN(),
             keras.callbacks.ModelCheckpoint(filepath=conf.save_model_path.as_posix(),
                                             verbose=1,
                                             save_weights_only=True,
                                             save_best_only=True),
             keras.callbacks.ReduceLROnPlateau(monitor=conf.rcnn_rlr_monitor,
                                               verbose=conf.rcnn_rlr_verbose,
                                               factor=conf.rcnn_rlr_factor,
                                               patience=conf.rcnn_rlr_patience,
                                               min_lr=conf.clf_lr)]
train_gen, val_gen = generator.dir_gen(conf)

logger.info("Training")
model.fit_generator(train_gen,
                    steps_per_epoch=conf.train_steps,
                    epochs=conf.epochs,
                    verbose=conf.verbose,
                    workers=conf.workers,
                    max_queue_size=conf.max_queue_size,
                    use_multiprocessing=conf.use_multiprocessing,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    validation_steps=conf.val_steps)
logger.info("Finished")

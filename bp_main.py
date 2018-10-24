import config
from logging import getLogger, StreamHandler, INFO
import tensorflow as tf
from keras import backend as K
from keras import utils
import keras.callbacks

from clf import lenet
import generator
import bp_utils

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
    logger.info("Loaded model weight for fine-tuning")
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

if conf.stage in ['train', 'all']:
    logger.info("Training")
    model.fit_generator(train_gen,
                        steps_per_epoch=conf.train_steps,
                        epochs=conf.epochs,
                        verbose=conf.train_verbose,
                        workers=conf.workers,
                        max_queue_size=conf.max_queue_size,
                        use_multiprocessing=conf.use_multiprocessing,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=conf.val_steps)
    logger.info("Finished training")

if conf.stage in ['eval', 'all']:
    logger.info("Evaluating")
    eval_data, eval_label = bp_utils.fetch_datalist_with_label(conf.eval_data_path, conf)
    if conf.save_model_path.exists():
        model.load_weights(conf.save_model_path.as_posix(), by_name=True)
        logger.info("Loaded save model weight")

    eval = model.evaluate(x=eval_data,
                          y=eval_label,
                          batch_size=conf.batch_size,
                          verbose=conf.eval_verbose,
                          sample_weight=conf.sample_weight)
    print("Evaluate result: {}".format(eval))
    logger.info("Finished evaluating")

    preds = model.predict(eval_data)
    precision, recall, f_measure = bp_utils.calc_f_measure(tests=eval_label,
                                                           preds=preds)
    fpr, tpr, roc_auc = bp_utils.calc_auc_roc(eval_label, preds)
    bp_utils.draw_roc_auc([fpr],[tpr],[roc_auc],
                          conf.save_model_root,
                          {"class_title_list": ["class1", "class2", "class3"],
                           "mic_mac_titles": ["micro", "macro"],
                           "overall_title": 'All class ROC'})

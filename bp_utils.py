from keras import utils
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy import interp
from sklearn import metrics

# About Data
def make_img_list(data_path):
    data_list = sorted(Path(data_path).glob('**/*.bmp')) \
    + sorted(Path(data_path).glob('**/*.jpeg')) \
    + sorted(Path(data_path).glob('**/*.jpg')) \
    + sorted(Path(data_path).glob('**/*.png'))
    return data_list

def fetch_datalist_with_label(data_path, config, label_txt_path=''):
    data_path = Path(data_path)
    data_list = []
    label_list = []
    b_gray = False
    if config.gen_color != 'rgb':
        b_gray = True

    if not label_txt_path:
        subdirs = [subdir for subdir in sorted(data_path.glob('*')) if subdir.is_dir()]
        for (i, subdir) in enumerate(subdirs):
            subdir_img_paths = make_img_list(subdir)
            for sub in subdir_img_paths:
                sub_img = load_img(sub.as_posix(),
                                   grayscale=b_gray,
                                   target_size=(config.img_size[0],
                                                config.img_size[1]))
                sub_img = img_to_array(sub_img)/255.
                data_list.append(sub_img)
            label_list.extend([i]*len(subdir_img_paths))
    label_list = np.asarray(label_list)
    label_list = utils.np_utils.to_categorical(label_list)
    return np.asarray(data_list), label_list

# About Evaluate
def calc_f_measure(tests, preds):
    ## initalize
    cell_size = len(tests[0])
    cell = np.zeros((cell_size, 1, cell_size))
    for (test, pred) in zip(tests, preds):
        cell[np.argmax(test)][0][np.argmax(pred)] += 1

    ## calc precision
    precision = np.zeros((cell_size))
    recall = np.zeros((cell_size))

    for i in range(cell_size):
        TP = cell[i][0][i]
        FP = sum(cell[i][0][:i]) + sum(cell[i][0][(i+1):])
        precision[i] = TP/(TP + FP)

    cell = cell.T
    for i in range(cell_size):
        TN = cell[i][0][i]
        FN = sum(cell[i][0][:i]) + sum(cell[i][0][(i+1):])
        recall[i] = TP/(TP + FN)

    f_measure = 2*precision*recall/(precision + recall)
    return precision, recall, f_measure

def calc_auc_roc(y_tests, y_preds):
    y_tests = np.array(y_tests)
    y_preds = np.array(y_preds)
    class_len = len(y_tests[0])
    # FPR, TPR(, しきい値) を算出
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(class_len):
        fpr[i], tpr[i], thresholds = metrics.roc_curve(y_tests[:, i], y_preds[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_tests.ravel(), y_preds.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_len)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_len):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= class_len

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc

def draw_roc_auc(fprs, tprs, roc_aucs,
                 save_path,
                 names={"class_title_list": [],
                        "mic_mac_titles": [],
                        "overall_title": ''}):
        # origin_colors = cycle(['aqua', 'darkorange', 'lime'])
        # masked_colors = cycle(['aqua', 'darkorange', 'lime'])
        if not "DISPLAY" in os.environ:
            plt.switch_backend("Agg")

        for (i, name) in enumerate(names["class_title_list"]):
            for (fpr, tpr, roc_auc) in zip(fprs, tprs, roc_aucs):
                # ROC曲線をプロット
                plt.plot(fpr[i],
                         tpr[i],
                         label='ROC curve of class {0} (area = {1:0.2f})'.format(name, roc_auc[i]))
                plt.legend()
                plt.title('ROC curve {}'.format(name))
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.grid(True)

            save_name = Path(save_path, 'ROCcurve_class{}.png'.format(i))
            plt.savefig(str(save_name))
            plt.close()
        for (fpr, tpr, roc_auc) in zip(fprs, tprs, roc_aucs):
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro ROC curve of {0} (area = {1:0.2f})'
                           ''.format(names["mic_mac_titles"][0], roc_auc["micro"]),
                     linestyle=':', linewidth=2)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro ROC curve of {0} (area = {1:0.2f})'
                           ''.format(names["mic_mac_titles"][1], roc_auc["macro"]),
                     linewidth=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(names["overall_title"])
        plt.legend(loc="lower right")
        save_miacro_name = Path(save_path, 'micromacroROC.png')
        plt.savefig(str(save_miacro_name))
        plt.close()

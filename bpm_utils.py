import numpy as np
from pathlib import Path
from keras import utils
from keras.preprocessing.image import load_img, img_to_array

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
                data_list.append(img_to_array(sub_img))
            label_list.extend([i]*len(subdir_img_paths))
    label_list = np.asarray(label_list)
    label_list = utils.np_utils.to_categorical(label_list)
    return np.asarray(data_list), label_list

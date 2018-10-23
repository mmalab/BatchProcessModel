from pathlib import Path

def calc_steps(data_path):
    data_list = sorted(Path(data_path).glob('**/*.bmp')) \
    + sorted(Path(data_path).glob('**/*.jpeg')) \
    + sorted(Path(data_path).glob('**/*.jpg')) \
    + sorted(Path(data_path).glob('**/*.png'))
    steps = len(data_list)
    return data_list, steps

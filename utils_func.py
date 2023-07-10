import os


def get_base_filename(fullpath):
    # fullpath = /root/autodl-tmp/data/gc14w/img/000000000036.jpg
    # return 000000000036.jpg
    return os.path.split(os.path.sep)[-1]


def get_base_filename_without_ext(fullpath):
    # fullpath = /root/autodl-tmp/data/gc14w/img/000000000036.jpg
    # return 000000000036
    return os.path.splitext(fullpath)[0].split(os.path.sep)[-1]


def fill_idx(idx, ext=".jpg"):
    return str(idx).zfill(12) + ext

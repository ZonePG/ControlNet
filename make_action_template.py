import os
from collections import defaultdict
from glob import glob
import time
import json
import random
import numpy as np
import math
import itertools
import cv2
from coco_util import coco, catIds, img_ids, get_center_keypoints

TEMPLATE_PATH = "/root/autodl-tmp/datasets/caption"

jpgs = sorted(glob(os.path.join(TEMPLATE_PATH, "**", "*.jpg"), recursive=True))

dict_save = {}

for jpg in jpgs:
    jpg_list = jpg.split(os.path.sep)
    action = jpg_list[-2]
    item = jpg_list[-1]
    if dict_save.get(action) is None:
        dict_save[action] = {}

    dict_save[action][item] = {
        "small": [],
        "middle": [],
        "large": [],
    }

    img_idx = int(item.split(".")[0])
    for size in ["small", "middle", "large"]:
        kpts = get_center_keypoints(img_idx, size=size)
        dict_save[action][item][size] = kpts

json.dump(dict_save, open("dict_save.json", "w"))

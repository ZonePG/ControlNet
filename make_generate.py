import json
import os
import random
import math
import numpy as np
from coco_util import get_center_keypoints, get_box, kp_trans_dst, affine_points, show_skelenton
import cv2
from coco_format import *

H, W = 600, 720

SAVE_PATH = "/root/autodl-tmp/datasets/random_generate_3"
SAVE_IMAGE_HQ_PATH = os.path.join(SAVE_PATH, "hq")
SAVE_TXT_PATH = os.path.join(SAVE_PATH, "coco_txt")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_TXT_PATH, exist_ok=True)

table_num = ["zero", "one", "two", "three", "four"]

if __name__ == "__main__":
    data_dict = json.load(open("/root/autodl-tmp/datasets/dict/dict.json", "r"))
    size = ["small", "middle", "large"]
    actions = list(data_dict.keys())
    img_idx = 1
    while img_idx <= 3000:
        img_name = str(img_idx).zfill(12) + ".jpg"
        txt_name = str(img_idx).zfill(12) + ".txt"
        action_num = random.randint(1, 4)

        choice_action = []
        choice_img_name = []
        choice_size = []
        action_count = {}
        keypoints = []
        json_box = []
        txt = None

        for _ in range(action_num):
            random_action = random.choice(actions)
            choice_action.append(random_action)
            choice_size.append(random.choice(size))
            if random_action not in action_count:
                action_count[random_action] = 1
            else:
                action_count[random_action] = action_count[random_action] + 1

        for action in choice_action:
            choice_img_name.append(random.choice(list(data_dict[action].keys())))

        txt = f"There are {table_num[action_num]} people in the picture, "
        for action, count in action_count.items():
            txt += f"{table_num[count]} people is {action}ing, "
        txt = txt[:-2] + "."

        center = []
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for idx in range(len(choice_img_name)):
            # img_idx = int(choice_img_name[i])
            keypoint = data_dict[choice_action[idx]][choice_img_name[idx]][choice_size[idx]]
            kpts = np.array(keypoint).reshape(-1, 3)
            box = get_box(keypoint)
            X_min, Y_min, X_max, Y_max = box
            w = X_max - X_min
            h = Y_max - Y_min

            target_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
            target_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
            while_count = 100
            if len(center) > 0:
                np_center = np.array(center)
                distances = np.min(np.sqrt(np.sum((np_center - np.array([target_center_X, target_center_Y]))**2, axis=1)))
                while distances < 300 and while_count > 0:
                    target_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
                    target_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
                    distances = np.min(np.sqrt(np.sum((np_center - np.array([target_center_X, target_center_Y]))**2, axis=1)))
                    while_count -= 1
                if while_count == 0:
                    break
            dst_center = np.array([target_center_X, target_center_Y])
            center.append(dst_center)
            trans = kp_trans_dst(box, dst_shape=(H, W), dst_center=dst_center)
            kpts1 = affine_points(kpts, trans)
            json_box.append([target_center_X - w / 2, target_center_Y - h / 2, w, h])

            ones = np.ones((kpts1.shape[0], 1), dtype=float)
            kpts1 = np.concatenate([kpts1, ones], axis=1)
            for i in range(len(kpts)):
                if kpts[i][2] == 0:
                    kpts1[i][0] = kpts[i][0]
                    kpts1[i][1] = kpts[i][1]
                    kpts1[i][2] = kpts[i][2]

            kpts1 = np.array(kpts1).reshape(-1).tolist()
            keypoints.append(kpts1)
            img = show_skelenton(canvas, kpts1)

        if while_count == 0:
            print("pass")
            continue
        cv2.imwrite(os.path.join(SAVE_IMAGE_HQ_PATH, img_name), img)
        with open(os.path.join(SAVE_TXT_PATH, txt_name), "w") as f:
            f.write(txt)
        print(img_idx)

        img_json = image_format.copy()
        img_json['file_name'] = img_name
        img_json['height'] = H
        img_json['width'] = W
        img_json['id'] = img_idx
        data['images'].append(img_json)

        for i, keypoint, box in zip(range(len(keypoints)), keypoints, json_box):
            annotation_json = annotation_format.copy()
            annotation_json['keypoints'] = keypoint
            annotation_json['image_id'] = img_idx
            annotation_json['bbox'] = box
            annotation_json['id'] = (img_idx * 100) + i
            data['annotations'].append(annotation_json)       
        img_idx += 1

    with open(os.path.join(SAVE_PATH, json_file_name), "w") as f:
        f.write(json.dumps(data))


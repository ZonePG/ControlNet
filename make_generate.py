import json
import os
import random
import math
import numpy as np
from coco_util import get_center_keypoints, get_box, kp_trans_dst, affine_points, show_skelenton
import cv2

H, W = 384, 512

SAVE_PATH = "/root/autodl-tmp/datasets/random_generate"
SAVE_IMAGE_HQ_PATH = os.path.join(SAVE_PATH, "hq")
SAVE_TXT_PATH = os.path.join(SAVE_PATH, "coco_txt")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_TXT_PATH, exist_ok=True)

if __name__ == "__main__":
    data = json.load(open("/root/autodl-tmp/dict/dict.json", "r"))
    size = ["small", "middle", "large"]
    actions = list(data.keys())
    img_idx = 100000000000
    while img_idx < 100000001000:
        action_num = random.randint(1, 7)

        choice_action = []
        choice_img_name = []
        choice_size = []
        action_count = {}
        keypoints = []
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
            choice_img_name.append(random.choice(list(data[action].keys())))

        txt = f"There are {action_num} people in the picture, "
        for action, count in action_count.items():
            txt += f"{count} people is {action}ing, "
        txt = txt[:-2] + "."

        center = []
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for idx in range(len(choice_img_name)):
            # img_idx = int(choice_img_name[i])
            keypoint = data[choice_action[idx]][choice_img_name[idx]][choice_size[idx]]
            keypoints.append(keypoint)
            kpts = np.array(keypoint).reshape(-1, 3)
            box = get_box(keypoint)
            X_min, Y_min, X_max, Y_max = box
            w = X_max - X_min
            h = Y_max - Y_min

            target_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
            target_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
            if len(center) > 0:
                np_center = np.array(center)
                distances = np.min(np.sqrt(np.sum((np_center - np.array([target_center_X, target_center_Y]))**2, axis=1)))
                while_count = 100
                while distances < 100 and while_count > 0:
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

            ones = np.ones((kpts1.shape[0], 1), dtype=float)
            kpts1 = np.concatenate([kpts1, ones], axis=1)
            for i in range(len(kpts)):
                if kpts[i][2] == 0:
                    kpts1[i][0] = kpts[i][0]
                    kpts1[i][1] = kpts[i][1]
                    kpts1[i][2] = kpts[i][2]

            kpts1 = np.array(kpts1).reshape(1, -1).tolist()
            img = show_skelenton(canvas, kpts1)

        if while_count == 0:
            print("pass")
            continue
        img_idx += 1
        print(img_idx)
        cv2.imwrite(os.path.join(SAVE_IMAGE_HQ_PATH, f"{img_idx}.jpg"), img)
        with open(os.path.join(SAVE_TXT_PATH, f"{img_idx}.txt"), "w") as f:
            f.write(txt)

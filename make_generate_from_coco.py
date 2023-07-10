import numpy as np
import random
import math
import os
import cv2
import json
from coco_util import (
    load_coco,
    get_imgs_id_have_all_keypoints,
    get_box,
    kp_trans_dst,
    affine_points,
    show_skelenton,
)
from coco_format import *
from utils_func import fill_idx

coco = load_coco("/root/autodl-tmp/data/test/rotate.json")
se_ids = get_imgs_id_have_all_keypoints(coco)
img_ids = sorted(se_ids)
print(len(img_ids))

H, W = 768, 620
SAVE_PATH = "/root/autodl-tmp/data/test"
SAVE_IMAGE_HQ_PATH = os.path.join(SAVE_PATH, "hq_rotate_trans")
SAVE_TXT_PATH = os.path.join(SAVE_PATH, "coco_txt")
people_count = {}
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_TXT_PATH, exist_ok=True)


def random_trans_keypoints(origin_keypoints, image_shape=(H, W)):
    # 在区域内随机平移关节
    origin_keypoints_np = np.array(origin_keypoints).reshape(-1, 3).astype(np.float32)
    origin_box = get_box(origin_keypoints)
    origin_X_min, origin_Y_min, origin_X_max, origin_Y_max = origin_box
    w = origin_X_max - origin_X_min
    h = origin_Y_max - origin_Y_min

    new_keypoints_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
    new_keypoints_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
    new_keypoints_center = np.array([new_keypoints_center_X, new_keypoints_center_Y])
    trans = kp_trans_dst(origin_box, dst_shape=image_shape, dst_center=new_keypoints_center)
    new_keypoints_np = affine_points(origin_keypoints_np, trans)
    ones = np.ones((new_keypoints_np.shape[0], 1), dtype=float)
    new_keypoints_np = np.concatenate([new_keypoints_np, ones], axis=1)
    for i in range(len(origin_keypoints_np)):
        if origin_keypoints_np[i][2] == 0:
            new_keypoints_np[i][0] = origin_keypoints_np[i][0]
            new_keypoints_np[i][1] = origin_keypoints_np[i][1]
            new_keypoints_np[i][2] = origin_keypoints_np[i][2]
    new_keypoints = np.array(new_keypoints_np).reshape(-1).tolist()
    return new_keypoints_center_X, new_keypoints_center_Y, new_keypoints


if __name__ == "__main__":
    # img_ids = get_imgs_id_have_all_keypoints()
    index = 0
    while index < len(img_ids):
        img_id = img_ids[index]
        print(index, img_id)
        # img_id = 251292
        img_name = fill_idx(img_id, ".jpg")
        txt_name = fill_idx(img_id, ".txt")
        if img_id == 129436 or img_id == 232463 or img_id == 325102 or img_id == 514222:
            # 旋转后超出了边框
            print("continue: ", index, ", ", img_name)
            index += 1
            continue

        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        keypoints = []
        json_box = []
        ids = []

        center = []
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for person_id, obj in enumerate(objs):
            if obj["num_keypoints"] < 5:
                continue

            origin_keypoints = obj["keypoints"]
            origin_box = get_box(origin_keypoints)
            origin_X_min, origin_Y_min, origin_X_max, origin_Y_max = origin_box
            w = origin_X_max - origin_X_min
            h = origin_Y_max - origin_Y_min

            new_keypoints_center_X, new_keypoints_center_Y, new_keypoints = random_trans_keypoints(origin_keypoints)
            while_count = 100
            if len(center) > 0:
                np_center = np.array(center)
                distances = np.min(np.sqrt(np.sum((np_center - np.array([new_keypoints_center_X, new_keypoints_center_Y])) ** 2, axis=1)))
                while distances < 200 and while_count > 0:
                    new_keypoints_center_X, new_keypoints_center_Y, new_keypoints = random_trans_keypoints(origin_keypoints)
                    distances = np.min(np.sqrt(np.sum((np_center - np.array([new_keypoints_center_X, new_keypoints_center_Y])) ** 2, axis=1)))
                    while_count -= 1
            if while_count == 0:
                    break

            dst_center = np.array([new_keypoints_center_X, new_keypoints_center_Y])
            center.append(dst_center)
            json_box.append([new_keypoints_center_X - w / 2, new_keypoints_center_Y - h / 2, w, h])
            keypoints.append(new_keypoints)
            ids.append(obj["id"])
            img = show_skelenton(canvas, new_keypoints)
        if while_count == 0:
            print("continue: ", index, ", ", img_name)
            index += 1
            continue
        cv2.imwrite(os.path.join(SAVE_IMAGE_HQ_PATH, img_name), img)

        # update data
        img_json = image_format.copy()
        img_json["file_name"] = img_name
        img_json["height"] = H
        img_json["width"] = W
        img_json["id"] = img_id
        data["images"].append(img_json)

        for anno_id, keypoint, box in zip(ids, keypoints, json_box):
            annotation_json = annotation_format.copy()
            annotation_json["keypoints"] = keypoint
            annotation_json["image_id"] = img_id
            annotation_json["bbox"] = box
            annotation_json["id"] = anno_id
            data["annotations"].append(annotation_json)

        if len(ids) not in people_count:
            people_count[len(ids)] = 1
        else:
            people_count[len(ids)] += 1
        index += 1
    with open(os.path.join(SAVE_PATH, "rotate_trans.json"), "w") as f:
        f.write(json.dumps(data))
    print(people_count)

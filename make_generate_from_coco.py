import numpy as np
import random
import math
import os
import cv2
import json
from coco_util import coco, get_imgs_id_have_all_keypoints, get_box, kp_trans_dst, affine_points, show_skelenton
from coco_format import *

H, W = 600, 720

SAVE_PATH = "/root/autodl-tmp/datasets/generate_from_coco"
SAVE_IMAGE_HQ_PATH = os.path.join(SAVE_PATH, "hq")
SAVE_TXT_PATH = os.path.join(SAVE_PATH, "coco_txt")

people_count = {}

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_TXT_PATH, exist_ok=True)

if __name__ == "__main__":
    imgs_id = get_imgs_id_have_all_keypoints()

    print(imgs_id)
    index = 0

    while index < len(imgs_id):
        img_id = imgs_id[index]
        print(img_id)
        img_name = str(img_id).zfill(12) + ".jpg"
        txt_name = str(img_id).zfill(12) + ".txt"

        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)

        keypoints = []
        json_box = []
        ids = []

        center = []
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for person_id, obj in enumerate(objs):
            origin_keypoint = obj["keypoints"]
            origin_keypoint_np = np.array(origin_keypoint).reshape(-1, 3).astype(np.float32)
            origin_box = get_box(origin_keypoint)
            origin_X_min, origin_Y_min, origin_X_max, origin_Y_max = origin_box
            w = origin_X_max - origin_X_min
            h = origin_Y_max - origin_Y_min
            
            target_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
            target_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
            while_count = 100
            if len(center) > 0:
                np_center = np.array(center)
                distances = np.min(np.sqrt(np.sum((np_center - np.array([target_center_X, target_center_Y]))**2, axis=1)))
                while distances < 200 and while_count > 0:
                    target_center_X = random.randint(math.ceil(w / 2), math.floor(W - w / 2))
                    target_center_Y = random.randint(math.ceil(h / 2), math.floor(H - h / 2))
                    distances = np.min(np.sqrt(np.sum((np_center - np.array([target_center_X, target_center_Y]))**2, axis=1)))
                    while_count -= 1
                if while_count == 0:
                    break

            dst_center = np.array([target_center_X, target_center_Y])
            center.append(dst_center)
            trans = kp_trans_dst(origin_box, dst_shape=(H, W), dst_center=dst_center)
            new_keypoint_np = affine_points(origin_keypoint_np, trans)
            json_box.append([target_center_X - w / 2, target_center_Y - h / 2, w, h])

            ones = np.ones((new_keypoint_np.shape[0], 1), dtype=float)
            new_keypoint_np = np.concatenate([new_keypoint_np, ones], axis=1)
            for i in range(len(origin_keypoint_np)):
                if origin_keypoint_np[i][2] == 0:
                    new_keypoint_np[i][0] = origin_keypoint_np[i][0]
                    new_keypoint_np[i][1] = origin_keypoint_np[i][1]
                    new_keypoint_np[i][2] = origin_keypoint_np[i][2]

            new_keypoint = np.array(new_keypoint_np).reshape(-1).tolist()
            keypoints.append(new_keypoint)
            ids.append(obj['id'])
            img = show_skelenton(canvas, new_keypoint)
        if while_count == 0:
            print("pass")
            continue
        cv2.imwrite(os.path.join(SAVE_IMAGE_HQ_PATH, img_name), img)


        img_json = image_format.copy()
        img_json['file_name'] = img_name
        img_json['height'] = H
        img_json['width'] = W
        img_json['id'] = img_id
        data['images'].append(img_json)

        for anno_id, keypoint, box in zip(ids, keypoints, json_box):
            annotation_json = annotation_format.copy()
            annotation_json['keypoints'] = keypoint
            annotation_json['image_id'] = img_id
            annotation_json['bbox'] = box
            annotation_json['id'] = anno_id
            data['annotations'].append(annotation_json)

        if len(ids) not in people_count:
            people_count[len(ids)] = 1
        else:
            people_count[len(ids)] += 1
        index += 1
    with open(os.path.join(SAVE_PATH, json_file_name), "w") as f:
        f.write(json.dumps(data))
    print(people_count)


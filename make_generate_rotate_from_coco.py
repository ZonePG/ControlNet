import numpy as np
import random
import math
import os
import cv2
import json
from coco_util import (
    load_coco,
    get_imgs_id_have_all_keypoints,
    get_useful_point,
    show_skelenton,
)
from coco_format import *
from utils_func import fill_idx

coco = load_coco("/root/autodl-tmp/data/person_keypoints_train2017.json")
se_ids = get_imgs_id_have_all_keypoints(coco)
img_ids = sorted(se_ids)
print(len(img_ids))

H, W = 768, 620
SAVE_PATH = "/root/autodl-tmp/data/test"
SAVE_IMAGE_HQ_PATH = os.path.join(SAVE_PATH, "hq_rotate")
SAVE_IMAGE_ORI_HQ_PATH = os.path.join(SAVE_PATH, "hq_ori")
SAVE_TXT_PATH = os.path.join(SAVE_PATH, "coco_txt")
people_count = {}
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_IMAGE_ORI_HQ_PATH, exist_ok=True)
os.makedirs(SAVE_TXT_PATH, exist_ok=True)

rotate_list = [8, 7, 14, 13, 6, 5, 12, 11]

def construct_box(keypoint, connection_keypoint):
    half_w = math.fabs(keypoint[0] - connection_keypoint[0])
    half_h = math.fabs(keypoint[1] - connection_keypoint[1])
    X_min = keypoint[0] - half_w
    Y_min = keypoint[1] - half_h
    X_max = keypoint[0] + half_w
    Y_max = keypoint[1] + half_h
    return [X_min, Y_min, X_max, Y_max]


def rotate_point(center_point, rotate_point, rotation=[-30, 30]):
    # 假设对图片上任意点(x,y)，绕一个坐标点 (rx0,ry0) 逆时针旋转a角度后的新的坐标设为(x0, y0)，有公式：
    # x0 = (x - rx0) * cos(a) - (y - ry0)*sin(a) + rx0;
    # y0 = (x - rx0) * sin(a) + (y - ry0)*cos(a) + ry0;
    angle = random.randint(*rotation)  # 角度制
    angle = angle / 180 * math.pi      # 弧度制
    x, y = rotate_point[0], rotate_point[1]
    rx0, ry0 = center_point[0], center_point[1]
    x0 = (x - rx0) * math.cos(angle) - (y - ry0) * math.sin(angle) + rx0
    y0 = (x - rx0) * math.sin(angle) + (y - ry0) * math.cos(angle) + ry0
    return [x0, y0]


def random_rotate_one_keypoints(origin_keypoints, keypoint_num, image_shape=(H, W)):
    # keypoint num 下标从 0 开始
    assert keypoint_num in rotate_list
    connection_keypoint_table = {
        8: 10,
        7: 9,
        14: 16,
        13: 15,
        6: 8,
        5: 7,
        12: 14,
        11: 13
    }
    origin_keypoints_np = np.array(origin_keypoints).reshape(-1, 3).astype(np.float32)
    connection_keypoint_num = connection_keypoint_table[keypoint_num]

    # 以 keypoint_num 为中心构建 box
    keypoint = origin_keypoints_np[keypoint_num].tolist()
    connection_keypoint = origin_keypoints_np[connection_keypoint_num].tolist()
    if keypoint[2] == 0 or connection_keypoint[2] == 0:
        # 不存在连接点，直接返回
        return origin_keypoints

    rotate_connection_keypoint = rotate_point(keypoint, connection_keypoint)

    origin_keypoints_np[connection_keypoint_num][0] = rotate_connection_keypoint[0]
    origin_keypoints_np[connection_keypoint_num][1] = rotate_connection_keypoint[1]
    if keypoint_num in [6, 5, 12, 11]:
        # 需要平移对应的 [10, 9, 16, 15] 关键点
        special_keypoint_table = {
            6: 10,
            5: 9,
            12: 16,
            11: 15
        }
        special_keypoint_num = special_keypoint_table[keypoint_num]
        special_keypoint = origin_keypoints_np[special_keypoint_num].tolist()
        if special_keypoint[2] != 0:
            special_relative_x = special_keypoint[0] - connection_keypoint[0]
            special_relative_y = special_keypoint[1] - connection_keypoint[1]
            origin_keypoints_np[special_keypoint_num][0] = rotate_connection_keypoint[0] + special_relative_x
            origin_keypoints_np[special_keypoint_num][1] = rotate_connection_keypoint[1] + special_relative_y

    new_keypoints = np.array(origin_keypoints_np).reshape(-1).tolist()
    return new_keypoints


def random_rotate_keypoints(origin_keypoints, image_shape=(H, W)):
    # 固定原始中心点不动，旋转关节
    for keypoint_num in rotate_list:
        origin_keypoints = random_rotate_one_keypoints(origin_keypoints, keypoint_num, image_shape)
    return origin_keypoints


if __name__ == "__main__":
    # img_ids = get_imgs_id_have_all_keypoints()
    index = 0
    while index < len(img_ids):
        img_id = img_ids[index]
        print(img_id)
        img_name = fill_idx(img_id, ".jpg")
        txt_name = fill_idx(img_id, ".txt")

        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        keypoints = []
        json_box = []
        ids = []

        center = []
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        ori_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for person_id, obj in enumerate(objs):
            if obj["num_keypoints"] < 5:
                continue
            origin_keypoints = obj["keypoints"]
            new_keypoints = random_rotate_keypoints(origin_keypoints)
            X_min, Y_min, X_max, Y_max, X_center, Y_center = get_useful_point(new_keypoints)
            w = X_max - X_min
            h = Y_max - Y_min
            dst_center = np.array([X_center, Y_center])
            center.append(dst_center)
            json_box.append([X_center - w / 2, Y_center - h / 2, w, h])
            keypoints.append(new_keypoints)
            ids.append(obj["id"])
            ori_img = show_skelenton(ori_canvas, origin_keypoints)
            img = show_skelenton(canvas, new_keypoints)
        cv2.imwrite(os.path.join(SAVE_IMAGE_ORI_HQ_PATH, img_name), ori_img)
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
    with open(os.path.join(SAVE_PATH, "rotate.json"), "w") as f:
        f.write(json.dumps(data))
    print(people_count)

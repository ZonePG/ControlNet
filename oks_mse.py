import os
from glob import glob
import cv2
from annotator.openpose import OpenposeDetector
from coco_util import load_coco, show_skelenton, get_box
import numpy as np
import math

from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")


apply_openpose = OpenposeDetector()

# TODO fix
BASE_DIR = "/root/autodl-tmp/data/gc14w"
IMG_PATH = os.path.join(BASE_DIR, "img")
HT_PATH = os.path.join(BASE_DIR, "hq")
DHT_PATH = os.path.join(BASE_DIR, "tthq")
coco = load_coco("/root/autodl-tmp/data/gc14w/person_keypoints_coco_controlnet1k.json")
imgs_id = coco.getImgIds()


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


def compute_similarity(emb_one, emb_two):
    """Computes cosine similarity between two vectors."""
    scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
    return scores.numpy().tolist()


def resnet18_score(imgA, imgB):
    input1 = feature_extractor(imgA, return_tensors="pt")
    with torch.no_grad():
        logit1 = model(**input1).logits
    input2 = feature_extractor(imgB, return_tensors="pt")
    with torch.no_grad():
        logit2 = model(**input2).logits
    return compute_similarity(logit1, logit2)[0]


def mse(imgA, imgB):
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err


def compute_objs_score(objs, ht_image, openpose_image, test=False, resolution=256):
    score = 0
    for person_id, obj in enumerate(objs):
        obj_keypoints = obj["keypoints"]
        xmin, ymin, xmax, ymax = get_box(obj_keypoints)
        xmin, ymin = math.floor(xmin) - 5, math.floor(ymin) - 5
        xmax, ymax = math.ceil(xmax) + 5, math.ceil(ymax) + 5
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, ht_image.shape[1]), min(ymax, ht_image.shape[0])

        crop_ht_image = ht_image[ymin:ymax, xmin:xmax, :]
        crop_ht_image = cv2.resize(
            crop_ht_image,
            dsize=(resolution, resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        crop_openpose_image = openpose_image[ymin:ymax, xmin:xmax, :]
        crop_openpose_image = cv2.resize(
            crop_openpose_image,
            dsize=(resolution, resolution),
            interpolation=cv2.INTER_LINEAR,
        )

        # test
        if test:
            canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            # show_skelenton(canvas, obj_keypoints)
            cv2.imwrite("test1.jpg", crop_ht_image)
            cv2.imwrite("test2.jpg", crop_openpose_image)
            cv2.imwrite("test3.jpg", canvas)
            test_score = resnet18_score(crop_ht_image, canvas)
            print("(crop_ht_image, canvas) score: ", test_score)
        score += resnet18_score(crop_ht_image, crop_openpose_image)
    return score / len(objs)


def compute_img_score(img_id, test=False):
    img_name = fill_idx(img_id)
    full_path_image_name = os.path.join(IMG_PATH, img_name)
    full_path_ht_image_name = os.path.join(HT_PATH, img_name)
    full_path_dtht_image_name = os.path.join(DHT_PATH, img_name)

    image = cv2.imread(full_path_image_name)
    ht_image = cv2.imread(full_path_ht_image_name)
    # openpose_image, _ = apply_openpose(image)
    openpose_image = cv2.imread(full_path_dtht_image_name)

    annIds = coco.getAnnIds(img_id)
    objs = coco.loadAnns(annIds)

    return compute_objs_score(objs, ht_image, openpose_image, test)


if __name__ == "__main__":
    result = []
    # 看某一张图的效果
    # score = compute_img_mse(735, test=True)
    # print("score: ", score)
    for img_id in imgs_id:
        score = compute_img_score(img_id)
        print("img_id={}, score={}".format(img_id, score))
        if score > 0.9:  #< 15000 10222
            result.append(img_id)
    print(result, len(result))

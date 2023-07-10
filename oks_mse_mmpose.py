import os
from glob import glob
import cv2
from annotator.openpose import OpenposeDetector
from coco_util import load_coco, show_skelenton, get_box
import numpy as np
import math
from utils_func import fill_idx
import seaborn as sns

from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model

mmpose_model = init_pose_model(
    "/root/autodl-tmp/zoupeng/HumanSD/humansd_data/models/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py",
    "/root/autodl-tmp/zoupeng/HumanSD/humansd_data/checkpoints/higherhrnet_w48_humanart_512x512_udp.pth",
    device=DEVICE,
)

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

# TODO fix
BASE_DIR = "/root/autodl-tmp/data/tt_SD"
IMG_PATH = os.path.join(BASE_DIR, "79img")
HT_PATH = os.path.join(BASE_DIR, "sst")
DHT_PATH = os.path.join(BASE_DIR, "thq2")
os.makedirs(DHT_PATH, exist_ok=True)
coco = load_coco("/root/autodl-tmp/data/628/rotate_628.json")
imgs_id = coco.getImgIds()


def draw_humansd_skeleton(image, present_pose, mmpose_detection_thresh):
    humansd_skeleton = [
        [0, 0, 1],
        [1, 0, 2],
        [2, 1, 3],
        [3, 2, 4],
        [4, 3, 5],
        [5, 4, 6],
        [6, 5, 7],
        [7, 6, 8],
        [8, 7, 9],
        [9, 8, 10],
        [10, 5, 11],
        [11, 6, 12],
        [12, 11, 13],
        [13, 12, 14],
        [14, 13, 15],
        [15, 14, 16],
    ]
    humansd_skeleton_width = 10
    humansd_color = sns.color_palette("hls", len(humansd_skeleton))

    def plot_kpts(img_draw, kpts, color, edgs, width):
        for idx, kpta, kptb in edgs:
            if (
                kpts[kpta, 2] > mmpose_detection_thresh
                and kpts[kptb, 2] > mmpose_detection_thresh
            ):
                line_color = tuple([int(255 * color_i) for color_i in color[idx]])

                cv2.line(
                    img_draw,
                    (int(kpts[kpta, 0]), int(kpts[kpta, 1])),
                    (int(kpts[kptb, 0]), int(kpts[kptb, 1])),
                    line_color,
                    width,
                )
                cv2.circle(
                    img_draw,
                    (int(kpts[kpta, 0]), int(kpts[kpta, 1])),
                    width // 2,
                    line_color,
                    -1,
                )
                cv2.circle(
                    img_draw,
                    (int(kpts[kptb, 0]), int(kpts[kptb, 1])),
                    width // 2,
                    line_color,
                    -1,
                )

    pose_image = np.zeros_like(image)
    for person_i in range(len(present_pose)):
        if np.sum(present_pose[person_i]["keypoints"]) > 0:
            plot_kpts(
                pose_image,
                present_pose[person_i]["keypoints"],
                humansd_color,
                humansd_skeleton,
                humansd_skeleton_width,
            )

    return pose_image


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

    ht_image = cv2.imread(full_path_ht_image_name)
    ### 先对生成姿态图进行可视化 ###
    image = cv2.imread(full_path_image_name)
    mmpose_results = inference_bottom_up_pose_model(
        mmpose_model,
        image,
        dataset="BottomUpCocoDataset",
        dataset_info=None,
        pose_nms_thr=1.0,
        return_heatmap=False,
        outputs=None,
    )[0]
    mmpose_filtered_results = []
    for mmpose_result in mmpose_results:
        if mmpose_result["score"] > 0.05:
            mmpose_filtered_results.append(mmpose_result)
    # mmpose_image = draw_humansd_skeleton(image, mmpose_filtered_results, 0.05)
    # cv2.imwrite(full_path_dtht_image_name, mmpose_image)


    ####  直接读取生成好的数据  ####
    mmpose_image = cv2.imread(full_path_dtht_image_name)

    annIds = coco.getAnnIds(img_id)
    objs = coco.loadAnns(annIds)

    return compute_objs_score(objs, ht_image, mmpose_image, test)


if __name__ == "__main__":
    result = []
    # 看某一张图的效果
    # score = compute_img_mse(735, test=True)
    # print("score: ", score)
    imgs_id = list(range(2230))
    
    for img_id in imgs_id:
        score = compute_img_score(img_id)
        print("img_id={}, score={}".format(img_id, score))
        # 根据分数进行筛选
        # if score > 0.93:  #< 15000 10222
        #     result.append(img_id)
    # print(result, len(result))

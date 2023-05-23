import os
import json

import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import time as time
import itertools
import math
import random


def kp_trans(box, dst_shape=(384, 512), size="small"):
    scale = None
    rotation = None

    src_xmin, src_ymin, src_xmax, src_ymax = box[:4]
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin

    if src_h / dst_shape[0] > src_w / dst_shape[1]:
        if size == "small":
            h_N = 0.25
        elif size == "middle":
            h_N = 0.5
        elif size == "large":
            h_N = 0.8
        w_N = dst_shape[0] * h_N / src_h
        fixed_size = (dst_shape[0] * h_N, src_w * w_N)
    else:
        if size == "small":
            w_N = 0.25
        elif size == "middle":
            w_N = 0.5
        elif size == "large":
            w_N = 0.8
        h_N = dst_shape[1] * w_N / src_w
        fixed_size = (src_h * h_N, dst_shape[1] * w_N)
    src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
    src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    # dst_center = np.array([(fixed_size[1] + 1) / 2, (fixed_size[0] + 1) / 2])
    # dst_p2 = dst_center + np.array([(fixed_size[1]) / 2, 0])  # top middle
    # dst_p3 = np.array([fixed_size[1], (fixed_size[0]) / 2])  # right middle
    dst_center = np.array([dst_shape[1] / 2, dst_shape[0] / 2])
    dst_p2 = dst_center + np.array([0, -fixed_size[0] / 2])  # top middle
    dst_p3 = dst_center + np.array([fixed_size[1] / 2, 0])  # right middle

    if scale is not None:
        scale = random.uniform(*scale)
        src_w = src_w * scale
        src_h = src_h * scale
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    if rotation is not None:
        angle = random.randint(*rotation)  # 角度制
        angle = angle / 180 * math.pi  # 弧度制
        src_p2 = src_center + np.array(
            [src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)]
        )
        src_p3 = src_center + np.array(
            [src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)]
        )

    src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
    dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

    trans = cv2.getAffineTransform(src, dst)
    return trans


def kp_trans_dst(box, dst_shape=(384, 512), dst_center=None):
    scale = None
    rotation = None

    src_xmin, src_ymin, src_xmax, src_ymax = box[:4]
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin
    fixed_size = (src_h, src_w)
    src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
    src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    # dst_center = np.array([(fixed_size[1] + 1) / 2, (fixed_size[0] + 1) / 2])
    # dst_p2 = dst_center + np.array([(fixed_size[1]) / 2, 0])  # top middle
    # dst_p3 = np.array([fixed_size[1], (fixed_size[0]) / 2])  # right middle
    # dst_center = np.array([dst_shape[1] / 2, dst_shape[0] / 2])
    dst_p2 = dst_center + np.array([0, -fixed_size[0] / 2])  # top middle
    dst_p3 = dst_center + np.array([fixed_size[1] / 2, 0])  # right middle

    if scale is not None:
        scale = random.uniform(*scale)
        src_w = src_w * scale
        src_h = src_h * scale
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    if rotation is not None:
        angle = random.randint(*rotation)  # 角度制
        angle = angle / 180 * math.pi  # 弧度制
        src_p2 = src_center + np.array(
            [src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)]
        )
        src_p3 = src_center + np.array(
            [src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)]
        )

    src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
    dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

    trans = cv2.getAffineTransform(src, dst)
    return trans


def affine_points(pt, t):
    npt = pt[:, :2]
    ones = np.ones((npt.shape[0], 1), dtype=float)
    npt = np.concatenate([npt, ones], axis=1).T
    new_pt = np.dot(t, npt)
    return new_pt.T


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if annotation_file is not None:
            print("loading annotations into memory...")
            tic = time.time()
            dataset = json.load(open(annotation_file, "r"))
            assert (
                type(dataset) == dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]

        else:
            cats = self.dataset["categories"]
            cats = (
                cats
                if len(catNms) == 0
                else [cat for cat in cats if cat["name"] in catNms]
            )
            cats = (
                cats
                if len(supNms) == 0
                else [cat for cat in cats if cat["supercategory"] in supNms]
            )
            cats = (
                cats
                if len(catIds) == 0
                else [cat for cat in cats if cat["id"] in catIds]
            )

        ids = [cat["id"] for cat in cats]
        return ids

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def getImgIds(self, imgIds=[], catIds=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset["annotations"]
        else:
            # 根据imgIds找到所有的ann
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            # 通过各类条件如catIds对anns进行筛选
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]


def show_skelenton(img, kpts, thr=0.01):
    stickwidth = 2
    kpts = np.array(kpts).reshape(-1, 3)
    kp6 = kpts[5]
    kp7 = kpts[6]
    if kpts[5][2] != 0 and kpts[6][2] != 0:
        kp18 = [
            (kp6[0] + kp7[0]) / 2,
            (kp6[1] + kp7[1]) / 2,
            1,
        ]  # 找两个肩膀的中间点，如果其中一个肩膀的坐标为0，则将中间点坐标赋值为存在的点坐标
    else:
        kp18 = [(kp6[0] + kp7[0]), (kp6[1] + kp7[1]), 0]

    kpts = np.append(kpts, kp18)
    kpts = np.array(kpts).reshape(-1, 3)
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    skelenton = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [18, 1],
        [18, 6],
        [18, 7],
        [18, 12],
        [18, 13],
    ]
    # 18个点
    for n in range(len(kpts)):
        x, y = kpts[n][0:2]
        cv2.circle(img, (int(x), int(y)), 3, colors[n], thickness=-1)

    i = 0
    for sk in skelenton:
        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))

        if (
            pos1[0] > 0
            and pos1[1] > 0
            and pos2[0] > 0
            and pos2[1] > 0
            and kpts[sk[0] - 1, 2] > thr
            and kpts[sk[1] - 1, 0] > thr
        ):
            X = [pos1[1], pos2[1]]
            Y = [pos1[0], pos2[0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            #             ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta)
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )  # 画连接的椭圆
            cv2.fillConvexPoly(img, polygon, colors[i])  # 填充颜色
            i += 1
    return img


def get_center_keypoints(img_idx, dst_shape=(384, 512, 3), size="small"):
    annIds = coco.getAnnIds(imgIds=img_idx, iscrowd=False)
    objs = coco.loadAnns(annIds)
    for person_id, obj in enumerate(objs):
        if obj["num_keypoints"] < 5:
            continue
        keypoints = obj["keypoints"]
        kpts = np.array(keypoints).reshape(-1, 3)

        mask = np.logical_and(kpts[:, 0] != 0, kpts[:, 1] != 0)

        # 通过坐标点，找框，然后找中心点坐标，从而生成仿射矩阵，进行坐标点的变化
        konghang = []
        for i in range(len(kpts)):
            if kpts[i][2] == 0:
                konghang.append(i)
        kpt_new = np.delete(kpts, konghang, axis=0)

        MAX = np.max(kpt_new, axis=0)
        X_max, Y_max = MAX[0], MAX[1]
        MIN = np.min(kpt_new, axis=0)
        X_min, Y_min = MIN[0], MIN[1]
        box = [X_min, Y_min, X_max, Y_max]

        trans = kp_trans(box, dst_shape=(dst_shape[0], dst_shape[1]), size=size)

        # 大小缩方、位置变化
        kpts1 = affine_points(kpts, trans)

        ones = np.ones((kpts1.shape[0], 1), dtype=float)
        kpts1 = np.concatenate([kpts1, ones], axis=1)
        for i in range(len(kpts)):
            if kpts[i][2] == 0:
                kpts1[i][0] = kpts[i][0]
                kpts1[i][1] = kpts[i][1]
                kpts1[i][2] = kpts[i][2]

        kpts1 = np.array(kpts1).reshape(1, -1).tolist()
        # canvas = np.zeros(dst_shape, dtype=np.uint8)
        # img = show_skelenton(canvas, kpts1)
        # htpath = "test.jpg"
        # cv2.imwrite(htpath, img)
        return kpts1[0]


def get_imgs_id_have_all_keypoints():
    imgs_id_have_all_keypoints = []
    for img_id in img_ids:
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        is_have_all_keypoints = True
        for person_id, obj in enumerate(objs):
            if obj["num_keypoints"] != 17:
                is_have_all_keypoints = False
                break
        if is_have_all_keypoints:
            imgs_id_have_all_keypoints.append(img_id)
    return sorted(imgs_id_have_all_keypoints)


def get_box(keypoint):
    kpts = np.array(keypoint).reshape(-1, 3)
    mask = np.logical_and(kpts[:, 0] != 0, kpts[:, 1] != 0)
    # 通过坐标点，找框，然后找中心点坐标，从而生成仿射矩阵，进行坐标点的变化
    konghang = []
    for i in range(len(kpts)):
        if kpts[i][2] == 0:
            konghang.append(i)
    kpt_new = np.delete(kpts, konghang, axis=0)

    MAX = np.max(kpt_new, axis=0).tolist()
    X_max, Y_max = MAX[0], MAX[1]
    MIN = np.min(kpt_new, axis=0).tolist()
    X_min, Y_min = MIN[0], MIN[1]
    return [X_min, Y_min, X_max, Y_max]


coco_json_path = "/root/autodl-tmp/datasets/person_keypoints_train2017.json"
coco_img_path = "."

coco = COCO(coco_json_path)
catIds = coco.getCatIds(catNms=["person"])
img_ids = coco.getImgIds(catIds=catIds)

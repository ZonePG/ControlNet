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


def kp_trans(box):
    scale = None
    rotation = None
    N = 1.5  # 缩方的大小比例 越大人体姿态越小

    src_xmin, src_ymin, src_xmax, src_ymax = box[:4]
    src_w = src_xmax - src_xmin
    src_h = src_ymax - src_ymin
    fixed_size = (src_h / N, src_w / N)

    src_center = np.array([(src_xmin + src_xmax) / 2 + 100, (src_ymin + src_ymax) / 2])
    src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
    src_p3 = src_center + np.array([src_w / 2, 0])  # right middle

    # dst_center = np.array([(fixed_size[1] + 1) / 2, (fixed_size[0] + 1) / 2])
    # dst_p2 = dst_center + np.array([(fixed_size[1]) / 2, 0])  # top middle
    # dst_p3 = np.array([fixed_size[1], (fixed_size[0]) / 2])  # right middle
    dst_center = src_center.copy()
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
        if not annotation_file == None:
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


coco_json_path = "./person_keypoints_train2017.json"
coco_img_path = "."
save_plt_path = "."

if not os.path.exists(save_plt_path):
    os.makedirs(save_plt_path)

coco = COCO(coco_json_path)
catIds = coco.getCatIds(catNms=["person"])
img_ids = coco.getImgIds(catIds=catIds)
# img_ids = coco.getImgIds()


num = 0
img_ids = [86]
for img_idx in tqdm(img_ids):
    img_name = str(img_idx).zfill(12) + ".jpg"
    img_path = coco_img_path + "/" + img_name
    img = cv2.imread(img_path)
    # oripath = save_plt_path+'/ori_img/'+img_name
    # cv2.imwrite(oripath,img)

    canvas = np.zeros_like(img)  # 创建黑色画布
    annIds = coco.getAnnIds(imgIds=img_idx, iscrowd=False)
    objs = coco.loadAnns(annIds)
    save_p = False  # 是否保存图，如果没有标注任何点信息，则不画图像

    # ob={
    #     'segmentation': 0,
    #     'num_keypoints': 0,
    #     'area': 0,
    #     'iscrowd':0,
    #     'keypoints': kyp,
    #     'image_id':0,
    #     'bbox':0,
    #     'category_id':0,
    #     'id':0,
    # }
    # objs.append(ob)

    for person_id, obj in enumerate(objs):
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
        trans = kp_trans(box)

        # 大小缩方、位置变化
        kpts1 = affine_points(kpts, trans)

        print(kpts)
        ones = np.ones((kpts1.shape[0], 1), dtype=float)
        kpts1 = np.concatenate([kpts1, ones], axis=1)
        for i in range(len(kpts)):
            if kpts[i][2] == 0:
                kpts1[i][0] = kpts[i][0]
                kpts1[i][1] = kpts[i][1]
                kpts1[i][2] = kpts[i][2]

        kpts1 = np.array(kpts1).reshape(1, -1).tolist()

        print(kpts1)
        print(keypoints)

        img = show_skelenton(canvas, kpts1)
        htpath = save_plt_path + "/ht/" + img_name
        cv2.imwrite(htpath, img)


# num_list = [161586, 360716, 108056, 45920, 500814, 156375, 256035, 440045, 5038, 267643, 331250, 576607, 461491, 183843, 5247, 432239, 336166, 425462, 177407, 539888, 267251, 538092, 252020, 281111, 461860, 159372, 186026, 356708, 331403, 298331, 360182, 8923, 330348, 113989, 542938, 549683, 336802, 296848, 562345, 255112, 395665, 161079, 519706, 304008, 530619, 264853, 62198, 186711, 246649, 515219, 527718, 250594, 155873, 499252, 136285, 283186, 331180, 270738, 211722, 571012, 355137, 202825, 418092, 449191, 382797, 99615, 251404, 74832, 297676, 416059, 54277, 526576, 463325, 64902, 165012, 305195, 149500, 190081, 552188, 223276, 494139, 450599, 303320, 330754, 196053, 564163, 475398, 371135, 342711, 84460, 32947, 421131, 478550, 25005, 448175, 203734, 53431, 511066, 154254, 301670, 499755, 189845, 170181, 505788, 483368, 150358, 112905, 388654, 81303, 357096, 192656, 544334, 158887, 320715, 140007, 308630, 460676, 134285, 402405, 563680, 408327, 431208, 14874, 258078, 522163, 537124, 284350, 140860, 432146, 35313, 576463, 244151, 259513, 12269, 299631, 376521, 579073, 578344, 442962, 554114, 264568, 578427, 216198, 289263, 361382, 134551, 554582, 313169, 195408, 181786, 75595, 476925, 240137, 322937, 513129, 240028, 1948, 29080, 282225, 362369, 107360, 47498, 182167, 208135, 141509, 320864, 578292, 374368, 445567, 279420, 324634, 455414, 430259, 25864, 356937, 451951, 33645, 405529, 275034, 459301, 77473, 33900, 478621, 508119, 103873, 200058, 192095, 378334, 543692, 295092, 261050, 160351, 403672, 127451, 194499, 510182, 382848, 289152, 507935, 511117, 443429, 13892, 512116, 434805, 150533, 166624, 241318, 366502, 479379, 462486, 194525, 229889, 428015, 435136, 111109, 450400, 528091, 265374, 363126, 347170, 331196, 437732, 145378, 390435, 323552, 475856, 165133, 238065, 112066, 525705, 311309, 95051, 184810, 220277, 62878, 199403, 207545, 476569, 519744, 434976, 501762, 254638, 199540, 185988, 465986, 304834, 480663, 333691, 275393, 207178, 34234, 404367, 40016, 170000, 269311, 357356, 509192, 551107, 390585, 84097, 258399, 259345, 516813, 179199, 45388, 560108, 316107, 558406, 426453, 367706, 114158, 514083, 444719, 375755, 248701, 511622, 232054, 536831, 315908, 426070, 546029, 553852, 33158, 458275, 207239, 158015, 450567, 187352, 345961, 535106, 502766, 48169, 97479, 524047, 560718, 350668, 144379, 410339, 404613, 323288, 486172, 523696, 279259, 539079, 15017, 246064, 512139, 303342, 343394, 535506, 240340, 320106, 555131, 143333, 271639, 559470, 530998, 536791, 460781, 187244, 177194, 515792, 526892, 241174, 102903, 63549, 446473, 316795, 15559, 277533, 335472, 334713, 517807, 508822, 32812, 391365, 44029, 505242, 112830, 191000, 71699, 259037, 25508]
# print(len(num_list))

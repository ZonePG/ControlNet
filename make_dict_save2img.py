import json
import os
import numpy as np
import math
import cv2


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


if __name__ == "__main__":
    data = json.load(open("./dict/dict.json", "r"))
    print(data)

    WRITE_DIR = "./dict"

    os.makedirs(WRITE_DIR, exist_ok=True)

    for action, value in data.items():
        os.makedirs(os.path.join(WRITE_DIR, action), exist_ok=True)
        for jpg, value_jpg in value.items():
            for size in ["small", "middle", "large"]:
                os.makedirs(os.path.join(WRITE_DIR, action, size), exist_ok=True)
                keypoints = value_jpg[size]
                canvas = np.zeros((384, 512, 3), dtype=np.uint8)
                img = show_skelenton(canvas, keypoints)
                img_path = os.path.join(WRITE_DIR, action, size, jpg)
                cv2.imwrite(img_path, img)

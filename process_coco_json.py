import os
import json

THREE_PEOPLE_FILE = "/root/autodl-tmp/coco_controlnet/1_3people.txt"

with open(THREE_PEOPLE_FILE, "r") as f:
    S = f.read().replace(",", "")
    O = S.split()
    same_index = list(int(i) for i in O)

COCO_DIR_PATH = "/root/autodl-tmp/coco_controlnet"
COCO_TXT_DIR_PATH = os.path.join(COCO_DIR_PATH, "coco_txt")
COCO_HQ_DIR_PATH = os.path.join(COCO_DIR_PATH, "hq")
COCO_ORI_IMG_DIR_PATH = os.path.join(COCO_DIR_PATH, "ori_img")
WRITE_JSON_PATH = "/root/autodl-tmp/coco_controlnet/prompt.json"


def check(coco_txt_path_list, coco_hq_path_list, coco_ori_img_path_list):
    coco_txt_path_list = [x.replace(".txt", ".jpg") for x in coco_hq_path_list]

    inter1 = list(set(coco_txt_path_list).intersection(coco_hq_path_list))
    inter2 = list(set(coco_txt_path_list).intersection(coco_ori_img_path_list))
    assert coco_txt_path_list == sorted(inter1)
    assert coco_txt_path_list == sorted(inter2)


if __name__ == "__main__":
    coco_txt_path_list = sorted(os.listdir(COCO_TXT_DIR_PATH))
    coco_hq_path_list = sorted(os.listdir(COCO_HQ_DIR_PATH))
    coco_ori_img_path_list = sorted(os.listdir(COCO_ORI_IMG_DIR_PATH))
    # print(coco_txt_path_list[0])
    # print(coco_hq_path_list[0])
    # print(coco_ori_img_path_list[0])
    check(coco_txt_path_list, coco_hq_path_list, coco_ori_img_path_list)

    with open(WRITE_JSON_PATH, "w") as f:
        for img_path in coco_hq_path_list:
            txt_path = img_path.replace(".jpg", ".txt")
            txt = open(os.path.join(COCO_TXT_DIR_PATH, txt_path), "r").read()
            hq_img = os.path.join("hq", img_path)
            ori_img = os.path.join("ori_img", img_path)
            data = {"source": hq_img, "target": ori_img, "prompt": txt}
            f.write(json.dumps(data) + "\n")

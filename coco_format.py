json_file_name = "person_keypoints_coco_controlnet1k.json"

data = {
    "info": None,
    "license": None,
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "left_ankle",
                "right_ankle",
            ],
            "skeleton": [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [2, 3],
                [1, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
            ],
        }
    ],
}

image_format = {
    "license": None,
    "file_name": "",
    "coco_url": None,
    "height": 0,
    "width": 0,
    "date_captured": None,
    "flickr_url": None,
    "id": 0,
}

annotation_format = {
    "segmentation": None,
    "num_keypoints": 17,  # TODO
    "area": None,
    "iscrowd": 0,
    "keypoints": [],
    "image_id": 0,
    "bbox": [0, 0, 0, 0], # [upper_x, left_y, width, height]
    "category_id": 1,
    "id": 0,   # image_id << 2
}

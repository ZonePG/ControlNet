import json
import cv2
import numpy as np
import os
from annotator.util import resize_image

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.dataset_path = dataset_path
        self.prompt_path = os.path.join(dataset_path, 'prompt.json')
        with open(self.prompt_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source_filename = os.path.join(self.dataset_path, source_filename)
        target_filename = os.path.join(self.dataset_path, target_filename)

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)
        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
        target = resize_image(target, 512)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float16) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float16) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

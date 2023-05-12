from share import *
import config
import json
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from collections import defaultdict
import time as time
import itertools
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from PIL import Image

import torchvision


preprocessor = None

model_name = "control_v11p_sd15_openpose"
model_config_name = f"/root/autodl-tmp/zoupeng/ControlNet/models/{model_name}.yaml"
model_checkpoint_name = "/root/autodl-tmp/zoupeng/ControlNet/checkpoint-coco-batchsize32/lightning_logs/version_0/checkpoints/epoch=26-step=35343.pt"
base_model_checkpoint_name = "/root/autodl-tmp/zoupeng/ControlNet/models/v1-5-pruned.ckpt"
model = create_model(model_config_name).cpu()
model.load_state_dict(
    load_state_dict(base_model_checkpoint_name, location="cuda"),
    strict=False,
)
model.load_state_dict(
    load_state_dict(model_checkpoint_name, location="cuda"),
    strict=False,
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

COCO_CONTROLNET_PATH = "/root/autodl-tmp/coco_controlnet"
SAVE_PAH = "/root/autodl-tmp/coco_controlnet/generate_author_batch32_epoch26_3people"
os.makedirs(SAVE_PAH, exist_ok=True)


with torch.no_grad():
    seed_everything(0)
    with open('/root/autodl-tmp/coco_controlnet/prompt_3people.json') as f:
        train_data = [json.loads(line) for line in f]
        for data in train_data:
            print(data['target'])
            # data['source'] = 'hq/000000434992.jpg'
            # data['target'] = 'ori_img/000000434992.jpg'
            source = cv2.imread(os.path.join(COCO_CONTROLNET_PATH, data['source']))
            target = cv2.imread(os.path.join(COCO_CONTROLNET_PATH, data['target']))
            source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)
            target = resize_image(target, 512)
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0
            prompt = data['prompt']
            # prompt = "A man sitting on a bench with his bag and a small dog. A young man sits on a bench with a dog. A man sitting on a bench, unzipping a bag, with a chihuahua in a vest standing with front paws on the man's leg, by a parking lot and building with a woman walking and a person sitting in a chair under awning in the background.A man opening a bag, with a small dog on his lap.A man that is sitting on a bench with a dog."
            num_samples = 1

            jpg = torch.from_numpy(target.copy()).float().cuda()
            jpg = torch.stack([jpg for _ in range(num_samples)], dim=0)
            txt = [prompt]
            hint = torch.from_numpy(source.copy()).float().cuda()
            hint = torch.stack([hint for _ in range(num_samples)], dim=0)
            batch = {
                "jpg": jpg,
                "txt": txt,
                "hint": hint,
            }

            images = model.log_images(batch)
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().squeeze(0).cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)
            result = torch.stack([images[k] for k in images], dim=0)

            grid = torchvision.utils.make_grid(result, nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            path = os.path.join(SAVE_PAH, data['target'].split('/')[-1])
            # path = data['target'].split('/')[-1]
            Image.fromarray(grid).save(path)
